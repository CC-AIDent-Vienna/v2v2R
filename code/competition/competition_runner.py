#!/usr/bin/env python3
"""
code/competition/competition_runner.py

One case or a whole split, start to finished report.

ONE CASE is the shape it was written for -- Grand Challenge gives a fresh
container, an A10G-class card, and 15 minutes of wall clock for everything --
and that constraint is what makes the schedule below worth having.

A WHOLE SPLIT is the same code with the case loop around the CPU half:
--dataset-dir instead of --volume/--mask, and the four stages after rendering
are already directory-shaped, so they run once over every case rather than once
per case. The model then loads once for forty cases instead of forty times,
which is the only thing that ever made a batch driver a separate script. What
it does NOT do is render cases in parallel -- one case at a time, its four
generators concurrent -- so for a big split the faster front half is still
gen_images_cpu.sh on a CPU partition, and pool_infer.sh over the qa_pairs.jsonl
it writes. Batch here is for when one command is worth more than the wall
clock.

WHAT THE 15-MINUTE BUDGET BOUGHT EVERYONE ELSE
──────────────────────────────────────────────
A pipeline built for throughput over 40 cases amortises its startup to nothing,
and three habits follow from that which are expensive the moment the container
starts once PER CASE:

  1. generate all images, THEN start vLLM -- two serial waits where the first
     needs no GPU and the second needs no CPU;
  2. `sleep 90` before the first health check, then poll every 10 s, so a
     server that came up at 40 s is not noticed until 90;
  3. concurrency fixed at a number chosen for an A100.

This runner overlaps (1), polls /health every 0.5 s for (2), and sizes
concurrency off the KV cache the server actually reports for (3). None of the
three is worth less in batch: (1) hides the whole model load behind the first
case's rendering, and (3) is the difference between using the KV pool and
guessing at it.

THE OVERLAP, AND WHAT LIMITS IT
───────────────────────────────
    t0  ├─ vLLM starts loading ──────────────────────┐
        ├─ facts audit ─┐                            │
        │   ├─ panoramic ─┬─ 3d ─┬─ sinus ──┐        │
        │   │             └──────┴─ tooth composites ┼──┐
        │                                            ▼  ▼
        │                          server ready + globals ready
        │                                     └─ WAVE 1: global calls
        │                                        (composites still rendering)
        │                                            └─ WAVE 2: tooth calls

The facts audit comes first because every generator reads the facts file: it
decides which teeth are outlined and what the captions say, so auditing after a
render would mean rendering twice. It costs ~2 s against a ~200 s model load.

create_tooth_detail.py takes --pano-dir and --render-dir, so the composites
cannot start until the panoramic and the 3D renders exist; that dependency, not
scheduling, is what serialises the image half. The globals are dispatched as
soon as their four images and the server are both ready, which is the point:
those calls run while the 32 composites are still being written.

Wave 2 is not a barrier for correctness, only for images -- every call in a case
is independent (see run_vqa_inference.run_calls_concurrently and the
TRUST_MODEL_ABSENCE note there). It exists because the tooth calls need tooth
images.

WHERE THE TIME WENT
───────────────────
Every phase is timed and printed as a table at the end, and vLLM's own startup
milestones are parsed out of its stdout -- weight load, torch.compile, CUDA
graph capture, KV allocation -- so a slow start can be attributed instead of
guessed at. That breakdown is the point of the --report-timing output: "vLLM
took 11 minutes" is not actionable, "9 of those 11 were graph capture" is.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Repo bootstrap. Finds code/ by walking up for _repo.py, so this file does not
# care how deep it sits, and puts every code group on sys.path so the flat
# `import postprocess_pred` works across groups. See code/_repo.py.
import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(next(
    p for p in _pathlib.Path(__file__).resolve().parents
    if (p / "_repo.py").is_file())))
from _repo import REPO_ROOT, add_code_paths  # noqa: E402
add_code_paths()
from _repo import module_path  # noqa: E402



# Tokens of KV cache one of this pipeline's requests occupies, measured on the
# v7.1 validate runs. Used to turn the server's reported pool into a concurrency
# ceiling; see code/pipeline/aksssr_pipeline.sh's header for why the "Maximum concurrency"
# line vLLM prints is the wrong number to read.
TOKENS_PER_REQUEST = 6200
FALLBACK_CONCURRENCY = 8

# vLLM startup milestones worth attributing, in the order they occur. Each entry
# is (label, compiled regex with one float group of SECONDS).
VLLM_MILESTONES = [
    ("weight load",     re.compile(r"Loading weights took ([\d.]+) seconds")),
    ("model -> GPU",    re.compile(r"Model loading took [\d.]+ GiB memory and ([\d.]+) seconds")),
    ("torch.compile",   re.compile(r"torch\.compile took ([\d.]+) s")),
    ("cuda graphs",     re.compile(r"Graph capturing finished in (\d+) secs")),
    ("engine init",     re.compile(r"init engine .*took ([\d.]+) s")),
]
KV_RE = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
WEIGHTS_RE = re.compile(r"Model loading took ([\d.]+) GiB")


@dataclass
class Phases:
    """Wall-clock spans, recorded against a single t0 so they can overlap."""
    t0: float = field(default_factory=time.time)
    spans: List[tuple] = field(default_factory=list)
    _open: Dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def start(self, name: str) -> None:
        with self._lock:
            self._open[name] = time.time()

    def end(self, name: str) -> None:
        with self._lock:
            t = self._open.pop(name, None)
            if t is not None:
                self.spans.append((name, t - self.t0, time.time() - self.t0))

    def collapsed(self) -> List[tuple]:
        """One row per phase NAME, so a 40-case batch prints five rendering
        rows rather than two hundred. A name that occurs once collapses to
        itself, which is why single-case output is unchanged: earliest start,
        latest end, and the summed occupancy in `secs` -- summed, not
        end-minus-start, because between two cases' img:panoramic spans the
        phase is not running."""
        agg: Dict[str, list] = {}
        for name, s, e in self.spans:
            row = agg.setdefault(name, [name, s, e, 0.0, 0])
            row[1] = min(row[1], s)
            row[2] = max(row[2], e)
            row[3] += e - s
            row[4] += 1
        return sorted(agg.values(), key=lambda r: r[1])

    def table(self) -> str:
        total = time.time() - self.t0
        rows = self.collapsed()
        labels = {r[0] if r[4] == 1 else f"{r[0]} x{r[4]}" for r in rows}
        width = max((len(n) for n in labels), default=10)
        out = [f"{'phase':{width}}  {'start':>7} {'end':>7} {'secs':>7}   timeline"]
        out.append("-" * (width + 34 + 40))
        for name, s, e, secs, n in rows:
            name = name if n == 1 else f"{name} x{n}"
            # A crude gantt: where this span sits inside the whole run. The
            # overlap is the thing being demonstrated, so it has to be visible.
            bar_start = int(40 * s / total) if total else 0
            bar_len = max(1, int(40 * (e - s) / total)) if total else 1
            bar = " " * bar_start + "#" * bar_len
            out.append(f"{name:{width}}  {s:7.1f} {e:7.1f} {secs:7.1f}   {bar}")
        out.append(f"{'TOTAL':{width}}  {0:7.1f} {total:7.1f} {total:7.1f}")
        return "\n".join(out)


class VLLMServer:
    """Starts the server, tails its stdout, and reads what it says about itself.

    Stdout is tailed rather than discarded because it is the only place the
    startup breakdown and the KV cache size appear -- both are needed here, one
    to explain a slow start and one to size concurrency.
    """

    def __init__(self, model: str, port: int, max_model_len: int,
                 gpu_mem_util: float, max_images: int, log_path: Path,
                 extra_args: Optional[List[str]] = None):
        self.model, self.port, self.log_path = model, port, log_path
        self.cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model, "--served-model-name", "qwen3.5-vl",
            "--port", str(port), "--max-model-len", str(max_model_len),
            "--gpu-memory-utilization", str(gpu_mem_util),
            "--limit-mm-per-prompt", json.dumps({"image": max_images}),
            "--enable-prefix-caching",
        ] + (extra_args or [])
        self.proc: Optional[subprocess.Popen] = None
        self.kv_tokens: Optional[int] = None
        self.weights_gib: Optional[float] = None
        self.milestones: List[tuple] = []
        self._lines: List[str] = []

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.log_path.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True,
                                     bufsize=1)
        threading.Thread(target=self._tail, daemon=True).start()

    def _tail(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            self._fh.write(line)
            self._fh.flush()
            self._lines.append(line)
            m = KV_RE.search(line)
            if m:
                self.kv_tokens = int(m.group(1).replace(",", ""))
            m = WEIGHTS_RE.search(line)
            if m:
                self.weights_gib = float(m.group(1))
            for label, rx in VLLM_MILESTONES:
                m = rx.search(line)
                if m:
                    self.milestones.append((label, float(m.group(1))))

    def wait_ready(self, timeout_s: float, poll_s: float = 0.5) -> float:
        """Poll /health until it answers. Returns seconds waited.

        Polling rather than sleeping is the whole point: a fixed pre-wait spends
        budget on a server that may already be up, and 15 minutes does not have
        90 seconds to give away.
        """
        url = f"http://127.0.0.1:{self.port}/health"
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited with code {self.proc.returncode} during startup "
                    f"-- see {self.log_path}")
            try:
                with urllib.request.urlopen(url, timeout=2):
                    return time.time() - t0
            except Exception:
                time.sleep(poll_s)
        raise TimeoutError(f"vLLM not ready after {timeout_s:.0f}s -- see {self.log_path}")

    def concurrency(self, n_calls: int) -> tuple:
        """(chosen concurrency, why) -- sized off the pool the server reports.

        Two ceilings, and the smaller wins. The KV pool is one; the number of
        calls in the case is the other, because the calls are batched within a
        case and there is nothing else to fill the slots with.
        """
        if not self.kv_tokens:
            return FALLBACK_CONCURRENCY, "KV size not seen in log; fallback"
        by_kv = max(1, self.kv_tokens // TOKENS_PER_REQUEST)
        if by_kv <= n_calls:
            return by_kv, f"KV pool {self.kv_tokens:,} tok / {TOKENS_PER_REQUEST} = {by_kv}"
        return n_calls, (f"case has {n_calls} calls (KV would allow {by_kv})")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def run_generator(script: str, args: List[str], phases: Phases,
                  label: str) -> None:
    phases.start(label)
    proc = subprocess.run([sys.executable, str(module_path(script))] + args,
                          capture_output=True, text=True)
    phases.end(label)
    if proc.returncode != 0:
        print(f"[FAIL] {script}: {proc.stderr[-1500:]}", file=sys.stderr)
        raise RuntimeError(f"{script} exited {proc.returncode}")


def prepare_facts(case: str, mask: Path, facts_in: Optional[Path],
                  out_dir: Path, phases: Phases) -> Optional[Path]:
    """
    Audit the case's facts against its mask, and hand back the audited copy.

    THE REAL INPUT SHAPE. A CBCT arrives; the segmentation AND the facts come
    from upstream (the segmentation component), so this runner does not extract
    them -- it audits what it is given and uses the result for captions and for
    the source rules. Omitting --facts-file derives them here instead, with
    extract_facts.py, for a case where no upstream facts exist; that costs ~65 s
    against the audit's ~2 s and is a fallback, not the shipping path.

    WHY THE AUDIT CANNOT BE SKIPPED, which is the thing that was missing:

      * `fov.maxilla := excluded` when the maxillary bone is below the
        renderer's visibility threshold. THE RULE -- absent teeth gates on
        exactly this field, and without it every upper position lands in
        `teeth_absent` for a maxilla that was never in the volume.
      * `bridge_arches`, derived from the mask's bridge label (8). THE RULE --
        fixed bridges is sourced from this field and does nothing when it is
        missing -- and "missing" is indistinguishable from "no bridge" unless
        the audit has run, which is why it writes the key even when empty.

    Neither is a correction to the upstream facts' clinical content; both are
    statements about the ACQUISITION that only the mask can make.

    THE COPY IS NOT INCIDENTAL. audit_facts.py rewrites in place with
    --execute, and on Grand Challenge the input is read-only, so the file is
    copied into the run directory first and the copy is what everything
    downstream reads.

    CPU-only, and off the critical path -- it runs while vLLM loads weights.
    """
    facts_dir = out_dir / "facts"
    facts_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = out_dir / "facts_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    link = mask_dir / f"{case}.nii.gz"
    if not link.exists():
        try:
            link.symlink_to(mask.resolve())
        except OSError:
            shutil.copy2(mask, link)

    target = facts_dir / f"{case}.json"
    if facts_in is not None and not Path(facts_in).exists():
        # Upstream said it would hand over facts and did not. Deriving them
        # from the mask is strictly better than running with none: the source
        # rules and the caption tooth list both come back, and the only thing
        # lost is whatever the upstream file knew that the mask does not.
        print(f"[WARN] upstream facts file missing: {facts_in} -- deriving "
              f"from the mask instead (~65 s)", file=sys.stderr)
        facts_in = None

    if facts_in is not None:
        phases.start("facts:copy")
        shutil.copy2(facts_in, target)
        phases.end("facts:copy")
    else:
        # extract_facts.py IS NOT OURS AND IS NOT SHIPPED. Facts come from the
        # segmentation component, which produces them alongside the mask, so on
        # Grand Challenge --facts-file is always available and this branch is a
        # research convenience for a case with no upstream facts (~65 s against
        # the audit's ~2 s). A container packaged from the competition manifest
        # therefore will not contain the script, and the resulting
        # FileNotFoundError points at a path rather than at the mistake.
        extractor = module_path("extract_facts.py")
        if not extractor.exists():
            sys.exit(
                f"[FAIL] no --facts-file given, and {extractor} is not present.\n"
                f"       Deriving facts here is the fallback path and "
                f"extract_facts.py belongs to the segmentation component, so it "
                f"is not packaged for the container.\n"
                f"       Pass --facts-file (the segmentation half produces it "
                f"next to the mask), or --no-facts to run without any.")
        phases.start("facts:extract")
        subprocess.run([sys.executable, str(extractor),
                        "-i", str(mask_dir), "-o", str(out_dir / "facts_build")],
                       check=True, capture_output=True, text=True)
        built = out_dir / "facts_build" / "facts" / f"{case}.json"
        if not built.exists():
            print(f"[WARN] extract_facts produced no facts for {case}",
                  file=sys.stderr)
            phases.end("facts:extract")
            return None
        shutil.copy2(built, target)
        phases.end("facts:extract")

    phases.start("facts:audit")
    proc = subprocess.run([sys.executable, str(module_path("audit_facts.py")),
                           "--facts-dir", str(facts_dir),
                           "--masks-dir", str(mask_dir),
                           # This case only. The dir holds one file today, but
                           # an out-dir reused across cases would otherwise
                           # re-audit and re-time every case that came before.
                           "--cases", case,
                           "--derive-bridge-arches", "--execute"],
                          capture_output=True, text=True)
    phases.end("facts:audit")
    # Print what the audit did. It is the one stage that rewrites an input the
    # rest of the run then treats as given, and captured-and-discarded stdout
    # is how "the FOV correction never fired" stays invisible for a week.
    for line in (proc.stdout or "").splitlines():
        if line.strip() and not line.startswith("[INFO] Auditing"):
            print(f"[facts:audit] {line}")
    if proc.returncode != 0:
        # A failed audit is not fatal: the facts still describe the case, they
        # simply have not been reconciled with the acquisition. Say so loudly
        # rather than dropping them, because the two rules above will then be
        # quietly inert.
        print(f"[WARN] audit_facts failed ({proc.returncode}); using unaudited "
              f"facts -- absent-teeth FOV gating and the bridge rule are OFF\n"
              f"{proc.stderr[-800:]}", file=sys.stderr)
    return target


def generate_images(case: str, volume: Path, mask: Path, facts: Optional[Path],
                    out_dir: Path, phases: Phases) -> None:
    """The four generators, in dependency order.

    panoramic / 3d / sinus are independent of each other; the composites need
    --pano-dir and --render-dir, so they wait for the first two. The three
    independent ones run concurrently because on a 4-vCPU A10G instance they are
    the only thing competing for CPU while the GPU loads weights.
    """
    common = ["--case-id", case, "--out-dir", str(out_dir)]
    vol = ["--volume", str(volume)]
    msk = ["--mask", str(mask)]
    fct = ["--facts-file", str(facts)] if facts else ["--no-facts"]

    errors: List[BaseException] = []

    def guard(fn, *a):
        try:
            fn(*a)
        except BaseException as e:          # noqa: BLE001 - reported below
            errors.append(e)

    first = [
        threading.Thread(target=guard, args=(
            run_generator, "create_panoramic.py",
            common + vol + msk + fct, phases, "img:panoramic")),
        threading.Thread(target=guard, args=(
            run_generator, "create_3d_renders.py",
            common + msk + fct, phases, "img:3d")),
        threading.Thread(target=guard, args=(
            run_generator, "create_sinus_detail.py",
            common + vol + msk, phases, "img:sinus")),
    ]
    for t in first:
        t.start()
    for t in first:
        t.join()
    if errors:
        raise errors[0]

    run_generator("create_tooth_detail.py",
                  common + vol + msk + fct
                  + ["--pano-dir", str(out_dir), "--render-dir", str(out_dir)],
                  phases, "img:composites")


def resolve_cases(args) -> List[tuple]:
    """(case_id, volume, mask, facts_in) for every case this run covers.

    Two shapes, and they are exclusive rather than merged: --case-id names one
    case and its three files explicitly, which is what a container is handed;
    --dataset-dir names a split and the files are found by the layout
    convention. A run that accepted both would have to decide which wins, and
    the answer would be a footnote instead of an error.
    """
    if args.case_id:
        return [(args.case_id, args.volume, args.mask, args.facts_file)]

    root = args.dataset_dir
    images, masks = root / "images", root / "masks"
    if not images.is_dir():
        sys.exit(f"[FAIL] {images} is not a directory -- --dataset-dir wants "
                 f"images/ and masks/ under it")
    found = sorted(f.name[:-len("_0000.nii.gz")]
                   for f in images.glob("*_0000.nii.gz"))
    if args.case_list:
        wanted = [l.strip() for l in args.case_list.read_text().splitlines()
                  if l.strip() and not l.startswith("#")]
    elif args.case_ids:
        wanted = list(args.case_ids)
    else:
        wanted = found
    missing = [c for c in wanted if c not in set(found)]
    if missing:
        sys.exit(f"[FAIL] no volume under {images} for: {missing[:8]}")
    if args.limit:
        # Seeded, so LIMIT=N picks the SAME N cases every run -- a smoke test
        # that samples differently each time cannot be compared with itself.
        rng = random.Random(args.seed)
        wanted = sorted(rng.sample(wanted, min(args.limit, len(wanted))))

    out = []
    for c in wanted:
        facts = root / "facts" / f"{c}.json"
        out.append((c, images / f"{c}_0000.nii.gz", masks / f"{c}.nii.gz",
                    facts if facts.exists() else None))
    return out


def already_rendered(case: str, images_dir: Path) -> bool:
    """Has this case's whole render finished?

    Keyed on {case}_coverage.json, which create_tooth_detail.py writes
    UNCONDITIONALLY and which the last generator in generate_images() writes
    last -- so its presence means all four ran, and its absence means at least
    one did not.

    NOT the tooth caption sidecar, which is the obvious choice and is wrong:
    create_tooth_detail.py writes it only `if captions`, so an edentulous case
    finishes perfectly and leaves none. 4 of the 40 validate cases are like
    that, and keying on it would re-render each of them on every resumed run
    while reporting nothing amiss."""
    return (images_dir / f"{case}_coverage.json").exists()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case-id",
                    help="ONE case, with --volume/--mask given explicitly. "
                         "This is the container shape. Mutually exclusive with "
                         "--dataset-dir.")
    ap.add_argument("--volume", type=Path)
    ap.add_argument("--mask", type=Path)
    ap.add_argument("--facts-file", type=Path,
                    help="The case's facts, from upstream. Audited against the "
                         "mask before use; the original is never written to. "
                         "Omit to derive them here with extract_facts.py.")
    ap.add_argument("--dataset-dir", type=Path,
                    help="A SPLIT: dataset/validate, holding images/ masks/ "
                         "and optionally facts/. Every case in it runs, sharing "
                         "one vLLM load. Files are found by the layout "
                         "convention -- {case}_0000.nii.gz, {case}.nii.gz, "
                         "{case}.json.")
    ap.add_argument("--case-ids", nargs="+", help="restrict --dataset-dir to these")
    ap.add_argument("--case-list", type=Path,
                    help="restrict --dataset-dir to the case ids in this file, "
                         "one per line")
    ap.add_argument("--limit", type=int,
                    help="smoke test: a seeded sample of --dataset-dir")
    ap.add_argument("--seed", type=int, default=42,
                    help="which cases --limit picks (default 42, so the same "
                         "ones every run)")
    ap.add_argument("--resume", action="store_true",
                    help="skip a case whose images and prediction already "
                         "exist. Rendering is the expensive half of a batch "
                         "run and a killed job should not repeat it.")
    ap.add_argument("--no-facts", action="store_true",
                    help="No facts at all: captions carry none and the source "
                         "rules stay off. The NO_FACTS=1 arm.")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="images/, facts/, predictions/, summaries/ and "
                         "reports/ are written under here, one set for the "
                         "whole run rather than one per case")
    ap.add_argument("--model", required=True,
                    help="path INSIDE the container. The shipping model is the "
                         "trained arm, models/best_model (the LoRA merged "
                         "into the AWQ base); code/competition/competition_sim.sh's header "
                         "records what that arm wins and where it regresses.")
    ap.add_argument("--schema", type=Path, default=Path("schema/schema.json"))
    ap.add_argument("--project-dir", type=Path, default=Path("."))
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-model-len", type=int, default=32768,
                    help="Must hold prompt + reply. The global calls request "
                         "8192 output tokens, so anything at or below 8192 "
                         "makes every one of them fail with a 400 while the "
                         "tooth calls still pass.")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-images-per-prompt", type=int, default=3)
    ap.add_argument("--startup-timeout", type=float, default=900.0)
    ap.add_argument("--max-concurrency", type=int, default=0,
                    help="0 = auto from the server's reported KV cache")
    ap.add_argument("--no-server", action="store_true",
                    help="images and timing only; skip vLLM entirely")
    args = ap.parse_args()

    if bool(args.case_id) == bool(args.dataset_dir):
        ap.error("give either --case-id (with --volume and --mask) or "
                 "--dataset-dir, not both and not neither")
    if args.case_id and not (args.volume and args.mask):
        ap.error("--case-id needs --volume and --mask")

    phases = Phases()
    out_dir = args.out_dir
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Resolved BEFORE the server starts: a typo in --case-list should cost a
    # second, not a model load.
    cases = resolve_cases(args)
    if len(cases) > 1:
        print(f"[INFO] {len(cases)} case(s): {', '.join(c[0] for c in cases[:6])}"
              f"{' ...' if len(cases) > 6 else ''}", file=sys.stderr)

    server: Optional[VLLMServer] = None
    server_err: List[BaseException] = []
    ready_s = 0.0

    # ── vLLM first, in the background: it is the long pole and needs no CPU ──
    if not args.no_server:
        server = VLLMServer(
            args.model, args.port, args.max_model_len,
            args.gpu_memory_utilization, args.max_images_per_prompt,
            out_dir / "vllm_startup.log")
        phases.start("vllm:startup")
        server.start()

        def wait():
            nonlocal ready_s
            try:
                ready_s = server.wait_ready(args.startup_timeout)
            except BaseException as e:       # noqa: BLE001
                server_err.append(e)
            finally:
                phases.end("vllm:startup")

        waiter = threading.Thread(target=wait, daemon=True)
        waiter.start()

    # ── facts audited against the mask, then the images -- while vLLM loads ─
    # --facts-file IS the competition path: upstream hands over the mask and
    # the facts together. Omitting it falls back to deriving them here, which
    # is the same information at ~65 s instead of ~2 s.
    #
    # Both facts and images are written per case into ONE directory each, which
    # is what makes the batch loop this small: prepare_facts() already audits
    # --cases <this one> so a reused out-dir does not re-audit its
    # predecessors, and every stage after the loop is directory-shaped
    # already.
    facts_dir: Optional[Path] = None
    phases.start("images:total")
    for case, volume, mask, facts_in in cases:
        # --resume needs BOTH halves present. The audited facts are checked
        # separately from the images because postprocess reads the facts
        # DIRECTORY: one case missing its copy turns the source rules off for
        # that case and says nothing. The audit is not free either -- it reads
        # the mask, which is tens of seconds on a network filesystem, not the
        # ~2 s it costs a container with the volume on local disk.
        facts_target = out_dir / "facts" / f"{case}.json"
        have_facts = args.no_facts or facts_target.exists()
        if args.resume and already_rendered(case, images_dir) and have_facts:
            if not args.no_facts:
                facts_dir = facts_target.parent
            print(f"[skip] {case} (images and facts present)", file=sys.stderr)
            continue

        if len(cases) > 1:
            print(f"[case] {case}", file=sys.stderr)
        facts_file = None
        if not args.no_facts:
            facts_file = prepare_facts(case, mask, facts_in, out_dir, phases)
            if facts_file is not None:
                facts_dir = Path(facts_file).parent
        if args.resume and already_rendered(case, images_dir):
            continue
        generate_images(case, volume, mask, facts_file, images_dir, phases)
    phases.end("images:total")

    if server:
        waiter.join()
        if server_err:
            print(f"[FAIL] {server_err[0]}", file=sys.stderr)
            server.stop()
            sys.exit(2)

    # ── qa_pairs, then inference ────────────────────────────────────────────
    phases.start("qa_pairs")
    qa_jsonl = out_dir / "qa_pairs.jsonl"
    # --cases, always, and not because the images dir usually holds only what
    # this run rendered. It does in a container; it does not when an out-dir is
    # reused or --resume points at a whole split's images, and build_vqa_pairs
    # plans from the FILES ON DISK. Without it, asking for three cases and
    # getting forty is silent -- the payload is well-formed, just not the one
    # that was requested, and the extra thirty-seven are then inferred.
    subprocess.run([sys.executable, str(module_path("build_vqa_pairs.py")),
                    "--images-dir", str(images_dir), "--schema", str(args.schema),
                    "--out", str(qa_jsonl), "--project-dir", str(args.project_dir),
                    "--cases"] + [c[0] for c in cases],
                   check=True, capture_output=True, text=True)
    phases.end("qa_pairs")

    n_calls = 0
    if qa_jsonl.exists():
        # The BIGGEST case, not the first: concurrency is sized so one case's
        # calls fit the KV pool, and under-sizing on a 20-tooth case would
        # queue the 32-tooth one behind itself.
        for line in qa_jsonl.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            n_calls = max(n_calls, sum(len(rec.get(k) or {})
                                       for k in ("global", "dental_elements")))

    if not args.no_server:
        conc, why = ((args.max_concurrency, "set by --max-concurrency")
                     if args.max_concurrency else server.concurrency(n_calls))
        print(f"[INFO] concurrency {conc} -- {why}", file=sys.stderr)

        phases.start("inference")
        subprocess.run([sys.executable, str(module_path("run_vqa_inference.py")),
                        "--vqa-jsonl", str(qa_jsonl),
                        "--out-dir", str(out_dir / "predictions"),
                        "--base-dir", str(args.project_dir),
                        "--model", "qwen3.5-vl",
                        "--vllm-url", f"http://127.0.0.1:{args.port}/v1",
                        "--max-concurrency", str(conc)]
                       + (["--resume"] if args.resume else []),
                       check=True)
        phases.end("inference")

        phases.start("postprocess")
        # --facts-dir turns on THE SOURCE RULES (docs/postprocess_pipeline.md): ten
        # findings re-sourced from the segmentation instead of the VLM reads.
        # Without it postprocess emits the pre-2026-08-16 summary and every
        # rule measured on validate-40 is silently absent from the submission.
        post = [sys.executable, str(module_path("postprocess_pred.py")),
                "--pred-dir", str(out_dir / "predictions"),
                "--out-dir", str(out_dir / "summaries")]
        if facts_dir is not None:
            post += ["--facts-dir", str(facts_dir)]
        subprocess.run(post, check=True)
        phases.end("postprocess")

        phases.start("synthesize")
        subprocess.run([sys.executable, str(module_path("synthesize_report.py")),
                        "--summary-dir", str(out_dir / "summaries"),
                        "--out-dir", str(out_dir / "reports")], check=True)
        phases.end("synthesize")

        server.stop()

    # ── the report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("COMPETITION SIMULATION -- "
          + (f"case {cases[0][0]}" if len(cases) == 1
             else f"{len(cases)} cases, one server"))
    print("=" * 78)
    print(phases.table())
    if server:
        print(f"\nvLLM: ready after {ready_s:.1f}s"
              f"{f', weights {server.weights_gib} GiB' if server.weights_gib else ''}"
              f"{f', KV {server.kv_tokens:,} tokens' if server.kv_tokens else ''}")
        if server.milestones:
            print("  startup breakdown (vLLM's own numbers):")
            for label, secs in server.milestones:
                print(f"    {label:16} {secs:7.1f}s")
            # "engine init" is the OUTER span -- it already contains
            # torch.compile and CUDA graph capture, so the top-level total is
            # model-load + engine-init only. Summing every line double-counts
            # and drives the remainder negative.
            outer = {"model -> GPU", "engine init"}
            named = sum(v for k, v in server.milestones if k in outer)
            print(f"    {'unattributed':16} {ready_s - named:7.1f}s"
                  "   <- spawn, imports, CUDA context, API server bind")
    total = time.time() - phases.t0
    budget = 15 * 60
    if len(cases) == 1:
        print(f"\n{'WITHIN' if total < budget else 'OVER'} the 15-minute budget: "
              f"{total:.0f}s of {budget}s ({100 * total / budget:.0f}%)")
    else:
        # The 15-minute budget is per CONTAINER START, and a batch run starts
        # one server for all of them -- so the budget does not apply and
        # printing it against the total would be a meaningless failure. What a
        # batch run says about the budget is the per-case cost with the load
        # already paid, which is the optimistic half of it.
        print(f"\n{len(cases)} cases in {total:.0f}s -- {total / len(cases):.0f}s "
              f"per case with the model load shared. The 15-minute budget is "
              f"per container start; use --case-id to measure it.")


if __name__ == "__main__":
    main()
