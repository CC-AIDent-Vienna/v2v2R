#!/usr/bin/env python3
"""
check_evidence_perceivable.py -- can the STUDENT see what the teacher described?

docs/vision_sft_plan.md §3.3's screen, at pool scale. build_fewshot_exemplars.py
--check-perceivable does this for a handful of exemplars with a human rewriting
the failures; here it runs over every drafted evidence string and reports a rate
per field, because at ~1,850 calls nobody rewrites anything by hand.

WHY THIS IS THE LOAD-BEARING CHECK
──────────────────────────────────
The teacher was handed the answer and told not to re-judge it (DRAFT_SYSTEM).
That is what makes the label safe, and it is also what makes the prose
dangerous: a teacher that cannot see a finding will still write confident,
specific support for it, because it was told the finding is established. Nothing
downstream can tell the difference -- the JSON is well-formed either way.

Description from knowledge of the feature class is fine and is the point of
having a teacher at all. What is not fine is memory dressed as observation:
prose naming THIS image's specifics -- a tile, a neighbour, a relation -- that
are not there. Training on that teaches the student to assert what it cannot
verify, which is the over-calling the whole experiment is trying to remove.

So the question is put to the STUDENT, not the teacher: PERCEIVE_SYSTEM asks it,
strictly, whether each concrete feature the description claims is visible in
this image. The teacher cannot be asked -- it wrote the prose.

`STAGE=perceive` measured ~40% cue confirmation on four tooth-class exemplar
files against ~85% on sinus, which is why this is a gate rather than a report.

READING IT
──────────
Per field: `visible / claimed`. Compare fields the teacher can FIND (endodontic,
fillings, restoration scored well at audition) against ones it cannot
(post_and_core). Comparable rates mean the teacher is describing honestly from
class knowledge; a much worse rate on the fields it cannot find means it is
inventing, and that field's evidence should be dropped.

--out-dir writes a FILTERED copy of the evidence: rows whose confirmation falls
below --min-visible are dropped, so build_sft_targets.py --evidence-from can be
pointed at screened prose rather than raw prose.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


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

ROOT = REPO_ROOT

import run_vqa_inference as RVI                              # noqa: E402
from build_fewshot_exemplars import PERCEIVE_SYSTEM          # noqa: E402


def judge(client, model, call: dict, text: str, max_tokens: int):
    blocks = RVI.build_captioned_image_blocks(call.get("images") or {},
                                              call.get("captions") or {})
    if not blocks:
        return None
    user = blocks + [{"type": "text", "text": f"DESCRIPTION TO CHECK:\n{text}"}]
    raw = RVI.call_vllm_messages(
        client, model,
        [{"role": "system", "content": PERCEIVE_SYSTEM},
         {"role": "user", "content": user}], max_tokens=max_tokens)
    v = RVI.parse_json(raw)
    feats = v.get("features") or []
    vis = [f for f in feats if f.get("visible")]
    return len(vis), len(feats), [f.get("feature") for f in feats
                                  if not f.get("visible")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evidence-dir", type=Path, required=True)
    ap.add_argument("--qa-jsonl", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path,
                    help="write a filtered copy keeping only rows at or above "
                         "--min-visible")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--base-dir", default=str(ROOT),
                    help="resolve qa_pairs image paths against this -- they are "
                         "stored relative to the project dir, so CWD must not "
                         "decide whether an image is attached")
    ap.add_argument("--vllm-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="student",
                    help="the STUDENT under test -- never the teacher")
    ap.add_argument("--min-visible", type=float, default=0.5,
                    help="fraction of claimed features the student must confirm")
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int)
    # Same crash/timeout rule as draft_evidence.py: the verdict stream is the
    # record, everything else is derived from it. Shorter pass, same failure --
    # a screen that dies near the end and reports nothing has cost a GPU hour
    # and taught nothing.
    ap.add_argument("--flush-every", type=int, default=200)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--max-failure-rate", type=float, default=0.05,
                    help="refuse to write filtered evidence if more than this "
                         "fraction of calls produced no verdict")
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.vllm_url, api_key="EMPTY")

    calls = {}
    for line in args.qa_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        for tk, c in (rec.get("dental_elements") or {}).items():
            c = dict(c)
            c["images"] = {k: RVI.resolve_path(v, args.base_dir)
                           for k, v in (c.get("images") or {}).items() if v}
            calls[(rec["case_id"], tk)] = c

    jobs = []
    for p in sorted(args.evidence_dir.glob("*_pred.json")):
        blob = json.loads(p.read_text(encoding="utf-8"))
        case = blob.get("case_id") or p.name.split("_pred")[0]
        for tk, facts in (blob.get("teeth") or {}).items():
            call = calls.get((case, tk))
            if not call:
                continue
            for fact_id, text in facts.items():
                if isinstance(text, str) and text.strip():
                    jobs.append((case, tk, fact_id, call, text))
    if args.limit:
        jobs = jobs[:args.limit]

    # The verdict stream: appended and flushed per answer, and also the resume
    # record. Everything below is derived from it, so a killed job keeps what it
    # screened and a rerun continues.
    stream = (args.report.parent if args.report else args.evidence_dir.parent)
    stream = stream / f"{args.evidence_dir.name}_verdicts.jsonl"
    stream.parent.mkdir(parents=True, exist_ok=True)

    results = []
    seen = set()
    if stream.exists() and not args.no_resume:
        for line in stream.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                     # torn by a kill -- skip, not fatal
            results.append(r)
            seen.add((r["case"], r["tooth"], r["fact"]))
        jobs = [j for j in jobs if (j[0], j[1], j[2]) not in seen]
        print(f"[INFO] resuming from {stream}: {len(seen)} already screened, "
              f"{len(jobs)} left")

    print(f"[INFO] {len(jobs)} evidence string(s) to screen against the STUDENT "
          f"({args.model}), concurrency {args.concurrency}")

    per_field = collections.defaultdict(lambda: [0, 0, 0, 0])   # vis, claimed, rows, kept
    for r in results:                        # fold resumed rows into the tally
        s = per_field[r["fact"].split("_", 2)[2]]
        s[0] += r["visible"]; s[1] += r["claimed"]; s[2] += 1; s[3] += int(r["kept"])
    lock = threading.Lock()
    failures = collections.Counter()

    def run(j):
        case, tk, fact_id, call, text = j
        try:
            return j, judge(client, args.model, call, text, args.max_tokens), None
        except Exception as e:                                   # noqa: BLE001
            return j, None, f"{type(e).__name__}"

    done = 0
    fh = stream.open("a", encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for fut in as_completed([pool.submit(run, j) for j in jobs]):
                (case, tk, fact_id, _c, text), out, err = fut.result()
                done += 1
                if err or out is None:
                    # NOT a pass. An unjudged string is unknown, and the safe
                    # reading of unknown is drop -- the opposite default let a
                    # screen in which every single call failed print
                    # "9301 string(s) kept" and hand unscreened prose to
                    # build_sft_targets.py (2026-08-14).
                    with lock:
                        failures[err or "no_image"] += 1
                    continue
                vis, claimed, missed = out
                field = fact_id.split("_", 2)[2]
                frac = (vis / claimed) if claimed else 0.0
                keep = frac >= args.min_visible
                row = {"case": case, "tooth": tk, "fact": fact_id,
                       "visible": vis, "claimed": claimed,
                       "kept": keep, "not_visible": missed[:4]}
                with lock:
                    s = per_field[field]
                    s[0] += vis; s[1] += claimed; s[2] += 1; s[3] += int(keep)
                    results.append(row)
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                    if done % args.flush_every == 0:
                        print(f"  {done}/{len(jobs)}", file=sys.stderr)
    finally:
        fh.close()

    print(f"\n{'field':34}{'rows':>7}{'claimed':>9}{'visible':>9}{'rate':>8}"
          f"{'rows kept':>11}")
    tot = [0, 0, 0, 0]
    for f in sorted(per_field):
        vis, cl, rows, kept = per_field[f]
        for i, v in enumerate((vis, cl, rows, kept)):
            tot[i] += v
        print(f"{f:34}{rows:>7}{cl:>9}{vis:>9}"
              f"{(vis/cl if cl else 0):>8.2f}{kept:>7} ({100*kept/rows if rows else 0:.0f}%)")
    print(f"{'TOTAL':34}{tot[2]:>7}{tot[1]:>9}{tot[0]:>9}"
          f"{(tot[0]/tot[1] if tot[1] else 0):>8.2f}{tot[3]:>7} "
          f"({100*tot[3]/tot[2] if tot[2] else 0:.0f}%)")
    print("\n[NOTE] compare fields the teacher can FIND (endodontic, fillings,")
    print("       restoration) against ones it cannot (post_and_core). A similar")
    print("       rate means honest class-level description; a much worse one")
    print("       means invention, and that field's evidence should be dropped.")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(
            {"min_visible": args.min_visible,
             "per_field": {f: dict(zip(("visible", "claimed", "rows", "kept"),
                                       v)) for f, v in per_field.items()},
             "rows": results}, indent=2) + "\n", encoding="utf-8")
        print(f"[INFO] {args.report}")

    if failures:
        rate = sum(failures.values()) / max(1, sum(failures.values()) + len(results))
        print(f"\n[WARN] {sum(failures.values())} call(s) produced no verdict "
              f"({100 * rate:.0f}%): {dict(failures)}")
        if rate > args.max_failure_rate:
            print(f"[FAIL] failure rate {100 * rate:.0f}% exceeds "
                  f"--max-failure-rate {100 * args.max_failure_rate:.0f}%. "
                  f"Refusing to write a filtered evidence dir: a screen that "
                  f"did not run is not a screen that passed.")
            return 1

    if args.out_dir:
        judged = {(r["case"], r["tooth"], r["fact"]) for r in results}
        drop = {k for k in judged
                if not next(r["kept"] for r in results
                            if (r["case"], r["tooth"], r["fact"]) == k)}
        args.out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in sorted(args.evidence_dir.glob("*_pred.json")):
            blob = json.loads(p.read_text(encoding="utf-8"))
            case = blob.get("case_id") or p.name.split("_pred")[0]
            teeth = {}
            for tk, facts in (blob.get("teeth") or {}).items():
                kept = {k: v for k, v in facts.items()
                        if (case, tk, k) in judged and (case, tk, k) not in drop}
                if kept:
                    teeth[tk] = kept
                    n += len(kept)
            (args.out_dir / p.name).write_text(json.dumps(
                {"case_id": case, "teeth": teeth}, ensure_ascii=False,
                indent=2) + "\n", encoding="utf-8")
        print(f"[INFO] filtered evidence: {n} string(s) kept -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
