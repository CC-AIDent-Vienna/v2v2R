#!/usr/bin/env python3
"""
draft_evidence.py -- the teacher writes visual_evidence for answers it is handed.

docs/vision_sft_plan.md §3.3, at pool scale. build_fewshot_exemplars.py does this for
a handful of exemplar files with a human reading every one; this does it for the
~1,850 tooth calls of the SFT pool, and imports that file's DRAFT_SYSTEM and
DRAFT_USER_TEMPLATE rather than restating them -- two copies of a prompt drift,
and the drift is invisible.

WHAT THE TEACHER IS AND IS NOT ASKED
────────────────────────────────────
It is handed the finished answer, derived from the reports and facts.structured,
and told (in DRAFT_SYSTEM) that those findings "are ALREADY ESTABLISHED ... you
must not re-judge them". It writes only what in this image supports them. The
label never comes from the teacher, which is the property that makes rationale
distillation safe here at all.

That also means DETECTION and DESCRIPTION are different jobs, and the teacher is
only doing the second. A model that cannot reliably find a post and core unaided
can still write what one looks like -- a wide radiopaque column running from the
crown into the coronal third, wider and brighter than the filling continuing to
the apex -- because that is knowledge of the feature class, not a reading of
this picture. Audition scores (code/pipeline/infer/pool_infer.sh) therefore bound what the
teacher can FIND, not what it can usefully SAY.

The failure that remains is memory dressed as observation: prose that invents
this image's specifics -- a tile, a neighbour, a relation that is not there.
DRAFT_SYSTEM already carries the guard ("If a finding's evidence is not clearly
visible in this image, say plainly which part is not visible rather than
inventing support for it"), and --check-perceivable is what tests whether it was
obeyed. This script's job is to produce the prose and record what was asked.

WHAT IS SENT
────────────
Only fields that are actually SUPERVISED get an evidence request. A null in the
GT means the reference did not answer there (§3.2), and asking a teacher to
justify a null is asking it to invent one.

Output is written as {case}_pred.json shaped exactly like a predictions file --
{"teeth": {"tooth_46": {"tooth_46_morphology": "<prose>", ...}}} -- because that
is what build_sft_targets.py --evidence-from already reads.

    python code/train/draft_evidence.py --qa-jsonl outputs/training_results/vsft_pool_training/qa_pairs.jsonl \\
        --gt-dir dataset/training/outputs/ground_truth \\
        --case-list outputs/training_results/sft_pool/wide.txt \\
        --out-dir outputs/training_results/vsft_pool_training/evidence_27b \\
        --vllm-url http://localhost:8000/v1 --model qwen-under-test
"""

from __future__ import annotations

import argparse
import collections
import json
import os
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

import run_vqa_inference as RVI                       # noqa: E402
from evidence_prompts import (                        # noqa: E402  -- one copy of the prompt
    DRAFT_SYSTEM, DRAFT_USER_TEMPLATE,
)
from build_sft_targets import (                       # noqa: E402  -- one copy of the mask
    DEFAULT_REFUSED, EVIDENCE_FIELD, PROSE_FIELDS, schema_field_order,
)


def established_answer(block: dict, fdi: int, order: dict, refused: set):
    """The finished findings for this tooth, minus anything not supervised.

    Nulls are dropped rather than sent as null: the reference did not answer
    there, and a teacher asked to support a null will support something.
    """
    out = {}
    for tmpl, fields in order.items():
        fact_id = tmpl.replace("{fdi}", str(fdi))
        src = block.get(fact_id)
        if src is None:
            continue
        kept = {f: src[f] for f in fields
                if f in src and src[f] is not None
                and f != EVIDENCE_FIELD and f not in PROSE_FIELDS
                and f not in refused}
        if kept:
            out[fact_id] = kept
    return out


class NoImage(RuntimeError):
    """Refuse to draft evidence for an image that was not attached.

    build_captioned_image_blocks() drops any path that does not resolve and
    says nothing, and qa_pairs.jsonl stores paths RELATIVE to the project dir
    so that the same file works on the host and in the container. Run from any
    other working directory, every image silently disappears -- and a
    text-only request is still perfectly valid, so the teacher happily writes
    fluent prose from the answer JSON alone. That is exactly the
    memory-dressed-as-observation failure this pipeline exists to avoid,
    manufactured by the harness rather than the model, and it produced 9,301
    unusable strings on 2026-08-14 before anything noticed.

    So: no image, no draft. Loud, not skipped.
    """


def draft_one(client, model, call: dict, fdi: int, answer: dict,
              max_tokens: int) -> dict:
    keys = list(answer)
    user_text = DRAFT_USER_TEMPLATE.format(
        answer_json=json.dumps(answer, ensure_ascii=False, indent=1),
        keys=", ".join(keys),
        key_list=", ".join(f'"{k}": "..."' for k in keys))
    blocks = [{"type": "text", "text": user_text}]
    img_blocks = RVI.build_captioned_image_blocks(call.get("images") or {},
                                                  call.get("captions") or {})
    if not any(b.get("type") == "image_url" for b in img_blocks):
        raise NoImage(f"tooth {fdi}: no image resolved from "
                      f"{list((call.get('images') or {}).values())[:2]}")
    blocks += img_blocks
    raw = RVI.call_vllm_messages(
        client, model,
        [{"role": "system", "content": DRAFT_SYSTEM},
         {"role": "user", "content": blocks}],
        max_tokens=max_tokens)
    got = RVI.parse_json(raw)
    return {k: v for k, v in got.items() if k in answer and isinstance(v, str)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa-jsonl", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--case-list", type=Path)
    ap.add_argument("--schema", type=Path, default=ROOT / "schema/schema.json")
    ap.add_argument("--vllm-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="qwen-under-test")
    # qa_pairs.jsonl stores image paths relative to the project dir, on purpose,
    # so one payload resolves on the host and at /project in the container.
    # Resolve them explicitly rather than relying on the working directory.
    ap.add_argument("--base-dir", default=str(ROOT),
                    help="resolve qa_pairs image paths against this")
    # Prose, not a report: ~300 tokens is plenty, and decode dominates
    # wall-clock at these batch sizes. The pipeline's 4096 would triple it.
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int)
    # Crash/timeout safety. Every answer is appended to a JSONL the moment it
    # arrives, and the per-case files are rebuilt from it periodically. A job
    # that dies at hour 7 of 8 then keeps everything it had done, and rerunning
    # continues instead of starting over -- which for a teacher pass is hours of
    # GPU either way.
    ap.add_argument("--flush-every", type=int, default=100,
                    help="rebuild the {case}_pred.json files every N answers")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore an existing drafts.jsonl and redo everything")
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.vllm_url, api_key="EMPTY")

    order = schema_field_order(json.loads(args.schema.read_text()))
    wanted = None
    if args.case_list:
        wanted = {l.strip() for l in args.case_list.read_text().splitlines()
                  if l.strip() and not l.startswith("#")}

    jobs = []
    for line in args.qa_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        case = rec["case_id"]
        if wanted is not None and case not in wanted:
            continue
        gt_path = args.gt_dir / f"{case}_gt.json"
        if not gt_path.exists():
            continue
        teeth = (json.loads(gt_path.read_text()).get("teeth") or {})
        for tooth_key, call in (rec.get("dental_elements") or {}).items():
            fdi = int(tooth_key.split("_")[1])
            block = teeth.get(tooth_key)
            if not block:
                continue
            call = dict(call)
            call["images"] = {k: RVI.resolve_path(v, args.base_dir)
                              for k, v in (call.get("images") or {}).items() if v}
            if not any(os.path.exists(v) for v in call["images"].values()):
                continue
            answer = established_answer(block, fdi, order, set(DEFAULT_REFUSED))
            if answer:
                jobs.append((case, tooth_key, fdi, call, answer))
    if args.limit:
        jobs = jobs[:args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stream = args.out_dir / "drafts.jsonl"

    # Resume: the JSONL is the record of what has been answered. Rebuilt case
    # files are derived from it and can be thrown away safely.
    out = collections.defaultdict(dict)
    if stream.exists() and not args.no_resume:
        for line in stream.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue            # a line torn in half by a kill -- skip it
            out[r["case"]][r["tooth"]] = r["evidence"]
        prior = sum(len(t) for t in out.values())
        jobs = [j for j in jobs if j[1] not in out.get(j[0], {})]
        print(f"[INFO] resuming from {stream}: {prior} tooth block(s) already "
              f"drafted, {len(jobs)} left")

    print(f"[INFO] {len(jobs)} tooth call(s) to draft, "
          f"{len({j[0] for j in jobs})} case(s), concurrency {args.concurrency}")
    print(f"[INFO] refused (never sent to the teacher): {sorted(DEFAULT_REFUSED)}")

    fails = collections.Counter()
    lock = threading.Lock()

    def flush_cases():
        """Rebuild the {case}_pred.json files from what is in hand."""
        n = 0
        for case, teeth in sorted(out.items()):
            n += sum(len(v) for v in teeth.values())
            tmp = args.out_dir / f".{case}_pred.json.tmp"
            tmp.write_text(json.dumps({"case_id": case, "teeth": teeth},
                                      ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
            tmp.replace(args.out_dir / f"{case}_pred.json")   # atomic
        return n

    def run(j):
        case, tooth_key, fdi, call, answer = j
        try:
            return case, tooth_key, draft_one(client, args.model, call, fdi,
                                              answer, args.max_tokens), None
        except Exception as e:                          # noqa: BLE001
            return case, tooth_key, None, f"{type(e).__name__}: {e}"

    done = 0
    fh = stream.open("a", encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for fut in as_completed([pool.submit(run, j) for j in jobs]):
                case, tooth_key, ev, err = fut.result()
                done += 1
                with lock:
                    if err:
                        fails[err.split(":")[0]] += 1
                    else:
                        out[case][tooth_key] = ev
                        # Appended and flushed per answer: this line is what
                        # survives a timeout, so it is written before anything
                        # else and never buffered.
                        fh.write(json.dumps({"case": case, "tooth": tooth_key,
                                             "evidence": ev},
                                            ensure_ascii=False) + "\n")
                        fh.flush()
                    if done % args.flush_every == 0:
                        flush_cases()
                        print(f"  {done}/{len(jobs)}  failures "
                              f"{sum(fails.values())}  (case files flushed)",
                              file=sys.stderr)
    finally:
        fh.close()
        n_fields = flush_cases()

    print(f"\n[INFO] wrote {len(out)} case file(s), "
          f"{sum(len(t) for t in out.values())} tooth block(s), "
          f"{n_fields} evidence string(s) -> {args.out_dir}")
    print(f"[INFO] append-only record: {stream}")
    if fails:
        print(f"[WARN] {sum(fails.values())} call(s) failed: {dict(fails)}")
    print("[INFO] next: build_sft_targets.py --evidence-from "
          f"{args.out_dir}, then --check-perceivable against the STUDENT.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
