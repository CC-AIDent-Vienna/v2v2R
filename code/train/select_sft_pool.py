#!/usr/bin/env python3
"""
select_sft_pool.py -- the training pool for docs/vision_sft_plan.md §3.5, as a file.

Turns `outputs/gt_quality_training/case_ranking_*.csv` + `audit_report_facts.py`
into three case lists -- narrow, wide, held-out -- and prints the composition of
each, because composition is the part that goes wrong quietly.

WHY THIS IS NOT A ONE-LINE `awk`
────────────────────────────────
Three things have to be true at once and only one of them is a threshold.

1. GT QUALITY. `score_asis` from the ranking file, the same number
   survey_gt_quality.py produced. Narrow is >=0.90, wide is >=0.80.

2. AUDIT-CLEAN. A case whose report_facts still carries an ERROR from
   audit_report_facts.py is a case whose labels are known-wrong, and the
   ranking score does not know that. Passed in via --audit-json.

3. SPLIT. Nothing from `dataset/validate` may appear, ever -- plan §0. The
   ranking file is training-only so this cannot fire today, but it is checked
   rather than assumed: the cost of the check is a directory listing and the
   cost of the assumption is an experiment whose headline number is invalid.

THE PREFIX PROBLEM, WHICH IS THE REAL REASON THIS FILE EXISTS
─────────────────────────────────────────────────────────────
Case IDs carry their sub-dataset in the prefix letter, and the four differ in
scanner, field of view and reconstruction. Training is 69% P. Validate is
10/10/10/10 -- exactly 25% each. And the GT-quality threshold makes it worse
rather than better, because P's reports are the ones that extract cleanly:

    prefix   all training   pool >=0.80   validate
    A            15%             6%         25%
    F             9%             8%         25%
    P            69%            78%         25%
    S             7%             9%         25%

For a language arm that is a mild nuisance. For arm 2 -- which trains the
vision encoder, the one component that sees scanner physics rather than words
-- fitting on 78% P and scoring on 25% P is a way to produce a real perceptual
gain that the official metric cannot see, or a real regression on A/F/S that a
P-heavy gain hides. Neither is a finding about H1.

THE CHOSEN ANSWER, AND WHY IT IS NOT `--stratify`
─────────────────────────────────────────────────
Balancing the >=0.80 pool directly is useless: it collapses wide from 198 cases
to 28, because A/F/S run out long before P does. The skew is not the
threshold's doing -- the split is 70% P even at threshold 0 -- so no threshold
fixes it.

--per-prefix-top N inverts the order instead: rank WITHIN each prefix and take
the best N of each. The ceiling is the scarcest prefix (audit-clean: A 83,
F 41, P 385, S 40), so a balanced pool tops out near 40/prefix, and the price
is paid in GT quality by exactly the prefixes that are scarce:

    N/prefix   cases   calls    A floor  F floor  P floor  S floor
        30      120    2,340      0.71     0.68     1.00     0.71
        34      136    2,652      0.71     0.58     1.00     0.67
        40      160    3,120      0.67     0.31     1.00     0.46

Past ~34 the balance is bought with labels that are simply bad, so 30 is the
default. P's floor stays 1.00 throughout: it has 385 audit-clean cases and only
needs its best 30.

Held-out and training are drawn from the SAME per-prefix band (the top N+H) and
split at random within it, not by taking the best H for the gate. A held-out
set made of each prefix's cleanest cases would report a gate number the
training distribution cannot reproduce -- flattering, and in the direction that
makes a null look like a result.

The held-out set is stratified REGARDLESS of --per-prefix-top. It is what the
stage-C decision gate reads, and a gate that reads a 78%-P sample to predict a
25%-P result is not a gate. That one is not a knob.

Usage:
    python code/train/select_sft_pool.py \\
        --ranking outputs/gt_quality_training/case_ranking_20260810_232836.csv \\
        --audit-json <audit.json> --out-dir outputs/training_results/sft_pool
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import re
import sys
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

NARROW_MIN = 0.90
WIDE_MIN = 0.80
HELDOUT_N = 25


def case_of(text: str) -> str | None:
    m = re.match(r"([A-Z]+\d+)", str(text))
    return m.group(1) if m else None


def audit_error_cases(path: Path | None) -> set[str]:
    """Case ids with at least one surviving ERROR.

    Tolerant about the JSON shape on purpose -- audit_report_facts.py's --json
    has changed shape once already, and a pool silently built from zero
    exclusions is worse than a crash.
    """
    if not path:
        return set()
    blob = json.loads(path.read_text(encoding="utf-8"))
    items = blob if isinstance(blob, list) else (
        blob.get("findings") or blob.get("errors") or [])
    if not items:
        sys.exit(f"[FAIL] {path}: no findings parsed -- shape changed? "
                 f"top-level keys: {list(blob)[:8] if isinstance(blob, dict) else 'list'}")
    bad = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        level = str(it.get("level") or it.get("severity") or "").upper()
        if level != "ERROR":
            continue
        c = case_of(it.get("case") or it.get("stem") or it.get("file") or "")
        if c:
            bad.add(c)
    return bad


def validate_case_ids() -> set[str]:
    """Every case id under dataset/validate. Plan §0's hard boundary."""
    d = ROOT / "dataset/validate/images"
    return {c for c in (case_of(p.name) for p in d.glob("*_0000.nii.gz")) if c}


def composition(cases) -> dict:
    return dict(sorted(collections.Counter(c[0] for c in cases).items()))


def fmt(cases, label: str, total_ref: int | None = None) -> str:
    comp = composition(cases)
    n = len(cases)
    parts = " ".join(f"{k}:{v:>3} ({100 * v / n:>3.0f}%)" for k, v in comp.items()) if n else "-"
    calls = f"~{round(n * 19.5):,} calls"
    return f"  {label:<22} {n:>4} cases  {calls:>14}   {parts}"


def stratify(cases, per_prefix: int | None, rng) -> list[str]:
    """Down-sample each prefix to the scarcest one (or to `per_prefix`)."""
    by = collections.defaultdict(list)
    for c in cases:
        by[c[0]].append(c)
    k = per_prefix or min(len(v) for v in by.values())
    out = []
    for p in sorted(by):
        pool = sorted(by[p])
        rng.shuffle(pool)
        out.extend(pool[:k])
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranking", type=Path, required=True)
    ap.add_argument("--audit-json", type=Path,
                    help="audit_report_facts.py --json output; cases with a "
                         "surviving ERROR are excluded")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--stratify", action="store_true",
                    help="balance the TRAINING pools across A/F/P/S by "
                         "down-sampling the >=0.80 pool. Collapses it to ~28 "
                         "cases; kept only to make that visible. Prefer "
                         "--per-prefix-top.")
    ap.add_argument("--per-prefix-top", type=int, metavar="N",
                    help="rank within each prefix and take the best N for the "
                         "wide pool, ignoring the global score thresholds. The "
                         "balanced mode -- see the module docstring for the "
                         "GT-quality floor each N costs.")
    ap.add_argument("--per-prefix-narrow", type=int, metavar="K",
                    help="narrow pool size per prefix (default: N//2), taken "
                         "as the best K of that prefix's training allocation.")
    ap.add_argument("--heldout-per", type=int, default=HELDOUT_N // 4,
                    help="held-out cases per prefix (default 6 -> 24 total, "
                         "matching validate's 10/10/10/10 shape)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = list(csv.DictReader(args.ranking.open(encoding="utf-8")))
    bad = audit_error_cases(args.audit_json)
    forbidden = validate_case_ids()

    scored = {r["case_id"]: float(r.get("score_asis") or 0.0) for r in rows}

    leaked = sorted(set(scored) & forbidden)
    if leaked:
        sys.exit(f"[FAIL] plan §0: {len(leaked)} validate case(s) in the "
                 f"ranking file: {leaked[:10]}")

    def eligible(threshold):
        return sorted(c for c, s in scored.items()
                      if s >= threshold and c not in bad and c not in forbidden)

    wide_all = eligible(WIDE_MIN)
    narrow_all = eligible(NARROW_MIN)

    print(f"[INFO] ranking {args.ranking.name}: {len(rows)} cases")
    print(f"[INFO] audit ERRORs exclude {len(bad)} case(s); "
          f"{len(forbidden)} validate ids are forbidden and none appeared")
    print()
    print("BEFORE held-out is removed:")
    print(fmt(narrow_all, f"narrow >={NARROW_MIN}"))
    print(fmt(wide_all, f"wide   >={WIDE_MIN}"))
    print()

    H = args.heldout_per

    if args.per_prefix_top:
        # Balanced mode. Rank within each prefix, take the best N+H, then split
        # that band at RANDOM into training and held-out so both see the same
        # GT-quality distribution. Ordering by score and giving the gate the top
        # H would make the gate easier than the task.
        N = args.per_prefix_top
        K = args.per_prefix_narrow or max(1, N // 2)
        eligible_all = sorted(
            (c for c, s in scored.items() if c not in bad and c not in forbidden),
            key=lambda c: (-scored[c], c))

        by_prefix = collections.defaultdict(list)
        for c in eligible_all:
            by_prefix[c[0]].append(c)

        held, train_wide, train_narrow = [], [], []
        print("balanced mode -- per prefix:")
        for p in sorted(by_prefix):
            band = by_prefix[p][:N + H]
            if len(band) < N + H:
                print(f"  [WARN] {p}: only {len(band)} audit-clean case(s), "
                      f"wanted {N + H} -- this prefix is short")
            shuffled = list(band)
            rng.shuffle(shuffled)
            h, t = shuffled[:H], shuffled[H:]
            # narrow is the best K of THIS prefix's training allocation
            t_by_score = sorted(t, key=lambda c: (-scored[c], c))
            held.extend(h)
            train_wide.extend(t)
            train_narrow.extend(t_by_score[:K])
            floor = min((scored[c] for c in band), default=float("nan"))
            print(f"  {p}: band {len(band):>3} (score floor {floor:.2f})  ->  "
                  f"held-out {len(h)}, wide {len(t)}, narrow {min(K, len(t))}")
        print()
        held, train_wide, train_narrow = sorted(held), sorted(train_wide), sorted(train_narrow)
        held_set = set(held)
    else:
        # Held-out first, and stratified, so the gate reads validate's shape.
        # Taken from WIDE: a case held out of the wide set is held out of narrow
        # too (narrow is a subset), which is what "never train on it" requires.
        held = stratify(wide_all, H, rng)[:H * 4]
        held_set = set(held)

        train_wide = [c for c in wide_all if c not in held_set]
        train_narrow = [c for c in narrow_all if c not in held_set]

        if args.stratify:
            train_wide = stratify(train_wide, None, rng)
            train_narrow = stratify(train_narrow, None, rng)

    mode = ("balanced top-%d/prefix" % args.per_prefix_top if args.per_prefix_top
            else "stratified" if args.stratify else "unstratified")
    print(f"SELECTED ({mode}):")
    print(fmt(train_narrow, "narrow (arm 1)"))
    print(fmt(train_wide, "wide (arms 2,3)"))
    print(fmt(held, "held-out (the gate)"))
    print()

    # The three invariants worth crashing over. Held-out leaking into a
    # training pool is the one that would invalidate the stage-C gate without
    # changing any number that looks wrong.
    overlap = set(train_wide) & held_set
    assert not overlap, f"held-out leaked into wide: {sorted(overlap)[:5]}"
    assert not (set(train_narrow) & held_set), "held-out leaked into narrow"
    assert set(train_narrow) <= set(train_wide), "narrow is not a subset of wide"
    assert not ((set(train_wide) | held_set) & forbidden), "validate case in a pool"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, cases in (("narrow", train_narrow), ("wide", train_wide),
                        ("heldout", held)):
        p = args.out_dir / f"{name}.txt"
        p.write_text("\n".join(cases) + "\n", encoding="utf-8")
        written[name] = {"n": len(cases), "composition": composition(cases),
                         "cases": cases}
        print(f"[INFO] {p} ({len(cases)} cases)")

    meta = args.out_dir / "pool.json"
    meta.write_text(json.dumps({
        "ranking": str(args.ranking), "seed": args.seed,
        "stratified": args.stratify,
        "narrow_min": NARROW_MIN, "wide_min": WIDE_MIN,
        "audit_error_cases": sorted(bad),
        "sets": written,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] {meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
