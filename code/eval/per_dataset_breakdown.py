#!/usr/bin/env python3
"""
code/eval/per_dataset_breakdown.py

Break an official_ranking run down by source dataset (the case_id prefix:
A / F / P / S) and report BLEU-4, METEOR, Captioning, RadFact P/R, Clinical
Score and Final Score for each one.

Everything is derived from the run's own results.jsonl, so the breakdown can
never drift from the run it describes:

  BLEU-4 / METEOR   mean of the per-case values stored in results.jsonl
                    (macro-average over the dataset's cases)
  Captioning        mean(BLEU-4, METEOR)
  Clinical          harmonic mean of the dataset's mean RadFact logical
                    precision and mean logical recall -- the official
                    formula, applied to the subset
  Final             0.8 * Clinical + 0.2 * Captioning

Caveat on BLEU-4: the official aggregate in summary.json is CORPUS-level
BLEU, which does not decompose into subsets -- corpus BLEU pools n-gram
counts across all cases, so a per-dataset corpus BLEU is a different (and
generally higher) number than the macro-average reported here. Pass
--corpus-bleu to additionally compute the true corpus-level BLEU-4 per
dataset; that re-reads the candidate/reference .txt files from disk, and the
script refuses to do so if those files are newer than results.jsonl (i.e.
they have been regenerated since the run was scored) unless --allow-stale is
given.

Caveat on Clinical: cases whose RadFact judging failed carry logical_f1 =
null and clinical_score = 0 in results.jsonl. Those cases are EXCLUDED from
the dataset's mean precision/recall (matching official_ranking.py's
aggregate) but their zeros still drag down the per-case clinical mean. The
"RF-n" column reports how many cases actually contributed RadFact scores.

Because BLEU is non-linear in --corpus-bleu mode and Clinical is a harmonic
mean, the four per-dataset Final Scores do not average back to the overall
Final Score. That is expected.

Usage:
  python3 code/eval/per_dataset_breakdown.py outputs/no_VLM/official_ranking
  python3 code/eval/per_dataset_breakdown.py <run_dir> --corpus-bleu
  python3 code/eval/per_dataset_breakdown.py <run_dir> --json out.json
"""

import argparse
import json
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

METRIC_KEYS = ("bleu_4", "meteor", "captioning_score", "logical_precision",
               "logical_recall", "logical_f1", "clinical_score", "final_score")


def mean_sd(values):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, var ** 0.5


def load_rows(run_dir: Path) -> dict:
    """Read results.jsonl into {case_id: row}, last row per case_id wins."""
    path = run_dir / "results.jsonl"
    if not path.exists():
        sys.exit(f"[ERROR] no results.jsonl in {run_dir}")
    rows = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[row["case_id"]] = row
    if not rows:
        sys.exit(f"[ERROR] {path} is empty")
    return rows


def resolve_dirs(run_dir: Path, args) -> tuple:
    """Candidate/reference dirs: CLI override, else the run's summary.json."""
    synth, reports, multi_ref = args.synthesized_dir, args.reports_dir, False
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        synth = synth or summary.get("synthesized_dir")
        reports = reports or summary.get("reports_dir")
        multi_ref = bool(summary.get("multi_reference", False))
    if not synth or not reports:
        sys.exit("[ERROR] could not determine --synthesized-dir / --reports-dir "
                 f"(no usable summary.json in {run_dir}); pass them explicitly")
    return Path(synth), Path(reports), multi_ref


def check_freshness(run_dir: Path, case_ids, synth_dir: Path, reports_dir: Path,
                    allow_stale: bool) -> None:
    """Refuse to mix a scored run with report .txt files written after it."""
    scored_at = (run_dir / "results.jsonl").stat().st_mtime
    stale = []
    for case_id in case_ids:
        for base in (synth_dir, reports_dir):
            path = base / f"{case_id}.txt"
            if path.exists() and path.stat().st_mtime > scored_at:
                stale.append(path)
    if not stale:
        return
    msg = (f"{len(stale)} report file(s) are NEWER than results.jsonl "
           f"(e.g. {stale[0]}) -- they were regenerated after this run was "
           f"scored, so corpus BLEU computed from them would not match the "
           f"RadFact scores in results.jsonl.")
    if allow_stale:
        print(f"[WARN] {msg} Proceeding because --allow-stale was given.\n",
              file=sys.stderr)
    else:
        sys.exit(f"[ERROR] {msg}\n        Re-score the run, or pass "
                 f"--allow-stale to compute corpus BLEU anyway.")


def corpus_bleu_for(case_ids, synth_dir: Path, reports_dir: Path,
                    multi_ref: bool) -> float:
    """True corpus-level BLEU-4 for a subset, via official_ranking's scorer."""
    from official_ranking import bleu_4_local, load_cases
    texts = load_cases(synth_dir, reports_dir, list(case_ids), multi_ref)
    if not texts:
        return 0.0
    return bleu_4_local([t[1] for t in texts], [t[2] for t in texts])


def score_group(case_ids, rows) -> dict:
    """Official aggregate formulas restricted to one group of case_ids."""
    group = [rows[cid] for cid in case_ids if cid in rows]

    bleu4, _ = mean_sd([r.get("bleu_4") for r in group])
    meteor, _ = mean_sd([r.get("meteor") for r in group])
    captioning = (bleu4 + meteor) / 2.0

    radfact_rows = [r for r in group if r.get("logical_f1") is not None]
    if radfact_rows:
        mean_p = sum(r["logical_precision"] for r in radfact_rows) / len(radfact_rows)
        mean_r = sum(r["logical_recall"] for r in radfact_rows) / len(radfact_rows)
        clinical = (2 * mean_p * mean_r / (mean_p + mean_r)) if (mean_p + mean_r) else 0.0
    else:
        mean_p = mean_r = None
        clinical = 0.0

    per_case = {}
    for key in METRIC_KEYS:
        vals = [r.get(key) for r in group if r.get(key) is not None]
        if vals:
            m, sd = mean_sd(vals)
            per_case[key] = {"mean": m, "sd": sd, "n": len(vals)}

    return {
        "num_cases": len(group),
        "num_radfact_cases": len(radfact_rows),
        "case_ids": list(case_ids),
        "bleu_4": bleu4,
        "meteor": meteor,
        "captioning_score": captioning,
        "logical_precision": mean_p,
        "logical_recall": mean_r,
        "clinical_score": clinical,
        "final_score": 0.8 * clinical + 0.2 * captioning,
        "per_case_stats": per_case,
    }


def fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def print_table(groups: dict, overall: dict, corpus: bool) -> None:
    cols = [("Dataset", 9), ("N", 4), ("RF-n", 6), ("BLEU-4", 10)]
    if corpus:
        cols.append(("cBLEU-4", 10))
    cols += [("METEOR", 10), ("Caption", 10), ("RF-P", 10), ("RF-R", 10),
             ("Clinical", 10), ("Final", 10)]
    header = "".join(f"{n:<9}" if w == 9 else f"{n:>{w}}" for n, w in cols)

    def row(label, g):
        cells = [f"{label:<9}", f"{g['num_cases']:>4}", f"{g['num_radfact_cases']:>6}",
                 f"{fmt(g['bleu_4']):>10}"]
        if corpus:
            cells.append(f"{fmt(g.get('corpus_bleu_4')):>10}")
        cells += [f"{fmt(g['meteor']):>10}", f"{fmt(g['captioning_score']):>10}",
                  f"{fmt(g['logical_precision']):>10}", f"{fmt(g['logical_recall']):>10}",
                  f"{fmt(g['clinical_score']):>10}", f"{fmt(g['final_score']):>10}"]
        return "".join(cells)

    print("=" * len(header))
    print("PER-DATASET BREAKDOWN")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for key in sorted(groups):
        print(row(key, groups[key]))
    print("-" * len(header))
    print(row("ALL", overall))
    print("=" * len(header))
    print("BLEU-4/METEOR are per-case means over the dataset's cases; "
          "Caption = mean of the two.")
    if corpus:
        print("cBLEU-4 is true corpus-level BLEU-4 over the dataset's "
              "candidate/reference pairs.")
    print("RF-n = cases with a RadFact score; RF-P/RF-R/Clinical use only those "
          "cases.")


def print_per_case(groups: dict) -> None:
    print("\nPer-case mean +/- sd within each dataset:")
    for key in sorted(groups):
        print(f"\n  [{key}]  n={groups[key]['num_cases']}")
        stats = groups[key]["per_case_stats"]
        for mk in METRIC_KEYS:
            if mk in stats:
                s = stats[mk]
                suffix = f"   (n={s['n']})" if s["n"] != groups[key]["num_cases"] else ""
                print(f"    {mk:<20} {s['mean']:.4f} +/- {s['sd']:.4f}{suffix}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-dataset (A/F/P/S) breakdown of an official_ranking run")
    ap.add_argument("run_dir", help="official_ranking output dir "
                                    "(contains results.jsonl + summary.json)")
    ap.add_argument("--corpus-bleu", action="store_true",
                    help="also compute true corpus-level BLEU-4 per dataset "
                         "(re-reads the report .txt files)")
    ap.add_argument("--allow-stale", action="store_true",
                    help="with --corpus-bleu, proceed even if report files are "
                         "newer than results.jsonl")
    ap.add_argument("--synthesized-dir", default=None,
                    help="override candidate report dir from summary.json")
    ap.add_argument("--reports-dir", default=None,
                    help="override reference report dir from summary.json")
    ap.add_argument("--json", default=None, help="also write the breakdown here")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    rows = load_rows(run_dir)

    by_prefix = {}
    for case_id in sorted(rows):
        by_prefix.setdefault(case_id[0], []).append(case_id)

    groups = {k: score_group(v, rows) for k, v in by_prefix.items()}
    overall = score_group(sorted(rows), rows)

    if args.corpus_bleu:
        synth_dir, reports_dir, multi_ref = resolve_dirs(run_dir, args)
        check_freshness(run_dir, sorted(rows), synth_dir, reports_dir,
                        args.allow_stale)
        for key, g in groups.items():
            g["corpus_bleu_4"] = corpus_bleu_for(g["case_ids"], synth_dir,
                                                 reports_dir, multi_ref)
        overall["corpus_bleu_4"] = corpus_bleu_for(overall["case_ids"], synth_dir,
                                                   reports_dir, multi_ref)

    print_table(groups, overall, args.corpus_bleu)
    print_per_case(groups)

    if args.json:
        out = {
            "run_dir": str(run_dir),
            "bleu_mode": "corpus+macro" if args.corpus_bleu else "macro",
            "per_dataset": groups,
            "overall": overall,
        }
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\n[WROTE] {args.json}")


if __name__ == "__main__":
    main()
