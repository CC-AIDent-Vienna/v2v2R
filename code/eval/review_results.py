#!/usr/bin/env python3
"""
code/eval/review_results.py

Review ground truth AND/OR VLM prediction results, side by side if both given.
Same tooth-centered display style as review_qa_pairs.py.

Handles two input shapes automatically (auto-detected per line):

1) Ground truth records (from parse_reports_to_gt.py) — one record per file/line:
   {
     "case_id": "A004",
     "radiologist_id": "1",            # or "radiologists": [...] for consensus
     "ground_truth": [
       {"fact_id": "...", "field": "...", "value": ..., "tooth_fdi": 11 (optional),
        "confidence": "...", "source": "..."},
       ...
     ]
   }

2) Prediction records (from run_vqa_inference.py) — flat, one QA pair per line:
   {"question_id": "...", "case_id": "A004", "category": "global"|"detail",
    "prediction": ... (or "answer"/"value"/"output"), ...}

Usage:
  # Review ground truth only
  python code/eval/review_results.py --results-jsonl test_5/outputs/ground_truth/A004_ground_truth.jsonl

  # Review predictions only
  python code/eval/review_results.py --results-jsonl test_5/outputs/predictions/A004_pred.json

  # Review GT + predictions side by side (multiple files, same case)
  python3 code/eval/review_results.py \
      --results-jsonl test_5/outputs/ground_truth/A004_1_ground_truth.jsonl \
                       test_5/outputs/predictions/A004_pred.json \
      --format text

  # All individual radiologists + consensus + predictions together
  python code/eval/review_results.py \
      --results-jsonl test_5/outputs/ground_truth/A004_*.jsonl \
                       test_5/outputs/predictions/A004_pred.json
"""

import json
import sys
import glob
import argparse
from pathlib import Path
from collections import defaultdict


ALL_FDIS = [11,12,13,14,15,16,17,18,
            21,22,23,24,25,26,27,28,
            31,32,33,34,35,36,37,38,
            41,42,43,44,45,46,47,48]

PREDICTION_VALUE_KEYS = ["prediction", "answer", "value", "output", "response"]


def extract_fdi_from_fact_id(fact_id: str):
    """tooth_11_eruption -> 11 (int). Returns None if not a tooth fact."""
    parts = fact_id.split("_")
    if len(parts) >= 2 and parts[0] == "tooth":
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def get_prediction_value(record: dict):
    """Find the actual predicted value under whichever key the inference script used."""
    for key in PREDICTION_VALUE_KEYS:
        if key in record:
            return record[key]
    return None


def source_label_for_gt_record(record: dict) -> str:
    """Build a readable source label for a GT record (individual radiologist or consensus)."""
    if "radiologists" in record:
        n = record.get("n_radiologists", len(record.get("radiologists", [])))
        return f"consensus({n})"
    rid = record.get("radiologist_id")
    if rid is not None:
        return f"doctor_{rid}"
    return "ground_truth"


def normalize_line(record: dict, source_file: str):
    """
    Convert one JSON line into a list of normalized finding dicts:
      {case_id, fact_id, category, tooth_fdi, value, confidence, source}
    """
    findings = []

    if "ground_truth" in record and isinstance(record["ground_truth"], list):
        # Ground truth style record
        case_id = record.get("case_id")
        source = source_label_for_gt_record(record)

        for f in record["ground_truth"]:
            fact_id = f.get("fact_id")
            tooth_fdi = f.get("tooth_fdi")
            category = "detail" if tooth_fdi is not None else "global"

            findings.append({
                "case_id": case_id,
                "fact_id": fact_id,
                "category": category,
                "tooth_fdi": tooth_fdi,
                "value": f.get("value"),
                "confidence": f.get("confidence"),
                "source": source,
            })

    elif "question_id" in record:
        # Flat prediction-style record (one QA pair per line)
        case_id = record.get("case_id")
        fact_id = record.get("question_id")
        category = record.get("category", "global")
        tooth_fdi = record.get("tooth_fdi")
        if tooth_fdi is None and category == "detail":
            tooth_fdi = extract_fdi_from_fact_id(fact_id)

        findings.append({
            "case_id": case_id,
            "fact_id": fact_id,
            "category": category,
            "tooth_fdi": tooth_fdi,
            "value": get_prediction_value(record),
            "confidence": record.get("confidence"),
            "source": "prediction",
        })

    elif "teeth" in record and "global" in record:
        # Prediction-style record from run_vqa_inference.py:
        # {"case_id": ..., "global": {field: value, ...},
        #  "teeth": {"tooth_11": {field: value, ..., "detected": "yes"}, ...}}
        case_id = record.get("case_id")

        for field, value in record.get("global", {}).items():
            findings.append({
                "case_id": case_id,
                "fact_id": field,
                "category": "global",
                "tooth_fdi": None,
                "value": value,
                "confidence": None,
                "source": "prediction",
            })

        for tooth_key, tooth_data in record.get("teeth", {}).items():
            fdi_str = tooth_key.replace("tooth_", "")
            fdi = int(fdi_str) if fdi_str.isdigit() else None

            for field, value in tooth_data.items():
                if field == "detected":
                    continue
                fact_id = f"{tooth_key}_{field}" if fdi is not None else field
                findings.append({
                    "case_id": case_id,
                    "fact_id": fact_id,
                    "category": "detail",
                    "tooth_fdi": fdi,
                    "value": value,
                    "confidence": None,
                    "source": "prediction",
                })

    else:
        print(f"[WARN] Unrecognized record shape in {source_file}, skipping: "
              f"{list(record.keys())}", file=sys.stderr)

    return findings


def load_results(paths):
    """
    Load and normalize one or more result files (GT and/or predictions).
    Expands glob patterns. Returns pairs_by_case[case_id][fact_id] = [finding, ...]

    Every file we produce (GT per-radiologist, GT consensus, VLM predictions)
    contains exactly ONE JSON object -- just formatted differently: GT files
    are single-line (json.dumps + "\n"), predictions files are pretty-printed
    multi-line (json.dumps(..., indent=2)). So we always read the WHOLE file
    as one JSON document, never line-by-line.
    """
    resolved_paths = []
    for p in paths:
        matches = glob.glob(p)
        if matches:
            resolved_paths.extend(sorted(matches))
        elif Path(p).exists():
            resolved_paths.append(p)
        else:
            print(f"[WARN] No file matched: {p}", file=sys.stderr)

    pairs_by_case = defaultdict(lambda: defaultdict(list))

    for path in resolved_paths:
        with open(path) as f:
            content = f.read().strip()
        if not content:
            continue
        try:
            record = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"[WARN] Bad JSON in {path}, skipping: {e}", file=sys.stderr)
            continue

        for finding in normalize_line(record, path):
            case_id = finding["case_id"]
            fact_id = finding["fact_id"]
            if case_id is None or fact_id is None:
                continue
            pairs_by_case[case_id][fact_id].append(finding)

    return {k: dict(v) for k, v in pairs_by_case.items()}


def format_value(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def format_summary(pairs_by_case: dict) -> str:
    lines = []
    total_global = 0
    total_detail = 0
    total_findings = 0

    for case_id in sorted(pairs_by_case.keys()):
        facts = pairs_by_case[case_id]

        global_count = sum(1 for fid, entries in facts.items()
                            if entries[0]["category"] == "global")
        detail_count = sum(1 for fid, entries in facts.items()
                            if entries[0]["category"] == "detail")
        n_findings = sum(len(entries) for entries in facts.values())

        lines.append(f"{case_id}: {global_count} global fact(s) + {detail_count} tooth fact(s) "
                      f"= {n_findings} total finding(s) across all sources")

        total_global += global_count
        total_detail += detail_count
        total_findings += n_findings

    lines.append("")
    lines.append(f"TOTAL: {total_global} global + {total_detail} tooth facts, "
                  f"{total_findings} findings across all sources")

    return "\n".join(lines)


def format_text(pairs_by_case: dict, case_id_filter=None) -> str:
    lines = []

    for case_id in sorted(pairs_by_case.keys()):
        if case_id_filter and case_id != case_id_filter:
            continue

        facts = pairs_by_case[case_id]

        lines.append(f"\n{'='*70}")
        lines.append(f"CASE: {case_id}")
        lines.append(f"{'='*70}")

        # --- Global facts ---
        global_facts = {fid: entries for fid, entries in facts.items()
                         if entries[0]["category"] == "global"}
        lines.append(f"\n--- GLOBAL FACTS ({len(global_facts)}) ---\n")

        for i, fact_id in enumerate(sorted(global_facts.keys()), 1):
            entries = global_facts[fact_id]
            lines.append(f"[{i}] {fact_id}")
            for e in entries:
                conf = f" (confidence: {e['confidence']})" if e.get("confidence") else ""
                lines.append(f"    [{e['source']}] {format_value(e['value'])}{conf}")
            lines.append("")

        # --- Tooth facts, organized by FDI ---
        detail_facts = {fid: entries for fid, entries in facts.items()
                         if entries[0]["category"] == "detail"}
        lines.append(f"\n--- TOOTH FACTS ({len(detail_facts)} fact-instances) ---\n")

        facts_by_fdi = defaultdict(dict)
        for fact_id, entries in detail_facts.items():
            fdi = entries[0].get("tooth_fdi")
            if fdi is None:
                fdi = extract_fdi_from_fact_id(fact_id)
            facts_by_fdi[fdi][fact_id] = entries

        for fdi in ALL_FDIS:
            if fdi not in facts_by_fdi:
                lines.append(f"tooth {fdi}: no findings, skip")
                continue

            lines.append(f"tooth {fdi}:")
            for j, fact_id in enumerate(sorted(facts_by_fdi[fdi].keys()), 1):
                entries = facts_by_fdi[fdi][fact_id]
                # field name without the tooth_{fdi}_ prefix, if present
                field_label = entries[0].get("field") or fact_id
                lines.append(f"  [{j}] {field_label}")
                for e in entries:
                    conf = f" (confidence: {e['confidence']})" if e.get("confidence") else ""
                    lines.append(f"      [{e['source']}] {format_value(e['value'])}{conf}")
            lines.append("")

    return "\n".join(lines)


def format_markdown(pairs_by_case: dict, case_id_filter=None) -> str:
    lines = []

    for case_id in sorted(pairs_by_case.keys()):
        if case_id_filter and case_id != case_id_filter:
            continue

        facts = pairs_by_case[case_id]
        lines.append(f"## Case {case_id}\n")

        global_facts = {fid: entries for fid, entries in facts.items()
                         if entries[0]["category"] == "global"}
        lines.append(f"### Global Facts ({len(global_facts)})\n")

        for i, fact_id in enumerate(sorted(global_facts.keys()), 1):
            entries = global_facts[fact_id]
            lines.append(f"{i}. **{fact_id}**")
            for e in entries:
                conf = f" _(confidence: {e['confidence']})_" if e.get("confidence") else ""
                lines.append(f"   - `{e['source']}`: {format_value(e['value'])}{conf}")
            lines.append("")

        detail_facts = {fid: entries for fid, entries in facts.items()
                         if entries[0]["category"] == "detail"}
        lines.append(f"### Tooth Facts ({len(detail_facts)} fact-instances)\n")

        facts_by_fdi = defaultdict(dict)
        for fact_id, entries in detail_facts.items():
            fdi = entries[0].get("tooth_fdi")
            if fdi is None:
                fdi = extract_fdi_from_fact_id(fact_id)
            facts_by_fdi[fdi][fact_id] = entries

        for fdi in ALL_FDIS:
            if fdi not in facts_by_fdi:
                lines.append(f"**Tooth {fdi}**: no findings (skip)\n")
                continue

            lines.append(f"**Tooth {fdi}**\n")
            for j, fact_id in enumerate(sorted(facts_by_fdi[fdi].keys()), 1):
                entries = facts_by_fdi[fdi][fact_id]
                field_label = entries[0].get("field") or fact_id
                lines.append(f"  {j}. {field_label}")
                for e in entries:
                    conf = f" _(confidence: {e['confidence']})_" if e.get("confidence") else ""
                    lines.append(f"     - `{e['source']}`: {format_value(e['value'])}{conf}")
            lines.append("")

    return "\n".join(lines)


def format_json(pairs_by_case: dict, case_id_filter=None) -> str:
    if case_id_filter:
        filtered = {k: v for k, v in pairs_by_case.items() if k == case_id_filter}
    else:
        filtered = pairs_by_case
    return json.dumps(filtered, indent=2, ensure_ascii=False)


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Review ground truth and/or prediction results (tooth-centered).",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument("--results-jsonl", required=True, nargs="+",
                     help="One or more result files (ground_truth or predictions). "
                          "Glob patterns accepted, e.g. test_5/outputs/ground_truth/A004_*.jsonl")
    ap.add_argument("--format", default="summary",
                     choices=["summary", "text", "markdown", "json"],
                     help="Output format (default: summary)")
    ap.add_argument("--case-id", default=None,
                     help="Filter to single case (optional)")
    ap.add_argument("--out", default=None,
                     help="Output file (default: stdout)")

    args = ap.parse_args()

    pairs_by_case = load_results(args.results_jsonl)

    if not pairs_by_case:
        print(f"[FAIL] No results found in: {args.results_jsonl}", file=sys.stderr)
        sys.exit(1)

    if args.format == "summary":
        output = format_summary(pairs_by_case)
    elif args.format == "text":
        output = format_text(pairs_by_case, args.case_id)
    elif args.format == "markdown":
        output = format_markdown(pairs_by_case, args.case_id)
    elif args.format == "json":
        output = format_json(pairs_by_case, args.case_id)

    if args.out:
        with open(args.out, "w") as f:
            f.write(output)
        print(f"[INFO] Written to {args.out}", file=sys.stderr)
    else:
        print(output)