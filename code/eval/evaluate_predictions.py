#!/usr/bin/env python3
"""
code/eval/evaluate_predictions.py

Fact-based accuracy evaluation: compares each case's (normalized)
{case_id}_pred.json against its {case_id}_gt.json (parse_reports_to_gt.py's
output) field by field, at the SAME granularity the VLM was actually
asked at -- see parse_reports_to_gt.py's docstring for why this compares
against the raw prediction and not postprocess_pred.py's summary.

This is a diagnostic tool, not the competition metric (RadFact/BLEU/
METEOR score the synthesized report text) -- its purpose is "which
specific fact does the VLM get wrong," matching this project's original
job4_evaluate.sh design, written against schema.json's actual fact
shapes (arch-split facts, list-valued pattern fields, list[object]
sub-facts that need matching before comparing, etc.).

FIVE COMPARISON STRATEGIES, chosen per field type:
  1. set_overlap    -- list[int] fields (absent_teeth_*, primary_teeth_*,
                        adjacent_teeth, intrasinusal_teeth, abutment_teeth,
                        implant_supported_teeth, pontic_teeth, fdi_numbers):
                        precision/recall/F1 over the FDI sets.
  2. exact_match     -- bool/enum scalar sub-fields: accuracy + per-class
                        precision/recall/F1.
  3. list_pattern    -- periodontal_bone_resorption.pattern specifically
                        (list[string], both values can apply at once) --
                        same set_overlap logic as #1, just not FDI numbers.
  4. list_object_match -- regions / remnants / affected_teeth / implants /
                        fixed_bridges: each ground-truth entry is matched
                        to its closest prediction entry by a KEY field
                        (side / location / tooth / fdi_numbers / span),
                        unmatched entries count as false negatives/
                        positives, and only MATCHED pairs get their other
                        sub-fields compared (comparing sub-fields of two
                        entries that don't even refer to the same
                        finding would be meaningless).
  5. skipped (free text) -- radiopaque_findings/radiolucent_findings/
                        findings/details: not fact-comparable by exact
                        match; excluded from scoring rather than scored
                        by a metric (string similarity) that doesn't
                        actually answer "was the fact right."

dentition_type IS NOT a field on either side (see parse_reports_to_gt.py)
-- it's DERIVED, so it's evaluated by running the SAME derivation
(postprocess_pred.py's build_dentition_type) against both the ground
truth's and the prediction's own primary_teeth/eruption facts, then
comparing the two derived labels. Comparing two derivations only means
something if both sides ran the identical derivation logic.

Usage:
    python code/eval/evaluate_predictions.py \
        --pred-dir dataset/validate/outputs/predictions \
        --gt-dir   dataset/validate/outputs/ground_truth \
        --schema   schema/schema.json \
        --out      dataset/validate/outputs/evaluation/evaluation_$(date +%Y%m%d_%H%M%S).json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from normalize_pred import normalize_prediction  # noqa: E402
from postprocess_pred import build_dentition_type, ALL_FDIS  # noqa: E402
# The same gate build_vqa_pairs.py asks before it puts a per-tooth question in
# a call, so a fact can never be scored at an FDI it was never asked at.
from build_vqa_pairs import fact_applies_to_fdi  # noqa: E402


# ── Generic metric primitives ──────────────────────────────────────────────

_MAXILLA_SCOPE_FOLD = {"fully_included": "included", "partially_included": "included"}


def fold_maxilla_scope(pred: Dict) -> None:
    """Collapse a legacy prediction's maxilla_scope to the two values both
    sides now use.

    A NO-OP FOR SCHEMA v6.6 AND LATER, kept for replaying older predictions.
    The two sides used to be asked at different granularity: the VLM answers
    off the 3d_frontal render, where a volume boundary cutting across the
    maxilla is visible, so the schema kept fully_included|partially_included|
    not_included, while the ground truth comes from a REPORT, which does not
    settle that -- of 51 training reports whose extraction said
    fully_included, 4 state an extent at all and 35 state nothing about the
    scan's extent anywhere -- so parse_reports_to_gt.py emits
    included|not_included and nothing else (see MAXILLA_SCOPES there).
    Comparing the two vocabularies directly would have marked every correct
    "partially_included" wrong.

    v6.6 retired the three-way answer: maxilla_included is included|
    not_included, the same pair, so a current prediction arrives already
    folded and passes through untouched. The fold stays because a v6.5
    prediction file replayed against this code still needs it, and because
    the folding belongs here at the comparison rather than in
    normalize_pred.py -- the prediction on disk stays exactly what the model
    said.

    Mutates `pred` in place; a missing or already-folded field is a no-op.
    """
    scope = pred.get("global", {}).get("maxilla_scope")
    if not isinstance(scope, dict):
        return
    folded = _MAXILLA_SCOPE_FOLD.get(scope.get("maxilla_included"))
    if folded:
        scope["maxilla_included"] = folded


def set_overlap_metric(gt_set: set, pred_set: set) -> Dict:
    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if not pred_set and not gt_set else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else (1.0 if not pred_set and not gt_set else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"type": "set_overlap", "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "n_instances": len(gt_set)}


def exact_match_metric(gt_values: List, pred_values: List) -> Dict:
    """gt_values/pred_values are parallel lists (one pair per case) of a
    scalar field's value -- accuracy + per-class precision/recall/F1.

    A GT value of None means THE REPORT DID NOT SAY (parse_reports_to_gt.py
    fills a neutral value wherever the schema has one and leaves null only
    where it does not -- mandible_canal_*.location is enum lingual|buccal,
    and a report that never traces the canal supports neither). Scoring the
    prediction against that would mark it wrong for answering a question
    the reference never answered, so those pairs are dropped from the
    comparison and counted in n_report_silent instead of being hidden."""
    n_all = len(gt_values)
    pairs = [(g, p) for g, p in zip(gt_values, pred_values) if g is not None]
    n_silent = n_all - len(pairs)
    n = len(pairs)
    if n == 0:
        return {"type": "exact_match", "f1": None, "accuracy": None, "per_class": {},
                "n": 0, "n_report_silent": n_silent}
    gt_values = [g for g, _ in pairs]
    pred_values = [p for _, p in pairs]
    correct = sum(1 for g, p in pairs if g == p)
    accuracy = correct / n

    # Sort by (type name, str) rather than by value: a field can legitimately
    # carry both a bool and a string across cases (the VLM answers "true"
    # where the extractor answers True, or an enum field degrades to a free
    # string), and Python refuses to order str against bool. Only the label
    # ordering is at stake here, so a total order that never raises is
    # strictly better than a crash that loses the entire evaluation.
    classes = sorted({v for v in gt_values if v is not None} | {v for v in pred_values if v is not None},
                     key=lambda v: (type(v).__name__, str(v)))
    per_class = {}
    f1s = []
    for c in classes:
        tp = sum(1 for g, p in zip(gt_values, pred_values) if g == c and p == c)
        fp = sum(1 for g, p in zip(gt_values, pred_values) if g != c and p == c)
        fn = sum(1 for g, p in zip(gt_values, pred_values) if g == c and p != c)
        support = sum(1 for g in gt_values if g == c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        if support:
            f1s.append(f1)

    macro_f1 = sum(f1s) / len(f1s) if f1s else None
    return {"type": "exact_match", "f1": macro_f1, "accuracy": accuracy,
            "per_class": per_class, "n": n, "n_report_silent": n_silent}


# ── list[object] matching ──────────────────────────────────────────────────

def match_list_objects(gt_list: List[Dict], pred_list: List[Dict],
                       key_fn) -> Tuple[List[Tuple[Dict, Dict]], List[Dict], List[Dict]]:
    """
    Pairs gt/pred entries whose key_fn(entry) is equal. Returns
    (matched_pairs, unmatched_gt, unmatched_pred). key_fn must return a
    hashable value (or a frozenset for set-valued keys like fdi_numbers).
    An entry whose key_fn raises/returns None is treated as unmatchable
    (goes straight to the unmatched list) rather than crashing the whole
    comparison over one malformed entry.
    """
    def safe_key(e):
        try:
            k = key_fn(e)
            return k if k is not None else object()  # unique sentinel, never matches
        except Exception:
            return object()

    gt_by_key = {}
    for e in gt_list:
        gt_by_key.setdefault(safe_key(e), []).append(e)

    matched, unmatched_pred = [], []
    for p in pred_list:
        k = safe_key(p)
        bucket = gt_by_key.get(k)
        if bucket:
            matched.append((bucket.pop(0), p))
            if not bucket:
                del gt_by_key[k]
        else:
            unmatched_pred.append(p)

    unmatched_gt = [e for bucket in gt_by_key.values() for e in bucket]
    return matched, unmatched_gt, unmatched_pred


def list_object_metric(gt_list: List[Dict], pred_list: List[Dict], key_fn,
                       subfield_comparisons: Dict[str, str]) -> Dict:
    """
    subfield_comparisons: {field_name: 'exact'|'set_overlap'} -- only
    applied to MATCHED pairs (comparing a sub-field of two entries that
    don't even refer to the same finding is meaningless).
    """
    matched, unmatched_gt, unmatched_pred = match_list_objects(gt_list, pred_list, key_fn)
    tp, fn, fp = len(matched), len(unmatched_gt), len(unmatched_pred)
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if not pred_list and not gt_list else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else (1.0 if not pred_list and not gt_list else 0.0)
    entry_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    subfield_results = {}
    for field, kind in subfield_comparisons.items():
        gt_vals = [g.get(field) for g, p in matched]
        pred_vals = [p.get(field) for g, p in matched]
        if kind == "exact":
            subfield_results[field] = exact_match_metric(gt_vals, pred_vals)
        elif kind == "set_overlap":
            per_pair = [set_overlap_metric(set(g or []), set(p or [])) for g, p in
                       zip(gt_vals, pred_vals)]
            if per_pair:
                subfield_results[field] = {
                    "type": "set_overlap",
                    "precision": sum(m["precision"] for m in per_pair) / len(per_pair),
                    "recall": sum(m["recall"] for m in per_pair) / len(per_pair),
                    "f1": sum(m["f1"] for m in per_pair) / len(per_pair),
                    "n_instances": len(per_pair),
                }

    return {"type": "list_object_match", "entry_precision": precision,
            "entry_recall": recall, "entry_f1": entry_f1,
            "tp": tp, "fp": fp, "fn": fn, "subfields": subfield_results}


# ── Field-by-field walk over one section ────────────────────────────────────

# Free text, not fact-comparable by exact match.
#   visual_evidence -- v6.1 puts one on EVERY object fact. Scoring it by
#                      string equality would be scoring nothing.
#   findings        -- kept for older predictions only: v6.1 dropped the prose
#                      tooth_{fdi}_periodontal_status.findings field. It does
#                      NOT skip dental_arch_findings_{arch}.findings, which is
#                      a structured FDI -> finding map and is scored below --
#                      the two are told apart by the spec's type head, not by
#                      the shared name.
#   uncertain_teeth -- the model's own doubt about its arch-findings answer
#                      (see postprocess_pred's uncertainty gate). It is
#                      introspection, not a fact about the patient: a report
#                      cannot state it, so the extractor can only ever return
#                      [], and scoring pred against that empty set would
#                      report precision 0.0 for a field where the model was
#                      doing exactly what it was asked.
SKIP_FIELDS = {"visual_evidence", "uncertain_teeth", "findings", "details",
               "radiopaque_findings", "radiolucent_findings"}


def _as_set(value) -> set:
    """A list-valued field as a set, tolerating a bare scalar or None."""
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {v for v in value if isinstance(v, (str, int, float, bool))}
    return {value} if isinstance(value, (str, int, float, bool)) else set()


def _spec_of(fact: Dict, field: str) -> str:
    spec = fact.get("object_fields", {}).get(field, "")
    return spec if isinstance(spec, str) else ""


def _is_list_object_field(fact: Dict, field: str) -> bool:
    return _spec_of(fact, field).startswith("list[object]")


def _is_list_int_field(fact: Dict, field: str) -> bool:
    return _spec_of(fact, field).startswith("list[int]")


def _is_list_string_field(fact: Dict, field: str) -> bool:
    """
    periodontal_bone_resorption_{arch}.pattern is the only one: list[string]
    where BOTH values (vertical, horizontal) can apply at once, so it is
    scored as a set, not by equality. A table declaring that intent existed
    but was never actually consulted, and a list reaching the exact-match
    path raised "unhashable type: 'list'" the moment a ground
    truth carried a pattern -- comparison strategy #3 in the docstring above
    never ran.
    """
    return _spec_of(fact, field).startswith("list[string]")


def _is_map_field(fact: Dict, field: str) -> bool:
    """A free-keyed object sub-field (dental_arch_findings' FDI -> finding)."""
    spec = _spec_of(fact, field)
    return spec.startswith("object") and not spec.startswith("object_")


def tooth_map_metric(samples: List[Tuple[Dict, Dict]]) -> Dict:
    """
    dental_arch_findings_{arch}.findings: a per-tooth map, scored two ways at
    once, because both halves of it are real claims.

      * which teeth are listed at all (they are the PRESENT teeth) --
        set_overlap over the key sets;
      * what each shared tooth was called -- exact match over the values of
        the teeth BOTH sides listed. Scoring a value against a tooth the other
        side never mentioned would double-count the key-set error.
    """
    key_metrics, gt_values, pred_values = [], [], []
    for gt_map, pred_map in samples:
        gt_map = gt_map if isinstance(gt_map, dict) else {}
        pred_map = pred_map if isinstance(pred_map, dict) else {}
        gt_keys = {str(k) for k in gt_map}
        pred_keys = {str(k) for k in pred_map}
        key_metrics.append(set_overlap_metric(gt_keys, pred_keys))
        for key in sorted(gt_keys & pred_keys):
            gt_values.append(gt_map.get(key, gt_map.get(int(key)) if key.isdigit() else None))
            pred_values.append(pred_map.get(key, pred_map.get(int(key)) if key.isdigit() else None))

    n = len(key_metrics) or 1
    return {
        "type": "tooth_map",
        "teeth_listed": {
            "type": "set_overlap",
            "precision": sum(m["precision"] for m in key_metrics) / n,
            "recall": sum(m["recall"] for m in key_metrics) / n,
            "f1": sum(m["f1"] for m in key_metrics) / n,
            "n_instances": len(key_metrics),
        },
        "finding_per_tooth": exact_match_metric(gt_values, pred_values),
    }


# Match keys + sub-field comparisons for each list[object] shape in the schema.
# v6.1 renamed/reshaped every one of these: implants and bridges moved INSIDE
# their arch fact as sub-fields (so they are matched here, not at fact level),
# and regions / remnants / affected_teeth are gone with the facts that held
# them.
_LIST_OBJECT_SPECS = {
    "implants": (lambda e: e.get("fdi_number"),
                 {"with_crown": "exact", "osseointegration_status": "exact"}),
    "bridges":  (lambda e: (e.get("span_start"), e.get("span_end"))
                 if e.get("span_start") is not None and e.get("span_end") is not None else None,
                 {"abutment_teeth": "set_overlap",
                  "implant_supported_teeth": "set_overlap",
                  "pontic_teeth": "set_overlap"}),
}


def _implants_key(e):
    """Legacy v4 shape (a bare list[object] fact with an fdi_numbers list)."""
    fdis = e.get("fdi_numbers")
    return frozenset(fdis) if fdis else None


def _bridges_key(e):
    if e.get("span_start") is None or e.get("span_end") is None:
        return None
    return (e["span_start"], e["span_end"])


def evaluate_case_fact(fact: Dict, gt_val: Any, pred_val: Any, results: Dict) -> None:
    """
    Accumulates ONE case's contribution to `results[fact_id][field]` for
    every scalar sub-field, and appends this case's (gt_list, pred_list)
    for later per-field list_object_metric computation for list[object]
    top-level facts. `results` is mutated in place across many calls (one
    per case) before final aggregation.
    """
    fid = fact["output_field"]
    results.setdefault(fid, {"_fact": fact, "_pairs": {}})
    bucket = results[fid]["_pairs"]

    if fact["type"] == "enum" and "object_fields" not in fact:
        # Bare enum fact (tooth_{fdi}_restoration) -- no sub-fields at all.
        bucket.setdefault("__value__", []).append((gt_val, pred_val))
        return

    if fact["type"] == "list[int]":
        bucket.setdefault("__value__", []).append(
            (set(gt_val or []), set(pred_val or [])))
        return

    if fact["type"] == "list[object]":
        bucket.setdefault("__value__", []).append(
            (gt_val if isinstance(gt_val, list) else [],
             pred_val if isinstance(pred_val, list) else []))
        return

    # Plain object fact: walk its declared sub-fields.
    gt_obj = gt_val if isinstance(gt_val, dict) else {}
    pred_obj = pred_val if isinstance(pred_val, dict) else {}
    for field in fact.get("object_fields", {}):
        # SKIP_FIELDS is by NAME, so it also matches
        # dental_arch_findings_{arch}.findings -- which is a structured
        # per-tooth map, not the prose the skip list is for. The declared type
        # decides, not the name.
        if field in SKIP_FIELDS and not _is_map_field(fact, field):
            continue
        if _is_list_object_field(fact, field):
            bucket.setdefault(field, []).append(
                (gt_obj.get(field) if isinstance(gt_obj.get(field), list) else [],
                 pred_obj.get(field) if isinstance(pred_obj.get(field), list) else []))
        elif _is_list_int_field(fact, field) or _is_list_string_field(fact, field):
            bucket.setdefault(field, []).append(
                (_as_set(gt_obj.get(field)), _as_set(pred_obj.get(field))))
        elif _is_map_field(fact, field):
            bucket.setdefault(field, []).append(
                (gt_obj.get(field) if isinstance(gt_obj.get(field), dict) else {},
                 pred_obj.get(field) if isinstance(pred_obj.get(field), dict) else {}))
        else:
            bucket.setdefault(field, []).append((gt_obj.get(field), pred_obj.get(field)))


def aggregate_list_object_results(per_case: List[Dict]) -> Optional[Dict]:
    """
    Averages entry-level precision/recall/F1 across cases AND aggregates
    each sub-field's own metric across cases -- the earlier version of
    this dropped the subfield results during aggregation entirely,
    silently losing the actual "did we get osseointegration_status /
    periodontal_bone_resorption / orientation right" signal on every
    matched entry, which defeats much of the point of comparing these
    facts at all. Caught by testing against a synthetic pair with a
    deliberate sub-field mismatch on an otherwise-matched entry.

    Exact-match sub-fields are pooled (weighted by n) rather than
    averaging each case's own accuracy, so a case with one matched entry
    doesn't count as much as a case with ten.
    """
    if not per_case:
        return None
    result = {
        "type": "list_object_match",
        "entry_precision": sum(m["entry_precision"] for m in per_case) / len(per_case),
        "entry_recall": sum(m["entry_recall"] for m in per_case) / len(per_case),
        "entry_f1": sum(m["entry_f1"] for m in per_case) / len(per_case),
        "tp": sum(m["tp"] for m in per_case), "fp": sum(m["fp"] for m in per_case),
        "fn": sum(m["fn"] for m in per_case),
    }
    subfield_names = set()
    for m in per_case:
        subfield_names.update(m.get("subfields", {}).keys())
    if subfield_names:
        pooled = {}
        for name in subfield_names:
            kind_entries = [m["subfields"][name] for m in per_case if name in m.get("subfields", {})]
            if not kind_entries:
                continue
            if kind_entries[0]["type"] == "exact_match":
                # A case where this fact matched NO entries (no implants at
                # all, say) yields accuracy=None/n=0 from exact_match_metric.
                # Those carry no signal and must be dropped BEFORE the
                # weighted sum -- None * 0 is a TypeError, not a no-op, so
                # leaving them in crashes the whole evaluation on the first
                # case that happens to lack the finding.
                scored = [e for e in kind_entries
                          if e.get("n") and e.get("accuracy") is not None]
                total_n = sum(e["n"] for e in scored)
                weighted_acc = (sum(e["accuracy"] * e["n"] for e in scored) / total_n
                               if total_n else None)
                pooled[name] = {"type": "exact_match", "accuracy": weighted_acc,
                                "n": total_n, "n_cases_scored": len(scored)}
            else:
                # Same guard as above: an entry missing any of the three
                # metrics contributes nothing and must not poison the mean.
                scored = [e for e in kind_entries
                          if all(e.get(k) is not None for k in ("precision", "recall", "f1"))]
                if not scored:
                    continue
                pooled[name] = {
                    "type": "set_overlap",
                    "precision": sum(e["precision"] for e in scored) / len(scored),
                    "recall": sum(e["recall"] for e in scored) / len(scored),
                    "f1": sum(e["f1"] for e in scored) / len(scored),
                    "n_cases_scored": len(scored),
                }
        result["subfields"] = pooled
    return result


def finalize_fact_results(results: Dict) -> Dict:
    out = {}
    for fid, entry in results.items():
        fact = entry["_fact"]
        pairs = entry["_pairs"]
        field_out = {}
        for field, samples in pairs.items():
            if field == "__value__" and fact["type"] == "list[int]":
                per_case = [set_overlap_metric(g, p) for g, p in samples]
                field_out[fid] = {
                    "type": "set_overlap",
                    "precision": sum(m["precision"] for m in per_case) / len(per_case),
                    "recall": sum(m["recall"] for m in per_case) / len(per_case),
                    "f1": sum(m["f1"] for m in per_case) / len(per_case),
                    "n_instances": len(per_case),
                }
            elif field == "__value__" and fact["type"] == "list[object]":
                if fid.startswith("implants"):
                    key_fn = _implants_key
                    subfields = {"with_crown": "exact", "osseointegration_status": "exact"}
                elif fid.startswith("fixed_bridges"):
                    key_fn = _bridges_key
                    subfields = {"abutment_teeth": "set_overlap", "implant_supported_teeth": "set_overlap",
                                "pontic_teeth": "set_overlap"}
                else:
                    key_fn, subfields = None, {}
                per_case = [list_object_metric(g, p, key_fn, subfields) for g, p in samples]
                aggregated = aggregate_list_object_results(per_case)
                if aggregated:
                    field_out[fid] = aggregated
            elif field == "__value__":
                gts, preds = zip(*samples)
                field_out[fid] = exact_match_metric(list(gts), list(preds))
            elif field in _LIST_OBJECT_SPECS and _is_list_object_field(fact, field):
                key_fn, subfields = _LIST_OBJECT_SPECS[field]
                per_case = [list_object_metric(g, p, key_fn, subfields) for g, p in samples]
                aggregated = aggregate_list_object_results(per_case)
                if aggregated:
                    field_out[f"{fid}.{field}"] = aggregated
            elif _is_map_field(fact, field):
                field_out[f"{fid}.{field}"] = tooth_map_metric(samples)
            elif isinstance(samples[0][0], set):
                per_case = [set_overlap_metric(g, p) for g, p in samples]
                field_out[f"{fid}.{field}"] = {
                    "type": "set_overlap",
                    "precision": sum(m["precision"] for m in per_case) / len(per_case),
                    "recall": sum(m["recall"] for m in per_case) / len(per_case),
                    "f1": sum(m["f1"] for m in per_case) / len(per_case),
                    "n_instances": len(per_case),
                }
            else:
                gts, preds = zip(*samples)
                field_out[f"{fid}.{field}"] = exact_match_metric(list(gts), list(preds))
        out.update(field_out)
    return out


# ── Per-case walk ────────────────────────────────────────────────────────────

def walk_case(gt: Dict, pred: Dict, schema: Dict, results: Dict) -> None:
    gt_g, pred_g = gt.get("global", {}) or {}, pred.get("global", {}) or {}
    for section in ("mandible", "maxilla"):
        for fact in schema.get(section, []) or []:
            fid = fact["output_field"]
            evaluate_case_fact(fact, gt_g.get(fid), pred_g.get(fid), results)

    gt_teeth, pred_teeth = gt.get("teeth", {}) or {}, pred.get("teeth", {}) or {}
    for fdi in ALL_FDIS:
        gt_tooth = gt_teeth.get(f"tooth_{fdi}", {}) or {}
        pred_tooth = pred_teeth.get(f"tooth_{fdi}", {}) or {}
        for fact_template in schema.get("dental_elements", []) or []:
            # A fact gated to some FDIs (tooth_{fdi}_mandible_canal -> lower
            # molars) was never asked at the other 26 positions, so neither
            # side has a value there. Scoring them anyway would pair None with
            # None 26 times per case and count each as a hit -- an accuracy
            # made of questions nobody was asked.
            if not fact_applies_to_fdi(fact_template, fdi):
                continue
            fid_template = fact_template["output_field"]
            fid = fid_template.replace("{fdi}", str(fdi))
            fact = {**fact_template, "output_field": fid_template}  # group all teeth under the template id
            evaluate_case_fact(fact, gt_tooth.get(fid), pred_tooth.get(fid), results)

    # dentition_type: DERIVED on both sides from primary_teeth + eruption,
    # not a directly-comparable field -- see module docstring.
    gt_dtype = build_dentition_type(gt_g.get("primary_teeth_mandible", []),
                                    gt_g.get("primary_teeth_maxilla", []), gt_teeth)
    pred_dtype = build_dentition_type(pred_g.get("primary_teeth_mandible", []),
                                      pred_g.get("primary_teeth_maxilla", []), pred_teeth)
    results.setdefault("dentition_type (derived)", {"_fact": {"type": "enum"}, "_pairs": {}})
    results["dentition_type (derived)"]["_pairs"].setdefault("__value__", []).append(
        (gt_dtype["dentition_type"], pred_dtype["dentition_type"]))


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--schema", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--case-ids", nargs="+", default=None)
    args = ap.parse_args()

    with open(args.schema) as f:
        schema = json.load(f)

    pred_dir, gt_dir = Path(args.pred_dir), Path(args.gt_dir)
    pred_files = sorted(pred_dir.glob("*_pred.json"))
    if args.case_ids:
        pred_files = [p for p in pred_files if p.stem.replace("_pred", "") in args.case_ids]

    results: Dict = {}
    n_evaluated, n_missing_gt = 0, 0
    for pf in pred_files:
        case_id = pf.stem.replace("_pred", "")
        gt_path = gt_dir / f"{case_id}_gt.json"
        if not gt_path.exists():
            print(f"[{case_id}] [SKIP] no ground truth found: {gt_path}", file=sys.stderr)
            n_missing_gt += 1
            continue

        pred = json.loads(pf.read_text())
        # Fold BEFORE normalizing, not after: v6.6's enum no longer contains
        # fully_included, so normalize_prediction would null a v6.5 answer as
        # schema-violating and the fold would find nothing left to fold.
        fold_maxilla_scope(pred)
        pred, _ = normalize_prediction(pred, args.schema)
        gt = json.loads(gt_path.read_text())

        walk_case(gt, pred, schema, results)
        n_evaluated += 1

    final = finalize_fact_results(results)

    out = {"n_cases_evaluated": n_evaluated, "n_cases_missing_gt": n_missing_gt,
          "fields": final}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Evaluated {n_evaluated} case(s) ({n_missing_gt} missing ground truth) -> {args.out}")


if __name__ == "__main__":
    main()