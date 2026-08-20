#!/usr/bin/env python3
"""
code/ground_truth/survey_gt_quality.py

Data-quality survey of one split's LABEL side, before any model is involved:
how much the radiologists agree with each other, how much the `facts/`
sidecar agrees with the reports, and how much of the disagreement is the
known left-right MIRROR error. Then one score per case and a ranking.

This is the label-side twin of structured_findings_evaluation.py, which
surveys the PREDICTION side. Same reflexes: count first, score only what the
sources can actually settle, write a timestamped file and diff it against the
previous one.

WHAT IS COMPARED AGAINST WHAT
─────────────────────────────
  reports  ->  dataset/<split>/outputs/ground_truth/{case}[_{reader}]_report_facts.json
               parse_reports_to_gt.py stage 1: the small, report-shaped
               extraction (present/absent lists, crowns, post-and-core,
               implants, bridges, scopes, canal course). One file per
               radiologist, plus a consensus file per case.
  facts    ->  dataset/<split>/facts/{case}.json .structured
               teeth_present / teeth_absent (always a COMPLETE enumeration of
               all 32 positions), crowns, implants, bridge_present,
               ian_close_teeth, fov. Derived from the segmentation mask --
               `present_label_ids` is the mask's own label set -- so facts is
               a mask-space claim, and the report is a report-space claim.

The reports are the label (they are what the challenge scores against) and
the facts are context that the image generators bake into captions, so the
question this file answers is asymmetric on purpose: NOT "do they agree"
but "given the report, how correct is the facts file".

WHY THE STAGE-1 INTERMEDIATE AND NOT {case}_gt.json
────────────────────────────────────────────────────
{case}_gt.json is the expansion: every unstated position gets a default, so
comparing it to facts would score the DEFAULTS -- an agreement number
dominated by teeth no radiologist ever mentioned. report_facts holds only
what the text states, which is exactly the set of positions on which the
two sources make competing claims. Silence is excluded rather than
scored, in both directions.

THREE SECTIONS
──────────────
  1. INTER-READER    cases with >= 2 reports. Pairwise, per channel; a case
                     with 3 readers contributes 3 pairs. Presence is scored
                     only where BOTH readers state the position (Cohen's
                     kappa over the pooled 2x2), set findings by Jaccard,
                     scalars by exact match where both are non-null.
                     This is the ceiling: no extraction can be more right
                     than the readers are consistent.
  2. REPORT-vs-FACTS all cases, consensus report_facts vs facts. Presence
                     over report-stated positions only; crowns / implants /
                     ian_close as set precision-recall-F1 with the report as
                     label; bridge_present as a 2x2.
  3. MIRROR          every channel recomputed with the FACTS mirrored
                     (quadrant 1<->2, 3<->4), and the two framings compared
                     on INFORMATIVE items only.

INFORMATIVE ITEMS ARE THE WHOLE TRICK IN SECTION 3
───────────────────────────────────────────────────
Most positions cannot tell the two framings apart. If facts says both 18
and 28 are absent, a report claiming "18 absent" agrees under either
framing and is evidence for neither. So an item counts only when the facts
value at the position DIFFERS from the facts value at its mirror partner --
which is the only situation where mirroring changes the answer. A case whose
dentition is left-right symmetric yields zero informative items and gets the
verdict `no-signal` rather than being forced into `aligned`; that is a real
category here, not a rounding of one.

Mirroring is applied to the FACTS side, never to the report. The report is
the label; a label you rotate to fit is not a label.

SCORING AND RANKING
───────────────────
One score per case in [0,1]: the weighted mean of the channels the case
actually supports, renormalized over those channels, so a case whose report
never mentions an implant is not penalised for the implant channel. Weights
(presence .40, crowns .20, implants .15, ian_close .15, bridge .10) follow
how much each channel steers the downstream captions: presence decides which
teeth create_panoramic.py outlines at all, bridge is one sentence.

Two scores are reported for every case -- `asis` (facts exactly as they sit
on disk, which is what every generator reads today) and `mirrored` (facts
with the quadrants swapped). The ranking uses `asis`, because that is what
"how correct are the facts" means for a pipeline that does not currently
correct them; `mirrored` sits next to it so the gap is visible per case.

n_evidence -- how many comparable items the case offered -- is reported and
breaks ties, because a 1.00 built on two claims is not the same object as a
0.95 built on forty, and the ranking should not pretend it is.
"""

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# ── FDI space ──────────────────────────────────────────────────────────────

ARCH_FDIS = {
    "mandible": [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38],
    "maxilla":  [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28],
}
ALL_FDIS = sorted(ARCH_FDIS["mandible"] + ARCH_FDIS["maxilla"])
_MIRROR_QUADRANT = {1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5, 7: 8, 8: 7}


def mirror_fdi(fdi: int) -> int:
    """18<->28, 38<->48: same tooth, other side. Primary quadrants too."""
    q, d = divmod(int(fdi), 10)
    return _MIRROR_QUADRANT.get(q, q) * 10 + d


def mirror_set(s: Set[int]) -> Set[int]:
    return {mirror_fdi(f) for f in s}


# ── Channels ───────────────────────────────────────────────────────────────
#
# SET_CHANNELS are compared between readers; the report-vs-facts side can
# only use the four the facts file also carries (crowns, implants,
# ian_close, bridge) plus presence.

SET_CHANNELS = [
    "crowns", "post_and_core", "fillings", "caries", "endodontic", "implants",
    "root_remnants", "root_fractures", "periapical_lesions", "primary_teeth",
    "unerupted", "uncertain_teeth",
]

# (label, arch, path) -- read from one arch's report-facts block.
SCALAR_CHANNELS = [
    ("periodontal_extent.mandible", "mandible", ("periodontal_extent",)),
    ("periodontal_extent.maxilla",  "maxilla",  ("periodontal_extent",)),
    ("alveolar_atrophy.mandible",   "mandible", ("alveolar_atrophy",)),
    ("alveolar_atrophy.maxilla",    "maxilla",  ("alveolar_atrophy",)),
    ("condyle_right_scope",         "mandible", ("condyle_right_scope",)),
    ("condyle_left_scope",          "mandible", ("condyle_left_scope",)),
    ("canal_right.location",        "mandible", ("canal_right", "location")),
    ("canal_left.location",         "mandible", ("canal_left", "location")),
    ("maxilla_scope",               "maxilla",  ("maxilla_scope",)),
    ("sinus_right.mucosa_state",    "maxilla",  ("sinus_right", "mucosa_state")),
    ("sinus_left.mucosa_state",     "maxilla",  ("sinus_left", "mucosa_state")),
    ("sinus_right.sinus_content",   "maxilla",  ("sinus_right", "sinus_content")),
    ("sinus_left.sinus_content",    "maxilla",  ("sinus_left", "sinus_content")),
]

# Report-vs-facts scoring weights. Renormalized per case over the channels
# that case supports -- see the header.
SCORE_WEIGHTS = {
    "presence": 0.40,
    "crowns": 0.20,
    "implants": 0.15,
    "ian_close": 0.15,
    "bridge": 0.10,
}

# Inter-reader score: presence carries the same relative weight it does
# above; the other two are the mean over whatever channels are defined.
INTERREADER_WEIGHTS = {"presence": 0.45, "sets": 0.30, "scalars": 0.25}

TIERS = [("A", 0.90), ("B", 0.75), ("C", 0.55), ("D", 0.0)]


# ── Small metrics ──────────────────────────────────────────────────────────

def jaccard(a: Set[int], b: Set[int]) -> Optional[float]:
    """None when both sides are empty: two readers who both said nothing have
    not agreed about anything, and averaging in a 1.0 there would drown the
    channel in cases it never applied to."""
    if not a and not b:
        return None
    return len(a & b) / len(a | b)


def prf(pred: Set[int], label: Set[int]) -> Optional[Dict[str, Any]]:
    """Set precision/recall/F1 with `label` as ground truth. None when both
    are empty (same reason as jaccard)."""
    if not pred and not label:
        return None
    tp, fp, fn = len(pred & label), len(pred - label), len(label - pred)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def cohen_kappa(table: Counter) -> Optional[float]:
    """Kappa over a pooled 2x2 Counter keyed (a_label, b_label)."""
    n = sum(table.values())
    if not n:
        return None
    labels = sorted({k for pair in table for k in pair})
    po = sum(table[(l, l)] for l in labels) / n
    pe = 0.0
    for l in labels:
        ra = sum(v for (a, _), v in table.items() if a == l) / n
        rb = sum(v for (_, b), v in table.items() if b == l) / n
        pe += ra * rb
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def mean(vals: Sequence[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def pct(x: Optional[float]) -> str:
    return "  n/a " if x is None else f"{x:6.1%}"


def num(x: Optional[float], w: int = 6, d: int = 3) -> str:
    return " " * (w - 3) + "n/a" if x is None else f"{x:{w}.{d}f}"


# ── Reading one side ───────────────────────────────────────────────────────

def _ints(v: Any) -> Set[int]:
    out = set()
    for x in v or []:
        if isinstance(x, dict):
            x = x.get("fdi", x.get("tooth"))
        try:
            out.add(int(x))
        except (TypeError, ValueError):
            pass
    return out


def flatten_report_facts(rf: Dict) -> Dict[str, Any]:
    """Both arches folded into one flat, comparable view.

    Arch membership is not itself a claim under test here -- an extractor
    that files 47 under maxilla is a stage-1 bug, and sanitize_report_facts
    already rejects those -- so the arches are unioned and every channel is
    one set of FDI numbers.
    """
    out: Dict[str, Any] = {k: set() for k in SET_CHANNELS}
    out["present"] = set()
    out["absent"] = set()
    out["canal_adjacent"] = set()
    out["bridges"] = []
    out["presence_enumerated"] = {}
    for arch in ("mandible", "maxilla"):
        f = rf.get(arch) or {}
        out["present"] |= _ints(f.get("teeth_present"))
        out["absent"] |= _ints(f.get("teeth_absent"))
        for key in SET_CHANNELS:
            out[key] |= _ints(f.get(key))
        out["bridges"] += list(f.get("bridges") or [])
        out["presence_enumerated"][arch] = f.get("presence_enumerated")
        if arch == "mandible":
            for side in ("right", "left"):
                out["canal_adjacent"] |= _ints((f.get(f"canal_{side}") or {}).get("adjacent_teeth"))
    # A position claimed both present and absent by one reader is a stage-1
    # contradiction; resolve_presence() ranks them downstream, but for a
    # quality survey it is cleaner to drop it from BOTH sides and count it.
    out["self_conflict"] = out["present"] & out["absent"]
    out["present"] -= out["self_conflict"]
    out["absent"] -= out["self_conflict"]
    out["stated"] = out["present"] | out["absent"]
    return out


def flatten_facts(structured: Dict) -> Dict[str, Any]:
    """`stated` is the whole point of this function.

    The facts pool used to enumerate all 32 positions in every case, so
    "not in teeth_present" could be read as "absent". As of the 2026-08-10
    fov update it does not: 354 of 622 cases carry fov.maxilla "excluded"
    and list only the 16 mandibular positions, because a tooth outside the
    scanned volume is not a tooth the segmentation can call missing. A
    position in NEITHER list is now the facts declining to answer.

    Reading that silence as "absent" would score the facts wrong at every
    maxillary position of every excluded-maxilla case -- and would do it in
    the direction that looks like a regression, since the report happily
    names maxillary teeth the scan never covered. So `stated` is carried
    alongside, and every comparison below intersects with it. Same rule the
    report side already uses for the same reason.
    """
    present = _ints(structured.get("teeth_present"))
    absent = _ints(structured.get("teeth_absent"))
    return {
        "present": present,
        "absent": absent,
        "stated": present | absent,
        "crowns": _ints(structured.get("crowns")),
        "implants": _ints(structured.get("implants")),
        "ian_close": _ints(structured.get("ian_close_teeth")),
        "bridge_present": bool(structured.get("bridge_present")),
        "fov_maxilla": (structured.get("fov") or {}).get("maxilla"),
        "fov_condyles": (structured.get("fov") or {}).get("condyles"),
        "canals": structured.get("canals") or {},
    }


def mirror_facts(ff: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(ff)
    for key in ("present", "absent", "stated", "crowns", "implants", "ian_close"):
        out[key] = mirror_set(ff[key])
    return out


# ── Section 1: inter-reader ────────────────────────────────────────────────

def compare_readers(a: Dict, b: Dict) -> Dict[str, Any]:
    """One reader pair, on the pre-flattened views."""
    both = a["stated"] & b["stated"]
    agree = sum(1 for f in both if (f in a["present"]) == (f in b["present"]))
    table = Counter((("present" if f in a["present"] else "absent"),
                     ("present" if f in b["present"] else "absent")) for f in both)

    sets = {k: jaccard(a[k], b[k]) for k in SET_CHANNELS}
    return {
        "presence_both_stated": len(both),
        "presence_agree": agree,
        "presence_acc": agree / len(both) if both else None,
        "presence_table": table,
        # How much the two readers even talk about the same teeth. A low
        # value with a high presence_acc means they agree wherever they
        # overlap but describe different mouths.
        "coverage_jaccard": jaccard(a["stated"], b["stated"]),
        "sets": sets,
        "bridge_present": (bool(a["bridges"]) == bool(b["bridges"])),
    }


def interreader_case(readers: List[Tuple[str, Dict]], raw: List[Dict]) -> Dict[str, Any]:
    """readers: [(reader_id, flattened)]; raw: the matching unflattened dicts
    (the scalar channels live per arch and are read from those)."""
    pairs = []
    for i in range(len(readers)):
        for j in range(i + 1, len(readers)):
            cmp = compare_readers(readers[i][1], readers[j][1])
            cmp["scalars"] = {}
            for label, arch, path in SCALAR_CHANNELS:
                va, vb = raw[i].get(arch) or {}, raw[j].get(arch) or {}
                for k in path:
                    va = (va or {}).get(k) if isinstance(va, dict) else None
                    vb = (vb or {}).get(k) if isinstance(vb, dict) else None
                cmp["scalars"][label] = None if (va is None or vb is None) else (va == vb)
            cmp["readers"] = (readers[i][0], readers[j][0])
            pairs.append(cmp)

    presence = mean([p["presence_acc"] for p in pairs])
    setscore = mean([v for p in pairs for v in p["sets"].values()])
    scalars = [float(v) for p in pairs for v in p["scalars"].values() if v is not None]
    scalars += [float(p["bridge_present"]) for p in pairs]
    scalarscore = mean(scalars)

    parts = {"presence": presence, "sets": setscore, "scalars": scalarscore}
    wsum = sum(INTERREADER_WEIGHTS[k] for k, v in parts.items() if v is not None)
    score = (sum(INTERREADER_WEIGHTS[k] * v for k, v in parts.items() if v is not None) / wsum
             if wsum else None)

    return {
        "n_readers": len(readers),
        "n_pairs": len(pairs),
        "presence_acc": presence,
        "presence_both_stated": sum(p["presence_both_stated"] for p in pairs),
        "presence_contradictions": sum(p["presence_both_stated"] - p["presence_agree"]
                                       for p in pairs),
        "coverage_jaccard": mean([p["coverage_jaccard"] for p in pairs]),
        "set_agreement": setscore,
        "scalar_agreement": scalarscore,
        "score": score,
        "_pairs": pairs,
    }


# ── Section 2 + 3: report vs facts, both framings ──────────────────────────

def compare_report_facts(rep: Dict, ff: Dict) -> Dict[str, Any]:
    """One framing (facts as given, or mirrored). Report is the label.

    Scored only where BOTH sources speak: the report's stated positions
    intersected with the facts' stated positions. A maxillary tooth the
    report names in a case whose scan excluded the maxilla is not a fact
    the segmentation got wrong -- it is a region it never saw. Those pairs
    are counted in presence_n_facts_silent rather than scored, exactly as
    structured_findings_evaluation.py drops null-GT pairs instead of marking the
    prediction wrong for answering an unasked question.

    The positive-only channels (crowns, implants, ian_close) get the same
    treatment by filtering the REPORT side down to facts-covered positions
    -- otherwise every crown the report names in an unscanned arch counts
    as a miss by the facts.
    """
    stated = rep["stated"] & ff["stated"]
    n = len(stated)
    correct = sum(1 for f in stated if (f in rep["present"]) == (f in ff["present"]))
    table = Counter((("present" if f in rep["present"] else "absent"),
                     ("present" if f in ff["present"] else "absent")) for f in stated)
    covered = ff["stated"]

    out = {
        "presence_n": n,
        "presence_n_facts_silent": len(rep["stated"] - ff["stated"]),
        "presence_correct": correct,
        "presence_acc": correct / n if n else None,
        "presence_table": table,
        "crowns": prf(ff["crowns"], {t for t in rep["crowns"] if t in covered}),
        "implants": prf(ff["implants"], {t for t in rep["implants"] if t in covered}),
        "ian_close": prf(ff["ian_close"],
                         {t for t in rep["canal_adjacent"] if t in covered}),
        "bridge_facts": ff["bridge_present"],
        "bridge_report": bool(rep["bridges"]),
    }
    out["bridge_match"] = float(out["bridge_facts"] == out["bridge_report"])

    channels = {
        "presence": out["presence_acc"],
        "crowns": out["crowns"]["f1"] if out["crowns"] else None,
        "implants": out["implants"]["f1"] if out["implants"] else None,
        "ian_close": out["ian_close"]["f1"] if out["ian_close"] else None,
        "bridge": out["bridge_match"],
    }
    wsum = sum(SCORE_WEIGHTS[k] for k, v in channels.items() if v is not None)
    out["channels"] = channels
    out["score"] = (sum(SCORE_WEIGHTS[k] * v for k, v in channels.items() if v is not None)
                    / wsum) if wsum else None
    out["n_evidence"] = n + sum(
        len({t for t in rep[k] if t in covered} | ff[fk])
        for k, fk in (("crowns", "crowns"), ("implants", "implants"))
    ) + len({t for t in rep["canal_adjacent"] if t in covered} | ff["ian_close"]) + 1
    return out


def mirror_evidence(rep: Dict, ff: Dict) -> Dict[str, Any]:
    """Direct-vs-mirrored tally over INFORMATIVE items only -- see header.

    An item is informative when mirroring the facts changes the answer at
    that position, i.e. facts(pos) != facts(mirror(pos)). Everything else is
    silent about the question and is excluded rather than counted as
    agreement for both framings.

    A position the facts do not state is excluded too, and so is one whose
    mirror partner they do not state: comparing a claim against a silence
    would make "the maxilla was not scanned" look like mirror evidence.
    Mirroring never crosses arches (31<->41, 11<->21), so in practice this
    drops the maxilla wholesale in the fov.maxilla == excluded cases, which
    is right -- those cases carry mandibular evidence only.
    """
    tally = {"n": 0, "direct": 0, "mirror": 0, "by_channel": {}}
    detail = {}

    covered = ff["stated"]

    def add(name: str, items: Set[int], in_facts):
        inf = [t for t in items
               if t in covered and mirror_fdi(t) in covered
               and in_facts(t) != in_facts(mirror_fdi(t))]
        d = sum(1 for t in inf if in_facts(t))
        m = sum(1 for t in inf if in_facts(mirror_fdi(t)))
        tally["by_channel"][name] = {"n": len(inf), "direct": d, "mirror": m}
        tally["n"] += len(inf)
        tally["direct"] += d
        tally["mirror"] += m
        detail[name] = sorted(inf)

    # Presence: the report claim is "present" or "absent"; the facts answer at
    # a position is its presence bit. Agreement means the bits match, so the
    # indicator is "does facts agree with what the report said here".
    pres_items = rep["stated"] & covered
    inf = [t for t in pres_items
           if mirror_fdi(t) in covered
           and (t in ff["present"]) != (mirror_fdi(t) in ff["present"])]
    d = sum(1 for t in inf if (t in rep["present"]) == (t in ff["present"]))
    m = sum(1 for t in inf if (t in rep["present"]) == (mirror_fdi(t) in ff["present"]))
    tally["by_channel"]["presence"] = {"n": len(inf), "direct": d, "mirror": m}
    tally["n"] += len(inf); tally["direct"] += d; tally["mirror"] += m
    detail["presence"] = sorted(inf)

    add("crowns", rep["crowns"], lambda t: t in ff["crowns"])
    add("implants", rep["implants"], lambda t: t in ff["implants"])

    # Thresholds: at least two informative items must break the same way AND
    # the winning framing must take 70% of them. One item is a typo away from
    # a verdict, and a 3-2 split is not a finding.
    n, d, m = tally["n"], tally["direct"], tally["mirror"]
    if n == 0:
        verdict = "no-signal"
    elif m - d >= 2 and m / n >= 0.70:
        verdict = "mirrored"
    elif d - m >= 2 and d / n >= 0.70:
        verdict = "aligned"
    else:
        verdict = "ambiguous"
    tally["verdict"] = verdict
    tally["margin"] = (m - d) / n if n else None
    tally["informative_items"] = detail
    return tally


# ── The extraction-independent control ─────────────────────────────────────
#
# Section 3 is a claim about the data, and everything it reads went through
# an LLM first. If stage 1 swapped quadrants while extracting, the mirror
# tally would be measuring the extractor. So the same tally is recomputed
# from a regex over the raw report text -- sentences containing an absence
# word, FDI numbers pulled out of them, compared against facts.teeth_absent.
# It recovers far fewer items (it cannot read "the second quadrant is
# edentulous") and it will pick up a stray number now and then, but it
# shares no code and no model with stage 1: if the two disagree, believe
# neither, and if they agree, the flip is in the data.

_ABSENCE_WORD = re.compile(r"absen|missing|agenes|edentul", re.I)
_FDI_IN_TEXT = re.compile(r"\b([1-4])\s*\.?\s*([1-8])\b")


def regex_mirror_control(reports_dir: Path, facts: Dict[str, Dict]) -> Dict[str, Any]:
    out = {"n": 0, "direct": 0, "mirror": 0, "cases_with_signal": 0, "reports_read": 0}
    if not reports_dir.is_dir():
        out["skipped"] = f"no reports dir at {reports_dir}"
        return out
    for case, structured in sorted(facts.items()):
        reports = sorted(reports_dir.glob(f"{case}_*.txt"))
        if not reports:
            continue
        out["reports_read"] += 1
        text = reports[0].read_text(encoding="utf-8", errors="replace")
        f_absent = _ints(structured.get("teeth_absent"))
        claimed: Set[int] = set()
        for sentence in re.split(r"[.;]", text):
            if _ABSENCE_WORD.search(sentence):
                claimed |= {int(m.group(1)) * 10 + int(m.group(2))
                            for m in _FDI_IN_TEXT.finditer(sentence)}
        n = d = m = 0
        for t in claimed:
            if (t in f_absent) == (mirror_fdi(t) in f_absent):
                continue                      # uninformative, same rule as above
            n += 1
            d += t in f_absent
            m += mirror_fdi(t) in f_absent
        out["n"] += n; out["direct"] += d; out["mirror"] += m
        out["cases_with_signal"] += bool(n)
    return out


# ── Loading ────────────────────────────────────────────────────────────────

_READER_RE = re.compile(r"^([A-Za-z]\d+)(?:_(\d+))?_report_facts\.json$")


def load_split(gt_dir: Path, facts_dir: Path) -> Tuple[Dict, Dict, List[str], Dict]:
    consensus: Dict[str, Dict] = {}
    per_reader: Dict[str, List[Tuple[str, Dict]]] = defaultdict(list)
    problems: List[str] = []

    for p in sorted(gt_dir.glob("*_report_facts.json")):
        m = _READER_RE.match(p.name)
        if not m:
            problems.append(f"unparseable report_facts filename: {p.name}")
            continue
        case, reader = m.group(1), m.group(2)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # a truncated stage-1 write is a data fact
            problems.append(f"{p.name}: unreadable ({exc})")
            continue
        if reader is None:
            consensus[case] = data
        else:
            per_reader[case].append((reader, data))

    facts: Dict[str, Dict] = {}
    for p in sorted(facts_dir.glob("*.json")):
        try:
            facts[p.stem] = json.loads(p.read_text(encoding="utf-8")).get("structured") or {}
        except Exception as exc:
            problems.append(f"{p.name}: unreadable ({exc})")

    for case in sorted(per_reader):
        per_reader[case].sort(key=lambda t: int(t[0]))
    return consensus, dict(per_reader), problems, facts


# ── Aggregation + report ───────────────────────────────────────────────────

def prefix_of(case: str) -> str:
    return case[0]


def survey(gt_dir: Path, facts_dir: Path, reports_dir: Optional[Path]) -> Dict[str, Any]:
    consensus, per_reader, problems, facts = load_split(gt_dir, facts_dir)

    cases = sorted(set(facts) | set(consensus))
    rows: List[Dict[str, Any]] = []

    pooled_reader_table: Counter = Counter()
    pooled_reader_sets: Dict[str, List[int]] = defaultdict(lambda: [0, 0])   # [inter, union]
    pooled_reader_scalars: Dict[str, List[int]] = defaultdict(lambda: [0, 0])  # [agree, n]

    FRAMINGS = ("asis", "mirror", "corrected")
    pooled_rf_table = {f: Counter() for f in FRAMINGS}
    pooled_rf_sets = {f: {c: [0, 0, 0] for c in ("crowns", "implants", "ian_close")}
                      for f in FRAMINGS}                                        # tp, fp, fn
    pooled_bridge = {f: Counter() for f in FRAMINGS}
    pooled_mirror = {"n": 0, "direct": 0, "mirror": 0,
                     "by_channel": defaultdict(lambda: {"n": 0, "direct": 0, "mirror": 0})}
    fov_maxilla_tab: Counter = Counter()
    fov_condyle_tab: Counter = Counter()
    verdicts: Counter = Counter()
    ambiguous_lean: Counter = Counter()
    self_conflicts = 0

    for case in cases:
        if case not in facts:
            problems.append(f"{case}: report_facts without a facts file")
            continue
        if case not in consensus:
            problems.append(f"{case}: facts file without report_facts")
            continue

        rep = flatten_report_facts(consensus[case])
        ff = flatten_facts(facts[case])
        ff_m = mirror_facts(ff)
        self_conflicts += len(rep["self_conflict"])

        asis = compare_report_facts(rep, ff)
        mirr = compare_report_facts(rep, ff_m)
        mev = mirror_evidence(rep, ff)
        verdicts[mev["verdict"]] += 1
        if mev["verdict"] == "ambiguous":
            ambiguous_lean["leans mirrored" if mev["mirror"] > mev["direct"]
                           else "leans aligned" if mev["direct"] > mev["mirror"]
                           else "dead even"] += 1
        # `corrected` is the conservative repair: swap the quadrants only for
        # the cases the evidence actually convicts, leave `ambiguous` and
        # `no-signal` alone. It is deliberately NOT "mirror everything" --
        # that framing is already the `mirror` column, and the gap between the
        # two is how much of the mirror problem is confidently localisable.
        corr = mirr if mev["verdict"] == "mirrored" else asis

        # inter-reader
        ir = None
        if len(per_reader.get(case, [])) >= 2:
            raw = [d for _, d in per_reader[case]]
            flat = [(r, flatten_report_facts(d)) for r, d in per_reader[case]]
            ir = interreader_case(flat, raw)
            # Micro pooling: sum the intersections and unions themselves
            # rather than averaging per-pair Jaccards, so a case with many
            # crowned teeth weighs as much as it costs.
            for i in range(len(flat)):
                for j in range(i + 1, len(flat)):
                    for k in SET_CHANNELS:
                        sa, sb = flat[i][1][k], flat[j][1][k]
                        if sa or sb:
                            pooled_reader_sets[k][0] += len(sa & sb)
                            pooled_reader_sets[k][1] += len(sa | sb)
            for p in ir["_pairs"]:
                pooled_reader_table.update(p["presence_table"])
                for label, v in p["scalars"].items():
                    if v is None:
                        continue
                    pooled_reader_scalars[label][0] += int(v)
                    pooled_reader_scalars[label][1] += 1
                pooled_reader_scalars["bridge_present"][0] += int(p["bridge_present"])
                pooled_reader_scalars["bridge_present"][1] += 1
            ir.pop("_pairs")

        for framing, res in (("asis", asis), ("mirror", mirr), ("corrected", corr)):
            pooled_rf_table[framing].update(res["presence_table"])
            for c in ("crowns", "implants", "ian_close"):
                if res[c]:
                    pooled_rf_sets[framing][c][0] += res[c]["tp"]
                    pooled_rf_sets[framing][c][1] += res[c]["fp"]
                    pooled_rf_sets[framing][c][2] += res[c]["fn"]
            pooled_bridge[framing][(res["bridge_report"], res["bridge_facts"])] += 1

        pooled_mirror["n"] += mev["n"]
        pooled_mirror["direct"] += mev["direct"]
        pooled_mirror["mirror"] += mev["mirror"]
        for k, v in mev["by_channel"].items():
            for f in ("n", "direct", "mirror"):
                pooled_mirror["by_channel"][k][f] += v[f]

        fov_maxilla_tab[(ff["fov_maxilla"], (consensus[case].get("maxilla") or {}).get("maxilla_scope"))] += 1
        mand = consensus[case].get("mandible") or {}
        fov_condyle_tab[(ff["fov_condyles"], mand.get("condyle_right_scope"), mand.get("condyle_left_scope"))] += 1

        rows.append({
            "case_id": case,
            "prefix": prefix_of(case),
            "n_reports": len(per_reader.get(case, [])) or 1,
            "presence_stated": len(rep["stated"]),
            "presence_scored": asis["presence_n"],
            "presence_facts_silent": asis["presence_n_facts_silent"],
            "facts_positions_stated": len(ff["stated"]),
            "presence_enumerated": consensus[case].get("mandible", {}).get("presence_enumerated"),
            "self_conflicts": len(rep["self_conflict"]),
            "score_asis": asis["score"],
            "score_mirrored": mirr["score"],
            "score_corrected": corr["score"],
            "score_best": max([s for s in (asis["score"], mirr["score"]) if s is not None],
                              default=None),
            "n_evidence": asis["n_evidence"],
            "presence_acc_asis": asis["presence_acc"],
            "presence_acc_mirrored": mirr["presence_acc"],
            "presence_acc_corrected": corr["presence_acc"],
            "crowns_f1_corrected": corr["crowns"]["f1"] if corr["crowns"] else None,
            "crowns_f1_asis": asis["crowns"]["f1"] if asis["crowns"] else None,
            "crowns_f1_mirrored": mirr["crowns"]["f1"] if mirr["crowns"] else None,
            "implants_f1_asis": asis["implants"]["f1"] if asis["implants"] else None,
            "implants_f1_mirrored": mirr["implants"]["f1"] if mirr["implants"] else None,
            "ian_f1_asis": asis["ian_close"]["f1"] if asis["ian_close"] else None,
            "bridge_match_asis": asis["bridge_match"],
            "mirror_verdict": mev["verdict"],
            "mirror_n_informative": mev["n"],
            "mirror_direct": mev["direct"],
            "mirror_mirrored": mev["mirror"],
            "interreader_score": ir["score"] if ir else None,
            "interreader_presence_acc": ir["presence_acc"] if ir else None,
            "interreader_contradictions": ir["presence_contradictions"] if ir else None,
            "interreader_coverage_jaccard": ir["coverage_jaccard"] if ir else None,
            "interreader_set_agreement": ir["set_agreement"] if ir else None,
            "interreader_scalar_agreement": ir["scalar_agreement"] if ir else None,
        })

    for r in rows:
        s = r["score_asis"]
        r["tier"] = next(t for t, lo in TIERS if s is not None and s >= lo) if s is not None else "?"

    ranked = sorted(rows, key=lambda r: (-(r["score_asis"] if r["score_asis"] is not None else -1),
                                         -r["n_evidence"], r["case_id"]))
    for i, r in enumerate(ranked, 1):
        r["rank"] = i

    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gt_dir": str(gt_dir), "facts_dir": str(facts_dir),
        "n_cases": len(rows),
        "n_multi_report": sum(1 for r in rows if r["n_reports"] >= 2),
        "problems": problems,
        "self_conflicts": self_conflicts,
        "pooled": {
            "reader_presence_table": {f"{a}|{b}": v for (a, b), v in pooled_reader_table.items()},
            "reader_presence_kappa": cohen_kappa(pooled_reader_table),
            "reader_sets": {k: {"inter": v[0], "union": v[1],
                                "jaccard": (v[0] / v[1]) if v[1] else None}
                            for k, v in sorted(pooled_reader_sets.items())},
            "reader_scalars": {k: {"agree": v[0], "n": v[1],
                                   "rate": (v[0] / v[1]) if v[1] else None}
                               for k, v in sorted(pooled_reader_scalars.items())},
            "rf_presence_table": {f: {f"{a}|{b}": v for (a, b), v in t.items()}
                                  for f, t in pooled_rf_table.items()},
            "rf_sets": {f: {c: {"tp": v[0], "fp": v[1], "fn": v[2],
                                "precision": v[0] / (v[0] + v[1]) if (v[0] + v[1]) else None,
                                "recall": v[0] / (v[0] + v[2]) if (v[0] + v[2]) else None,
                                "f1": 2 * v[0] / (2 * v[0] + v[1] + v[2]) if v[0] else 0.0}
                            for c, v in cs.items()}
                        for f, cs in pooled_rf_sets.items()},
            "rf_bridge": {f: {f"report={a}|facts={b}": v for (a, b), v in t.items()}
                          for f, t in pooled_bridge.items()},
            "mirror": {"n": pooled_mirror["n"], "direct": pooled_mirror["direct"],
                       "mirror": pooled_mirror["mirror"],
                       "by_channel": dict(pooled_mirror["by_channel"])},
            "fov_maxilla": {f"facts={a}|report={b}": v for (a, b), v in fov_maxilla_tab.items()},
            "fov_condyles": {f"facts={a}|report_r={b}|report_l={c}": v
                             for (a, b, c), v in fov_condyle_tab.items()},
            "mirror_verdicts": dict(verdicts),
            "mirror_ambiguous_lean": dict(ambiguous_lean),
            "mirror_regex_control": regex_mirror_control(reports_dir, facts) if reports_dir else {},
        },
        "cases": ranked,
    }


# ── Text rendering ─────────────────────────────────────────────────────────

def render(res: Dict[str, Any], top: int, bottom: int) -> str:
    L: List[str] = []
    A = L.append
    p = res["pooled"]
    rows = res["cases"]

    A("=" * 78)
    A("GROUND-TRUTH QUALITY SURVEY")
    A(f"  generated : {res['generated']}")
    A(f"  reports   : {res['gt_dir']}")
    A(f"  facts     : {res['facts_dir']}")
    A(f"  cases     : {res['n_cases']}   multi-report: {res['n_multi_report']}")
    A("=" * 78)

    # ── 1 ──────────────────────────────────────────────────────────────
    A("")
    A("1. INTER-READER AGREEMENT  (cases with >= 2 reports, pooled over reader pairs)")
    A("-" * 78)
    multi = [r for r in rows if r["n_reports"] >= 2]
    if not multi:
        A("  no multi-report case in this split.")
    else:
        t = p["reader_presence_table"]
        n = sum(t.values())
        agree = t.get("present|present", 0) + t.get("absent|absent", 0)
        A(f"  presence, positions BOTH readers state : n={n}")
        A(f"    raw agreement                        : {pct(agree / n if n else None)}")
        A(f"    Cohen's kappa                        : {num(p['reader_presence_kappa'])}")
        A(f"    present|present {t.get('present|present',0):5d}   present|absent {t.get('present|absent',0):5d}")
        A(f"    absent |present {t.get('absent|present',0):5d}   absent |absent {t.get('absent|absent',0):5d}")
        A(f"    contradictions (one says present, the other absent): "
          f"{t.get('present|absent',0)+t.get('absent|present',0)}")
        A(f"  coverage overlap (Jaccard of the position sets each reader states),")
        A(f"    mean over cases                      : "
          f"{pct(mean([r['interreader_coverage_jaccard'] for r in multi]))}")
        A("")
        A("  set findings, micro-Jaccard over all reader pairs:")
        A(f"    {'channel':<22} {'inter':>7} {'union':>7} {'jaccard':>9}")
        for k, v in p["reader_sets"].items():
            A(f"    {k:<22} {v['inter']:7d} {v['union']:7d} {pct(v['jaccard']):>9}")
        A("    A low Jaccard here is two readers describing different teeth, which")
        A("    is not the same claim as two readers contradicting each other -- a")
        A("    report states what its author found notable and omits the rest.")
        A("    crowns / post_and_core / caries are the channels where that omission")
        A("    bites hardest, and where the vocabulary is most indirect, so their")
        A("    numbers bound reader+extraction noise together, not reader noise alone.")
        A("")
        A("  scalar findings, exact match where both readers are non-null:")
        A(f"    {'channel':<28} {'agree':>6} {'n':>6} {'rate':>8}")
        for k, v in p["reader_scalars"].items():
            A(f"    {k:<28} {v['agree']:6d} {v['n']:6d} {pct(v['rate']):>8}")
        A("")
        A(f"  per-case inter-reader score: mean {pct(mean([r['interreader_score'] for r in multi]))}"
          f"   min {pct(min([r['interreader_score'] for r in multi if r['interreader_score'] is not None], default=None))}")
        worst = sorted([r for r in multi if r["interreader_score"] is not None],
                       key=lambda r: r["interreader_score"])[:12]
        A("  least-consistent cases (these are the ones whose label is least trustworthy):")
        A(f"    {'case':<8} {'rdrs':>4} {'score':>7} {'presence':>9} {'contra':>7} {'sets':>7} {'scalars':>8}")
        for r in worst:
            A(f"    {r['case_id']:<8} {r['n_reports']:4d} {pct(r['interreader_score']):>7} "
              f"{pct(r['interreader_presence_acc']):>9} {r['interreader_contradictions']:7d} "
              f"{pct(r['interreader_set_agreement']):>7} {pct(r['interreader_scalar_agreement']):>8}")

    # ── 2 ──────────────────────────────────────────────────────────────
    A("")
    A("2. REPORT vs FACTS  (report is the label; facts scored as given AND mirrored)")
    A("-" * 78)
    half = [r for r in rows if r["facts_positions_stated"] < 32]
    silent = sum(r["presence_facts_silent"] for r in rows)
    if half or silent:
        A(f"  facts state fewer than 32 positions in {len(half)} of {len(rows)} case(s)")
        A(f"  -- fov.maxilla 'excluded' means the scan never covered those teeth, so")
        A(f"  they are not claims the facts got wrong. {silent} report-stated")
        A(f"  position(s) are excluded from the presence comparison for that reason,")
        A(f"  and the positive channels below are filtered to covered positions too.")
        A("")
    TAGS = {"asis": "facts as given",
            "mirror": "facts MIRRORED (every case)",
            "corrected": "facts mirror-CORRECTED (convicted cases only)"}
    for framing in ("asis", "mirror", "corrected"):
        t = p["rf_presence_table"][framing]
        n = sum(t.values())
        acc = (t.get("present|present", 0) + t.get("absent|absent", 0)) / n if n else None
        A(f"  presence over report-stated positions, {TAGS[framing]}: n={n}  accuracy {pct(acc)}")
        A(f"    report present & facts present {t.get('present|present',0):6d}"
          f"   report present & facts absent {t.get('present|absent',0):6d}")
        A(f"    report absent  & facts present {t.get('absent|present',0):6d}"
          f"   report absent  & facts absent {t.get('absent|absent',0):6d}")
    A("")
    A("  set channels, micro over all cases (facts = prediction, report = label):")
    A(f"    {'channel':<12} {'framing':<10} {'tp':>6} {'fp':>6} {'fn':>6} {'prec':>8} {'rec':>8} {'F1':>8}")
    for c in ("crowns", "implants", "ian_close"):
        for framing in ("asis", "mirror", "corrected"):
            v = p["rf_sets"][framing][c]
            A(f"    {c:<12} {framing:<10} {v['tp']:6d} {v['fp']:6d} {v['fn']:6d} "
              f"{pct(v['precision']):>8} {pct(v['recall']):>8} {pct(v['f1']):>8}")
    A("    crowns is the weakest channel on BOTH sides and the F1 is not a clean")
    A("    read on the facts: the report's crown list carries pontics and")
    A("    implant-borne crowns that no tooth position owns, and per _definitions")
    A("    a posted tooth is filed under post_and_core rather than crown. Treat it")
    A("    as an upper bound on the disagreement, not a measured error rate.")
    A("")
    A("  bridge_present (unaffected by mirroring; shown once):")
    for k, v in sorted(p["rf_bridge"]["asis"].items()):
        A(f"    {k:<32} {v:5d}")
    A("")
    A("  fov.maxilla (facts) vs maxilla_scope (report) -- cross-tab, NOT scored:")
    A("    the two vocabularies are not the same enum, and facts uses null for")
    A("    'not flagged partial' rather than for 'unknown', so a match rate here")
    A("    would be an artefact of the mapping chosen, not a measurement.")
    for k, v in sorted(p["fov_maxilla"].items(), key=lambda kv: -kv[1]):
        A(f"    {k:<52} {v:5d}")
    A("")
    A("  fov.condyles (facts) vs condyle scopes (report) -- cross-tab, NOT scored:")
    for k, v in sorted(p["fov_condyles"].items(), key=lambda kv: -kv[1])[:10]:
        A(f"    {k:<52} {v:5d}")

    # ── 3 ──────────────────────────────────────────────────────────────
    A("")
    A("3. MIRROR ERROR  (informative items only: facts(pos) != facts(mirror(pos)))")
    A("-" * 78)
    m = p["mirror"]
    A(f"  {'channel':<12} {'informative':>12} {'facts agree':>12} {'facts agree':>13}")
    A(f"  {'':<12} {'items':>12} {'as given':>12} {'when mirrored':>13}")
    for k in ("presence", "crowns", "implants"):
        v = m["by_channel"].get(k, {"n": 0, "direct": 0, "mirror": 0})
        A(f"  {k:<12} {v['n']:12d} {pct(v['direct']/v['n'] if v['n'] else None):>12} "
          f"{pct(v['mirror']/v['n'] if v['n'] else None):>13}")
    A(f"  {'ALL':<12} {m['n']:12d} {pct(m['direct']/m['n'] if m['n'] else None):>12} "
      f"{pct(m['mirror']/m['n'] if m['n'] else None):>13}")
    A("")
    A("  per-case verdict:")
    tot = sum(p["mirror_verdicts"].values()) or 1
    for k in ("mirrored", "aligned", "ambiguous", "no-signal"):
        v = p["mirror_verdicts"].get(k, 0)
        A(f"    {k:<12} {v:5d}  {v/tot:6.1%}")
    A("    convicted = >= 2 informative items and >= 70% of them breaking the")
    A("                same way; no-signal = left-right symmetric evidence, so")
    A("                the two framings are indistinguishable for that case,")
    A("                which is not the same as 'aligned'.")
    for k, v in sorted(p.get("mirror_ambiguous_lean", {}).items()):
        A(f"      of the ambiguous: {k:<16} {v:5d}")
    A("")
    rc = p.get("mirror_regex_control") or {}
    if rc.get("n"):
        A("  CONTROL, no LLM in the loop: absence sentences regexed straight off")
        A("  the report text and compared to facts.teeth_absent, same informative-")
        A("  item rule. Shares no code and no model with the extraction above.")
        A(f"    reports read {rc['reports_read']}   cases with signal {rc['cases_with_signal']}"
          f"   informative items {rc['n']}")
        A(f"    facts agree as given {pct(rc['direct']/rc['n'])}"
          f"    when mirrored {pct(rc['mirror']/rc['n'])}")
    elif rc.get("skipped"):
        A(f"  CONTROL skipped: {rc['skipped']}")
    A("")
    A("  by sub-dataset prefix:")
    A(f"    {'prefix':<8} {'cases':>6} {'mirrored':>9} {'aligned':>8} {'ambig':>7} {'no-sig':>7}")
    byp = defaultdict(Counter)
    for r in rows:
        byp[r["prefix"]][r["mirror_verdict"]] += 1
        byp[r["prefix"]]["_n"] += 1
    for pre in sorted(byp):
        c = byp[pre]
        A(f"    {pre:<8} {c['_n']:6d} {c['mirrored']:9d} {c['aligned']:8d} "
          f"{c['ambiguous']:7d} {c['no-signal']:7d}")

    # ── 4 ──────────────────────────────────────────────────────────────
    A("")
    A("4. PER-CASE SCORE AND RANKING  (how correct the facts file is, report = label)")
    A("-" * 78)
    scored = [r for r in rows if r["score_asis"] is not None]
    A(f"  scored cases: {len(scored)} / {len(rows)}")
    A(f"  score (facts as given)      : mean {pct(mean([r['score_asis'] for r in scored]))}"
      f"   median {pct(sorted(r['score_asis'] for r in scored)[len(scored)//2] if scored else None)}")
    A(f"  score (facts mirrored)      : mean {pct(mean([r['score_mirrored'] for r in scored]))}")
    A(f"  score (mirror-corrected)    : mean {pct(mean([r['score_corrected'] for r in scored]))}"
      f"   -- ceiling reachable by fixing only the convicted cases")
    A("")
    A("  tier distribution (on the as-given score):")
    tt = Counter(r["tier"] for r in rows)
    for t, lo in TIERS:
        A(f"    {t}  (>= {lo:.2f})  {tt.get(t,0):5d}")
    A("")
    A("  by sub-dataset prefix:")
    A(f"    {'prefix':<8} {'cases':>6} {'as given':>10} {'mirrored':>10} {'corrected':>10} {'presence':>10}")
    bys = defaultdict(list)
    for r in scored:
        bys[r["prefix"]].append(r)
    for pre in sorted(bys):
        g = bys[pre]
        A(f"    {pre:<8} {len(g):6d} {pct(mean([x['score_asis'] for x in g])):>10} "
          f"{pct(mean([x['score_mirrored'] for x in g])):>10} "
          f"{pct(mean([x['score_corrected'] for x in g])):>10} "
          f"{pct(mean([x['presence_acc_asis'] for x in g])):>10}")
    A("")
    # The reason this survey exists on the few-shot branch: which cases are
    # safe to hand a model as an exemplar. Three independent gates, because a
    # case can fail any one of them for a different reason.
    clean = [r for r in rows
             if r["score_asis"] is not None and r["score_asis"] >= 0.90
             and r["n_evidence"] >= 10
             and r["mirror_verdict"] in ("aligned", "no-signal")
             and (r["interreader_score"] is None or r["interreader_score"] >= 0.80)]
    A(f"  CLEAN SET: {len(clean)} case(s) pass all three gates -- score >= 0.90,")
    A(f"    >= 10 comparable items, no mirror conviction, and (where a second")
    A(f"    reader exists) inter-reader agreement >= 0.80. These are the cases")
    A(f"    whose facts sidecar can be trusted as written:")
    for i in range(0, min(len(clean), 60), 12):
        A("      " + " ".join(f"{r['case_id']:<7}" for r in clean[i:i + 12]))
    if len(clean) > 60:
        A(f"      ... and {len(clean)-60} more (full list in the CSV: filter tier==A)")
    A("")
    hdr = (f"    {'rank':>4} {'case':<8} {'tier':<4} {'asis':>7} {'mirr':>7} {'corr':>7} "
           f"{'pres':>7} {'crown':>7} {'impl':>7} {'ev':>4} {'rdr':>3} {'IRA':>7} {'mirror?':<10}")

    def line(r):
        return (f"    {r['rank']:4d} {r['case_id']:<8} {r['tier']:<4} {pct(r['score_asis']):>7} "
                f"{pct(r['score_mirrored']):>7} {pct(r['score_corrected']):>7} "
                f"{pct(r['presence_acc_asis']):>7} "
                f"{pct(r['crowns_f1_asis']):>7} {pct(r['implants_f1_asis']):>7} "
                f"{r['n_evidence']:4d} {r['n_reports']:3d} {pct(r['interreader_score']):>7} "
                f"{r['mirror_verdict']:<10}")

    A(f"  TOP {top} (facts most consistent with the report):")
    A(hdr)
    for r in rows[:top]:
        A(line(r))
    A("")
    A(f"  BOTTOM {bottom} (facts least consistent with the report -- inspect before use):")
    A(hdr)
    for r in rows[-bottom:]:
        A(line(r))

    if res["problems"]:
        A("")
        A("PROBLEMS")
        A("-" * 78)
        for x in res["problems"][:40]:
            A(f"  {x}")
        if len(res["problems"]) > 40:
            A(f"  ... and {len(res['problems'])-40} more")
    if res["self_conflicts"]:
        A("")
        A(f"  stage-1 self-contradictions (one reader listing a position as BOTH "
          f"present and absent): {res['self_conflicts']} position(s), excluded from "
          f"every presence figure above.")
    return "\n".join(L)


# ── Diff against the previous survey ───────────────────────────────────────

_HEADLINE = [
    ("reader kappa", lambda r: r["pooled"]["reader_presence_kappa"]),
    ("presence acc (as given)", lambda r: _acc(r["pooled"]["rf_presence_table"]["asis"])),
    ("presence acc (mirrored)", lambda r: _acc(r["pooled"]["rf_presence_table"]["mirror"])),
    ("crowns F1 (as given)", lambda r: r["pooled"]["rf_sets"]["asis"]["crowns"]["f1"]),
    ("mirror rate", lambda r: (r["pooled"]["mirror"]["mirror"] / r["pooled"]["mirror"]["n"])
     if r["pooled"]["mirror"]["n"] else None),
    ("mean score (as given)", lambda r: mean([c["score_asis"] for c in r["cases"]])),
    ("mean score (mirrored)", lambda r: mean([c["score_mirrored"] for c in r["cases"]])),
    ("mean score (corrected)", lambda r: mean([c.get("score_corrected") for c in r["cases"]])),
]


def _acc(table: Dict[str, int]) -> Optional[float]:
    n = sum(table.values())
    return (table.get("present|present", 0) + table.get("absent|absent", 0)) / n if n else None


def render_diff(cur: Dict, prev: Dict, prev_name: str) -> str:
    L = ["", f"DIFF vs {prev_name}", "-" * 78,
         f"    {'metric':<28} {'previous':>10} {'now':>10} {'delta':>10}"]
    for label, fn in _HEADLINE:
        try:
            a, b = fn(prev), fn(cur)
        except Exception:
            continue
        d = None if (a is None or b is None) else b - a
        L.append(f"    {label:<28} {pct(a):>10} {pct(b):>10} "
                 f"{('    n/a' if d is None else f'{d:+9.1%}'):>10}")
    pa = {c["case_id"]: c for c in prev["cases"]}
    moved = [(c["case_id"], pa[c["case_id"]]["mirror_verdict"], c["mirror_verdict"])
             for c in cur["cases"]
             if c["case_id"] in pa and pa[c["case_id"]]["mirror_verdict"] != c["mirror_verdict"]]
    if moved:
        L.append(f"    mirror verdict changed for {len(moved)} case(s): "
                 + ", ".join(f"{c}({a}->{b})" for c, a, b in moved[:12])
                 + (" ..." if len(moved) > 12 else ""))
    return "\n".join(L)


# ── CLI ────────────────────────────────────────────────────────────────────

CSV_FIELDS = ["rank", "case_id", "prefix", "tier", "score_asis", "score_mirrored",
              "score_corrected", "score_best", "n_evidence", "n_reports",
              "presence_stated",
              "presence_acc_asis", "presence_acc_mirrored", "presence_acc_corrected",
              "crowns_f1_asis", "crowns_f1_corrected",
              "crowns_f1_mirrored", "implants_f1_asis", "implants_f1_mirrored",
              "ian_f1_asis", "bridge_match_asis", "mirror_verdict",
              "mirror_n_informative", "mirror_direct", "mirror_mirrored",
              "interreader_score", "interreader_presence_acc",
              "interreader_contradictions", "interreader_coverage_jaccard",
              "interreader_set_agreement", "interreader_scalar_agreement",
              "self_conflicts"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="training")
    ap.add_argument("--project-dir", default=".")
    ap.add_argument("--gt-dir", default=None,
                    help="default dataset/<split>/outputs/ground_truth")
    ap.add_argument("--facts-dir", default=None, help="default dataset/<split>/facts")
    ap.add_argument("--out-dir", default=None,
                    help="default outputs/gt_quality_<split>; timestamped, never overwritten")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--bottom", type=int, default=25)
    ap.add_argument("--no-diff", action="store_true")
    ap.add_argument("--stdout-only", action="store_true")
    args = ap.parse_args()

    root = Path(args.project_dir).resolve()
    gt_dir = Path(args.gt_dir) if args.gt_dir else root / "dataset" / args.split / "outputs" / "ground_truth"
    facts_dir = Path(args.facts_dir) if args.facts_dir else root / "dataset" / args.split / "facts"
    for d in (gt_dir, facts_dir):
        if not d.is_dir():
            print(f"[FAIL] not a directory: {d}", file=sys.stderr)
            return 2

    res = survey(gt_dir, facts_dir, root / "dataset" / args.split / "reports")
    text = render(res, args.top, args.bottom)

    out_dir = Path(args.out_dir) if args.out_dir else root / "outputs" / f"gt_quality_{args.split}"
    prev_path = None
    if not args.no_diff and out_dir.is_dir():
        prevs = sorted(out_dir.glob("gt_quality_*.json"))
        if prevs:
            prev_path = prevs[-1]
            try:
                text += "\n" + render_diff(res, json.loads(prev_path.read_text(encoding="utf-8")),
                                           prev_path.name)
            except Exception as exc:
                text += f"\n[WARN] could not diff against {prev_path.name}: {exc}"

    print(text)
    if args.stdout_only:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (out_dir / f"gt_quality_{stamp}.txt").write_text(text + "\n", encoding="utf-8")
    (out_dir / f"gt_quality_{stamp}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    csv_path = out_dir / f"case_ranking_{stamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in res["cases"]:
            w.writerow(r)
    print(f"\n[OK] wrote {out_dir}/gt_quality_{stamp}.{{txt,json}} and {csv_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
