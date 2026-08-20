#!/usr/bin/env python3
"""
audit_report_facts.py -- can the report TEXT support what stage 1 extracted?

A095 is why this exists. Its report says "Mucosal hyperplasia of the floor of
the right maxillary sinus, bilaterally, greater on the left, with likely
endodontic material on the sinus floor corresponding to 25", and stage 1 filed
tooth 25 -- an upper LEFT tooth -- under sinus_RIGHT. Nothing downstream could
see that: the value is a well-formed list of ints in a field that takes a list
of ints, and stage 2 expands it faithfully into a ground truth that says the
wrong side. It surfaced only because a schema change happened to make that one
field visible, which is not a way to find the other ones.

So this is the double-check for stage 1, and every screen in it is MECHANICAL
-- decidable from the FDI numbering and the report text alone, no model, no
GPU, no judgement. A screen that needed judgement would just be a second
opinion of the same kind as the first, and two LLM reads that agree are not
evidence; they are the same failure twice.

  --screen laterality   an FDI in a side- or arch-scoped field that belongs to
                        the other side or the other arch. FDI numbering is
                        positional -- 1x/2x upper right/left, 4x/3x lower
                        right/left -- so "tooth 25 in the right sinus" is not
                        a matter of opinion. This is the A095 class.
  --screen not-in-text  an FDI asserted in report_facts that never appears in
                        the report at all, in any notation the corpus uses
                        ("25", "2.5", "e.d. 25"). The extractor invented a
                        tooth number, or copied one from the other arch's
                        paragraph.
  --screen contradiction  the same FDI in both teeth_present and teeth_absent,
                        or a finding hung on a tooth the same block calls
                        absent. The second one has a legitimate case -- a
                        pontic's crown, a remnant at an extracted site -- so it
                        reports as a note, not an error.

Usage:
    python3 code/ground_truth/audit_report_facts.py --split training
    python3 code/ground_truth/audit_report_facts.py --split training --screen laterality
    python3 code/ground_truth/audit_report_facts.py --split training --json audit.json
    python3 code/ground_truth/audit_report_facts.py --split training --case-ids A095 A003

Exit code is 1 when any ERROR-level finding survives, so this can gate a
rebuild: stage 2 is deterministic, and replaying it over bad stage-1 output
just produces a confidently wrong ground truth faster.
"""

import argparse
import json
import re
import sys
from collections import Counter
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

# FDI is positional, and that is the whole basis of the laterality screen.
UPPER_RIGHT = set(range(11, 19))
UPPER_LEFT = set(range(21, 29))
LOWER_LEFT = set(range(31, 39))
LOWER_RIGHT = set(range(41, 49))
PRIMARY = set(range(51, 56)) | set(range(61, 66)) | set(range(71, 76)) | set(range(81, 86))

ARCH_FDIS = {"maxilla": UPPER_RIGHT | UPPER_LEFT,
             "mandible": LOWER_LEFT | LOWER_RIGHT}
ARCH_PRIMARY = {"maxilla": set(range(51, 56)) | set(range(61, 66)),
                "mandible": set(range(71, 76)) | set(range(81, 86))}

# field -> the FDIs it may legally contain, per arch block. Anything not listed
# is checked against the arch as a whole.
SIDE_SCOPED = {
    "maxilla": {"sinus_right": UPPER_RIGHT, "sinus_left": UPPER_LEFT},
    "mandible": {"canal_right": LOWER_RIGHT, "canal_left": LOWER_LEFT},
}
SIDE_FIELD_LIST = {"sinus_right": "intrasinusal_teeth",
                   "sinus_left": "intrasinusal_teeth",
                   "canal_right": "adjacent_teeth",
                   "canal_left": "adjacent_teeth"}

# Flat int lists that must stay inside their own arch.
FLAT_LISTS = ("teeth_present", "teeth_absent", "primary_teeth", "post_and_core",
              "crowns", "fillings", "caries", "root_remnants", "root_fractures",
              "periapical_lesions", "uncertain_teeth")
# Lists of objects, and the key holding the FDI.
OBJ_LISTS = (("endodontic", "fdi"), ("unerupted", "fdi"), ("implants", "fdi"))

FINDING_FIELDS = ("post_and_core", "crowns", "fillings", "caries",
                  "root_fractures", "periapical_lesions")

# Fields a report routinely fills without writing a number: presence stated
# wholesale ("presence of all dental elements"), a pontic implied by the span
# either side of it, a third molar named as "the maxillary third molars", a
# tooth the extractor was unsure of. not-in-text is a NOTE for these and an
# ERROR everywhere else.
DERIVED_FIELDS = {"teeth_present", "teeth_absent", "primary_teeth",
                  "unerupted", "uncertain_teeth", "bridges"}


# The arch sweep, in the order a report walks it, so "from 13 to 15" and
# "from 18 to 28" both expand to the teeth actually between the endpoints.
ARCH_SWEEP = {
    "upper": [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28],
    "lower": [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38],
}
_RANGE_RE = re.compile(r"\b([1-4][1-8])\s*(?:to|-|–|—|until|through)\s*([1-4][1-8])\b")


def split_sentences(text: str):
    """Sentence split that does not cut a tooth number off its abbreviation.

    These reports write "e.d. 48" (elemento dentale) and "4.8", and a naive
    split on ". " strands the number: "in proximity to e.d. 48" becomes a
    sentence ending at "e.d." plus one starting "48", and every check that
    asks "does this sentence name tooth 48" then answers no. survey_findings
    guards the same thing for the same reason.
    """
    guarded = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", text or "")
    guarded = re.sub(r"\b([a-zA-Z])\.(?=[a-zA-Z]\.|\s*\d)", r"\1<DOT>", guarded)
    return [s.replace("<DOT>", ".").strip()
            for s in re.split(r"(?<=[.;:])\s+", guarded) if s.strip()]


# ── Decisions a reader has already made ────────────────────────────────────
#
# A screen cannot see that a positional phrase names a tooth, so it re-raises
# the same finding after every rebuild and the reviewer answers it again. This
# is where an answer is kept, in the project's own style: hand-checked, in
# code, with the sentence that settles it beside the entry -- the same reason
# Kept here rather than in a data file, on the reasoning the deleted
# survey_findings.py used for its own tables.
#
# Keyed (stage-1 file stem, arch, field, FDI). Entries downgrade the finding
# from ERROR to NOTE: the audit still reports what stage 1 said, it just stops
# asking. A drop is NOT recorded here -- that is applied to report_facts by
# apply_triage_decisions.py and the finding then no longer exists.
ACKNOWLEDGED = {
    ("A022", "maxilla", "root_remnants", 23):
        'reviewed 2026-08-13: "a root remnant in the canine region of the 2nd '
        'quadrant" -- quadrant 2 has one canine, so the phrase names 23',
    ("F003_1", "maxilla", "endodontic", 16):
        'reviewed 2026-08-13: "The distal teeth of the 1st quadrant have been '
        'endodontically treated" covers 16, 17',
    ("F003_1", "maxilla", "endodontic", 17):
        'reviewed 2026-08-13: "The distal teeth of the 1st quadrant have been '
        'endodontically treated" covers 16, 17',

    # -- training, second batch of 2026-08-13: the arch-range conflicts, where
    # -- the reviewer read the range against what else the report says at that
    # -- position and the range lost, plus the rows read and left undecided.
    # -- Both are recorded for the same reason: a decision that is not written
    # -- down is a decision that gets asked again.
    ("A094", "maxilla", "arch range", 15):
        "reviewed 2026-08-13: the range does not win here -- A094_1.txt: \"15 and 26 are edentulous gaps;\"",
    ("A098", "mandible", "present/absent", 41):
        "reviewed 2026-08-13: read and left undecided -- the text does not settle it either way",
    ("F045", "maxilla", "arch range", 16):
        "reviewed 2026-08-13: the range does not win here -- F045_1.txt: \"Dental implants in positions 16 and 25, which appear radiologically well osseointegrated and\"",
    ("F045", "maxilla", "arch range", 25):
        "reviewed 2026-08-13: the range does not win here -- F045_1.txt: \"Dental implants in positions 16 and 25, which appear radiologically well osseointegrated and\"",
    ("F045_2", "maxilla", "arch range", 16):
        "reviewed 2026-08-13: the range does not win here -- F045_1.txt: \"Dental implants in positions 16 and 25, which appear radiologically well osseointegrated and\"",
    ("F045_2", "maxilla", "arch range", 25):
        "reviewed 2026-08-13: the range does not win here -- F045_1.txt: \"Dental implants in positions 16 and 25, which appear radiologically well osseointegrated and\"",
    ("F048", "maxilla", "arch range", 17):
        "reviewed 2026-08-13: the range does not win here -- F048_2.txt: \"17 is absent;\"",
    ("F048_2", "maxilla", "arch range", 17):
        "reviewed 2026-08-13: the range does not win here -- F048_2.txt: \"17 is absent;\"",
    ("F061", "maxilla", "arch range", 17):
        "reviewed 2026-08-13: the range does not win here -- F061_2.txt: \"Teeth 17 and 28 are missing.\"",
    ("F061_2", "maxilla", "arch range", 17):
        "reviewed 2026-08-13: the range does not win here -- F061_2.txt: \"Teeth 17 and 28 are missing.\"",
    ("P015", "mandible", "arch range", 36):
        "reviewed 2026-08-13: the range does not win here -- P015_2.txt: \"in the III quadrant, teeth from 38 to 31 are present, with tooth 36 missing.\"",
    ("P015_2", "mandible", "arch range", 36):
        "reviewed 2026-08-13: the range does not win here -- P015_2.txt: \"in the III quadrant, teeth from 38 to 31 are present, with tooth 36 missing.\"",
    ("P042_1", "mandible", "root_remnants", 33):
        "reviewed 2026-08-13: read and left undecided -- the text does not settle it either way",
    ("P042_1", "mandible", "root_remnants", 34):
        "reviewed 2026-08-13: read and left undecided -- the text does not settle it either way",
    ("P042_1", "mandible", "root_remnants", 35):
        "reviewed 2026-08-13: read and left undecided -- the text does not settle it either way",
    ("P192", "mandible", "endodontic", 31):
        "reviewed 2026-08-13: read and left undecided -- the text does not settle it either way",
    ("P192", "mandible", "endodontic", 32):
        "reviewed 2026-08-13: read and left undecided -- the text does not settle it either way",
    ("P192", "mandible", "endodontic", 41):
        "reviewed 2026-08-13: read and left undecided -- the text does not settle it either way",
    ("P192", "mandible", "endodontic", 42):
        "reviewed 2026-08-13: read and left undecided -- the text does not settle it either way",

    # -- training, batch of 2026-08-13. Each entry carries the sentence the
    # -- reviewer had in front of them, from whichever reader wrote it; the
    # -- keeps that ADDED a tooth (the arch-range ones) are not here, because
    # -- applying them removed the finding rather than affirming it.
    ("F005_1", "maxilla", "root_remnants", 17):
        "reviewed 2026-08-13 -- F005_1.txt: \"Presence of root remnants in the molar region of the 1st quadrant.\"",
    ("F008", "maxilla", "sinus_left.intrasinusal_teeth", 26):
        "reviewed 2026-08-13 -- F008_2.txt: \"The roots of the first, second and third molars bilaterally protrude into the maxillary sinuses.\"",
    ("F008", "maxilla", "sinus_left.intrasinusal_teeth", 27):
        "reviewed 2026-08-13 -- F008_2.txt: \"The roots of the first, second and third molars bilaterally protrude into the maxillary sinuses.\"",
    ("F008", "maxilla", "sinus_right.intrasinusal_teeth", 16):
        "reviewed 2026-08-13 -- F008_2.txt: \"The roots of the first, second and third molars bilaterally protrude into the maxillary sinuses.\"",
    ("F008", "maxilla", "sinus_right.intrasinusal_teeth", 17):
        "reviewed 2026-08-13 -- F008_2.txt: \"The roots of the first, second and third molars bilaterally protrude into the maxillary sinuses.\"",
    ("F008_2", "maxilla", "sinus_left.intrasinusal_teeth", 26):
        "reviewed 2026-08-13 -- F008_2.txt: \"The roots of the first, second and third molars bilaterally protrude into the maxillary sinuses.\"",
    ("F008_2", "maxilla", "sinus_left.intrasinusal_teeth", 27):
        "reviewed 2026-08-13 -- F008_2.txt: \"The roots of the first, second and third molars bilaterally protrude into the maxillary sinuses.\"",
    ("F008_2", "maxilla", "sinus_right.intrasinusal_teeth", 16):
        "reviewed 2026-08-13 -- F008_2.txt: \"The roots of the first, second and third molars bilaterally protrude into the maxillary sinuses.\"",
    ("F008_2", "maxilla", "sinus_right.intrasinusal_teeth", 17):
        "reviewed 2026-08-13 -- F008_2.txt: \"The roots of the first, second and third molars bilaterally protrude into the maxillary sinuses.\"",
    ("P026_2", "mandible", "endodontic", 46):
        "reviewed 2026-08-13 -- P026_1.txt: \"46 is the only tooth present in the arch, severely periodontally compromised, with endodontic in\"",
    ("P033_2", "mandible", "canal_left.adjacent_teeth", 38):
        "reviewed 2026-08-13 -- P033_1.txt: \"The left mandibular canal is in close relationship with impacted tooth 38.\"",
    ("P033_2", "mandible", "canal_right.adjacent_teeth", 48):
        "reviewed 2026-08-13 -- P033_1.txt: \"The right mandibular canal is in close relationship with impacted tooth 48.\"",
    ("P041_2", "mandible", "canal_left.adjacent_teeth", 38):
        "reviewed 2026-08-13 -- P041_1.txt: \"The left mandibular canal is in close relationship with the tooth germ of impacted tooth 38.\"",
    ("P041_2", "mandible", "canal_right.adjacent_teeth", 48):
        "reviewed 2026-08-13 -- P041_1.txt: \"The right mandibular canal is in close relationship with the tooth germ of impacted tooth 48.\"",
    ("P042_1", "mandible", "endodontic", 33):
        "reviewed 2026-08-13, kept on the reader's own reading",
    ("P042_1", "mandible", "endodontic", 34):
        "reviewed 2026-08-13, kept on the reader's own reading",
    ("P049", "mandible", "caries", 34):
        "reviewed 2026-08-13 -- P049_2.txt: \"the fourth tooth shows destructive caries involving the distal crown surface.\"",
    ("P049", "mandible", "fillings", 45):
        "reviewed 2026-08-13 -- P049_2.txt: \"the mesial surface of the fifth tooth is in contact with the distal surface of the fourth tooth,\"",
    ("P049_2", "mandible", "caries", 34):
        "reviewed 2026-08-13 -- P049_2.txt: \"the fourth tooth shows destructive caries involving the distal crown surface.\"",
    ("P049_2", "mandible", "fillings", 45):
        "reviewed 2026-08-13 -- P049_2.txt: \"the mesial surface of the fifth tooth is in contact with the distal surface of the fourth tooth,\"",
    ("P114_1", "mandible", "canal_left.adjacent_teeth", 38):
        "reviewed 2026-08-13 -- P114_1.txt: \"The mandibular canal has a predominantly lingual course in relation to the third molars.\"",
    ("P217", "mandible", "canal_left.adjacent_teeth", 38):
        "reviewed 2026-08-13 -- P217_1.txt: \"The inferior alveolar canal shows a predominantly buccal course on the left and a lingual course\"",
    ("P217", "mandible", "canal_right.adjacent_teeth", 48):
        "reviewed 2026-08-13 -- P217_1.txt: \"The inferior alveolar canal shows a predominantly buccal course on the left and a lingual course\"",
    ("P232", "mandible", "canal_left.adjacent_teeth", 38):
        "reviewed 2026-08-13 -- P232_1.txt: \"Impacted third molars, mesiolinguoversed, in close relationship to the mandibular canal, but not\"",
    ("P232", "mandible", "canal_right.adjacent_teeth", 48):
        "reviewed 2026-08-13 -- P232_1.txt: \"Impacted third molars, mesiolinguoversed, in close relationship to the mandibular canal, but not\"",
    ("P256_2", "mandible", "canal_right.adjacent_teeth", 48):
        "reviewed 2026-08-13 -- P256_1.txt: \"The right mandibular canal has a regular course, predominantly in an apico-lingual position, and\"",
    ("P258_2", "mandible", "canal_right.adjacent_teeth", 48):
        "reviewed 2026-08-13 -- P258_1.txt: \"Right mandibular canal with a regular course, predominantly in an apico-lingual position and dis\"",
    ("P360_2", "mandible", "canal_left.adjacent_teeth", 38):
        "reviewed 2026-08-13 -- P360_1.txt: \"The left mandibular canal shows a regular course, predominantly apico-lingual, in close contigui\"",
    ("P360_2", "mandible", "canal_right.adjacent_teeth", 48):
        "reviewed 2026-08-13 -- P360_1.txt: \"The right mandibular canal shows a regular course, predominantly apico-lingual, in close contigu\"",
    ("P437_1", "mandible", "canal_left.adjacent_teeth", 38):
        "reviewed 2026-08-13 -- P437_1.txt: \"The mandibular canal is in close buccal relationship with the apices of the third molars bilater\"",
    ("P437_1", "mandible", "canal_right.adjacent_teeth", 48):
        "reviewed 2026-08-13 -- P437_1.txt: \"The mandibular canal is in close buccal relationship with the apices of the third molars bilater\"",
    ("S0008_2", "mandible", "canal_left.adjacent_teeth", 37):
        "reviewed 2026-08-13 -- S0008_1.txt: \"The mandibular canal follows a predominantly lingual course bilaterally, in close relationship w\"",
    ("S0008_2", "mandible", "canal_right.adjacent_teeth", 47):
        "reviewed 2026-08-13 -- S0008_1.txt: \"Close proximity relationship of the right mandibular canal with teeth 48 and 47 is noted.\"",
    ("S0040_2", "mandible", "fillings", 38):
        "reviewed 2026-08-13 -- S0040_1.txt: \"Presence of restorations on tooth 38.\"",
}


def fdis_in_text(text: str) -> set:
    """Every FDI the report names, in each notation the corpus uses.

    "48", "4.8" and "e.d. 4.8" all mean tooth 48. The dotted form is why a
    naive \\b\\d\\d\\b scan misses teeth: the split_sentences of the deleted
    survey_findings.py
    guards the same thing for the same reason.

    A range NAMES the teeth inside it. "Prosthetic bridge between teeth 13 and
    15" states something about 14 as surely as if it had written the number,
    and stage 1 is right to record the pontic -- so the span is expanded here
    rather than counted against the extraction.
    """
    flat = re.sub(r"(\d)\.(\d)", r"\1\2", text)
    named = {int(m) for m in re.findall(r"\b([1-8][1-8])\b", flat)}
    for a, b in _RANGE_RE.findall(flat):
        a, b = int(a), int(b)
        for sweep in ARCH_SWEEP.values():
            if a in sweep and b in sweep:
                i, j = sorted((sweep.index(a), sweep.index(b)))
                named |= set(sweep[i:j + 1])
    return named


def report_text_for(stem: str, case_id: str, reports_dir: Path) -> str:
    """The text a stage-1 file has to be justified against.

    A per-reader file answers to its own reader: {case}_2_report_facts.json ->
    {case}_2.txt. The un-suffixed file is not a reader at all -- --consensus
    rewrites it by merging every reader -- so it legitimately carries claims
    the first radiologist never made, and it is scored against ALL of the
    case's reports at once.

    Getting this wrong invents errors that read exactly like real ones. Scored
    against report 1 alone, F014's consensus looked like it had fabricated
    crowns at 33-36 and 45; reader 2 had written "Prosthetic crowns are present
    at 45 and a bridge involving 33-34-35-36". 97 of validate's 102 flagged
    findings were this, and a reviewer shown only report 1 has no way to see it.
    """
    own = reports_dir / f"{stem}.txt"
    paths = [own] if own.exists() else sorted(reports_dir.glob(f"{case_id}_*.txt"))
    return "\n".join(p.read_text(encoding="utf-8") for p in paths)


# A report names teeth positionally as often as it numbers them, and an
# absence stated that way is still stated: A028's "Bilateral absence of third
# molars" is 18 and 28 as surely as if it had written the numbers. FDI's
# second digit IS the tooth type, which is what makes this resolvable rather
# than guessed.
# "Teeth from N to N are present in the arch" is the corpus's commonest way of
# stating presence (24 occurrences of that exact template, more with variants),
# and a range there runs along the ARCH, not through the numbers: 17 to 27
# sweeps 17-16-15-14-13-12-11-21-22-23-24-25-26-27, fourteen teeth. Read as a
# numeric interval it yields 17, 18, 21..27 -- 18 wrongly in, and the whole of
# 11-16 silently lost. Measured over both splits: 51 of the 58 such sentences
# had part of their sweep missing from teeth_present.
# A RANGE, NOT AN ENUMERATION. The corpus writes a run of teeth two ways --
# "from 17 to 27" (a range) and "47-46-34-36-37" (a list) -- and a bare hyphen
# belongs to the second. Accepting it turned "a bridge involving 33-34-35-36"
# into a presence claim about the whole sweep, on a sentence that says nothing
# about presence and whose report calls 34 and 36 absent. So only the word
# forms and the dashes that are not the corpus's list separator count.
_ARCH_RANGE = re.compile(
    r"\b([1-4][1-8])\s*(?:to|until|through|–|—)\s*([1-4][1-8])\b", re.I)
_PRESENCE_CLAUSE = re.compile(
    r"in the arch|are present|present in the arch|are visible|complete dentition", re.I)
# The same sweep, stated for a finding rather than for presence: "the
# prosthetic crowns of the maxillary teeth from 17 to 26", "Teeth from 13 to 23
# endodontically treated". Each maps to the field the sentence is about, so the
# screen can check the right list.
_FINDING_CLAUSE = (
    ("crowns", re.compile(r"crown|prosthe|capsul|bridge|rehabilitat", re.I)),
    ("endodontic", re.compile(r"endodont|root canal|endocanal", re.I)),
    ("fillings", re.compile(r"filling|restorat|conservative|composite|amalgam", re.I)),
    ("caries", re.compile(r"cari|decay", re.I)),
)


def arch_range_spans(report_text: str, arch: str, field: str = "teeth_present"):
    """[(sentence, set(FDI))] for every range this arch's text states.

    field="teeth_present" reads presence sentences; any other field reads the
    sentences about that finding, since a range means the same sweep wherever
    it appears.
    """
    out = []
    clause = _PRESENCE_CLAUSE if field == "teeth_present" else dict(
        (k, v) for k, v in _FINDING_CLAUSE).get(field)
    if clause is None:
        return out
    flat = re.sub(r"(\d)\.(\d)", r"\1\2", report_text or "")
    for sentence in re.split(r"(?<=[.;:])\s+", flat):
        if not clause.search(sentence):
            continue
        # Clause by clause, not sentence by sentence. "Elements from 18 to 26
        # present in the arch, 27-28 absent" carries a presence range AND an
        # absence pair in one breath, and reading it whole made the screen
        # demand 27 and 28 in teeth_present -- the very teeth it calls missing.
        for part in re.split(r"[,;]", sentence):
            if _ABSENCE_WORD.search(part):
                continue
            for a, b in _ARCH_RANGE.findall(part):
                a, b = int(a), int(b)
                for sweep in ARCH_SWEEP.values():
                    if a in sweep and b in sweep:
                        i, j = sorted((sweep.index(a), sweep.index(b)))
                        span = set(sweep[i:j + 1]) & ARCH_FDIS[arch]
                        if span:
                            out.append((sentence.strip(), span))
    return out


# Grouped by family, because a sentence names more than one at a time: "the
# first, second and third molars bilaterally" is 6, 7 and 8, and an
# implementation that stops at the first match reads it as the third molar
# alone -- which is how F008's six intrasinusal roots looked unsupported.
# The generic word only fires when no specific one in its family did, so
# "third molar" stays {8} rather than becoming every molar.
_TOOTH_FAMILY = (
    ((r"third molar|wisdom", {8}), (r"second molar", {7}), (r"first molar", {6}),
     (r"\bmolars?\b", {6, 7, 8})),
    ((r"second premolar|second bicuspid", {5}), (r"first premolar|first bicuspid", {4}),
     (r"\b(?:premolars?|bicuspids?)\b", {4, 5})),
    ((r"lateral incisor", {2}), (r"central incisor", {1}),
     (r"\bincisors?\b", {1, 2})),
    ((r"\b(?:canines?|cuspids?)\b", {3}),),
)
_ORDINAL_WORD = {"first": 1, "1st": 1, "second": 2, "2nd": 2,
                 "third": 3, "3rd": 3}
_SHARED_HEAD = re.compile(
    r"((?:(?:first|second|third|1st|2nd|3rd)\b[\s,]*(?:and\s+)?){2,3})"
    r"(molars?|premolars?|bicuspids?)", re.I)
_ORDINAL_TOOTH = [
    (r"\b(?:first|1st)\s+(?:tooth|element|dental element|e\.?d\.?)", 1),
    (r"\b(?:second|2nd)\s+(?:tooth|element|dental element|e\.?d\.?)", 2),
    (r"\b(?:third|3rd)\s+(?:tooth|element|dental element|e\.?d\.?)", 3),
    (r"\b(?:fourth|4th)\s+(?:tooth|element|dental element|e\.?d\.?)", 4),
    (r"\b(?:fifth|5th)\s+(?:tooth|element|dental element|e\.?d\.?)", 5),
    (r"\b(?:sixth|6th)\s+(?:tooth|element|dental element|e\.?d\.?)", 6),
    (r"\b(?:seventh|7th)\s+(?:tooth|element|dental element|e\.?d\.?)", 7),
    (r"\b(?:eighth|8th)\s+(?:tooth|element|dental element|e\.?d\.?)", 8),
]
_QUADRANT_WORD = [
    (r"\b(?:1st|first|I)\s+quadrant|quadrant\s+(?:1|I)\b", {1}),
    (r"\b(?:2nd|second|II)\s+quadrant|quadrant\s+(?:2|II)\b", {2}),
    (r"\b(?:3rd|third|III)\s+quadrant|quadrant\s+(?:3|III)\b", {3}),
    (r"\b(?:4th|fourth|IV)\s+quadrant|quadrant\s+(?:4|IV)\b", {4}),
]
_ARCH_QUADRANTS = {"maxilla": {1, 2}, "mandible": {3, 4}}


def positional_fdis(sentence: str, arch: str) -> set:
    """The FDIs a sentence names by POSITION rather than by number.

    Deliberately used for one thing only -- deciding whether an absence was
    stated -- and NOT for the not-in-text screen. There, "the distal teeth of
    the 1st quadrant have been endodontically treated" (a claim about all of
    them) and "presence of root remnants in the molar region of the 1st
    quadrant" (a claim that some exist there) resolve to the same three teeth
    and mean different things, and only a reader can tell them apart. An
    absence has no such split: whichever teeth the phrase covers, it says they
    are gone.
    """
    # Matched case-insensitively rather than against a lowercased string: the
    # quadrant words are Roman numerals ("the IV quadrant"), and lowercasing
    # the text while the patterns still read "IV" silently matched nothing,
    # which fell back to both quadrants of the arch and doubled the answer.
    low = sentence
    types = set()
    for family in _TOOTH_FAMILY:
        specific = set()
        for pattern, digits in family[:-1] if len(family) > 1 else ():
            if re.search(pattern, low, re.I):
                specific |= digits
        if specific:
            types |= specific
        elif re.search(family[-1][0], low, re.I):
            types |= family[-1][1]
    # An enumeration can share one head noun: "the first, second and third
    # MOLARS" is three teeth written once. Without this the specific patterns
    # above see only "third molars" -- F008 again, whose six intrasinusal roots
    # are stated in exactly this form.
    for ordinals, head in _SHARED_HEAD.findall(low):
        base = 5 if re.match(r"molar", head, re.I) else 3
        for word, n in _ORDINAL_WORD.items():
            if re.search(rf"\b{word}\b", ordinals, re.I):
                types.add(base + n)
    for pattern, digit in _ORDINAL_TOOTH:
        if re.search(pattern, low, re.I):
            types.add(digit)
    if not types:
        return set()

    quadrants = set()
    for pattern, q in _QUADRANT_WORD:
        if re.search(pattern, low, re.I):
            quadrants |= q
    if not quadrants:
        quadrants = set(_ARCH_QUADRANTS[arch])
        if re.search(r"maxill|upper", low, re.I):
            quadrants &= {1, 2}
        if re.search(r"mandib|lower", low, re.I):
            quadrants &= {3, 4}
        # "The right AND left mandibular canals ... in close relationship with
        # the ipsilateral lower third molar" names both sides, and testing each
        # word separately intersects right with left and returns nothing. A
        # sentence carrying both, or "ipsilateral"/"bilateral", covers both.
        both = bool(re.search(r"bilat|both|ipsilateral|contralateral", low, re.I)) or (
            re.search(r"\bright\b", low, re.I) and re.search(r"\bleft\b", low, re.I))
        if not both:
            if re.search(r"\bright\b", low, re.I):
                quadrants &= {1, 4}
            if re.search(r"\bleft\b", low, re.I):
                quadrants &= {2, 3}
    quadrants &= _ARCH_QUADRANTS[arch]
    return {q * 10 + d for q in quadrants for d in types}


_ABSENCE_WORD = re.compile(r"absen|missing|edentul|extract|agenes|avuls", re.I)


def _absence_stated(report_text: str, fdi: int, arch: str = "") -> bool:
    """Does a sentence naming this tooth also call the position empty?

    "Naming" covers both notations the corpus uses: the number, and the
    position ("Bilateral absence of third molars" -- A028, whose presence list
    kept 18 and 28 anyway).
    """
    if not report_text:
        return False
    for s in split_sentences(report_text):
        if not _ABSENCE_WORD.search(s):
            continue
        if arch and fdi in positional_fdis(s, arch):
            return True
        if re.search(rf"\b{fdi}\b", re.sub(r"(\d)\.(\d)", r"\1\2", s)):
            return True
    return False


def _ints(value) -> list:
    if not isinstance(value, (list, tuple)):
        return []
    return [v for v in value if isinstance(v, int) and not isinstance(v, bool)]


OTHER_SIDE = {"canal_right": "canal_left", "canal_left": "canal_right",
              "sinus_right": "sinus_left", "sinus_left": "sinus_right"}


def fix_laterality(facts: dict) -> list:
    """Move every side-scoped FDI to the side its own number names.

    Only the side-scoped lists are repaired, and only within their arch. That
    is not a shortcut -- it is the whole set of cases that HAVE a mechanical
    answer. FDI numbering says which side a tooth is on, so a lower-left tooth
    under canal_right has exactly one correct home and no judgement is
    involved. A cross-ARCH stray would not: the other arch was read by its own
    LLM call from its own paragraph, so moving a claim into it would inject
    something that call declined to make, and dropping it would delete
    something it may have missed. Those stay reported, unfixed.

    Two shapes, both from one cause -- a sentence that names both sides, or
    names a side in words rather than in the tooth number:

      duplicate  the correct side ALREADY lists it (A063: "the roots of tooth
                 37 on the left" went into both canals). Delete the wrong one;
                 the right one is already there and needs no help.
      move       only the wrong side lists it (A095: 25 under sinus_right).
                 Move it across.

    Returns [(message, kind)] describing what it did, and mutates `facts`.
    """
    changes = []
    for arch, sides in SIDE_SCOPED.items():
        block = facts.get(arch)
        if not isinstance(block, dict):
            continue
        for side, allowed in sides.items():
            sub, other_name = block.get(side), OTHER_SIDE[side]
            other = block.get(other_name)
            if not isinstance(sub, dict) or not isinstance(other, dict):
                continue
            key = SIDE_FIELD_LIST[side]
            listed = _ints(sub.get(key))
            wrong = [f for f in listed if f not in allowed]
            if not wrong:
                continue
            # An FDI of the OTHER ARCH is not this repair's business.
            arch_fdis = ARCH_FDIS[arch]
            for f in wrong:
                if f not in arch_fdis:
                    changes.append((f"{arch}.{side}.{key}: {f} is not even in "
                                    f"this arch -- left for review", "skipped"))
                    continue
                sub[key] = [x for x in _ints(sub.get(key)) if x != f]
                target = _ints(other.get(key))
                if f in target:
                    changes.append((f"{arch}.{side}.{key}: dropped {f}, already "
                                    f"on {other_name}", "duplicate"))
                else:
                    other[key] = sorted(target + [f])
                    changes.append((f"{arch}.{side}.{key}: moved {f} to "
                                    f"{other_name}", "move"))
    return changes


def fix_arch_range(facts: dict, report_text: str) -> list:
    """Add back the teeth an arch-sweep range states and the extraction lost.

    Mechanical for the same reason the laterality repair is: the sentence
    names both ends and the arch fixes everything between them, so there is
    one answer and no judgement. "Teeth from 17 to 27 in the arch" is those
    fourteen teeth whatever the extractor made of it.

    Only ADDS to teeth_present, and only teeth the sweep covers. A tooth the
    same block calls absent is left out of the addition -- "Teeth from 18 to
    28 in the arch" beside "absence of 25" is a report describing a gap in the
    run it just gave, and the specific claim wins.
    """
    changes = []
    for arch in ("mandible", "maxilla"):
        block = facts.get(arch)
        if not isinstance(block, dict):
            continue
        absent = set(_ints(block.get("teeth_absent")))
        for sentence, span in arch_range_spans(report_text, arch):
            present = set(_ints(block.get("teeth_present")))
            missing = sorted((span - present) - absent)
            if not missing:
                continue
            block["teeth_present"] = sorted(present | set(missing))
            changes.append((f"{arch}: added {missing} from \"{sentence[:52]}\"",
                            "arch-range"))
    return changes


# An adjacency is a CLAIM OF CONTACT, and the schema says so: "AN EMPTY LIST IS
# THE NORMAL ANSWER ... some tooth is always the nearest one, and naming it
# because it is nearest is a false finding, not a cautious one". So the report
# has to say contact, about that tooth, for the entry to have a source.
# "continuity" is the corpus's own word for it and is NOT a typo for
# contiguity here -- "in intimate continuity with the mesial root of tooth 38",
# "a relationship of close continuity with the root apices". Missing it made
# two dozen supported adjacencies look unsupported. What is deliberately NOT
# in this list is a course description: "the canal courses lingual to the roots
# of 38 and 48" states where the canal runs, not that it touches anything, and
# the schema's adjacency means touching, indenting or crossing.
_CONTIGUITY = re.compile(
    r"close relationship|relationship with|contiguity|continuity|continuous with"
    r"|in contact|in relation|proximity|adjacen|contact with|abut|impinge"
    r"|encroach|intimate", re.I)


def canal_adjacency_support(report_text: str, arch: str, fdi: int):
    """The sentence that supports this adjacency, or None.

    Named numerically or positionally -- "in close relationship with impacted
    tooth 48" and "in relation to the third molars" both count.
    """
    for sentence in split_sentences(report_text):
        if not _CONTIGUITY.search(sentence):
            continue
        flat = re.sub(r"(\d)\.(\d)", r"\1\2", sentence)
        if re.search(rf"\b{fdi}\b", flat) or fdi in positional_fdis(sentence, arch):
            return sentence.strip()
    return None


def fix_canal_adjacency(facts: dict, report_text: str) -> list:
    """Drop adjacency entries no contiguity sentence supports.

    Mechanical in the same sense as the laterality repair: the question is not
    whether the roots look close on the image -- nobody here can see the image
    -- but whether the REPORT this file is derived from says they touch. If no
    sentence pairs contact with that tooth, the entry came from somewhere other
    than the text. P197's whole canal paragraph is "the mandibular canal is
    well assessable bilaterally, with a predominantly lingual course", and the
    extraction asserted an adjacency at 47.
    """
    changes = []
    for side, key in (("canal_right", "adjacent_teeth"), ("canal_left", "adjacent_teeth")):
        sub = (facts.get("mandible") or {}).get(side)
        if not isinstance(sub, dict):
            continue
        listed = _ints(sub.get(key))
        kept = [f for f in listed
                if canal_adjacency_support(report_text, "mandible", f)]
        for f in listed:
            if f not in kept:
                changes.append((f"mandible.{side}.{key}: dropped {f}, no sentence "
                                f"puts the canal in contact with it", "canal-adjacency"))
        if kept != listed:
            sub[key] = kept
    return changes


# The sinus twin of _CONTIGUITY, and the same base rate applies: the schema
# calls an empty intrasinusal_teeth "the normal answer for most sides". A root
# is in the sinus only if the report says it goes there -- protrudes, projects,
# extends into, dehiscence -- or says it touches the floor. A sinus that is
# merely described above the roots is not a claim about any tooth.
# Any sentence about the sinus that also names the tooth. Narrower wording was
# tried first -- protrusion, contact, the floor -- and it flagged F060_2's "the
# left sinus shows ... displacement of endodontic material from the root canal
# treatment of tooth 25", which is a sinus finding about 25 by any reading and
# the same shape as A095's, the case that made this field visible at all. The
# discrimination this screen can honestly make is "is the sinus discussed with
# this tooth", not "is it a root rather than material".
_INTRASINUSAL = re.compile(r"sinus", re.I)


def intrasinusal_support(report_text: str, arch: str, fdi: int):
    """The sentence that puts this root in the sinus, or None."""
    for sentence in split_sentences(report_text):
        if not _INTRASINUSAL.search(sentence):
            continue
        flat = re.sub(r"(\d)\.(\d)", r"\1\2", sentence)
        if re.search(rf"\b{fdi}\b", flat) or fdi in positional_fdis(sentence, arch):
            return sentence.strip()
    return None


def fix_intrasinusal(facts: dict, report_text: str) -> list:
    """Drop intrasinusal roots no sentence puts in the sinus."""
    changes = []
    for side in ("sinus_right", "sinus_left"):
        sub = (facts.get("maxilla") or {}).get(side)
        if not isinstance(sub, dict):
            continue
        listed = _ints(sub.get("intrasinusal_teeth"))
        kept = [f for f in listed if intrasinusal_support(report_text, "maxilla", f)]
        for f in listed:
            if f not in kept:
                changes.append((f"maxilla.{side}.intrasinusal_teeth: dropped {f}, "
                                f"no sentence puts a root of it in the sinus",
                                "intrasinusal"))
        if kept != listed:
            sub["intrasinusal_teeth"] = kept
    return changes


def _obj_fdis(value, key) -> list:
    if not isinstance(value, (list, tuple)):
        return []
    return [e[key] for e in value
            if isinstance(e, dict) and isinstance(e.get(key), int)]


def audit_case(case_id: str, facts: dict, report_text: str, screens: set,
               stem: str = "") -> list:
    """[(level, screen, message)] for one stage-1 file."""
    out = []
    stated = fdis_in_text(report_text)

    def emit(level, screen, message, field=None, fdi=None):
        """Record a finding, unless a reader has already settled it.

        The field is looked up both whole and truncated at its first dot,
        because one row can be raised under either name: the not-in-text screen
        knows it as "sinus_left.intrasinusal_teeth" and the sinus screen as
        "sinus_left". Trying only one silently failed to suppress 25 recorded
        decisions -- they kept appearing, which is the exact thing the table
        exists to stop.
        """
        why = None
        if field:
            for key in (field, field.split(".")[0]):
                why = ACKNOWLEDGED.get((stem or case_id, arch, key, fdi))
                if why:
                    break
        if why and level == "ERROR":
            level = "NOTE"
            message = f"{message} [acknowledged -- {why}]"
        out.append((level, screen, message))

    for arch in ("mandible", "maxilla"):
        block = facts.get(arch)
        if not isinstance(block, dict):
            continue
        legal = ARCH_FDIS[arch] | ARCH_PRIMARY[arch]

        # -- collect every (field, fdi) this block asserts -------------------
        asserted = []
        for field in FLAT_LISTS:
            asserted += [(field, f) for f in _ints(block.get(field))]
        for field, key in OBJ_LISTS:
            asserted += [(field, f) for f in _obj_fdis(block.get(field), key)]
        for bridge in block.get("bridges") or []:
            if not isinstance(bridge, dict):
                continue
            for key in ("abutment_teeth", "pontic_teeth", "implant_supported_teeth"):
                asserted += [(f"bridges.{key}", f) for f in _ints(bridge.get(key))]
        for side, allowed in SIDE_SCOPED[arch].items():
            sub = block.get(side)
            if not isinstance(sub, dict):
                continue
            listed = _ints(sub.get(SIDE_FIELD_LIST[side]))
            asserted += [(f"{side}.{SIDE_FIELD_LIST[side]}", f) for f in listed]
            if "laterality" in screens:
                for f in listed:
                    if f not in allowed:
                        out.append(("ERROR", "laterality",
                                    f"{arch}.{side}.{SIDE_FIELD_LIST[side]} holds "
                                    f"{f}, which is not on that side"))

        if "laterality" in screens:
            for field, f in asserted:
                if f in PRIMARY or f in legal:
                    continue
                out.append(("ERROR", "laterality",
                            f"{arch}.{field} holds {f}, a tooth of the other arch"))

        if "not-in-text" in screens:
            for field, f in sorted(set(asserted)):
                if f in stated:
                    continue
                # A report states presence wholesale -- "presence of all dental
                # elements", "the molar group of quadrant IV is absent" -- and
                # stage 1 is RIGHT to expand that into sixteen numbers nobody
                # wrote down. Same for a pontic, and for a third molar named
                # only as "the maxillary third molars". Those fields drop to a
                # NOTE. A FINDING is different: a report that hangs a crown or
                # a canal filling on a tooth names the tooth, so an unnamed one
                # there is the extractor supplying a number of its own.
                level = "NOTE" if field.split(".")[0] in DERIVED_FIELDS else "ERROR"
                emit(level, "not-in-text",
                     f"{arch}.{field} asserts {f}, which the report never names",
                     field=field, fdi=f)

        if "arch-range" in screens:
            # PRESENCE ONLY, and the attempt to extend it is the reason for
            # this comment. A range means the same sweep in a finding sentence
            # as in a presence one -- prompt rule 14a says so, and F047's
            # "prosthetic crowns of the maxillary teeth from 17 to 26" is
            # thirteen teeth -- but the SCREEN cannot tell that sentence from
            # the two shapes it also matched on the validate split alone:
            # "Probable restorative treatments involving 18-17-16-14-24-26",
            # where this corpus uses en dashes as a LIST separator, and
            # "Prosthetic bridge from teeth 34 to 36", where 35 is a pontic and
            # carries no crown of its own. It fired 181 times on training
            # against 96 for presence, and its false-positive rate made the
            # sheet worse rather than better. The model reads the context; a
            # regex over dashes does not.
            for sentence, span in arch_range_spans(report_text, arch):
                missing = sorted(span - set(_ints(block.get("teeth_present"))))
                # One message carries a LIST, so a recorded decision removes its
                # tooth from that list rather than silencing the row -- two teeth
                # in one sentence can be settled separately.
                settled = [f for f in missing
                           if ACKNOWLEDGED.get((stem or case_id, arch, "arch range", f))]
                missing = [f for f in missing if f not in settled]
                if missing:
                    out.append(("ERROR", "arch-range",
                                f"{arch}: \"{sentence[:60]}\" sweeps the arch and "
                                f"teeth_present is missing {missing}"))
                for f in settled:
                    out.append(("NOTE", "arch-range",
                                f"{arch}: {f} is inside \"{sentence[:44]}\" and not "
                                f"in teeth_present [acknowledged -- "
                                f"{ACKNOWLEDGED[(stem or case_id, arch, 'arch range', f)]}]"))

        if "canal-adjacency" in screens and arch == "mandible":
            for side in ("canal_right", "canal_left"):
                sub = block.get(side)
                if not isinstance(sub, dict):
                    continue
                for f in _ints(sub.get("adjacent_teeth")):
                    if not canal_adjacency_support(report_text, arch, f):
                        emit("ERROR", "canal-adjacency",
                             f"mandible.{side} claims contact with {f}, which no "
                             f"contiguity sentence names",
                             field=f"{side}.adjacent_teeth", fdi=f)

        if "intrasinusal" in screens and arch == "maxilla":
            for side in ("sinus_right", "sinus_left"):
                sub = block.get(side)
                if not isinstance(sub, dict):
                    continue
                for f in _ints(sub.get("intrasinusal_teeth")):
                    if not intrasinusal_support(report_text, arch, f):
                        emit("ERROR", "intrasinusal",
                             f"maxilla.{side} puts {f} in the sinus, which no "
                             f"sentence states",
                             field=f"{side}.intrasinusal_teeth", fdi=f)

        if "contradiction" in screens:
            present, absent = set(_ints(block.get("teeth_present"))), \
                set(_ints(block.get("teeth_absent")))
            for f in sorted(present & absent):
                # Only a contradiction the READER has to settle is an ERROR.
                # Where the report states the absence outright -- A005's "18
                # and 28 absent in the maxilla", whose presence list kept 28
                # anyway -- resolve_presence already ranks the explicit
                # absence above the presence list, so the ground truth is
                # right and nobody needs to look. It is still worth seeing
                # that stage 1 said both. 22 of the 61 on the training
                # re-extraction are this shape.
                level = "NOTE" if _absence_stated(report_text, f, arch) else "ERROR"
                emit(level, "contradiction",
                     f"{arch}: {f} is in BOTH teeth_present and teeth_absent"
                     + (" (the report states the absence, so the GT resolves it)"
                        if level == "NOTE" else ""),
                     field="present/absent", fdi=f)
            for field in FINDING_FIELDS:
                for f in sorted(set(_ints(block.get(field))) & absent):
                    out.append(("NOTE", "contradiction",
                                f"{arch}.{field} names {f}, which the same block "
                                f"calls absent (legitimate for a pontic or a "
                                f"remnant -- read the sentence)"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="training", choices=("training", "validate"))
    ap.add_argument("--gt-dir", default=None,
                    help="Defaults to dataset/<split>/outputs/ground_truth")
    ap.add_argument("--reports-dir", default=None,
                    help="Defaults to dataset/<split>/reports")
    ap.add_argument("--screen", nargs="+",
                    default=["laterality", "not-in-text", "contradiction",
                             "arch-range", "canal-adjacency", "intrasinusal"],
                    choices=["laterality", "not-in-text", "contradiction",
                             "arch-range", "canal-adjacency", "intrasinusal"])
    ap.add_argument("--case-ids", nargs="+", default=None)
    ap.add_argument("--json", default=None, help="also write findings as JSON")
    ap.add_argument("--quiet", action="store_true",
                    help="counts only, no per-finding lines")
    ap.add_argument("--fix-intrasinusal", action="store_true",
                    help="drop intrasinusal roots no sentence puts in the "
                         "sinus. Same base rate as canal adjacency.")
    ap.add_argument("--fix-canal-adjacency", action="store_true",
                    help="drop canal adjacencies no contiguity sentence "
                         "supports. The schema's own base rate for this field "
                         "is an empty list.")
    ap.add_argument("--fix-arch-range", action="store_true",
                    help="add back the teeth an arch-sweep presence range "
                         "states and the extraction dropped. Same character "
                         "as --fix-laterality: the sentence has one reading.")
    ap.add_argument("--fix-laterality", action="store_true",
                    help="rewrite the side-scoped lists so every FDI sits on "
                         "the side its own number names. Only that screen is "
                         "repairable without reading the sentence. Writes a "
                         "*.bak beside each file it touches, and prints the "
                         "stage-2 replay command for the cases it changed.")
    args = ap.parse_args()

    root = REPO_ROOT
    gt_dir = Path(args.gt_dir or root / f"dataset/{args.split}/outputs/ground_truth")
    reports_dir = Path(args.reports_dir or root / f"dataset/{args.split}/reports")
    screens = set(args.screen)

    files = sorted(gt_dir.glob("*_report_facts.json"))
    if not files:
        sys.exit(f"[FAIL] no *_report_facts.json under {gt_dir}")

    findings, by_screen, bad_files = [], Counter(), set()
    fix_counts, fixed_cases = Counter(), set()
    for path in files:
        stem = path.name.replace("_report_facts.json", "")
        case_id = re.match(r"([A-Z]+\d+)", stem).group(1)
        if args.case_ids and case_id not in args.case_ids:
            continue
        report_text = report_text_for(stem, case_id, reports_dir)
        if not report_text:
            continue
        facts = json.loads(path.read_text())
        if (args.fix_laterality or args.fix_arch_range
                or args.fix_canal_adjacency or args.fix_intrasinusal):
            applied = (fix_laterality(facts) if args.fix_laterality else [])
            if args.fix_arch_range:
                applied += fix_arch_range(facts, report_text)
            if args.fix_canal_adjacency:
                applied += fix_canal_adjacency(facts, report_text)
            if args.fix_intrasinusal:
                applied += fix_intrasinusal(facts, report_text)
            if applied:
                backup = path.with_suffix(".json.bak")
                if not backup.exists():
                    backup.write_text(path.read_text())
                path.write_text(json.dumps(facts, indent=2, ensure_ascii=False) + "\n")
                fixed_cases.add(case_id)
                for message, kind in applied:
                    fix_counts[kind] += 1
                    if not args.quiet:
                        print(f"[FIX  ] {path.name:34} {message}")
        for level, screen, message in audit_case(case_id, facts, report_text,
                                                 screens, stem=stem):
            findings.append({"file": path.name, "case_id": case_id,
                             "level": level, "screen": screen, "message": message})
            by_screen[(screen, level)] += 1
            if level == "ERROR":
                bad_files.add(path.name)

    if not args.quiet:
        for f in findings:
            print(f"[{f['level']:5}] {f['file']:34} {f['screen']:14} {f['message']}")
        print()

    n = len({p.name for p in files}) if not args.case_ids else len(
        {f["file"] for f in findings})
    print(f"audited {len(files)} stage-1 file(s) in {gt_dir}")
    for (screen, level), count in sorted(by_screen.items()):
        print(f"  {level:5} {screen:14} {count}")
    print(f"  files with at least one ERROR: {len(bad_files)}")

    if args.json:
        Path(args.json).write_text(json.dumps(findings, indent=2))
        print(f"  -> {args.json}")

    if (args.fix_laterality or args.fix_arch_range or args.fix_canal_adjacency
            or args.fix_intrasinusal):
        print()
        print(f"repaired {sum(fix_counts.values())} finding(s) in "
              f"{len(fixed_cases)} case(s): "
              + ", ".join(f"{k} {v}" for k, v in sorted(fix_counts.items())))
        if fixed_cases:
            # Stage 2 is deterministic, so the ground truth is only as current
            # as its last replay -- the edit above changes nothing on disk that
            # anything downstream reads until this runs.
            print("  originals kept beside each file as *_report_facts.json.bak")
            print("  now replay stage 2 for them (CPU, no model):\n")
            print(f"    python3 code/ground_truth/parse_reports_to_gt.py \\\n"
                  f"        --reports-dir {reports_dir} --schema schema/schema.json \\\n"
                  f"        --out-dir {gt_dir} --from-report-facts --consensus \\\n"
                  f"        --case-ids {' '.join(sorted(fixed_cases))}")

    sys.exit(1 if bad_files else 0)


if __name__ == "__main__":
    main()
