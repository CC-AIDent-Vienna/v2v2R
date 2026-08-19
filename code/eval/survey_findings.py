#!/usr/bin/env python3
"""
code/eval/survey_findings.py

Per-finding survey of one output directory: for each of five findings --
impaction, endodontic treatment, post-and-core, crown, fillings -- how
many teeth the pipeline claims, at each stage, and how many of those
claims the reference reports support.

(Supersedes survey_impaction.py, which covered impaction only.)

OUTPUT LAYOUT: one SUMMARY table of every scored axis first -- absent
teeth, the five findings, implants and bridges, canal and sinus -- then a
section per group breaking its own rows down. The table is read off the
same totals the sections print, so it cannot drift from them. The reason
for that order is the survey-to-survey diff: after a postprocess flag
change the question is which axis moved, and that has to be answerable
from the top of the file rather than by reading all four sections.

ABSENT TEETH ARE SCORED AGAINST THE MASKS, not the reports -- the one row
in this file that is. No report enumerates all 32 positions, while
facts.structured.teeth_present/teeth_absent always does, and it is the
same segmentation every image is rendered from. Two sources can claim
absence: the panoramic arch read ("absent" is one of its enum values at
each of the sixteen positions per arch) and the four wisdom-tooth 3D
renders (eruption_state absent). The composite per-tooth facts cannot --
build_vqa_pairs only expands them for teeth the segmentation already
places. Note the two ways the mask reaches that row other than as the
label, both printed as caveats beside it: on a run built WITH facts the
panoramic caption already names the absent teeth (NO_FACTS=1 is the arm
that separates transcription from perception), and postprocess's own
absence derivation leans first on `detected == no_image`, i.e. on whether
the segmentation produced a composite crop at all -- so the SUMMARY column
recovers part of the label by construction. PRED is restricted to the two
sources that are actual reads.

TWO STAGES, scored separately because they disagree sharply:

  PRED     -- predictions/{case}_pred.json, the raw reads, unioned over
              every source that can independently assert the finding:
                composite  tooth_{fdi}_morphology.with_endo
                           tooth_{fdi}_morphology.with_{fillings,full_crown,
                                                        post_and_core}
                           (v6.4 folded tooth_{fdi}_restoration and the
                           endodontic fact's root_canal_treatment bool into
                           the morphology block; the pre-v6.4 shapes are still
                           read as a fallback so an older run still surveys)
                3d_render  {lower,upper}_{left,right}_wisdom_tooth.impacted
                           + .orientation
                panoramic  dental_arch_findings_{arch}.findings[fdi]
              Source coverage is NOT uniform, by schema design:
                impaction      2 (3d_render, panoramic) -- the composite
                                 read tooth_{fdi}_impaction was retired in
                                 v6.3; older predictions still carry it, and
                                 it is reported but not counted, see
                                 RETIRED_SOURCES
                endodontic     2 (composite, panoramic)
                post_and_core  2 (composite, panoramic)
                fillings       2 (composite, panoramic)
                crown          1 -- the panoramic arch read has no crown
                                 value at all ("too easily confused with a
                                 large filling or dense enamel at panoramic
                                 resolution"), so nothing can cross-check it.
  SUMMARY  -- summaries/{case}_summary.json: what survived
              postprocess_pred.py's cross-source vote and can therefore
              reach the synthesized report (impacted_teeth,
              endodontic_summary.teeth, restoration_summary.groups).

ONE FINDING PER TOOTH ON THE PANORAMIC SIDE. dental_arch_findings_{arch}
answers a single enum per tooth with the priority impacted > post_and_core
> root_canal_treatment > filling > caries > normal, so a crowned,
root-filled, posted tooth votes "post_and_core" only. That caps the
panoramic source's recall on the lower-priority categories; it is a
property of the schema, not a defect in the tooth's reading.

POST-AND-CORE OUTRANKS CROWN, per _definitions.restoration_types: "A
tooth with a post belongs under post_and_core even if a crown also sits
on top -- never double-count as crown." The ground truth below follows
the same rule: a tooth the report describes as post-and-core UNDER a
prosthetic crown is ground truth for post_and_core, and is moved to
crown's unscored set rather than counted as a crown the model missed.

GROUND TRUTH IS HAND-CODED, ON PURPOSE (REPORT_GT).
Read off dataset/<split>/reports/*.txt by a human, not extracted by a
model. parse_reports_to_gt.py already does LLM extraction of the whole
schema and is the right tool for a full evaluation; these five fields are
small enough to read directly, and reading them directly is what keeps
the judgement calls visible:
  - FDI numbers appear as "48" and as "4.8", inside prose an FDI regex
    mangles, and after abbreviations ("e.d. 34, 35 and 43") that a naive
    sentence splitter cuts in half;
  - the vocabulary is enormous and indirect -- "sequelae of endodontic
    treatment", "conservative coronal restoration", "prosthetic crown on
    a radicular stump", "in semi-bony inclusion", "in disodontiasis";
  - reports for the same case disagree (S0010's two readers list almost
    disjoint sets of restored maxillary teeth; F036's two disagree on
    whether 38 is impacted or extruded).
Re-audit with --dump-report-sentences; every run also cross-checks the
table against that scan and warns about drift.

UNSCORED TEETH are the second half of the ground truth and matter as much
as the positives. A claim on an unscored tooth counts as neither right
nor wrong, because the report cannot settle it:
  - crowns carried by an IMPLANT, not a tooth (A019/14, A037/15-17,
    F030/25, P345/46-47, P397/47, S0000, S0044/36-47) -- a different
    fact owns those;
  - PONTICS, the crowns of a bridge that span an edentulous site with no
    tooth under them (A037/45 cantilever, F067/15 and /25 "no dental
    stump or implant supporting the crown");
  - findings the radiologist explicitly could not resolve -- F014/F015's
    "a post is likely in rr 15 and in teeth 25-26-27 (not well
    distinguishable between post-and-core or endodontic treatment)",
    F003's "the distal teeth of the 1st quadrant have been
    endodontically treated" (which teeth?), P345's "bridge prosthetic
    work in the first and second quadrants, not further characterizable";
  - teeth one report calls crowned and another calls conservatively
    restored (F014/14, F014/16).

LATERALITY: NOTHING IS MIRRORED ANY MORE. LR_INVERTED_REPORTS is empty.
This once held A008, A018, A019 and A097, on the reading that those
reports were left-right inverted relative to the CBCT -- A008's "teeth
25, 28, 16 absent" against a segmentation whose absent teeth were then
15, 18, 26, which is exactly that set mirrored. The direction turned out
to be the other way round: the flip was in the SEGMENTATION upload, not
in the reports, and the masks were corrected in the v03 dataset. A008's
segmentation now reads 16, 25, 28 -- the report's own numbers. So the
reports were always in label space, and mirroring them would now CREATE
the error the flag existed to remove. See the table beside
LR_INVERTED_REPORTS for the re-check on all three testable cases.
--label-space therefore does nothing until some case is put back in that
set; it is kept for exactly that possibility.

SPLITS OTHER THAN validate: --no-gt.
Every ground-truth table in this file -- REPORT_GT, UNSCORED, IMPLANT_GT,
BRIDGE_GT, CANAL_POSITION_GT, IAN_CLOSE_GT, SINUS_MUCOSA_GT,
INTRASINUSAL_GT -- is hand-coded over the 40 validate cases, so the survey
is scored only on that split; resolve_gt exits on any other case_id rather
than silently scoring it against an absent entry.

--no-gt drops the scoring instead of the case. It reports the two CLAIM
COUNTS the survey exists to compare -- PRED (unioned over every source that
can assert the finding) and SUMMARY (what survived postprocess_pred.py's
cross-source vote) -- and prints nothing derived from ground truth: no
tp/fp/prec/rec, no per-source precision, no "never claimed" list. That
makes it usable on training (582 cases), where the question is how much the
vote discards, not whether the survivors are right.

It also skips the prosthetics and anatomy sections outright. Those are
scored end to end against IMPLANT_GT / CANAL_POSITION_GT / the rest, so
without ground truth they would print 0.00 precision everywhere -- a wrong
answer rather than a missing one. Run them on validate, without --no-gt.

Usage:
    python code/eval/survey_findings.py --run-dir outputs/aksssr_v6_combined_validate
    python code/eval/survey_findings.py --run-dir outputs/aksssr_v5_validate \\
        --category crown fillings --per-case
    python code/eval/survey_findings.py --dump-report-sentences --category endodontic
    python code/eval/survey_findings.py --run-dir outputs/aksssr_v6_training \\
        --no-gt --out outputs/aksssr_v6_training/survey/survey.json
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

CATEGORIES = ("impaction", "endodontic", "post_and_core", "crown", "fillings",
              "restoration", "caries", "root_remnant")

# The subset REPORT_GT hand-codes, and therefore the only categories THIS
# script can score. `caries` and `root_remnant` are extracted (pred_claims and
# summary_claims carry them) but have no REPORT_GT entry, so scoring them here
# would read GT 0 and file every claim as a false positive. They exist for
# compare_sources.py, which scores against the GENERATED ground truth and needs
# no hand-coded table -- see docs/postprocess_pipeline.md, THE TABLE. Add them here
# the day REPORT_GT gains the two keys, not before.
SCORED_CATEGORIES = ("impaction", "endodontic", "post_and_core", "crown",
                     "fillings", "restoration")

# "restoration" is the PANORAMIC's own category, added with schema v7.1.
# That read can no longer tell a crown from a large filling from a bridge
# abutment retainer -- the three collapsed into one enum value, because at
# arch resolution they are one bright capped tooth -- so it can no longer vote
# in the crown or fillings rows. Giving it a row of its own is the honest
# alternative to splitting a claim it never made: crown and fillings stay
# composite-only, and this row is where the panoramic's restorative claims are
# counted and scored. Its ground truth is derived, not hand-coded -- crown u
# fillings out of REPORT_GT, since a report's crowns and fillings are exactly
# what a panoramic "restoration" would be right about (see resolve_gt).

# The four enum values of _definitions.impaction_direction that assert
# impaction. "none" denies it; anything else the model has emitted here
# ("absent", "not_included", "not_erupted", "not_impacted") is an answer to
# a different question and is not a claim either way.
# The direction an impaction is tipped toward. Schema v6.4 replaced the old
# single enum (none|mesial|distal|horizontal|vertical, where "none" meant not
# impacted) with a bool + a separate `orientation`, and swapped
# horizontal/vertical for the buccolingual pair -- so both vocabularies are
# accepted here, since a survey run is often pointed at an older run's
# predictions.
IMPACTION_VALUES = {"mesial", "distal", "horizontal", "vertical",
                    "lingual", "buccal", "unspecified"}

WISDOM = {18, 28, 38, 48}

# The 32 permanent positions, per arch, in FDI order. Fixed by the schema's
# dental_arch_findings_{arch}.findings, which answers exactly these sixteen
# keys per arch -- so this is also the denominator of the absent-teeth read.
ARCH_TEETH = {
    "mandible": tuple(range(31, 39)) + tuple(range(41, 49)),
    "maxilla": tuple(range(11, 19)) + tuple(range(21, 29)),
}
ALL_TEETH = frozenset(f for positions in ARCH_TEETH.values() for f in positions)

# 3D-render fact name -> the FDI it reads. Fixed by schema.json.
WISDOM_SLOT = {
    "lower_right_wisdom_tooth": 48,
    "upper_right_wisdom_tooth": 18,
    "lower_left_wisdom_tooth": 38,
    "upper_left_wisdom_tooth": 28,
}

# dental_arch_findings_{arch}.findings value -> category it votes for.
#
# Both vocabularies, because a survey run is often pointed at an older run's
# predictions. v7.1 (top) collapsed the enum to five values; the v6.x words
# below it can no longer be emitted but are still on disk in every stored
# prediction.
#
# "defect" maps to nothing on purpose, exactly as its v6.x half "caries" did:
# neither caries nor a root remnant is one of the five findings surveyed here.
# And note what v7.1 costs the post_and_core row -- a posted tooth now reads
# root_canal_treatment on the panoramic, so that row loses its panoramic
# source entirely and becomes composite-only, which is the truthful account of
# what the image can support.
ARCH_VALUE = {
    "impacted": "impaction",
    "root_canal_treatment": "endodontic",
    "restoration": "restoration",
    # v7.1 spells caries `defect`, because at arch resolution a carious lesion
    # and a retained root are one dark irregular crown -- so this value votes
    # on caries and CANNOT vote on root_remnant, which is composite-only.
    "defect": "caries",
    # -- v6.x, read-only --
    "post_and_core": "post_and_core",
    "filling": "fillings",
    "caries": "caries",
}

# restoration_summary.groups key / tooth_{fdi}_restoration.restoration_type
# value -> category. The schema spells the filling value plural.
RESTORATION_VALUE = {"crown": "crown", "post_and_core": "post_and_core",
                     "fillings": "fillings"}

# EMPTY ON PURPOSE -- see the LATERALITY note in the module docstring.
#
# This held {"A008", "A018", "A019", "A097"} until the flip was traced to the
# SEGMENTATION upload rather than to the reports. The masks were corrected in
# the v03 dataset, so those reports' FDI numbers now agree with the
# segmentation as written, and mirroring them would introduce the very error
# the flag was added to remove. Re-checked against the current data:
#
#   case  report absent          segmentation absent      direct  mirrored
#   A008  16, 18, 25, 28         16, 25, 28                    3         1
#   A019  35, 37, 46, 47         38, 46, 47                    2         1
#   A097  31,32,36,37,41,46      31,32,36,37,41,46             6         4
#
# (A018's segmentation finds no teeth at all, so it cannot be tested this way;
# it is dropped with the other three, having been added with them for the same
# reason.) No validate report contains a laterality statement any more --
# grep -i 'invert|mirror|left-right|laterality' over dataset/validate/reports
# matches nothing, A022 included.
#
# Kept as an empty set rather than deleted so --label-space stays valid and a
# genuinely inverted case found later has somewhere to go.
LR_INVERTED_REPORTS: Set[str] = set()

# (category, source) pairs that predictions may still carry but the pipeline
# no longer consumes. tooth_{fdi}_impaction was dropped from the schema in
# v6.3 as unreliable -- this survey is what measured it (18 claims, 0 correct)
# -- and build_impacted_teeth deliberately ignores it even on older
# predictions that still have the field. It is excluded from the PRED column
# so that column means "what the pipeline can act on", and reported on its own
# line underneath so the evidence for retiring it stays visible.
RETIRED_SOURCES = {("impaction", "composite")}

# ---------------------------------------------------------------------------
# Ground truth, read by hand from dataset/validate/reports/*.txt.
#
#   case -> category -> list of FDI numbers the reports assert
#   UNSCORED[case][category] -> FDI numbers the reports cannot settle
#
# Union across a case's reports: this table records a finding stated by ANY
# radiologist. Scoring then applies the agreement rule -- see
# READER_DISAGREEMENT below -- so a finding only some readers assert is
# unscored rather than counted. The table stays the union so it remains a
# faithful record of what was read; the rule lives in resolve_gt().
# Every one of the 40 validate cases is listed, including the ones with
# nothing to report -- an absent case would be indistinguishable from an
# unreviewed one, and "the report says nothing here" is exactly the claim
# that turns a prediction into a false positive.
# ---------------------------------------------------------------------------
REPORT_GT: Dict[str, Dict[str, List[int]]] = {
    # -- impaction wording is quoted in IMPACTION_NOTES below ----------------
    "A008": {"impaction": [48], "endodontic": [24, 36, 46],
             "post_and_core": [24, 36, 46], "crown": [], "fillings": []},
    "A018": {},                                    # edentulous, nothing stated
    "A019": {"endodontic": [15, 23, 24, 34, 36, 45, 48], "fillings": [28]},
    "A022": {"endodontic": [32, 33, 42, 43],
             "crown": [31, 32, 33, 41, 42, 43],    # "circular fixed prosthetic
             "fillings": [16, 17]},                #  rehabilitation" 33-43
    "A034": {"endodontic": [16, 17, 18, 24, 36], "crown": [16, 17, 18, 36],
             "fillings": [25, 26, 27, 37, 47, 48]},
    "A037": {"impaction": [38, 48], "endodontic": [21, 25, 35, 46],
             "crown": [14, 18, 21, 25, 35, 44, 46]},
    "A041": {"endodontic": [11, 13, 21, 22, 23, 34, 35, 43],
             "crown": [11, 12, 13, 14, 21, 22, 23, 24, 26, 34, 35, 43]},
    "A078": {"impaction": [48], "endodontic": [12], "fillings": [46]},
    "A092": {"endodontic": [14, 15, 35]},
    "A097": {"endodontic": [25], "fillings": [14, 18]},
    "F002": {"endodontic": [25, 26, 41], "crown": [25, 35], "fillings": [26]},
    "F003": {"impaction": [48], "endodontic": [24, 25, 26, 27, 34, 35],
             "crown": [24, 25, 26, 27, 34, 35], "fillings": [11, 12, 21]},
    "F006": {"impaction": [13, 18], "endodontic": [36, 37, 45],
             "crown": [36, 37], "fillings": [15, 16, 17, 26, 27, 34, 35]},
    "F014": {"endodontic": [14, 25], "post_and_core": [45],
             "crown": [25, 26, 27, 33, 34, 35, 36, 45, 17], "fillings": [38, 48]},
    "F015": {"endodontic": [45], "post_and_core": [45],
             "crown": [25, 26, 27, 45], "fillings": [14, 16, 48]},
    "F030": {"endodontic": [15, 35, 36, 45, 46, 47], "crown": [15, 36, 46, 47],
             "fillings": [14, 16, 17, 18, 24, 26, 27, 28, 34, 45]},
    "F036": {"impaction": [38, 48], "endodontic": [27, 47],
             "fillings": [16, 17, 26, 34, 35, 36, 37, 44, 45, 46]},
    "F041": {"impaction": [18, 48], "fillings": [16, 17, 27, 36, 37]},
    "F043": {"post_and_core": [32, 33, 42, 43]},
    "F067": {"impaction": [28], "endodontic": [16],
             "crown": [14, 16, 17, 24, 26, 27], "fillings": [45, 46]},
    "P014": {"endodontic": [44, 45],
             "crown": [31, 32, 33, 35, 41, 42, 43, 45]},
    "P112": {"impaction": [38, 48]},
    "P123": {},                                    # edentulous, nothing stated
    "P330": {"impaction": [48], "fillings": [26, 46]},
    "P345": {"endodontic": [35, 45], "crown": [35, 45],
             "fillings": [33, 34, 43]},
    "P397": {},
    "P405": {"endodontic": [31, 37, 41, 42, 46], "fillings": [36]},
    "P452": {"endodontic": [36], "crown": [36, 46, 47]},
    "P482": {"impaction": [38, 48]},
    "P492": {"impaction": [38, 48]},
    "S0000": {"impaction": [38]},
    "S0009": {},
    "S0010": {"impaction": [38, 48],
              "fillings": [11, 12, 14, 15, 16, 17, 25, 27, 36, 37, 46, 47]},
    "S0017": {"impaction": [38, 48]},
    "S0021": {"impaction": [38, 48]},
    "S0027": {"fillings": [15, 16, 26, 34, 35, 36, 37, 43, 44, 45, 47, 48]},
    "S0037": {"impaction": [38, 48]},
    "S0044": {"crown": [45], "fillings": [44, 45]},
    "S0048": {"impaction": [38, 48],
              "fillings": [17, 27, 36, 37, 46, 47]},
    "S0050": {"impaction": [28, 38],
              "fillings": [16, 17, 22, 26, 33, 36, 37, 43, 46, 47]},

    # -- TRAINING split ------------------------------------------------------
    # Development for the few-shot probe scores on training, so validate stays
    # held out. Case IDs do not collide between splits,
    # so one table serves both. Read the same way as the validate entries:
    # union of the case's reports, with the agreement rule applied at scoring.
    "A031": {"endodontic": [15, 25]},           # single report
    "F058": {"endodontic": [36, 37, 45]},       # mandible; maxilla partial
    "P008": {"impaction": [38, 48], "endodontic": [36], "fillings": [46]},
    "S0043": {"endodontic": [35], "crown": [35], "fillings": [37]},

    # -- TRAINING split, SCALE-UP TO 24 SCORED CASES -------------------------
    # HAND-CHECKED 2026-08-12. Every entry read against its own quoted report
    # sentence, with the reports open. Three passes got here:
    #
    #   1. drafted by draft_report_gt.py (keyword pass + the Qwen3-14B
    #      extraction, marked where they disagree) -- 204 findings;
    #   2. a mechanical pass, 28 deleted: double-counts from one sentence,
    #      findings on teeth the mask says are absent, and one anaphora
    #      (P351's "root remnant of 32, previously treated endodontically"
    #      attaching to seven absent teeth);
    #   3. the impaction victims, 7 deleted: "48 impacting on the distal root
    #      of 47" does not make 47 impacted;
    #   4. the reviewer's read of all 176 survivors and the 17 sentences the
    #      drafter had declined -- 17 more deleted, mostly crowns that belong
    #      to a bridge rather than to a tooth.
    #
    # 204 -> 152 findings. For the error rates measured along the way, by
    # which pass asserted a finding, see the plan: both passes agreeing was
    # wrong 6% of the time, keyword-only 20%, LLM-only 40%.
    "A003": {"endodontic": [12, 35],
               "post_and_core": [33, 35, 36, 47],
               "crown": [12, 21, 23, 24, 33, 35, 36, 44, 45, 46, 47]},
    "A044": {"endodontic": [26, 35, 36],
               "crown": [26],
               "fillings": [15, 16, 17, 24, 25, 27, 37, 44]},
    "A060": {"endodontic": [14, 15, 26, 27],
               "crown": [15, 26, 27],
               "fillings": [34, 35, 37, 38, 45, 46, 47, 48]},
    "A085": {},                                 # reports assert none of the five
    "A127": {"endodontic": [15, 46],
               "crown": [15, 46],
               "fillings": [11, 12, 13, 21, 22, 23, 24, 26, 27, 31, 32, 33, 41, 42, 43, 44]},
    "F018": {"impaction": [48]},
    "F022": {"impaction": [18, 38],
               "endodontic": [26, 37],
               "fillings": [14, 15, 16, 24, 25, 26, 35, 37, 44, 45, 46]},
    "F033": {"impaction": [48], "fillings": [15, 16, 24, 26, 36]},  # 47 is what 48 impacts ON, not impacted (see rule 6 in draft_report_gt)
    "F055": {"impaction": [18, 28, 38, 48]},
    "F064": {"impaction": [38, 48]},
    "P248": {"endodontic": [34, 45], "fillings": [32, 45]},
    "P328": {"endodontic": [34, 38]},
    "P333": {"fillings": [11, 16, 21, 26, 27, 37, 46, 47]},
    "P340": {"endodontic": [35, 36],
               "post_and_core": [35, 36],
               "crown": [35, 36],
               "fillings": [23, 24, 25, 26, 27, 34, 35, 37, 44, 45]},
    "P351": {"endodontic": [32]},
    "S0014": {"impaction": [38, 48]},
    "S0022": {"impaction": [38, 48], "fillings": [26]},
    "S0029": {"fillings": [16, 17, 27, 36, 37, 46, 47, 48]},
    "S0033": {"impaction": [48], "fillings": [36, 45, 47]},
    "S0046": {"endodontic": [25], "fillings": [15, 17, 24, 25, 26, 27, 34, 37, 44, 46]},
}

# ---------------------------------------------------------------------------
# THE AGREEMENT RULE
#
# REPORT_GT above records the UNION of a case's reports -- what was actually
# read, reader by reader, which is the right thing for the table to preserve.
# But a finding only ONE of two radiologists asserts is not something the model
# can be marked wrong for missing, and not something it can be credited for
# finding either. resolve_gt() therefore subtracts these from the positive set
# and adds them to the unscored set: stated, but not settled.
#
# SILENCE IS NOT DISAGREEMENT, and the distinction is the whole reason this
# table is hand-checkable rather than derived on the fly. Readers write at
# different granularity: F014_1 says "partial edentulism of the 3rd and 4th
# quadrants" and names no tooth at all, while F014_2 enumerates a post-and-core
# at 45, crowns, and a 33-36 bridge. Reader 1 is not contradicting reader 2 --
# they are describing regions. Treating that as disagreement would have moved
# 62 further findings out of the scored set and stopped crediting the model for
# findings one radiologist documented explicitly.
#
# So a finding lands here ONLY when the other reader DOES enumerate that
# category and omits this tooth. Generated by the GT drafter's
# inter-reader agreement audit over dataset/validate/reports (that
# tool belongs to the label-quality work and is not in this release)
# and pasted back after checking. 37 findings across 12 cases; 9 of S0027's are
# fillings, which is where readers disagree most -- small restorations are the
# most subjective call in the set, and also the category the pipeline scores
# worst on, so a chunk of that apparent error was never settled ground truth.
#
# Attribution comes from the keyword pass, which misses ~5% of findings, so an
# entry here can be a wording the tables did not catch rather than a true
# disagreement. Re-run the audit after editing REPORT_GT.
# ---------------------------------------------------------------------------
READER_DISAGREEMENT: Dict[str, Dict[str, List[int]]] = {
    "F006": {"impaction": [13, 18]},
    "F015": {"endodontic": [45]},
    "F030": {"endodontic": [15]},
    "F036": {"endodontic": [47], "impaction": [38]},
    "F067": {"crown": [14, 17, 24], "fillings": [45, 46]},
    "P014": {"crown": [35]},
    "P112": {"impaction": [38]},
    "P330": {"fillings": [26]},
    "P405": {"endodontic": [31, 42], "fillings": [36]},
    "S0010": {"fillings": [11, 12, 14, 15, 16, 17], "impaction": [38]},
    "S0027": {"fillings": [15, 16, 26, 34, 37, 44, 45, 47, 48]},
    "S0050": {"fillings": [22, 33, 43], "impaction": [28]},
}

# Teeth the reports cannot settle -- see the module docstring. A claim here
# is counted as neither true nor false. Crown entries shadowed by a
# post_and_core ground truth are added automatically (see resolve_gt).
UNSCORED: Dict[str, Dict[str, List[int]]] = {
    # -- TRAINING, from the 2026-08-12 hand-check. Crowns that belong to a
    # BRIDGE rather than to the tooth under them: the report names them, the
    # schema has no value for them in dental_arch_findings, and the model
    # cannot be right or wrong about them. Stated, not settled.
    "P328": {"crown": [34, 38]},
    "P351": {"crown": [31, 33, 34, 41, 42, 43, 44]},
    "A019": {"crown": [14, 35]},                   # implant-supported
    "A034": {"post_and_core": [16, 17, 18, 36],    # "crown on a radicular
             "crown": [46]},                       #  stump" -- post implied?
    "A037": {"crown": [15, 16, 17, 45]},           # implants; 45 cantilever
    "A041": {"crown": [15, 17, 25, 27]},           # implant-supported
    "F003": {"endodontic": [16, 17, 18]},          # "distal teeth of Q1"
    "F014": {"crown": [14, 15, 16],                # bridge vs conservative
             "fillings": [14, 16],                 #  treatment, same report
             "post_and_core": [14, 15, 25, 26, 27]},
    "F015": {"endodontic": [15, 25, 26, 27],       # "post or endo, not
             "post_and_core": [15, 25, 26, 27]},   #  distinguishable"
    "F030": {"crown": [25]},                       # implant-supported
    "F043": {"endodontic": [32, 33, 42, 43]},      # post implies endo, unsaid
    "F067": {"crown": [15, 25]},                   # pontics, no stump/implant
    "P345": {"crown": [11, 12, 13, 14, 15, 16, 17, 18,
                       21, 22, 23, 24, 25, 26, 27, 28]},  # Q1/Q2 bridge work,
    "P397": {"crown": [47]},                       # "not characterizable"
    "S0000": {"crown": [31, 42, 43, 45, 46]},      # implant bridge + cantilever
    "S0044": {"crown": [36, 47]},                  # implants

    # -- TRAINING split ------------------------------------------------------
    # Three of these four cases have a maxilla the volume cuts across, which is
    # why the maxillary findings sit here rather than in REPORT_GT: the reports
    # state them, the scan cannot settle them, and a model that declines to
    # call a finding it cannot see should not be marked wrong for it. This is
    # the same judgement schema v6.5 now asks the model to make via
    # dental_arch_findings_{arch}.uncertain_teeth.
    "F058": {"endodontic": [24, 25, 26]},          # "maxilla partially included"
    # Reader 1's "Black Class II restoration" on 36, which the keyword pass
    # misses (the sentence leads with endodontic material) and the Qwen3-14B
    # extraction caught. Reader 2 does not mention it -- single reader, so
    # unscored rather than ground truth.
    "P008": {"fillings": [36]},
    "S0043": {"endodontic": [12, 13, 17, 21, 22, 23, 27],   # "not possible to
              "crown": [11, 12, 13, 17, 23, 24, 27,        #  determine adequacy,
                        36, 46]},                          #  apices outside scan";
                                                   # 36/46 are implant-supported
}

# Report statements that mention a finding but are deliberately NOT ground
# truth for it. Without this the drift check re-raises them on every run and
# the real warnings get lost in them; with it, the reason each was rejected
# is written down where the decision was made.
ACKNOWLEDGED: Dict[Tuple[str, str], str] = {
    ("P330", "endodontic"): "punctate radiopacity at the EDENTULOUS 36 site, "
                            "'residual endodontic material' -- material left "
                            "in bone after extraction, not a treated tooth",
    ("S0000", "endodontic"): "same -- remnant endodontic material in the "
                             "edentulous 36 region",
    ("P452", "fillings"): "'root canal filling reaching the apices' is the "
                          "endodontic fill, not a restorative filling",
    ("P014", "fillings"): "'prosthetic restoration extending from 33 to 43' "
                          "is crown work; scored under crown",
    ("F043", "endodontic"): "post-and-core on 32/33/42/43 implies endodontic "
                            "treatment but no report states it -- unscored",
    ("S0050", "crown"): "report explicitly denies it: 'Absence of prosthetic "
                        "restorations and/or endosseous implants'",
    ("F002", "crown"): "'prosthetic rehabilitation on the 4 implants' is "
                       "implant-borne; the tooth crowns 25/35 are scored",
    ("F003", "crown"): "implant 36 is noted as LACKING its crown",
    ("P397", "crown"): "the only crown is on implant 47, plus a removable "
                       "appliance at 36-37",
    ("F067", "post_and_core"): "'no dental stump or implant supporting the "
                               "crown' at 15/25 is a negation -- those are "
                               "pontics, unscored under crown",
}

# ---------------------------------------------------------------------------
# Prosthetics: implants and fixed bridges.
#
# These are scored differently from the five tooth findings above, and have a
# THIRD source to compare -- dataset/<split>/facts/{case}.json, derived from
# the segmentation masks rather than read out of images by the model. So each
# case gets FACTS / PRED / SUMMARY all measured against the same report.
#
# IMPLANTS USE SLOT SEMANTICS, not a flat tooth set. Reports place an implant
# by region as often as by tooth ("In the 1.5-1.6-1.7 region, presence of an
# endosseous implant" -- ONE implant, three candidate positions), and two
# readers of the same case can number the same implant differently (P397's
# reports say 47 and 46; S0044's say 46 and 47). Each slot below is one
# implant and the set of positions that would satisfy it: a claim landing
# anywhere in the slot finds it, a slot nothing lands in is a miss, and a
# claim matching no slot is a false positive. A flat set would either punish
# a correct region answer (2 of 3 positions "wrong") or silently accept three
# claims for one implant.
# ---------------------------------------------------------------------------
IMPLANT_GT: Dict[str, List[Set[int]]] = {
    "A018": [{43}],                      # "canine region of the 4th quadrant"
    "A019": [{14}, {35}],                # 35 "replaced by an implant"
    "A034": [{46}],
    "A037": [{15, 16, 17}],              # one implant, region-level
    "A041": [{36}, {37}, {44}, {45}, {46}, {47}, {15}, {17}, {25}, {27}],
    "F002": [{45}, {46}, {47}],          # rpt1 says "4 endosseous implants";
                                         # rpt2 names only three -- see notes
    "F003": [{36}, {44}],
    "F030": [{25}],
    "P345": [{46}, {47}],
    "P397": [{46, 47}],                  # one implant, rpt1 calls it 47,
                                         # rpt2 calls it 46
    "S0000": [{31}, {41}, {42}, {43}, {44}, {45}],
    "S0044": [{36}, {46, 47}],           # rpt1 46, rpt2 47, same implant

    # -- TRAINING split. Both readers agree on every slot below; S0043's
    # maxillary implant at 24 is deliberately absent, since reader 1 describes
    # no maxillary structure at all (see BRIDGE_UNSCORED).
    "A031": [{11}, {21}],                # "implants in the 21-11 region"
    "S0043": [{36}, {46}],
}
IMPLANT_NOTES = {
    "F002": "rpt1 states '4 endosseous implants' but names none; rpt2 names "
            "45-46-47. The fourth implant has no position and is not scored.",
    "A008": "facts assert an implant at 15 and the pipeline claims 45, but "
            "NEITHER report mentions an implant in this case at all.",
    "S0050": "report explicitly denies both: 'Absence of prosthetic "
             "restorations and/or endosseous implants'.",
}

# Fixed bridges are scored per CASE, not per tooth: facts carry only a
# `bridge_present` boolean, and the reports describe spans too variably to
# match tooth for tooth ("splinted crowns from 1.4 to 1.7", "circular fixed
# prosthetic rehabilitation", "crowns splinted as a bridge"). Presence is the
# claim all three sources can actually be compared on.
BRIDGE_GT: Dict[str, bool] = {
    "A022": True,    # "circular fixed prosthetic rehabilitation" 33-43
    "A037": True,    # crowns "soldered to distal prosthetic crowns", cantilever
    "A041": True,    # "prosthetic bridges in the 13-14, 15-17 and 21-23 regions"
    "F003": True,    # "prosthetic bridge from teeth 34 to 36"
    "F014": True,    # bridges 33-36, 14-16, 25-27
    "F015": True,    # "fixed bridge prosthetic crown involving 25-26-27"
    "F043": True,    # "prosthetic crowns splinted as a bridge"
    "F067": True,    # "splinted crowns from 1.4 to 1.7" and "2.4 to 2.7"
    "P014": True,    # "splinted coronal elements from 3.3 to 4.3"
    "P345": True,    # "bridge prosthetic work in the first and second quadrants"
    "S0000": True,   # two fixed bridge prostheses on implants
}

# Cases excluded from the bridge score entirely: the report describes
# prosthetic work without committing to whether it spans. F002's "prosthetic
# rehabilitation on the 4 implants has already been performed" is a fixed
# full-arch bridge in most clinics and four single crowns in some, and the
# report never says which -- so neither answer can be called wrong.
BRIDGE_UNSCORED: Dict[str, str] = {
    "F002": "'prosthetic rehabilitation on the 4 implants' -- spanning or not "
            "is never stated",
    "S0043": "three maxillary bridges, reader 2 only; reader 1 stops at "
             "'maxilla partially included' and describes no maxillary work, "
             "so the readers do not agree that a bridge is present",
}

# ---------------------------------------------------------------------------
# Mandibular canal and maxillary sinus.
#
# THE COURSE ENUM CONFLATES TWO AXES the reports keep separate. Radiologists
# write both "the canal has a REGULAR course" (an anomaly judgement) and "in a
# predominantly LINGUAL position" (where it sits buccolingually), and often
# both in one sentence -- A008's "Course of the mandibular canal is regular,
# predominantly lingual bilaterally". schema.json offers one field for both:
# course = regular|lingual|buccal. So "regular" and "lingual" are not
# alternatives in the reports' vocabulary, and a prediction of "regular" for a
# canal the report calls lingual is not a disagreement about anatomy -- it is
# the model answering the other axis.
#
# CANAL_POSITION_GT therefore records ONLY the buccolingual axis, per side,
# and only where a report states it. An empty set means the reports described
# the canal without placing it (usually "regular course" alone), and that side
# is skipped rather than scored. Predictions of "regular" where a position IS
# stated are counted separately as conflated, not as wrong.
#
# "central" (A022, P330) is not in the enum at all: the model cannot answer
# those two correctly whatever it sees. They are counted as unanswerable.
# ---------------------------------------------------------------------------
_L, _B, _C = {"lingual"}, {"buccal"}, {"central"}
CANAL_POSITION_GT: Dict[str, Dict[str, Set[str]]] = {
    "A008": {"right": _L, "left": _L},   "A018": {"right": _L, "left": _L},
    "A019": {"right": _L, "left": _L},   "A022": {"right": _C, "left": _C},
    "A034": {"right": _B, "left": _B},   "A037": {"right": _L, "left": _L},
    "A041": {"right": _L, "left": _L},   "A078": {"right": _L, "left": _L},
    "A092": {"right": _L, "left": _L},   "A097": {"right": set(), "left": set()},
    "F002": {"right": _L, "left": _L},   "F003": {"right": _L, "left": _L},
    "F006": {"right": _L, "left": _L},
    "F014": {"right": set(), "left": _B},          # right is "regular" only
    "F015": {"right": _L, "left": set()},
    "F030": {"right": set(), "left": set()},
    "F036": {"right": _L, "left": _L},   "F041": {"right": _L, "left": _L},
    "F043": {"right": _L, "left": _L},   "F067": {"right": _L, "left": _L},
    "P014": {"right": _L, "left": _L},
    "P112": {"right": set(), "left": set()},
    "P123": {"right": _L, "left": _L},   "P330": {"right": _C, "left": _C},
    "P345": {"right": _L, "left": _L},   "P397": {"right": _L, "left": _L},
    "P405": {"right": _L, "left": _L},
    "P452": {"right": _L, "left": _B},             # the asymmetric cases are
    "P482": {"right": _B, "left": _L},             # what test laterality
    "P492": {"right": _L, "left": _L},
    "S0000": {"right": set(), "left": set()},
    "S0009": {"right": _L, "left": _L},  "S0010": {"right": _L, "left": _L},
    "S0017": {"right": _L, "left": _L},  "S0021": {"right": _L, "left": _L},
    "S0027": {"right": _L, "left": _L},  "S0037": {"right": _L, "left": _L},
    "S0044": {"right": _L, "left": _L},  "S0048": {"right": _L, "left": _L},
    "S0050": {"right": _L, "left": _L},
}

# Teeth the reports place in close relation to the canal / IAN. Compared
# three ways, like implants: facts (structured.ian_close_teeth, mask-derived),
# prediction (mandible_canal_{side}.adjacent_teeth) and summary.
IAN_CLOSE_GT: Dict[str, Set[int]] = {
    "A078": {48}, "F003": {48}, "F015": {48}, "F030": {48}, "F036": {38, 48},
    "F041": {48}, "P112": {38, 48}, "P330": {48}, "P405": {38},
    "S0010": {38, 48}, "S0017": {38}, "S0021": {38, 48}, "S0037": {38, 48},
    "S0048": {37, 38, 48}, "S0050": {47},
}
# A034 says "in contiguity with the roots of the third molar" without naming
# which side's; P482, P492 and P397 state the OPPOSITE ("does not come into
# direct contact"), so a claim there is a real false positive, not unscored.
IAN_CLOSE_UNSCORED: Dict[str, Set[int]] = {"A034": {38, 48}}

# Sinus mucosa per side, where a report states it. thickening covers the
# reports' hypertrophy / hyperemia / hyperplasia / retention pseudocyst /
# "not aerated", all of which the enum can only express as thickening.
SINUS_MUCOSA_GT: Dict[str, Dict[str, str]] = {
    "A008": {"right": "normal", "left": "normal"},
    "A018": {"right": "normal", "left": "normal"},
    "A019": {"right": "normal", "left": "normal"},
    "A022": {"right": "normal", "left": "normal"},
    "A041": {"right": "thickening", "left": "thickening"},
    "A078": {"right": "normal", "left": "normal"},
    "A092": {"left": "thickening"},
    "A097": {"right": "thickening", "left": "thickening"},
    "F002": {"left": "thickening"},
    "F003": {"right": "thickening", "left": "normal"},
    "F014": {"right": "normal", "left": "normal"},
    "F030": {"right": "normal", "left": "normal"},
    "F041": {"right": "thickening", "left": "thickening"},
    "F067": {"right": "normal", "left": "normal"},

    # -- TRAINING split ------------------------------------------------------
    # Read off the sinus sentence of each scored training case. Unlike the
    # REPORT_GT block above these were NOT drafted by a tool: the sentence is
    # one clause with one of two verdicts in it, so the whole judgement is the
    # quote beside it. The dump they were read from is
    #   outputs/fewshot_probe_training/sinus_gt_draft_24.txt
    #
    # Only 11 of the 24 scored cases appear. THIRTEEN SAY THE SINUSES ARE NOT
    # IN THE VOLUME -- every S case and four of six P cases ("Maxillary
    # sinuses: not included in the acquisition volume and not assessable").
    # create_sinus_detail.py renders nothing for those, build_vqa_pairs.py
    # emits no call, and both arms are silent on them identically. That is the
    # real denominator of the sinus arm, and it is a property of how these
    # sub-datasets were acquired, not something the scored set can be
    # re-picked to avoid.
    "A003": {"right": "thickening", "left": "thickening"},   # "normally pneumatized WITH minimal inflammatory mucosal thickening"
    "A031": {"right": "normal", "left": "normal"},           # "sinuses with normal pneumatization"
    "A044": {"right": "normal", "left": "normal"},           # "included in the most caudal portion, normally pneumatized"
    "A060": {"right": "normal", "left": "normal"},           # "included in the scan volume and are normally aerated"
    "A085": {"left": "thickening"},                          # "Increased opacity of the left maxillary sinus"; right not stated
    "A127": {"right": "normal", "left": "normal"},           # "correctly pneumatized bilaterally"
    "F018": {"right": "thickening", "left": "normal"},       # "aerated, with right-sided mucosal hypertrophy"
    "F022": {"right": "thickening", "left": "thickening"},   # "minimally included, not aerated"
    "F055": {"right": "normal", "left": "normal"},           # "the sinuses are aerated"
    "F058": {"right": "thickening", "left": "thickening"},   # "bilateral mucosal hypertrophy"
    "F064": {"right": "thickening", "left": "thickening"},   # both readers: "bilateral mucosal thickening ... sinusitis" / "not aerated"
    # F033 is deliberately absent: "the right is minimally included, the left
    # is not included in the scan volume" states scope and says nothing at all
    # about mucosa. A side with no entry is unread, not wrong.
    # P008 likewise -- neither report mentions the sinuses.
}

# Roots the reports place inside the sinus. Up to schema v7.0 this field was
# only permitted when scope == 'fully_included' AND mucosa_state ==
# 'thickening', which capped its recall on two other reads being right first;
# v7.1 removed that gate, so a miss here is now the sinus read's own.
INTRASINUSAL_GT: Dict[str, Set[int]] = {
    "F036": {18, 27},
    "F041": {17, 18, 26, 27},
}

# The report wording behind each impaction ground truth, kept because these
# are the entries a second reader is most likely to score differently.
IMPACTION_NOTES = {
    ("F036", 38): "impacted (rpt1) / extruded (rpt2) -- reports conflict",
    ("P482", 38): "'in the arch, semi-included'",
    ("S0050", 28): "'not completely erupted' / 'partially erupted' -- never "
                   "the word impacted, but the same finding by the schema",
    ("S0048", 48): "also an impacted SUPERNUMERARY distal to 48, which has no "
                   "FDI and no schema field, so it is not scored either way",
}

# ---------------------------------------------------------------------------
# Report sentence scan -- over-inclusive by design; it exists to prove no
# statement was missed, and is filtered by eye, not by regex.
# ---------------------------------------------------------------------------
KEYWORD_RE = {
    "impaction": re.compile(
        r"impact|inclus|included|retain|unerupt|erupt|disodont|dysodont"
        r"|verted|ectopic|semi", re.I),
    "endodontic": re.compile(r"endodont|root canal|endocanal|obturat", re.I),
    "post_and_core": re.compile(r"post[- ]and[- ]core|endocanal post|\bpost\b"
                                r"|stump|abutment", re.I),
    "crown": re.compile(r"crown|capsul|prosthe|bridge|splint|rehabilitat", re.I),
    "fillings": re.compile(r"filling|restorat|conservative|amalgam|composite"
                           r"|reconstruct", re.I),
}
KEYWORD_RE["restoration"] = re.compile(
    KEYWORD_RE["crown"].pattern + "|" + KEYWORD_RE["fillings"].pattern, re.I)
# "included in the scan volume" is scope, not impaction; "post-extraction
# socket" is not a post; "the crown of tooth X" is anatomy, not a prosthesis.
# Dropped from the strong scan unless the sentence says something only the
# finding itself says.
NOISE_RE = {
    # Three senses of "retain-" appear in these reports and NONE of them is a
    # tooth held in the bone. Measured over all 622 reports: "retainer" x22 in
    # 15 cases is the orthodontic appliance ("lingual splint/retainer from 33
    # to 43", "bar-retained prosthesis"); "retention" x9 in 8 cases is a
    # maxillary sinus retention CYST; "retained" x8 is a root fragment in 3 of
    # its 4 cases. The one genuine impaction sense -- F011's "73 retained with
    # 33, which is impacted" -- says "impacted" in the same sentence and is
    # matched on that word instead, so nothing is lost by demoting these.
    #
    # This cost two fabricated GT entries before it was caught: S0014 and
    # S0022 both drafted impaction [31,32,33,41,42,43] out of "orthodontic
    # splint/retainer extending from tooth 33 to tooth 43", and an invented
    # impaction is worse than a missing one -- it scores correct model silence
    # as a false negative. The crown table above already learned the same
    # lesson about bare "splint"; impaction had not.
    "impaction": re.compile(r"(included|inclusion)\s+(in|within|only|of|at)?\s*"
                            r"(the|its|their)?\s*(scan|acquisition|available|most"
                            r"|volume|portion|caudal|field)"
                            r"|retainer|retention\s+cyst|retained\s+root"
                            r"|bar[- ]retained", re.I),
    "post_and_core": re.compile(r"post[- ]extraction|posterior|post[- ]operative", re.I),
}
# "(?<!in)verted" keeps mesioverted/distoverted/normoverted but not the
# "Inverted Panorex" / "right-left inverted" disclaimers, which are about
# LR_INVERTED_REPORTS, not about a tooth.
STRONG_RE = {
    # "retain" is NOT strong: the noise clause above can only fire on a
    # sentence that is not strong, and every "retain-" sense in this corpus is
    # noise. A real "retained tooth" sentence still matches KEYWORD_RE and is
    # printed as weak/filtered with its FDIs, so it is offered to the reader
    # rather than dropped.
    "impaction": re.compile(r"impact|(?<!in)verted|unerupt|ectopic|semi"
                            r"|disodont|dysodont", re.I),
    "endodontic": re.compile(r"endodont|root canal|endocanal", re.I),
    "post_and_core": re.compile(r"post[- ]and[- ]core|endocanal post|stump"
                                r"|\bpost\b(?![- ]extraction)", re.I),
    # "crown" alone is anatomy far more often than prosthetics -- "the crown
    # of tooth 3.6", "includes the crowns of the upper dental elements",
    # "caries of the buccal crown". A prosthetic crown is always said with
    # one of these.
    # Bare "splint" is usually a lingual orthodontic retainer (A092, P482,
    # S0017, S0048) -- prosthetic only when it is the crowns being splinted.
    "crown": re.compile(r"prosthe|capsul|bridge|rehabilitat|cantilever"
                        r"|splinted\s+crowns|crowns?\s+splinted"
                        r"|crowns?\s+(on|at|are|is|from|supported)", re.I),
    "fillings": re.compile(r"filling|restorat|conservative|amalgam", re.I),
}
# The panoramic's coarse row matches whatever either of its two halves matches
# -- it is right about a crowned tooth and about a filled one, so its audit has
# to look for both. Composed rather than restated, so the two patterns above
# stay the single place either wording is maintained.
STRONG_RE["restoration"] = re.compile(
    STRONG_RE["crown"].pattern + "|" + STRONG_RE["fillings"].pattern, re.I)


def mirror_fdi(fdi: int) -> int:
    """Reflect an FDI number across the midline: 48 <-> 38, 16 <-> 26."""
    quadrant, tooth = divmod(fdi, 10)
    return {1: 2, 2: 1, 3: 4, 4: 3}[quadrant] * 10 + tooth


def resolve_gt(case_id: str, category: str, label_space: bool,
               no_gt: bool = False) -> Tuple[Set[int], Set[int]]:
    """(ground truth teeth, unscored teeth) for one case and category.

    A tooth carrying a post-and-core under a crown is ground truth for
    post_and_core and unscored for crown, per _definitions.restoration_types
    ("never double-count as crown").

    Under --no-gt an unknown case yields empty sets. Nothing derived from
    them is printed (see print_report), so they never become a score of
    zero -- they mark the case as unscored, not as wrong.
    """
    if no_gt and case_id not in REPORT_GT:
        return set(), set()
    if case_id not in REPORT_GT:
        sys.exit(f"[FAIL] {case_id} has no REPORT_GT entry -- add one (an "
                 f"empty dict if its reports state nothing) before scoring it")
    entry = REPORT_GT[case_id]
    if category == "restoration":
        # DERIVED, not hand-coded. A panoramic "restoration" is right about a
        # tooth the report calls crowned OR filled -- including the bridge
        # retainers already recorded under "crown" -- so the union of those two
        # rows is this row's ground truth, and no case had to be re-read to get
        # it. post_and_core is deliberately not unioned in: v7.1 files a posted
        # tooth under root_canal_treatment, not here.
        positive = set(entry.get("crown", [])) | set(entry.get("fillings", []))
        unscored = (set(UNSCORED.get(case_id, {}).get("crown", []))
                    | set(UNSCORED.get(case_id, {}).get("fillings", [])))
    else:
        positive = set(entry.get(category, []))
        unscored = set(UNSCORED.get(case_id, {}).get(category, []))

    # The agreement rule: a finding only some of a case's readers assert is
    # neither credited nor penalised. Applied HERE rather than by editing
    # REPORT_GT, so that table stays the faithful record of what each report
    # said and the rule remains one reversible line. Must precede the
    # `unscored -= positive` below, which would otherwise cancel it.
    disputed = set(READER_DISAGREEMENT.get(case_id, {}).get(category, []))
    if category == "restoration":
        rd = READER_DISAGREEMENT.get(case_id, {})
        disputed = set(rd.get("crown", [])) | set(rd.get("fillings", []))
    positive -= disputed
    unscored |= disputed

    if category == "crown":
        unscored |= set(REPORT_GT[case_id].get("post_and_core", []))
    if category == "restoration":
        # A posted tooth is scored in the post_and_core row and is not a claim
        # this row can be wrong about -- the arch read files it elsewhere.
        unscored |= set(REPORT_GT[case_id].get("post_and_core", []))
    unscored -= positive
    if label_space and case_id in LR_INVERTED_REPORTS:
        return {mirror_fdi(f) for f in positive}, {mirror_fdi(f) for f in unscored}
    return positive, unscored


def _as_dict(value) -> Dict:
    return value if isinstance(value, dict) else {}


def as_list(value) -> List:
    """Coerce a maybe-scalar, maybe-missing list field into a list."""
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


def split_sentences(text: str) -> List[str]:
    """Split on sentence punctuation without cutting FDI numbers.

    Reports write teeth as both "48" and "4.8", and refer to them as
    "e.d. 34" (elemento dentale). A naive split on ". " turns
    "tooth 4.8 impacted" into "8 impacted" and strands "e.d." from its
    tooth list -- in both cases the tooth number is lost.
    """
    guarded = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", text)
    guarded = re.sub(r"\b([a-zA-Z])\.(?=[a-zA-Z]\.|\s*\d)", r"\1<DOT>", guarded)
    return [s.replace("<DOT>", ".").strip()
            for s in re.split(r"(?<=[.;:])\s+", guarded) if s.strip()]


def report_paths(reports_dir: Path, case_id: str) -> List[Path]:
    return sorted(reports_dir.glob(f"{case_id}_*.txt")) or \
        sorted(reports_dir.glob(f"{case_id}.txt"))


def finding_sentences(reports_dir: Path, case_id: str,
                      category: str) -> List[Tuple[str, str, bool]]:
    """[(report name, sentence, is_strong)] for one case and category."""
    hits = []
    noise = NOISE_RE.get(category)
    strong_re = STRONG_RE[category]
    for path in report_paths(reports_dir, case_id):
        for sentence in split_sentences(path.read_text(encoding="utf-8")):
            if not KEYWORD_RE[category].search(sentence):
                continue
            strong = bool(strong_re.search(sentence))
            if noise and noise.search(sentence) and not strong:
                continue
            hits.append((path.name, sentence, strong))
    return hits


def audit_gt_table(reports_dir: Path, case_ids: List[str],
                   categories: List[str]) -> List[str]:
    """Warn where REPORT_GT and the report text may have drifted apart.

    Catches the two ways a hand-coded table goes stale: a case whose
    reports discuss a finding but has no ground truth for it, and ground
    truth for a case whose reports no longer mention it (a re-split or
    renamed report).
    """
    warnings = []
    for case_id in case_ids:
        for category in categories:
            if (case_id, category) in ACKNOWLEDGED:
                continue
            strong = [s for _, s, ok in finding_sentences(reports_dir, case_id,
                                                          category) if ok]
            positive, unscored = resolve_gt(case_id, category, label_space=False)
            coded = positive | unscored
            if strong and not coded:
                warnings.append(f"{case_id}/{category}: reports discuss it but "
                                f"the table is empty -- e.g. {strong[0][:80]!r}")
            if coded and not strong:
                warnings.append(f"{case_id}/{category}: table has "
                                f"{sorted(coded)} but no report sentence "
                                f"mentions it")
    return warnings


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------

def pred_claims(pred: Dict) -> Dict[str, Dict[int, Dict[str, str]]]:
    """{category: {fdi: {source: value}}} for every claim in one prediction."""
    claims: Dict[str, Dict[int, Dict[str, str]]] = {c: {} for c in CATEGORIES}

    def record(category: str, fdi: int, value: str, source: str) -> None:
        claims[category].setdefault(fdi, {})[source] = value

    for key, tooth in _as_dict(pred.get("teeth")).items():
        fdi = int(key.split("_")[1])
        tooth = _as_dict(tooth)

        # No composite impaction claim: tooth_{fdi}_impaction left the schema
        # in v6.3 and postprocess_pred stopped reading it, so surveying it
        # would score a source the pipeline does not use. A legacy prediction
        # that still carries the fact is ignored here for the same reason.

        # Schema v6.4 folded the composite's restorative and endodontic claims
        # into tooth_{fdi}_morphology as bools. Read them the way
        # postprocess_pred.composite_restoration_type does -- same priority,
        # so the survey scores what the pipeline actually consumes -- and fall
        # back to the pre-v6.4 shapes when surveying an older run.
        morphology = _as_dict(tooth.get(f"tooth_{fdi}_morphology"))
        endo = _as_dict(tooth.get(f"tooth_{fdi}_endodontic_treatment"))

        rct = morphology.get("with_endo")
        if rct is None:
            rct = endo.get("root_canal_treatment")     # pre-v6.4 predictions
        # The field is a bool, but the VLM has emitted the strings "true"/"false".
        if str(rct).lower() == "true":
            record("endodontic", fdi,
                   str(endo.get("filling_quality") or "unspecified"), "composite")

        restoration = None
        for flag, value in (("with_post_and_core", "post_and_core"),
                            ("with_full_crown", "crown"),
                            ("with_fillings", "fillings")):
            if str(morphology.get(flag)).lower() == "true":
                restoration = value
                break
        if restoration is None:                        # pre-v6.4 predictions
            restoration = _as_dict(tooth.get(f"tooth_{fdi}_restoration")).get("restoration_type")
        if restoration in RESTORATION_VALUE:
            record(RESTORATION_VALUE[restoration], fdi, restoration, "composite")
        # The composite's second vote, into the panoramic's coarser row: a
        # crown or a filling IS a "restoration" as the arch read means it. A
        # post is not -- v7.1 files a posted tooth under root_canal_treatment,
        # so counting it here would make the two sources disagree about a
        # tooth they actually agree on. The composite picks ONE value by
        # priority, so a post-and-core UNDER a crown votes post_and_core only
        # and is silent here; that lossiness is the composite's, and predates
        # this row.
        if restoration in ("crown", "fillings"):
            record("restoration", fdi, restoration, "composite")

        # Two more composite facts, independent of the restorative one above --
        # a tooth can be carious AND filled, and a remnant carries findings of
        # its own. Neither is part of the restoration priority chain.
        if str(morphology.get("with_caries")).lower() == "true":
            record("caries", fdi, "unspecified", "composite")
        if str(morphology.get("is_remnant")).lower() == "true":
            record("root_remnant", fdi, "unspecified", "composite")

    facts = _as_dict(pred.get("global"))
    for slot, fdi in WISDOM_SLOT.items():
        fact = _as_dict(facts.get(slot))
        impacted = fact.get("impacted")
        if impacted is True or str(impacted).lower() == "true":
            # v6.4: the bool says WHETHER, `orientation` says which way. A
            # tooth impacted with no orientation given still counts as a claim.
            direction = fact.get("orientation")
            record("impaction", fdi,
                   direction if direction in IMPACTION_VALUES else "unspecified",
                   "3d_render")
        elif impacted in IMPACTION_VALUES:
            # pre-v6.4: one enum carried both, with "none" meaning not impacted
            record("impaction", fdi, impacted, "3d_render")

    for arch in ("mandible", "maxilla"):
        findings = _as_dict(_as_dict(facts.get(f"dental_arch_findings_{arch}")).get("findings"))
        for fdi, value in findings.items():
            # This fact answers one enum per tooth and has no direction or
            # quality field -- it only names the finding.
            if value in ARCH_VALUE:
                record(ARCH_VALUE[value], int(fdi), "unspecified", "panoramic")
    return claims


def uses_legacy_shape(pred: Dict) -> bool:
    """Was this prediction generated before v6.4 restructured the per-tooth facts?

    v6.4 folded the composite restorative claims into tooth_{fdi}_morphology
    as bools (with_full_crown / with_fillings / with_post_and_core) and
    retired tooth_{fdi}_restoration. pred_claims reads both, so the PRED
    column is unaffected -- but postprocess_pred.py reads ONLY the current
    shape, so rebuilding summaries from a legacy prediction silently yields
    empty restoration groups. That looks exactly like a quality collapse in
    the SUMMARY column and is not one, so it is detected and stated.
    """
    for key, tooth in _as_dict(pred.get("teeth")).items():
        fdi = int(key.split("_")[1])
        tooth = _as_dict(tooth)
        if tooth.get(f"tooth_{fdi}_restoration") is not None:
            return True
        morphology = _as_dict(tooth.get(f"tooth_{fdi}_morphology"))
        if morphology and "with_full_crown" not in morphology:
            return True
    return False


def composite_coverage(pred: Dict) -> Dict[str, Set[int]]:
    """{category: teeth whose composite fact was actually answered}."""
    # No entry for impaction: tooth_{fdi}_impaction was dropped from the
    # schema (v6.3), so no composite fact backs that category any more and
    # "was this tooth even read" is not a question about it.
    #
    # Each category lists the facts that would carry it, NEWEST FIRST, and any
    # one of them being present means the tooth was read. v6.4 folded the
    # restorative claims out of tooth_{fdi}_restoration and into
    # tooth_{fdi}_morphology, so naming only the old fact reported every v6.4+
    # tooth as uncropped -- which is the opposite of what this function is for,
    # and made "got no composite crop at all" count teeth that had one. Same
    # fallback pattern as pred_claims, for the same reason.
    fact_for = {"endodontic": ("morphology", "endodontic_treatment"),
                "post_and_core": ("morphology", "restoration"),
                "crown": ("morphology", "restoration"),
                "fillings": ("morphology", "restoration"),
                "restoration": ("morphology", "restoration"),
                "caries": ("morphology",),
                "root_remnant": ("morphology",)}
    covered: Dict[str, Set[int]] = {c: set() for c in CATEGORIES}
    for key, tooth in _as_dict(pred.get("teeth")).items():
        fdi = int(key.split("_")[1])
        for category, facts in fact_for.items():
            if any(isinstance(_as_dict(tooth).get(f"tooth_{fdi}_{f}"), dict)
                   for f in facts):
                covered[category].add(fdi)
    return covered


def summary_claims(summary: Dict) -> Dict[str, Dict[int, str]]:
    """{category: {fdi: value}} from the post-vote summary blocks."""
    kept: Dict[str, Dict[int, str]] = {c: {} for c in CATEGORIES}
    for arch in ("mandible", "maxilla"):
        block = _as_dict(summary.get(arch))

        for entry in block.get("impacted_teeth") or []:
            kept["impaction"][entry["tooth"]] = entry.get("impacted", "unspecified")

        endo = _as_dict(block.get("endodontic_summary"))
        quality = {fdi: q for q, teeth in
                   _as_dict(endo.get("quality_groups")).items() for fdi in teeth}
        for fdi in endo.get("teeth") or []:
            kept["endodontic"][fdi] = quality.get(fdi, "unspecified")

        groups = _as_dict(_as_dict(block.get("restoration_summary")).get("groups"))
        for group, teeth in groups.items():
            if group in RESTORATION_VALUE:
                for fdi in teeth:
                    kept[RESTORATION_VALUE[group]][fdi] = group
            # The coarse row, same rule as the PRED side in pred_claims: a
            # kept crown or filling is a kept "restoration". Without this the
            # row would read SUM 0 on every run and look like a vote that
            # threw everything away, when in fact the summary has no
            # restoration group of its own to read.
            if group in ("crown", "fillings"):
                for fdi in teeth:
                    kept["restoration"][fdi] = group

        # Caries rides in the per-tooth morphology block and is written only
        # when true; root remnants have a list of their own, which is also
        # where their resorption clause lives.
        for entry in block.get("tooth_findings") or []:
            fdi = entry.get("fdi")
            if not isinstance(fdi, int):
                continue
            if str(_as_dict(entry.get("morphology")).get("caries")).lower() == "true":
                kept["caries"][fdi] = "unspecified"
        for entry in block.get("root_remnants") or []:
            fdi = _as_dict(entry).get("fdi")
            if isinstance(fdi, int):
                kept["root_remnant"][fdi] = "unspecified"
    return kept


def summary_dropped(summary: Dict) -> Dict[str, List[int]]:
    """{category: teeth} the cross-source vote threw away."""
    key_to_category = {"impacted": "impaction",
                       "root_canal_treatment": "endodontic",
                       "restorations": None}   # not split by type; see below
    dropped: Dict[str, List[int]] = {c: [] for c in CATEGORIES}
    for arch in ("mandible", "maxilla"):
        cross = _as_dict(_as_dict(_as_dict(summary.get(arch)).get("arch_findings"))
                         .get("cross_source_dropped"))
        for key, teeth in cross.items():
            category = key_to_category.get(key)
            if category:
                dropped[category].extend(teeth)
            elif key == "restorations":
                # One bucket for crown/post_and_core/fillings together --
                # postprocess_pred.py does not record which type was dropped.
                dropped.setdefault("restorations", []).extend(teeth)
    return dropped


# ---------------------------------------------------------------------------
# Survey
# ---------------------------------------------------------------------------

def _version_key(name: str) -> List[int]:
    """Sort key for a 'predictions_v6.10'-style directory name.

    Numeric, component by component, so v6.10 sorts ABOVE v6.9 rather than
    below it the way a plain string compare would put it.
    """
    return [int(n) for n in re.findall(r"\d+", name)] or [-1]


def _resolve_stage_dir(parent: Path, prefix: str, pattern: str) -> Path:
    """Find the directory actually holding a stage's files.

    aksssr_pipeline.sh nests each stage under a schema-version tag --
    predictions/predictions_v6.9/ rather than predictions/ -- so that a v6.9
    run cannot land on top of a v6.4 one. Older arms, which are not being
    re-run, still use the flat layout. Accept both rather than pick one and
    fail on the other: flat first, then the newest-versioned subdirectory that
    is actually non-empty (an empty one is skipped, not chosen and then
    reported as having no cases). Falls back to `parent`, so the caller's own
    error message names the directory the user expected.
    """
    if parent.is_dir() and any(parent.glob(pattern)):
        return parent
    subs = sorted((d for d in parent.glob(f"{prefix}_*") if d.is_dir()),
                  key=lambda d: _version_key(d.name), reverse=True)
    for d in subs:
        if any(d.glob(pattern)):
            return d
    return parent


def load_cases(run_dir: Path, case_ids: Optional[List[str]],
               label_space: bool, facts_dir: Optional[Path] = None,
               no_gt: bool = False) -> List[Dict]:
    pred_dir = _resolve_stage_dir(run_dir / "predictions",
                                  "predictions", "*_pred.json")
    # Summaries must come from the SAME version as the predictions. Pairing
    # v6.9 predictions with v6.4 summaries would silently survey two arms at
    # once, which is the exact failure the version nesting exists to prevent,
    # so this mirrors the tag rather than resolving independently.
    tag = (pred_dir.name[len("predictions_"):]
           if pred_dir.name.startswith("predictions_") else "")
    if tag:
        summary_dir = run_dir / "summaries" / f"summaries_{tag}"
    else:
        summary_dir = run_dir / "summaries"
    if not summary_dir.is_dir():
        summary_dir = _resolve_stage_dir(run_dir / "summaries",
                                         "summaries", "*_summary.json")
    if tag:
        print(f"[INFO] predictions: {pred_dir}", file=sys.stderr)
        print(f"[INFO] summaries  : {summary_dir}", file=sys.stderr)

    pred_files = sorted(pred_dir.glob("*_pred.json"))
    if case_ids:
        pred_files = [p for p in pred_files
                      if p.stem.replace("_pred", "") in case_ids]
    if not pred_files:
        sys.exit(f"[FAIL] no *_pred.json under {pred_dir}")

    cases = []
    for pred_path in pred_files:
        case_id = pred_path.stem.replace("_pred", "")
        summary_path = summary_dir / f"{case_id}_summary.json"
        if not summary_path.exists():
            print(f"[{case_id}] [SKIP] no summary at {summary_path}", file=sys.stderr)
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        pred = json.loads(pred_path.read_text(encoding="utf-8"))
        cases.append({
            "case_id": case_id,
            # Which teeth actually carry the composite fact behind each
            # category. Every tooth gets a `tooth_{fdi}` entry, but only ~21
            # of 32 per case carry the facts themselves -- and crown is
            # reported ONLY by the composite read (the panoramic arch fact
            # has no crown value), so a tooth missing that fact is a crown
            # the pipeline could not have found however well it read.
            "read": composite_coverage(pred),
            "legacy": uses_legacy_shape(pred),
            "pred": pred_claims(pred),
            "summary": summary_claims(summary),
            "dropped": summary_dropped(summary),
            "gt": {c: resolve_gt(case_id, c, label_space, no_gt)
                   for c in SCORED_CATEGORIES},
        })

        pred_implants, pred_bridge = pred_prosthetics(pred)
        summary_implants, summary_bridge = summary_prosthetics(summary)
        # One read of the facts file feeds four columns: implants, bridges,
        # canal-adjacent teeth, and the absent-teeth label.
        facts_implants, facts_bridge = set(), False
        facts_ian: Set[int] = set()
        facts_gone: Set[int] = set()
        facts_enumerated: Set[int] = set()
        if facts_dir:
            facts_path = facts_dir / f"{case_id}.json"
            if facts_path.exists():
                facts = json.loads(facts_path.read_text(encoding="utf-8"))
                facts_implants, facts_bridge = facts_prosthetics(facts)
                facts_gone, facts_enumerated = facts_absent(facts)
                structured = _as_dict(facts.get("structured"))
                facts_ian = {f for f in as_list(structured.get("ian_close_teeth"))
                             if isinstance(f, int)}
            else:
                print(f"[{case_id}] [WARN] no facts file at {facts_path}",
                      file=sys.stderr)
        cases[-1].update(facts_implants=facts_implants, facts_bridge=facts_bridge,
                         pred_implants=pred_implants, pred_bridge=pred_bridge,
                         summary_implants=summary_implants,
                         summary_bridge=summary_bridge,
                         facts_ian_close=facts_ian,
                         facts_absent=facts_gone,
                         facts_enumerated=facts_enumerated,
                         pred_absent=pred_absent(pred),
                         summary_absent=summary_absent(summary),
                         pred_anatomy=pred_anatomy(pred),
                         summary_anatomy=summary_anatomy(summary))
    return cases


def score(claimed: Set[int], positive: Set[int],
          unscored: Set[int]) -> Tuple[Set[int], Set[int], Set[int], Set[int]]:
    """-> (true positives, false positives, false negatives, unscorable)."""
    skipped = claimed & unscored
    scored = claimed - unscored
    return scored & positive, scored - positive, positive - claimed, skipped


def ratio(num: int, den: int) -> str:
    return f"{num / den:.2f}" if den else "n/a"


def counts(counter: Counter, *keys: str) -> Dict[str, int]:
    """Counter -> plain dict with EVERY key present, zeros included.

    dict(Counter) omits the counts that never happened, which makes a real
    zero indistinguishable from a field that was never measured -- in the
    JSON, and as a KeyError in the printers that read these back.
    """
    return {k: counter[k] for k in keys}


IMPLANT_KEYS = ("claims", "tp", "fp", "fn")
BRIDGE_KEYS = ("tp", "fp", "fn", "tn")
ABSENT_KEYS = ("claims", "tp", "fp", "fn", "unread", "fn_unread", "tp_mirrored")
CANAL_KEYS = ("scored", "right", "conflated", "wrong", "unread", "unanswerable")
IAN_KEYS = ("claims", "tp", "fp", "fn", "tp_mirrored")
SINUS_KEYS = ("scored", "right", "wrong", "unread", "abnormal_gt",
              "abnormal_found")


def survey(cases: List[Dict], categories: List[str], per_case: bool,
           no_gt: bool = False) -> Tuple[Dict, Dict]:
    """-> (totals for the JSON, view data for print_report).

    Nothing is printed here. The per-case dump is collected instead, so the
    summary table can be printed ahead of every detail block -- see
    print_overview.
    """
    out = {}
    lines: List[str] = []
    for category in categories:
        totals = Counter()
        source_stats: Dict[str, List[int]] = {}
        retired_stats: Dict[str, List[int]] = {}
        agreement = Counter()

        if per_case:
            header = (f"[{category}]  {'case':7} {'PRED':46} {'SUMMARY':34}"
                      if no_gt else
                      f"[{category}]  {'case':7} {'PRED':46} {'SUMMARY':34} "
                      f"{'REPORT GT':22} TP/FP/FN")
            lines += ["", header, "-" * min(len(header), 160)]

        for case in cases:
            positive, unscored = case["gt"][category]
            raw = case["pred"][category]
            live = {fdi: {s: v for s, v in srcs.items()
                          if (category, s) not in RETIRED_SOURCES}
                    for fdi, srcs in raw.items()}
            pred = {fdi: srcs for fdi, srcs in live.items() if srcs}
            for fdi, srcs in raw.items():
                for source in srcs:
                    if (category, source) in RETIRED_SOURCES and fdi not in unscored:
                        stat = retired_stats.setdefault(source, [0, 0])
                        stat[0] += 1
                        stat[1] += fdi in positive
            kept = case["summary"][category]

            p_tp, p_fp, p_fn, p_skip = score(set(pred), positive, unscored)
            s_tp, s_fp, s_fn, s_skip = score(set(kept), positive, unscored)

            totals["gt"] += len(positive)
            totals["cases_with_gt"] += bool(positive)
            totals["pred"] += len(pred)
            totals["pred_tp"] += len(p_tp)
            totals["pred_fp"] += len(p_fp)
            totals["pred_skip"] += len(p_skip)
            totals["summary"] += len(kept)
            totals["summary_tp"] += len(s_tp)
            totals["summary_fp"] += len(s_fp)
            totals["summary_fn"] += len(s_fn)
            totals["summary_skip"] += len(s_skip)
            dropped = case["dropped"].get(category, [])
            totals["dropped"] += len(dropped)
            totals["dropped_true"] += sum(1 for f in dropped if f in positive)

            for fdi, sources in pred.items():
                if fdi in unscored:
                    continue
                agreement[(len(sources), fdi in positive)] += 1
                for source in sources:
                    stat = source_stats.setdefault(source, [0, 0])
                    stat[0] += 1
                    stat[1] += fdi in positive

            if per_case:
                fmt = lambda t: ", ".join(str(f) for f in sorted(t)) or "-"
                indent = " " * (len(category) + 4)
                line = (f"{indent}{case['case_id']:7} "
                        f"{fmt(pred):46} {fmt(kept):34}")
                if not no_gt:
                    line += (f" {fmt(positive):22} "
                             f"{len(s_tp)}/{len(s_fp)}/{len(s_fn)}")
                lines.append(line)

        out[category] = dict(totals)
        out[f"{category}_sources"] = {s: v for s, v in source_stats.items()}
        out[f"{category}_retired"] = {s: v for s, v in retired_stats.items()}
        out[f"{category}_agreement"] = {f"{n}_{'true' if t else 'false'}": c
                                        for (n, t), c in agreement.items()}
    return out, {"per_case": lines}


def print_legacy_note(cases: List[Dict], categories: List[str],
                      results: Dict) -> None:
    """The one caveat that does not depend on ground truth being available."""
    # A SUMMARY column far below its PRED column, on predictions that predate
    # the current schema, is a version mismatch rather than a postprocess
    # regression -- see uses_legacy_shape. Stated here because the two look
    # identical in the table and only one of them is worth investigating.
    legacy = [c["case_id"] for c in cases if c["legacy"]]
    starved = [cat for cat in categories
               if results[cat]["pred"] and not results[cat]["summary"]]
    if not legacy:
        return
    print()
    print(f"  NOTE: {len(legacy)}/{len(cases)} predictions predate schema "
          f"v6.4 -- they carry tooth_{{fdi}}_restoration rather than the\n"
          f"        morphology bools. The PRED column reads both shapes, "
          f"but postprocess_pred.py reads only the\n"
          f"        current one, so any summary rebuilt from these is "
          f"empty for the restorative findings"
          + (f" ({', '.join(starved)} scored 0 here)" if starved else "")
          + ".\n        Re-run inference against the current schema; "
            "re-running postprocess alone cannot fix it.")


GT_DERIVED_TOTALS = ("gt", "cases_with_gt", "pred_tp", "pred_fp", "pred_skip",
                     "summary_tp", "summary_fp", "summary_fn", "summary_skip",
                     "dropped_true")


def strip_gt_from_totals(results: Dict, categories: List[str]) -> Dict:
    """Drop every GT-derived field from a --no-gt result set.

    survey() computes them unconditionally against empty ground truth, so
    they come out zero. Zero in a saved JSON is indistinguishable from a
    real score of zero, hence removal rather than serialisation.
    """
    out = dict(results)
    for category in categories:
        out[category] = {k: v for k, v in out[category].items()
                         if k not in GT_DERIVED_TOTALS}
        for suffix in ("_sources", "_retired"):
            key = f"{category}{suffix}"
            # [claims, correct] -> claims
            out[key] = {s: n for s, (n, _correct) in out[key].items()}
        key = f"{category}_agreement"
        collapsed = Counter()
        for label, count in out[key].items():
            collapsed[label.split("_")[0]] += count
        out[key] = dict(collapsed)
    return out


def print_report_no_gt(cases: List[Dict], categories: List[str],
                       results: Dict, view: Dict) -> None:
    """Claim counts only -- the survey with every scored column removed."""
    print()
    print(f"{len(cases)} cases, NO ground truth (--no-gt): claim counts only")
    print()
    head = (f"{'finding':14} {'PRED':>6} {'SUM':>6} {'kept':>6} "
            f"{'dropped':>8}")
    print(head)
    print("-" * len(head))
    for category in categories:
        t = results[category]
        print(f"{category:14} {t['pred']:6} {t['summary']:6} "
              f"{ratio(t['summary'], t['pred']):>6} {t['dropped']:8}")
    print()
    print("PRED = teeth claimed by any source.  SUM = teeth surviving the "
          "cross-source vote.\nkept = SUM/PRED.  dropped = what postprocess "
          "recorded as discarded by the vote; it is\n       its own counter, "
          "NOT PRED minus SUM (a claim can also vanish by regrouping).\n"
          "Nothing here is scored for correctness -- no ground truth exists "
          "outside the 40\nvalidate cases.")

    print_legacy_note(cases, categories, results)

    for category in categories:
        sources = results[f"{category}_sources"]
        agreement = results[f"{category}_agreement"]
        t = results[category]
        print()
        print(f"[{category}] {t['pred']} claims across {len(cases)} cases")
        for source, (n, _correct) in sorted(sources.items()):
            print(f"  source {source:10} {n:4} claims")
        for source, (n, _correct) in sorted(results[f"{category}_retired"].items()):
            print(f"  RETIRED {source:10} {n:4} claims"
                  f"  -- fact dropped from the schema, not consumed")
        if agreement:
            # Collapse the (n_sources, is_true) key onto n_sources alone:
            # the truth half is meaningless without ground truth.
            by_n = Counter()
            for key, count in agreement.items():
                by_n[int(key.split("_")[0])] += count
            parts = [f"{n} source(s): {c}" for n, c in sorted(by_n.items())]
            print(f"  by agreement -- {';  '.join(parts)}")

    for line in view["per_case"]:
        print(line)


def _pr(totals: Dict, n: int, tp: str = "tp", fp: str = "fp") -> str:
    """A prec/rec pair as one cell, matching the "prec/rec" metric label.

    Keyed off the totals dict rather than two ints because these come from
    Counters that were dict()ed for the JSON, so a count that never happened
    is a missing key rather than a zero.
    """
    hit, miss = totals.get(tp, 0), totals.get(fp, 0)
    return f"{ratio(hit, hit + miss)}/{ratio(hit, n)}"


def _acc(totals: Dict) -> str:
    """right/scored as one cell, for the axes with no precision to report."""
    return ratio(totals.get("right", 0), totals.get("scored", 0))


def print_overview(cases: List[Dict], categories: List[str], results: Dict,
                   label_space: bool, absent: Optional[Dict],
                   prosthetics: Optional[Dict], anatomy: Optional[Dict],
                   anatomy_view: Optional[Dict]) -> None:
    """Every scored axis in the survey, on one line each, before the detail.

    One table rather than four means the two questions this file exists to
    answer -- where the pipeline stands, and what the last flag change moved --
    are answered without reading to the bottom. Nothing is computed here: each
    row is read off the same totals its own section prints below, so the
    summary cannot drift from the detail.
    """
    rows: List[Tuple[str, str, int, str, str, str, str]] = []

    # Absent teeth first: everything else is a finding ON a tooth, so which
    # positions hold a tooth at all is the claim the rest are conditioned on.
    if absent and absent["cases"]:
        rows.append(("absent teeth", "vs segmentation mask", absent["gt"],
                     "teeth", "prec/rec",
                     _pr(absent["pred"], absent["gt"]),
                     _pr(absent["summary"], absent["gt"])))
    for category in categories:
        t = results[category]
        rows.append(("findings", category, t["gt"], "teeth", "prec/rec",
                     _pr(t, t["gt"], "pred_tp", "pred_fp"),
                     _pr(t, t["gt"], "summary_tp", "summary_fp")))
    if prosthetics:
        n_slots, n_bridge = prosthetics["implant_slots"], prosthetics["bridge_cases"]
        i, b = prosthetics["implants"], prosthetics["bridges"]
        rows.append(("prosthetics", "implants", n_slots, "slots", "prec/rec",
                     _pr(i["pred"], n_slots), _pr(i["summary"], n_slots)))
        rows.append(("prosthetics", "fixed bridges", n_bridge, "cases", "prec/rec",
                     _pr(b["pred"], n_bridge), _pr(b["summary"], n_bridge)))
    if anatomy and anatomy_view:
        canal, sinus = anatomy["canal"], anatomy["sinus"]
        ian, intra = anatomy["ian_close"], anatomy["intrasinusal"]
        rows.append(("anatomy", "canal position", canal["pred"].get("scored", 0),
                     "sides", "acc", _acc(canal["pred"]), _acc(canal["summary"])))
        n_ian = anatomy_view["ian_teeth"]
        rows.append(("anatomy", "canal-adjacent teeth", n_ian, "teeth", "prec/rec",
                     _pr(ian["pred"], n_ian), _pr(ian["summary"], n_ian)))
        rows.append(("anatomy", "sinus mucosa", anatomy_view["sinus_sides"],
                     "sides", "acc", _acc(sinus["pred"]), _acc(sinus["summary"])))
        n_intra = anatomy_view["intrasinusal_teeth"]
        rows.append(("anatomy", "intrasinusal roots", n_intra, "teeth", "prec/rec",
                     _pr(intra["pred"], n_intra), _pr(intra["summary"], n_intra)))

    print()
    print(f"{len(cases)} cases, ground truth in {gt_space(label_space)}")
    print()
    print("SUMMARY -- every scored axis; each section below breaks its own rows down")
    head = (f"{'section':13} {'axis':21} {'N':>4} {'unit':6} {'metric':9} "
            f"{'PRED':11} {'SUMMARY':11}").rstrip()
    print(head)
    print("-" * len(head))
    for section, axis, n, unit, metric, pred, summary in rows:
        print(f"{section:13} {axis:21} {n:4} {unit:6} {metric:9} "
              f"{pred:11} {summary:11}".rstrip())
    print()
    print("N       = what the label asserts -- teeth, implant slots, sides or cases.\n"
          "PRED    = the raw VQA reads, unioned over every source that can assert it.\n"
          "SUMMARY = what survived postprocess_pred.py's cross-source vote, i.e. what\n"
          "          can still reach the synthesized report.\n"
          "prec/rec are against N; acc = right/scored over the sides that stage\n"
          "answered at all. The label is the reference reports on every row except\n"
          "absent teeth, which no report enumerates -- that one is against the masks.\n"
          "The `facts` stage -- what the segmentation alone asserts -- is a third\n"
          "column in the sections below, on the axes where it can make the claim.")

    print_legacy_note(cases, categories, results)


def gt_space(label_space: bool) -> str:
    """How the ground truth is oriented, for the header line.

    With LR_INVERTED_REPORTS empty, --label-space mirrors nothing, so this
    must not claim otherwise -- it prints into the survey file that is the
    record of how a number was produced.
    """
    if label_space and LR_INVERTED_REPORTS:
        return "label space (inverted reports mirrored)"
    if label_space:
        return "report space (--label-space passed, but no cases are marked inverted)"
    return "report space"


def print_report(cases: List[Dict], categories: List[str], results: Dict,
                 view: Dict) -> None:
    print()
    print("=" * 78)
    print("FINDINGS -- per-tooth claims, raw prediction vs cross-source vote, "
          "against the reports")
    print()
    head = (f"{'finding':14} {'GT':>4} {'PRED':>5} {'tp':>4} {'fp':>4} {'prec':>5} "
            f"{'rec':>5} | {'SUM':>5} {'tp':>4} {'fp':>4} {'prec':>5} {'rec':>5} "
            f"| {'skip':>4}")
    print(head)
    print("-" * len(head))
    for category in categories:
        t = results[category]
        print(f"{category:14} {t['gt']:4} {t['pred']:5} {t['pred_tp']:4} "
              f"{t['pred_fp']:4} {ratio(t['pred_tp'], t['pred_tp'] + t['pred_fp']):>5} "
              f"{ratio(t['pred_tp'], t['gt']):>5} | "
              f"{t['summary']:5} {t['summary_tp']:4} {t['summary_fp']:4} "
              f"{ratio(t['summary_tp'], t['summary_tp'] + t['summary_fp']):>5} "
              f"{ratio(t['summary_tp'], t['gt']):>5} | "
              f"{t['pred_skip']:4}")
    print()
    print("GT   = teeth the reports assert.  PRED/SUM = teeth claimed at each "
          "stage.  skip = claims\n       on unscored teeth (implant crowns, "
          "pontics, findings the report could not resolve),\n       counted "
          "as neither right nor wrong.")

    for category in categories:
        sources = results[f"{category}_sources"]
        agreement = results[f"{category}_agreement"]
        t = results[category]
        print()
        print(f"[{category}] {t['gt']} teeth across {t['cases_with_gt']} cases")
        for source, (n, correct) in sorted(sources.items()):
            print(f"  source {source:10} {n:4} claims  {correct:4} correct  "
                  f"precision {ratio(correct, n)}")
        for source, (n, correct) in sorted(results[f"{category}_retired"].items()):
            print(f"  RETIRED {source:10} {n:4} claims  {correct:4} correct  "
                  f"precision {ratio(correct, n)}  "
                  f"-- fact dropped from the schema, not consumed")
        if agreement:
            parts = [f"{k.replace('_', ' source(s), ')}: {v}"
                     for k, v in sorted(agreement.items())]
            print(f"  by agreement -- {';  '.join(parts)}")
        if t["dropped"]:
            print(f"  cross_source_dropped: {t['dropped']} discarded by the vote, "
                  f"{t['dropped_true']} of them true")
        missed = [(c["case_id"], f) for c in cases
                  for f in c["gt"][category][0] if f not in c["pred"][category]]
        by_case = {c["case_id"]: c["read"][category] for c in cases}
        unread = [(c, f) for c, f in missed if f not in by_case[c]]
        print(f"  never claimed by any source: {len(missed)}"
              + (f" -- {', '.join(f'{c}/{f}' for c, f in missed[:14])}"
                 + (" ..." if len(missed) > 14 else "") if missed else ""))
        if unread:
            print(f"  ...of which {len(unread)} got no composite crop at all, "
                  f"so no per-tooth fact was ever asked about them")

    for line in view["per_case"]:
        print(line)


# ---------------------------------------------------------------------------
# Absent teeth -- the one axis scored in MASK space, not report space
#
# No report enumerates all 32 positions: they name the absences worth
# mentioning and stay silent about the rest, so "the report does not say 15 is
# there" is not a claim that it is missing. The segmentation does enumerate all
# 32 (facts.structured.teeth_present + teeth_absent), and it is the very thing
# every image in this pipeline is rendered from -- so it is the label that can
# actually settle whether the VLM read the dentition it was shown. Scored
# against the masks for that reason, and labelled as such everywhere it is
# printed, because every other number in this survey is against the reports.
# ---------------------------------------------------------------------------

def facts_absent(facts: Dict) -> Tuple[Set[int], Set[int]]:
    """(absent positions, positions this facts file enumerates at all)."""
    structured = _as_dict(facts.get("structured"))
    present = {f for f in as_list(structured.get("teeth_present"))
               if isinstance(f, int)}
    absent = {f for f in as_list(structured.get("teeth_absent"))
              if isinstance(f, int)}
    # Deciduous FDIs (5x-8x) appear in these lists for the mixed-dentition
    # cases; the arch fact only answers the permanent positions.
    return absent & ALL_TEETH, (present | absent) & ALL_TEETH


def pred_absent(pred: Dict) -> Dict:
    """Absent-tooth claims in one raw prediction, per source.

    Two sources can assert absence, and neither is the composite: the
    panoramic arch read answers all sixteen positions per arch with `absent`
    as an explicit value, and the four wisdom-tooth 3D renders answer
    eruption_state `absent`. The composite per-tooth facts cannot say it at
    all -- build_vqa_pairs only expands them for teeth the segmentation
    already places, so their silence at a position is the pipeline's own
    doing rather than a read, and counting it as a claim would score the
    mask against itself.
    """
    facts = _as_dict(pred.get("global"))
    by_source: Dict[str, Set[int]] = {"panoramic": set(), "3d_render": set()}
    read: Set[int] = set()
    for arch, positions in ARCH_TEETH.items():
        findings = _as_dict(_as_dict(facts.get(f"dental_arch_findings_{arch}"))
                            .get("findings"))
        for fdi in positions:
            value = findings.get(str(fdi), findings.get(fdi))
            if not isinstance(value, str):
                continue                   # position never answered
            read.add(fdi)
            if value.strip().lower() == "absent":
                by_source["panoramic"].add(fdi)
    for fact_name, fdi in WISDOM_SLOT.items():
        state = _as_dict(facts.get(fact_name)).get("eruption_state")
        if isinstance(state, str):
            read.add(fdi)
            if state.strip().lower() == "absent":
                by_source["3d_render"].add(fdi)
    return {"absent": by_source["panoramic"] | by_source["3d_render"],
            "by_source": by_source, "read": read}


def summary_absent(summary: Dict) -> Dict:
    """What postprocess's absent_teeth block asserts, per arch.

    Derived from `present` wherever it exists rather than from `teeth`: the
    block omits `teeth` when the pattern is "none", so reading `teeth` alone
    would turn a full arch into an unanswered one. An arch whose pattern
    could not be settled ("unknown") carries neither key -- that is sixteen
    UNREAD positions, not sixteen teeth called present.
    """
    absent, read = set(), set()
    for arch, positions in ARCH_TEETH.items():
        block = _as_dict(_as_dict(_as_dict(summary.get(arch))
                                  .get("arch_findings")).get("absent_teeth"))
        listed = {f for f in as_list(block.get("teeth")) if isinstance(f, int)}
        if "present" in block:
            present = {f for f in as_list(block.get("present"))
                       if isinstance(f, int)}
            read |= set(positions)
            absent |= (set(positions) - present) | listed
        elif listed:
            read |= set(positions)
            absent |= listed
    return {"absent": absent & ALL_TEETH, "read": read}


def survey_absent(cases: List[Dict], per_case: bool) -> Tuple[Dict, Dict]:
    """-> (totals for the JSON, view data for print_absent)."""
    stages = ("pred", "summary")
    stat = {s: Counter() for s in stages}
    sources: Dict[str, List[int]] = {}
    scored_cases = 0
    n_absent = n_positions = 0
    worst: List[Tuple[int, str, List[int], List[int]]] = []
    lines: List[str] = []

    # Counts and disagreements only: an edentulous case lists 32 FDIs per
    # column, which no per-case table can hold and nobody reads anyway.
    def fdis(teeth: Set[int]) -> str:
        shown = sorted(teeth)
        return (",".join(str(f) for f in shown[:8])
                + (" ..." if len(shown) > 8 else "")) or "-"

    if per_case:
        header = (f"{'case':7} {'mask':>4} {'PRED':>5} {'SUM':>4}  "
                  f"{'summary wrongly absent':34} summary missed")
        lines += ["", header, "-" * len(header)]

    for case in cases:
        gt, enumerated = case["facts_absent"], case["facts_enumerated"]
        if not enumerated:
            continue        # no facts file for this case: nothing to score against
        scored_cases += 1
        n_absent += len(gt)
        n_positions += len(enumerated)

        for stage in stages:
            claim = case[f"{stage}_absent"]
            claimed = claim["absent"] & enumerated
            tp, fp, fn, _ = score(claimed, gt, set())
            unread = enumerated - claim["read"]
            stat[stage]["claims"] += len(claimed)
            stat[stage]["tp"] += len(tp)
            stat[stage]["fp"] += len(fp)
            stat[stage]["fn"] += len(fn)
            stat[stage]["unread"] += len(unread)
            stat[stage]["fn_unread"] += len(fn & unread)
            # Laterality, as for implants: 46 vs 36 is one mirror apart.
            mirrored, _, _, _ = score({mirror_fdi(f) for f in claimed}, gt, set())
            stat[stage]["tp_mirrored"] += len(mirrored)

        for source, claimed in case["pred_absent"]["by_source"].items():
            stat_source = sources.setdefault(source, [0, 0])
            scored = claimed & enumerated
            stat_source[0] += len(scored)
            stat_source[1] += len(scored & gt)

        kept = case["summary_absent"]["absent"] & enumerated
        if kept ^ gt:
            worst.append((len(kept ^ gt), case["case_id"],
                          sorted(kept - gt), sorted(gt - kept)))
        if per_case:
            claimed = case["pred_absent"]["absent"] & enumerated
            lines.append(f"{case['case_id']:7} {len(gt):4} {len(claimed):5} "
                         f"{len(kept):4}  {fdis(kept - gt):34} {fdis(gt - kept)}")

    results = {"gt": n_absent, "positions": n_positions, "cases": scored_cases,
               "sources": dict(sources),
               **{s: counts(stat[s], *ABSENT_KEYS) for s in stages}}
    view = {"worst": sorted(worst, reverse=True)[:6], "per_case": lines}
    return results, view


def print_absent(results: Dict, view: Dict) -> None:
    print()
    print("=" * 78)
    print("ABSENT TEETH -- which positions hold no tooth, against the "
          "SEGMENTATION masks")
    print()
    print(f"{results['gt']} positions are empty in the masks, of "
          f"{results['positions']} enumerated across {results['cases']} cases")
    head = (f"  {'source':9} {'claims':>7} {'tp':>4} {'fp':>4} {'fn':>4} "
            f"{'unread':>7} {'prec':>6} {'rec':>6} {'tp mirrored':>12}")
    print(head)
    for stage in ("pred", "summary"):
        t = results[stage]
        print(f"  {stage:9} {t['claims']:7} {t['tp']:4} {t['fp']:4} {t['fn']:4} "
              f"{t['unread']:7} {ratio(t['tp'], t['tp'] + t['fp']):>6} "
              f"{ratio(t['tp'], results['gt']):>6} {t['tp_mirrored']:>12}")
    for source, (claims, correct) in sorted(results["sources"].items()):
        print(f"  source {source:10} {claims:4} claims  {correct:4} correct  "
              f"precision {ratio(correct, claims)}")
    for stage in ("pred", "summary"):
        t = results[stage]
        if t["fn_unread"]:
            print(f"  {stage}: {t['fn_unread']} of its {t['fn']} misses sit on "
                  f"positions it never answered at all")
    if view["worst"]:
        def fmt(teeth: List[int]) -> str:
            shown = ",".join(str(f) for f in teeth[:8])
            return (shown + (" ..." if len(teeth) > 8 else "")) or "none"
        print("  worst cases (summary vs mask):")
        for _n, cid, extra, gone in view["worst"]:
            print(f"    {cid:7} wrongly absent {fmt(extra):32} "
                  f"missed {fmt(gone)}")
    print("  label = facts/<case>.json structured.teeth_absent, off the same mask "
          "the images are\n          rendered from. unread = positions the stage "
          "never answered at all; the\n          absent ones among those count as "
          "misses, since a report written from them\n          says nothing about "
          "that position either way.")
    # Two ways the mask reaches these numbers other than as the label. Neither
    # is a defect to fix -- both are how the pipeline is built -- but without
    # them on the page a transcription reads as perception, and postprocess
    # looks like it is correcting a read when it is partly consulting the mask.
    print("  CAVEATS -- the mask is not only the label on this axis:")
    print("    * on a run generated WITH facts, the panoramic caption already "
          "names the absent\n      teeth, so PRED scores a transcription as much "
          "as a read; NO_FACTS=1 is the arm\n      that separates the two.")
    print("    * SUMMARY is not a pure read either: collect_arch_absent's "
          "strongest source is\n      `detected == no_image`, i.e. whether the "
          "segmentation produced a composite crop\n      at that position, so "
          "postprocess recovers part of this label by construction.\n      PRED "
          "here is deliberately restricted to the two sources that are reads.")
    for line in view["per_case"]:
        print(line)


# ---------------------------------------------------------------------------
# Prosthetics survey
# ---------------------------------------------------------------------------

def facts_prosthetics(facts: Dict) -> Tuple[Set[int], bool]:
    """(implant positions, bridge present) from a masks-derived facts file."""
    structured = _as_dict(facts.get("structured"))
    # The list carries a stray None for at least one case (A018).
    implants = {f for f in as_list(structured.get("implants")) if isinstance(f, int)}
    return implants, bool(structured.get("bridge_present"))


def pred_prosthetics(pred: Dict) -> Tuple[Set[int], bool]:
    facts = _as_dict(pred.get("global"))
    implants, bridge = set(), False
    for arch in ("mandible", "maxilla"):
        for entry in as_list(_as_dict(facts.get(f"implants_{arch}")).get("implants")):
            fdi = _as_dict(entry).get("fdi_number")
            if isinstance(fdi, int):
                implants.add(fdi)
        bridges = _as_dict(facts.get(f"fixed_bridges_{arch}"))
        bridge = bridge or bool(bridges.get("present")) or bool(bridges.get("bridges"))
    return implants, bridge


def summary_prosthetics(summary: Dict) -> Tuple[Set[int], bool]:
    implants, bridge = set(), False
    for arch in ("mandible", "maxilla"):
        block = _as_dict(_as_dict(summary.get(arch)).get("prosthetics"))
        for entry in as_list(block.get("implants")):
            fdi = _as_dict(entry).get("fdi_number")
            if isinstance(fdi, int):
                implants.add(fdi)
        bridge = bridge or bool(as_list(block.get("bridges")))
    return implants, bridge


_IMPLANT_RE = re.compile(r"implant", re.I)
# "no dental stump or implant supporting the crown" (F067) denies one.
_IMPLANT_DENIAL_RE = re.compile(r"no (dental stump or )?implant|absence of "
                                r"prosthetic restorations", re.I)
_BRIDGE_RE = re.compile(r"bridge|splinted|soldered|circular fixed|cantilever"
                        r"|fixed prosth", re.I)


def audit_prosthetics(reports_dir: Path, case_ids: List[str]) -> List[str]:
    """Same drift check as audit_gt_table, for IMPLANT_GT and BRIDGE_GT."""
    warnings = []
    for case_id in case_ids:
        text = " ".join(p.read_text(encoding="utf-8") for p in report_paths(reports_dir, case_id))
        sentences = split_sentences(text)
        mentions_implant = any(_IMPLANT_RE.search(s) and
                               not _IMPLANT_DENIAL_RE.search(s) for s in sentences)
        mentions_bridge = any(_BRIDGE_RE.search(s) for s in sentences)
        if mentions_implant and not IMPLANT_GT.get(case_id) \
                and case_id not in IMPLANT_NOTES:
            warnings.append(f"{case_id}/implant: reports mention one but "
                            f"IMPLANT_GT is empty")
        if IMPLANT_GT.get(case_id) and not mentions_implant:
            warnings.append(f"{case_id}/implant: IMPLANT_GT has "
                            f"{len(IMPLANT_GT[case_id])} but no report says so")
        if case_id in BRIDGE_UNSCORED:
            continue
        if mentions_bridge and not BRIDGE_GT.get(case_id):
            warnings.append(f"{case_id}/bridge: reports mention bridge/splinted "
                            f"work but BRIDGE_GT says no")
        if BRIDGE_GT.get(case_id) and not mentions_bridge:
            warnings.append(f"{case_id}/bridge: BRIDGE_GT says yes but no "
                            f"report sentence mentions one")
    return warnings


def score_slots(claimed: Set[int],
                slots: List[Set[int]]) -> Tuple[int, int, int]:
    """(found slots, false positives, missed slots) under slot semantics.

    Each slot is one implant and the positions that would satisfy it. A slot
    is found if any claim lands in it; each slot consumes at most one claim,
    so three claims inside one region slot score one hit and two false
    positives rather than three hits.
    """
    unclaimed = set(claimed)
    found = 0
    for slot in slots:
        hit = sorted(unclaimed & slot)
        if hit:
            found += 1
            unclaimed.discard(hit[0])
    return found, len(unclaimed), len(slots) - found


def survey_prosthetics(cases: List[Dict], per_case: bool) -> Tuple[Dict, Dict]:
    """Three-way implant and fixed-bridge comparison: facts, pred, summary.

    -> (totals for the JSON, view data for print_prosthetics). Prints nothing,
    so the summary table can come first; see print_overview.
    """
    stages = ("facts", "pred", "summary")
    implant = {s: Counter() for s in stages}
    # The same claims scored again with every FDI mirrored across the midline.
    # Not a fix -- a measurement of how much of the error is pure laterality.
    implant_mirrored = {s: Counter() for s in stages}
    bridge = {s: Counter() for s in stages}
    mirror_gain = []
    bridge_missed: Dict[str, List[str]] = {s: [] for s in stages}
    lines: List[str] = []

    if per_case:
        header = (f"{'case':7} {'FACTS':22} {'PRED':26} {'SUMMARY':22} "
                  f"{'REPORT GT':24} bridge f/p/s/GT")
        lines += ["", header, "-" * len(header)]

    for case in cases:
        slots = IMPLANT_GT.get(case["case_id"], [])
        gt_bridge = BRIDGE_GT.get(case["case_id"], False)
        claims = {stage: case[f"{stage}_implants"] for stage in stages}

        for stage in stages:
            found, fp, missed = score_slots(claims[stage], slots)
            implant[stage]["tp"] += found
            implant[stage]["fp"] += fp
            implant[stage]["fn"] += missed
            implant[stage]["claims"] += len(claims[stage])

            found, fp, missed = score_slots({mirror_fdi(f) for f in claims[stage]},
                                            slots)
            implant_mirrored[stage]["tp"] += found
            implant_mirrored[stage]["fp"] += fp
            implant_mirrored[stage]["fn"] += missed
            implant_mirrored[stage]["claims"] += len(claims[stage])

            if case["case_id"] in BRIDGE_UNSCORED:
                continue
            has = case[f"{stage}_bridge"]
            bridge[stage]["tp" if has and gt_bridge else
                          "fp" if has else
                          "fn" if gt_bridge else "tn"] += 1
            if gt_bridge and not has:
                bridge_missed[stage].append(case["case_id"])

        # Does the segmentation-derived answer line up with the report only
        # after mirroring? See the laterality note in the module docstring.
        flat_gt = {f for slot in slots for f in slot}
        if flat_gt and claims["facts"]:
            direct = len(claims["facts"] & flat_gt)
            mirrored = len({mirror_fdi(f) for f in claims["facts"]} & flat_gt)
            mirror_gain.append((case["case_id"], direct, mirrored))

        if per_case:
            fmt = lambda t: ", ".join(str(f) for f in sorted(t)) or "-"
            gt_text = " / ".join("|".join(str(f) for f in sorted(s))
                                 for s in slots) or "-"
            flags = "".join("Y" if case[f"{s}_bridge"] else "." for s in stages)
            lines.append(f"{case['case_id']:7} {fmt(claims['facts']):22} "
                         f"{fmt(claims['pred']):26} {fmt(claims['summary']):22} "
                         f"{gt_text:24} {flags}/{'Y' if gt_bridge else '.'}")

    n_slots = sum(len(IMPLANT_GT.get(c["case_id"], [])) for c in cases)
    n_bridge = sum(1 for c in cases if BRIDGE_GT.get(c["case_id"])
                   and c["case_id"] not in BRIDGE_UNSCORED)
    better = [c for c, direct, mirrored in mirror_gain if mirrored > direct]
    worse = [c for c, direct, mirrored in mirror_gain if direct > mirrored]

    results = {"implants": {s: counts(implant[s], *IMPLANT_KEYS) for s in stages},
               "bridges": {s: counts(bridge[s], *BRIDGE_KEYS) for s in stages},
               "implant_slots": n_slots, "bridge_cases": n_bridge,
               "mirror_better": better, "mirror_worse": worse}
    view = {
        "stages": stages,
        "implants_mirrored": {s: counts(implant_mirrored[s], *IMPLANT_KEYS)
                              for s in stages},
        "bridge_missed": bridge_missed,
        "implant_cases": sum(1 for c in cases if IMPLANT_GT.get(c["case_id"])),
        "bridge_scored": sum(1 for c in cases
                             if c["case_id"] not in BRIDGE_UNSCORED),
        "bridge_unscored": {cid: reason
                            for cid, reason in sorted(BRIDGE_UNSCORED.items())
                            if any(c["case_id"] == cid for c in cases)},
        "notes": {cid: note for cid, note in sorted(IMPLANT_NOTES.items())
                  if any(c["case_id"] == cid for c in cases)},
        "mirror_cases": len(mirror_gain),
        "per_case": lines,
    }
    return results, view


def print_prosthetics(results: Dict, view: Dict) -> None:
    stages = view["stages"]
    n_slots, n_bridge = results["implant_slots"], results["bridge_cases"]
    print()
    print("=" * 78)
    print("PROSTHETICS -- facts (masks) vs prediction vs summary, "
          "all against the reports")
    print()
    print(f"IMPLANTS -- {n_slots} implants stated across "
          f"{view['implant_cases']} cases")
    head = f"  {'source':9} {'claims':>7} {'found':>6} {'fp':>4} {'missed':>7} {'prec':>6} {'rec':>6}"
    print(head)
    for stage in stages:
        t = results["implants"][stage]
        print(f"  {stage:9} {t['claims']:7} {t['tp']:6} {t['fp']:4} {t['fn']:7} "
              f"{ratio(t['tp'], t['tp'] + t['fp']):>6} {ratio(t['tp'], n_slots):>6}")
    print("  -- the same claims, every FDI mirrored across the midline --")
    for stage in stages:
        t = view["implants_mirrored"][stage]
        print(f"  {stage + '*':9} {t['claims']:7} {t['tp']:6} {t['fp']:4} {t['fn']:7} "
              f"{ratio(t['tp'], t['tp'] + t['fp']):>6} {ratio(t['tp'], n_slots):>6}")

    print()
    print(f"FIXED BRIDGES -- present in {n_bridge} of {view['bridge_scored']} "
          f"scored cases (scored per case, since facts carry only a boolean)")
    for case_id, reason in view["bridge_unscored"].items():
        print(f"  unscored {case_id}: {reason}")
    print(f"  {'source':9} {'tp':>4} {'fp':>4} {'fn':>4} {'tn':>4} {'prec':>6} {'rec':>6}")
    for stage in stages:
        t = results["bridges"][stage]
        print(f"  {stage:9} {t['tp']:4} {t['fp']:4} {t['fn']:4} {t['tn']:4} "
              f"{ratio(t['tp'], t['tp'] + t['fp']):>6} {ratio(t['tp'], n_bridge):>6}")
    for stage in stages:
        if view["bridge_missed"][stage]:
            print(f"  {stage} misses: {', '.join(view['bridge_missed'][stage])}")

    print()
    print("LATERALITY -- does the segmentation-derived implant list match the "
          "report\n              better after mirroring left<->right?")
    print(f"  better mirrored: {len(results['mirror_better'])} cases"
          + (f" -- {', '.join(results['mirror_better'])}"
             if results["mirror_better"] else ""))
    print(f"  better as-is:    {len(results['mirror_worse'])} cases"
          + (f" -- {', '.join(results['mirror_worse'])}"
             if results["mirror_worse"] else ""))
    print(f"  (of {view['mirror_cases']} cases where both the facts and the "
          f"report name an implant)")
    for case_id, note in view["notes"].items():
        print(f"  note {case_id}: {note}")

    for line in view["per_case"]:
        print(line)


# ---------------------------------------------------------------------------
# Anatomy survey: mandibular canal and maxillary sinus
# ---------------------------------------------------------------------------

def pred_anatomy(pred: Dict) -> Dict:
    facts = _as_dict(pred.get("global"))
    out = {"canal": {}, "ian_close": set(), "sinus": {}, "intrasinusal": set()}
    for side in ("right", "left"):
        canal = _as_dict(facts.get(f"mandible_canal_{side}"))
        # schema v6.4: `location` (lingual|buccal) replaced `course`
        # (regular|lingual|buccal). Fall back to the old key so runs predicted
        # under the previous schema still score.
        out["canal"][side] = canal.get("location", canal.get("course"))
        out["ian_close"] |= {f for f in as_list(canal.get("adjacent_teeth"))
                             if isinstance(f, int)}
        sinus = _as_dict(facts.get(f"maxilla_sinus_{side}"))
        out["sinus"][side] = sinus.get("mucosa_state")
        out["intrasinusal"] |= {f for f in as_list(sinus.get("intrasinusal_teeth"))
                                if isinstance(f, int)}
    return out


def summary_anatomy(summary: Dict) -> Dict:
    out = {"canal": {}, "ian_close": set(), "sinus": {}, "intrasinusal": set()}
    canals = _as_dict(_as_dict(summary.get("mandible")).get("canals"))
    for side in ("right", "left"):
        entry = _as_dict(canals.get(side))
        out["canal"][side] = entry.get("location", entry.get("course"))
        out["ian_close"] |= {f for f in as_list(entry.get("adjacent_teeth"))
                             if isinstance(f, int)}
    maxilla = _as_dict(summary.get("maxilla"))
    sinuses = _as_dict(maxilla.get("sinuses"))
    for side in ("right", "left"):
        out["sinus"][side] = _as_dict(sinuses.get(side)).get("mucosa_state")
    intrasinusal = _as_dict(maxilla.get("sinus_intrasinusal_teeth"))
    out["intrasinusal"] = {f for f in as_list(intrasinusal.get("teeth"))
                           if isinstance(f, int)}
    return out


def survey_anatomy(cases: List[Dict], per_case: bool) -> Tuple[Dict, Dict]:
    """-> (totals for the JSON, view data for print_anatomy). Prints nothing."""
    canal = {s: Counter() for s in ("pred", "summary")}
    canal_swapped = {s: Counter() for s in ("pred", "summary")}
    sinus = {s: Counter() for s in ("pred", "summary")}
    ian = {s: Counter() for s in ("facts", "pred", "summary")}
    intrasinusal = {s: Counter() for s in ("pred", "summary")}
    unanswerable = []
    lines: List[str] = []

    for case in cases:
        cid = case["case_id"]
        gt_canal = CANAL_POSITION_GT.get(cid, {})
        for stage in ("pred", "summary"):
            claimed = case[f"{stage}_anatomy"]["canal"]
            for side, accepted in gt_canal.items():
                if not accepted:
                    continue                       # position never stated
                if accepted == _C:
                    if stage == "pred" and cid not in unanswerable:
                        unanswerable.append(cid)
                    canal[stage]["unanswerable"] += 1
                    continue
                value = claimed.get(side)
                canal[stage]["scored"] += 1
                if value in accepted:
                    canal[stage]["right"] += 1
                elif value == "regular":
                    # Answered the anomaly axis instead of the position axis.
                    # Only reachable for runs predicted under schema <= 6.3,
                    # whose `course` enum still carried a "regular" value; the
                    # v6.4 `location` enum is position-only and cannot hit this.
                    canal[stage]["conflated"] += 1
                elif value is None:
                    canal[stage]["unread"] += 1
                else:
                    canal[stage]["wrong"] += 1
                # Same answer scored against the OTHER side's ground truth.
                other = gt_canal.get("left" if side == "right" else "right") or set()
                if other and other != _C:
                    canal_swapped[stage]["scored"] += 1
                    canal_swapped[stage]["right"] += value in other

        gt_sinus = SINUS_MUCOSA_GT.get(cid, {})
        for stage in ("pred", "summary"):
            claimed = case[f"{stage}_anatomy"]["sinus"]
            for side, expected in gt_sinus.items():
                value = claimed.get(side)
                if value is None:
                    sinus[stage]["unread"] += 1
                    continue
                sinus[stage]["scored"] += 1
                sinus[stage]["right" if value == expected else "wrong"] += 1
                if expected == "thickening":
                    sinus[stage]["abnormal_gt"] += 1
                    sinus[stage]["abnormal_found"] += value == "thickening"

        gt_ian = IAN_CLOSE_GT.get(cid, set())
        skip_ian = IAN_CLOSE_UNSCORED.get(cid, set())
        for stage in ("facts", "pred", "summary"):
            claimed = (case["facts_ian_close"] if stage == "facts"
                       else case[f"{stage}_anatomy"]["ian_close"])
            tp, fp, fn, _ = score(claimed, gt_ian, skip_ian)
            ian[stage]["tp"] += len(tp)
            ian[stage]["fp"] += len(fp)
            ian[stage]["fn"] += len(fn)
            ian[stage]["claims"] += len(claimed - skip_ian)
            # Laterality, as for implants: 48 vs 38 is one mirror apart.
            mirrored, _, _, _ = score({mirror_fdi(f) for f in claimed},
                                      gt_ian, skip_ian)
            ian[stage]["tp_mirrored"] += len(mirrored)

        gt_intra = INTRASINUSAL_GT.get(cid, set())
        for stage in ("pred", "summary"):
            claimed = case[f"{stage}_anatomy"]["intrasinusal"]
            tp, fp, fn, _ = score(claimed, gt_intra, set())
            intrasinusal[stage]["tp"] += len(tp)
            intrasinusal[stage]["fp"] += len(fp)
            intrasinusal[stage]["fn"] += len(fn)

        if per_case and (gt_canal or gt_sinus):
            pa = case["pred_anatomy"]
            lines.append(f"{cid:7} canal pred={pa['canal']} gt="
                         f"{ {s: sorted(v) for s, v in gt_canal.items()} }  "
                         f"sinus pred={pa['sinus']} gt={gt_sinus}")

    n_ian = sum(len(IAN_CLOSE_GT.get(c["case_id"], set())) for c in cases)
    n_sinus = sum(len(v) for k, v in SINUS_MUCOSA_GT.items()
                  if any(c["case_id"] == k for c in cases))
    n_abnormal = sum(1 for k, v in SINUS_MUCOSA_GT.items()
                     for state in v.values() if state == "thickening"
                     and any(c["case_id"] == k for c in cases))
    n_intra = sum(len(INTRASINUSAL_GT.get(c["case_id"], set())) for c in cases)

    results = {"canal": {s: counts(canal[s], *CANAL_KEYS) for s in canal},
               "canal_swapped": {s: counts(canal_swapped[s], "scored", "right")
                                 for s in canal_swapped},
               "ian_close": {s: counts(ian[s], *IAN_KEYS) for s in ian},
               "sinus": {s: counts(sinus[s], *SINUS_KEYS) for s in sinus},
               "intrasinusal": {s: counts(intrasinusal[s], "tp", "fp", "fn")
                                for s in intrasinusal}}
    view = {"unanswerable": unanswerable, "ian_teeth": n_ian,
            "sinus_sides": n_sinus, "sinus_abnormal": n_abnormal,
            "intrasinusal_teeth": n_intra, "per_case": lines}
    return results, view


def print_anatomy(results: Dict, view: Dict) -> None:
    canal, canal_swapped = results["canal"], results["canal_swapped"]
    ian, sinus = results["ian_close"], results["sinus"]
    intrasinusal = results["intrasinusal"]
    unanswerable = view["unanswerable"]
    print()
    print("=" * 78)
    print("ANATOMY -- mandibular canal and maxillary sinus, against the reports")
    print()
    print("MANDIBULAR CANAL -- buccolingual position, per side, only where a "
          "report states it")
    head = (f"  {'source':9} {'scored':>7} {'right':>6} {'conflated':>10} "
            f"{'wrong':>6} {'unread':>7} {'acc':>6}")
    print(head)
    for stage in ("pred", "summary"):
        t = canal[stage]
        print(f"  {stage:9} {t['scored']:7} {t['right']:6} {t['conflated']:10} "
              f"{t['wrong']:6} {t['unread']:7} {ratio(t['right'], t['scored']):>6}")
    print(f"  conflated = answered 'regular' (the anomaly axis) where the "
          f"report gave a position -- pre-v6.4 predictions only")
    # A stage that read NOTHING is a schema-version mismatch, not a bad score,
    # and 0.00 accuracy next to a populated row reads like a regression unless
    # it says so. It happens whenever the summaries are rebuilt by a v6.4
    # postprocess (which reads `location`) from predictions generated before
    # v6.4 (which only carry `course`): the field the summary wants is simply
    # not in the prediction, so it is null everywhere. Re-running inference is
    # what fixes it, not re-running postprocess.
    for stage in ("pred", "summary"):
        t = canal[stage]
        if t["scored"] and t["unread"] == t["scored"]:
            print(f"  NOTE: {stage} carries no canal position at all "
                  f"({t['unread']} sides unread). These predictions predate "
                  f"schema v6.4,\n        so they answer `course` and the "
                  f"v6.4 summary reads `location` -- nothing to score until "
                  f"inference\n        is re-run against the current schema.")
    if unanswerable:
        print(f"  unanswerable: {canal['pred']['unanswerable']} sides in "
              f"{', '.join(unanswerable)} -- the reports say 'central', which "
              f"the location enum\n                (lingual|buccal) "
              f"cannot express")
    for stage in ("pred", "summary"):
        t, s = canal[stage], canal_swapped[stage]
        if s["scored"]:
            print(f"  {stage} scored against the OPPOSITE side: "
                  f"{s['right']}/{s['scored']} ({ratio(s['right'], s['scored'])}) "
                  f"vs {t['right']}/{t['scored']} ({ratio(t['right'], t['scored'])}) "
                  f"as-is")

    print()
    n_ian = view["ian_teeth"]
    print(f"CANAL-ADJACENT TEETH -- {n_ian} teeth the reports place against "
          f"the canal")
    print(f"  {'source':9} {'claims':>7} {'tp':>4} {'fp':>4} {'fn':>4} "
          f"{'prec':>6} {'rec':>6} {'tp mirrored':>12}")
    for stage in ("facts", "pred", "summary"):
        t = ian[stage]
        print(f"  {stage:9} {t['claims']:7} {t['tp']:4} {t['fp']:4} {t['fn']:4} "
              f"{ratio(t['tp'], t['tp'] + t['fp']):>6} {ratio(t['tp'], n_ian):>6} "
              f"{t['tp_mirrored']:>12}")

    print()
    n_sinus, n_abnormal = view["sinus_sides"], view["sinus_abnormal"]
    print(f"MAXILLARY SINUS MUCOSA -- {n_sinus} sides stated "
          f"({n_abnormal} of them abnormal)")
    print(f"  {'source':9} {'scored':>7} {'right':>6} {'wrong':>6} "
          f"{'unread':>7} {'acc':>6} {'abnormal found':>15}")
    for stage in ("pred", "summary"):
        t = sinus[stage]
        print(f"  {stage:9} {t['scored']:7} {t['right']:6} {t['wrong']:6} "
              f"{t['unread']:7} {ratio(t['right'], t['scored']):>6} "
              f"{str(t['abnormal_found']) + '/' + str(t['abnormal_gt']):>15}")

    print()
    n_intra = view["intrasinusal_teeth"]
    print(f"INTRASINUSAL ROOTS -- {n_intra} teeth stated; since schema v7.1 "
          f"this field is\n                     answered on its own evidence, "
          f"with no scope/mucosa gate")
    for stage in ("pred", "summary"):
        t = intrasinusal[stage]
        print(f"  {stage:9} tp {t['tp']:3}  fp {t['fp']:3}  fn {t['fn']:3}  "
              f"prec {ratio(t['tp'], t['tp'] + t['fp'])}  rec {ratio(t['tp'], n_intra)}")

    for line in view["per_case"]:
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path,
                    help="output dir holding predictions/ and summaries/, "
                         "e.g. outputs/aksssr_v5_validate")
    ap.add_argument("--reports-dir", type=Path,
                    default=Path("dataset/validate/reports"),
                    help="reference reports, used to audit REPORT_GT for drift "
                         "and by --dump-report-sentences")
    ap.add_argument("--category", nargs="+", choices=SCORED_CATEGORIES,
                    default=list(SCORED_CATEGORIES))
    ap.add_argument("--label-space", action="store_true",
                    help="mirror the ground truth of the left-right inverted "
                         "reports into the predictions' own space. "
                         + (f"Currently affects: "
                            f"{', '.join(sorted(LR_INVERTED_REPORTS))}."
                            if LR_INVERTED_REPORTS else
                            "NO-OP: LR_INVERTED_REPORTS is empty -- the flip "
                            "was in the segmentation, not the reports, and was "
                            "fixed in v03. Passing this mirrors nothing."))
    ap.add_argument("--facts-dir", type=Path,
                    default=Path("dataset/validate/facts"),
                    help="mask-derived facts, the third source compared "
                         "against the reports for implants and fixed bridges")
    ap.add_argument("--no-gt", action="store_true",
                    help="report PRED/SUMMARY claim counts only, for splits "
                         "with no hand-coded ground truth (everything outside "
                         "the 40 validate cases). Implies --no-prosthetics "
                         "and --no-anatomy, which are scored end to end "
                         "against their own GT tables. The absent-teeth "
                         "section survives it: its label is the segmentation, "
                         "which exists for every split")
    ap.add_argument("--no-prosthetics", action="store_true",
                    help="skip the implant / fixed-bridge comparison")
    ap.add_argument("--no-absent", action="store_true",
                    help="skip the absent-teeth comparison (the one section "
                         "scored against the masks rather than the reports)")
    ap.add_argument("--no-anatomy", action="store_true",
                    help="skip the mandibular-canal / maxillary-sinus comparison")
    ap.add_argument("--per-case", action="store_true",
                    help="print the per-case table as well as the totals")
    ap.add_argument("--case-ids", nargs="+", default=None)
    ap.add_argument("--dump-report-sentences", action="store_true",
                    help="print the report sentences behind each category and "
                         "exit -- the input for re-auditing REPORT_GT")
    ap.add_argument("--out", type=Path, default=None, help="also write JSON")
    args = ap.parse_args()

    if args.dump_report_sentences:
        if not args.reports_dir.is_dir():
            sys.exit(f"[FAIL] reports dir not found: {args.reports_dir}")
        case_ids = args.case_ids or sorted({p.stem.split("_")[0]
                                            for p in args.reports_dir.glob("*.txt")})
        for case_id in case_ids:
            print(f"===== {case_id}")
            for category in args.category:
                hits = finding_sentences(args.reports_dir, case_id, category)
                coded = REPORT_GT.get(case_id, {}).get(category, [])
                skip = UNSCORED.get(case_id, {}).get(category, [])
                if not hits and not coded:
                    continue
                print(f"  -- {category}: GT {coded or 'none'}"
                      + (f", unscored {skip}" if skip else ""))
                for name, sentence, strong in hits:
                    print(f"     [{name}]{'' if strong else ' (weak)'} {sentence}")
        return

    if not args.run_dir:
        sys.exit("[FAIL] --run-dir is required (or pass --dump-report-sentences)")

    if args.no_gt:
        args.no_prosthetics = True
        args.no_anatomy = True

    # The facts file feeds the prosthetics columns AND the absent-teeth label,
    # so it is read unless both of those are off.
    facts_dir = (None if args.no_prosthetics and args.no_absent
                 else args.facts_dir)
    if facts_dir and not facts_dir.is_dir():
        print(f"[WARN] facts dir not found ({facts_dir}) -- the implant and "
              f"bridge comparison will have an empty facts column",
              file=sys.stderr)
    cases = load_cases(args.run_dir, args.case_ids, args.label_space, facts_dir,
                       args.no_gt)
    if args.no_gt:
        # The drift check compares REPORT_GT against the reports; with no
        # table to check it would warn on every case of every category.
        print("[INFO] --no-gt: claim counts only, no scoring, and no "
              "REPORT_GT drift check", file=sys.stderr)
    elif args.reports_dir.is_dir():
        for warning in audit_gt_table(args.reports_dir,
                                      [c["case_id"] for c in cases], args.category):
            print(f"[WARN] GT table drift -- {warning}", file=sys.stderr)
        if not args.no_prosthetics:
            for warning in audit_prosthetics(args.reports_dir,
                                             [c["case_id"] for c in cases]):
                print(f"[WARN] GT table drift -- {warning}", file=sys.stderr)
    else:
        print(f"[WARN] reports dir not found ({args.reports_dir}) -- skipping "
              f"the REPORT_GT drift check", file=sys.stderr)

    # Everything is measured before anything is printed: the summary table has
    # to carry rows from all four sections, so it cannot be written until the
    # last of them has been counted.
    results, findings_view = survey(cases, args.category, args.per_case,
                                   args.no_gt)
    absent = absent_view = None
    if not args.no_absent:
        absent, absent_view = survey_absent(cases, args.per_case)
        if not absent["cases"]:
            absent = absent_view = None      # no facts file reached: nothing to say
    prosthetics = prosthetics_view = None
    if not args.no_prosthetics:
        prosthetics, prosthetics_view = survey_prosthetics(cases, args.per_case)
    anatomy = anatomy_view = None
    if not args.no_anatomy:
        anatomy, anatomy_view = survey_anatomy(cases, args.per_case)

    print(f"Findings survey: {args.run_dir}")
    if args.no_gt:
        print_report_no_gt(cases, args.category, results, findings_view)
        results = strip_gt_from_totals(results, args.category)
    else:
        print_overview(cases, args.category, results, args.label_space,
                       absent, prosthetics, anatomy, anatomy_view)
        print_report(cases, args.category, results, findings_view)
    if absent:
        results["absent"] = absent
        print_absent(absent, absent_view)
    if prosthetics:
        results["prosthetics"] = prosthetics
        print_prosthetics(prosthetics, prosthetics_view)
    if anatomy:
        results["anatomy"] = anatomy
        print_anatomy(anatomy, anatomy_view)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        def case_json(c: Dict) -> Dict:
            out = {"case_id": c["case_id"],
                   "pred": {cat: {str(f): v for f, v in claims.items()}
                            for cat, claims in c["pred"].items()},
                   "summary": {cat: {str(f): v for f, v in claims.items()}
                               for cat, claims in c["summary"].items()}}
            # Omitted rather than emitted empty under --no-gt: an empty list
            # would read as "the reports assert nothing", not "unscored".
            if not args.no_gt:
                out["gt"] = {cat: sorted(c["gt"][cat][0])
                             for cat in SCORED_CATEGORIES}
                out["unscored"] = {cat: sorted(c["gt"][cat][1])
                                   for cat in SCORED_CATEGORIES}
            return out

        args.out.write_text(json.dumps(
            {"run_dir": str(args.run_dir), "label_space": args.label_space,
             "no_gt": args.no_gt,
             "totals": results,
             "cases": [case_json(c) for c in cases]}, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
