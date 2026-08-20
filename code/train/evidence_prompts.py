#!/usr/bin/env python3
"""
code/train/evidence_prompts.py -- the two prompts of the evidence pass.

docs/vision_sft_plan.md §3.3 is one idea run twice, by two different models, and
these are the words each is given:

  DRAFT_SYSTEM + DRAFT_USER_TEMPLATE   the TEACHER is handed a finished answer
      and told the findings "are ALREADY ESTABLISHED ... you must not re-judge
      them". It writes only what in THIS image supports them. The label never
      comes from the teacher, and that is the whole reason rationale
      distillation is safe here.

  PERCEIVE_SYSTEM   the STUDENT is asked whether the features that prose names
      are actually visible. Never the teacher: it wrote the prose, so asking it
      confirms nothing. This is the screen that catches a teacher writing
      confident, specific support for something it could not see -- which it
      will, precisely because it was told the finding is established.

WHY THEY ARE A FILE OF THEIR OWN
────────────────────────────────
Two copies of a prompt drift, and the drift is invisible: both render, both
parse, and the two runs stop being the same experiment without anything
failing. So there is one copy, and the two modules of the evidence pass import
it: draft_evidence.py writes the prose over the pool's ~1,850 tooth calls, and
check_evidence_perceivable.py screens it.

These three strings sat inside an abandoned arm's exemplar builder until
2026-08-20, which is why the SFT path used to import from it. They were never
that arm's -- the teacher/student split above is the SFT evidence pass's own
design -- and nothing here depends on that arm any more.
"""

from __future__ import annotations

DRAFT_SYSTEM = """\
You are an expert dental/maxillofacial radiologist writing TEACHING captions
for a CBCT reading exercise.

The findings you are given are ALREADY ESTABLISHED from the written report and
from the segmentation. They are not in question and you must not re-judge them,
soften them, or contradict them. Your only job is to write, for each finding,
the "visual_evidence" sentence: what in THIS image a reader should point at.

Write for a reader with WEAKER eyes than yours. Every feature you name must be:
  - LOCATABLE  -- say where it is (which tooth by FDI, which quadrant, which
                  region), not just that it exists;
  - COARSE     -- large and high-contrast enough to be unmistakable at this
                  resolution. Do not cite fine texture, subtle gradients, or
                  anything you would need to zoom in to confirm;
  - RELATIONAL -- describe brightness, size and position RELATIVE to a named
                  neighbouring structure ("brighter than the enamel of 45",
                  "sits above the canal", "shorter than the adjacent roots"),
                  because absolute descriptions are not checkable.

Never hedge: no "possibly", "suggestive of", "cannot be excluded", "appears to
may". If a finding's evidence is not clearly visible in this image, say plainly
which part is not visible rather than inventing support for it.

Never mention the report, the segmentation, the ground truth, or the fact that
the answer was given to you. The sentence must read as a first-hand reading of
the picture.

Answer ONLY in valid JSON: one key per finding, one string value each. No
prose, no markdown fences, no extra keys.
"""

DRAFT_USER_TEMPLATE = """\
ESTABLISHED FINDINGS for this image (do not re-judge; describe the support):
{answer_json}

Write "visual_evidence" for each of these keys: {keys}

One to three sentences each. Return exactly:
{{{key_list}}}
"""

PERCEIVE_SYSTEM = """\
You are looking at ONE CBCT image. You will be given a written description of
it. For each concrete visual feature the description claims, say whether you
can actually see that feature in this image.

Judge only visibility, not clinical correctness. Be strict: if you cannot point
to the feature at the location described, it is not visible. Do not give the
description the benefit of the doubt.

Answer ONLY in valid JSON:
{"features": [{"feature": "<short quote or paraphrase>", "visible": true|false,
               "why": "<one clause>"}]}
"""
