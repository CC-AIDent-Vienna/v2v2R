#!/usr/bin/env python3
"""
build_fewshot_exemplars.py -- the mechanical half of exemplar authoring.

Choosing few-shot exemplars is a human judgement, but finding the candidates
and getting the answer SHAPE right are not. This script does those two, so the
human pass is "read three rendered images and write three visual_evidence
strings" rather than a hunt through 582 training cases.

Two modes:

  --candidates   Rank training cases for one target category, joining what the
                 segmentation already knows (facts/<case>.json .structured)
                 against what the reports say (survey_findings.finding_sentences,
                 whose keyword/strong/noise tables are the single source for
                 that -- they are not restated here).

                 Prints POSITIVE candidates (report asserts the finding) and
                 CLEAN NEGATIVE candidates (no report sentence, and the facts
                 agree there is nothing). The negatives matter as much as the
                 positives: the negative-hard exemplar is the one that teaches
                 the model NOT to over-call, and it has to be a case where
                 nothing is really there.

  --skeleton     Emit a code/arms/fewshot_examples/<name>.json with the answer keys
                 read off the pool's own json_schema, so the exemplar cannot
                 have the wrong output shape. Values are written as "TODO"
                 ON PURPOSE: fewshot_probe.py hard-fails on placeholder text,
                 so a skeleton that was never filled in cannot silently reach
                 the model.

  --draft-visual-evidence
                 Have a BIGGER model write the visual_evidence prose, given the
                 image and the ALREADY-ESTABLISHED answer. This is rationale
                 distillation, and it is safe here for one reason only: the
                 label does not come from the drafter. The other fields are
                 settled from the training report plus facts.structured, and
                 the drafter is told not to re-judge them -- it only describes
                 what in the image supports them. Letting it diagnose instead
                 would turn a wrong answer into a confidently-worded exemplar,
                 which is worse than having no exemplar at all.

  --check-perceivable
                 Ask the STUDENT model (the local Qwen3.5-9B under test)
                 whether it can actually see each feature the drafted prose
                 names. A stronger drafter cites cues the 9B cannot resolve --
                 faint periapical lucency, subtle trabecular texture -- and an
                 exemplar built on an invisible cue teaches the student to
                 assert findings it cannot verify. That is precisely the
                 over-calling this probe exists to measure, so it would poison
                 the experiment silently. Anything unconfirmed here should be
                 rewritten around a coarser feature.

Neither drafting mode marks an exemplar usable. Both leave "verified": false,
and fewshot_probe.py refuses to run an unverified exemplar -- a human still
reads the prose against the image. Drafting turns the human step from WRITING
into REVIEWING; it does not remove it.

EXEMPLARS COME FROM THE TRAINING SPLIT ONLY. A validate case as an exemplar is
leakage into the scored set; fewshot_probe.py refuses to run if one appears.

Usage
─────
    # who are the candidates for the negative-hard crown exemplar?
    python3 code/arms/build_fewshot_exemplars.py --candidates --category crown \\
        --reports-dir dataset/training/reports --facts-dir dataset/training/facts

    # implants / bridges are read from the segmentation, not keywords
    python3 code/arms/build_fewshot_exemplars.py --candidates --category implants \\
        --reports-dir dataset/training/reports --facts-dir dataset/training/facts

    # once chosen, emit the file to fill in
    python3 code/arms/build_fewshot_exemplars.py --skeleton \\
        --pool-qa-jsonl outputs/fewshot_pool_training/qa_pairs.jsonl \\
        --call 3d_left --cases A011,A064,A073 \\
        --out code/arms/fewshot_examples/3d_left.json

    # fill in every field EXCEPT visual_evidence (from the report + facts), then:
    export OPENAI_API_KEY=sk-...
    python3 code/arms/build_fewshot_exemplars.py --draft-visual-evidence \\
        --exemplar-file code/arms/fewshot_examples/3d_left.json

    # then ask the student model what it can actually see (needs the local server)
    python3 code/arms/build_fewshot_exemplars.py --check-perceivable \\
        --exemplar-file code/arms/fewshot_examples/3d_left.json \\
        --vllm-url http://localhost:8000/v1
"""

import argparse
import json
import os
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

from structured_findings_helper import (  # noqa: E402
    CATEGORIES,
    KEYWORD_RE,
    _as_dict,
    as_list,
    finding_sentences,
    report_paths,
)
from run_vqa_inference import (  # noqa: E402
    build_captioned_image_blocks,
    call_vllm_messages,
    parse_json,
)
# resolve_image_paths and fdis_in are imported LAZILY, inside the two functions
# that use them. Both live in modules this one otherwise has no business
# pulling in -- fewshot_probe.py is the probe arm, draft_report_gt.py is the
# GT drafter -- and at module scope they made every importer of this file a
# dependent of both. That matters because the only importers left are
# draft_evidence.py and check_evidence_perceivable.py, which take three prompt
# CONSTANTS from here and never reach either function. Same reasoning that
# moved generate_report_from_pred inside a function in run_vqa_inference.py.

# An explicit denial. Must be a negation, not just a sentence with no tooth
# number in it -- see the denials filter in candidates() for why that
# distinction is load-bearing.
NEGATION_RE = re.compile(
    r"\b(no|not|none|without|absence|absent|negative for|free of|"
    r"unremarkable|neither)\b", re.IGNORECASE)

# Categories this script can rank. The five text categories come from
# survey_findings; implants and bridges are read off the segmentation instead,
# because that is where facts.structured actually records them.
PROSTHETIC_CATEGORIES = ("implants", "bridges", "crowns_seg")

# Which FDI each wisdom-tooth fact is about, so a GT "absent" can be checked
# against the segmentation. Fixed by schema.json / CALL_PLAN.
WISDOM_FACT_FDI = {"lower_right_wisdom_tooth": 48, "upper_right_wisdom_tooth": 18,
                   "lower_left_wisdom_tooth": 38, "upper_left_wisdom_tooth": 28}
ALL_CATEGORIES = tuple(CATEGORIES) + PROSTHETIC_CATEGORIES


def is_denial(sentence: str, category: str) -> bool:
    """Does this sentence RULE OUT the category, rather than assert it?

    The negation has to scope over the finding, and in these reports it always
    precedes it: "NO impacted teeth", "no evidence of impacted teeth". A
    negation appearing AFTER the finding word negates something else --
    "Impacted third molars ... but NOT in contact with the canal" (P232) and
    "third molar partially impacted and NOT assessable" (P061) both assert
    impaction. Position is what separates the two, and getting it wrong picks a
    positive case as the negative exemplar.
    """
    kw = KEYWORD_RE[category].search(sentence)
    if not kw:
        return False
    neg = NEGATION_RE.search(sentence)
    return bool(neg) and neg.start() < kw.start()


def load_structured(facts_dir: Path, case_id: str) -> dict:
    path = facts_dir / f"{case_id}.json"
    if not path.exists():
        return {}
    return _as_dict(json.loads(path.read_text(encoding="utf-8")).get("structured"))


def seg_signal(structured: dict, category: str):
    """What the mask alone says about this category, or None if it says nothing.

    Facts are segmentation-derived structural data, not report ground truth --
    they are already fed to the pipeline through captions and tooth outlining,
    so using them to pre-screen exemplar candidates adds no information the run
    does not already have.
    """
    # Some facts files carry nulls inside these lists (an implant the
    # segmentation found but could not assign an FDI to), which made a bare
    # sorted() raise on None < None. Keep only real FDIs.
    def fdis(key):
        return sorted(v for v in as_list(structured.get(key))
                      if isinstance(v, int))

    if category == "implants":
        return fdis("implants")
    if category == "bridges":
        return bool(structured.get("bridge_present"))
    if category == "crowns_seg" or category == "crown":
        return fdis("crowns")
    return None


def candidates(reports_dir: Path, facts_dir: Path, category: str, limit: int):
    # Shared so "which teeth does this sentence name" means the same thing in
    # the GT drafter and here -- including span expansion and its
    # hyphen-is-a-list rule. Lazy: see the note at the imports.
    from draft_report_gt import fdis_in

    case_ids = sorted({p.name.split("_")[0].removesuffix(".txt")
                       for p in reports_dir.glob("*.txt")})
    if not case_ids:
        raise SystemExit(f"[FAIL] no reports under {reports_dir}")

    positives, denied, negatives = [], [], []
    for case_id in case_ids:
        structured = load_structured(facts_dir, case_id)
        seg = seg_signal(structured, category)

        if category in CATEGORIES:
            hits = finding_sentences(reports_dir, case_id, category)
            # A sentence can match the category keyword while DENYING it --
            # "No impacted teeth, signs of periodontal disease, fracture lines
            # ... are present" matches on "impacted". Requiring the sentence to
            # name a tooth separates the two: an assertion places the finding on
            # an FDI, a denial has none to place. Without this, A017/A020/A021
            # ranked as positive impaction candidates on the strength of
            # sentences that say the opposite -- and a positive exemplar built
            # from one would teach the model to call a finding the report
            # explicitly rules out.
            strong = [h for h in hits if h[2] and fdis_in(h[1])]
            # A DENIAL needs an explicit negation, not merely the absence of a
            # tooth number. "Impacted third molars:", "Third molars in partial
            # bony impaction" and "third molars are erupted and mesioverted"
            # all name no FDI, and none of them denies anything -- F051 was
            # ranked as a denial candidate on the last of those while its other
            # reader calls 18, 28, 38 and 48 impacted. Picking it as the
            # negative exemplar would have taught the exact inversion of the
            # finding.
            denials = [h for h in hits
                       if h[2] and not fdis_in(h[1]) and is_denial(h[1], category)]
            asserted = bool(strong)
        else:
            hits = []
            asserted = bool(seg)
            for path in report_paths(reports_dir, case_id):
                text = path.read_text(encoding="utf-8")
                word = {"implants": "implant", "bridges": "bridge",
                        "crowns_seg": "crown"}[category]
                hits += [(path.name, s, True) for s in text.split(".")
                         if word in s.lower()]
            asserted = asserted or bool(hits)
            strong = hits

        row = {"case_id": case_id, "seg": seg, "n_hits": len(hits),
               "sentences": [h[1][:110] for h in strong[:2]],
               "denials": [h[1][:110] for h in (denials if category in CATEGORIES
                                                else [])[:2]]}
        if asserted:
            positives.append(row)
        elif row["denials"] and not seg:
            # The report explicitly RULES the finding out. This is the strongest
            # negative exemplar available: not merely "nothing was mentioned",
            # but a radiologist looking and saying no. Ranked ahead of silent
            # cases below.
            denied.append(row)
        elif not hits and not seg:
            # Clean negative: neither the report nor the segmentation asserts
            # anything here. These are the pool the negative-hard exemplar is
            # drawn from -- run --arm zeroshot over them and take one the model
            # answers positive anyway.
            negatives.append(row)

    print(f"\n=== {category}: {len(positives)} positive, {len(denied)} explicit-denial, "
          f"{len(negatives)} silent (of {len(case_ids)} training cases) ===\n")

    print(f"-- POSITIVE candidates (a report places the finding ON A TOOTH) --")
    for row in positives[:limit]:
        print(f"  {row['case_id']:7} seg={row['seg']}  hits={row['n_hits']}")
        for s in row["sentences"]:
            print(f"            \"{s}\"")

    print(f"\n-- EXPLICIT-DENIAL candidates (the report RULES IT OUT) --")
    print(f"   The strongest negative exemplar available: a radiologist looked")
    print(f"   and said no, rather than simply not mentioning it.")
    for row in denied[:limit]:
        print(f"  {row['case_id']:7}")
        for s in row["denials"]:
            print(f"            \"{s}\"")

    print(f"\n-- SILENT candidates (nothing asserted anywhere) --")
    print(f"   Feed these to: fewshot_probe.py --arm zeroshot, then pick one the")
    print(f"   model answers POSITIVE. That is the negative-hard exemplar.")
    for row in negatives[:limit]:
        print(f"  {row['case_id']:7} seg={row['seg']}")

    print(f"\n[NOTE] All three lists are training-split cases. Never use a "
          f"validate case as an exemplar.")
    return {"positive": positives, "denied": denied, "negative": negatives}


def skeleton(pool_qa: Path, call: str, case_ids, fdis, out: Path):
    rows = {}
    with open(pool_qa, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[row["case_id"]] = row

    is_tooth = call.startswith("tooth:")
    tooth_cls = call.split(":", 1)[1] if is_tooth else None

    exemplars = []
    schema_props = None
    for i, case_id in enumerate(case_ids):
        row = rows.get(case_id)
        if row is None:
            raise SystemExit(f"[FAIL] {case_id} not in {pool_qa} -- render it into "
                             f"the exemplar pool first (STAGE=render_pool)")
        if is_tooth:
            fdi = fdis[i]
            call_data = row.get("dental_elements", {}).get(f"tooth_{fdi}")
            where = f"tooth_{fdi}"
        else:
            fdi = None
            call_data = row.get("global", {}).get(call)
            where = call
        if not call_data:
            raise SystemExit(f"[FAIL] {case_id} has no '{where}' call in {pool_qa}")

        # An FDI the mask does not contain still HAS a call in qa_pairs -- it
        # just has no image, and fewshot_probe.py refuses it at load time
        # ("an exemplar with no picture teaches nothing"). Refuse it here
        # instead, before --fill-from-gt and a drafting model have been spent
        # on it. A013's mask holds no molar at all, so tooth_36 shaped up
        # perfectly and filled cleanly from GT as "absent" -- a fine answer
        # about a picture that does not exist.
        if not call_data.get("images"):
            raise SystemExit(
                f"[FAIL] {case_id}/{where} has no image -- the segmentation "
                f"does not contain that tooth, so nothing is rendered for it. "
                f"Pick an FDI from this case's mask.")

        props = ((call_data.get("questions", {}) or {}).get("json_schema") or {}
                 ).get("properties") or {}
        if not props:
            raise SystemExit(f"[FAIL] {case_id}/{where} has no json_schema to shape "
                             f"the answer from")
        schema_props = schema_props or props

        answer = {}
        for fact, spec in props.items():
            fields = (spec.get("properties") or {})
            answer[fact] = {name: f"TODO({_hint(sub)})" for name, sub in fields.items()} \
                if fields else "TODO"
        entry = {"case_id": case_id,
                 "kind": ["positive", "negative_hard", "mixed"][i % 3],
                 "why": "TODO: what this example is here to teach",
                 # Set to true BY HAND, only after reading the prose against
                 # the image. fewshot_probe.py refuses to run until then --
                 # the placeholder check alone cannot catch a machine-drafted
                 # string that nobody ever looked at.
                 "verified": False,
                 "answer": answer}
        if fdi is not None:
            entry["fdi"] = fdi
        exemplars.append(entry)

    spec = {"call_key": "tooth" if is_tooth else call,
            "split": "training",
            "qa_jsonl": str(pool_qa).replace("\\", "/"),
            "exemplars": exemplars}
    if tooth_cls:
        spec["tooth_class"] = tooth_cls

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out} -- {len(exemplars)} exemplar(s), "
          f"{len(schema_props)} fact(s) per answer.")
    print("Every value is a TODO placeholder and fewshot_probe.py will REFUSE to "
          "run until they are written against the rendered images.")


# ── Drafting visual_evidence with a bigger model ──────────────────────────

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


def _openai_client(base_url: str, api_key_env: str):
    from openai import OpenAI
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise SystemExit(
            f"[FAIL] ${api_key_env} is not set. Export it before running:\n"
            f"       export {api_key_env}=sk-...")
    return OpenAI(base_url=base_url or None, api_key=api_key)


def _plain_chat(client, model: str, messages: list, max_tokens: int = 2048) -> str:
    """
    A bare chat call, with NO extra_body.

    run_vqa_inference.call_vllm_messages sends vLLM-specific parameters
    (guided_json, chat_template_kwargs) in extra_body; the real OpenAI API
    rejects unknown body parameters outright, so the drafter cannot reuse it.
    """
    resp = client.chat.completions.create(model=model, messages=messages,
                                          max_tokens=max_tokens)
    return resp.choices[0].message.content


def _call_context(spec: dict, ex: dict, rows: dict, project_dir: Path):
    """(call_data, where) for one exemplar, from the pool qa_pairs."""
    # Lazy: see the note at the imports.
    from fewshot_probe import resolve_image_paths

    row = rows.get(ex["case_id"])
    if row is None:
        raise SystemExit(f"[FAIL] {ex['case_id']} is not in the pool qa_pairs")
    if spec.get("tooth_class"):
        where = f"tooth_{ex['fdi']}"
        call_data = row.get("dental_elements", {}).get(where)
    else:
        where = spec["call_key"]
        call_data = row.get("global", {}).get(where)
    if not call_data:
        raise SystemExit(f"[FAIL] {ex['case_id']} has no '{where}' call")
    return resolve_image_paths(call_data, project_dir), where


def _load_pool(spec: dict, project_dir: Path) -> dict:
    qa_path = Path(spec["qa_jsonl"])
    if not qa_path.is_absolute():
        qa_path = project_dir / qa_path
    rows = {}
    with open(qa_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[row["case_id"]] = row
    return rows


def draft_visual_evidence(exemplar_file: Path, project_dir: Path, model: str,
                          base_url: str, api_key_env: str):
    spec = json.loads(exemplar_file.read_text(encoding="utf-8"))
    rows = _load_pool(spec, project_dir)
    client = _openai_client(base_url, api_key_env)

    for ex in spec["exemplars"]:
        call_data, where = _call_context(spec, ex, rows, project_dir)
        answer = ex["answer"]

        # Everything EXCEPT visual_evidence must already be filled in -- that
        # is what makes this rationale distillation rather than diagnosis.
        established = {fact: {k: v for k, v in fields.items()
                              if k != "visual_evidence"}
                       for fact, fields in answer.items()
                       if isinstance(fields, dict)}
        unfilled = [f"{fact}.{k}" for fact, fields in established.items()
                    for k, v in fields.items()
                    if isinstance(v, str) and v.startswith("TODO")]
        if unfilled:
            raise SystemExit(
                f"[FAIL] {exemplar_file}: {ex['case_id']}/{where} still has "
                f"unfilled answer fields: {unfilled}\n"
                f"       Fill these from the report and facts.structured FIRST. "
                f"The drafter writes prose, it does not decide findings.")

        keys = sorted(established)
        blocks = build_captioned_image_blocks(call_data.get("images", {}),
                                              call_data.get("captions", {}))
        if not blocks:
            raise SystemExit(f"[FAIL] {ex['case_id']}/{where}: no image on disk")

        user_text = DRAFT_USER_TEMPLATE.format(
            answer_json=json.dumps(established, indent=2, ensure_ascii=False),
            keys=", ".join(keys),
            key_list=", ".join(f'"{k}": "..."' for k in keys))

        print(f"[draft] {ex['case_id']}/{where} -> {len(keys)} finding(s) via {model}")
        raw = _plain_chat(client, model,
                          [{"role": "system", "content": DRAFT_SYSTEM},
                           {"role": "user",
                            "content": blocks + [{"type": "text", "text": user_text}]}])
        try:
            drafted = parse_json(raw)
        except Exception as e:
            print(f"  [WARN] unparseable draft ({e}) -- leaving this one alone")
            continue

        for fact in keys:
            text = drafted.get(fact)
            if isinstance(text, str) and text.strip():
                answer[fact]["visual_evidence"] = text.strip()
            else:
                print(f"  [WARN] {fact}: drafter returned nothing")

        ex["drafted_by"] = model
        # Drafting never certifies an exemplar. fewshot_probe.py refuses to run
        # while this is false, so a machine-written string cannot reach the
        # model without a human having read it against the image.
        ex["verified"] = False

    exemplar_file.write_text(json.dumps(spec, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    print(f"\nWrote {exemplar_file}")
    print("Every exemplar is marked verified:false. Read each visual_evidence "
          "against its image,\nthen run --check-perceivable, then set "
          "\"verified\": true by hand.")


# ── Can the student actually see it? ──────────────────────────────────────

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


def check_perceivable(exemplar_file: Path, project_dir: Path, vllm_url: str,
                      model: str):
    from openai import OpenAI
    spec = json.loads(exemplar_file.read_text(encoding="utf-8"))
    rows = _load_pool(spec, project_dir)
    client = OpenAI(base_url=vllm_url, api_key="not-needed")

    total, unconfirmed = 0, 0
    for ex in spec["exemplars"]:
        call_data, where = _call_context(spec, ex, rows, project_dir)
        blocks = build_captioned_image_blocks(call_data.get("images", {}),
                                              call_data.get("captions", {}))
        if not blocks:
            print(f"[SKIP] {ex['case_id']}/{where}: no image on disk")
            continue

        for fact, fields in ex["answer"].items():
            if not isinstance(fields, dict):
                continue
            text = fields.get("visual_evidence")
            if not isinstance(text, str) or not text.strip():
                continue

            user = blocks + [{"type": "text", "text":
                              f"DESCRIPTION TO CHECK:\n{text}"}]
            raw = call_vllm_messages(client, model,
                                     [{"role": "system", "content": PERCEIVE_SYSTEM},
                                      {"role": "user", "content": user}],
                                     max_tokens=1024)
            try:
                verdict = parse_json(raw)
            except Exception as e:
                print(f"  [WARN] {ex['case_id']}/{fact}: unparseable verdict ({e})")
                continue

            features = verdict.get("features") or []
            missed = [f for f in features if not f.get("visible")]
            total += len(features)
            unconfirmed += len(missed)
            flag = "OK " if not missed else "CHECK"
            print(f"[{flag}] {ex['case_id']}/{fact}: "
                  f"{len(features) - len(missed)}/{len(features)} features visible")
            for f in missed:
                print(f"        NOT VISIBLE: {f.get('feature')} "
                      f"-- {f.get('why', '')}")

    print(f"\n{total - unconfirmed}/{total} claimed features confirmed visible "
          f"by {model}.")
    if unconfirmed:
        print("Rewrite the unconfirmed ones around a coarser, more locatable "
              "feature.\nAn exemplar that cites a cue the student cannot see "
              "teaches it to assert\nfindings it cannot verify -- the exact "
              "over-calling this probe measures.")


def _gt_fact(gt: dict, spec: dict, where: str, fact: str) -> dict:
    """One fact's GT object, from the half of the file that holds it.

    parse_reports_to_gt.py writes the pipeline's prediction layout, so a
    global fact lives at gt["global"][fact] but a per-tooth fact lives at
    gt["teeth"]["tooth_46"]["tooth_46_morphology"] -- one level deeper, under
    a key that repeats the FDI.

    This used to read gt["global"] unconditionally. For a tooth exemplar that
    finds nothing, and finding nothing is not an error here -- the field is
    simply left as TODO -- so a tooth file would have come back with every
    one of its ~25 fields unfilled and no explanation. The four exemplar
    files that existed when this was written were all global calls, so the
    path was never exercised.
    """
    if spec.get("tooth_class"):
        return _as_dict(_as_dict(_as_dict(gt.get("teeth")).get(where)).get(fact))
    return _as_dict(_as_dict(gt.get("global")).get(fact))


def fill_from_gt(exemplar_file: Path, gt_dir: Path, facts_dir: Path,
                 project_dir: Path):
    """
    Fill an exemplar's NON-PROSE answer fields from parse_reports_to_gt.py's
    output, validating every value instead of trusting it.

    The extraction is a draft, and on the pool it demonstrably contained:
      - invented fields ("how_the_finding_is_defined") that are in no schema,
        which would teach the model a key that does not exist;
      - READER DISAGREEMENT presented as fact -- F008's two radiologists
        differ on five of the 3d_left fields, including whether 18 is impacted
        at all (True vs False). An exemplar is supposed to be an unambiguous
        worked example; a contested one teaches noise;
      - SILENCE READ AS ABSENCE -- A020's 48 came back "eruption_state":
        "absent" because the report never mentions it, while the segmentation
        has the tooth. A report describes what is notable; it does not
        enumerate every normal tooth. That failure mode is exactly backwards
        for a negative-hard exemplar, whose whole point is a tooth that IS
        there and IS unremarkable.

    So: only schema fields are copied, enum values are checked, readers must
    agree, and presence is cross-checked against facts.structured -- the
    segmentation, which is what the rendered image actually shows. Anything
    that fails is left as TODO and reported, so the probe's placeholder guard
    still blocks the run until a human resolves it.
    """
    spec = json.loads(exemplar_file.read_text(encoding="utf-8"))
    rows = _load_pool(spec, project_dir)
    conflicts = []

    for ex in spec["exemplars"]:
        case_id = ex["case_id"]
        call_data, where = _call_context(spec, ex, rows, project_dir)
        props = ((call_data.get("questions", {}) or {}).get("json_schema") or {}
                 ).get("properties") or {}

        gt_files = sorted(gt_dir.glob(f"{case_id}_gt.json")) + \
            sorted(gt_dir.glob(f"{case_id}_*_gt.json"))
        if not gt_files:
            conflicts.append(f"{case_id}: no GT file in {gt_dir}")
            continue
        readers = {p.name: json.loads(p.read_text(encoding="utf-8"))
                   for p in gt_files}

        structured = load_structured(facts_dir, case_id)
        present = {v for v in as_list(structured.get("teeth_present"))
                   if isinstance(v, int)}

        for fact, fact_schema in props.items():
            fields = (fact_schema.get("properties") or {})
            target = ex["answer"].setdefault(fact, {})
            for field, sub in fields.items():
                if field == "visual_evidence":
                    continue                      # the drafter's job

                values = []
                for name, gt in readers.items():
                    v = _as_dict(_gt_fact(gt, spec, where, fact)).get(field)
                    if v is not None:
                        values.append((name, v))
                if not values:
                    continue

                distinct = {json.dumps(v, sort_keys=True) for _n, v in values}
                if len(distinct) > 1:
                    conflicts.append(
                        f"{case_id}/{fact}.{field}: readers disagree "
                        f"{[(n, v) for n, v in values]} -- left TODO")
                    continue

                value = values[0][1]
                enum = sub.get("enum")
                if enum is not None and value not in enum:
                    conflicts.append(
                        f"{case_id}/{fact}.{field}: {value!r} is not in the "
                        f"schema enum {enum} -- left TODO")
                    continue

                # Presence guard, BOTH DIRECTIONS, for the wisdom-tooth facts
                # whose FDI the fact name identifies. The render is built from
                # the mask, so the mask decides what is in the picture and the
                # report cannot overrule it.
                #
                # Both directions matter and only one was checked at first:
                #   - GT "absent", mask HAS it: the report is silent about a
                #     normal tooth and the extractor read silence as absence.
                #   - GT erupted/impacted, mask LACKS it: the report describes
                #     a tooth that is not in this segmentation, usually because
                #     the report is laterality-flipped. This one slipped
                #     through and put "48 fully_erupted" on A017, whose mask
                #     has only 38 of the four wisdom teeth.
                # A tooth exemplar names its FDI in the file rather than in the
                # fact name, but it is the same check and the same failure: a
                # composite is cropped around the mask's tooth, so a GT that
                # calls it absent while the mask has it (or the reverse) is
                # describing a different picture from the one being shown.
                fdi = (ex.get("fdi") if spec.get("tooth_class")
                       else WISDOM_FACT_FDI.get(fact))
                if fdi is not None and field == "eruption_state":
                    if value == "absent" and fdi in present:
                        conflicts.append(
                            f"{case_id}/{fact}: GT says absent but the "
                            f"segmentation HAS tooth {fdi}. Silence in the "
                            f"report is not absence -- left TODO, decide from "
                            f"the image.")
                        continue
                    if value != "absent" and fdi not in present:
                        conflicts.append(
                            f"{case_id}/{fact}: GT says {value!r} but tooth "
                            f"{fdi} is NOT in the segmentation, so it is not "
                            f"drawn. Usually a laterality-flipped report -- "
                            f"left TODO, decide from the image.")
                        continue

                target[field] = value

    exemplar_file.write_text(json.dumps(spec, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    print(f"Filled {exemplar_file.name} from {gt_dir}")
    if conflicts:
        print(f"\n{len(conflicts)} field(s) NOT filled -- resolve by hand:")
        for c in conflicts:
            print(f"  - {c}")
    else:
        print("  every non-prose field filled and validated")
    return conflicts


def _hint(sub: dict) -> str:
    """A short shape hint in the placeholder, so the author knows what goes there."""
    if "enum" in sub:
        return "|".join(str(v) for v in sub["enum"])
    t = sub.get("type")
    if isinstance(t, list):
        t = "/".join(x for x in t if x != "null")
    return str(t or "?")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--candidates", action="store_true")
    mode.add_argument("--skeleton", action="store_true")
    mode.add_argument("--draft-visual-evidence", action="store_true",
                      help="a bigger model writes the prose for an already-filled answer")
    mode.add_argument("--fill-from-gt", action="store_true",
                      help="fill non-prose answer fields from parse_reports_to_gt.py "
                           "output, validating enums / reader agreement / presence")
    mode.add_argument("--check-perceivable", action="store_true",
                      help="ask the student model whether it can see what the prose claims")

    ap.add_argument("--category", choices=ALL_CATEGORIES)
    ap.add_argument("--reports-dir", type=Path, default=Path("dataset/training/reports"))
    ap.add_argument("--facts-dir", type=Path, default=Path("dataset/training/facts"))
    ap.add_argument("--limit", type=int, default=25)

    ap.add_argument("--pool-qa-jsonl", type=Path)
    ap.add_argument("--call", help="a global call key, or tooth:<incisor|canine|premolar|molar>")
    ap.add_argument("--cases", help="comma-separated exemplar case IDs (training split)")
    ap.add_argument("--fdis", help="comma-separated FDIs, one per case (tooth calls only)")
    ap.add_argument("--out", type=Path)

    ap.add_argument("--exemplar-file", type=Path,
                    help="the exemplar file to draft into / check")
    ap.add_argument("--gt-dir", type=Path,
                    default=Path("dataset/training/outputs/ground_truth"),
                    help="where gen_ground_truth.sh wrote {case}[_{reader}]_gt.json "
                         "(--fill-from-gt). NOTE this is the per-split path, not "
                         "the repo-root outputs/ground_truth used for validate.")
    ap.add_argument("--project-dir", type=Path, default=Path("."),
                    help="root the pool's image paths are relative to")
    # Drafting model: same env-var conventions as code/eval/evaluation.sh's judge.
    ap.add_argument("--draft-model", default=os.environ.get("OPENAI_MODEL", "gpt-4o"))
    ap.add_argument("--draft-base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    # Student model, for --check-perceivable: the local server under test.
    ap.add_argument("--vllm-url", default="http://localhost:8000/v1")
    ap.add_argument("--student-model", default="qwen3.5-vl")
    args = ap.parse_args()

    if args.fill_from_gt:
        if not args.exemplar_file:
            ap.error("--fill-from-gt needs --exemplar-file")
        fill_from_gt(args.exemplar_file, args.gt_dir, args.facts_dir,
                     args.project_dir.resolve())
        return

    if args.draft_visual_evidence or args.check_perceivable:
        if not args.exemplar_file:
            ap.error("--exemplar-file is required for this mode")
        if not args.exemplar_file.exists():
            raise SystemExit(f"[FAIL] no exemplar file at {args.exemplar_file}")
        project_dir = args.project_dir.resolve()
        if args.draft_visual_evidence:
            draft_visual_evidence(args.exemplar_file, project_dir,
                                  args.draft_model, args.draft_base_url,
                                  args.api_key_env)
        else:
            check_perceivable(args.exemplar_file, project_dir,
                              args.vllm_url, args.student_model)
        return

    if args.candidates:
        if not args.category:
            ap.error("--candidates needs --category")
        candidates(args.reports_dir, args.facts_dir, args.category, args.limit)
        return

    for required in ("pool_qa_jsonl", "call", "cases", "out"):
        if getattr(args, required) is None:
            ap.error(f"--skeleton needs --{required.replace('_', '-')}")
    case_ids = [c.strip() for c in args.cases.split(",")]
    fdis = [int(f) for f in args.fdis.split(",")] if args.fdis else []
    if args.call.startswith("tooth:") and len(fdis) != len(case_ids):
        ap.error("a tooth exemplar file needs one --fdis entry per --cases entry")
    skeleton(args.pool_qa_jsonl, args.call, case_ids, fdis, args.out)


if __name__ == "__main__":
    main()
