#!/usr/bin/env python3
"""
code/eval/review_qa_pairs.py

Review generated (batched) QA records before inference.

The jsonl has ONE record per case. Each record is:
  {
    "case_id": "P018",
    "<section>": {                       # e.g. "global", "dental_elements"
      "<block>": {                       # e.g. "panoramic", "3d_left", "tooth_14"
        "images":    {"<name>": "path/to.png", ...},   # {} when the block has no image
        "questions": {"questions": "1. [key] ...",
                      "output_schema": "{...}",        # human-readable schema string
                      "json_schema":   {...}},         # strict JSON schema for the VLM
        "captions":  {"<name>": "what the image shows", ...}
      },
      ...
    },
    ...
  }

One block with >=1 image == one VLM call. Blocks with no image are skipped at
inference time and are reported here as "no image".

Usage:
  # counts per case/section
  python code/eval/review_qa_pairs.py --qa-jsonl outputs/aksssr_v4_training/qa_pairs.jsonl

  # everything for one case, human readable
  python3 code/eval/review_qa_pairs.py --qa-jsonl outputs/aksssr_v4_validate/qa_pairs.jsonl \
      --format text --case-id F036

  # just the panoramic block of every case, including the strict json schema
  python3 code/eval/review_qa_pairs.py --qa-jsonl outputs/aksssr_v4_training/qa_pairs.jsonl \
      --format text --block panoramic --json-schema

  # only blocks that actually have an image, as markdown
  python3 code/eval/review_qa_pairs.py --qa-jsonl outputs/aksssr_v4_training/qa_pairs.jsonl \
      --format markdown --with-images-only --out review.md

  # exactly what the VLM is sent, one sample per DISTINCT call: one tooth
  # stands for all 32, one arch for both, one side for left+right
  python3 code/eval/review_qa_pairs.py --qa-jsonl outputs/aksssr_v5_validate/qa_pairs.jsonl \
      --format calls --out model_input.txt
"""

import json
import re
import sys
import argparse
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

QUESTION_LINE = re.compile(r"^\s*\d+\.\s")


# ── loading / traversal ───────────────────────────────────────────────────

def load_qa_pairs(qa_jsonl: str) -> dict:
    """Load records from jsonl. Returns {case_id: record} in file order."""
    records_by_case = {}
    with open(qa_jsonl) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] line {lineno}: bad json ({e}), skipped", file=sys.stderr)
                continue
            records_by_case[record.get('case_id', f'line{lineno}')] = record
    return records_by_case


def is_block(value) -> bool:
    """A block is a leaf dict carrying questions/images/captions."""
    return isinstance(value, dict) and (
        'questions' in value or 'images' in value or 'captions' in value)


def iter_sections(record: dict):
    """Yield (section_name, {block_name: block}) for every section of a record.

    Tolerates a section that is itself a single block (older layouts where
    `global` held images/questions directly) by wrapping it as one block.
    """
    for name, value in record.items():
        if name == 'case_id' or not isinstance(value, dict):
            continue
        if is_block(value):
            yield name, {name: value}
        else:
            yield name, {k: v for k, v in value.items() if is_block(v)}


def block_sort_key(name: str):
    """Sort tooth_11..tooth_48 numerically, everything else alphabetically."""
    m = re.match(r"^tooth_(\d+)$", name)
    return (1, int(m.group(1)), "") if m else (0, 0, name)


def count_questions(block: dict) -> int:
    """Number of numbered questions in a block's combined question text."""
    text = (block.get('questions') or {}).get('questions', '') or ''
    n = len([l for l in text.splitlines() if QUESTION_LINE.match(l)])
    # fall back to non-empty lines if the block isn't numbered
    return n or len([l for l in text.splitlines() if l.strip()])


def question_keys(block: dict) -> list:
    """The [bracketed] fact keys of a block, in order."""
    text = (block.get('questions') or {}).get('questions', '') or ''
    return re.findall(r"^\s*\d+\.\s*\[([^\]]+)\]", text, flags=re.M)


def image_paths(block: dict) -> dict:
    return {k: v for k, v in (block.get('images') or {}).items() if v}


def select(records_by_case, case_id=None, section=None, block=None, with_images_only=False):
    """Filter into an ordered list of (case_id, section_name, block_name, block)."""
    out = []
    for cid in sorted(records_by_case):
        if case_id and cid != case_id:
            continue
        for sec_name, blocks in iter_sections(records_by_case[cid]):
            if section and sec_name != section:
                continue
            for blk_name in sorted(blocks, key=block_sort_key):
                if block and block not in blk_name:
                    continue
                blk = blocks[blk_name]
                if with_images_only and not image_paths(blk):
                    continue
                out.append((cid, sec_name, blk_name, blk))
    return out


def trim(text: str, limit: int) -> str:
    if limit and len(text) > limit:
        return text[:limit].rstrip() + f" ... [+{len(text) - limit} chars]"
    return text


# ── formatters ────────────────────────────────────────────────────────────

def format_summary(rows, records_by_case) -> str:
    lines = []
    tot_calls = tot_blocks = tot_facts = 0

    for cid in sorted({r[0] for r in rows}):
        case_rows = [r for r in rows if r[0] == cid]
        lines.append(f"{cid}")
        case_calls = 0
        for sec in dict.fromkeys(r[1] for r in case_rows):
            sec_rows = [r for r in case_rows if r[1] == sec]
            n_blocks = len(sec_rows)
            with_img = [r for r in sec_rows if image_paths(r[3])]
            facts = sum(count_questions(r[3]) for r in with_img)
            case_calls += len(with_img)
            tot_blocks += n_blocks
            tot_facts += facts
            missing = [r[2] for r in sec_rows if not image_paths(r[3])]
            lines.append(f"  {sec:<16} {len(with_img):>2}/{n_blocks} block(s) with image "
                         f"-> {len(with_img):>2} VLM call(s), {facts:>3} question(s)")
            if missing:
                shown = ', '.join(missing[:12])
                more = f" (+{len(missing) - 12} more)" if len(missing) > 12 else ""
                lines.append(f"  {'':<16} no image: {shown}{more}")
        lines.append(f"  {'TOTAL':<16} {case_calls} VLM call(s) for this case")
        tot_calls += case_calls

    lines.append("")
    lines.append(f"TOTAL: {len({r[0] for r in rows})} case(s), {tot_blocks} block(s), "
                 f"{tot_calls} VLM call(s), {tot_facts} question(s)")
    return "\n".join(lines)


def format_text(rows, show_captions=True, show_schema=True, show_json_schema=False,
                caption_chars=0) -> str:
    lines = []
    current = (None, None)

    for cid, sec, blk_name, blk in rows:
        if cid != current[0]:
            lines += ["", "=" * 74, f"CASE: {cid}", "=" * 74]
            current = (cid, None)
        if sec != current[1]:
            lines += ["", f"### SECTION: {sec}", "-" * 74]
            current = (cid, sec)

        imgs = image_paths(blk)
        head = f"[{blk_name}]" + ("" if imgs else "  (no image -- skipped at inference)")
        lines += ["", head]

        for name, path in imgs.items():
            lines.append(f"  image  {name}: {Path(path).name}   ({path})")

        if show_captions:
            for name, cap in (blk.get('captions') or {}).items():
                lines.append(f"  caption {name}:")
                lines.append(f"    {trim(cap, caption_chars)}")

        q = blk.get('questions') or {}
        # Shown before the questions because that is where it sits in the
        # prompt: schema v6.1's shared _definitions vocabulary, which the
        # facts' "how to judge" lines point at by name.
        if q.get('guidance'):
            lines.append("  guidance:")
            for gline in q['guidance'].splitlines():
                lines.append(f"    {gline}")
        lines.append(f"  questions ({count_questions(blk)}):")
        for qline in (q.get('questions') or '(none)').splitlines():
            lines.append(f"    {qline}")

        if show_schema:
            lines.append("  output_schema:")
            for sline in (q.get('output_schema') or '(none)').splitlines():
                lines.append(f"    {sline}")

        if show_json_schema and q.get('json_schema'):
            lines.append("  json_schema:")
            for sline in json.dumps(q['json_schema'], indent=2).splitlines():
                lines.append(f"    {sline}")

    return "\n".join(lines)


def format_markdown(rows, show_captions=True, show_schema=True, show_json_schema=False,
                    caption_chars=0) -> str:
    lines = []
    current = (None, None)

    for cid, sec, blk_name, blk in rows:
        if cid != current[0]:
            lines += [f"\n# Case {cid}\n"]
            current = (cid, None)
        if sec != current[1]:
            lines += [f"\n## {sec}\n"]
            current = (cid, sec)

        imgs = image_paths(blk)
        lines.append(f"### `{blk_name}`" + ("" if imgs else " — no image (skipped)"))
        lines.append("")
        for name, path in imgs.items():
            lines.append(f"- **{name}**: `{path}`")
        if show_captions:
            for name, cap in (blk.get('captions') or {}).items():
                lines.append(f"- **caption ({name})**: {trim(cap, caption_chars)}")
        lines.append("")

        q = blk.get('questions') or {}
        if q.get('guidance'):
            lines.append("**Definitions used by these questions:**")
            lines.append(f"```\n{q['guidance']}\n```")
        lines.append(f"**Questions ({count_questions(blk)}):**")
        lines.append(f"```\n{q.get('questions') or '(none)'}\n```")
        if show_schema:
            lines.append("**Output schema:**")
            lines.append(f"```json\n{q.get('output_schema') or '(none)'}\n```")
        if show_json_schema and q.get('json_schema'):
            lines.append("**JSON schema:**")
            lines.append(f"```json\n{json.dumps(q['json_schema'], indent=2)}\n```")
        lines.append("")

    return "\n".join(lines)


# ── "what the model actually sees" ────────────────────────────────────────
#
# Every other formatter here shows the RECORD -- the questions, the schema,
# the caption -- laid out for reading. This one shows the MESSAGE: the exact
# text run_vqa_inference.py hands the VLM, in the order it hands it over,
# with the prompt templates imported from that module rather than restated,
# so a prompt edit shows up here without anyone remembering to mirror it.
#
# The system prompt and the user template depend on the call, exactly as they
# do at inference time:
#   dental_elements  -> TOOTH_SYSTEM    + TOOTH_USER_TEMPLATE, always with
#                       HOW_TO_READ_A_TOOTH
#   global           -> CATEGORY_SYSTEM + CATEGORY_USER_TEMPLATE, with that
#                       block only for the calls in TOOTH_READING_CALLS

def _inference_prompts():
    """Import run_vqa_inference lazily -- it is only needed by this format.

    It pulls in postprocess_pred/generate_report_from_pred, which the other
    formats have no reason to load (and which need a heavier environment than
    a plain jsonl read does).
    """
    import run_vqa_inference as r  # noqa: E402
    return r


# Which parts of a call name vary between otherwise identical calls. Two calls
# that differ only here ask the same questions of the same kind of picture, so
# the report shows one of them and says how many it stands for.
# `\b` would be wrong for the side: an underscore is a word character, so
# "right\b" never matches inside lower_right_wisdom_tooth, and 3d_left/3d_right
# stayed unmerged. Anchor on the separator instead.
_VARIANT_PATTERNS = [
    (re.compile(r"tooth_\d+"),                  "tooth_{fdi}"),
    (re.compile(r"_(mandible|maxilla)(?=_|$)"), "_{arch}"),
    (re.compile(r"_(left|right)(?=_|$)"),       "_{side}"),
    (re.compile(r"\b(\d\d)\b"),                 "{fdi}"),   # FDI inside fact keys
]


def call_signature(section: str, block_name: str, block: dict) -> str:
    """A key that is equal for calls built from the same schema template.

    Built from the block name AND its fact keys, both with the varying part
    (which tooth, which arch, which side) masked out. Name alone would merge
    two different panoramic calls; fact keys alone would merge every tooth.
    """
    def mask(text):
        for pattern, repl in _VARIANT_PATTERNS:
            text = pattern.sub(repl, text)
        return text
    return f"{section}/{mask(block_name)}::{','.join(mask(k) for k in question_keys(block))}"


def dedupe_rows(rows):
    """One row per distinct call shape. Returns [(row, n_represented)]."""
    groups = {}
    for row in rows:
        groups.setdefault(call_signature(row[1], row[2], row[3]), []).append(row)
    return [(members[0], len(members)) for members in groups.values()]


def format_calls(rows, dedupe=True) -> str:
    """Render each distinct call as the message the VLM receives."""
    r = _inference_prompts()
    pairs = dedupe_rows(rows) if dedupe else [(row, 1) for row in rows]
    lines = []

    for (cid, sec, blk_name, blk), n_same in pairs:
        imgs = image_paths(blk)
        if not imgs:
            continue
        caps = blk.get('captions') or {}
        q = blk.get('questions') or {}
        is_tooth = sec == 'dental_elements'
        fdi = (re.search(r"tooth_(\d+)", blk_name) or [None, None])[1]

        stands_for = f"  [1 of {n_same} identical calls]" if n_same > 1 else ""
        lines += ["", "#" * 78,
                  f"# CALL: {sec}/{blk_name}   (case {cid}){stands_for}",
                  "#" * 78]

        lines += ["", "-" * 78, "-- 1. SYSTEM", "-" * 78,
                  r.TOOTH_SYSTEM if is_tooth else r.CATEGORY_SYSTEM]

        # User content, in wire order: [caption, image] per image, then one
        # text block with the questions -- see build_captioned_image_blocks.
        for i, (name, path) in enumerate(imgs.items(), 1):
            cap = caps.get(name)
            lines += ["", "-" * 78,
                      f"-- {i + 1}. USER text, sent BEFORE image {i}",
                      "-" * 78,
                      f"Image {i} ({name})" + (f": {cap}" if cap else ":"),
                      "", f"    <IMAGE: {path}>"]

        # Two different reading blocks: the composite crops get the per-tooth
        # one, the panoramic arch survey gets the panoramic one (see
        # run_vqa_inference.TOOTH_READING_CALLS for why they are not the same).
        if is_tooth:
            tooth_reading = "\n" + r.HOW_TO_READ_A_TOOTH
        elif blk_name in r.TOOTH_READING_CALLS:
            tooth_reading = "\n" + r.HOW_TO_READ_THE_PANORAMIC
        else:
            tooth_reading = ""
        template = r.TOOTH_USER_TEMPLATE if is_tooth else r.CATEGORY_USER_TEMPLATE
        # images_intro / fdi_tags are per-image-kind (see run_vqa_inference's
        # IMAGE_KIND_INTROS): built from the same image names the call sends,
        # so this dump keeps showing exactly what the VLM is handed.
        fmt = {"n_images": len(imgs),
               "images_intro": r.build_images_intro(imgs),
               "fdi_tags": r.build_fdi_tag_block(imgs),
               "tooth_reading": tooth_reading,
               "guidance": r.build_guidance_section(q.get('guidance')),
               "output_schema": q.get('output_schema') or '{}',
               "questions": q.get('questions') or '(none)'}
        if is_tooth:
            fmt["fdi"] = fdi
        lines += ["", "-" * 78,
                  f"-- {len(imgs) + 2}. USER text, sent AFTER the image(s)",
                  "-" * 78,
                  template.format(**fmt)]

    if not lines:
        return "[no call had an image on disk to render]"
    return "\n".join(lines)


def format_keys(rows) -> str:
    """Compact view: just the [fact_key] list per block."""
    lines = []
    for cid, sec, blk_name, blk in rows:
        flag = "" if image_paths(blk) else "  (no image)"
        lines.append(f"{cid}  {sec}/{blk_name}{flag}")
        for k in question_keys(blk):
            lines.append(f"    - {k}")
    return "\n".join(lines)


def format_json(rows) -> str:
    out = {}
    for cid, sec, blk_name, blk in rows:
        out.setdefault(cid, {}).setdefault(sec, {})[blk_name] = blk
    return json.dumps(out, indent=2)


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Review batched QA records before inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument("--qa-jsonl", required=True, help="Path to qa_pairs.jsonl")
    ap.add_argument("--format", default="summary",
                    choices=["summary", "text", "markdown", "keys", "json", "calls"],
                    help="Output format (default: summary). 'calls' renders the "
                         "exact message the VLM receives -- system prompt, each "
                         "caption, each image, then the questions -- one per "
                         "DISTINCT call, so 32 teeth collapse to one tooth and "
                         "mandible/maxilla to one arch (see --all-calls)")
    ap.add_argument("--all-calls", action="store_true",
                    help="--format calls: render every call, including the "
                         "per-tooth and per-arch repeats")
    ap.add_argument("--case-id", default=None, help="Only this case (e.g. P018)")
    ap.add_argument("--section", default=None,
                    help="Only this section (e.g. global, dental_elements)")
    ap.add_argument("--block", default=None,
                    help="Only blocks whose name contains this (e.g. panoramic, tooth_14)")
    ap.add_argument("--with-images-only", action="store_true",
                    help="Skip blocks that have no image (i.e. show only real VLM calls)")
    ap.add_argument("--no-captions", action="store_true", help="Hide image captions")
    ap.add_argument("--no-schema", action="store_true", help="Hide the output_schema string")
    ap.add_argument("--json-schema", action="store_true",
                    help="Also print the strict json_schema sent to the VLM")
    ap.add_argument("--caption-chars", type=int, default=0,
                    help="Truncate captions to N chars (0 = full text)")
    ap.add_argument("--out", default=None, help="Output file (default: stdout)")

    args = ap.parse_args()

    if not Path(args.qa_jsonl).exists():
        print(f"[FAIL] QA pairs file not found: {args.qa_jsonl}", file=sys.stderr)
        sys.exit(1)

    records_by_case = load_qa_pairs(args.qa_jsonl)
    if not records_by_case:
        print(f"[FAIL] No QA records found in {args.qa_jsonl}", file=sys.stderr)
        sys.exit(1)

    rows = select(records_by_case, args.case_id, args.section, args.block,
                  args.with_images_only)
    if not rows:
        print("[FAIL] No blocks matched the filters "
              f"(case-id={args.case_id}, section={args.section}, block={args.block})",
              file=sys.stderr)
        print(f"[INFO] Cases in file: {', '.join(sorted(records_by_case))}", file=sys.stderr)
        sys.exit(1)

    kw = dict(show_captions=not args.no_captions,
              show_schema=not args.no_schema,
              show_json_schema=args.json_schema,
              caption_chars=args.caption_chars)

    if args.format == "summary":
        output = format_summary(rows, records_by_case)
    elif args.format == "calls":
        # One case is enough: the calls are identical across cases bar the
        # image paths, and rendering all of them would repeat the same
        # prompts per case -- the exact repetition this format exists to cut.
        if not args.case_id:
            first = sorted(records_by_case)[0]
            rows = [r for r in rows if r[0] == first]
            print(f"[INFO] --format calls: showing case {first} "
                  f"(use --case-id to pick another)", file=sys.stderr)
        output = format_calls(rows, dedupe=not args.all_calls)
    elif args.format == "text":
        output = format_text(rows, **kw)
    elif args.format == "markdown":
        output = format_markdown(rows, **kw)
    elif args.format == "keys":
        output = format_keys(rows)
    else:
        output = format_json(rows)

    if args.out:
        with open(args.out, "w") as f:
            f.write(output)
        print(f"[INFO] Written to {args.out}", file=sys.stderr)
    else:
        print(output)
