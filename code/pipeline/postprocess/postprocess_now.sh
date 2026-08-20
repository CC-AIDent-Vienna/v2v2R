#!/usr/bin/env bash
# =============================================================================
# postprocess_now.sh — rebuild summaries, reports and the finding survey from
# an existing predictions/ directory, in one command, on the login node.
#
# WHY THIS EXISTS
#   postprocess_pred.py, synthesize_report.py and structured_findings_evaluation.py are three
#   separate CPU-only scripts that always have to be run in that order, and
#   running them by hand is where the mistakes happen: surveying summaries
#   that predate the postprocess change being measured, or reading a report
#   rendered from a stale summary. Every tuning decision recorded in
#   postprocess_pred.py's flag comments (CROSS_VALIDATE_IMPACTION,
#   CROSS_VALIDATE_ENDODONTIC, CROSS_VALIDATE_RESTORATIONS,
#   DROP_IMPLANTS_ON_PRESENT_TEETH) came from this exact loop, so it is worth
#   one script.
#
#   AS OF 2026-08-07 THOSE GATES ARE ALL OFF. Every one of them that was ever
#   surveyed in this loop cost recall and left precision flat, so postprocess
#   is now a bridge -- it derives, groups and shapes, and does not drop the
#   model's findings. The gates are kept and still run for the audit trail;
#   POSTPROCESS_ARGS="--cross-validate --demote-uncertain" restores the old
#   arm, and this loop is still how the two get compared.
#
#   No GPU, no model, no sbatch -- predictions in, JSON summaries, prose
#   reports and a survey out. Run it as often as you like.
#
# WHAT IT OVERWRITES
#   <run_dir>/summaries/ and <run_dir>/synthesized_reports/ are rebuilt in
#   place. Both are git-tracked, so `git diff` is the record of what a flag
#   change did to the output -- which is the point. The survey is written to
#   <run_dir>/survey/ under a timestamp and never overwrites a previous one.
#
# READING THE SURVEY
#   As of 2026-08-12 the survey is structured_findings_evaluation.py, and it scores against the
#   GENERATED ground truth -- parse_reports_to_gt.py's *_gt.json, one per case,
#   emitted in PREDICTION SHAPE. That is what makes it fact-level: absence,
#   atrophy, implants, bridges and every per-tooth finding are ordinary fields
#   in that file, so all of them are scored the same way, against the same
#   label, with no axis falling back to anything else.
#
#   It replaced survey_findings.py here, which scored the findings against a
#   hand-coded REPORT_GT table and absence against the SEGMENTATION MASK --
#   no report enumerates all 32 positions, so absence had nowhere else to go,
#   and the mask is not always right. survey_findings.py was deleted on
#   2026-08-19; its hand-read tables live on in
#   code/ground_truth/report_gt_tables.py, with no survey around them.
#
#   Two views. Four CALL GROUP sections -- 3d, sinus, panoramic, detail --
#   break down per FIELD, so "which read is failing" is answerable rather than
#   just "the score moved". Then one OVERALL table puts PRED beside SUMMARY for
#   every key axis, which is where a flag change shows up: PRED cannot move
#   (postprocess does not touch the raw reads), so any row whose SUMMARY column
#   shifted is what the flag did.
#
#   Files are written as survey_facts_<stamp>.{txt,json} -- a different prefix
#   from the deleted survey_findings.py's survey_<stamp>.*, on purpose. The two score
#   different questions against different labels, so diffing one against the
#   other would report every row as changed. Each prefix only ever diffs
#   against its own kind.
#

# Usage:
#   code/pipeline/postprocess/postprocess_now.sh <run_dir>
#   code/pipeline/postprocess/postprocess_now.sh outputs/aksssr_v5_validate
#
# Env overrides:
#   SPLIT=validate|training   which dataset the survey's reports/facts come
#                             from (default: inferred from the run dir name)
#   CASE_IDS="A008 A019"      restrict every stage to these cases
#   NO_SURVEY=1               stop after the reports
#   NO_REPORTS=1              stop after the summaries
#   SURVEY_ARGS=...          extra flags for structured_findings_evaluation.py
#   GT_DIR=...               generated ground truth (default:
#                            dataset/$SPLIT/outputs/ground_truth)
#   POSTPROCESS_ARGS="--cross-validate --demote-uncertain"
#                             extra flags for postprocess_pred.py; this pair
#                             re-runs the pre-2026-08-07 gated arm
#
# Exit codes: 1 = bad arguments / missing inputs, 2 = a stage failed.
# =============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [ "$d" != "/" ] && [ ! -f "$d/schema/schema.json" ]; do d="$(dirname "$d")"; done; echo "$d")}"
CODE_DIR="$PROJECT_DIR/code"

RUN_DIR="${1:?Usage: code/pipeline/postprocess/postprocess_now.sh <run_dir>   (e.g. outputs/aksssr_v5_validate)}"
case "$RUN_DIR" in
    /*) : ;;
    *) RUN_DIR="$PROJECT_DIR/$RUN_DIR" ;;
esac

# Per-stage directories. aksssr_pipeline.sh nests each stage's output under a
# schema-version tag -- predictions/predictions_v6.9/ rather than predictions/ --
# so that a v6.9 run never lands on top of a v6.4 one. A run dir may therefore
# hold EITHER layout: the flat one from before that change (and from the older
# arms, which are not being re-run), or one-or-more versioned subdirectories.
#
# Resolve it by looking for where the *_pred.json actually are, newest version
# first, instead of picking a layout and failing on the other. An explicit
# PRED_DIR= still wins, which is how you re-postprocess an OLDER version after
# a newer one exists.
resolve_stage_dir() {
    # $1 = parent dir, $2 = subdir name prefix, $3 = glob that must match inside
    local parent="$1" prefix="$2" glob="$3" d
    # Flat layout: the files sit directly in the parent.
    if compgen -G "$parent/$glob" >/dev/null 2>&1; then echo "$parent"; return; fi
    # Versioned layout: newest-sorting subdir that actually has the files.
    # `sort -V` so v6.10 would come after v6.9 rather than before it.
    while IFS= read -r d; do
        [ -n "$d" ] || continue
        if compgen -G "$d/$glob" >/dev/null 2>&1; then echo "$d"; return; fi
    done < <(find "$parent" -mindepth 1 -maxdepth 1 -type d -name "${prefix}_*" 2>/dev/null | sort -Vr)
    # Nothing found -- hand back the parent so the caller's own check reports it.
    echo "$parent"
}

PRED_DIR="${PRED_DIR:-$(resolve_stage_dir "$RUN_DIR/predictions" predictions '*_pred.json')}"
# summaries/reports are WRITTEN, not read, so they can't be resolved by content.
# Mirror whatever version tag PRED_DIR resolved to, so a versioned predictions
# dir produces versioned summaries and reports rather than scattering them flat.
_ver_tag=""
_pred_base="$(basename "$PRED_DIR")"
case "$_pred_base" in
    predictions_*) _ver_tag="${_pred_base#predictions_}" ;;
esac
if [ -n "$_ver_tag" ]; then
    SUMMARY_DIR="${SUMMARY_DIR:-$RUN_DIR/summaries/summaries_${_ver_tag}}"
    REPORT_DIR="${REPORT_DIR:-$RUN_DIR/synthesized_reports/synthesized_reports_${_ver_tag}}"
else
    SUMMARY_DIR="${SUMMARY_DIR:-$RUN_DIR/summaries}"
    REPORT_DIR="${REPORT_DIR:-$RUN_DIR/synthesized_reports}"
fi
SURVEY_DIR="${SURVEY_DIR:-$RUN_DIR/survey}"

# The survey needs the reference reports and the mask-derived facts, which
# live per split. Infer from the run dir name so the common case needs no env.
if [ -z "${SPLIT:-}" ]; then
    case "$(basename "$RUN_DIR")" in
        *validate*) SPLIT="validate" ;;
        *train*)    SPLIT="training" ;;
        *) echo "[FAIL] cannot infer SPLIT from $(basename "$RUN_DIR") -- pass SPLIT=validate|training"
           exit 1 ;;
    esac
fi
# The survey scores against the GENERATED ground truth -- parse_reports_to_gt.py
# output, one *_gt.json per case in prediction shape -- not against the reports
# or the mask-derived facts, which is why neither of those dirs is needed here
# any more.
GT_DIR="${GT_DIR:-$PROJECT_DIR/dataset/$SPLIT/outputs/ground_truth}"

# THE SOURCE RULES (docs/postprocess_pipeline.md) re-source ten findings from the
# mask-derived facts file. On by default, because every one of them was
# measured better than what it replaces; NO_SOURCE_RULES=1 turns the lot off
# and postprocess then behaves exactly as it did before 2026-08-16, which is
# how the two arms get compared.
#
# --facts-dir carries ONE MORE THING that is not a source rule and is not under
# the NO_SOURCE_RULES switch's name: the maxilla's FOV scope. With the facts,
# `fov.maxilla` decides whether the maxilla arch is reported at all, in both
# directions, instead of the model's own maxilla_scope answer -- see
# postprocess_pred.py's drop_excluded_maxilla gate and THE RULE -- maxilla FOV
# scope. NO_SOURCE_RULES=1 drops the flag, so it turns this off too.
FACTS_DIR="${FACTS_DIR:-$PROJECT_DIR/dataset/$SPLIT/facts}"
FACTS_ARGS=1
[ -n "${NO_SOURCE_RULES:-}" ] && FACTS_ARGS=""
[ -d "$FACTS_DIR" ] || FACTS_ARGS=""

CASE_ARGS=()
[ -n "${CASE_IDS:-}" ] && CASE_ARGS=(--case-ids ${CASE_IDS})

STAMP="$(date +%Y%m%d_%H%M%S)"

echo "[INFO] ========== Postprocess -> Report -> Survey =========="
echo "[INFO] Start:       $(date)"
echo "[INFO] Run dir:     $RUN_DIR"
echo "[INFO] Split:       $SPLIT"
echo "[INFO] Predictions: $PRED_DIR"
[ -n "${CASE_IDS:-}" ] && echo "[INFO] Case IDs:    $CASE_IDS"
echo ""

# ── Sanity checks ──────────────────────────────────────────────────────────

if [ ! -d "$PRED_DIR" ]; then
    echo "[FAIL] No predictions directory: $PRED_DIR"
    exit 1
fi
# Trailing slash on purpose: postprocess_val_now.sh builds its arms with
# predictions/ as a SYMLINK back to the source run, and `find` does not
# descend into a symlinked start point without it -- it reports the link
# itself and this check then fails on a directory that has all 40 files.
NUM_PRED=$(find "$PRED_DIR"/ -maxdepth 1 -name '*_pred.json' | wc -l)
if [ "$NUM_PRED" -eq 0 ]; then
    echo "[FAIL] No *_pred.json in $PRED_DIR"
    exit 1
fi
echo "[INFO] $NUM_PRED prediction file(s)"

# ── Stage 1: predictions -> summaries ──────────────────────────────────────

echo ""
echo "[INFO] --- Stage 1/3: postprocess_pred.py ---"
python3 "$CODE_DIR/pipeline/postprocess/postprocess_pred.py" \
    --pred-dir "$PRED_DIR" \
    --out-dir "$SUMMARY_DIR" \
    ${FACTS_ARGS:+--facts-dir "$FACTS_DIR"} \
    "${CASE_ARGS[@]}" \
    ${POSTPROCESS_ARGS:-} \
    || { echo "[FAIL] postprocess_pred.py failed"; exit 2; }

NUM_SUMMARY=$(find "$SUMMARY_DIR" -maxdepth 1 -name '*_summary.json' | wc -l)
echo "[PASS] $NUM_SUMMARY summary file(s) -> $SUMMARY_DIR"
if [ -z "${CASE_IDS:-}" ] && [ "$NUM_SUMMARY" -ne "$NUM_PRED" ]; then
    echo "[WARN] $NUM_PRED predictions produced $NUM_SUMMARY summaries -- some case failed above"
fi

if [ -n "${NO_REPORTS:-}" ]; then
    echo "[INFO] NO_REPORTS=1 -- stopping after the summaries."
    exit 0
fi

# ── Stage 2: summaries -> reports ──────────────────────────────────────────

echo ""
echo "[INFO] --- Stage 2/3: synthesize_report.py ---"
python3 "$CODE_DIR/pipeline/postprocess/synthesize_report.py" \
    --summary-dir "$SUMMARY_DIR" \
    --out-dir "$REPORT_DIR" \
    "${CASE_ARGS[@]}" \
    || { echo "[FAIL] synthesize_report.py failed"; exit 2; }

NUM_REPORT=$(find "$REPORT_DIR" -maxdepth 1 -name '*.txt' | wc -l)
echo "[PASS] $NUM_REPORT report(s) -> $REPORT_DIR"

if [ -n "${NO_SURVEY:-}" ]; then
    echo "[INFO] NO_SURVEY=1 -- stopping after the reports."
    exit 0
fi

# ── Stage 3: survey the summaries against the reference reports ────────────
#
# Text and JSON both land under $SURVEY_DIR/, timestamped, so successive runs
# are directly diffable against each other -- that comparison is the whole
# reason to re-run this after a flag change.

echo ""
echo "[INFO] --- Stage 3/3: structured_findings_evaluation.py ---"
if [ ! -d "$GT_DIR" ]; then
    echo "[FAIL] Ground truth not found: $GT_DIR"
    echo "[FAIL] Generate it with parse_reports_to_gt.py first."
    echo "[FAIL] Summaries and reports ARE built -- only the survey was skipped."
    exit 2
fi
NUM_GT=$(find "$GT_DIR" -maxdepth 1 -name '*_gt.json' | wc -l)
echo "[INFO] $NUM_GT ground-truth file(s) in $GT_DIR"

mkdir -p "$SURVEY_DIR"
# A DIFFERENT prefix from the deleted survey_findings.py's `survey_*`, deliberately. Those
# files score absence against the segmentation mask and the findings against the
# hand-coded REPORT_GT; these score everything against the generated GT. Sharing
# a prefix would diff the two against each other and report every row as changed
# on the first run, which is noise, not a result.
SURVEY_TXT="$SURVEY_DIR/survey_facts_${STAMP}.txt"
SURVEY_JSON="$SURVEY_DIR/survey_facts_${STAMP}.json"

set +e
python3 "$CODE_DIR/eval/structured_findings_evaluation.py" "$RUN_DIR" \
    --gt-dir "$GT_DIR" \
    --schema "$PROJECT_DIR/schema/schema.json" \
    --json-out "$SURVEY_JSON" \
    "${CASE_ARGS[@]}" \
    ${SURVEY_ARGS:-} 2>&1 | tee "$SURVEY_TXT"
STATUS=${PIPESTATUS[0]}
set -e
if [ "$STATUS" -ne 0 ]; then
    echo "[FAIL] structured_findings_evaluation.py exited $STATUS"
    echo "[FAIL] Summaries and reports ARE built -- only the survey failed."
    exit 2
fi
echo ""
echo "[PASS] Survey -> $SURVEY_TXT"
echo "[PASS] Survey -> $SURVEY_JSON"

PREV=$(find "$SURVEY_DIR" -maxdepth 1 -name 'survey_facts_*.txt' | sort | tail -2 | head -1)
if [ -n "$PREV" ] && [ "$PREV" != "$SURVEY_TXT" ]; then
    echo ""
    echo "[INFO] Change against the previous survey ($(basename "$PREV")):"
    diff "$PREV" "$SURVEY_TXT" | head -40 || true
fi

echo ""
echo "[INFO] ========== Complete =========="
echo "[INFO] End: $(date)"
