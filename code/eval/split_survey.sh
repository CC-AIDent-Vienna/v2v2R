#!/usr/bin/env bash
# =============================================================================
# code/eval/split_survey.sh -- one prediction set, three surveys, by what the arm saw
#
# A survey over all 582 training cases is dominated by cases the arm memorised,
# so its headline number measures fit, not skill. But the same predictions also
# contain two clean slices, and splitting them costs no GPU and no re-inference:
#
#   trained   497 cases  -- in the arm's training rows. Memorisation.
#   evalonly   30 cases  -- the --eval-cases split. Seen only as eval loss,
#                           never trained on, never used to pick a checkpoint.
#   unseen     55 cases  -- in the 582-case payload, in no training row at all.
#
# `unseen` is the honest number, and at 55 cases it is 2.5x the 22-case held-out
# set every arm has been compared on. The gap between `trained` and `unseen` is
# a direct read on how much of an arm's gain is memorisation -- the question the
# aggregate cannot answer and the held-out set is too small to answer sharply.
#
# `evalonly` is the control between them: if it tracks `unseen`, eval loss was
# an honest generalisation signal; if it tracks `trained`, then watching it
# during training leaked more than intended.
#
# CPU only, seconds. structured_findings_evaluation.py takes --case-ids, so nothing is copied or
# symlinked and all three read the one predictions dir.
#
#   code/eval/split_survey.sh outputs/training_results/vsft_arm5/train_arm5
#
# Case lists come from the run's own train_meta.json, written next to it as
# cases_{trained,evalonly,unseen}.txt -- never retyped, so a survey cannot be
# labelled with a split it does not describe.
# =============================================================================
set -euo pipefail

RUN_DIR="${1:?usage: split_survey.sh <run_dir>}"
ARM_DIR="${ARM_DIR:-$(dirname "$RUN_DIR")}"
GT_DIR="${GT_DIR:-dataset/training/outputs/ground_truth}"
SCHEMA="${SCHEMA:-schema/schema.json}"

[ -d "$RUN_DIR/predictions" ] || { echo "[FAIL] no predictions in $RUN_DIR"; exit 1; }
mkdir -p "$RUN_DIR/survey"

for split in trained evalonly unseen; do
    list="$ARM_DIR/cases_${split}.txt"
    [ -f "$list" ] || { echo "[FAIL] no case list: $list"; exit 1; }
    n=$(grep -cvE '^\s*(#|$)' "$list")
    # Only cases that actually have a prediction -- a list naming cases the run
    # never wrote would silently score a smaller split than its label claims.
    have=0
    while read -r c; do
        [ -n "$c" ] && [ -f "$RUN_DIR/predictions/${c}_pred.json" ] && have=$((have + 1))
    done < "$list"
    echo ""
    echo "===== $split -- $have of $n case(s) present ====="
    [ "$have" -eq 0 ] && { echo "[WARN] nothing to score, skipping"; continue; }
    python3 code/eval/structured_findings_evaluation.py "$RUN_DIR" \
        --gt-dir "$GT_DIR" --schema "$SCHEMA" \
        --case-ids $(grep -vE '^\s*(#|$)' "$list" | tr '\n' ' ') \
        --json-out "$RUN_DIR/survey/survey_${split}.json" \
        | tee "$RUN_DIR/survey/survey_${split}.txt"
done

echo ""
echo "[PASS] three surveys -> $RUN_DIR/survey/survey_{trained,evalonly,unseen}.{txt,json}"
echo "[INFO] read 'unseen' as the arm's real number; trained-minus-unseen is the memorisation gap."
