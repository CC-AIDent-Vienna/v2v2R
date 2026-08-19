#!/usr/bin/env bash
# code/pipeline/preprocess/seed_v6_review_case.sh -- build ONE case's v6 images the way
# NO_FACTS=1 code/pipeline/aksssr_pipeline.sh will, then dump the exact prompts.
#
# Written into the real v6 images directory on purpose: the pipeline's
# per-step "already exists" checks then skip this case, so nothing is
# generated twice and the reviewed prompts are the ones the run will send.
#
# Usage: bash code/pipeline/preprocess/seed_v6_review_case.sh [case_id] [split]
set -euo pipefail
cd "$(d="$(cd "$(dirname "$0")" && pwd)"; while [ "$d" != "/" ] && [ ! -f "$d/schema/schema.json" ]; do d="$(dirname "$d")"; done; echo "$d")"
# Locate conda.sh by path rather than via `conda info --base`: a
# non-interactive shell (ssh host 'bash code/...') never sourced ~/.bashrc, so
# the conda shell function does not exist, and on this cluster `conda` is not
# on PATH even on the login node. Unlike the batch scripts this one at least
# failed loudly (set -e on a failed source), but with a confusing error.
# CONDA_BASE overrides the search.
CONDA_ENV="${CONDA_ENV:-cbct_base}"
_conda_sh=""
if command -v conda >/dev/null 2>&1; then
    _conda_sh="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
fi
for _cand in "$_conda_sh" "${CONDA_BASE:-}/etc/profile.d/conda.sh" \
             "$HOME/miniconda3/etc/profile.d/conda.sh" \
             "$HOME/anaconda3/etc/profile.d/conda.sh" \
             "$HOME/miniforge3/etc/profile.d/conda.sh" \
             "/opt/conda/etc/profile.d/conda.sh"; do
    if [ -n "$_cand" ] && [ -f "$_cand" ]; then
        # shellcheck disable=SC1090
        source "$_cand"
        break
    fi
done
if ! command -v conda >/dev/null 2>&1; then
    echo "[FAIL] no conda.sh found on this node. Set CONDA_BASE=/path/to/conda and retry."
    exit 1
fi
conda activate "$CONDA_ENV" \
    || { echo "[FAIL] conda activate $CONDA_ENV failed"; exit 1; }

CASE="${1:-A008}"
SPLIT="${2:-validate}"
OUT="outputs/aksssr_v6_${SPLIT}"
IMG="$OUT/images"
mkdir -p "$IMG"

VOL="dataset/$SPLIT/images/${CASE}_0000.nii.gz"
MASK="dataset/$SPLIT/masks/${CASE}.nii.gz"

echo "=== [1/5] panoramic (--no-facts) ==="
python3 code/pipeline/preprocess/create_panoramic.py --volume "$VOL" --mask "$MASK" --no-facts \
    --out-dir "$IMG" --case-id "$CASE"
echo "=== [2/5] 3D renders (no facts) ==="
python3 code/pipeline/preprocess/create_3d_renders.py --mask "$MASK" --out-dir "$IMG" --case-id "$CASE"
echo "=== [3/5] tooth composites (no facts) ==="
python3 code/pipeline/preprocess/create_tooth_detail.py --volume "$VOL" --mask "$MASK" \
    --out-dir "$IMG" --case-id "$CASE"
echo "=== [4/5] sinus detail ==="
python3 code/pipeline/preprocess/create_sinus_detail.py --volume "$VOL" --mask "$MASK" \
    --out-dir "$IMG" --case-id "$CASE"

echo "=== [5/5] qa pairs + prompt dump ==="
python3 code/pipeline/preprocess/build_vqa_pairs.py \
    --schema schema/schema.json \
    --images-dir "$IMG" \
    --project-dir "$PWD" \
    --cases "$CASE" \
    --out "$OUT/qa_pairs_review_${CASE}.jsonl"

python3 code/eval/review_qa_pairs.py \
    --qa-jsonl "$OUT/qa_pairs_review_${CASE}.jsonl" \
    --format calls --out "$OUT/model_input_${CASE}.txt"

python3 code/eval/review_qa_pairs.py \
    --qa-jsonl "$OUT/qa_pairs_review_${CASE}.jsonl" \
    --format text --json-schema --out "$OUT/model_input_${CASE}_full.txt"

echo "=== done ==="
ls "$IMG" | wc -l
wc -l "$OUT/model_input_${CASE}.txt" "$OUT/model_input_${CASE}_full.txt"
