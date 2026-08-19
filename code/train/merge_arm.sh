#!/usr/bin/env bash
# =============================================================================
# code/train/merge_arm.sh -- a trained adapter -> a servable AWQ checkpoint
#
#   ADAPTER=outputs/training_results/vsft_arm6/weights/adapter \
#   OUT_MODEL=models/Qwen3.5-9B-AWQ-arm6 sbatch code/train/merge_arm.sh
#
# WHY THIS EXISTS AS A SCRIPT
# ---------------------------
# code/train/merge_vision_lora.py has always been run by hand, from the srun line in
# its own docstring. That was fine while a merge followed a training run the
# same afternoon; it is not fine as the middle link of a dependency chain,
# because `--dependency=afterok:<train>` needs something to point at. Written
# 2026-08-16 for arm 6, and it is the same command the docstring gives.
#
# CPU, NOT GPU. Nothing is instantiated: shards in, shards out, no model class,
# no quantizer, no CUDA. Peak RAM is one shard, ~3 GB, and 2 of the 5 AWQ shards
# hold the tensors an arm can touch -- the other 3 are hardlinked and never
# read. ~1-3 min. It would fit the login node's 8 GB cgroup, but the cpu queue
# is the safe habit and costs nothing.
#
# THE INTERPRETER IS cbct_sft_cu128, NOT cbct_base, and that is not a
# preference: cbct_base has no `safetensors`. The merge script's own guards
# (per-tensor base equality, all-zero lora_B, alpha sanity) do the verifying --
# this wrapper only refuses to overwrite a model directory that already exists,
# because a half-written checkpoint that is complete to `ls` and wrong to vLLM
# is the one failure the script cannot catch after the fact.
#
# WHAT THIS DOES NOT PROVE. That the merged tensors are arithmetically right is
# checked; that the SERVED model behaves like the adapter-applied one is not.
# That check has NEVER been run, for any arm, and cannot be run as things
# stand: nothing in this repo applies a LoRA adapter at load time, so the
# reference half of the diff has no harness. Arms 1, 2, 5 and 6 were all scored
# without it. Audited 2026-08-17 -- see docs/vision_sft_plan_light.md §9 for what
# building it would take. Do not read the merge script's closing note as a
# chore someone forgot; read it as a gap.
# =============================================================================
#SBATCH --job-name=merge_arm
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/merge_arm_%j.log
#SBATCH --error=logs/merge_arm_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/V2V2R_ToothFairy4}"
ADAPTER="${ADAPTER:?set ADAPTER=<dir with adapter_model.safetensors>}"
OUT_MODEL="${OUT_MODEL:?set OUT_MODEL=<models/Qwen3.5-9B-AWQ-armN>}"
# The two bases are merge_vision_lora.py's own defaults (bf16 to fit against,
# AWQ to serve); named here only so the existence check can fail before a
# 30-minute allocation rather than inside it. Pass them through if an arm ever
# needs different ones.
TRAIN_BASE="${TRAIN_BASE:-$PROJECT_DIR/models/Qwen3.5-9B}"
TARGET_BASE="${TARGET_BASE:-$PROJECT_DIR/models/Qwen3.5-9B-AWQ}"
SFT_PY="${SFT_PY:-$HOME/miniconda3/envs/cbct_sft_cu128/bin/python3}"
export PYTHONUNBUFFERED=1

cd "$PROJECT_DIR"
[ -d "$ADAPTER" ]   || { echo "[FAIL] no adapter dir: $ADAPTER"; exit 1; }
[ -f "$ADAPTER/adapter_model.safetensors" ] || {
    echo "[FAIL] no adapter_model.safetensors in $ADAPTER"; exit 1; }
[ -d "$TRAIN_BASE" ]  || { echo "[FAIL] no train base: $TRAIN_BASE"; exit 1; }
[ -d "$TARGET_BASE" ] || { echo "[FAIL] no target base: $TARGET_BASE"; exit 1; }
[ -x "$SFT_PY" ]    || { echo "[FAIL] no interpreter: $SFT_PY"; exit 1; }
# Refuse rather than overwrite. Re-merging onto a populated directory would
# leave whichever shards this run did not rewrite from the PREVIOUS arm, and
# nothing downstream could tell the mixture from a clean checkpoint.
[ -e "$OUT_MODEL" ] && { echo "[FAIL] $OUT_MODEL exists -- move it aside "; exit 1; }

echo "[INFO] adapter=$ADAPTER"
echo "[INFO] train-base=$TRAIN_BASE  target-base=$TARGET_BASE"
echo "[INFO] out=$OUT_MODEL  Start: $(date)"

"$SFT_PY" code/train/merge_vision_lora.py --adapter "$ADAPTER" --out "$OUT_MODEL" \
    --train-base "$TRAIN_BASE" --target-base "$TARGET_BASE"

echo ""
echo "[INFO] End: $(date)"
