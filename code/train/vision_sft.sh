#!/usr/bin/env bash
# =============================================================================
# code/train/vision_sft.sh -- docs/vision_sft_plan_stale.md stage B onward. STAGE picks the step.
#
#   STAGE=probe sbatch code/train/vision_sft.sh          # open item 3 / R3, ~2 min
#   STAGE=smoke sbatch code/train/vision_sft.sh          # stage B: overfit 200 calls
#   STAGE=train ARM=vision+merger sbatch code/train/vision_sft.sh    # stage D
#
# WHY probe AND smoke ARE ONE JOB BY DEFAULT (STAGE=b)
# ────────────────────────────────────────────────────
# They ask different questions of the same loaded model, and the load is the
# expensive part. `probe` answers "do gradients reach the arm on the AWQ
# checkpoint" (§4.3 -- if they do, R3 stops existing rather than needing a
# gate); `smoke` answers "can the rig overfit 200 calls" and measures the step
# time §4.4 could only do arithmetic for. A probe that fails makes the smoke run
# meaningless, so it runs FIRST and the script stops on it.
#
# THE ARM IS DECLARED, NOT DEFAULTED, FOR STAGE=train. Arms 2 and 3 differ by
# 1.5% in trainable parameters and that contrast is the experiment; a default
# would let one of them run under the wrong name.
# =============================================================================
# THE CARD: a100 (40 GB), which is what §4.5's table asks for -- and the queue
# is the reason, not a change of mind about the memory budget.
#
# The earlier default was the h100, on the argument that 80 GB takes the memory
# question off stage B's list entirely: §4.4 budgets 25-28 GB against a 40 GB
# card with the GatedDeltaNet backward as the least predictable term, and an OOM
# there costs a whole allocation to learn something a bigger card measures for
# free. That argument is still sound and still the fallback. What overruled it
# on 2026-08-14 is arithmetic of a different kind: all 4 h100s were held by jobs
# with 22 h to 2 days left and the estimated start was 46 h out, against ~2 h for
# an a100. A two-hour job is not worth a two-day wait to de-risk a term the run
# reports anyway.
#
# So the memory question comes BACK onto stage B's list, deliberately, and the
# run answers it either way: `[INFO] peak GPU memory N GiB` prints on success,
# and an OOM is itself the measurement that §4.4's arithmetic was optimistic.
# If it OOMs, resubmit unchanged with:
#     sbatch --qos=a100-sxm4-80gb --gres=gpu:a100-sxm4-80gb:1 code/train/vision_sft.sh
#
# The step time it measures is now honest for stage D's a100 ask rather than
# optimistic (open item 5) -- the one thing the queue bought us.
#
# --time IS SIZED FOR STAGE D, NOT FOR THE PROBE. It was 02:00:00 while this
# script's job was stage B on a 120-case pool. Full scope is 10,791 rows at
# grad_accum 16 = ~675 optimizer steps, and the measured step time is 36.0 s,
# so one epoch is ~6.75 h -- a 2 h wall clock kills it around step 200 with a
# half-trained adapter and no checkpoint worth keeping. 12 h leaves room for
# arm 5's slightly heavier backward and for a slower card than the a100 the
# 36.0 s was measured on. STAGE=probe and STAGE=smoke still finish in minutes;
# an over-long --time costs them nothing but a slightly worse queue position.
#
# RAISED TO 20 h ON 2026-08-16, for the arch rows. sft_wide_arch.jsonl adds the
# nine `global` calls per case that every arm before it trained without, and
# under train_minus_heldout.txt that is 10,207 rows against arm 5's 6,162.
# Sized from arm 5's own run rather than from the 36.0 s figure, which was
# measured on a smoke set: job 556129 did 707 steps in 31,440 s including five
# eval passes, i.e. ~41.9 s/step. Here EVAL_CASES=30 leaves ~9,658 training
# rows = ~1,207 steps, and the arch rows are SHORTER than the tooth ones (mean
# prompt 3,456 tokens against 5,224), so the mix averages ~4,523 -- call it
# 0.9x the per-step cost and the estimate is ~12.6 h of steps, ~13.5 h with
# eval and load. 20 h is that with room for the discount being wrong: at no
# discount at all it is still ~14.9 h. It is NOT sized for a card slower than
# an a100; on one, re-estimate rather than trusting this number.
#SBATCH --job-name=vsft
#SBATCH --partition=gpu
#SBATCH --qos=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/vsft_%j.log
#SBATCH --error=logs/vsft_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/project_ToothFairy4}"
STAGE="${STAGE:-b}"
ARM="${ARM:-vision+merger}"
ROWS="${ROWS:-$PROJECT_DIR/outputs/training_results/vsft_pool_training/sft_wide.jsonl}"
# TRAIN ON bf16, MERGE INTO AWQ. The deployed model is still Qwen3.5-9B-AWQ;
# this is where the ADAPTER is learned, and the two are the same weights where
# it matters. Verified tensor by tensor, not assumed: all 142 tensors the three
# arms target are BIT-IDENTICAL between models/Qwen3.5-9B and
# models/Qwen3.5-9B-AWQ (merger 2, vision blocks 108, lm self_attn 32; zero
# missing, zero differing). AWQ quantizes only mlp.{down,up,gate}_proj in
# layers 1-31 and copies everything else verbatim, so a LoRA trained here
# merges into the AWQ checkpoint exactly.
#
# The reason is that loading AWQ for TRAINING drags in gptqmodel, and gptqmodel
# is at war with this architecture: it monkey-patches Triton's autotuner
# globally at import, and Qwen3.5's GatedDeltaNet layers run on fla's Triton
# kernels, whose CachedAutotuner subclass then dies on a _cache_lock that only
# gptqmodel's patched __init__ sets. Nothing about that is the arm's problem.
# Dropping AWQ from the training path also drops transformers' forced fp16
# downcast ("bfloat16 is not supported for AWQ CUDA kernels yet") and gives
# back the bf16 §4.4 budgets for.
#
# What this DOES cost: the forward now runs bf16 MLPs where deployment runs
# 4-bit ones. That gap is not new and it is not unmeasured -- it is exactly
# what §4.3's transfer gate exists to check, and it is now load-bearing rather
# than a formality. Weights are ~19 GB instead of ~12 GB, so the 80 GB card
# matters more than it did.
MODEL="${MODEL:-$PROJECT_DIR/models/Qwen3.5-9B}"
OUT="${OUT:-$PROJECT_DIR/outputs/vsft_${STAGE}_${ARM//+/_}}"
CASE_LIST="${CASE_LIST:-}"
LIMIT="${LIMIT:-200}"
EPOCHS="${EPOCHS:-2}"
# Cases held out of TRAINING to give an eval-loss curve (STAGE=train only --
# a smoke run's job is to overfit, and an eval split on 200 rows measures
# nothing). 0 keeps the old blind behaviour. 30 of 558 is ~5% of rows; the
# same 30 on the old 112-case pool would have been 23%, so this number is
# sized for full scope and is wrong for a small one.
EVAL_CASES="${EVAL_CASES:-0}"
EVIDENCE_WEIGHT="${EVIDENCE_WEIGHT:-0.04}"
SFT_PY="${SFT_PY:-$HOME/miniconda3/envs/cbct_sft_cu128/bin/python3}"
# Python block-buffers stdout when it is a file rather than a tty, so every
# {'loss': ...} and {'eval_loss': ...} line sits in an 8 KB buffer instead of
# reaching the log. At ~90 chars a line that is ~90 lines, i.e. the first flush
# lands somewhere past step 450 of 708 -- and on a job that is killed or OOMs
# the buffer is simply lost. Job 556129 ran this way: tqdm's progress bar was
# live on stderr the whole time while the eval-loss curve the run exists to
# produce was invisible. One export, and the log is readable as it happens.
export PYTHONUNBUFFERED=1

cd "$PROJECT_DIR"
[ -f "$ROWS" ]   || { echo "[FAIL] no rows: $ROWS"; exit 1; }
[ -x "$SFT_PY" ] || { echo "[FAIL] no training interpreter: $SFT_PY"; exit 1; }
[ -d "$MODEL" ]  || { echo "[FAIL] no model: $MODEL"; exit 1; }

echo "[INFO] stage=$STAGE arm=$ARM model=$(basename "$MODEL") Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

common=(--arm "$ARM" --rows "$ROWS" --model "$MODEL"
        --evidence-weight "$EVIDENCE_WEIGHT")
[ -n "$CASE_LIST" ] && common+=(--case-list "$CASE_LIST")

run_probe() {
    echo ""
    echo "── probe: do gradients reach the arm? (§4.3, open item 3) ──────"
    "$SFT_PY" code/train/train_vision_lora.py "${common[@]}" \
        --out "$OUT" --probe-backward
}

run_smoke() {
    echo ""
    echo "── smoke: overfit $LIMIT call(s), measure step time (stage B) ──"
    # More epochs on purpose: the question is whether the rig CAN drive the
    # loss down on a small set, not whether this is a good recipe. A rig that
    # cannot overfit 200 calls has a defect that 3,500 calls will only hide.
    "$SFT_PY" code/train/train_vision_lora.py "${common[@]}" \
        --out "$OUT" --limit "$LIMIT" --epochs "${SMOKE_EPOCHS:-6}" \
        --workers 6
}

case "$STAGE" in
    probe) run_probe ;;
    smoke) run_smoke ;;
    b)     run_probe && run_smoke ;;
    train)
        "$SFT_PY" code/train/train_vision_lora.py "${common[@]}" \
            --out "$OUT" --epochs "$EPOCHS" --workers 6 \
            --eval-cases "$EVAL_CASES"
        ;;
    *) echo "[FAIL] unknown STAGE=$STAGE (probe|smoke|b|train)"; exit 1 ;;
esac

echo ""
echo "[INFO] End: $(date)"
