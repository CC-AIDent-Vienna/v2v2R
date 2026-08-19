#!/usr/bin/env bash
# =============================================================================
# code/train/screen_evidence.sh -- step 1 of two: can the STUDENT see it?
#
# Boots the STUDENT (Qwen3.5-9B-AWQ, the model that will actually be trained and
# deployed) and asks it, per drafted evidence string, whether the features that
# string names are visible in that image. Never the teacher: the teacher wrote
# the prose, so asking it confirms nothing.
#
# Then runs build_sft_targets.py against the FILTERED evidence, so the audit at
# the end is of the targets that would really be trained on.
#
#   EV_DIR=evidence_27b sbatch code/train/screen_evidence.sh
#
# TWO DEFAULTS HERE POINT AT THE 120-CASE TRIAL POOL, AND BOTH FEED STEP 2.
# CASE_LIST defaults to wide.txt and QA_JSONL to vsft_pool_training's payload,
# which after the rescope holds 110 records against a 582-case corpus. Neither
# is only the screen's input: build_sft_targets.py at the bottom takes both and
# overwrites sft_wide.jsonl. Left at the defaults, this screens what it can,
# writes targets for a fifth of the corpus, and exits 0.
#
# That is not hypothetical -- job 556051 was submitted with CASE_LIST corrected
# and QA_JSONL left alone, and was killed during model load. The guard below
# exists because a comment did not catch it.
#
#   EV_DIR=evidence_27b \
#   CASE_LIST=$PWD/outputs/training_results/sft_pool/all_582.txt \
#   QA_JSONL=$PWD/outputs/vsft_full_training/qa_pairs.jsonl \
#       sbatch code/train/screen_evidence.sh
# =============================================================================
#SBATCH --job-name=screen_ev
#SBATCH --partition=gpu
#SBATCH --qos=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/screen_ev_%j.log
#SBATCH --error=logs/screen_ev_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/V2V2R_ToothFairy4}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"
CONTAINER="${SIF_PATH:-$HOME/containers/vllm019_cu128.sqsh}"
export PYTHONUNBUFFERED=1     # see the note in code/train/draft_evidence.sh
STUDENT="${STUDENT:-Qwen3.5-9B-AWQ}"
POOL="$PROJECT_DIR/outputs/training_results/vsft_pool_training"
EV_DIR="${EV_DIR:-evidence_27b}"
QA_JSONL="${QA_JSONL:-$POOL/qa_pairs.jsonl}"
GT_DIR="${GT_DIR:-$PROJECT_DIR/dataset/training/outputs/ground_truth}"
CASE_LIST="${CASE_LIST:-$PROJECT_DIR/outputs/training_results/sft_pool/wide.txt}"
PORT="${PORT:-8000}"
MIN_VISIBLE="${MIN_VISIBLE:-0.5}"
TOKENS_PER_REQUEST="${TOKENS_PER_REQUEST:-6200}"
MAX_CONCURRENCY_CAP="${MAX_CONCURRENCY_CAP:-40}"
CONDA_ENV="${CONDA_ENV:-cbct_base}"

if [ -z "${JOB_LOG:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    JOB_LOG=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
              | grep -oP "StdOut=\K\S+" || true)
fi
JOB_LOG="${JOB_LOG:-$PROJECT_DIR/logs/screen_ev_${SLURM_JOB_ID:-local}.log}"

cd "$PROJECT_DIR"
[ -d "$POOL/$EV_DIR" ] || { echo "[FAIL] no drafted evidence: $POOL/$EV_DIR"; exit 1; }

# The payload must at least cover the case list. A payload SMALLER than the
# list is the silent-shrink failure above: every later stage is keyed off what
# the payload contains, so a short one narrows the corpus without narrowing any
# of the numbers that get reported. A payload LARGER than the list is fine and
# ordinary -- the list is then a deliberate subset.
n_payload=$(wc -l < "$QA_JSONL")
n_list=$(grep -cvE '^\s*(#|$)' "$CASE_LIST" 2>/dev/null || echo 0)
echo "[INFO] payload $n_payload record(s) vs case list $n_list case(s)"
if [ "$n_payload" -lt "$n_list" ] && [ -z "${ALLOW_PARTIAL:-}" ]; then
    echo "[FAIL] payload has $n_payload record(s) but the case list names $n_list."
    echo "       $QA_JSONL"
    echo "       $CASE_LIST"
    echo "       This is how a full-scope run quietly becomes a trial-pool run:"
    echo "       build_sft_targets.py below reads the same payload and would"
    echo "       overwrite sft_wide.jsonl with the smaller corpus. Point"
    echo "       QA_JSONL at the payload built for this case list, or set"
    echo "       ALLOW_PARTIAL=1 if the subset is deliberate."
    exit 1
fi
echo "[INFO] student=$STUDENT  evidence=$EV_DIR  min-visible=$MIN_VISIBLE  Start: $(date)"

VLLM_PID=""
cleanup() { [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT

srun --overlap --container-image="$CONTAINER" \
     --container-mounts="$MODEL_DIR:/models,$PROJECT_DIR:/project" \
     python3 -m vllm.entrypoints.openai.api_server \
        --model "/models/$STUDENT" --served-model-name student \
        --port "$PORT" --max-model-len 32768 --reasoning-parser qwen3 \
        --limit-mm-per-prompt '{"image": 3}' \
        --gpu-memory-utilization 0.90 --enable-prefix-caching &
VLLM_PID=$!

deadline=$(( $(date +%s) + 2700 ))
until curl -s --max-time 5 "http://localhost:${PORT}/health" >/dev/null 2>&1; do
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "[FAIL] vLLM died"; exit 1; }
    [ "$(date +%s)" -lt "$deadline" ] || { echo "[FAIL] vLLM not ready"; exit 1; }
    sleep 1
done
echo "[PASS] student ready: $(date)"

# THE SAME RACE draft_evidence.sh HIT, AND IT COSTS THE SAME WAY. /health
# answering does not mean the engine is up: on job 555799 the port answered at
# 23:13:20 while the model was still being registered, and the 546 requests
# sent in that window came back 404 and were counted as failures, not retried.
# A single grep straight after the health check reads a log that does not yet
# contain the KV line. So poll for the LINE, with a bound, and read stderr too
# -- which stream vLLM's startup lands on depends on how srun buffers it.
kv=""
kv_deadline=$(( $(date +%s) + ${KV_WAIT:-900} ))
while [ "$(date +%s)" -lt "$kv_deadline" ]; do
    kv=$(grep -ho "GPU KV cache size: [0-9,]* tokens" "$JOB_LOG" "${JOB_LOG%.log}.err" 2>/dev/null \
         | tail -1 | grep -o "[0-9,]*" | tr -d ',' || true)
    [ -n "$kv" ] && break
    sleep 5
done
# CONCURRENCY= overrides the derived number only; it never skips the wait above,
# because that wait is the readiness gate rather than an input to this sum.
if [ -n "${CONCURRENCY:-}" ]; then
    CONC="$CONCURRENCY"; echo "[INFO] concurrency $CONC (forced)"
elif [ -n "$kv" ] && [ "$kv" -gt 0 ] 2>/dev/null; then
    CONC=$(( kv / TOKENS_PER_REQUEST )); [ "$CONC" -lt 1 ] && CONC=1
    [ "$CONC" -gt "$MAX_CONCURRENCY_CAP" ] && CONC=$MAX_CONCURRENCY_CAP
    echo "[INFO] concurrency $CONC (KV pool ${kv} tok)"
else
    echo "[FAIL] the student never reported its KV pool within ${KV_WAIT:-900}s."
    echo "       Refusing to screen: that is the state in which requests go to"
    echo "       an engine that is not accepting them, and the failures are"
    echo "       silent. Force with CONCURRENCY=N if you know it is up."
    exit 1
fi

# ── 1. screen ──────────────────────────────────────────────────────────────
srun --overlap --container-image="$CONTAINER" \
     --container-mounts="$MODEL_DIR:/models,$PROJECT_DIR:/project" \
     python3 /project/code/train/check_evidence_perceivable.py \
        --evidence-dir "/project/outputs/training_results/vsft_pool_training/$EV_DIR" \
        --qa-jsonl "${QA_JSONL/$PROJECT_DIR//project}" \
        --out-dir "/project/outputs/training_results/vsft_pool_training/${EV_DIR}_screened" \
        --report "/project/outputs/training_results/vsft_pool_training/${EV_DIR}_perceivable.json" \
        --base-dir /project --vllm-url "http://localhost:${PORT}/v1" --model student \
        --min-visible "$MIN_VISIBLE" --concurrency "$CONC"
cleanup

# ── 2. targets, from the SCREENED evidence ─────────────────────────────────
for base in "$HOME/miniconda3" "$HOME/miniforge3" /opt/conda; do
    [ -f "$base/etc/profile.d/conda.sh" ] && { . "$base/etc/profile.d/conda.sh"; break; }
done
conda activate "$CONDA_ENV"

python3 code/train/build_sft_targets.py \
    --qa-jsonl "$QA_JSONL" --gt-dir "$GT_DIR" --case-list "$CASE_LIST" \
    --evidence-from "$POOL/${EV_DIR}_screened" \
    --out "$POOL/sft_wide.jsonl"

echo "[INFO] End: $(date)"
