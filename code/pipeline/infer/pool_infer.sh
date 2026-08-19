#!/usr/bin/env bash
# =============================================================================
# code/pipeline/infer/pool_infer.sh -- run ANY model over an EXISTING payload, and score it
#
# One model, one qa_pairs.jsonl, the full call the pipeline actually sends,
# scored per field against the generated GT. That is the whole job.
#
# WHY THIS IS NOT code/pipeline/aksssr_pipeline.sh
# ---------------------------------------
# That script is organised around a SPLIT: it discovers cases, renders images,
# rebuilds qa_pairs.jsonl and then infers. Here the payload already exists --
# outputs/training_results/vsft_pool_training/qa_pairs.jsonl, the 120-case SFT
# pool, tooth calls only -- and the variable under test is the MODEL. Rebuilding
# the payload per model would reintroduce the one thing every arm comparison
# must not have: a difference in what was asked.
#
# WHY THE FULL CALL AND NOT ONE FIELD
# -----------------------------------
# An earlier probe (b05c6ca, since deleted) asked a single question with a
# three-key schema of its own. It answered its question, but its numbers
# described a prompt that exists nowhere in production: the pipeline sends five
# facts per tooth -- six on the lower molars -- decoded in one constrained
# object, and fields condition each other in that order. A model that reads
# post_and_core well in isolation and badly beside eruption_state and
# bone_loss has not been measured by the isolated test. So this runs the real
# call, unchanged, through the same run_vqa_inference.py the pipeline uses.
#
# WHAT IT IS FOR
#   * teacher audition -- can the candidate teacher actually do this task,
#     BLIND, before it is trusted to write visual_evidence for answers it is
#     handed (plan §3.3). A teacher that cannot see a finding will still write
#     confident prose about it, because DRAFT_SYSTEM tells it not to re-judge.
#   * the student's own baseline on the same pool, which the same command
#     produces by changing one variable.
#
#   QWEN_MODEL_NAME=Qwen3.5-27B  RUN_NAME=teacher27b  sbatch code/pipeline/infer/pool_infer.sh
#   QWEN_MODEL_NAME=Qwen3.5-9B   RUN_NAME=student9b   sbatch code/pipeline/infer/pool_infer.sh
#
# Then read the two survey_facts tables side by side, per field. Size is not
# the answer to "is this a good teacher"; the [detail] rows are.
# =============================================================================
#SBATCH --job-name=pool_infer
#SBATCH --partition=gpu
#SBATCH --qos=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/pool_infer_%j.log
#SBATCH --error=logs/pool_infer_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/V2V2R_ToothFairy4}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"
CONTAINER="${SIF_PATH:-$HOME/containers/vllm019_cu128.sqsh}"
export PYTHONUNBUFFERED=1     # see the note in code/train/draft_evidence.sh
QWEN_MODEL_NAME="${QWEN_MODEL_NAME:?set QWEN_MODEL_NAME}"
RUN_NAME="${RUN_NAME:?set RUN_NAME}"
QA_JSONL="${QA_JSONL:-$PROJECT_DIR/outputs/training_results/vsft_pool_training/qa_pairs.jsonl}"
GT_DIR="${GT_DIR:-$PROJECT_DIR/dataset/training/outputs/ground_truth}"
OUT_DIR="$PROJECT_DIR/outputs/${RUN_NAME}"
PORT="${PORT:-8000}"
# Sized from the server's own KV pool, like aksssr_pipeline.sh step 7b. A 27B
# teacher leaves a much smaller pool than a 9B student, so a constant here
# would be wrong for one of the two runs by construction.
TOKENS_PER_REQUEST="${TOKENS_PER_REQUEST:-6200}"
MAX_CONCURRENCY_CAP="${MAX_CONCURRENCY_CAP:-40}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
CONDA_ENV="${CONDA_ENV:-cbct_base}"
# The job's OWN stdout file, resolved from Slurm rather than guessed from the
# job name: --output is routinely overridden per candidate (aud_9b_%j.log,
# aud_27b_%j.log), and a guessed path silently misses, which drops concurrency
# to the fallback 8 and triples the wall clock without failing anything. Both
# 2026-08-13 auditions ran that way.
if [ -z "${JOB_LOG:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    JOB_LOG=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
              | grep -oP "StdOut=\\K\\S+" || true)
fi
JOB_LOG="${JOB_LOG:-$PROJECT_DIR/logs/pool_infer_${SLURM_JOB_ID:-local}.log}"

cd "$PROJECT_DIR"
mkdir -p "$OUT_DIR"
[ -f "$QA_JSONL" ] || { echo "[FAIL] no payload: $QA_JSONL"; exit 1; }
[ -d "$MODEL_DIR/$QWEN_MODEL_NAME" ] || { echo "[FAIL] no model: $MODEL_DIR/$QWEN_MODEL_NAME"; exit 1; }

echo "[INFO] model=$QWEN_MODEL_NAME  payload=$QA_JSONL ($(wc -l < "$QA_JSONL") case records)  out=$OUT_DIR"
echo "[INFO] Start: $(date)"

VLLM_PID=""
cleanup() { [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT

srun --overlap --container-image="$CONTAINER" \
     --container-mounts="$MODEL_DIR:/models,$PROJECT_DIR:/project" \
     python3 -m vllm.entrypoints.openai.api_server \
        --model "/models/$QWEN_MODEL_NAME" --served-model-name qwen-under-test \
        --port "$PORT" --max-model-len 32768 --reasoning-parser qwen3 \
        --limit-mm-per-prompt '{"image": 3}' \
        --gpu-memory-utilization "$GPU_MEM_UTIL" --enable-prefix-caching &
VLLM_PID=$!

deadline=$(( $(date +%s) + 2700 ))
until curl -s --max-time 5 "http://localhost:${PORT}/health" >/dev/null 2>&1; do
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "[FAIL] vLLM died"; tail -60 "$JOB_LOG" || true; exit 1; }
    [ "$(date +%s)" -lt "$deadline" ] || { echo "[FAIL] vLLM not ready"; exit 1; }
    sleep 1
done
echo "[PASS] vLLM ready: $(date)"

# THE LOG PATH WAS ONLY HALF THE BUG. Resolving JOB_LOG from scontrol fixed a
# guessed path; it did not fix the RACE. /health answers before the engine has
# reported its KV pool -- on job 555799 the port was up at 23:13:20 and the KV
# line appeared at 23:17:05 -- so a single grep straight after the health check
# reads a log that does not contain the line yet, takes the fallback 8, and is
# indistinguishable from the path bug this comment used to describe. Worse, in
# that window the model is not registered either, and requests come back 404.
#
# So poll for the LINE, bounded, across .log and .err, and refuse if it never
# arrives rather than proceeding at a guess.
kv=""
kv_deadline=$(( $(date +%s) + ${KV_WAIT:-900} ))
while [ "$(date +%s)" -lt "$kv_deadline" ]; do
    kv=$(grep -ho "GPU KV cache size: [0-9,]* tokens" "$JOB_LOG" "${JOB_LOG%.log}.err" 2>/dev/null \
         | tail -1 | grep -o "[0-9,]*" | tr -d ',' || true)
    [ -n "$kv" ] && break
    sleep 5
done
if [ -n "${CONCURRENCY:-}" ]; then
    CONC="$CONCURRENCY"; echo "[INFO] concurrency $CONC (forced)"
elif [ -n "$kv" ] && [ "$kv" -gt 0 ] 2>/dev/null; then
    CONC=$(( kv / TOKENS_PER_REQUEST )); [ "$CONC" -lt 1 ] && CONC=1
    [ "$CONC" -gt "$MAX_CONCURRENCY_CAP" ] && CONC=$MAX_CONCURRENCY_CAP
    echo "[INFO] concurrency $CONC (KV pool ${kv} tok / $TOKENS_PER_REQUEST)"
else
    echo "[FAIL] the server never reported its KV pool within ${KV_WAIT:-900}s."
    echo "       Refusing to infer at a guessed concurrency against an engine"
    echo "       that may not be accepting requests yet."
    exit 1
fi

srun --overlap --container-image="$CONTAINER" \
     --container-mounts="$MODEL_DIR:/models,$PROJECT_DIR:/project" \
     python3 /project/code/pipeline/infer/run_vqa_inference.py \
        --vqa-jsonl "${QA_JSONL/$PROJECT_DIR//project}" \
        --out-dir "/project/outputs/${RUN_NAME}/predictions" \
        --base-dir /project --model qwen-under-test \
        --vllm-url "http://localhost:${PORT}/v1" \
        --max-concurrency "$CONC"
cleanup

for base in "$HOME/miniconda3" "$HOME/miniforge3" /opt/conda; do
    [ -f "$base/etc/profile.d/conda.sh" ] && { . "$base/etc/profile.d/conda.sh"; break; }
done
conda activate "$CONDA_ENV"

python3 code/pipeline/postprocess/postprocess_pred.py --pred-dir "$OUT_DIR/predictions" \
        --out-dir "$OUT_DIR/summaries"
mkdir -p "$OUT_DIR/survey"
python3 code/eval/survey_facts.py "$OUT_DIR" --gt-dir "$GT_DIR" \
        --schema schema/schema.json \
        --json-out "$OUT_DIR/survey/survey_facts.json" \
        | tee "$OUT_DIR/survey/survey_facts.txt"

echo "[INFO] End: $(date)"
