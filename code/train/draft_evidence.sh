#!/usr/bin/env bash
# =============================================================================
# code/train/draft_evidence.sh -- the teacher pass (plan §3.3)
#
# Boots the teacher and has it write visual_evidence for every supervised field
# of every tooth call in the SFT pool, given the ESTABLISHED answer. Output is a
# predictions-shaped dir that build_sft_targets.py --evidence-from reads.
#
#   QWEN_MODEL_NAME=Qwen3.5-27B EV_DIR=evidence_27b sbatch code/train/draft_evidence.sh
#
# Speed notes, because the 27B audition ran 2h18 and did not need to:
#   * concurrency is sized from the KV pool the server reports, and the log path
#     comes from scontrol -- guessing it is what pinned the audition to the
#     fallback 8 (d7fb8ac);
#   * --max-tokens 512, not the pipeline's 4096. This writes prose, not a
#     report, and decode dominates wall-clock at these batch sizes;
#   * a 52 GB bf16 27B on an 80 GB card starves its own batch -- ~20 GB of KV
#     left. Qwen3.5-27B-FP8 halves the weights, roughly doubles the pool, and
#     H100 does FP8 natively. Same script, different QWEN_MODEL_NAME.
# =============================================================================
#SBATCH --job-name=draft_ev
#SBATCH --partition=gpu
#SBATCH --qos=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/draft_ev_%j.log
#SBATCH --error=logs/draft_ev_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/project_ToothFairy4}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"
CONTAINER="${SIF_PATH:-$HOME/containers/vllm019_cu128.sqsh}"
# Without this, draft_evidence.py's [INFO] lines sit in a block buffer until
# the process exits: on job 555799 the scope line -- "N tooth call(s) to draft,
# M case(s)" -- was unreadable for the whole 5h41m run, and the run had to be
# reconstructed from its output files instead. Progress was only ever visible
# because those prints go to stderr.
export PYTHONUNBUFFERED=1
QWEN_MODEL_NAME="${QWEN_MODEL_NAME:?set QWEN_MODEL_NAME}"
EV_DIR="${EV_DIR:-evidence_$(echo "$QWEN_MODEL_NAME" | tr 'A-Z.' 'a-z_')}"
QA_JSONL="${QA_JSONL:-$PROJECT_DIR/outputs/training_results/vsft_pool_training/qa_pairs.jsonl}"
GT_DIR="${GT_DIR:-$PROJECT_DIR/dataset/training/outputs/ground_truth}"
CASE_LIST="${CASE_LIST:-$PROJECT_DIR/outputs/training_results/sft_pool/wide.txt}"
OUT_DIR="$PROJECT_DIR/outputs/training_results/vsft_pool_training/$EV_DIR"
PORT="${PORT:-8000}"
TOKENS_PER_REQUEST="${TOKENS_PER_REQUEST:-6200}"
MAX_CONCURRENCY_CAP="${MAX_CONCURRENCY_CAP:-32}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
# 32768 is the model's context, not this job's need: a tooth call is ~6,200
# tokens in and --max-tokens out. On a 40 GB card the 27B leaves ~10 GB after
# weights, and asking vLLM to reserve KV for 32k-token requests inside that
# OOMs during CUDA-graph capture (job 555794, "tried to allocate 1.53 GiB,
# 695 MiB free"). Raising --gpu-memory-utilization makes it worse rather than
# better: it hands MORE of the budget to the KV pool and leaves less for the
# graphs, which are allocated after it.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# Skips CUDA-graph capture entirely -- a few GB back, ~10-20% slower decode.
# Worth it only when the alternative is not running at all.
#
# FOR Qwen3.5-27B-FP8 ON A 40 GB a100 THE ALTERNATIVE IS NOT RUNNING AT ALL.
# Weights are 28.87 GiB of a 39.49 GiB card. Job 555799 succeeded with
# ENFORCE_EAGER=1; job 556042 was the same command with the flag left off and
# died in 6 minutes -- OOM inside profile_cudagraph_memory, "tried to allocate
# 1.53 GiB, 693.50 MiB free", the same shape as 555794. The known-good set for
# that model on that card is, all three together:
#
#   ENFORCE_EAGER=1 MAX_MODEL_LEN=12288 CONCURRENCY=8
#
# It is left defaulting to empty because an 80 GB card does not need it and
# pays 10-20% decode for nothing.
ENFORCE_EAGER="${ENFORCE_EAGER:-}"
MAX_TOKENS="${MAX_TOKENS:-512}"

if [ -z "${JOB_LOG:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    JOB_LOG=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
              | grep -oP "StdOut=\K\S+" || true)
fi
JOB_LOG="${JOB_LOG:-$PROJECT_DIR/logs/draft_ev_${SLURM_JOB_ID:-local}.log}"

cd "$PROJECT_DIR"
[ -d "$MODEL_DIR/$QWEN_MODEL_NAME" ] || { echo "[FAIL] no model: $QWEN_MODEL_NAME"; exit 1; }
echo "[INFO] teacher=$QWEN_MODEL_NAME  out=$OUT_DIR  Start: $(date)"

VLLM_PID=""
cleanup() { [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT

srun --overlap --container-image="$CONTAINER" \
     --container-mounts="$MODEL_DIR:/models,$PROJECT_DIR:/project" \
     python3 -m vllm.entrypoints.openai.api_server \
        --model "/models/$QWEN_MODEL_NAME" --served-model-name teacher \
        --port "$PORT" --max-model-len "$MAX_MODEL_LEN" --reasoning-parser qwen3 \
        ${ENFORCE_EAGER:+--enforce-eager} \
        --limit-mm-per-prompt '{"image": 3}' \
        --gpu-memory-utilization "$GPU_MEM_UTIL" --enable-prefix-caching &
VLLM_PID=$!

deadline=$(( $(date +%s) + 2700 ))
until curl -s --max-time 5 "http://localhost:${PORT}/health" >/dev/null 2>&1; do
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "[FAIL] vLLM died"; exit 1; }
    [ "$(date +%s)" -lt "$deadline" ] || { echo "[FAIL] vLLM not ready"; exit 1; }
    sleep 1
done
echo "[PASS] vLLM ready: $(date)"

# CONCURRENCY is a RACE, not a parse. /health answered at 23:13:20 on job 555799
# while the engine was still choosing an attention backend at 23:14:21 -- so a
# single grep straight after the health check reads a log that does not yet
# contain the KV line, silently takes the fallback 8, and turns a 36-minute
# draft into a 2h18 one. That is what pinned the audition too; it looked like a
# path bug and was fixed as one (d7fb8ac), which is why it came back.
#
# So poll for the LINE, not for the port, with a bound: readiness for this
# purpose means "the server has reported its KV pool", which is strictly later
# than "the server accepts connections". Also read stderr, since which stream
# vLLM's startup lands on depends on how srun buffers it.
#
# 180 s WAS TOO SHORT AND IT COST 546 BLOCKS (job 555799). /health answered at
# 23:13:20 -- the same second the job started, i.e. against something that was
# not yet this job's server -- the engine reported its KV pool at 23:17:05, and
# the poll had already given up at 23:16:20.
#
# What draft_evidence.py then hit is worth naming exactly, because it is not
# what "not ready" sounds like: 546 NotFoundError, i.e. HTTP 404. The port was
# open and answering; the model `teacher` was simply not registered yet. A
# connection check cannot see that state at all. Final tally 632 failures =
# 546 NotFoundError + 86 JSONDecodeError, and zero NoImage.
#
# So: a bound long enough for a cold 27B on squashfs, and the absence of the
# line is now FATAL. The old `[WARN] ... concurrency 8` fallback is what turned
# a race into lost data -- it kept going, and looked fine.
kv=""
kv_deadline=$(( $(date +%s) + ${KV_WAIT:-900} ))
while [ "$(date +%s)" -lt "$kv_deadline" ]; do
    kv=$(grep -ho "GPU KV cache size: [0-9,]* tokens" "$JOB_LOG" "${JOB_LOG%.log}.err" 2>/dev/null \
         | tail -1 | grep -o "[0-9,]*" | tr -d ',' || true)
    [ -n "$kv" ] && break
    sleep 5
done
# CONCURRENCY= overrides the DERIVED NUMBER ONLY. It deliberately does not skip
# the wait above: that wait is the readiness gate, not merely an input to this
# arithmetic, and forcing a value past a server that is still registering its
# model is precisely how 546 blocks were lost.
#
# Forcing is the right move more often than it looks. The pool-derived value is
# a floor, not a capacity: 555799 ran at the fallback 8 against an 18,816-token
# pool that this formula scores as 3, and sat at 60% KV usage -- because 6,200
# tokens is the prompt, and prefix caching plus staggered completion means the
# eight are never all resident at once.
if [ -n "${CONCURRENCY:-}" ]; then
    CONC="$CONCURRENCY"; echo "[INFO] concurrency $CONC (forced)"
elif [ -n "$kv" ] && [ "$kv" -gt 0 ] 2>/dev/null; then
    CONC=$(( kv / TOKENS_PER_REQUEST )); [ "$CONC" -lt 1 ] && CONC=1
    [ "$CONC" -gt "$MAX_CONCURRENCY_CAP" ] && CONC=$MAX_CONCURRENCY_CAP
    echo "[INFO] concurrency $CONC (KV pool ${kv} tok)"
else
    echo "[FAIL] the server never reported its KV pool within ${KV_WAIT:-900}s."
    echo "       Refusing to draft: this is the state in which requests are sent"
    echo "       to an engine that is not accepting them, and the failures are"
    echo "       silent (draft_evidence.py counts them and carries on)."
    echo "       Force it with CONCURRENCY=N if you know the server is up."
    exit 1
fi

srun --overlap --container-image="$CONTAINER" \
     --container-mounts="$MODEL_DIR:/models,$PROJECT_DIR:/project" \
     python3 /project/code/train/draft_evidence.py \
        --qa-jsonl "${QA_JSONL/$PROJECT_DIR//project}" \
        --gt-dir "${GT_DIR/$PROJECT_DIR//project}" \
        --case-list "${CASE_LIST/$PROJECT_DIR//project}" \
        --out-dir "${OUT_DIR/$PROJECT_DIR//project}" \
        --base-dir /project --vllm-url "http://localhost:${PORT}/v1" --model teacher \
        --max-tokens "$MAX_TOKENS" --concurrency "$CONC"

echo "[INFO] End: $(date)"
