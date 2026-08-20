#!/usr/bin/env bash
# =============================================================================
# code/train/prompt_parity.sh -- docs/vision_sft_plan.md §3.4, and it runs before stage B.
#
# Boots the STUDENT and asks it to tokenize the exact conversation the training
# collator builds, then requires the two token-id sequences to be equal. Not a
# text diff: the three failures this is for -- enable_thinking, the generation
# prompt, the caption/image interleave order -- all render as prompts that look
# right.
#
# WHY THE a30 AND NOT AN A100
# ───────────────────────────
# This job generates NOTHING. One POST /tokenize, a few hundred milliseconds of
# work behind a ~10 min weight load, so throughput is irrelevant and the only
# question is whether 11.53 GiB of AWQ weights fit. They fit 24 GiB with room to
# spare. s0-n12's a30 is the one 24 GB card on the cluster and it is normally
# idle, while every A100 and H100 is allocated with a queue behind it -- so this
# runs now instead of after a wait measured in hours, and leaves the pool for
# the jobs that need it (stage B onward).
#
#   sbatch code/train/prompt_parity.sh                 # 46 then 12, one boot
#   FDIS="46" sbatch code/train/prompt_parity.sh       # just the lower molar
#
# Two shapes by default and both behind one weight load: 46 is a lower molar,
# so its call carries mandible_canal and is the longest shape the collator will
# ever build; 12 has no canal fact. A parity that holds for one field count and
# not the other is worth an extra 2 s of GPU to rule out.
#
# Reads outputs/training_results/vsft_pool_training/sft_wide.jsonl, writes
# nothing but the log.
# =============================================================================
#SBATCH --job-name=parity
#SBATCH --partition=gpu
#SBATCH --qos=a30
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/parity_%j.log
#SBATCH --error=logs/parity_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/project_ToothFairy4}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"
CONTAINER="${SIF_PATH:-$HOME/containers/vllm019_cu128.sqsh}"
STUDENT="${STUDENT:-Qwen3.5-9B-AWQ}"
ROWS="${ROWS:-$PROJECT_DIR/outputs/training_results/vsft_pool_training/sft_wide.jsonl}"
FDIS="${FDIS:-46 12}"
PORT="${PORT:-8000}"
# The training env, BY PATH: the checker needs transformers 5.9.0 (qwen3_5) and
# the container ships 4.x, so this half cannot run inside it.
SFT_PY="${SFT_PY:-$HOME/miniconda3/envs/cbct_sft_cu128/bin/python3}"

cd "$PROJECT_DIR"
[ -f "$ROWS" ]    || { echo "[FAIL] no SFT rows: $ROWS"; exit 1; }
[ -x "$SFT_PY" ]  || { echo "[FAIL] no training interpreter: $SFT_PY"; exit 1; }
[ -f "$CONTAINER" ] || { echo "[FAIL] no container: $CONTAINER"; exit 1; }
[ -d "$MODEL_DIR/$STUDENT" ] || { echo "[FAIL] no model: $MODEL_DIR/$STUDENT"; exit 1; }

echo "[INFO] student=$STUDENT  fdis=$FDIS  rows=$ROWS  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

VLLM_PID=""
cleanup() { [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT

# Same flags the pipeline serves with -- the chat template, the mm limit and the
# context length are all inputs to what /tokenize returns, so a parity check
# against a differently-configured server proves nothing about the real one.
srun --overlap --container-image="$CONTAINER" \
     --container-mounts="$MODEL_DIR:/models" \
     python3 -m vllm.entrypoints.openai.api_server \
        --model "/models/$STUDENT" --served-model-name student \
        --port "$PORT" --max-model-len 32768 --reasoning-parser qwen3 \
        --limit-mm-per-prompt '{"image": 3}' \
        --gpu-memory-utilization 0.85 --enable-prefix-caching &
VLLM_PID=$!

deadline=$(( $(date +%s) + 2400 ))
until curl -s --max-time 5 "http://localhost:${PORT}/health" >/dev/null 2>&1; do
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "[FAIL] vLLM died"; exit 1; }
    [ "$(date +%s)" -lt "$deadline" ] || { echo "[FAIL] vLLM not ready"; exit 1; }
    sleep 1
done
echo "[PASS] student ready: $(date)"

# Every shape is checked even after one fails -- a partial answer here is what
# sends someone back for a second allocation.
status=0
for fdi in $FDIS; do
    echo ""
    echo "── tooth $fdi ──────────────────────────────────────────────────"
    "$SFT_PY" code/train/check_prompt_parity.py \
        --rows "$ROWS" --fdi "$fdi" \
        --model-dir "$MODEL_DIR/$STUDENT" \
        --vllm-url "http://localhost:${PORT}" \
        --save-ids "$PROJECT_DIR/logs/parity_ids_${fdi}_${SLURM_JOB_ID:-local}.json" \
        --dump "$PROJECT_DIR/logs/parity_tooth${fdi}_${SLURM_JOB_ID:-local}.txt" \
        || status=1
done

cleanup
echo "[INFO] End: $(date)  exit=$status"
exit $status
