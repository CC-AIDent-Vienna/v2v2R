#!/bin/bash
# =============================================================================
# scripts/run_sft.sh -- the LoRA SFT pipeline, one stage per submission.
#
#   STAGE=pool                 scripts/run_sft.sh          # CPU, login node
#   STAGE=draft   sbatch ...   scripts/run_sft.sh          # teacher    [GPU]
#   STAGE=screen  sbatch ...   scripts/run_sft.sh          # student    [GPU]
#   STAGE=targets              scripts/run_sft.sh          # CPU, the long pole
#   STAGE=parity  sbatch ...   scripts/run_sft.sh          # student    [GPU]
#   STAGE=train   sbatch ...   scripts/run_sft.sh          # the arm    [GPU]
#   STAGE=merge   sbatch ...   scripts/run_sft.sh          # CPU is fine
#
# Everything after `--` goes to the stage's module verbatim:
#
#   STAGE=train ARM=vision+merger sbatch --partition=gpu --qos=a100 \
#       --gres=gpu:a100:1 scripts/run_sft.sh -- --rows sft_wide.jsonl --epochs 2
#
# THIS FILE IS DELIBERATELY THIN
# ------------------------------
# code/train/sft.py holds the stage table, the vLLM lifecycle and the
# interpreter choice; code/train/README.md says what the modules are. What is
# here is the SLURM directives and the three site paths, because those are the
# only parts a Python entry point cannot decide for itself.
#
# THE THREE ENVIRONMENTS, AND WHY THE STAGE PICKS ONE
# ---------------------------------------------------
#   base        nibabel / PIL / the schema tooling      pool, targets
#   container   vLLM -- exists nowhere else             draft, screen (server
#                                                       AND client)
#   SFT_PY      torch 2.11 / transformers 5.9           parity, train, merge
#
# vLLM 0.19.0 pins torch==2.10.0 and transformers<5 while training needs 2.11.0
# and 5.9.0, which is the single row that forced three environments rather than
# two. sft.py checks the interpreter BEFORE starting a server, so a missing
# training env costs a second instead of a queue wait plus a model load.
#
# ONE STAGE PER SUBMISSION, ON PURPOSE
# ------------------------------------
# They want different cards and different wall clocks -- `targets` is CPU and
# hours, `train` is ~12 h on one a100, `merge` needs no GPU at all -- and each
# is a decision point: the screen's keep-rate is read before targets are built,
# and the parity gate is meant to stop the run. Chaining them would hide both.
# =============================================================================
#SBATCH --job-name=sft
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sft_%j.log
#SBATCH --error=logs/sft_%j.err
#
# --partition/--qos/--gres are not here: they differ per stage and per site.
#   STAGE=train  sbatch --partition=gpu --qos=a100 --gres=gpu:a100:1 scripts/run_sft.sh
#   STAGE=merge  sbatch --partition=cpu --qos=cpu scripts/run_sft.sh

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

STAGE="${STAGE:?set STAGE=pool|targets|draft|screen|parity|train|merge}"
case "$STAGE" in pool|targets|draft|screen|parity|train|merge) ;; *)
    echo "[FAIL] unknown STAGE=$STAGE" >&2; exit 1 ;;
esac

PROJECT_DIR="${PROJECT_DIR:-$(d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; \
    while [ "$d" != "/" ] && [ ! -f "$d/schema/schema.json" ]; do d="$(dirname "$d")"; done; echo "$d")}"
[ -f "$PROJECT_DIR/schema/schema.json" ] || { echo "[FAIL] no schema/schema.json above $0" >&2; exit 1; }

ARGS=(--stage "$STAGE" --project-dir "$PROJECT_DIR"
      --model-dir "${MODEL_DIR:-$PROJECT_DIR/models}"
      --container "${CONTAINER:-${SIF_PATH:-$HOME/containers/vllm019_cu128.sqsh}}"
      --sft-python "${SFT_PY:-$HOME/miniconda3/envs/cbct_sft_cu128/bin/python3}")

[ -n "${TEACHER_MODEL:-}" ] && ARGS+=(--teacher-model "$TEACHER_MODEL")
[ -n "${STUDENT_MODEL:-}" ] && ARGS+=(--student-model "$STUDENT_MODEL")
[ -n "${PORT:-}"          ] && ARGS+=(--port "$PORT")
[ -n "${GPU_MEM_UTIL:-}"  ] && ARGS+=(--gpu-memory-utilization "$GPU_MEM_UTIL")
[ -n "${MAX_MODEL_LEN:-}" ] && ARGS+=(--max-model-len "$MAX_MODEL_LEN")

# The base env holds nibabel and the schema tooling, and `pool`/`targets` are
# the two stages that need it. Sourced by path, never via the `conda` shell
# function: a non-interactive submission never read ~/.bashrc, so that function
# does not exist on the compute node. Job 551011 is what assuming otherwise
# costs. CONDA_ENV= skips activation for a shell already in the right env.
CONDA_ENV="${CONDA_ENV-cbct_base}"
if [ -n "$CONDA_ENV" ]; then
    _conda_sh=""
    command -v conda >/dev/null 2>&1 && \
        _conda_sh="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
    for _cand in "$_conda_sh" "${CONDA_BASE:-}/etc/profile.d/conda.sh" \
                 "$HOME/miniconda3/etc/profile.d/conda.sh" \
                 "$HOME/anaconda3/etc/profile.d/conda.sh" \
                 "$HOME/miniforge3/etc/profile.d/conda.sh" \
                 "/opt/conda/etc/profile.d/conda.sh"; do
        # shellcheck disable=SC1090
        [ -n "$_cand" ] && [ -f "$_cand" ] && { source "$_cand"; break; }
    done
    if command -v conda >/dev/null 2>&1; then
        conda activate "$CONDA_ENV" || { echo "[FAIL] conda activate $CONDA_ENV failed" >&2; exit 1; }
    else
        echo "[WARN] no conda.sh on this node; using $(command -v python3)" >&2
    fi
fi
ARGS+=(--base-python "$(command -v python3)")

mkdir -p "$PROJECT_DIR/logs"
echo "[INFO] stage=$STAGE  base=$(command -v python3)"
exec python3 "$PROJECT_DIR/code/train/sft.py" "${ARGS[@]}" "$@"
