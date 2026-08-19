#!/usr/bin/env bash
# =============================================================================
# gen_ground_truth.sh — Generate ground truth from clinical reports
#
# Handles multiple reports per case (different radiologists). Extracts GT
# from each report separately, optionally computes consensus.
#
# TWO-STAGE EXTRACTION (parse_reports_to_gt.py, current design)
#   stage 1 (LLM, needs this job's GPU)  report text -> {case}_report_facts.json,
#           a small report-shaped intermediate. TWO calls per report (mandible,
#           maxilla), not the 34 the per-fact/per-tooth design needed.
#   stage 2 (pure CPU, no model)         report_facts -> {case}_gt.json, the full
#           schema shape, derived deterministically so the arch map, the
#           per-tooth facts and the wisdom-tooth facts cannot contradict
#           each other the way the LLM-answers-everything version did.
#   Stage 2 alone re-runs anywhere, no GPU and no sbatch:
#     python code/ground_truth/parse_reports_to_gt.py --from-report-facts \
#         --reports-dir dataset/$SPLIT/reports --schema schema/schema.json \
#         --out-dir dataset/$SPLIT/outputs/ground_truth
#   Both files land in $GT_DIR; only *_gt.json is read downstream.
#
# ARCHITECTURE NOTE -- this rewrite matches parse_reports_to_gt.py's actual
# design, which differs from what this script used to assume:
#   - HTTP call to an ALREADY-RUNNING vLLM server (--vllm-url/--model), NOT
#     an offline tensor-parallel engine loaded directly inside the
#     extraction script (--model-dir/--model-name/--tensor-parallel never
#     existed as real flags on that script). This matches every other
#     model-calling script in this project (run_vqa_inference.py) --
#     one server, started once, called over HTTP.
#   - Output is one JSON object per case (or per case+radiologist), not
#     one JSONL line per finding -- there is no "agreement_ratio" grep
#     target anymore; per-field agreement now lives in a structured
#     "_agreement" block inside the consensus JSON file itself.
#
# Report naming conventions:
#   Single report per case:   A003.txt
#   Multiple reports:         A003_1.txt, A003_2.txt (or A003_doctor1.txt)
#
# Output (all under $OUT_DIR/ground_truth/):
#   Single-report case:            {case_id}_gt.json
#   Multi-report, per radiologist: {case_id}_{radiologist}_gt.json
#   Multi-report consensus:        {case_id}_gt.json (only written with --consensus)
#
# Usage:
#   sbatch code/ground_truth/gen_ground_truth.sh <training|validate>
#   sbatch code/ground_truth/gen_ground_truth.sh validate --consensus   # Include consensus
#   LIMIT=2 sbatch code/ground_truth/gen_ground_truth.sh training        # Smoke test (cases, not files)
#   DRY_RUN=1 sbatch code/ground_truth/gen_ground_truth.sh training      # Validate only, no LLM calls
#   RESUME=1 sbatch code/ground_truth/gen_ground_truth.sh training       # Skip cases already done
#
#   # One report per case (cheapest path to one {case_id}_gt.json per case),
#   # then score a prediction dir against it -- per-category accuracy:
#   FIRST_REPORT_ONLY=1 EVAL_PRED_DIR=outputs/aksssr_v5_validate/predictions \
#       sbatch code/ground_truth/gen_ground_truth.sh validate
#
# WHY FIRST_REPORT_ONLY MATTERS FOR EVALUATION
#   evaluate_predictions.py looks up exactly one file per case:
#   {case_id}_gt.json. A multi-report case with neither --consensus nor
#   --first-report-only writes ONLY {case_id}_{radiologist}_gt.json, so that
#   case is skipped with no error and the run reports a plausible score over
#   a subset. On dataset/validate that is 26 of 40 cases.
#
#SBATCH --job-name=gen_gt
#SBATCH --partition=gpu
#SBATCH --qos=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gen_gt_%j.log
#SBATCH --error=logs/gen_gt_%j.err

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# ── Configuration ──────────────────────────────────────────────────────────

PROJECT_DIR="${PROJECT_DIR:-$HOME/V2V2R_ToothFairy4}"
CODE_DIR="$PROJECT_DIR/code"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"
CONTAINER="${SIF_PATH:-$HOME/containers/extraction.sqsh}"
QWEN_MODEL_NAME="${QWEN_MODEL_NAME:-Qwen3-14B}"
# 8011, NOT 8001 -- code/eval/judge_server.sh also defaults to 8001 and can be
# running on the same node (both take --qos=a100). vLLM binds its socket with
# SO_REUSEPORT, so the second server does NOT fail with "address already in
# use": both bind and the kernel round-robins connections between them. The
# judge serves the model under the name `qwen3-14b-text` and this job asks for
# `qwen3-text`, so roughly half of every extraction call landed on the judge
# and came back "404 Not Found" -- see logs/gen_gt_549316.log (342 x 200 here,
# 379 x 404 in logs/judge_server_549303.log at the same timestamps). A report
# needs its calls to all succeed and any single failure aborts that reader, so
# at a ~50% per-call collision rate essentially no case survived. That was 34
# calls per case then and is 2 per report now, which shortens the exposure but
# does not remove it. PORT_SCAN below is the belt-and-braces guard for the same
# class of collision with any other job.
PORT="${PORT:-8011}"           # different default port than the VQA VL model's server (8000) and the judge (8001)
PORT_SCAN="${PORT_SCAN:-16}"   # how many consecutive ports to try before giving up

# 24576, not 32768: --qos=a100 can land on s0-n00/s0-n01, which hold A100-40GB
# cards. There, Qwen3-14B's 27.5 GiB of weights plus CUDA graphs and activation
# overhead leave only ~3.9 GiB for the KV cache at --gpu-memory-utilization
# 0.85, while 32768 tokens need 5.0 GiB -- vLLM then refuses to start at all
# ("estimated maximum model length is 25424", see logs/gen_gt_548408.log).
# 24576 fits under that ceiling. It is also far more than this job needs:
# build_extraction_prompt() sends one section's fact specs plus a single
# report, so the worst case across every report in both splits is ~2.4k
# prompt tokens + max_tokens=2000 output = ~4.5k. The two-stage rewrite sends
# one report-facts spec (~1.5k tokens) plus the report instead, so the ceiling
# only moved down.
# On an 80GB card there is room to spare, so pin one and restore the full
# context with:
#   MAX_MODEL_LEN=32768 sbatch --qos=a100-sxm4-80gb \
#       --gres=gpu:a100-sxm4-80gb:1 code/ground_truth/gen_ground_truth.sh <split>
MAX_MODEL_LEN="${MAX_MODEL_LEN:-24576}"

# Required: "training" or "validate".
SPLIT="${1:?Usage: sbatch code/ground_truth/gen_ground_truth.sh <training|validate> [--consensus]}"
if [ "$SPLIT" != "training" ] && [ "$SPLIT" != "validate" ]; then
    echo "[FAIL] SPLIT must be 'training' or 'validate', got: $SPLIT"
    exit 1
fi
shift || true

BASE_DIR="$PROJECT_DIR/dataset/$SPLIT"
REPORTS_DIR="$BASE_DIR/reports"
SCHEMA="$PROJECT_DIR/schema/schema.json"
OUT_DIR="$BASE_DIR/outputs"
GT_DIR="$OUT_DIR/ground_truth"

# Optional flags
LIMIT="${LIMIT:-}"
CASE_IDS="${CASE_IDS:-}"
DRY_RUN="${DRY_RUN:-}"
RESUME="${RESUME:-}"
CONSENSUS="${CONSENSUS:-}"

# FIRST_REPORT_ONLY=1 -- extract from one report per case only (the
# lowest-numbered radiologist), written straight to {case_id}_gt.json.
# Without it AND without --consensus, a multi-report case writes only
# {case_id}_{radiologist}_gt.json, which evaluate_predictions.py never looks
# up -- on validate that silently drops 26 of 40 cases from the evaluation.
FIRST_REPORT_ONLY="${FIRST_REPORT_ONLY:-}"

# EVAL_PRED_DIR=<dir> -- after extraction, score that prediction dir against
# the ground truth just built (per-category accuracy, evaluate_predictions.py).
# Pure CPU, no model, so it runs on the same allocation after vLLM is stopped.
# Empty (the default) keeps this script's original GT-only behaviour.
EVAL_PRED_DIR="${EVAL_PRED_DIR:-}"
EVAL_OUT="${EVAL_OUT:-}"

# --consensus can also be passed as a positional flag after the split, to
# match how this script has always been invoked.
for arg in "$@"; do
    if [ "$arg" = "--consensus" ]; then
        CONSENSUS="1"
    fi
done

mkdir -p logs "$GT_DIR"

# Activate conda. Do NOT reach it through `conda info --base`: a
# non-interactive submission (ssh host 'sbatch ...') never sourced ~/.bashrc,
# so the conda shell function does not exist on the compute node, and on this
# cluster `conda` is not on PATH even on the login node -- the guard below
# would silently skip activation and leave python3 as /usr/bin/python3. See
# job 551011 for what that costs. CONDA_BASE overrides the search.
# Same fix as code/pipeline/preprocess/gen_images_cpu.sh.
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
    echo "[FAIL] no conda.sh found on this node. Set CONDA_BASE=/path/to/conda and resubmit."
    exit 1
fi
conda activate "$CONDA_ENV" \
    || { echo "[FAIL] conda activate $CONDA_ENV failed"; exit 1; }
echo "[INFO] python3: $(command -v python3)"

# ── Main ───────────────────────────────────────────────────────────────────

{
    echo "[INFO] ========== Generate Ground Truth (Multi-Report) =========="
    echo "[INFO] Start: $(date)"
    echo "[INFO] Job ID: ${SLURM_JOB_ID:-local}"
    echo "[INFO] Split: $SPLIT"
    echo "[INFO] Output dir: $GT_DIR"
    echo "[INFO] Model: $QWEN_MODEL_NAME"
    [ -n "$LIMIT" ] && echo "[INFO] LIMIT: $LIMIT (cases, not report files)"
    [ -n "$CASE_IDS" ] && echo "[INFO] CASE_IDS: $CASE_IDS"
    [ -n "$DRY_RUN" ] && echo "[INFO] DRY-RUN: ON (no LLM calls)"
    [ -n "$RESUME" ] && echo "[INFO] RESUME: ON"
    [ -n "$CONSENSUS" ] && echo "[INFO] CONSENSUS: ON"
    echo ""

    # ── Sanity Checks ──────────────────────────────────────────────────────

    if [ ! -f "$SCHEMA" ]; then
        echo "[FAIL] Schema not found: $SCHEMA"
        exit 1
    fi
    echo "[INFO] Schema: $SCHEMA"

    if [ ! -d "$REPORTS_DIR" ]; then
        echo "[FAIL] Reports directory not found: $REPORTS_DIR"
        echo "[HINT] Expected dataset/$SPLIT/reports/ to hold the reference reports"
        exit 1
    fi

    num_reports=$(find "$REPORTS_DIR" -maxdepth 1 -name "*.txt" | wc -l)
    echo "[INFO] Found $num_reports report file(s) in $REPORTS_DIR"

    if [ ! -f "$CODE_DIR/ground_truth/parse_reports_to_gt.py" ]; then
        echo "[FAIL] Extraction script not found: $CODE_DIR/ground_truth/parse_reports_to_gt.py"
        exit 1
    fi

    # ── Build Optional Arguments ───────────────────────────────────────────

    EXTRA_ARGS=()
    [ -n "$LIMIT" ] && EXTRA_ARGS+=(--limit "$LIMIT")
    [ -n "$CASE_IDS" ] && EXTRA_ARGS+=(--case-ids $CASE_IDS)
    [ -n "$RESUME" ] && EXTRA_ARGS+=(--resume)
    [ -n "$CONSENSUS" ] && EXTRA_ARGS+=(--consensus)
    [ -n "$FIRST_REPORT_ONLY" ] && EXTRA_ARGS+=(--first-report-only)

    if [ -n "$FIRST_REPORT_ONLY" ] && [ -n "$CONSENSUS" ]; then
        echo "[FAIL] FIRST_REPORT_ONLY and CONSENSUS are mutually exclusive."
        exit 1
    fi

    # ── DRY-RUN BRANCH: no vLLM server needed at all ────────────────────────
    # parse_reports_to_gt.py's own --dry-run just groups/validates report
    # files and prints the plan -- no model call, so no reason to pay for
    # a GPU allocation's vLLM startup time just to validate inputs.

    if [ -n "$DRY_RUN" ]; then
        echo "[DRY-RUN] Validating report grouping and schema (no vLLM server, no LLM calls)"
        python3 "$CODE_DIR/ground_truth/parse_reports_to_gt.py" \
            --reports-dir "$REPORTS_DIR" \
            --schema "$SCHEMA" \
            --out-dir "$GT_DIR" \
            "${EXTRA_ARGS[@]}" \
            --dry-run
        echo ""
        echo "[INFO] ========== Dry-run Complete =========="
        echo "[INFO] End: $(date)"
        exit 0
    fi

    # ── GPU Check ────────────────────────────────────────────────────────────

    if [ ! -f "$CONTAINER" ]; then
        echo "[FAIL] Container not found: $CONTAINER"
        exit 1
    fi
    if [ ! -d "$MODEL_DIR/$QWEN_MODEL_NAME" ]; then
        echo "[FAIL] Model not found: $MODEL_DIR/$QWEN_MODEL_NAME"
        exit 1
    fi

    echo "[INFO] Checking GPU..."
    set +o pipefail
    nvidia-smi | head -10
    set -o pipefail
    echo ""

    # ── Start vLLM Server (text-only Qwen3-14B, same lifecycle pattern as
    #    aksssr_pipeline.sh's VL model server) ──────────────────────────────

    VLLM_PID=""
    cleanup() {
        if [ -n "$VLLM_PID" ]; then
            echo "[INFO] Stopping vLLM (PID $VLLM_PID)..."
            kill "$VLLM_PID" 2>/dev/null || true
            wait "$VLLM_PID" 2>/dev/null || true
            VLLM_PID=""
        fi
    }
    trap cleanup EXIT

    # ── Pick a port nobody else is already serving on ────────────────────
    # SO_REUSEPORT means a busy port does not announce itself at bind time,
    # so probe it BEFORE starting: anything that answers /health owns the
    # port, and we move on to the next one.
    start_port="$PORT"
    port_found=0
    for _ in $(seq 1 "$PORT_SCAN"); do
        if curl -s --max-time 3 "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "[WARN] Port $PORT is already serving (another job on $(hostname)); trying $((PORT + 1))"
            PORT=$((PORT + 1))
        else
            port_found=1
            break
        fi
    done
    if [ "$port_found" -eq 0 ]; then
        echo "[FAIL] No free port in ${start_port}..$((start_port + PORT_SCAN - 1)) on $(hostname)."
        echo "[HINT] Another vLLM job is likely hogging this node. Check 'squeue -u $USER'"
        echo "[HINT] and either wait for it, or rerun with PORT=<free port>."
        exit 1
    fi

    echo "[INFO] Starting vLLM server (text model)..."
    echo "  Model: $QWEN_MODEL_NAME"
    echo "  Port: $PORT"
    echo ""

    srun \
        --overlap \
        --container-image="$CONTAINER" \
        --container-mounts="$MODEL_DIR:/models" \
        python3 -m vllm.entrypoints.openai.api_server \
            --model "/models/$QWEN_MODEL_NAME" \
            --served-model-name qwen3-text \
            --port "$PORT" \
            --dtype bfloat16 \
            --max-model-len "$MAX_MODEL_LEN" \
            --reasoning-parser qwen3 \
            --gpu-memory-utilization 0.85 \
            --enable-prefix-caching &

    VLLM_PID=$!
    echo "[INFO] vLLM PID: $VLLM_PID"
    echo ""

    echo "[INFO] Waiting for vLLM to load (A100: 10-20 min for a 14B text model)..."
    sleep 60

    echo "[INFO] Health check loop..."
    SERVER_READY=0
    TIMEOUT=120  # 120 x 10s = 20 min total

    for i in $(seq 1 $TIMEOUT); do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[FAIL] vLLM process died at iteration $i"
            exit 1
        fi
        # /v1/models, not /health: /health proves *a* server is on this port,
        # not that it is OURS serving qwen3-text.
        if curl -s --max-time 5 "http://localhost:${PORT}/v1/models" 2>/dev/null \
             | grep -q '"id"[[:space:]]*:[[:space:]]*"qwen3-text"'; then
            elapsed=$((i * 10 + 60))
            echo "[PASS] vLLM ready after ${elapsed}s (iteration $i/$TIMEOUT)"
            SERVER_READY=1
            break
        fi
        if [ $((i % 6)) -eq 0 ]; then
            elapsed=$((i * 10 + 60))
            echo "  Waiting... (iteration $i/$TIMEOUT, ~$(( (elapsed + 30) / 60 )) min elapsed)"
        fi
        sleep 10
    done

    if [ $SERVER_READY -eq 0 ]; then
        echo "[FAIL] vLLM never became ready after $((TIMEOUT * 10 + 60))s"
        exit 1
    fi

    # One /v1/models probe can pass by luck if a second server co-bound this
    # port between the preflight scan and now -- connections are handed out
    # round-robin, so ask repeatedly and require every answer to be ours.
    for probe in $(seq 1 8); do
        if ! curl -s --max-time 5 "http://localhost:${PORT}/v1/models" 2>/dev/null \
               | grep -q '"id"[[:space:]]*:[[:space:]]*"qwen3-text"'; then
            echo "[FAIL] Port $PORT is shared with another vLLM server (probe $probe/8"
            echo "       did not answer as qwen3-text). Extraction calls would be"
            echo "       split between the two servers and 404 on the wrong one."
            echo "[HINT] Rerun with an explicit free PORT, or wait for the other job."
            exit 1
        fi
    done
    echo "[PASS] Port $PORT verified exclusive to this job's qwen3-text server"
    echo ""

    # ── Run Ground Truth Extraction (plain host python3, calls the vLLM
    #    server over HTTP -- NOT run inside the container, since this script
    #    only makes network calls, no model-loading of its own) ────────────

    echo "[INFO] Running ground truth extraction..."
    python3 "$CODE_DIR/ground_truth/parse_reports_to_gt.py" \
        --reports-dir "$REPORTS_DIR" \
        --schema "$SCHEMA" \
        --out-dir "$GT_DIR" \
        --vllm-url "http://localhost:${PORT}/v1" \
        --model qwen3-text \
        "${EXTRA_ARGS[@]}" \
        || { echo "[FAIL] Extraction failed"; exit 1; }

    echo ""
    echo "[INFO] ========== Generation Complete =========="
    echo "[INFO] End: $(date)"
    echo ""

    # ── Summary ──────────────────────────────────────────────────────────────
    # Output shape changed from one-JSONL-line-per-finding to one JSON
    # object per case (or per case+radiologist) -- no more agreement_ratio
    # grep; per-field agreement now lives inside each consensus file's own
    # "_agreement" block, inspected directly per case if needed.

    num_single_or_consensus=$(find "$GT_DIR" -maxdepth 1 -name "*_gt.json" ! -regex '.*_[^_]*_gt\.json$' | wc -l)
    num_individual=$(find "$GT_DIR" -maxdepth 1 -regex '.*_[^_]*_gt\.json$' | wc -l)

    echo "[INFO] Output files in $GT_DIR:"
    echo "  - Single-report / consensus case files: $num_single_or_consensus"
    echo "  - Individual per-radiologist files: $num_individual"

    if [ "$num_individual" -gt 0 ] && [ -z "$CONSENSUS" ]; then
        echo ""
        echo "[INFO] Multi-report cases were extracted individually but no consensus was"
        echo "[INFO] computed (pass --consensus to combine them into one {case_id}_gt.json)."
    fi
    echo ""

    # ── Per-category accuracy (optional; EVAL_PRED_DIR=<dir>) ────────────────
    # evaluate_predictions.py needs no GPU and no model -- it walks the schema
    # comparing each {case_id}_pred.json against {case_id}_gt.json. Stop vLLM
    # first so the GPU is released while this CPU-only stage runs.

    if [ -z "$EVAL_PRED_DIR" ]; then
        echo "[INFO] EVAL_PRED_DIR not set -- skipping per-category evaluation."
        echo ""
    else
        cleanup   # release the GPU before the CPU-only stage

        case "$EVAL_PRED_DIR" in
            /*) : ;;
            *) EVAL_PRED_DIR="$PROJECT_DIR/$EVAL_PRED_DIR" ;;
        esac
        [ -z "$EVAL_OUT" ] && EVAL_OUT="$OUT_DIR/evaluation/evaluation_$(date +%Y%m%d_%H%M%S).json"
        mkdir -p "$(dirname "$EVAL_OUT")"

        echo "[INFO] ========== Per-Category Evaluation =========="
        echo "[INFO] Predictions: $EVAL_PRED_DIR"
        echo "[INFO] Ground truth: $GT_DIR"
        echo "[INFO] Output: $EVAL_OUT"

        if [ ! -d "$EVAL_PRED_DIR" ]; then
            echo "[FAIL] Prediction dir not found: $EVAL_PRED_DIR"
            echo "[FAIL] Ground truth WAS written to $GT_DIR -- rerun the evaluation alone with:"
            echo "[FAIL]   python code/eval/evaluate_predictions.py --pred-dir <dir> \\"
            echo "[FAIL]       --gt-dir $GT_DIR --schema $SCHEMA --out $EVAL_OUT"
            exit 1
        fi

        NUM_PREDS=$(find "$EVAL_PRED_DIR" -maxdepth 1 -name "*_pred.json" | wc -l)
        NUM_GT=$(find "$GT_DIR" -maxdepth 1 -name "*_gt.json" ! -regex '.*_[^_]*_gt\.json$' | wc -l)
        echo "[INFO] $NUM_PREDS prediction(s) vs $NUM_GT case-level GT file(s)"
        if [ "$NUM_PREDS" -eq 0 ]; then
            echo "[FAIL] No *_pred.json in $EVAL_PRED_DIR -- did the v5 pipeline finish?"
            exit 1
        fi
        # Every prediction without a matching {case_id}_gt.json is skipped
        # silently by evaluate_predictions.py, which then reports a healthy
        # score over a fraction of the set. Name them here instead.
        MISSING=0
        for p in "$EVAL_PRED_DIR"/*_pred.json; do
            cid=$(basename "$p" _pred.json)
            [ -f "$GT_DIR/${cid}_gt.json" ] || { echo "[WARN] no GT for $cid"; MISSING=$((MISSING + 1)); }
        done
        # A warning is not enough here. Job 549067 extracted GT for only 2 of
        # 36 cases (every other case died on a per-case [ERROR]), then scored
        # those 2, printed [PASS] and exited 0 -- a green job whose evaluation
        # covered 6% of the split. Partial coverage must be opted into.
        if [ "$MISSING" -gt 0 ]; then
            echo "[WARN] $MISSING/$NUM_PREDS prediction(s) have no case-level GT."
            if [ -z "${ALLOW_PARTIAL_GT:-}" ]; then
                echo "[FAIL] Refusing to report an evaluation over $((NUM_PREDS - MISSING))/$NUM_PREDS cases."
                echo "[FAIL] Search this log for '[ERROR]' for the per-case extraction cause."
                echo "[FAIL] Set ALLOW_PARTIAL_GT=1 to score the available cases anyway."
                exit 1
            fi
            echo "[WARN] ALLOW_PARTIAL_GT=1 -- scoring the remaining $((NUM_PREDS - MISSING)) case(s) only."
        fi
        echo ""

        STATUS=0
        python3 "$CODE_DIR/eval/evaluate_predictions.py" \
            --pred-dir "$EVAL_PRED_DIR" \
            --gt-dir "$GT_DIR" \
            --schema "$SCHEMA" \
            --out "$EVAL_OUT" \
            ${CASE_IDS:+--case-ids $CASE_IDS} || STATUS=$?

        echo ""
        if [ "$STATUS" -ne 0 ]; then
            echo "[FAIL] evaluate_predictions.py exited $STATUS"
            echo "[FAIL] Ground truth is intact in $GT_DIR -- only the scoring step failed."
            exit 1
        fi
        echo "[PASS] Per-category evaluation written to $EVAL_OUT"
        echo ""
    fi

    echo "[INFO] ========== Job Complete =========="
    echo "[INFO] End: $(date)"

} 2>&1