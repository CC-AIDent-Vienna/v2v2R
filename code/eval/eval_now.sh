#!/usr/bin/env bash
# =============================================================================
# eval_now.sh — run the official ranking against an ALREADY-RUNNING judge.
#
# Companion to code/eval/judge_server.sh. That job holds a vLLM RadFact judge on a
# GPU node; this script runs on the LOGIN node, finds it, and evaluates. No
# model load, no sbatch, no waiting -- run it as often as you like.
#
# Usage:
#   code/eval/eval_now.sh <training|validate> [synthesized_reports_dir]
#
#   code/eval/eval_now.sh validate outputs/aksssr_v6_combined_validate/synthesized_reports
#   code/eval/eval_now.sh validate                     # pipeline's default dir
#
# Env overrides:
#   OUT_DIR=...          results location  (default: alongside the reports dir)
#   JUDGE_URL=...        skip auto-discovery, use this URL
#   MULTI_REFERENCE=1    score BLEU/METEOR against EVERY reference report for a
#                        case, not just the first. 59.7% of cases have 2+.
#                        Diagnostic only -- leave off for leaderboard parity.
#   NO_RADFACT=1         BLEU/METEOR only; no judge needed at all.
#   BATCH_SIZE=10        cases per incremental write (default 10)
#   RESUME=1             continue a prior run, skipping completed cases
#
# Exit codes: 1 = bad arguments / missing inputs, 2 = no judge running.
# =============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/project_ToothFairy4}"
cd "$PROJECT_DIR"

SPLIT="${1:?Usage: code/eval/eval_now.sh <training|validate> [synthesized_reports_dir]}"
if [ "$SPLIT" != "training" ] && [ "$SPLIT" != "validate" ]; then
    echo "[FAIL] SPLIT must be 'training' or 'validate', got: $SPLIT"; exit 1
fi

BASE_DIR="$PROJECT_DIR/dataset/$SPLIT"
SYNTH_DIR="${2:-${SYNTH_DIR:-$BASE_DIR/outputs/synthesized_reports}}"
REPORTS_DIR="$BASE_DIR/reports"
OUT_DIR="${OUT_DIR:-$(dirname "$SYNTH_DIR")/official_ranking}"
SERVED_NAME="${SERVED_NAME:-qwen3-14b-text}"

[ -d "$SYNTH_DIR" ]   || { echo "[FAIL] No such reports dir: $SYNTH_DIR"; exit 1; }
[ -d "$REPORTS_DIR" ] || { echo "[FAIL] No reference reports: $REPORTS_DIR"; exit 1; }

# official_ranking.py needs nltk + radfact_lite, which live in cbct_base.
PYTHON="${PYTHON:-$HOME/miniconda3/envs/cbct_base/bin/python3}"
[ -x "$PYTHON" ] || PYTHON="python3"
if ! "$PYTHON" -c "import nltk" 2>/dev/null; then
    echo "[FAIL] nltk not importable by $PYTHON -- activate cbct_base or set PYTHON=..."
    exit 1
fi

ARGS=(--schema "$PROJECT_DIR/schema/schema.json"
      --synthesized-dir "$SYNTH_DIR"
      --reports-dir "$REPORTS_DIR"
      --out-dir "$OUT_DIR"
      --batch-size "${BATCH_SIZE:-10}")
[ -n "${RESUME:-}" ]          && ARGS+=(--resume)
[ -n "${MULTI_REFERENCE:-}" ] && ARGS+=(--multi-reference)
# The normal-finding filter is ON by default in official_ranking.py: normal
# boilerplate is dropped from BOTH sides before entailment, so it cannot prop
# up precision and recall alike. NO_FILTER_NORMAL=1 reproduces the pre-2026-08
# unfiltered scores -- the two are not comparable, so keep it off unless you
# are explicitly making that comparison.
[ -n "${NO_FILTER_NORMAL:-}" ] && ARGS+=(--no-filter-normal-findings)

if [ -n "${NO_RADFACT:-}" ]; then
    echo "[INFO] NO_RADFACT=1 -- BLEU/METEOR only, Clinical Score will be 0.0"
    ARGS+=(--no-radfact)
else
    # ── Find the judge: explicit override, else the file judge_server.sh
    #    writes, else ask SLURM for the node running the 'judge' job. ────────
    if [ -z "${JUDGE_URL:-}" ] && [ -f "$PROJECT_DIR/.judge/url" ]; then
        JUDGE_URL="$(cat "$PROJECT_DIR/.judge/url")"
    fi
    if [ -z "${JUDGE_URL:-}" ]; then
        NODE="$(squeue -u "$USER" -n judge -h -t RUNNING -o %N 2>/dev/null | head -1)"
        [ -n "$NODE" ] && JUDGE_URL="http://${NODE}:8001/v1"
    fi
    if [ -z "${JUDGE_URL:-}" ]; then
        echo "[FAIL] No judge server found. Start one with:"
        echo "[FAIL]   sbatch --partition=gpu --qos=a100 --gres=gpu:a100:1 code/eval/judge_server.sh"
        echo "[FAIL] ...or run BLEU/METEOR only:  NO_RADFACT=1 $0 $*"
        exit 2
    fi

    # Reachability check. A judge that is up but unreachable would otherwise
    # yield logical_f1=None for every case -- a complete-looking results file
    # that is entirely garbage.
    HEALTH="${JUDGE_URL%/v1}/health"
    if ! curl -s --max-time 10 "$HEALTH" >/dev/null 2>&1; then
        echo "[FAIL] Judge not answering at $HEALTH"
        if [ -f "$PROJECT_DIR/.judge/url" ] && [ ! -f "$PROJECT_DIR/.judge/ready" ]; then
            echo "[FAIL] The job is queued/loading -- vLLM takes 15-25 min. Watch:"
            echo "[FAIL]   tail -f logs/judge_server_*.log"
        fi
        exit 2
    fi
    echo "[INFO] Judge: $JUDGE_URL  (model $SERVED_NAME)"
    ARGS+=(--judge-backend vllm --vllm-url "$JUDGE_URL" --model "$SERVED_NAME")
fi

echo "[INFO] Reports : $SYNTH_DIR  ($(ls "$SYNTH_DIR"/*.txt 2>/dev/null | wc -l) cases)"
echo "[INFO] Refs    : $REPORTS_DIR"
echo "[INFO] Out     : $OUT_DIR"
echo ""
exec "$PYTHON" "$PROJECT_DIR/code/eval/official_ranking.py" "${ARGS[@]}"
