#!/usr/bin/env bash
# =============================================================================
# aksssr_pipeline.sh — End-to-end: CBCT + mask + schema -> predictions -> reports
#
# Dataset layout:
#   dataset/training/{images,masks,reports,facts}
#   dataset/validate/{images,masks,reports,facts}
#
# For smoke tests, use LIMIT=5 against whichever split you're working with
# rather than a separate small test split. LIMIT samples RANDOMLY from the
# split's cases, but the sample is SEEDED (SEED=42 by default), so it stays
# the same across runs unless you change SEED yourself.
#
# Pipeline:
#   Every image-gen step writes its images AND a caption sidecar JSON, keyed
#   by schema.json's own images_needed names. Captions are produced at
#   GENERATION time, not by a separate captioning pass, which is why steps
#   1-3 all need this split's facts/{case}.json: the case findings get baked
#   into the caption text there and then.
#
#   NO_FACTS=1 cuts every one of those facts paths -- no fact reaches a caption,
#   and step 1 stops filtering its outlines by facts.teeth_present. Use it when
#   the facts files are suspect (e.g. laterality that may be flipped): a wrong
#   fact is worse than no fact, because the caption states it with the same
#   authority as a right one. See the NO_FACTS block in the config section.
#
#   1.   panoramic         -> {case}_panoramic.png
#                              + {case}_panoramic_caption.json
#                              An edentulous case still gets a panoramic:
#                              create_panoramic.py seeds the arch curve from
#                              the jawbone centerline when there aren't 4
#                              tooth centroids to fit, and only returns None
#                              when there's no jaw tissue either.
#                              REQUIRES --facts-file: facts.json is the source
#                              of truth for which teeth get outlined/tagged,
#                              so create_panoramic.py hard-errors without it.
#   2.   3D renders        -> {case}_3d_left/frontal/right.png
#                              + {case}_3d_captions.json
#                              --facts-file optional (adds the canal-proximity
#                              sentence to the left/right captions).
#   3.   tooth composites  -> {case}_tooth{fdi}_composite.png, ONE per tooth
#                              + {case}_tooth_captions.json
#                              create_tooth_detail.py. A 3x3 grid of axial /
#                              coronal / sagittal cuts through the tooth, all
#                              from this case's own volume -- it reads nothing
#                              from steps 1-2, so it no longer depends on the
#                              panoramic or the 3D renders existing. The target
#                              tooth is outlined in red in every panel, and on
#                              LOWER teeth the mandibular canal is FILLED with
#                              a translucent yellow wash (CANAL_ALPHA) in the
#                              same yellow step 2 gives it, so the root-to-canal
#                              distance is readable off the composite instead of
#                              only off the 3D views.
#                              --facts-file optional (adds per-tooth findings
#                              to that tooth's caption).
#   4.   sinus              -> {case}_sinus_right_detail.png,
#                              {case}_sinus_left_detail.png
#                              + {case}_sinus_captions.json
#                              -- ONE image per side, create_sinus_detail.py.
#                              No facts input: no schema fact maps to these.
#                              The nasal-cavity composite is GONE, not merely
#                              unused: schema v6.1 dropped the nasal_cavity
#                              fact, so nothing asked a question against it and
#                              create_sinus_detail.py stopped rendering it.
#                              Each sinus now carries a green CROSS-MARKER on
#                              the region rather than an outline around it.
#   5.   Build batched qa_pairs.jsonl from schema.json. No fact in the
#        current schema needs more than ONE image per call (every _left/
#        _right/_frontal/detail/composite image is used by exactly one
#        fact at a time) -- MAX_IMAGES_PER_PROMPT is a safety margin, not
#        a real requirement, unlike the old 9-image-per-call design.
#   6.   GPU / container / model preflight checks
#   7.   Start vLLM server (Qwen3.5-9B)
#   8.   Run VQA inference ONLY: qa_pairs.jsonl -> {case_id}_pred.json.
#        No report generation happens here anymore -- see below.
#
#   REPORT GENERATION IS NO LONGER AN LLM STEP.
#   ─────────────────────────────────────────────
#   This project moved from an LLM-generated report (reusing the vLLM
#   connection, postprocessed JSON in / prose out) to a deterministic
#   TEMPLATE renderer, per this session's decision: RadFact is only as
#   good as the underlying inference, and a template can't introduce a
#   NEW clinical error the way free-text generation can, on top of
#   whatever the VQA predictions already got wrong.
#
#   Concretely, that means steps 9-10 below need NEITHER the model NOR the
#   GPU at all -- postprocess_pred.py and synthesize_report.py are plain
#   Python (schema-driven classification + string templates, no VLM
#   calls), so they run as host python3, not inside the container, and
#   don't need vLLM alive. The vLLM server is torn down right after
#   inference (step 8) instead of being kept alive through report
#   generation the old design needed it for.
#
#   9.   Postprocess predictions -> {case_id}_summary.json
#        (postprocess_pred.py -- classification: absent-teeth pattern,
#        quality grouping, cross-source reconciliation)
#   10.  Synthesize reports: summaries -> {case_id}.txt
#        (synthesize_report.py -- the actual sentence templates)
#
# NOTE: image generation (steps 1-4, CPU-only work) still runs on a
# GPU-allocated node for the whole duration when you run this script alone --
# unrelated to the report-generation change above, that's about the vLLM
# server's lifetime, not the node's.
#
# When the gpu/a100 queue is deep, that idle-A100 time is paid twice: once
# pending for a GPU steps 1-4 never needed, then again generating images while
# holding one. code/pipeline/preprocess/gen_images_cpu.sh does steps 1-4 on --partition=cpu, in
# parallel across cases, into THIS run's images directory:
#
#     sbatch code/pipeline/preprocess/gen_images_cpu.sh validate      # short cpu queue, ~45 min
#     sbatch code/pipeline/aksssr_pipeline.sh validate     # steps 1-4 all [SKIP]
#
# It is not a separate arm -- same four generators, same facts arguments, same
# filenames and same completion signals, so the per-step existence checks below
# skip whatever it finished and generate the rest here as usual. The two may be
# queued together. Keep NO_FACTS the same across both: the checks below test
# whether a file exists, not how it was made.
#
# Usage:
#   sbatch code/pipeline/aksssr_pipeline.sh training
#   sbatch code/pipeline/aksssr_pipeline.sh validate
#   LIMIT=5 sbatch code/pipeline/aksssr_pipeline.sh training       # smoke test: same 5 cases every run
#                                                     # (SEED=42 by default)
#   LIMIT=5 SEED=7 sbatch code/pipeline/aksssr_pipeline.sh training  # a DIFFERENT fixed 5 cases
#   LIMIT=5 SEED= sbatch code/pipeline/aksssr_pipeline.sh training   # unseeded: reshuffles every run
#   RESUME=1 sbatch code/pipeline/aksssr_pipeline.sh training       # skip cases with existing _pred.json
#                                                     # (safe with the default seed -- a
#                                                     # resumed run re-samples the same cases)
#   
#   DRY_RUN=1 sbatch code/pipeline/aksssr_pipeline.sh training      # inference dry-run (no vLLM calls,
#                                                     # postprocessing/synthesis skipped too)
#   VERSION_TAG=v6.9 sbatch code/pipeline/aksssr_pipeline.sh validate # pin the output version tag
#                                                     # by hand instead of reading it from
#                                                     # schema.json. The per-stage outputs
#                                                     # nest under it:
#                                                     #   predictions/predictions_<tag>/
#                                                     #   summaries/summaries_<tag>/
#                                                     #   synthesized_reports/synthesized_reports_<tag>/
#                                                     # so a v6.9 run never lands on top of a
#                                                     # v6.4 one -- the two disagree about which
#                                                     # facts exist, so mixing them per-file
#                                                     # would compare different arms silently.
#   NO_FACTS=1 sbatch code/pipeline/aksssr_pipeline.sh validate     # facts/<case>.json never opened:
#                                                     # no fact in any caption, and the
#                                                     # panoramic outlines every SEGMENTED
#                                                     # tooth instead of the facts-listed ones
#   MAX_CONCURRENCY=16 sbatch code/pipeline/aksssr_pipeline.sh validate # calls in flight per case
#                                                     # (default "auto", from the server's own
#                                                     # KV pool; 1 = strictly sequential)
#
# THE MODEL LOADS WHILE THE IMAGES RENDER
#
#   Step 6a starts vLLM in the background BEFORE steps 1-4, and step 7b is the
#   only place that waits for it. The two halves need different hardware --
#   image generation is nibabel/VTK/PIL on the host CPU, the weight load is GPU
#   and disk -- and neither reads the other's output, so run serially they
#   simply add. On a 40-case run that was ~20 min of load charged after ~2
#   min/case of rendering, with the card idle throughout the first hour.
#
#   Borrowed from the competition container, where the same overlap is what
#   makes a single case fit inside Grand Challenge's 15 minutes. Two habits
#   came with it: poll /health every second rather than `sleep 90` and then
#   every 10 s (that pre-wait bought nothing and could hide a ready server for
#   100 s), and size concurrency from the pool the server reports rather than
#   from a constant.
#
#   The container and model existence checks stay AHEAD of the launch so a bad
#   QWEN_MODEL_NAME still fails in seconds, and `trap cleanup EXIT` is armed
#   with the launch so a generator crash cannot leave a server holding the GPU.
#   DRY_RUN starts no server at all.
#
# INFERENCE IS BATCHED PER CASE, AND WHY THAT IS SAFE
#
#   run_vqa_inference.py issues all ~38 of a case's calls concurrently, up to
#   MAX_CONCURRENCY at once, instead of one at a time. It used to send one
#   request and wait, so vLLM's continuous batching had nothing to batch and the
#   GPU idled between calls: 1092 calls over the 40 validate cases took 2h34m,
#   4.0 min/case (logs/aksssr_v6_553351.log). The 781 tooth calls were 71.5% of
#   the requests and 80.5% of the time, but the ~8 global calls per case were
#   being paid for serially too, which is why the batch covers both.
#
#   What makes it correct is TRUST_MODEL_ABSENCE being off in
#   run_vqa_inference.py. The only thing a per-tooth call ever needed from the
#   global calls is model_absent_teeth() -- the panoramic read's list of teeth
#   it thinks are missing -- and with the flag off that list cannot cancel a
#   tooth call: the segmentation that produced the composite outranks a 2D
#   projection read, so every imaged tooth is asked about regardless. Which
#   teeth to call is therefore known from qa_pairs.jsonl before any request is
#   sent. Turn TRUST_MODEL_ABSENCE on and that dependency returns, and the code
#   reinstates the barrier -- two batched phases, global then teeth -- rather
#   than assuming it away.
#
#   Results are assembled in call order regardless of completion order, so a
#   prediction is byte-identical to the sequential one; verified against a stub
#   server, 38 calls both ways. Per-call diagnostics (the raw-response preview,
#   parse_json's repair warnings) are buffered per thread and flushed as a
#   block, so a call's log lines stay together and stay attributable -- those
#   lines are what the parse-failure and repair rates get counted from.
#
# SIZING IT FOR THE A10G (24 GiB) THE SUBMISSION ACTUALLY RUNS ON
#
#   Memory is NOT the binding constraint, and the arithmetic is worth writing
#   down because the obvious way to check it reads 5x too pessimistic.
#
#     Qwen3.5-9B-AWQ weights                          11.53 GiB (measured)
#     24 GiB x --gpu-memory-utilization 0.90           21.6  GiB
#     less weights, activations, CUDA graphs          ~ 8    GiB for KV
#     KV density, measured on the bf16 A100 run        35.6 KiB/token
#       (logs/aksssr_v6_553351.log: 13.34 GiB -> 392,741 tokens)
#     => KV pool                                      ~236,000 tokens
#     one pipeline call is ~6.2k tokens (1,764 vision + ~3.5k text + answer)
#     => ~38 concurrent calls fit
#
#   Qwen3.5 is hybrid -- only 8 of its 32 layers are full attention and carry a
#   per-token KV cache; the other 24 are Gated DeltaNet with a per-SEQUENCE
#   recurrent state. That is why the density is 35.6 KiB/token rather than the
#   ~128 you would get from a dense 32-layer model, and why concurrency costs a
#   little more than tokens alone suggest.
#
#   THE TRAP: vLLM prints at startup
#       "Maximum concurrency for 32,768 tokens per request: N.NNx"
#   and that N is computed at --max-model-len, not at the size of the requests
#   this pipeline actually sends. Ours are ~5x smaller, so the real headroom is
#   ~5x that number. Sizing MAX_CONCURRENCY off that line directly would leave
#   most of the card unused. Read "GPU KV cache size: N tokens" instead and
#   divide by ~6,200 -- which is exactly what MAX_CONCURRENCY=auto now does,
#   grepping that line out of the job log once the server answers /health.
#
#   The old fixed 8 was safe on an A10G and wrong on both of the cards this
#   actually runs on: an A100-40GB reports ~392,741 tokens, i.e. ~63 slots, so
#   a fixed 8 used an eighth of the pool. "auto" is capped at
#   MAX_CONCURRENCY_CAP (40) because a case holds only ~24-38 calls and they
#   are batched within a case -- past that there is nothing to fill slots
#   with. What limits an A10G is compute, not memory: it has roughly 1/2.5 the
#   bandwidth of the A100 these numbers were measured on. Exceeding the pool
#   does not fail either way: vLLM queues and preempts, it just stops helping.
#
#   TO VERIFY BEFORE SUBMITTING, on this cluster, which has no A10G (see
#   CLAUDE.md "Testing that budget"):
#     memory fit  : --qos=a100 --gres=gpu:a100:1 with --gpu-memory-utilization
#                   0.55 on a 40 GB card carves out ~22 GB, the A10G envelope
#     the real card: --qos=a30 --gres=gpu:a30:1 (24 GB, ONE unit cluster-wide --
#                   confirm with it, do not iterate on it)
#     throughput  : --qos=3g.20gb --gres=gpu:3g.20gb:1 is smaller and slower
#                   than an A10G, so it is a conservative pass/fail gate.
#                   NOTE vLLM 0.19.0 cannot run on MIG (it int()s a MIG- uuid).
#
#   NOT DONE, and deliberately: build_user_blocks puts the IMAGE first and the
#   shared instruction text (TOOTH_SYSTEM, HOW_TO_READ_A_TOOTH, guidance,
#   output_schema -- ~3.5k tokens identical across all 29 tooth calls) AFTER
#   it, so none of it is a common prefix and --enable-prefix-caching can only
#   ever hit the system turn. Moving that text ahead of the image would make it
#   cacheable across every tooth in a case, which on a compute-starved A10G is
#   likely the largest remaining prefill saving. It also changes the prompt,
#   and therefore the answers, so it is a measured experiment and not a free
#   optimisation -- one change at a time.
#
#SBATCH --job-name=aksssr_v6
#SBATCH --partition=gpu
#SBATCH --qos=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/aksssr_v6_%j.log
#SBATCH --error=logs/aksssr_v6_%j.err

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# ── Configuration ──────────────────────────────────────────────────────────

PROJECT_DIR="${PROJECT_DIR:-$HOME/V2V2R_ToothFairy4}"
CODE_DIR="$PROJECT_DIR/code"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"
CONTAINER="${SIF_PATH:-$HOME/containers/extraction.sqsh}"
QWEN_MODEL_NAME="${QWEN_MODEL_NAME:-Qwen3.5-9B}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

# Max images sent in a single VLM call. Every fact in the current schema
# needs exactly ONE image (tooth_{fdi}_composite, sinus_*_detail,
# panoramic, or a single 3d_left/frontal/right) --
# unlike the old 9-image-per-tooth/9-per-sinus design, nothing here
# genuinely needs more than 1. Kept above 1 only as a safety margin in
# case build_vqa_pairs.py ever batches multiple images sharing one call.
MAX_IMAGES_PER_PROMPT="${MAX_IMAGES_PER_PROMPT:-3}"

# How many of a case's ~38 calls run_vqa_inference.py keeps in flight at once.
# Every call in a case is independent (TRUST_MODEL_ABSENCE is off, so the
# panoramic read cannot cancel a tooth call), and issuing them one at a time
# left vLLM's continuous batching with nothing to batch: 1092 calls over 40
# cases took 2h34m at 4.0 min/case with the GPU mostly idle
# (logs/aksssr_v6_553351.log). Passed as a flag rather than left to the
# environment because srun/pyxis is a container boundary and an unexported
# variable would silently fall back to the default inside it.
# 1 restores the old strictly-sequential behaviour. Raising it past what the
# server's KV cache holds does not fail -- vLLM queues -- it just stops
# helping.
#
# The default is now "auto": the number is derived after startup from the KV
# pool the server reports, instead of being a constant picked for one card
# (see step 7b). A literal number still wins, and MAX_CONCURRENCY=1 is still
# how the strictly-sequential path is recovered.
MAX_CONCURRENCY="${MAX_CONCURRENCY:-auto}"

# Tokens of KV cache one of this pipeline's requests occupies -- ~1,764 vision
# + ~3,500 text + reply, measured on the v7.1 validate runs and shared with
# the competition container's TOKENS_PER_REQUEST.
TOKENS_PER_REQUEST="${TOKENS_PER_REQUEST:-6200}"
# Ceiling for "auto". A case holds ~24-38 calls and they are batched within a
# case, so slots beyond that have nothing to put in them.
MAX_CONCURRENCY_CAP="${MAX_CONCURRENCY_CAP:-40}"
# Used only when the KV line cannot be found in the job log.
MAX_CONCURRENCY_FALLBACK="${MAX_CONCURRENCY_FALLBACK:-8}"

# How long step 7b will wait for /health after the images are done. The load
# itself is 15-25 min on an A100, but it starts at step 6a alongside image
# generation, so on a multi-case run most of it is already spent by then.
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-1800}"

# Required: "training" or "validate" -- no more optional/empty split.
SPLIT="${1:?Usage: sbatch code/pipeline/aksssr_pipeline.sh <training|validate>}"
if [ "$SPLIT" != "training" ] && [ "$SPLIT" != "validate" ]; then
    echo "[FAIL] SPLIT must be 'training' or 'validate', got: $SPLIT"
    exit 1
fi

BASE_DIR="$PROJECT_DIR/dataset/$SPLIT"

CBCT_DIR="$BASE_DIR/images"
MASK_DIR="$BASE_DIR/masks"
# Per-case findings, consumed by steps 1-3 to bake case facts into the image
# captions at generation time. create_panoramic.py additionally treats
# facts.structured.teeth_present as the source of truth for which teeth get
# outlined/tagged, so for step 1 this is required, not merely enriching.
FACTS_DIR="$BASE_DIR/facts"

# NO_FACTS=1 -- generate every image with facts/<case>.json never opened.
#
# Set it when the facts themselves are suspect. The laterality of a facts file
# cannot be checked from inside these generators: a left/right-flipped fact
# produces a perfectly well-formed caption naming the wrong side, and step 1
# would additionally outline and tag the WRONG teeth, since create_panoramic.py
# filters by facts.structured.teeth_present. Cutting facts out entirely is the
# only way to keep a suspect file from reaching either the pixels or the text.
#
# What it changes, per generator:
#   1 panoramic   --no-facts: every tooth the SEGMENTATION found is outlined
#                 and tagged (no teeth_present filter), caption falls back to
#                 its static description. THIS CHANGES THE PIXELS.
#   2 3D renders  --facts-file dropped: no FOV, ian_close_teeth, wisdom-tooth
#                 or tooth-inventory addenda; static viewpoint captions only.
#   3 tooth       --facts-file dropped: static layout caption only, no
#                 per-tooth fact fragments. Pixels are unaffected either way
#                 (make_tooth_composite reads volume + mask, never facts).
#   4 sinus       nothing to change: it has never taken a facts input.
#
# The per-case "facts file missing -> skip the case" guard below is also lifted,
# since no step needs one any more.
NO_FACTS="${NO_FACTS:-}"
SCHEMA="$PROJECT_DIR/schema/schema.json"

CODE_DIR_C="/project/code"
MODEL_DIR_C="/models"

# output
RUN_NAME="${RUN_NAME:-aksssr_v6}"
OUT_DIR="$PROJECT_DIR/outputs/${RUN_NAME}_${SPLIT}"
OUT_DIR_C="/project/outputs/${RUN_NAME}_${SPLIT}"

# Schema version tag, e.g. "v6.9" -- the per-stage outputs below are nested
# under a directory carrying it, matching the convention already used by
# official_ranking_openai/official_ranking_openai_v6.4/ and
# survey/survey_<timestamp>_v6.4.json.
#
# Read from schema.json rather than hardcoded, because the whole point is that
# it tracks the schema that actually produced these answers: a run against
# v6.9 must not land in the same directory as one against v6.4, since the two
# differ in which facts exist at all, and comparing them file-by-file would
# silently mix the arms. Override with VERSION_TAG=... to pin it by hand.
#
# `|| true` plus the fallback matters under `set -euo pipefail`: a missing or
# unparseable schema.json would otherwise abort the job here, several minutes
# before the preflight checks that are supposed to report that clearly.
if [ -z "${VERSION_TAG:-}" ]; then
    _schema_ver=$( { python3 -c "import json,sys; print(json.load(open('$SCHEMA'))['version'])" 2>/dev/null ; } || true )
    if [ -n "$_schema_ver" ]; then
        VERSION_TAG="v${_schema_ver}"
    else
        VERSION_TAG="vunknown"
        echo "[WARN] Could not read version from $SCHEMA -- tagging outputs '$VERSION_TAG'" >&2
    fi
fi

# ── REPLAY OVERRIDES: IMAGES_DIR / QA_JSONL ──────────────────────────────
# An arm comparison must differ in WEIGHTS ONLY (R11). The generators are code
# and code changes between runs -- 83c95bd turned the mandibular canal from an
# outline into a translucent fill three days after the v6 composites were
# written -- so re-rendering silently makes two arms differ in PIXELS as well
# as weights, and the difference is invisible in the outputs.
#
#   IMAGES_DIR=<dir>   read images (and their caption sidecars) from an
#                      existing directory instead of this run's own.
#   QA_JSONL=<file>    replay an existing payload. Implies IMAGES_DIR is
#                      irrelevant: steps 1-5 are SKIPPED entirely, because
#                      there is nothing to render and the payload is the
#                      thing being reused.
#
# Typical use -- score a merged SFT checkpoint on the baseline's exact input:
#   QA_JSONL=$PWD/outputs/aksssr_v7_validate/qa_pairs.jsonl \
#   QWEN_MODEL_NAME=Qwen3.5-9B-AWQ-arm2 RUN_NAME=vsft_arm2 \
#     sbatch code/pipeline/aksssr_pipeline.sh validate
#
# `${VAR:+1}` is read BEFORE the default is applied -- that is the only moment
# at which "did the caller supply this?" is still answerable.
REPLAY_PAYLOAD="${QA_JSONL:+1}"
REPLAY_IMAGES="${IMAGES_DIR:+1}"

QA_JSONL="${QA_JSONL:-$OUT_DIR/qa_pairs.jsonl}"
# Container path derived from the host path rather than rebuilt from OUT_DIR_C,
# so an externally supplied payload maps correctly too. Same substitution the
# other job scripts use; it requires the path to live under $PROJECT_DIR, which
# is checked below.
QA_JSONL_C="${QA_JSONL/$PROJECT_DIR//project}"
IMAGES_DIR="${IMAGES_DIR:-$OUT_DIR/images}"
PRED_DIR="$OUT_DIR/predictions/predictions_${VERSION_TAG}"
PRED_DIR_C="$OUT_DIR_C/predictions/predictions_${VERSION_TAG}"
SYNTH_DIR="$OUT_DIR/synthesized_reports/synthesized_reports_${VERSION_TAG}"
# The postprocessed {case_id}_summary.json handed to synthesize_report.py.
# Kept on disk so a questionable report can be traced to postprocessing vs.
# the template renderer without re-running anything.
SUMMARY_DIR="$OUT_DIR/summaries/summaries_${VERSION_TAG}"

LIMIT="${LIMIT:-}"
# Seed for LIMIT's random case sampling. Defaults to a FIXED seed so the
# sample never changes on its own -- the same LIMIT gives the same cases on
# every run, and it only moves when you pass a different SEED yourself.
# `${SEED-...}` (no colon) is deliberate: an explicitly empty SEED= is the
# opt-in escape hatch for an unseeded, different-every-run sample.
SEED="${SEED-42}"
RESUME="${RESUME:-}"
DRY_RUN="${DRY_RUN:-}"

mkdir -p logs "$PRED_DIR" "$SYNTH_DIR" "$SUMMARY_DIR"
# Only this run's OWN images directory is created. A supplied one must already
# exist: `mkdir -p` on a typo'd path would produce an empty directory, and an
# empty images dir is not an error anywhere downstream -- build_vqa_pairs.py
# emits captions:{} without complaining -- so the run would proceed on a
# payload built from nothing.
if [ -z "$REPLAY_IMAGES" ]; then
    mkdir -p "$IMAGES_DIR"
elif [ ! -d "$IMAGES_DIR" ]; then
    echo "[FAIL] IMAGES_DIR does not exist: $IMAGES_DIR"; exit 1
fi
for _p in "$QA_JSONL" "$IMAGES_DIR"; do
    case "$_p" in
        "$PROJECT_DIR"/*) ;;
        *) echo "[FAIL] must live under \$PROJECT_DIR to be visible inside the"
           echo "       container (mounted as /project): $_p"; exit 1 ;;
    esac
done
if [ -n "$REPLAY_PAYLOAD" ]; then
    [ -f "$QA_JSONL" ] || { echo "[FAIL] QA_JSONL not found: $QA_JSONL"; exit 1; }
    echo "[INFO] REPLAY: reusing payload $QA_JSONL"
    echo "[INFO]         steps 1-5 (image generation, payload build) SKIPPED"
fi

# Where slurm is writing this job's stdout. Asked from slurm rather than
# rebuilt from the #SBATCH --output pattern, so renaming the job (v3 -> v4,
# etc.) never desyncs the failure diagnostics below. Falls back to a glob on
# the job id, then to empty (dump_job_log then just says so).
JOB_LOG=""
if [ -n "${SLURM_JOB_ID:-}" ]; then
    # `|| true` on both: set -e + pipefail would otherwise abort the whole job
    # here when scontrol is missing or the glob matches nothing.
    JOB_LOG=$( { scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
                 | tr ' ' '\n' | grep '^StdOut=' | head -1 | cut -d= -f2- ; } || true )
    if [ ! -f "$JOB_LOG" ]; then
        JOB_LOG=$( { ls -1t logs/*"${SLURM_JOB_ID}".log 2>/dev/null | head -1 ; } || true )
    fi
fi

# Tail the job log without ever aborting the script (set -e + missing file).
dump_job_log() {
    local n="${1:-50}"
    if [ -n "$JOB_LOG" ] && [ -f "$JOB_LOG" ]; then
        echo "Last $n lines of job log ($JOB_LOG):"
        tail -"$n" "$JOB_LOG" || true
    else
        echo "[WARN] Could not locate this job's log file; check logs/ for job ${SLURM_JOB_ID:-local}"
    fi
}
echo "[INFO] Container paths: QA=$QA_JSONL_C PRED=$PRED_DIR_C" >&2

# Activate conda -- needed for image-gen / build_vqa_pairs.py /
# postprocess_pred.py / synthesize_report.py, ALL of which run as plain
# python3 on the host, not in the container (schema-driven or
# nibabel-dependent work never runs inside the container in this project).
#
# Do NOT reach this through `conda info --base`. A batch job's environment is
# whatever the SUBMITTING shell exported, and a non-interactive submission
# (ssh host 'sbatch ...', a cron hook) never sourced ~/.bashrc, so the conda
# shell function does not exist on the compute node even though it does on an
# interactive login node. Job 551011 ran that way: `conda: command not found`,
# every generator fell through to /usr/bin/python3, and the run died on a
# missing numpy after the allocation was already spent. Source a real conda.sh
# by path instead, and hard-fail if none is found -- CONDA_BASE overrides the
# search. Same fix as code/pipeline/preprocess/gen_images_cpu.sh.
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

# Preflight the host-side imports ONCE, before any case runs. The per-case
# steps tolerate a failing generator on purpose, so an ENVIRONMENT fault is
# not a per-case problem but looks exactly like one -- it reports itself for
# every case, after the allocation has been spent. This turns that into a
# two-second exit.
echo "[INFO] python3: $(command -v python3)"
if ! python3 -c "import nibabel, numpy, scipy, PIL" 2>&1; then
    echo "[FAIL] '$CONDA_ENV' cannot import the image-generation dependencies"
    echo "[FAIL] (nibabel / numpy / scipy / PIL). Nothing would be generated."
    exit 1
fi

# ── Main ───────────────────────────────────────────────────────────────────

{
    echo "[INFO] ========== Pipeline: CBCT -> Predictions -> Reports =========="
    echo "[INFO] Start: $(date)"
    echo "[INFO] Job ID: ${SLURM_JOB_ID:-local}"
    echo "[INFO] Split: $SPLIT"
    echo "[INFO] Mask dir (per-split): $MASK_DIR"
    echo "[INFO] Output dir: $OUT_DIR"
    echo "[INFO] Version tag: $VERSION_TAG (from $(basename "$SCHEMA"))"
    echo "[INFO]   predictions -> $PRED_DIR"
    echo "[INFO]   summaries   -> $SUMMARY_DIR"
    echo "[INFO]   reports     -> $SYNTH_DIR"
    [ -n "$LIMIT" ] && echo "[INFO] LIMIT: $LIMIT (random sample${SEED:+, SEED=$SEED})"
    [ -n "$RESUME" ] && echo "[INFO] RESUME: ON"
    if [ -n "$NO_FACTS" ]; then
        echo "[INFO] NO_FACTS: ON -- facts/<case>.json is never opened."
        echo "[INFO]   panoramic: every SEGMENTED tooth outlined/tagged (no"
        echo "[INFO]              teeth_present filter) -- different pixels"
        echo "[INFO]              from a facts run, not just a shorter caption."
        echo "[INFO]   3D + tooth composites: static captions, no fact fragments."
    fi
    [ -n "$DRY_RUN" ] && echo "[INFO] DRY-RUN: ON (no vLLM calls, postprocessing/synthesis skipped too)"
    echo ""

    # ── STEP 0: Discover Case IDs from Volumes ──────────────────────────────

    echo "[INFO] Discovering case IDs from volumes..."
    CASE_IDS=()
    for volume_file in "$CBCT_DIR"/*_0000.nii.gz; do
        if [ ! -f "$volume_file" ]; then
            continue
        fi
        filename=$(basename "$volume_file" _0000.nii.gz)
        CASE_IDS+=("$filename")
    done

    if [ ${#CASE_IDS[@]} -eq 0 ]; then
        echo "[FAIL] No volumes found in $CBCT_DIR"
        exit 1
    fi

    echo "[INFO] Found ${#CASE_IDS[@]} case(s): ${CASE_IDS[*]}"

    # LIMIT picks a RANDOM sample of cases, not the first N. Taking the first
    # N is biased by whatever the filename ordering correlates with (site,
    # acquisition batch, case numbering), so a smoke test on it doesn't say
    # much about the split as a whole.
    #
    # The sample is seeded by default, so it's reproducible across runs
    # (RESUME needs this -- unseeded, a resumed run would sample a DIFFERENT 5
    # cases and redo the work). It only moves when SEED is changed by hand;
    # SEED= (explicitly empty) opts back into a fresh shuffle each run.
    # Selection is re-sorted afterwards so the processing order stays stable
    # for a given sample.
    if [ -n "$LIMIT" ]; then
        if [ -n "$SEED" ]; then
            mapfile -t CASE_IDS < <(printf '%s\n' "${CASE_IDS[@]}" \
                | shuf -n "$LIMIT" --random-source=<(yes "$SEED") | sort)
            echo "[INFO] Limited to ${#CASE_IDS[@]} random case(s) (SEED=$SEED): ${CASE_IDS[*]}"
        else
            mapfile -t CASE_IDS < <(printf '%s\n' "${CASE_IDS[@]}" | shuf -n "$LIMIT" | sort)
            echo "[INFO] Limited to ${#CASE_IDS[@]} random case(s) (unseeded): ${CASE_IDS[*]}"
        fi
    fi
    echo ""

    if [ ! -f "$SCHEMA" ]; then
        echo "[FAIL] Schema not found: $SCHEMA"
        exit 1
    fi
    echo "[INFO] Schema: $SCHEMA"
    echo ""

    # ── STEP 6a: GPU check + start vLLM, BEFORE the images ──────────────────
    #
    # The server load and the image generation want different hardware and
    # neither waits on the other, so they are started together and the shorter
    # one disappears inside the longer. Steps 1-4 are nibabel/VTK/PIL on the
    # host CPU and touch no GPU; the weight load is GPU and disk and touches no
    # generator output. Run serially -- which is what this script did until now,
    # loading the model only at step 7 -- a 40-case run pays ~20 min of load
    # AFTER ~2 min/case of rendering, and the card sits idle for the first hour.
    #
    # Lifted from the competition container, where the same overlap is what
    # makes one case fit in 15 minutes. There it is the whole game (5+10 -> 10);
    # here it is ~20 minutes off every arm's inference job, which over the five
    # arms of docs/vision_sft_plan.md is worth having.
    #
    # Ordering constraints this respects:
    #   * the container / model existence checks stay AHEAD of the launch, so a
    #     typo in QWEN_MODEL_NAME still fails in seconds rather than after an
    #     hour of rendering;
    #   * `trap cleanup EXIT` is installed with the launch, so a generator that
    #     dies mid-loop takes the server down with it instead of leaving it
    #     holding the GPU for the rest of the allocation;
    #   * DRY_RUN starts nothing -- it never reaches step 8.
    # The readiness WAIT stays at step 7b, after the images: waiting here would
    # rebuild the serial order this is removing.

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

    if [ -z "$DRY_RUN" ]; then
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

        echo "[INFO] Starting vLLM server (in parallel with image generation)..."
        echo "  Model: $QWEN_MODEL_NAME"
        echo "  Port: $PORT"
        echo "  Max context: $MAX_MODEL_LEN"
        echo "  Max images/prompt: $MAX_IMAGES_PER_PROMPT"
        echo ""

        srun \
            --overlap \
            --container-image="$CONTAINER" \
            --container-mounts="$MODEL_DIR:/models" \
            python3 -m vllm.entrypoints.openai.api_server \
                --model "/models/$QWEN_MODEL_NAME" \
                --served-model-name qwen3.5-vl \
                --port "$PORT" \
                --dtype bfloat16 \
                --max-model-len "$MAX_MODEL_LEN" \
                --reasoning-parser qwen3 \
                --limit-mm-per-prompt "{\"image\": $MAX_IMAGES_PER_PROMPT}" \
                --gpu-memory-utilization 0.85 \
                --enable-prefix-caching &

        VLLM_PID=$!
        echo "[INFO] vLLM PID: $VLLM_PID -- loading while images render"
        echo ""
    fi

    # ── STEP 1-4: Image Generation (Per Case) ───────────────────────────────
    # Skipped wholesale under QA_JSONL=: the payload already names every image
    # it needs, so rendering more would at best duplicate work and at worst
    # produce different pixels than the payload was built from.

    if [ -z "$REPLAY_PAYLOAD" ]; then

    for case_id in "${CASE_IDS[@]}"; do
        echo "[INFO] Processing case: $case_id"

        volume_file="$CBCT_DIR/${case_id}_0000.nii.gz"
        mask_file="$MASK_DIR/${case_id}.nii.gz"
        facts_file="$FACTS_DIR/${case_id}.json"

        if [ ! -f "$volume_file" ]; then
            echo "  [SKIP] Volume not found: $volume_file"
            continue
        fi
        if [ ! -f "$mask_file" ]; then
            echo "  [SKIP] Mask not found: $mask_file"
            continue
        fi
        # Skip the case rather than let step 1 hard-error: create_panoramic.py
        # calls ap.error() without --facts-file, which under `set -e` would
        # take down the entire job on one case's missing file. Under NO_FACTS
        # no step opens the file at all, so a missing one is not a reason to
        # drop the case.
        if [ -z "$NO_FACTS" ] && [ ! -f "$facts_file" ]; then
            echo "  [SKIP] Facts not found: $facts_file"
            continue
        fi

        # How steps 1-3 are told about facts. Under NO_FACTS step 1 takes its
        # explicit --no-facts branch (unfiltered outlines + static caption) and
        # steps 2-3 are simply handed no --facts-file, which they already treat
        # as "static caption only".
        if [ -n "$NO_FACTS" ]; then
            panoramic_facts_arg=(--no-facts)
            case_facts_arg=()
        else
            panoramic_facts_arg=(--facts-file "$facts_file")
            case_facts_arg=(--facts-file "$facts_file")
        fi

        # Step 1: Panoramic. create_panoramic.py skips entirely (produces no
        # file) if fewer than 4 teeth are detected -- not a failure,
        # downstream steps just won't have a panoramic image for this case.
        #
        # Skip on the CAPTION SIDECAR, not on the .png. The sidecar is the
        # completion signal for every other step here, and it is the thing
        # build_vqa_pairs.py actually consumes -- discover_captions() reads the
        # four sidecars and, finding none, emits captions:{} without erroring.
        # Keying on the .png meant a directory holding the images but not their
        # sidecars (e.g. after the sidecars were cleaned out) skipped straight
        # past regeneration and silently produced an uncaptioned arm.
        panoramic_file="$IMAGES_DIR/${case_id}_panoramic_caption.json"
        if [ -f "$panoramic_file" ]; then
            echo "  [SKIP] Panoramic already exists: $panoramic_file"
        else
            echo "  [STEP 1] Creating panoramic${NO_FACTS:+ (--no-facts)}..."
            python3 "$CODE_DIR/pipeline/preprocess/create_panoramic.py" \
                --volume "$volume_file" \
                --mask "$mask_file" \
                "${panoramic_facts_arg[@]}" \
                --out-dir "$IMAGES_DIR" \
                --case-id "$case_id"
        fi

        # Step 2: 3D renders -- 3 separate images (left/frontal/right) + a
        # captions sidecar. Use the captions sidecar as the completion signal.
        render_captions_file="$IMAGES_DIR/${case_id}_3d_captions.json"
        if [ -f "$render_captions_file" ]; then
            echo "  [SKIP] 3D renders already exist: $render_captions_file"
        else
            echo "  [STEP 2] Creating 3D renders${NO_FACTS:+ (no facts)}..."
            python3 "$CODE_DIR/pipeline/preprocess/create_3d_renders.py" \
                --mask "$mask_file" \
                "${case_facts_arg[@]}" \
                --out-dir "$IMAGES_DIR" \
                --case-id "$case_id"
        fi

        # Step 3: Tooth composites -- ONE multi-panel image per tooth,
        # {case}_tooth{fdi}_composite.png, from create_tooth_detail.py: a 3x3
        # grid of axial / coronal / sagittal cuts, every tile cropped from the
        # same tooth-centred window of this case's own volume. It reads no
        # image from steps 1-2 any more, so its only inputs are the volume,
        # the mask and (optionally) the facts file.
        #
        # Completion signal: the captions sidecar, written after the last
        # tooth, so an interrupted run re-does the case instead of resuming
        # from a half-written image set.
        tooth_captions_file="$IMAGES_DIR/${case_id}_tooth_captions.json"
        if [ -f "$tooth_captions_file" ]; then
            echo "  [SKIP] Tooth composites already exist for $case_id"
        else
            echo "  [STEP 3] Creating tooth composites${NO_FACTS:+ (no facts)}..."
            python3 "$CODE_DIR/pipeline/preprocess/create_tooth_detail.py" \
                --volume "$volume_file" \
                --mask "$mask_file" \
                "${case_facts_arg[@]}" \
                --out-dir "$IMAGES_DIR" \
                --case-id "$case_id"
        fi

        # Step 4: Sinus detail -- ONE image per sinus side, plus a captions
        # sidecar. The nasal-cavity composite is gone: schema.json has no
        # nasal_cavity fact for it to answer, so create_sinus_detail.py no
        # longer renders one. Completion is checked on the actual PNGs (this
        # generator writes single images per region, not the multi-image
        # sidecar scheme the old check assumed). Takes no facts input at all,
        # so NO_FACTS changes nothing here.
        # Skip on the caption sidecar for the same reason as step 1 above.
        # Note this never engages for a case whose mask has 0 sinus voxels:
        # create_sinus_detail.py writes no sidecar there, so the step re-runs
        # (and re-skips internally) on every pass. That is the same cheap
        # redundancy step 3 already accepts for a case with no tooth labels,
        # and is preferable to keying on a .png that may outlive its caption.
        sinus_captions_file="$IMAGES_DIR/${case_id}_sinus_captions.json"
        if [ -f "$sinus_captions_file" ]; then
            echo "  [SKIP] Sinus detail already exists for $case_id"
        else
            echo "  [STEP 4] Creating sinus detail..."
            python3 "$CODE_DIR/pipeline/preprocess/create_sinus_detail.py" \
                --volume "$volume_file" \
                --mask "$mask_file" \
                --out-dir "$IMAGES_DIR" \
                --case-id "$case_id"
        fi
    done

    # ── STEP 5: Build QA Pairs ───────────────────────────────────────────────

    echo ""
    echo "[STEP 5] Building QA pairs from schema..."
    python3 "$CODE_DIR/pipeline/preprocess/build_vqa_pairs.py" \
        --schema "$SCHEMA" \
        --images-dir "$IMAGES_DIR" \
        --project-dir "$PROJECT_DIR" \
        --cases "${CASE_IDS[@]}" \
        --out "$QA_JSONL"

    if [ ! -f "$QA_JSONL" ]; then
        echo "[FAIL] qa_pairs.jsonl was not created: $QA_JSONL"
        exit 1
    fi
    echo "[INFO] QA pairs: $QA_JSONL ($(wc -l < "$QA_JSONL") case record(s))"

    fi   # end: steps 1-5 skipped under REPLAY_PAYLOAD

    echo "[INFO] payload: $QA_JSONL ($(wc -l < "$QA_JSONL") case record(s))"
    echo ""

    # ── DRY-RUN BRANCH: Inference only, skips vLLM server entirely ─────────

    # Under REPLAY the payload is somebody else's and holds THEIR cases -- all
    # 40 of validate, not the LIMIT/SEED sample this run selected. `--limit N`
    # would then take the first N RECORDS of that file, which is a different
    # set of cases than the same LIMIT picks without replay, so two arms could
    # silently score different cases. Naming the cases explicitly makes LIMIT
    # mean the same thing in both modes. Harmless when not replaying: the
    # payload was built from exactly these ids.
    # Written as an `if` rather than `[ -n "$X" ] && arr=(...)`: under
    # `set -e` that one-liner's exit status is the failed test, and whether the
    # shell exits then depends on a subtle rule about which side of the final
    # `&&` the failure fell on. Not worth relying on in a job that runs for
    # hours before anyone finds out.
    replay_case_args=()
    if [ -n "$REPLAY_PAYLOAD" ]; then
        replay_case_args=(--case-ids "${CASE_IDS[@]}")
    fi

    if [ -n "$DRY_RUN" ]; then
        echo "[DRY-RUN] Inference (no vLLM server)"
        srun \
            --overlap \
            --container-image="$CONTAINER" \
            --container-mounts="$MODEL_DIR:/models,$PROJECT_DIR:/project" \
            python3 "$CODE_DIR_C/pipeline/infer/run_vqa_inference.py" \
                --vqa-jsonl "$QA_JSONL_C" \
                --out-dir "$PRED_DIR_C" \
                --base-dir "/project" \
                "${replay_case_args[@]}" \
                ${LIMIT:+--limit $LIMIT} \
                ${RESUME:+--resume} \
                --dry-run
        echo ""
        echo "[INFO] ========== Pipeline Complete (DRY-RUN) =========="
        echo "[INFO] Postprocessing/synthesis skipped (no real predictions in dry-run)"
        echo "[INFO] End: $(date)"
        exit 0
    fi

    # ── STEP 7b: Wait for vLLM (it has been loading since step 6a) ──────────
    #
    # Poll /health every second and stop as soon as it answers. The old shape
    # here -- `sleep 90`, then a 150-iteration loop sleeping 10 s -- charged
    # every run up to 100 s of not-noticing a server that was already up. That
    # was tolerable when nothing else had happened yet; now the images have
    # already run, so on a 40-case job the server is usually ready BEFORE this
    # point and the whole wait collapses to one successful curl.
    #
    # The deadline is wall-clock from here, not an iteration count, so the
    # polling interval and the timeout stop being the same number.

    START_WAIT=$(date +%s)
    WAIT_DEADLINE=$((START_WAIT + VLLM_STARTUP_TIMEOUT))
    SERVER_READY=0
    next_report=$((START_WAIT + 60))

    echo "[INFO] Waiting for vLLM readiness (timeout ${VLLM_STARTUP_TIMEOUT}s)..."
    while [ "$(date +%s)" -lt "$WAIT_DEADLINE" ]; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[FAIL] vLLM process died during startup"
            echo ""
            dump_job_log 50
            exit 1
        fi

        if curl -s --max-time 5 "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "[PASS] vLLM ready after $(( $(date +%s) - START_WAIT ))s of waiting"
            SERVER_READY=1
            break
        fi

        now=$(date +%s)
        if [ "$now" -ge "$next_report" ]; then
            echo "  Waiting... (~$(( (now - START_WAIT + 30) / 60 )) min at this step)"
            next_report=$((now + 60))
        fi

        sleep 1
    done

    if [ $SERVER_READY -eq 0 ]; then
        echo "[FAIL] vLLM never became ready after ${VLLM_STARTUP_TIMEOUT}s"
        echo ""
        dump_job_log 100
        exit 1
    fi
    echo ""

    # ── Size concurrency off the KV pool the server actually reports ────────
    #
    # MAX_CONCURRENCY=auto reads "GPU KV cache size: N tokens" out of the job
    # log and divides by the ~6,200 tokens one of this pipeline's requests
    # occupies. A fixed 8 was chosen once, for one card, and is now wrong in
    # both directions: an A100-40GB reports ~390k tokens (≈63 slots) and left
    # most of the card idle, while a smaller card can hold fewer than 8.
    #
    # Read the KV line, NOT vLLM's "Maximum concurrency for 32,768 tokens per
    # request: N" -- that is computed at --max-model-len, and our requests are
    # ~5x smaller, so it understates the real ceiling by about 5x. The
    # competition container parses the same line for the same reason.
    #
    # The cap exists because concurrency above the number of calls in a case
    # buys nothing: run_vqa_inference.py batches WITHIN a case, and a case has
    # ~24-38 calls. Anything past that just queues.
    if [ "$MAX_CONCURRENCY" = "auto" ]; then
        kv_tokens=""
        if [ -n "$JOB_LOG" ] && [ -f "$JOB_LOG" ]; then
            # `|| true` for the same reason as line 387: under `set -euo
            # pipefail` a grep that matches nothing exits 1 and takes the whole
            # job with it -- and "the line is not there" is precisely the case
            # this is supposed to fall back from, not die on.
            kv_tokens=$(grep -o "GPU KV cache size: [0-9,]* tokens" "$JOB_LOG" \
                        | tail -1 | grep -o "[0-9,]*" | tr -d ',' || true)
        fi
        if [ -n "$kv_tokens" ] && [ "$kv_tokens" -gt 0 ] 2>/dev/null; then
            MAX_CONCURRENCY=$((kv_tokens / TOKENS_PER_REQUEST))
            [ "$MAX_CONCURRENCY" -lt 1 ] && MAX_CONCURRENCY=1
            if [ "$MAX_CONCURRENCY" -gt "$MAX_CONCURRENCY_CAP" ]; then
                echo "[INFO] KV pool ${kv_tokens} tok allows $((kv_tokens / TOKENS_PER_REQUEST)) in flight; capping at $MAX_CONCURRENCY_CAP (calls per case)"
                MAX_CONCURRENCY=$MAX_CONCURRENCY_CAP
            else
                echo "[INFO] Concurrency $MAX_CONCURRENCY = KV pool ${kv_tokens} tok / ${TOKENS_PER_REQUEST} tok per request"
            fi
        else
            MAX_CONCURRENCY=$MAX_CONCURRENCY_FALLBACK
            echo "[WARN] Could not read 'GPU KV cache size' from the job log; using MAX_CONCURRENCY=$MAX_CONCURRENCY"
        fi
        echo ""
    fi

    # ── STEP 8: Run VQA Inference ONLY (predictions, no report generation) ──
    #
    # run_vqa_inference.py no longer takes --reports-out-dir/--summaries-out-dir
    # -- report generation moved OUT of this step entirely (see steps 9-10
    # below), since it no longer needs the model at all.

    echo "[INFO] Running VQA inference..."
    srun \
        --overlap \
        --container-image="$CONTAINER" \
        --container-mounts="$MODEL_DIR:/models,$PROJECT_DIR:/project" \
        python3 "$CODE_DIR_C/pipeline/infer/run_vqa_inference.py" \
            --vqa-jsonl "$QA_JSONL_C" \
            --vllm-url "http://localhost:${PORT}/v1" \
            --model qwen3.5-vl \
            --out-dir "$PRED_DIR_C" \
            --base-dir "/project" \
            --max-concurrency "$MAX_CONCURRENCY" \
            ${LIMIT:+--limit $LIMIT} \
            ${RESUME:+--resume} \
        || { echo "[FAIL] Inference failed"; dump_job_log 50; exit 1; }

    num_preds=$(find "$PRED_DIR" -maxdepth 1 -name "*_pred.json" | wc -l)
    echo ""
    echo "[INFO] Inference complete. Prediction files: $num_preds"
    echo ""

    # vLLM is no longer needed for anything below this point -- stop it now
    # rather than holding the GPU process alive through postprocessing and
    # report synthesis, which are pure CPU/Python and don't touch the model.
    echo "[INFO] Stopping vLLM (no longer needed -- report generation is model-free now)..."
    cleanup
    echo ""

    # ── STEP 9: Postprocess predictions -> summaries (CPU, no model) ────────
    #
    # Plain host python3, NOT inside the container/srun -- postprocess_pred.py
    # is schema-driven classification logic with no GPU/model dependency,
    # same convention as nibabel-dependent image-gen work in this project.

    echo "[STEP 9] Postprocessing predictions into summaries..."
    python3 "$CODE_DIR/pipeline/postprocess/postprocess_pred.py" \
        --pred-dir "$PRED_DIR" \
        --out-dir "$SUMMARY_DIR" \
        --schema "$SCHEMA" \
        || { echo "[FAIL] Postprocessing failed"; exit 1; }

    num_summaries=$(find "$SUMMARY_DIR" -maxdepth 1 -name "*_summary.json" | wc -l)
    echo "[INFO] Postprocessed summaries: $num_summaries"
    echo ""

    # ── STEP 10: Synthesize reports from summaries (CPU, no model) ──────────

    echo "[STEP 10] Synthesizing reports from summaries..."
    python3 "$CODE_DIR/pipeline/postprocess/synthesize_report.py" \
        --summary-dir "$SUMMARY_DIR" \
        --out-dir "$SYNTH_DIR" \
        || { echo "[FAIL] Report synthesis failed"; exit 1; }

    num_reports=$(find "$SYNTH_DIR" -maxdepth 1 -name "*.txt" | wc -l)
    echo "[INFO] Synthesized reports: $num_reports"
    echo ""

    # A per-case failure in postprocessing/synthesis doesn't abort the whole
    # run (each script processes every file it finds, one case's malformed
    # prediction shouldn't cost every other case its report) -- so the only
    # outward sign of a partial failure is these counts disagreeing. Say so
    # loudly instead of leaving it to be spotted by eye.
    if [ "$num_reports" -lt "$num_preds" ]; then
        echo "[WARN] $((num_preds - num_reports)) case(s) produced a prediction but NO report."
        echo "[WARN] This is now a CHEAP, CPU-only, no-vLLM-needed retry -- no GPU"
        echo "[WARN] allocation or model load required, unlike the old LLM-based report step:"
        echo "[WARN]   python code/pipeline/postprocess/postprocess_pred.py --pred-dir $PRED_DIR \\"
        echo "[WARN]       --out-dir $SUMMARY_DIR --schema $SCHEMA"
        echo "[WARN]   python code/pipeline/postprocess/synthesize_report.py --summary-dir $SUMMARY_DIR \\"
        echo "[WARN]       --out-dir $SYNTH_DIR"
    fi
    echo ""

    # ── Summary ──────────────────────────────────────────────────────────────

    echo "[INFO] ========== Pipeline Complete =========="
    echo "[INFO] End: $(date)"
    echo ""

    num_panoramic=$(find "$IMAGES_DIR" -maxdepth 1 -name "*_panoramic.png" | wc -l)
    num_3d_cases=$(find "$IMAGES_DIR" -maxdepth 1 -name "*_3d_captions.json" | wc -l)
    num_3d_images=$(find "$IMAGES_DIR" -maxdepth 1 \( -name "*_3d_left.png" -o -name "*_3d_frontal.png" -o -name "*_3d_right.png" \) | wc -l)
    num_tooth_images=$(find "$IMAGES_DIR" -maxdepth 1 -name "*_tooth*_composite.png" | wc -l)
    num_sinus_images=$(find "$IMAGES_DIR" -maxdepth 1 \( -name "*_sinus_right_detail.png" -o -name "*_sinus_left_detail.png" \) | wc -l)

    echo "[INFO] Generated images:"
    echo "  - Panoramic: $num_panoramic case(s)"
    echo "  - 3D renders: $num_3d_cases case(s), $num_3d_images image(s) (left/frontal/right, 3/case)"
    echo "  - Tooth composites: $num_tooth_images image(s) (1/tooth, up to 32/case)"
    echo "  - Sinus detail: $num_sinus_images image(s) (1 per sinus side, up to 2/case)"
    echo "[INFO] QA pairs: $QA_JSONL"
    echo "[INFO] Predictions: $PRED_DIR/ ($num_preds file(s))"
    echo "[INFO] Postprocessed summaries: $SUMMARY_DIR/ ($num_summaries file(s))"
    echo "[INFO] Synthesized reports: $SYNTH_DIR/ ($num_reports file(s))"
    echo ""

} 2>&1

# Usage:
#   sbatch code/pipeline/aksssr_pipeline.sh training
#   sbatch code/pipeline/aksssr_pipeline.sh validate
#   LIMIT=5 sbatch code/pipeline/aksssr_pipeline.sh training        # smoke test: 5 random cases
#   LIMIT=5 SEED=42 sbatch code/pipeline/aksssr_pipeline.sh training # same 5 cases every run
#   RESUME=1 sbatch code/pipeline/aksssr_pipeline.sh training        # skip cases with existing _pred.json
#   DRY_RUN=1 sbatch code/pipeline/aksssr_pipeline.sh validate        # inference dry-run only