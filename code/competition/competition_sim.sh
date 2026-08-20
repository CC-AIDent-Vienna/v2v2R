#!/bin/bash
# competition_sim.sh — ONE case under the Grand Challenge shape, on this cluster
#
# WHAT IT SIMULATES
#   Grand Challenge starts a FRESH container PER CASE on an A10G (24 GiB VRAM,
#   32 GiB RAM, 4-8 vCPU) with no network, and gives it 15 minutes. So the model
#   load is charged to the one case that will ever run, and the question is not
#   "how fast is inference" but "does the whole thing fit in 900 seconds".
#
#   code/competition/competition_runner.py is the thing being tested. It differs from
#   aksssr_pipeline.sh in three ways that only matter under that budget:
#   vLLM starts FIRST and loads while the images generate; /health is polled
#   every 0.5 s instead of after a fixed 90 s sleep; and concurrency is read off
#   the KV pool the server reports rather than fixed for an A100.
#
# WHICH CARD, AND WHY NOT THE OBVIOUS ONES
#   --qos=a30 --gres=gpu:a30:1 on s0-n12. The A30 is 24 GiB, the same envelope
#   as an A10G, and it is the only real 24 GiB card on the cluster -- ONE unit
#   cluster-wide, so use it to confirm a result, not to iterate on one.
#
#   NOT the 3g.20gb MIG slice, even though CLAUDE.md suggests it as a
#   conservative proxy: vLLM 0.19 cannot run on MIG at all. It calls int() on
#   CUDA_VISIBLE_DEVICES, which on a MIG slice is a "MIG-<uuid>" string, and
#   dies during model inspection before touching a weight.
#
#   NOT an A100: it flatters throughput by ~2.5x on bandwidth alone, so a pass
#   there says nothing about the A10G. The A30 is Ampere like the A10G but
#   sm_80 rather than sm_86 -- close on memory, not identical on kernels.
#
# THE CONTAINER IS NOT THE RESEARCH ONE
#   ~/containers/competition.sqsh = vllm019_cu128.sqsh + nibabel, scipy,
#   scikit-image, pillow, matplotlib. The research split (generators on the host
#   in cbct_base, vLLM in the container) cannot work here: the whole point is
#   that ONE process starts the server and renders the images concurrently, so
#   both halves have to be in the same image. A submission container needs the
#   same union.
#
# USAGE
#   sbatch code/competition/competition_sim.sh [CASE_ID] [SPLIT]
#   CASE=A008 SPLIT=validate sbatch code/competition/competition_sim.sh
#
# ONE CASE, AND THAT IS THE POINT OF THIS SCRIPT. competition_runner.py also
# takes --dataset-dir and runs a whole split on one model load, but a batch run
# cannot answer the question here: the budget is per CONTAINER START, and a
# server shared across forty cases has already stopped simulating the thing.
# For a split, call the runner directly, or use gen_images_cpu.sh (renders on
# the CPU partition, in parallel) followed by pool_infer.sh.
#
# THE INPUTS ARE UPSTREAM'S, AND THE AUDIT IS THE GATE
#   The mask AND the facts both come from the segmentation component (the pools
#   dataset/source/predictions and dataset/source/facts_all_622_cases_new,
#   split into dataset/$SPLIT/{masks,facts}). Neither is derived from a
#   reference report, so both exist on Grand Challenge and this is the real
#   input shape rather than a research convenience.
#
#   What does NOT come from upstream is the reconciliation between them, and
#   competition_runner.py runs audit_facts.py --derive-bridge-arches --execute
#   over a COPY before any generator opens the file. Two rules are inert
#   without it: absent teeth gates on `fov.maxilla == excluded` -- which no
#   upstream file sets, it says "partial" or nothing (12 of the 40 validate
#   cases change) -- and the fixed bridge rule reads `bridge_arches`, which
#   upstream has no key for at all. ~2 s of CPU while vLLM loads: it is free,
#   and skipping it silently turns off two of the ten source rules.
#
#   THE DEFAULT IS THE RAW POOL, NOT dataset/$SPLIT/facts, and the difference
#   is the whole point. The split copies have been audited IN PLACE by the
#   research path, so pointing at them makes the audit a no-op and the
#   simulation stops testing the gate it exists to prove. The raw pool is what
#   a container is actually handed. Both end at the same file -- that is the
#   claim being checked, once per run, in the log.
#
# ENV
#   FACTS_FILE=...              upstream facts for the case, PATH INSIDE THE
#                               CONTAINER. Default: the RAW pool copy,
#                               /project/dataset/source/facts_all_622_cases_new.
#   EXTRACT_FACTS=1             no upstream facts: derive them in-container
#                               from the mask (extract_facts.py, ~65 s). The
#                               fallback arm, not the shipping one.
#   NO_FACTS=1                  no facts at all -- captions carry none and the
#                               source rules stay off. The ablation arm.
#   MODEL_NAME=Qwen3.5-9B-AWQ   the un-adapted baseline; the default is the
#                               trained arm, named for its Hugging Face repo
#   MAX_MODEL_LEN=32768         do NOT shorten this. The context has to hold the
#                               prompt AND the reply, and the global calls ask
#                               for max_tokens=8192 of output on top of a ~12.5k
#                               character prompt. At 8192 every global call dies
#                               with a 400 and the prediction comes back with
#                               global={} -- the tooth calls still succeed, so
#                               the run LOOKS fine and the report is quietly
#                               synthesized from defaults.
#   GPU_MEM_UTIL=0.90           of a 24 GiB card
#   MAX_CONCURRENCY=0           0 = auto from the reported KV pool
#SBATCH --job-name=comp_sim
#SBATCH --partition=gpu
#SBATCH --qos=a30
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/comp_sim_%j.log
#SBATCH --error=logs/comp_sim_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/project_ToothFairy4}"
CONTAINER="${CONTAINER:-$HOME/containers/competition.sqsh}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"
# THE TRAINED MODEL IS THE DEFAULT AS OF 2026-08-16, AND IT IS ARM 6 AS OF
# 2026-08-17. models/Qwen3.5-9B-AWQ-dental-cbct-sft is the vision+language LoRA
# of docs/vision_sft_plan_light.md §8.3, merged into the AWQ base (214/214 tensors
# bit-identical at merge). Official 0.4557 on validate-40 against arm 5's
# 0.4115, all of it clinical: RadFact F1 0.4470 -> 0.5014 with precision and
# recall rising together, captioning flat.
#
# WHAT THIS DIRECTORY HELD BEFORE, AND WHY THAT ARGUMENT IS GONE. Until today it
# was arm 5 (§8.2), which won the groups it trained on and REGRESSED on the two
# it did not: [3d] alveolar atrophy and [sinus] scope. The note here used to say
# that trade was survivable only because the source rules covered exactly those
# weaknesses -- atrophy taken from the mask's edentulism, sinus scope reaching
# no sentence. Arm 6 repairs both at the source ([3d] +0.13, [sinus] +0.33), so
# the default no longer depends on that overlap. The rules still do the same
# work; they are simply no longer compensating for the model.
#
# Arm 5 is still on disk as models/Qwen3.5-9B-AWQ-arm5, and on the hub at
# revision tag `arm5`. Set MODEL_NAME=Qwen3.5-9B-AWQ-arm5 to reproduce the
# 0.4115 runs, or MODEL_NAME=Qwen3.5-9B-AWQ for the un-adapted baseline.
#
# PROMOTION IS BY RENAME, which is why arm 5 vacated this name rather than
# being overwritten: the directory name is the model's identity here, so the
# weights behind it must never change under a reader's feet.
#
# `best_model` IS THE NAME THE SUBMISSION USES, AND THAT IS THE POINT: it is
# indirection, so promoting an arm is a pointer change and NOT a code change.
# The submission is fully offline -- the weights are baked into the image and
# nothing is fetched at runtime -- so the container never learns which arm it is
# holding, and it does not need to.
#
# This reverses a note that stood here from 2026-08-16, which argued for naming
# the directory after the Hugging Face repo because "nothing that downloads the
# model produces `best_model`, so every fetch needs a rename or a MODEL_NAME
# override". That objection is real, and it is about FETCHING, on this cluster.
# It does not reach inside the container, where there is no fetch at all. So the
# two names split by role, deliberately:
#
#   models/best_model                       what the pipeline OPENS -- stable
#   models/Qwen3.5-9B-AWQ-dental-cbct-sft   what a download PRODUCES -- versioned
#
# best_model points at the second. Keep it a symlink and repoint it to promote;
# do not copy, and do not let the two diverge. Where the weights come from:
#
#   hf download lucent517/Qwen3.5-9B-AWQ-dental-cbct-sft \
#       --local-dir models/Qwen3.5-9B-AWQ-dental-cbct-sft
#   ln -sfn Qwen3.5-9B-AWQ-dental-cbct-sft models/best_model
#
# Use --local-dir, not the bare cache: snapshot_download puts symlinks into
# blobs/ under a sha-named snapshot dir, and those dangle when only the snapshot
# is bind-mounted. A symlink that dangles inside a bind mount is the one failure
# mode this arrangement can still produce, so check it resolves before a run.
MODEL_NAME="${MODEL_NAME:-best_model}"

CASE="${1:-${CASE:-A008}}"
SPLIT="${2:-${SPLIT:-validate}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-0}"

# The facts upstream hands over with the mask, as a container path. The runner
# copies it and audits the copy -- the original is never written to, which is
# also what Grand Challenge's read-only /input requires. The RAW pool, not the
# split copy: see the header. Set FACTS_FILE=/project/dataset/$SPLIT/facts/${CASE}.json
# to feed the already-audited research copy instead.
FACTS_FILE="${FACTS_FILE:-/project/dataset/source/facts_all_622_cases_new/${CASE}.json}"
EXTRACT_FACTS="${EXTRACT_FACTS:-0}"
NO_FACTS="${NO_FACTS:-0}"

FACTS_ARGS=()
if [ "$NO_FACTS" = "1" ]; then
    FACTS_ARGS=(--no-facts)
    FACTS_MODE="none (ablation)"
elif [ "$EXTRACT_FACTS" = "1" ]; then
    FACTS_ARGS=()
    FACTS_MODE="derived in-container from the mask"
else
    FACTS_ARGS=(--facts-file "$FACTS_FILE")
    FACTS_MODE="upstream $FACTS_FILE (audited before use)"
fi

OUT_DIR="${OUT_DIR:-$PROJECT_DIR/outputs/competition_${CASE}}"

echo "[INFO] ========== Competition simulation =========="
echo "[INFO] Start:     $(date)"
echo "[INFO] Case:      $CASE ($SPLIT)"
echo "[INFO] Model:     $MODEL_NAME"
echo "[INFO] Facts:     $FACTS_MODE"
echo "[INFO] Container: $CONTAINER"
echo "[INFO] Out:       $OUT_DIR"
echo ""

PREFLIGHT=("$CONTAINER" \
           "$MODEL_DIR/$MODEL_NAME" \
           "$PROJECT_DIR/dataset/$SPLIT/images/${CASE}_0000.nii.gz" \
           "$PROJECT_DIR/dataset/$SPLIT/masks/${CASE}.nii.gz")
# Check the facts HERE rather than discovering it 200 s in, once vLLM has
# already spent the budget's long pole. The runner would fall back to deriving
# them, which is right in the container and wrong in a simulation: it would
# quietly measure a different arm than the one that was asked for.
if [ ${#FACTS_ARGS[@]} -eq 2 ]; then
    PREFLIGHT+=("${FACTS_FILE/#\/project/$PROJECT_DIR}")
fi
for f in "${PREFLIGHT[@]}"; do
    [ -e "$f" ] || { echo "[FAIL] missing: $f"; \
        echo "       (no upstream facts for this case? EXTRACT_FACTS=1 derives" \
             "them from the mask)"; exit 1; }
done

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo ""

# One srun, one container, one process tree: the runner starts vLLM itself and
# renders images in threads beside it. Handing the server to a separate srun
# would put it in a different container instance and lose the overlap being
# measured.
# $PROJECT_DIR is mounted TWICE, at /project and at its own host path. The
# second is not redundant: dataset/{split}/images/*.nii.gz are ABSOLUTE symlinks
# into dataset/source/ (the split was made with --mode symlink, because the
# volumes are 45G and the filesystem is at 94%), so they point at
# /msc/home/.../project_ToothFairy4/... and dangle inside a container where only
# /project exists. Mounting the host path makes them resolve.
#
# This is an artifact of the cluster dataset layout, NOT something the real
# container needs: Grand Challenge hands over a real file in /input.
# THE FACTS ARE PASSED, AND THEY ARE AUDITED FIRST (2026-08-16).
# --facts-file names the upstream file for the case; competition_runner.py
# copies it into the run directory and runs audit_facts.py over the COPY before
# the first generator opens it, so the reconciliation with the mask -- the FOV
# verdict and bridge_arches -- is in place for the captions AND for the source
# rules that postprocess later applies. See the header block above for why
# those two rules are inert without it.
#
# This is not the leakage the caption rule guards against: this pool is derived
# from the segmentation, not from the reference report, so the same file exists
# on Grand Challenge. EXTRACT_FACTS=1 derives it here instead, and NO_FACTS=1
# runs the ablation with none.
srun --container-image="$CONTAINER" \
     --container-mounts="$MODEL_DIR:/models,$PROJECT_DIR:/project,$PROJECT_DIR:$PROJECT_DIR" \
     --container-workdir=/project \
     python3 /project/code/competition/competition_runner.py \
        --case-id "$CASE" \
        --volume "/project/dataset/$SPLIT/images/${CASE}_0000.nii.gz" \
        --mask "/project/dataset/$SPLIT/masks/${CASE}.nii.gz" \
        ${FACTS_ARGS[@]+"${FACTS_ARGS[@]}"} \
        --out-dir "/project/outputs/competition_${CASE}" \
        --model "/models/$MODEL_NAME" \
        --schema /project/schema/schema.json \
        --project-dir /project \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --max-concurrency "$MAX_CONCURRENCY"

echo ""
echo "[INFO] End: $(date)"
