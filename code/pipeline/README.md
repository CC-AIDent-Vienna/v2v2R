# `code/pipeline/` — the inference path

One CBCT volume in, one radiology report out.

`infer.py` is the entry point and `scripts/run_infer.sh` submits it. The four
subdirectories below are the stages, in order — though `infer.py` drives only
the last three: `segmentation/` is the handover, and the mask and facts arrive
as inputs.

| | |
|---|---|
| `segmentation/` | the mask + facts handover — inputs to this pipeline, not products of it. The facts audit is folded into `extract_facts.py` and runs at extraction |
| `preprocess/` | volume + mask → rendered images → `qa_pairs.jsonl` |
| `vqa/` | the VLM calls — the only stage that needs a GPU |
| `postprocess/` | predictions → summaries → report — the rules it applies are [`configs/postprocess/`](../../configs/postprocess/), not constants in the code |

## Run it

```bash
# One entry point, three stages. They are split because they want different
# hardware, not for modularity -- see scripts/run_infer.sh.
STAGE=images sbatch scripts/run_infer.sh validate   # CPU only, holds no GPU
STAGE=infer  sbatch scripts/run_infer.sh validate   # the only GPU stage
STAGE=post         scripts/run_infer.sh validate    # login node, seconds/case
STAGE=all    sbatch scripts/run_infer.sh validate   # one job, for one case

# Or call the Python directly: one case, every file named.
python code/pipeline/infer.py \
    --case-id A008 \
    --volume     dataset/validate/images/A008_0000.nii.gz \
    --mask       dataset/validate/masks/A008.nii.gz \
    --facts-file dataset/validate/facts/A008.json \
    --model      models/Qwen3.5-9B-AWQ-dental-cbct-sft \
    --out-dir    outputs/one_case
```

`--partition`/`--qos`/`--gres` are deliberately **not** in the script: they
differ per stage and per site, so they go on the `sbatch` line —
`--partition=cpu --qos=cpu` for `images`, `--partition=gpu --qos=a100
--gres=gpu:a100:1` for `infer` and `all`.

Common overrides: `MODEL_NAME`, `MODEL_DIR`, `RUN_NAME`, `OUT_DIR`, `GT_DIR`,
`CASE_ID` / `CASE_IDS` / `LIMIT`, `RESUME=1`, `NO_FACTS=1`, and
`POSTPROCESS_CONFIG=configs/postprocess/<arm>.yaml`.

## Who does what

| | owner | produces |
|---|---|---|
| **A. Segmentation** | *(segmentation)* | `mask.nii.gz` — per-tooth FDI labels + jaw structures |
| **B. Facts** | *(segmentation)* | `facts.json` — which FDI numbers are present/absent, already corrected against the mask |
| **C. Report** | *(VQA)* | rendered images → VLM reads → `report.txt` |

**vLLM belongs entirely to C.** B is the handover point: everything
downstream needs to know which teeth actually exist, and whether they were in
the volume at all.


## The pipeline

```
 t=0 ┌─────────────────────────────────────────────────────────────┐
     │ 1a. START vLLM  (background, GPU)                           │
     │     loads model weights, compiles kernels, allocates KV      │
     └─────────────────────────────────────────────────────────────┘
     ┌─────────────────────────────────────────────────────────────┐
     │ 1b. segmentation  ->  mask.nii.gz + facts.json (CPU/GPU)    │
     │      ...with the facts AUDITED against the mask as they are │
     │         extracted (~2 s), so 1c renders from audited facts  │
     │ 1c. render images from volume + mask + AUDITED facts  (CPU) │
     │       panoramic | 3D views | sinus  (parallel)              │
     │       then 32 tooth close-ups (needs panoramic + 3D)        │
     └─────────────────────────────────────────────────────────────┘
                              │
         vLLM ready ──────────┤ 2. INFERENCE starts as soon as the
                              │    server is up AND some images exist.
                              │    It does NOT wait for all 32 crops.
                              ▼
                        3. post-process -> report.txt
```

## Container requirements

Base must have **CUDA 12.8** — the platform runs driver 570, and CUDA 13 needs
driver 580+. A cu130 image will not start at all.

| | version | why |
|---|---|---|
| NVIDIA driver (theirs) | 570 | fixed by Grand Challenge |
| CUDA | **12.8** | anything 12.x works on driver 570; 13.x does not |
| vLLM | **0.19.0** | newest release still on CUDA 12.8 *and* supporting Qwen3.5 |
| torch | 2.10.0+cu128 | pulled in by vLLM 0.19.0 |

### Two environments in one image

Segmentation and VQA both want torch, and probably not the same version. To
avoid a conflict that only shows up at runtime, each half gets its own venv:

```dockerfile
# segmentation
RUN python3 -m venv /opt/seg   && /opt/seg/bin/pip   install -r seg_requirements.txt
# VQA + vLLM
RUN python3 -m venv /opt/vqa   && /opt/vqa/bin/pip   install -r vqa_requirements.txt
```

Each step is then run with its own interpreter (`/opt/seg/bin/python …`,
`/opt/vqa/bin/python …`). They never import each other — the handover is
`mask.nii.gz` and `facts.json` on disk — so they do not need to share anything.
