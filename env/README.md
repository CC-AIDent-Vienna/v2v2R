# Environments

Three files, two environments, and the split between them is a design decision
rather than an accident.

| file | what it builds | used for |
|---|---|---|
| `environment.yml` + `requirements.txt` | `cbct_base`, a host conda env | the entire CPU path |
| `sft_requirements.txt` | `cbct_sft_cu128`, a second host env | LoRA-SFT training only |

## `cbct_base` — the CPU path

```bash
conda env create -f env/environment.yml     # pulls requirements.txt via pip
conda activate cbct_base
```

Everything that touches nibabel / numpy / VTK / PIL or `schema/schema.json`
runs here, on the host, never inside a container: image generation,
`build_vqa_pairs.py`, `normalize_pred.py`, `postprocess_pred.py`,
`synthesize_report.py`, `survey_facts.py`, `parse_reports_to_gt.py` and
`official_ranking.py`.

That is most of the pipeline, and none of it needs a GPU — which is what makes
`code/pipeline/postprocess/postprocess_now.sh` usable as a fast iteration loop
on a login node or a laptop.

**`radfact-lite` is pinned to a git SHA, not a PyPI version, and the pin is
load-bearing.** PyPI's 0.1.0 predates a rewrite of the TOOTHFAIRY parse and
entailment prompts, so `pip install radfact-lite` gives the older prompts and a
score that is not comparable with any number in this repo.
`official_ranking.py` reports which revision it found, in
`_radfact_prompt_revision()` — check it before comparing runs.

## `cbct_sft_cu128` — training

```bash
conda create -n cbct_sft_cu128 python=3.12
conda activate cbct_sft_cu128
pip install -r env/sft_requirements.txt
```

Separate from `cbct_base` because training needs `peft` / `transformers`, which
the vLLM serving container does not carry, and because it pins a CUDA 12.8
torch build. Read the header of `sft_requirements.txt` before installing
anything into it — it records, at length, why `gptqmodel` must **not** be
present: it monkey-patches Triton's autotuner globally at import, which breaks
the Qwen3.5 GatedDeltaNet kernels on every forward pass, and merely having it
installed is enough because `peft` imports it during `get_peft_model`.

## Inference

Inference is not in either env. `run_vqa_inference.py` talks HTTP to a vLLM
server and runs inside that server's container, where the repo is mounted at
`/project`. That mount is the reason `build_vqa_pairs.py` writes image paths
relative to `--project-dir`: the same `qa_pairs.jsonl` has to resolve both on
the host and inside the container.

## A note on the pins

These files were captured from a working cluster environment, so the versions
are what that environment had rather than the minimum this code needs. If a pin
fights your platform, the ones that actually matter are `radfact-lite` (above),
`nltk` (BLEU/METEOR), `nibabel` + `numpy` (volume IO), `vtk` (3D renders) and
`Pillow` (compositing).
