# CBCT → radiology report, via schema-guided VQA

Dental CBCT volume + segmentation mask → anatomy-aware renders → a fine-tuned
vision-language model answers a fixed clinical schema → a deterministic template
writes the report.

Built for the **ODIN2026 / ToothFairy4** challenge, whose metric is
`0.8 × RadFact Logical F1 + 0.2 × mean(BLEU-4, METEOR)`.

## Results

Validate-40, RadFact judged by a local `qwen3-14b-text` server.

| | |
|---|---|
| **Final Score** | **0.4658** |
| Clinical — RadFact Logical F1 | 0.5139 (precision 0.5306 / recall 0.4982) |
| Captioning | 0.2737 (BLEU-4 0.1626, METEOR 0.3848) |

Where it came from, in order:

| arm | final | F1 |
|---|---|---|
| schema v6.9, prompt-only | 0.3307 | 0.3443 |
| schema v7.1, prompt-only | 0.3524 | 0.3740 |
| \+ the ten source rules | 0.4115 | 0.4470 |
| \+ LoRA SFT (arm 6) | **0.4658** | **0.5139** |

Those four are comparable to each other and to nothing else: the same judge
model and the same `radfact_lite` prompt revision throughout. Earlier numbers in
this project used a `gpt-4o` judge and are not on the same scale.

**The gains are clinical, not fluency.** Captioning sits between 0.266 and 0.276
in all four arms, without trending — the +0.135 of final score is RadFact, and
nothing else.

## Three decisions that did the work

**`schema/schema.json` is the single source of truth.** Fact names, enum values,
which images each question needs, and the shared clinical calibration text all
live there, and every stage derives from it — building the VQA calls, repairing
malformed model output, post-processing, scoring. Nothing downstream hardcodes a
fact name. Field *order* inside `object_fields` is load-bearing too: vLLM's
`guided_json` decodes in declaration order, so a field derived from others
(`impacted`, from `eruption_state` + `orientation`) must come after them, or the
model commits to the conclusion before the evidence.

**Each image is asked only what it can settle.** The panoramic arch survey
answers five findings, because at that resolution a crown, a large filling and a
bridge abutment are one bright capped tooth — as are a post and a canal filling,
and caries and a root remnant. The per-tooth composite crops keep the finer
vocabulary and are what separate them. Doubt goes into an explicit
`uncertain_teeth` list, never into a hedged value. Downstream, the *source
rules* then decide which read a finding is taken **from**, rather than voting
between every read that can assert it — worth +0.06 official on its own. Each
rule, its measurement, and what it gives up is in
[docs/postprocess_pipeline.md](docs/postprocess_pipeline.md).

**The report writer is a string template, not an LLM.** `synthesize_report.py`
is deterministic. A template cannot invent a *new* clinical error on top of
whatever the VQA already got wrong, and with a fact-level metric that ceiling is
worth more than fluency. This is why the captioning score barely moves and why
that is fine.

## Layout

```
code/
  pipeline/
    segmentation/   stage 1 — NOT in this repo; the interface it must satisfy
    preprocess/     volume + mask + facts -> PNGs + captions -> qa_pairs.jsonl
    infer/          the only GPU stage: vLLM, via OpenAI-standard response_format
    postprocess/    pred -> normalize -> classify + source rules -> report
  ground_truth/     reference reports -> generated GT, shaped like predictions
  eval/             official ranking, the fact-level survey, judge server
  train/            LoRA SFT: pool selection, targets, training, merge
docs/               postprocess_pipeline.md — every source rule, its
                    measurement and what it gives up
                    competition_pipeline.md — the 15-minute, one-case-per-start
                    Grand Challenge container this pipeline feeds (its code is
                    a separate repo)
schema/schema.json  the single source of truth
env/                one host env, one container spec
```

Each `.sh` sits next to the `.py` it drives. The `#`-comment header at the top of
every script is its real documentation — they carry the design decisions and the
reasons, and they are kept current.

Segmentation is the other half of the challenge and is **not** implemented here.
[code/pipeline/segmentation/README.md](code/pipeline/segmentation/README.md)
states exactly what a segmenter must produce for the rest to run.

## Data

Not included. Get the CBCT volumes, masks and reference reports from the
ToothFairy4 / ODIN2026 challenge under its data-use terms, and lay them out as:

```
dataset/{training,validate}/
  images/   {case}_0000.nii.gz
  masks/    {case}.nii.gz
  reports/  {case}_1.txt, {case}_2.txt, ...   (a case may have several readers)
  facts/    {case}.json
```

Case IDs carry their sub-dataset in the prefix letter (`A`, `F`, `P`, `S`) with
varying digit widths — `A004`, `S0030`. Splitting, sampling and the per-dataset
breakdown all group on that letter. Where a case has several reference reports,
official scoring uses the first alphabetically, matching the challenge's own
convention.

**Ground truth is not redistributed either — rebuild it:**

```bash
sbatch code/ground_truth/gen_ground_truth.sh validate
```

`parse_reports_to_gt.py` parses the reference reports into
`dataset/validate/outputs/ground_truth/{case}_gt.json`, in the same shape as a
prediction, which is what `structured_findings_evaluation.py` scores against.

## Run it

End to end — images, VQA pairs, inference, summaries, reports:

```bash
sbatch code/pipeline/aksssr_pipeline.sh validate

LIMIT=5 sbatch code/pipeline/aksssr_pipeline.sh validate   # smoke; SEED=42 pins the cases
RESUME=1  ...                                              # skip cases with a _pred.json
DRY_RUN=1 ...                                              # no vLLM calls
RUN_NAME=my_arm ...                                        # separate output directory
```

Everything after inference is CPU-only, so the tuning loop needs no GPU and no
model:

```bash
code/pipeline/postprocess/postprocess_now.sh outputs/<run>_validate
CASE_IDS="A008 A019" code/pipeline/postprocess/postprocess_now.sh outputs/<run>_validate
NO_SOURCE_RULES=1   code/pipeline/postprocess/postprocess_now.sh outputs/<run>_validate
```

Scoring — start the judge **once** and keep it, rather than reloading a 14B model
per evaluation:

```bash
sbatch --partition=gpu --qos=a100 --gres=gpu:a100:1 code/eval/judge_server.sh
tail -f logs/judge_server_*.log            # wait for "[PASS] Judge ready"
code/eval/eval_now.sh validate outputs/<run>_validate/synthesized_reports
scancel -n judge                           # it does not stop by itself

NO_RADFACT=1 code/eval/eval_now.sh ...     # BLEU/METEOR only, no GPU
```

There is no test suite and no linter. Verification here means `LIMIT=5` smoke
runs and the survey-to-survey diff that `structured_findings_evaluation.py`
prints automatically.

## Train it

The 0.4658 arm is LoRA SFT on the vision tower and merger, trained on the
training split minus a 24-case held-out gate:

| | |
|---|---|
| arm | `ARM=vision+merger` — vision blocks + the vision/text merger, not the LM |
| epochs | 1 |
| rows | `build_sft_targets.py --include-arch`, filtered by `train_minus_heldout.txt` → 10,207 |
| base | bf16 `Qwen3.5-9B`; the adapter is merged into `Qwen3.5-9B-AWQ` afterwards, tensor by tensor |
| held out | 24 cases, 6 per sub-dataset, for the eval-loss curve — never for the score |

`--include-arch` is the change that separated arm 6 from arm 5: it puts the nine
`global` arch-level calls per case into the training rows, which every arm
before it trained without.

```bash
# 0. the case lists: all_cases.txt, heldout.txt, train_minus_heldout.txt
python code/train/select_sft_pool.py --out-dir outputs/training_results/sft_pool
# 1. render the training split
sbatch code/pipeline/preprocess/gen_images_cpu.sh training
# 2. teacher writes visual_evidence for the answers it is handed
QWEN_MODEL_NAME=Qwen3.5-27B EV_DIR=evidence_27b sbatch code/train/draft_evidence.sh
# 3. screen it: can the STUDENT see what the teacher described
EV_DIR=evidence_27b sbatch code/train/screen_evidence.sh
# 4. train (build_sft_targets --include-arch runs inside this job)
MODEL=$PWD/models/Qwen3.5-9B STAGE=train ARM=vision+merger EPOCHS=1 \
  CASE_LIST=$PWD/outputs/training_results/sft_pool/train_minus_heldout.txt \
  sbatch code/train/vision_sft.sh
# 5. merge the adapter into the AWQ checkpoint, at tensor level
OUT_MODEL=models/Qwen3.5-9B-AWQ-arm6 sbatch code/train/merge_arm.sh
# 6. score it on an existing qa_pairs.jsonl
QWEN_MODEL_NAME=Qwen3.5-9B-AWQ-arm6 RUN_NAME=arm6 \
  sbatch code/pipeline/infer/pool_infer.sh
```

Step 0 is not optional — `train_vision_lora.py` treats a missing `heldout.txt`
as a failure rather than a skip, because a check that disappears when its input
moves is worse than no check. It has a second mode, `--ranking`, that selects a
pool by ground-truth quality using `survey_gt_quality.py` and
`audit_report_facts.py`; arm 6 uses neither, and the held-out 24 it draws are
not arm 6's own 24 — same 6/6/6/6 shape, different names, and the gate is not
what the 0.4658 is measured on.

`check_prompt_parity.py` is the acceptance test that matters: it asserts the
training path and the serving path produce **token-id-identical** prompts.
Break that and the model is fine-tuned for a prompt it will never see.

**Model:** the merged checkpoint is not public yet — the weights are released
with the paper. Until then, the recipe above is the whole of it: everything that
produced arm 6 is in this repo.

## Environment and hardware

Two: one host conda env and one container.

```bash
conda env create -f env/environment.yml     # cbct_base; pulls requirements.txt
conda activate cbct_base
python -m nltk.downloader wordnet omw-1.4   # DATA, not a pip package
```

**The third line is not optional if you intend to compare scores.** `nltk`
installs code, never corpora, and METEOR is half the captioning score — 10% of
the Final Score. Without wordnet, `official_ranking.py` scores METEOR with the
lite implementation adapted from the challenge's own `evaluate.py`: correct
code on a different scale, 0.3616 where the table above says 0.3848, i.e. a
Final Score of 0.4635. It warns on the fallback and records `meteor_backend` in
`summary.json` either way, so a run always states which one produced its number.
Every number here is `"nltk-wordnet"`.

`cbct_base` runs the whole CPU path — the four renderers, `build_vqa_pairs.py`,
postprocess, report synthesis, ground truth and scoring — and the inference
*client*. No GPU, no torch, which is what makes
`code/pipeline/postprocess/postprocess_now.sh` usable as a fast loop on a
laptop or a login node.

The container is the vLLM **server**, and by spec the training stack as well:
`env/container_requirements.txt`. Nothing in the pipeline imports vLLM —
`run_vqa_inference.py` is an HTTP client that sends OpenAI-standard
`response_format` — so the two halves only ever needed to agree over a socket.
Serving and training can share one image because vLLM 0.22.0 pins
`torch==2.11.0` and allows `transformers` 5.9.0, which is exactly what the LoRA
training needs; vLLM 0.19.0 could not, and that is why this used to be three
images. The header of that file carries the table and the build steps. **It is
a spec, not a built image** — verify a first build rather than trusting it.

Which means the scripts do not yet describe one image, and say so honestly:

- The **serving** default differs by job. `aksssr_pipeline.sh`,
  `judge_server.sh` and `gen_ground_truth.sh` default to `extraction.sqsh`
  (vLLM 0.22.0); `pool_infer.sh`, `draft_evidence.sh`, `screen_evidence.sh` and
  `prompt_parity.sh` to `vllm019_cu128.sqsh` (vLLM 0.19.0). Those are the two
  images that produced the numbers above, and the defaults are left pointing at
  them rather than at a merge that has not been measured. `SIF_PATH` overrides
  every one of them, so one built image is one variable.
- **Training runs on the host, not in a container.** `vision_sft.sh` calls
  `$SFT_PY`, an interpreter path. Once the merged image is built, point
  `SFT_PY` at its python; until then the training stack is its own env.

The repo is mounted at `/project` inside the container, which is why
`build_vqa_pairs.py` writes image paths relative to `--project-dir`: the same
`qa_pairs.jsonl` has to resolve on the host and in the container.

`radfact-lite` is pinned to a git SHA, not a PyPI version, and the pin is
load-bearing — PyPI's 0.1.0 predates a rewrite of the TOOTHFAIRY prompts, so
`pip install radfact-lite` gives a score not comparable with any number here.
`official_ranking.py` reports which revision it found.

Inference and the judge each want an A100; a vLLM load takes 15–25 minutes,
which is why the pipeline starts the server first and renders while it loads.

**These scripts encode one site's SLURM configuration.** Five things to change
for your cluster, and nothing else is site-specific:

| what | where | override |
|---|---|---|
| partition / QoS / GRES | `#SBATCH` lines — `--partition=gpu --qos=a100 --gres=gpu:a100:1`, and `--partition=cpu --qos=cpu` for the render jobs | edit in place |
| repo location | `$HOME/project_ToothFairy4`, the default in every `sbatch` job | `PROJECT_DIR` |
| container image | `$HOME/containers/*.sqsh`, launched through pyxis (`srun --container-image=`) | `SIF_PATH` |
| host interpreter | `$HOME/miniconda3/envs/cbct_base/bin/python3` | `PYTHON` |
| training interpreter | `$HOME/miniconda3/envs/cbct_sft_cu128/bin/python3` | `SFT_PY` |

`PROJECT_DIR` is a literal rather than derived from the script's own path
because SLURM copies a batch script into a spool directory before running it,
so `BASH_SOURCE` on a compute node points at `/var/spool/...` and not at the
checkout. The two scripts that are *not* submitted with `sbatch` need nothing
set: `postprocess_now.sh` walks up for `schema/schema.json`, and
`split_survey.sh` uses paths relative to the repo root it is run from.

They are left explicit rather than half-abstracted: a partially portable batch
script is harder to fix than an honest one.

## License

MIT, for the code — see [LICENSE](LICENSE). Challenge data and upstream model
weights carry their own terms and are not redistributed here.
