# `code/train/` — LoRA SFT on the vision path

Fine-tune Qwen3.5-9B's **vision tower** on this project's own tooth-composite
calls, so the VLM reads the images better rather than being told what to say.

Condensed from `docs/vision_sft_plan_light.md`, which keeps the full argument
and the arm-by-arm history.

## The modules

Roughly the conventional shape: config, prompt, dataset, collator, trainer.

| | |
|---|---|
| `select_sft_pool.py` | the training pool, as three case lists |
| `build_sft_targets.py` | composite calls + generated GT → SFT rows (`sft_wide.jsonl`) |
| `draft_evidence.py` | the **teacher** writes `visual_evidence` for answers it is handed |
| `check_evidence_perceivable.py` | the **screen**: can the *student* see what the teacher described? |
| `evidence_prompts.py` | the two prompts of the evidence pass. One copy, imported by both |
| `sft_prompt.py` | one row → the exact messages the pipeline would send |
| `sft_collator.py` | one row → `input_ids`, per-token labels, per-token weights |
| `dataset.py` | `RowDataset` (lazy encode in dataloader workers) and `load_rows` |
| `lora_arms.py` | the arm table, and `assert_arm()` — a config that missed its targets fails before a GPU is touched |
| `trainer.py` | the training loop. Drives the loss itself: the built-in one is an unweighted mean, and the evidence weight has to survive to the objective |
| `merge_vision_lora.py` | adapter → a servable AWQ checkpoint, at the tensor level |
| `check_prompt_parity.py` | token-id equality between training text and inference text |
| `dequantize_awq.py` | unpack AWQ's 4-bit MLPs to bf16 |

## The order things run

```
select_sft_pool  →  draft_evidence  →  check_evidence_perceivable
                                              ↓
                    build_sft_targets  →  check_prompt_parity
                                              ↓
                                          trainer  →  merge_vision_lora
```

One runner, one stage per submission:

```bash
STAGE=pool    scripts/run_sft.sh                       # CPU
STAGE=draft   sbatch ... scripts/run_sft.sh            # teacher   [GPU]
STAGE=screen  sbatch ... scripts/run_sft.sh            # student   [GPU]
STAGE=targets scripts/run_sft.sh                       # CPU, the long pole
STAGE=parity  sbatch ... scripts/run_sft.sh            # student   [GPU]
STAGE=train   sbatch ... scripts/run_sft.sh -- --arm vision+merger
STAGE=merge   sbatch ... scripts/run_sft.sh -- --adapter ... --out ...
```

`code/train/sft.py` holds the stage table; everything after `--` goes to that
stage's module verbatim, so a tuned default lives in one place. Stages are
separate submissions on purpose — each is a decision point, and the parity gate
is meant to stop the run.

## Train on bf16, merge into AWQ

Training targets `models/Qwen3.5-9B` (bf16); the adapter is merged into
`models/Qwen3.5-9B-AWQ` for serving.

Loading AWQ *for training* burned four jobs in 25 minutes and never reached a
forward pass — gptqmodel, then a Marlin shape constraint, then a JIT that wants
ninja and a CUDA toolkit, and only then the real question of whether backward
works through Marlin kernels at all.

**The merge is exact, not an approximation.** AWQ's `modules_to_not_convert`
covers `visual`, so `model.visual.*` is bit-identical between the two
checkpoints — same sha256 on `model.visual.blocks.0.attn.qkv.weight`. The arms
target only `model.visual.*`, so the LoRA is fit on exactly the tensors it is
merged into. Arm 6 merged 214/214 tensors exact.

What stays open is behavioural, not numerical: the adapter is fit with bf16
language MLPs downstream and served with 4-bit ones. AWQ measured against bf16
at mean F1 **0.572 vs 0.552** — no measurable fact-level cost — and every
validate run *is* the transfer measurement, since it serves the merged
checkpoint.

## The current arm

**`vsft_arm6` — 0.4658 official** (RadFact F1 0.5139, captioning 0.2737).
1,190 steps, 12 h 28 on one A100, peak 23.62 GiB, no dropped rows. The merged
checkpoint is `models/Qwen3.5-9B-AWQ-dental-cbct-sft`; promotion is by rename,
so that path always holds the current best.

Arm 6 is arm 5 plus two target-build changes: the nine `global` calls per case
built into training rows for the first time (`--include-arch`), and the two
`maxilla_sinus_*.scope` constants freed from `ARCH_REFUSED`. They separated
cleanly — `[3d]` +0.13 came from the arch rows, the `[sinus]` move came
entirely from the freed constant, and the tooth calls did not pay for either
(`[detail]` flat within noise on validate and on the held-out 22).

One caveat worth carrying: `maxilla_sinus_*.scope` went 0.000 → 1.000, but the
ground truth is `partially_included` on 80 of 80 sides because that is
`_resolve_scope`'s literal fallback in `parse_reports_to_gt.py`, not an
observation. The score says which constant the model settled on. Worth having —
a wrong constant is 38 wrong sentences — but it is **not** evidence the model
can read sinus extent.

## Two rules

**Nothing in `dataset/validate` ever enters training.** Not a case, not a tooth,
not a tuned threshold. The pool is drawn from `training` only, the held-out
cases are carved out of `training`, and `build_sft_targets.py` refuses a
validate case id at load time and prints the count it refused. A validate-tuned
arm produces a number that means nothing about the leaderboard.

**A silent degradation in the harness is indistinguishable from a model
result.** This project earned that twice in one week: a teacher pass that wrote
9,301 fluent strings having never seen an image (relative paths resolved against
the wrong working directory, and the screen counted unjudged as *kept*), and a
two-token prompt skew that would have sat under every training sample. Both were
caught by a component that **refused to proceed** rather than one that coped.
Prefer that shape.

### One thing that is claimed and has never run

`merge_vision_lora.py` is supposed to verify the merged checkpoint against
adapter-applied inference and fail loudly. **That check has never run, for any
arm** — arms 1, 2, 5 and 6 were all scored on merged checkpoints that never
faced it. Nothing in this repo applies a LoRA adapter at load time
(`--enable-lora`, `LoRARequest`, `PeftModel.load_adapter`: zero hits), so the
reference half has no harness, and no acceptance criterion was ever written.

What *is* verified per tensor and per arm is that the merge arithmetic is exact
and the bases are bit-identical where an arm touches them. The unverified step
is narrower: that serving the merged checkpoint reproduces adapter-applied
inference. Either build it with a criterion, or stop calling it non-negotiable.
