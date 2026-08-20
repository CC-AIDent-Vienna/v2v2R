# Vision-path LoRA SFT — competition-light execution plan

Written 2026-08-14. **This does not replace `docs/vision_sft_plan.md`** — that file stays as
the record of what was measured, why each design decision was taken, and what the full
experiment would look like. This file is the *execution subset* for the ODIN2026
deadline: what actually gets run, in what order, and what decides whether it ships.

Where the two disagree, this file wins for **what to do**; the parent wins for **why**.
Section numbers in `§x.y` form refer to the parent unless stated.

**The objective changed, and everything below follows from it.** The parent plan is
built to answer *why* the pipeline over-calls (H1 perceptual vs H2 threshold). This plan
is built to answer *does the leaderboard number go up, and is it safe to ship*. Those
need different arms and a different decision rule.

## 1. What was cut, and what that costs

| Arm | Parent | Here | Reason |
|---|---|---|---|
| **B** baseline (AWQ, validate-40) | done | **kept, already exists** | `outputs/aksssr_v7_validate/`, §5.0. Zero cost. |
| **0** rate-matched null | mandatory | **kept** | CPU-only, and it is the one control that is *also a candidate submission* — see below. |
| **1** projector only | stage C gate | **RAN ANYWAY, as a trial** | Cut here, but trained before this file was read (job 555522). Not wasted: it is what proved the rig end to end on real data — 911 rows / 55 cases, loss 0.6155 → 0.4764, **peak 22.15 GiB**, 29.3 s/step, zero dropped rows, and a merge into AWQ verified exact. Its held-out score (555716/555717) is left to land as information. It is **not** on the critical path and nothing waits for it. |
| **2** vision + projector | primary | **the whole plan** | Now trained on **all 582 training cases and all four image sources**, not the 120-case tooth-only pool. See §4.0. |
| **3** language control | *"arm 2 must never run without arm 3"* | **CUT — deferred, not deleted** | See below. |
| **4** capacity check r=64 | contingent | **CUT** | Contingent on arm 2 winning *and* time remaining. Neither is in evidence. |

### Arm 3, honestly

The stated reason for cutting it — *"I'm sure the problem is the vision part"* — is not
what the measurements say. Two results in the parent plan have the signature of an
**operating-point shift**, which is precisely what arm 3 exists to detect:

- the few-shot probe: claims −45%, precision 0.129 → 0.168, recall 0.347 → 0.250, Final
  Score +0.0081 with a 95% CI of −0.038 to +0.050;
- the zoom probe (§3.2b): 4× the patches on `post_and_core`, **7 fixed / 6 broken, exact
  McNemar p = 1.000**, and the assertion rate moving 25% → 43%.

So the prior on H1 is weaker than the cut assumes. **The cut is still right**, for a
different reason: arm 3 buys an *explanation*, and the leaderboard pays for an *outcome*.
Under a deadline, an 8 h train plus a 4 h inference run that cannot change what gets
submitted is the correct thing to lose.

**What it costs, stated plainly so it is not rediscovered later:** if arm 2 improves the
score, this plan **cannot distinguish** "the encoder learned to see" from "the model
learned to assert less". Arm 0 (§4.5 below) partially covers that — it is the cheap half
of the same question — but it is not the 1.5%-matched contrast.

**Arm 3 is deferrable, not lost.** It is two jobs (`ARM=language`, then inference)
against the same frozen payload, and the targets, collator and rig are already written.
Run it after the deadline if this is written up.

## 2. The decision rule changes

The parent pre-registers a scientific win: *"paired 95% CI excludes zero, and beats arm
0, and beats arm 3, and the gain concentrates in `[detail]`"*. On 40 cases that requires
roughly **+0.03 Final Score**, which the parent itself says is the detection floor.

That rule is wrong for this purpose. A competition does not need significance; it needs
the better point estimate and an assurance that nothing broke. **The ship rule is:**

1. **Ship arm 2 if its validate Final Score point estimate beats the AWQ baseline** —
   any margin — **and** nothing in the coverage reconciliation regressed (`called` /
   `unanswered` / `no_image`, and `n_facts_returned` vs `n_facts_asked` per call).
2. **Do not ship on a survey_facts gain alone.** The few-shot probe moved survey recall
   0.40 → 0.28 while RadFact logical recall moved +0.0023. `structured_findings_evaluation` explains
   *what* changed; it does not decide.
3. **If arm 2 and arm 0 land within noise of each other, ship arm 0.** Same number, no
   trained weights, no merge, no change to the submission container. That is strictly
   less risk on a 15-minute-per-case leaderboard.
4. **A tie against baseline ships the baseline.** A merged checkpoint that does not beat
   the stock one is pure added failure surface.

Report the paired CI anyway — it is one line of the same script — but as information,
not as a gate.

## 3. Train on bf16, merge into AWQ

**Decision (2026-08-14): arm 2 trains against `models/Qwen3.5-9B` (bf16), and the LoRA
is merged into `models/Qwen3.5-9B-AWQ` for serving.** This reverses §4.3's preferred
order and closes open item 3 by declining to answer it.

**Why.** Loading AWQ for *training* burned four jobs in 25 minutes on 2026-08-14 and
never reached a forward pass:

| job | died on |
|---|---|
| 555421 | `ImportError: Loading an AWQ quantized model requires gptqmodel` |
| 555431 | `AwqMarlinLinear`: `out_features 4304 must be divisible by [64]` |
| 555434 | Marlin fp16 JIT: `Ninja is required to load C++ extensions` |

The next step in that chain is `ninja`, then almost certainly a CUDA toolkit for the JIT,
and only *then* the actual question — whether backward works through Marlin AWQ kernels
at all. `transformers`' `is_trainable` gate returns True on a version check, which is a
statement about the package, not about the derivative. That is an open-ended cost with no
floor, spent before the experiment starts.

**Why it is safe.** §4.3 verified, on disk, that AWQ's `modules_to_not_convert` covers
`visual`, `self_attn`, `linear_attn`, `layers.0` and `mtp` — so `model.visual.*` is
**bit-identical** between the bf16 and AWQ checkpoints (same sha256 on
`model.visual.blocks.0.attn.qkv.weight`). Arm 2 targets only `model.visual.*`. Therefore:

- the LoRA is fit on exactly the tensors it will be merged into — **the merge is
  numerically exact**, not an approximation;
- **the memory budget does not change.** §4.4's 25–28 GB total already lists *"frozen
  base bf16 — 19.31 GB"*. Training bf16 was always what that arithmetic assumed; AWQ
  would have been the cheaper surprise, not the budgeted case.

**What remains open is behavioural, not numerical**: the adapter is fit with bf16
language MLPs downstream and served with 4-bit ones. Two things bound it — the parent
measured AWQ against bf16 on byte-identical images at **mean F1 0.572 vs 0.552**, i.e.
no measurable fact-level cost — and the arm-2 validate run *is itself* the transfer
measurement, since it serves the merged AWQ checkpoint. No separate transfer gate job.

**One thing to check before submitting, not to assume:** `train_vision_lora.py` defaults
`--model` to the AWQ path and carries `--awq-backend auto_trainable` plus an "AWQ:
rewrote 5 skip pattern(s)" step. Confirm those are inert when `--model` is the bf16
checkpoint — 5 minutes on the login node, against an 8 h job that silently takes a
quantizer branch.

## 4. The critical path

Everything upstream of this is **done**: pool selection, re-render, payload, teacher
evidence pass, perceivability screen, targets (`sft_wide.jsonl`, 1,850 rows, 40.5% of
field slots supervised), prompt parity, `lora_arms`, collator, trainer, job script.

> Each `sbatch` below needs a go-ahead before submission, and a card chosen by
> pending-jobs-per-card at the time.

### 4.0 The pool: all 582 cases, all four image sources

**Decision (2026-08-14): scope changes from a curated 120-case tooth-only pool to
every training case and every image.** The earlier runs are a trial; this is the
training run.

**Why the curation went away.** The 144-case pool was three filters — GT quality
(`score_asis` ≥ 0.80), audit-clean, and prefix balance. Each was re-examined:

- **`score_asis` is mostly redundant here.** It scores consensus-report against
  `dataset/*/facts/`, which is *mask*-derived — and the masks are not the label.
  Its heaviest channel is presence (0.40), and presence mismatches are **already
  dropped per tooth**: `build_row` returns `no_gt_block` for a tooth the mask has
  and the reports never mention, which is **1,145 of 4,352 calls (26%)** on the
  old pool. Dropping the *case* on top of that discards the teeth both sources
  agree on. Its non-redundant channels (crowns, implants, ian_close) are defined
  in only 307 / 109 / 301 of 582 cases.
- **The audit rejects nothing.** `audit_report_facts.py --split training` over all
  1,274 stage-1 files: **0 files with an ERROR.** The remaining findings are NOTEs
  — `not-in-text` 3,738 (overwhelmingly `teeth_present asserts 31`, i.e. stage 1's
  `presence_enumerated` expansion, not invention), `contradiction` 187,
  `arch-range` 11.
- **Prefix balance is now a stated, accepted risk.** All-582 is **69% P** against
  validate's 25%. The vision encoder sees scanner physics, so a P-heavy fit can
  hide an A/F/S regression. Accepted deliberately in exchange for 4.8× the data;
  **read §5's per-prefix breakdown before believing any aggregate gain.**

**Quality filtering moves from the case to the call**, which is where the
evidence is anyway:

| filter | granularity | effect |
|---|---|---|
| `no_gt_block` | per tooth | the reports never mention it — already ~26% of calls |
| `contradiction` (187) | per FDI | same tooth in `teeth_present` *and* `teeth_absent` — unusable, drop |
| `arch-range` (11) | per FDI | FDI outside its arch — unusable, drop |
| `null` / `gated` / `refused` / negative cap | per field | unchanged from §3 of the parent |
| `not-in-text` (3,738) | per FDI | **kept** — legitimate enumeration, not invention |

Nothing is dropped at case level. "If it is hard to decide, mask it" stays the
rule, and the mask machinery already has the vocabulary for it.

**`_gt.json` is NOT regenerated, and only stated facts are trained.** The question
was live because GT covers 11,502 tooth blocks (19.8 of 32 per case) where a
prediction covers 32: `_derivation` puts present 9,095 + absent 2,407 = exactly
those 11,502, and the 7,122 `unstated` are omitted. Filling them would reach
18,624 = 32 × 582, the prediction shape exactly, and there is a real argument for
it — a tooth outside the volume yields no call at all, so the unstated teeth that
would reach training are ones *visible in the image and unmentioned in the
report*, which for a radiologist who writes what is notable means unremarkable,
and teaching "most teeth are normal" attacks over-calling more directly than
anything else here.

**Declined anyway, deliberately.** `structured_findings_evaluation.py` walks the GT's keys, so
filling `unstated` turns 7,122 currently-unscored positions into scored negatives
and **every recorded baseline stops being comparable** — the 0.572/0.552 AWQ-vs-bf16
figures, §5's 27.6% overlay zero point, arm 1's held-out table. Changing the metric
and the model in the same step means the result cannot be attributed to either. The
field-level shape is already exact (every GT fact block carries precisely the
schema's fields, matching the prediction); only coverage differs, and `build_row`'s
`no_gt_block` drop already expresses "train only what the reports state".

**Scale, and the one thing it breaks.** ~20 tooth rows/case × 582 ≈ 11,600, plus
~9 arch calls/case ≈ 5,200 → **~16,800 rows** against today's 1,850. At accum 16
that is ~1,050 optimizer steps per epoch, and at the measured 36.0 s/step
**~10.5 h per epoch** — past the 8 h job §4.2 assumed. So the run is **1 epoch**,
which also sits inside parent R7's "≤2 epochs" forgetting bound rather than
straining it. Confirm the row count once the payload exists before booking the
wall clock.

### 4.1 Fix stage B and run it — 1×A100, ~2 h

```bash
MODEL=$PWD/models/Qwen3.5-9B STAGE=b ARM=vision+merger \
  sbatch --partition=gpu --qos=a100 --gres=gpu:a100:1 code/train/vision_sft.sh
```

`MODEL` is already an env knob (`code/train/vision_sft.sh:59`) — no code edit needed.

`STAGE=b` chains probe → smoke behind one weight load. On bf16 the probe's original
question (do gradients traverse quantized MLPs) is trivially yes, but **keep it**: it
also confirms the collator's mask produces a non-zero gradient at
`model.visual.blocks.0.attn.qkv`, which is the thing an 8 h job must not discover late.

**What stage B has to produce:** train loss on 200 calls → ~0, a printed **step time**,
and a printed **peak GPU memory**.

**If it cannot overfit 200 calls, stop.** Fix the rig; do not book the arm.

**[v] Ran 2026-08-14, job 555445.** R3 passed on bf16 — 110 `lora_B` carry a gradient at
step 0 and 110 `lora_A` come alive after one optimizer step, so the loop closes. Loss
0.97 → ~0.58 over ~2.7 epochs at **~37 s/step**.

*(Job 555443 first reported R3 as failing on a plain bf16 checkpoint containing no
quantized weight to blame. The probe was reading `lora_A`, which peft initialises against
a zero `lora_B` and which is therefore exactly zero at step 0 on every model. The
gradients were arriving; the probe was measuring the one matrix guaranteed to be zero.
`probe_backward`'s docstring now carries this.)*

### 4.1b Build the data — CPU only, and the long pole

Three steps, none needing a GPU, all of which can run while anything else queues.

**1. Render every image for every case.** The pool render was `STEPS=tooth` on 144
cases; this needs all four generators on 582. A 144-case `pano,3d,sinus` pass is
running now (job 555739) and is a strict subset — no work is wasted, the remainder
is a second job.

```bash
RUN_NAME=vsft_full STEPS=all sbatch code/pipeline/preprocess/gen_images_cpu.sh training
```

Budget from the measured rates: tooth was 15.5 min / 144 cases; 3D (VTK) is the
slow one. Expect a few hours across the `cpu` partition, which is uncontended.

**2. Build the payload.** `build_vqa_pairs.py` emits `global` *and*
`dental_elements` once the images exist — the tooth-only pool was tooth-only
because only tooth images were rendered, not because of a flag.

**3. Build the targets.** `build_sft_targets.py` gains `build_arch_row()`.
Three things differ from the tooth path and each silently corrupts the target if
the tooth code is reused:

- **Field names are not unique across facts.** `location` is free prose in
  `bone_quality_*` and an **enum** (`lingual|buccal`) in `mandible_canal_*`.
  `PROSE_FIELDS` matches bare names, so arch keys are fact-qualified.
- **There is no `unstated` signal.** `_derivation` carries present/absent/unstated
  for *teeth* only. For arch facts, `parse_reports_to_gt.py`'s doctrine is "null
  means the report did not say", and everything with a neutral value is defaulted
  **in code** — so a defaulted negative is indistinguishable from an observed one.
- **Some fields are constant.** Measured over 144 cases: `maxilla_sinus_*.scope` is
  `partially_included` in **100%** of cases — not because sinuses are, but because
  that is the literal fallback in `_resolve_scope`. Refused, on `DEFAULT_REFUSED`'s
  own grounds: a field where every example says the same thing can only teach
  "emit that value", which is R1's operating-point shift produced on purpose.
  15 of 59 arch fields have one value covering ≥95% of cases; the negative cap
  (§3.5) handles the rest.

Which facts a call covers is read from **`call.questions.json_schema.properties`**
— the grammar the server actually decodes against — rather than by re-deriving
`build_vqa_pairs.py`'s `images_needed` grouping. Re-deriving could disagree with
what was asked; reading the grammar cannot.

### 4.2 Arm 2 on everything — 1×A100 **40 GB**, ~11 h

**Pick the card by what is FREE, not by what is nominally right.** The a100 queue
has been the binding constraint all week. The job needs **22.15 GiB**, so the
constraint is availability, not capability:

| card | cards | fits 22 GiB? | note |
|---|---|---|---|
| `a100` (40 GB) | 21 | yes | most cards, but 10 are held by 5-day `a100_long` jobs |
| `a100-sxm4-80gb` | 8 | yes | ~2× memory bandwidth; step time here is optimistic for a 40 GB run |
| `h100` | 4 | yes | few cards, but often **0 pending** |
| `b200` | 8 | yes | **unverified for training — see below** |
| `a30` (24 GB) | 1 | **no, in practice** | 22.15 GiB + CUDA context leaves no margin for an 11 h job |
| `3g.20gb` MIG, `a16` | — | **no** | 20 GB / 16 GB, below the weights alone |

Before submitting, rank by *pending-per-card* **and** time-to-free — `squeue -t
RUNNING -o "%.9q %.12L"` shows when cards actually come back, which
pending-count alone hides. On 2026-08-14 the soonest b200 was 2 h 35 m out
against the soonest a100 at 11 h 29 m.

**The b200s are worth testing, and were written off for the wrong reason.**
Open item 8 records them as unusable — but that failure was **vLLM's**
(`cutlass.cute.core has no attribute 'ThrMma'`, a Blackwell kernel-library
mismatch inside the image), not PyTorch's. Training is plain torch + Triton, and
`libtorch_cuda.so` in `cbct_sft_cu128` carries **168 `sm_100` and 166 `sm_120`
references**, so the wheel is built for Blackwell. That does not settle whether
`fla`'s GatedDeltaNet Triton kernels compile on sm_100 — which is exactly the
class of thing that broke vLLM there. **Settle it with `STAGE=probe` (~2 min)
before booking an 11 h job on it**, never by reasoning.

**The 40 GB card is fine. The 60.28 GiB figure was an artifact and is withdrawn.**
`from_pretrained` leaves the model in **eval** mode, and
`GradientCheckpointingLayer.__call__` gates on `self.gradient_checkpointing **and
self.training**` — so the probe ran with checkpointing silently inactive and kept
every activation. The Trainer calls `model.train()` for you, and the real training
peak, printed after `trainer.train()`, is **22.15 GiB** (arm 1, job 555522). That is
*inside* the parent §4.4's 25–28 GB budget, which was right all along.

This matters more than a card: `a100` has **21 cards** against
`a100-sxm4-80gb`'s 8, and the queue has been the binding constraint all week.

```bash
MODEL=$PWD/models/Qwen3.5-9B STAGE=train ARM=vision+merger EPOCHS=1 \
  ROWS=$PWD/outputs/vsft_full_training/sft_all.jsonl \
  OUT=$PWD/outputs/vsft_arm2 \
  sbatch --partition=gpu --qos=a100 --gres=gpu:a100:1 --time=20:00:00 \
  code/train/vision_sft.sh
```

**Sizing, measured not assumed.** Stage B measured **36.0 s/step** (2,808 s / 78
steps, job 555445) against the parent's 25–40 s arithmetic, and arm 1 measured
29.3 s/step with fewer LoRA modules. At ~16,800 rows ÷ 16 = ~1,050 steps,
**1 epoch ≈ 10.5 h**. Ask 20 h: the queue does not refund unused wall-clock, and a
job that dies at the limit costs a full re-queue.

Everything else is unchanged and needs no decision: LR 1e-4 cosine, 3% warmup,
`lora_alpha=2r`, dropout 0.05, r=16, batch 1 × accum 16, `--evidence-weight 0.04`,
`--seed 42`, max_length 8192 with longer calls dropped and counted.

**Evidence is supervised on tooth calls only.** The teacher/screen chain (§3.3 of
the parent) was run for `dental_elements` and not for the arch facts. Rather than
spend two GPU passes extending it, arch rows carry `visual_evidence` **masked** —
the field is worth 0.04, it is auxiliary by construction, and an unscreened
teacher string is exactly the "confident prose about something it cannot see"
failure the screen exists to catch.

**No hyperparameter tuning.** There is no time for a sweep and no held-out budget
to pay for one. The `--evidence-weight 0` ablation (parent open item 11) is cut.

Everything else is unchanged from §4.4 and needs no decision: LR 1e-4 cosine, 3% warmup,
`lora_alpha=2r`, dropout 0.05, r=16, batch 1 × accum 16, `--evidence-weight 0.04`,
`--seed 42`, max_length 8192 with longer calls dropped and counted.

**No hyperparameter tuning.** There is no time for a sweep and no held-out budget to pay
for one. The `--evidence-weight 0` ablation (parent open item 11) is cut.

### 4.3 Merge → serve → score — 1×A100, ~1 h

**[v] `code/train/merge_vision_lora.py` exists and is proven** — run twice for real on arm 1
(2026-08-14), into both the AWQ and the bf16 checkpoint. Tensor surgery, CPU only,
~1 min: no model class, no quantizer, no GPU, and 4 of 5 AWQ shards hardlinked
because they hold no visual tensor. It re-verifies §3's claim per tensor rather
than citing it — **142/142 arm target tensors bit-identical**, zero missing, zero
differing — and reported `||Δ||/||W||` 2.1% for arm 1.

**[v] The `IMAGES_DIR` / `QA_JSONL` overrides now exist** (`9dee7e7`), so arm 2
replays the baseline's exact payload instead of re-rendering. `QA_JSONL=` skips
steps 1–5 wholesale. Two traps were closed while writing them: a supplied
`IMAGES_DIR` must already exist (`mkdir -p` on a typo leaves an empty dir, and an
empty images dir errors nowhere downstream), and a replayed payload carries *its*
cases — so `--case-ids` is now passed under replay, or `LIMIT=N` would take the
first N records rather than the seeded sample and two arms could score different
case sets while looking correctly configured.

```bash
python3 code/train/merge_vision_lora.py --adapter outputs/vsft_arm2/adapter \
        --out models/Qwen3.5-9B-AWQ-arm2 --arm vision+merger

SIF_PATH=$HOME/containers/vllm019_cu128.sqsh \
QWEN_MODEL_NAME=Qwen3.5-9B-AWQ-arm2 RUN_NAME=vsft_arm2 \
QA_JSONL=$PWD/outputs/aksssr_v7_validate/qa_pairs.jsonl \
  sbatch code/pipeline/aksssr_pipeline.sh validate
code/pipeline/postprocess/postprocess_now.sh outputs/vsft_arm2_validate
```

The baseline half of the comparison already exists (`outputs/aksssr_v7_validate/`, AWQ,
40/40, 17 min 19 s at `MAX_CONCURRENCY=40`) — **do not re-run it.**

### 4.4 Official ranking — the number that decides

```bash
sbatch --partition=gpu --qos=a100 --gres=gpu:a100:1 code/eval/judge_server.sh
tail -f logs/judge_server_*.log            # wait for "[PASS] Judge ready"
code/eval/eval_now.sh validate outputs/vsft_arm2_validate/synthesized_reports
scancel -n judge                           # it does not stop by itself
```

### 4.5 Arm 0 — CPU, after arm 2

**[v] `code/eval/rate_matched_null.py` is written and tested** (2026-08-14). It takes the
baseline's `predictions/`, suppresses claims until the per-field count matches arm 2's,
and writes a prediction dir `postprocess_now.sh` scores unchanged. Two blind orderings:
`--order random` (seeded uniform) and `--order prior` (drop where the finding is rarest
a priori, from per-(field, FDI) rates over the 1,274 **training** GT files — the script
refuses a `--prior-dir` under a validate path rather than trusting the caller).

**What counts as a claim follows `structured_findings_evaluation.py` exactly**, because arm 0 is scored by
it: a `bool` claims when true, a `list` claims once per element, and enums and maps are
left alone because they score as accuracy and have no false-positive column to game.
That is the same line §3.5 draws for the 6:1 cap.

It is structurally blocked until arm 2 exists — it matches *arm 2's* counts, and the
script says so rather than writing an empty dir. Zero GPU, seconds on the login node,
and it answers the one question the arm-3 cut left open that still bears on shipping:
**is the gain free?** Per §2 rule 3, if it is, ship arm 0.

**One thing the test surfaced that changes how arm 0 is read.** Suppression is
one-directional: where arm 2 claims *more* than baseline, arm 0 cannot match it and is
simply the baseline unchanged there. Running the two existing v7.1 arms against each
other (bf16 as baseline, AWQ as target — 1,018 claims down to 544) exercised the bulk
path, and the reverse direction showed 27 of 37 fields where the target claimed more. So
**arm 0 is a null for the fields where arm 2 said less, not globally**; the script prints
which fields those are, and on the rest the comparison is arm 2 vs baseline directly.
Read the per-field table, not just the total.

A `--census` mode prints any dir's per-field claim counts on its own. The AWQ baseline is
**544 claims over 31 fields**, concentrated in `with_full_crown` (150), `with_endo` (80),
`is_remnant` (65) and `with_fillings` (58) — which is where arm 0 will do most of its
work, and, per §5.0, three of the four weakest precision axes in the arm.

## 5. What to read, and in what order

Cut to four things. Everything else in §5.1 is diagnostic and does not gate a decision.

1. **`official_ranking.py` Final Score**, arm 2 vs the existing AWQ baseline, paired per
   case with a CI reported as information.
2. **Coverage reconciliation** — `called` / `unanswered` / `no_image`, and
   `n_facts_returned` vs `n_facts_asked`. *Read nothing else until this reconciles*: a
   lost call is a coverage difference between arms, not a finding about them.
3. **`structured_findings_evaluation.py`**, `PRED` beside `SUMMARY`, on the `[detail]` group — to explain a
   move, never to decide one. The baseline row is §5.0 of the parent.
4. **Overlay-mention rate** — fraction of `visual_evidence` strings citing the coloured
   outline. One `grep`-scale count, and it is the check that caught the int4 failure
   (`6e3f323`). **Record the baseline's rate before arm 2 is scored**, or the comparison
   has no zero point.

   **[v] Recorded 2026-08-14, before arm 2 exists**
   (`outputs/training_results/overlay_baseline.json`;
   kept here too because `outputs/` is gitignored and the zero point must outlive it):

   | arm | cases | evidence strings | overlay-mention |
   |---|---|---|---|
   | **AWQ validate-40 — the submission model** | 40 | 4,967 | **27.6%** |
   | bf16 validate-40 | 40 | 4,881 | 28.4% |
   | AWQ held-out 22 (tooth calls only) | 22 | 2,492 | 27.8% |

   Pattern: `red|colo(u)red|highlight*|overlay|outline(s|d)|segmentation|mask(ed)|
   contour*|delineat*`, case-insensitive, over every `visual_evidence` string.

   **The three agreeing to within 0.8 points is what makes this usable.** The rate is
   stable across quantisation *and* across case set, so it has no meaningful drift of its
   own — a move in arm 2 is attributable to arm 2. Read it as a **guard, not a target**:
   int4 failed by describing an overlay it could not see, so a large *rise* is as
   suspicious as a collapse. What would matter is a sharp fall, which would say the arm
   learned to answer without looking at the outline the whole pipeline depends on.

**`code/sft_holdout_metrics.py` is cut.** Held-out ΔAUROC and
precision-at-matched-recall are H1 instrumentation; with arm 3 gone they inform nothing
that changes the submission. If a go/no-go is wanted before spending the inference run,
score the 24 held-out *training* cases with `structured_findings_evaluation.py` — existing tooling, no new
file.

## 6. The parallel lane: caption-before-image

Runs **while arm 2 sits in the queue**, so it costs calendar time only if skipped. No
training, no new code — `code/train/check_prompt_parity.py` and `code/train/prompt_parity.sh` are
written and the a30 boots the student in ~80 s.

§3.4 established that vLLM guesses a `string` chat-template content format for Qwen3.5
and **hoists every image placeholder to the front of the user turn**. The
MedThinkVQA-style caption-before-image design — the entire reason
`build_captioned_image_blocks` exists — **has never reached the model in any run in this
repo**, the baseline included.

Serving with `--chat-template-content-format openai` restores the intended order. It is
two inference runs (baseline and fixed) against the frozen payload and costs no training
at all.

**Two rules, or this contaminates the main result:**

- It changes the prompt, so a fixed-order run is **not comparable** to §5.0's baseline.
  Score it as its own pair — fixed-order baseline vs fixed-order arm — never against a
  hoisted-order number.
- **The collator stays on `sft_prompt.CONTENT_FORMAT = "string"`.** Arm 2 trains against
  the prompt the server actually sends today. If the fixed order wins, retraining arm 2
  under it is a *later* decision with its own cost, not a mid-flight edit.

## 7. Risks — kept, and dropped

Kept as-is from parent §6: **R1** (operating-point shift — now carried entirely by arm 0
and the §2 rules), **R4** (leakage; enforced in `build_sft_targets.py`, which refuses a
validate case id at load), **R7** (forgetting — insulated by `synthesize_report.py` being
deterministic templates; still ≤2 epochs and a BLEU/METEOR non-inferiority check),
**R11** (image staleness — arm 2 must replay the frozen payload, §4.3), **R12**
(`--export` comma parsing — print the case count at job start).

Already closed and needing nothing: **R2** (structured decoding, `aa48c4b`), **R8**
(prompt parity, exact on both call shapes), **R9** (peft × transformers 5.9.0), **R10**
(`fla` chunked scan installed).

Changed:

- **R3 (train/serve quantization gap) — resolved by §3 rather than tested.** Train bf16,
  merge into AWQ, and let the arm-2 validate run be the transfer measurement. The
  `gptqmodel` route is abandoned, not deferred.
- **R5 (learning the overlay rather than the anatomy) — DROPPED from the critical path.**
  The `NO_FACTS=1` ablation is an interpretation tool. It cannot change what is submitted:
  if arm 2 scores better on validate *with* the overlay, and the overlay is present at
  submission time too, then learning the overlay is a legitimate way to win this
  challenge. Run it only if arm 2 ships and time remains.
- **R6 (label ceiling) — narrowed.** Keep the one operational instruction: `post_and_core`
  is scored separately and kept out of any aggregate a decision rests on (21 positives
  across 8 cases; baseline 0.03/0.18, the weakest axis in the arm).

## 8. Code still to write

Two files, both small. Everything else in the pipeline exists.

| Path | Why |
|---|---|
| ~~`code/train/merge_vision_lora.py`~~ | **DONE 2026-08-14 [v]**, and tested end-to-end against a synthetic arm-2-shaped adapter: 2m50s, CPU only, no model load and no quantizer. It merges at the tensor level and hardlinks the 3 of 5 AWQ shards that hold no visual tensor. **It also upgrades §3's central claim from cited to verified: all 110 arm-2 target tensors are bit-identical between bf16 and AWQ**, where the parent had spot-checked one sha256. Four refusals tested against fixtures, each leaving no output behind — all-zero `lora_B` (caught before a shard is read), a 109/110 adapter, a blown-up `lora_alpha`, and a base whose weights differ from the ones the LoRA was fit against. Output is built under `.incomplete` and renamed only after every check passes, so a walltime kill cannot leave a checkpoint that is complete to `ls` and wrong to vLLM. |
| ~~`code/eval/rate_matched_null.py`~~ | **DONE 2026-08-14 [v]**, §4.5. Imports `field_kind` / `schema_facts` from `structured_findings_evaluation.py` so "what is a claim" has one definition and arm 0 cannot drift from the metric that scores it. Tested both directions against the existing bf16 and AWQ v7.1 arms (480 claims suppressed in the direction that does real work), plus the §0 guard on a validate `--prior-dir` and the refusal to run before arm 2 exists. Re-counts the written dir and fails if any field is still above target, so a claim that was counted but not undone cannot pass silently. |

**[v] `IMAGES_DIR` / `QA_JSONL` overrides — DONE 2026-08-14** (`9dee7e7`), §4.3.
Tested against the real baseline payload under `DRY_RUN=1`: steps 1–5 skipped, all
40 records reused, container path mapped, off-project paths refused. Two traps
closed while writing them — a supplied `IMAGES_DIR` must already exist, and
`--case-ids` is now passed under replay so `LIMIT=N` cannot silently score a
different case set than the same `LIMIT` picks without replay.

**`build_arch_row()` in `code/train/build_sft_targets.py` — WRITTEN 2026-08-15, but
never switched on. Corrected 2026-08-16.**

This section previously read "the arch rows are in `sft_wide.jsonl` and arm 5
trained on them, which is why `[panoramic]` improved". **That is wrong, and it
was checked rather than argued**: `sft_wide.jsonl` is 6,567 rows and every one
of them is `kind: "tooth"`. Filtered by `train_minus_heldout.txt` it is 6,162
rows over 527 cases — the exact pair job 556129 printed — so that file, with no
arch row of any kind in it, is what arm 5 trained on.

`--include-arch` is off by default and no job ever passed it. So the gap is not
`[3d]` and `[sinus]`; it is **all nine `global` calls**, panoramic included.
Whatever moved `[panoramic]` in §8.2, it was not training on panoramic rows,
and any reading that rests on that sentence has to be redone.

The code itself was fine. `build_arch_row()`, `arch_field_order()`, the
`ARCH_PROSE` / `ARCH_REFUSED` / `ARCH_GATES` tables and `sft_prompt.py`'s
dispatch onto `CATEGORY_SYSTEM` / `CATEGORY_USER_TEMPLATE` all work: over the
582-case payload the flag builds 4,234 arch rows with **zero drops**, all nine
call kinds render and tokenize (max 6,084 tokens against the 8,192 limit).

Cut from the parent's file list: `code/sft_holdout_metrics.py` (§5).

### 8.1 What the trial established, so none of it is re-run

| | |
|---|---|
| **R3 — do gradients reach the arm** | **Yes** (job 555445): 110/110 `lora_B` non-zero at step 0, 110/110 `lora_A` after one optimizer step, including `visual.blocks.0`, the furthest point the signal travels. **A probe must read `lora_B`**: LoRA initialises B to zeros, so `dL/dA ∝ Bᵀ = 0` and every `lora_A` gradient is exactly zero at step 0 on *every* model. Reading `lora_A` reported R3 as failing on a checkpoint containing no quantized weight to blame. |
| **Step time** | **36.0 s/step** (arm 2 config, 78 steps), 29.3 s/step (arm 1, fewer modules). Parent §4.4 predicted 25–40 s. |
| **Peak memory** | **22.15 GiB** in real training. The 60.28 GiB probe figure was eval-mode with checkpointing inactive — withdrawn. A 40 GB card suffices. |
| **gptqmodel** | Unusable with this architecture at any version: it patches Triton's autotuner globally at import and kills `fla`'s GatedDeltaNet kernels, and peft imports it during `get_peft_model`, so merely having it *installed* broke bf16 training too. Recorded as a do-not-install in `env/sft_requirements.txt`. |
| **The merge** | Exact — 142/142 arm target tensors bit-identical across bf16 and AWQ, re-verified per tensor at merge time. |
| **Overlay zero point** | 27.6% on the AWQ validate-40 baseline (§5 item 4). |

### 8.2 What the full run established — 2026-08-15

The trial answered "can this train at all". This answers "does it help", and the
answer is **yes on what it trains, no on what it does not**.

**Arm 5 `vision+language` was added and is the arm that ran.** Arms 2 and 3
between them cover 8 of the language model's 32 layers -- only the
full_attention layers own a `self_attn`, and the 24 Gated DeltaNet layers were
in no arm at all. Arm 5 adds their `in_proj_qkv` / `in_proj_z` / `out_proj` to
the whole vision tower and projector: **214 modules, 22,928,896 parameters** at
r=16. It stops at `mlp.{down,up,gate}_proj` because those 93 modules are the
only ones AWQ quantizes, and `merge_vision_lora.py`'s per-tensor identity check
enforces that fence rather than a comment.

| | |
|---|---|
| **Training** (job 556129) | 708 steps, 8 h 45 m, **41.6 s/step**, peak **22.95 GiB**. Only +0.25 GiB over arm 2 for ~3x the adapter, so a 40 GB card still suffices. |
| **The merge** | **214/214 bit-identical**, all 5 shards rewritten. Confirms that language `self_attn` and `linear_attn` are as mergeable as the vision tensors -- an assertion when arm 5 was written, now checked. `\|\|Δ\|\|/\|\|W\|\|` median 1.06e-02 against arm 2's 7.64e-03. |
| **Epochs, finally measured** | eval loss **0.2725 → 0.2531 → 0.2506 → 0.2469** over 2 epochs, monotonic, never rising. Epoch 2 bought 2.4% against 7.1% for the second half of epoch 1. **One epoch is nearly as good as two on this corpus** -- worth halving future runs. |
| **The screen** | 20,878 of 30,450 evidence strings kept (69%). Per field: endodontic 0.82, canal 0.81, bone_quality 0.78, morphology 0.73, **eruption 0.48, periodontal 0.47**. |

**Held-out (22 cases) vs the un-adapted AWQ baseline**, judged against the
per-field noise band from two baseline draws: `with_fillings` +0.216 (22x
noise), `eruption_state` +0.151 on N=405 (6x), `with_full_crown` +0.111 (5x),
`with_endo` +0.198 (4x), `bone_loss` +0.036 (3x). One regression:
`canal.location` −0.038.

**Validate (40 cases) is where the real result is, and it is not the same
result.** The held-out payload is tooth-calls-only, so it measures exactly the
region arm 5 is strong in. The validate payload carries the `global` calls too:

| group | in the targets? | outcome |
|---|---|---|
| `[detail]` tooth calls | yes | `eruption_state` +0.116 (N=571), `with_endo` +0.294 (N=78), `canal.adjacent` +0.529 |
| `[panoramic]` arch | ~~yes~~ **no** (corrected 2026-08-16) | arch findings +0.088 (N=486) and +0.076 (N=330), periodontal extent +0.050 |
| `[3d]` | **no** | alveolar atrophy `present` **−0.509**, `atrophy` −0.421, maxilla −0.527 / −0.238, wisdom `impacted` −0.238 — **fixed by arm 6, §8.3** |
| `[sinus]` | **no** | `scope` **collapsed to 0.000** on both sides — **fixed by arm 6, §8.3: 0.000 → 1.000** |

**The `[panoramic]` row is why the correction above matters.** With no arch row
in `sft_wide.jsonl`, none of these four groups was supervised — so the split is
not trained-vs-untrained, it is **near-domain vs far-domain**. Panoramic is a
radiograph read with the tooth vocabulary and gained +0.088 by transfer alone;
the 3D renders are VTK surface renders in false colour, a different image
distribution entirely, and lost 0.2–0.5. That is an argument for training the
arch calls **in the mix**, not for bolting them on afterwards.

Aggregate over the 52 fields with N>=8: unweighted **0.5187 → 0.4981** (worse),
N-weighted **0.6391 → 0.6767** (better). Arm 5 wins the high-N fields it trained
on and loses many smaller untrained ones.

**THE FINDING THAT MATTERS: unsupervised calls are not left alone, they drift.**
`maxilla_sinus_{left,right}.scope` is `partially_included` in 100% of the ground
truth. The baseline answered it 15 `fully_included` / 23 `partially_included`;
arm 5 answers `fully_included` **38 times out of 38**. That field is in
`ARCH_REFUSED` -- excluded from supervision precisely *because* it is constant
and therefore carries no information. Getting no gradient, it drifted onto the
one answer that is always wrong. The `[3d]` group has the same disease in a
milder form: never built into rows, and down 0.2-0.5 across alveolar atrophy.

So "not worth supervising" and "safe to leave unsupervised" are different
claims, and this plan conflated them. A constant field is the *cheapest*
possible thing to anchor.

**What follows — done on branch `lora_sft_v2`, 2026-08-16, as arm 6.** Both
target-build changes, no new inference, one rebuild and one retrain:

1. **All nine `global` calls built into rows**, not just `3d` and `sinus` — see
   the correction in §8 for why that list was wrong. `sft_wide_arch.jsonl`:
   10,801 rows = the same 6,567 tooth rows arm 5 trained on, plus 4,234 arch
   (3D 1,746 / panoramic 2,228 / sinus 260), zero drops.
2. **The refused constants anchored.** `--supervise maxilla_sinus_left.scope
   maxilla_sinus_right.scope` frees them from `ARCH_REFUSED`; 260 slots, every
   one `partially_included`.

**ARM 6 THEREFORE CARRIES TWO CHANGES AGAINST ARM 5, AND THAT IS DECLARED, NOT
OVERLOOKED.** The alternative was a third run to separate them, on a 13-hour
job, to isolate a change whose blast radius is two fields. What makes it
readable anyway is that the second change can only touch sinus `scope`: any
move anywhere else is the arch rows. Read `scope` on its own and do not fold it
into an aggregate that is being used to judge change 1.

### 8.3 What arm 6 established — 2026-08-17

**It worked, and the two changes separated exactly as the paragraph above said
they would.** Trained 12:27:58 on one a100 (1,190 steps, ~35 s/step, peak 23.62
GiB, no dropped rows); merged exact, 214/214 tensors; scored on the same
validate-40 payload arm 5 was scored on (jobs 556541 / 556543 / 556544 / 556545).

Fields with N>=8, unweighted / N-weighted, against `val_base` and `val_arm5`:

| group | base | arm 5 | arm 6 | a6−a5 |
|---|---|---|---|---|
| `[3d]` | 0.5851 / 0.6101 | 0.5458 / 0.5800 | **0.6772 / 0.6753** | **+0.1315 / +0.0953** |
| `[sinus]` | 0.7807 | 0.5965 | **0.9298** | **+0.3333** |
| `[detail]` | 0.3743 / 0.6992 | 0.4730 / 0.7601 | **0.4893 / 0.7786** | +0.0163 / +0.0186 |
| `[panoramic]` | 0.4311 / 0.5576 | 0.4344 / 0.6164 | 0.4489 / 0.6072 | +0.0145 / −0.0093 |

- **`maxilla_sinus_{left,right}.scope`: 0.000 → 1.000**, N=19 each. The entire
  `[sinus]` group move is that one field — `mucosa_state`, `sinus_content` and
  `intrasinusal_teeth` are unchanged to three decimals. **Change 2, isolated**,
  and it confirms the §8.2 diagnosis: the field was never hard, it was
  unanchored. A constant is the cheapest thing to anchor and the most expensive
  thing to leave floating.
  **But be clear what that 1.000 is worth.** The GT is `partially_included` on
  80 of 80 sides because that is `_resolve_scope`'s literal fallback in
  `parse_reports_to_gt.py`, not an observation, so the score measures which
  constant the model settled on and nothing else — arm 5 emitted
  `fully_included` 38 times, arm 6 emits `partially_included` 38 times. Worth
  having, since a wrong constant is 38 wrong sentences, but it is **not**
  evidence that arm 6 can read sinus extent, and it should not be carried into
  any aggregate as if it were a recovered clinical finding. See
  `docs/postprocess_pipeline.md`, *The same view for `atrophy` and the three sinus
  rows*.
- **`[3d]` recovered past baseline.** Alveolar atrophy `present` 0.174 → 0.722,
  `atrophy` 0.118 → 0.667. No constant was freed in `[3d]`, so this is the arch
  rows alone. **Change 1, isolated.**
- **The tooth calls did not pay for it.** `[detail]` +0.016 on validate and
  **−0.008 on the held-out 22** (0.6612 → 0.6527) — flat within noise in both
  directions. This is the result that mattered most going in: the drift that
  §8.2 measured on *unsupervised* groups is what ruled out training the arch
  rows as a second stage on top of arm 5, and in the mixed fit it did not
  appear on the tooth side either.
- **Still below baseline:** `alveolar_bone_atrophy_maxilla.atrophy` 0.400 vs
  0.727, and `upper_left_wisdom_tooth.impacted` 0.000 on N=3. Small N, but the
  maxilla atrophy pair is worth a look — the mandible twin of the same field
  went the other way.

The eval-loss curve is **not** evidence here and should not be quoted as such:
0.2258 → 0.2132 → 0.2102 → 0.2072 against arm 5's 0.2725 → 0.2531 → 0.2506 →
0.2469 looks like a large win and is not one. The eval sets share only 6 of 30
cases (the split samples the case pool, which grew from 527 to 558), and arm 6's
is 37% arch rows by weight mass, which carry no evidence prose. An arch loss of
~0.147 reproduces the whole gap with the tooth loss unchanged.

**Arm 6 has no official-ranking number yet.** Everything above is the fact-based
eval. The 0.4115 headline is RadFact + BLEU-4 + METEOR and needs
`postprocess_now.sh` (with `--facts-dir`, for the source rules) then the judge.

**A methodological correction worth keeping.** Arm 2 was first read as a large
win because it was compared against **arm 1**, which is far below baseline on
nearly every field. Against the actual baseline it is a wash: `with_fillings`
+0.091 and `with_full_crown` +0.047, against `eruption_state` −0.074 on N=405,
`with_endo` −0.079 and `canal.location` −0.058. The reference point flipped the
conclusion, not the data. Every arm is scored against `ho_awq_base2`, never
against another arm.

Arm 2's regression on `eruption_state` is also the cleanest evidence the screen
works: eruption scored *lowest* on perceivability (0.48), arm 2 trained on
unscreened evidence and got worse there, arm 5 trained only where the student
could verify the prose and gained +0.151. That is the premise the whole evidence
pipeline rests on, tested on the field set up to falsify it.

Artifacts: `outputs/vsft_arm5/` (adapter, checkpoint-354 = 1 epoch,
checkpoint-708), `models/Qwen3.5-9B-AWQ-dental-cbct-sft` (formerly
`Qwen3.5-9B-AWQ-arm5`; named for its Hugging Face repo so a download lands on
the path the job expects), `outputs/ho_awq_arm5`,
`outputs/val_arm5`, `outputs/val_base`.

## 9. What is non-negotiable

Two rules survive the trimming completely intact, because both are cheap and both are
the kind of mistake that invalidates the work rather than degrading it.

**§0 — nothing in `dataset/validate` ever enters training.** Not a case, not a tooth, not
a tuned threshold. The pool is drawn from `training` only, the 24 held-out cases are
carved out of `training`, and `build_sft_targets.py` refuses a validate case id at load
time and prints the count it refused. Under deadline pressure this is exactly the rule
that looks tempting to bend and is worth the least to bend: a validate-tuned arm produces
a number that means nothing about the leaderboard, which is the only thing this plan is
for.

**A silent degradation in the harness is indistinguishable from a model result.** The
parent earned this twice in one week — the teacher pass that wrote 9,301 fluent strings
having never seen an image (relative paths resolved against the wrong working directory,
and the screen reported them as *kept* because unjudged counted as neither pass nor
fail), and the two-token prompt skew that would have sat under every training sample. Both
were caught by a component that **refused to proceed** rather than one that coped. Keep
that property in `merge_vision_lora.py`: verify the merged checkpoint against
adapter-applied inference on ~20 calls and fail loudly, rather than trusting that writing
the tensors worked.

**THAT CHECK HAS NEVER RUN, FOR ANY ARM. Audited 2026-08-17, and it is a gap, not
a habit anyone lapsed from.** Arms 1, 2, 5 and 6 were all scored on merged
checkpoints that never faced it. Three separate reasons, and the first is
fatal on its own:

- **The reference half does not exist as code.** Nothing in this repo applies a
  LoRA adapter at load time — `--enable-lora`, `--lora-modules`, `LoRARequest`,
  `PeftModel.load_adapter` are **zero hits** across `code/` and `jobs/`. The
  merged half is one `pool_infer.sh` away; the adapter-applied half has no
  harness at all. `merge_vision_lora.py` deliberately avoids instantiating a
  model, which is right for the merge and leaves nothing to compare against.
- **Whether it is even buildable here is unverified.** It would need
  `--enable-lora` on vLLM 0.19.0 against Qwen3.5-VL, over an AWQ base, with an
  arm that targets `model.visual.*`. This plan makes no claim that any of that
  is supported, and the container's internals are not inspectable from the login
  node (no `unsquashfs`).
- **No acceptance criterion was ever written.** "~20 calls", "diff", "fail
  loudly" is the entire specification — nothing says whether a match is token
  equality, claim-set equality or a metric within tolerance, nor how many of the
  20 may differ. The `LIMIT=1` in the printed command is one *case*, whose tooth
  calls merely happen to number ~20. And it would only be readable with
  `TEMPERATURE=0`, `RETRY_TEMPERATURE=0`, a pinned `CONCURRENCY` and
  `STRUCTURED_OUTPUTS=1` on both sides, none of which any job sets today.

**Do not read this as "the merges were unverified".** What *is* verified, per
tensor and per arm, is that the merge arithmetic is exact and that the bases are
bit-identical where an arm touches them (214/214 for arm 6). The unverified step
is narrower: that serving the merged checkpoint reproduces adapter-applied
inference. Note also that the obvious cheap substitute is confounded — diffing
merged-AWQ against bf16+adapter mixes the merge with the quantization gap, which
is §4.3's separate question.

**So either give it a criterion and build the harness, or stop calling it
non-negotiable.** A requirement restated in four places and never once executed
buys the confidence of a check without the cost of one, which is the exact
failure mode this section exists to name.
