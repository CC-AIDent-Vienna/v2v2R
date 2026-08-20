# Vision-path LoRA SFT on the tooth composite — successor to the few-shot probe

Rewritten 2026-08-12 for schema **v7.1**, the generated-GT scoring path
(`structured_findings_evaluation.py`), the cu128 / vLLM 0.19.0 submission environment, and
**`Qwen3.5-9B-AWQ`**. Supersedes the v6.9 draft entirely.

**Stage A1 has now been run** (2026-08-13, branch `lora_sft`) and this document is
updated to what it measured rather than what it predicted. Everything from stage A2
onward is still design only.

Conventions inherited from `docs/fewshot_probe_plan.md`: one payload replayed by every arm,
prompt parity through the pipeline's own renderer, and *a lost call is a difference in
coverage between the arms, not a finding about them*.

**[v]** = measured. **[?]** = not settled without changing system state.

## 0. The one rule that outranks the rest

**Nothing in `dataset/validate` ever enters training.** Not a case, not a tooth, not a
tuned threshold. The 40 validate cases exist to produce one number per arm at the end,
and a held-out set that has been looked at while fitting stops being one.

That is stronger than "do not put validate cases in the training file", and the extra
strength is the part that gets lost:

- **The training pool is drawn from `training` only** — §3.5's ranking file is
  `outputs/gt_quality_training/case_ranking_*.csv`, and there is no validate equivalent
  by design.
- **The held-out set for tuning is carved out of `training`**, not borrowed from
  validate (§3.5, 25 whole cases). Every decision that needs a number *during* the
  experiment — the stage-C gate, early stopping, ΔAUROC, deciding arm 4 is worth
  running — reads that set. Validate is read once per arm, at the end.
- **`build_sft_targets.py` enforces it at load time** rather than trusting the file
  list: it refuses to emit a row whose case id appears under `dataset/validate`, and
  prints the count it refused. R4's guard is this, not a convention.
- Validate's 93 `_gt.json` are regenerated and audited alongside training's because
  they are the *scoring* reference (`structured_findings_evaluation.py --gt-dir`). Building GT for a
  split is not training on it — but it is the one place the two paths touch, so the
  separation is worth stating where someone editing that code will read it.

## Context

### What the few-shot probe actually showed

Both arms ran to completion, 662/662 calls, one frozen payload, 24 training cases, and
the GT banner has since come off (`7aee275`, `REPORT_GT` 204 → 151 findings).

| | zero-shot | few-shot |
|---|---|---|
| claims | 583 | **322** (−45%) |
| false positives | 504 | **266** (−47%) |
| true positives | 75 | **54** |
| precision | 0.129 | **0.168** |
| recall | 0.347 | **0.250** |

**Few-shot cut false positives, not false negatives** — false negatives *rose*, which is
why recall fell. And on the metric that decides the challenge:

| | zero-shot | few-shot | Δ |
|---|---:|---:|---:|
| RadFact logical precision | 0.3209 | 0.3411 | +0.0202 |
| RadFact logical recall | 0.3390 | 0.3413 | +0.0023 |
| Final Score | 0.3241 | 0.3323 | **+0.0081** |

Paired per case: 14 better, 10 worse, mean +0.0062, **95% CI −0.038 to +0.050**. The
interval crosses zero and per-case variance is ~18× the mean difference.

**The correct reading is not "few-shot failed" but "few-shot moved the operating point
and nothing else".** Fewer claims, more of them right, more real findings missed, no
measurable movement in the official score. That is a pure precision/recall trade along
one ROC curve — which is exactly the null hypothesis this plan has to price before
claiming anything for SFT.

Two findings from that probe that survive and constrain this one:

1. **Survey recall fell 0.40 → 0.28 while RadFact logical recall did not move (+0.0023).**
   The claims few-shot dropped were largely ones the reference reports never made. So
   **trade-off arguments belong on the official metric, not on the survey's recall column** —
   the old plan's "recall must not fall by >0.05" success criterion was measuring the wrong
   thing and is replaced in §5.
2. **`post_and_core` collapsed to 20 claims / 0 correct.** The probe's own number stands.
   The conclusion drawn from it — *"combined with an inter-reader Jaccard of 4.2%, it is
   not a learnable field"* — **does not**, and is corrected in §3.2: that Jaccard measures
   how often two readers wrote about the same tooth, not whether they disagreed, and the
   labels have always been the union of both. The field is trained. What the 20/0 result
   still says is that the *baseline* cannot do it, which is a statement about the model.

### Why the vision path, and why now

`6e3f323` is the sharpest evidence in the project that the failure is perceptual.
`Qwen3.5-9B-int4-AutoRound` produces fluent, schema-valid answers that **invent
pathology** — `A008/tooth_11`, a normal upper central incisor, comes back
`is_remnant=True`, `complete_bony_inclusion`. The bf16 control on the same container,
same vLLM, same prompt, temperature 0 answers correctly. The mechanism is named in the
evidence strings: **int4 never mentions the coloured outline and describes grayscale
only**, while bf16 opens with *"The RED OUTLINE encompasses a single-rooted tooth
structure"*. Every generator draws coloured overlays precisely to say which structure to
read, so a model blind to the overlay is reading the wrong object — and that is the whole
route from a healthy incisor to a buried root remnant.

That is a perceptual defect producing over-calling, observed directly. It does not prove
the bf16 model's over-calling has the same cause, but it establishes the mechanism exists
in this pipeline and is invisible to every text-level check.

**Hypotheses.** H1 (perceptual): the encoder does not resolve the feature at composite
scale, so the LM falls back on a prior. H2 (decision threshold): the features are there,
the output policy is biased toward asserting. **H0 (rate): any intervention that makes the
model say "no" more often raises precision** — now not a hypothesis but an observed
result. A vision arm run alone cannot separate these.

**[v] One sub-case of H1 is already closed.** "The encoder cannot resolve the feature at
composite scale" was tested directly for `post_and_core` by re-rendering at ~4× the
patches and re-asking (§3.2b): no effect (McNemar p = 1.000), and what moved was the
assertion rate, 25% → 43%. So for that field the fix is not more pixels — and the shape
of the result is H0's, on an intervention nobody had suspected of it.

## 1. Scope: the tooth composite call only

Per the test-phase decision, this experiment trains and evaluates **only** the per-tooth
composite call. That is a large simplification and it is well justified by measurement.

**[v]** from `outputs/aksssr_v6_validate/qa_pairs.jsonl` + `logs/aksssr_v6_553351.log`
(40 cases):

| | count | share |
|---|---:|---:|
| tooth-composite requests | **781** | **71.5%** of calls |
| case-level (global) requests | 311 | 28.5% |
| tooth-composite inference time | **7,401 s (2h03m)** | **80.5%** |
| global inference time | 1,790 s (30m) | 19.5% |

9.5 s per tooth call against 5.8 s per global call; 19.5 composites per case (781/40, of
1,280 possible slots — an unsegmented tooth gets no composite, which is how
`postprocess_pred.py` infers absence).

The image: `code/pipeline/preprocess/create_tooth_detail.py` → `{case}_tooth{FDI}_composite.png`, a **3×3
grid** (axial crown/mid/root, coronal post/mid/ant, sagittal R/mid/L), target tooth
**outlined in red in every tile**, mandibular canal filled translucent yellow on lower
teeth. **[v] 1343×1356 px, identical across every file sampled → 1,764 vision tokens** —
against 768 for the entire panoramic. Rendering costs ~2 min/case, CPU.

All six `dental_elements` facts read this one image and are **deduplicated into a single
VLM request per tooth** — 5 facts per call, 6 for the lower molars (`tooth_{fdi}_mandible_canal`,
`applies_to_fdi: [36,37,38,46,47,48]`, coronal row only):

| fact | fields (decoded order) |
|---|---|
| `tooth_{fdi}_morphology` | `visual_evidence, is_remnant, with_caries, with_root_fracture, with_endo, with_fillings, with_full_crown, with_post_and_core` |
| `tooth_{fdi}_eruption` | `visual_evidence, eruption_state` |
| `tooth_{fdi}_endodontic_treatment` | `visual_evidence, filling_quality, periapical_lesion` (gated on `morphology.with_endo`) |
| `tooth_{fdi}_periodontal_status` | `visual_evidence, bone_loss, furcation_involvement, findings` |
| `tooth_{fdi}_bone_quality` | `visual_evidence, present, type` |
| `tooth_{fdi}_mandible_canal` | `visual_evidence, location, adjacent_to_teeth` |

**This is the ideal SFT target**: one call type, one image type, one uniform resolution,
one output shape. Every complication the v6.9 draft carried — per-call-type exemplar
families, length bucketing across a 3× token spread, sinus staleness, arch-vocabulary
aliasing — disappears.

**The largest test-phase lever was not SFT, and it is now done.**
`run_vqa_inference.py` used to issue its ~38 calls per case in a plain sequential loop,
one request in flight, so vLLM's continuous batching had nothing to batch and the GPU sat
idle at 4.0 min/case. It now issues **all** of them concurrently — global and per-tooth —
up to `MAX_CONCURRENCY`. This is safe because `TRUST_MODEL_ABSENCE` is off,
so nothing in a tooth call depends on the global calls; the barrier is reinstated
automatically if that flag is turned on. Results are assembled in call order, so
predictions are byte-identical to the sequential path (verified against a stub server:
38 calls both ways, identical output, 15.45 s → 2.76 s).

**Three more test-phase levers landed with it** (`e594329`), lifted from
`code/competition/competition_runner.py`, where the same three are what make one case fit in 15
minutes. They matter here because five arms each pay them over 40 cases:

- **vLLM starts before image generation, not after it.** The two halves want different
  hardware and neither reads the other's output, so run serially they simply add: a
  40-case run paid ~20 min of weight load *after* ~2 min/case of rendering, with the
  card idle throughout the first hour. `code/pipeline/aksssr_pipeline.sh` now launches at step
  6a and waits only at 7b.
- **`/health` is polled every second** instead of `sleep 90` then every 10 s. After the
  reordering the server is usually ready before the wait is even reached.
- **`MAX_CONCURRENCY=auto`** reads `GPU KV cache size: N tokens` out of the job log and
  divides by ~6,200, capped at 40 (a case holds ~24–38 calls and they batch within a
  case). The old fixed 8 was one *eighth* of an A100-40GB's ~392,741-token pool.

None of the three has been exercised by a real job yet — `bash -n` only. **[?]**

**A10G sizing — memory is not the binding constraint.** AWQ weights are **[v] 11.53 GiB**;
24 GiB × 0.90 leaves ~8 GiB for KV after activations and CUDA graphs; the measured KV
density is **[v] 35.6 KiB/token** (`logs/aksssr_v6_553351.log`: 13.34 GiB → 392,741
tokens), which is low because Qwen3.5 is hybrid — only 8 of 32 layers carry a per-token KV
cache, the other 24 being Gated DeltaNet with a per-*sequence* recurrent state. That gives
a ~237,000-token pool, and at ~6,200 tokens per call **~38 concurrent calls fit**. What
limits an A10G is compute, not memory.

**One trap worth naming:** vLLM's startup line *"Maximum concurrency for 32,768 tokens per
request: N"* is computed at `--max-model-len`, not at the size of the requests this
pipeline sends. Ours are ~5× smaller, so sizing `MAX_CONCURRENCY` off that number directly
leaves most of the card unused. Read *"GPU KV cache size: N tokens"* and divide by ~6,200.

**The remaining prefill saving, not taken.** `build_user_blocks` puts the image first and
the shared instruction text — `TOOTH_SYSTEM`, `HOW_TO_READ_A_TOOTH`, guidance,
`output_schema`, ~3.5k tokens identical across all 29 tooth calls — *after* it, so none of
it is a common prefix and `--enable-prefix-caching` can only hit the system turn. Moving
that text ahead of the image would make it cacheable across every tooth in a case, which
on a compute-starved A10G is probably the largest prefill win left. It also changes the
prompt and therefore the answers, so it is a measured experiment, not a free optimisation.

## 2. Arms

Parameter counts from the safetensors shapes. Vision block ×27 (`attn.qkv` [3456,1152],
`attn.proj` [1152,1152], `mlp.linear_fc1` [4304,1152], `mlp.linear_fc2` [1152,4304]) →
481,248·r. Merger → 17,920·r. Language full-attention layer ×8 (`layer_types` puts
`full_attention` at 3,7,…,31; the other 24 are Gated DeltaNet and own no `self_attn`) →
245,760·r.

| Arm | Targets | r | Trainable | % |
|---|---|---|---|---|
| **B** baseline | none | — | 0 | — |
| **0** rate-matched null | none (CPU post-hoc) | — | 0 | — |
| **1** projector only | `model.visual.merger.{linear_fc1,linear_fc2}` | 16 | **286,720** | 0.003% |
| **2** vision + projector ← *primary* | `model.visual.blocks.*.{attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}` + merger | 16 | **7,986,688** | 0.083% |
| **3** language control | `model.language_model.layers.{3,7,…,31}.self_attn.{q,k,v,o}_proj` | 32 | **7,864,320** | 0.084% |
| **4** capacity check (contingent) | as arm 2 | 64 | 31,946,752 | 0.340% |

**[v] All four verified** (`code/train/lora_arms.py`, meta device, no GPU, seconds): every
count above is what `print_trainable_parameters()` reports, with 2/2, 110/110, 32/32
and 110/110 target modules hit. Verified against **`Qwen3.5-9B-AWQ`** as well as bf16 —
AWQ's `modules_to_not_convert` covers `visual` *and* `self_attn`, so all three arms
target weights quantization never touched, and the module trees are identical. The
percentages are against a 9,409,813,744-parameter base. Arm 4's earlier figure
(30,799,872) counted the vision blocks but not the merger; "as arm 2" makes it
31,946,752.

**Arms 2 and 3 differ by 1.5% in trainable parameters.** Same data, same recipe, same
steps, same schedule, same seed — only the *location* differs. That contrast is the
experiment, and **arm 2 must never run without arm 3**.

| Comparison | Reading |
|---|---|
| **2 > 3** | H1 — over-calling is perceptual. The result this is for. |
| **3 ≥ 2** | H2 — the vision path is not where the fix lives; pivot to decoding/threshold work. |
| **1 ≈ 2** | The features were extracted; the *interface* to the LM was the bottleneck. One matrix pair fixes it. |
| **1 ≪ 2** | The encoder must change; a linear remap cannot invent resolution. |
| **any ≈ 0** | **H0** — nothing perceptual learned. Kills the result. |

**Arm 0 (rate-matched null) is now mandatory, not optional**, because the few-shot probe
observed exactly this outcome. Take the baseline's `predictions/` and suppress claims
until the per-field claim count matches arm 2's, by a deliberately non-perceptual
ordering (seeded uniform-random, and separately lowest-prior-field-first). Write a
prediction dir the normal CPU path scores unchanged. Zero GPU, reports before training
starts.

**Two naming traps, and the assertion that catches both.** The checkpoint uses
`model.visual.*` / `model.language_model.*`, **not** Qwen2-VL's `visual.*` /
`model.layers.*` — the naming every LoRA recipe on the internet was written against. A
copied PEFT `target_modules` regex matches nothing and `get_peft_model` reports 0
trainable parameters *without erroring*: the job starts, spends its wall-clock, and
writes an empty adapter.

The reverse trap is quieter and was not in the earlier draft. **[v]** the checkpoint
also carries an `mtp` (multi-token-prediction) head whose submodules are *also* named
`self_attn.{q,k,v,o}_proj`, so the obvious language pattern `.*self_attn\.[qkvo]_proj`
trains **9** blocks where arm 3 declares 8 — and the 1.5%-matched 2-vs-3 contrast the
whole experiment rests on is gone, silently. Every pattern in `lora_arms.py` is anchored
at `^` and matched with `re.fullmatch`.

The arm table and the assertion live in **`code/train/lora_arms.py`**, which
`train_vision_lora.py` imports; it refuses to start unless
`print_trainable_parameters()` equals the arm's declared number.

## 3. Targets

### 3.1 The generated GT is now the label source

The GT pipeline was rebuilt and is much better suited to this than the hand-curated
`REPORT_GT` tables were. `code/ground_truth/parse_reports_to_gt.py` runs two stages — stage 1 (LLM,
2 calls per report) report text → `{case}[_{reader}]_report_facts.json`; stage 2
(deterministic, CPU, replayable with `--from-report-facts`) → `{case}[_{reader}]_gt.json`.

**[v] 1,274 training + 93 validate `_gt.json` on disk**, shape **identical to
`{case}_pred.json`**: `{case_id, global, teeth, _derivation}`, with `teeth` carrying all
32 blocks `tooth_11 … tooth_48`, each holding all six fact objects, speaking v7.1
vocabulary.

It is audited, which the old GT was not:
- `code/ground_truth/audit_report_facts.py` — three mechanical screens (`laterality`, `not-in-text`,
  `contradiction`), exits 1 on surviving ERRORs so a rebuild can be gated on it.
  `--fix-laterality` applied 73 errors in 37 training cases, 10 in 6 validate cases.
- `code/ground_truth/build_triage_sheet.py` — the ERRORs laid out for a human, with keep/drop/relabel
  decisions emitted as JSON and never written back.
- `08b8e4b` (consensus file answers to every reader) cut `not-in-text` ERRORs from 102 → 5
  on validate and 626 → 195 on training, and reverted 13 wrongly-drafted drops.

### 3.2 The loss mask comes from the GT's own nulls, not from a hand-built tier table

This replaces the v6.9 draft's tier M/R/X provenance table with the repo's own machinery.

`{case}_gt.json._derivation` records, per arch, `present / absent / unstated / conflicts /
unlocated`. **Supervise only positions the GT marks as stated; `-100` everything
`unstated`.** That is the same rule `structured_findings_evaluation.py` already scores by — *"N = positions
BOTH sides answered … A field the prediction left null is not scored — neither is one the
report never stated."* Loss and metric then share one definition of "the reader looked
here", which is the property the old plan had to construct by hand and got only
approximately.

**[v] `_derivation` does real work, but only at the tooth level, and that is not the
whole mask.** Across the 1,274 training GT files its arch lists hold 17,682 `present`,
4,687 `absent` and **18,399 `unstated`** positions — so masking on it drops ~45% of
tooth slots, for the right reason. What it cannot say is which *fields within a stated
tooth* the reader addressed, and the GT gave no signal there either: every field was
densely populated, so an absent 16 still carried `furcation_involvement: false`,
`bone_loss: "none"`, `periapical_lesion: false`.

Two rules settle it, and the second is now implemented **in the GT itself** rather than
in the collator (`51154c1`, `code/ground_truth/parse_reports_to_gt.py`):

1. **On a tooth that is there, the dense default is right and is supervised.** Reports
   state findings by exception, so a present tooth the report never calls carious is a
   tooth without caries. `with_caries: false` is a real answer, worth scoring and worth
   training on. This is where the ~92%-negative prior in §3.5 comes from, and it is the
   clinical prior, not an artefact.
2. **On a tooth the report says is gone, the same silence says nothing.**
   `with_caries: false` on an extracted 16 is a category error — there is no crown to be
   sound. Those fields are now `null` in the GT, which is this pipeline's established
   word for "the reference did not answer here": `structured_findings_evaluation.py` and
   `evaluate_predictions.py` both drop null-GT pairs, so **the loss mask and the metric
   read one definition instead of two** — which was the stated goal of this section and
   was not previously true.

The rule is per *value*, not per field: what the report **asserted** at an absent
position survives, what only **silence** produced becomes null. So a bone lesion named at
an edentulous site and a bridge pontic's crown both stay `true`. `bone_loss` is the one
carve-out — its value comes from an arch-level assertion rather than from silence, and
"moderate generalized bone loss" is a claim about the dentition, not about the ridge
where 46 used to be, so it is null at an absent position either way.

**[v]** Effect, both splits re-expanded with `--from-report-facts --consensus`:

| | validate | training |
|---|---:|---:|
| absent tooth positions | 409 | 4,687 |
| fabricated negatives removed | 4,670 | **53,586** |
| findings kept at an edentulous site | 12 | 281 |
| changes on a **present** tooth | **0** | **0** |

Those 53,586 were, under the old GT, supervised negatives on positions with no referent
— roughly 11 per absent tooth. Training on them is a direct route to the "emit `false`"
degenerate solution §3.5 caps against, so removing them matters more to arm 2 than to
any metric.

#### The refusal list was derived from the wrong statistic (corrected 2026-08-13)

Every refusal below used to rest on a low **inter-reader Jaccard**. That is a coverage
number, not a contradiction number, and reading it as evidence of a bad label was a
mistake.

Radiology reports state findings by exception. Reader A writes *"post and core on 16"*;
reader B writes about the sinus and never mentions 16. Their pairwise Jaccard on that
field is 0 — **and nothing is in dispute**. Silence is not denial, which is the rule this
project already applies twice elsewhere: `parse_reports_to_gt.py` nulls an absent tooth's
fields rather than defaulting them (§3.2 above), and `verify_report_facts.combine()`
scores *one claims, one silent* as positive.

**So the union is the label** — a tooth is positive if any reader claims it. The GT
already does this, and it was verified rather than assumed: **[v]** over 341 multi-reader
training cases the consensus file equals the union of its readers on **341/341** for
`with_post_and_core` (43 teeth) and `with_endo` (418), via `_union_ints()` in
`consensus_report_facts()`.

Re-derived on **positive counts** in the 120-case wide pool, which is what a rare-label
problem actually turns on:

| field | positives | cases with ≥1 | neg:pos | |
|---|---:|---:|---:|---|
| `with_fillings` | 218 | 48 | 9:1 | supervised |
| `with_endo` | 185 | 58 | 11:1 | supervised |
| `with_full_crown` | 120 | 43 | 17:1 | supervised |
| `with_caries` | 54 | 27 | 38:1 | **supervised — was refused on 13.5%** |
| `periapical_lesion` | 42 | — | 50:1 | **supervised — was refused on 8.9%** |
| `is_remnant` | 33 | 18 | 64:1 | supervised |
| `with_post_and_core` | 21 | **8** | 101:1 | **supervised by decision** |
| `with_root_fracture` | 4 | **4** | 532:1 | refused — on count |
| `furcation_involvement` | **0** | 0 | — | refused — on count |

What survives as a refusal survives on **count**, an argument the union reading does not
touch:

- **`with_root_fracture` — 4 positives in 4 cases.** The held-out split works in whole
  cases, so one case moves the entire field. Nothing to fit, nothing to measure.
- **`furcation_involvement` — 0 positives in 1,785 slots.** Every example says `false`,
  so the only thing it can teach is "emit `false`" — R1's operating-point shift produced
  on purpose. Never an agreement question at all.

**`with_post_and_core` is supervised**, and the caveat that survives is also a count one,
not a quality one: its 21 positives sit in **8 cases**, and cases are the unit the
held-out split works in, so its variance is large for reasons that have nothing to do
with the readers. The AWQ baseline scores it 0.05/0.06 `PRED` and 0.03/0.18 `SUMMARY` on
N=17 — the worst findings axis in that arm, so there is room, but **score it separately
and keep it out of any aggregate a conclusion about H1 rests on**.
- **`filling_quality` is conditional by construction** — gated on
  `morphology.with_endo == true`. Mask it wherever the gate is false, or the model learns
  to answer a question it was told not to ask.

Trainable fields, in order of label quality: `eruption_state` (mask-corroborated presence
90.0% report-vs-facts), `with_endo` (inter-reader 62.4%, the best clinical channel),
`bone_loss`, `location` (canal, 6 FDIs), `with_full_crown` (20.5%, weak),
`with_fillings` (28.4%, weak), plus `with_caries`, `periapical_lesion` and
`with_post_and_core` — the three the corrected refusal list restores. Those inter-reader
percentages are coverage, not agreement, and are kept here only because they are what
earlier drafts cited; **the number that decides whether a field is learnable is its
positive count**, tabled above.

**[v] Two things the A3 audit found that this list did not predict**, both from
`code/train/build_sft_targets.py` over the 120-case wide pool (1,850 usable calls):

- **`furcation_involvement` has no positives at all — 0 true against 1,785 false.** It
  is not in any refusal list above because those were built from inter-reader agreement,
  and a field with zero positives has no agreement to measure. Supervising a boolean
  whose every example is `false` teaches "emit `false`", which is R1's operating-point
  shift produced deliberately — arm 0 wearing a costume. **Refuse it**, or the §3.5 cap
  has to do work it was not designed for.
- **Five more fields sit past §3.5's 6:1 cap**, which nothing yet enforces:
  `is_remnant` 73:1, `bone_quality.present` 35:1, `with_full_crown` 16:1, `with_endo`
  9:1, `with_fillings` 7.7:1. Only `mandible_canal.adjacent_to_teeth` (3.4:1) is inside
  it. The cap is a real intervention — it drops negative-only calls and shrinks the pool
  — so it is named here as outstanding rather than silently applied.

**[v] The `unstated` mask never fires, and that is correct.** `parse_reports_to_gt.py`
emits no tooth block at all for a position the report never placed, so those calls are
dropped as `no_gt_block` (1,018 of them) before any mask is consulted. Same outcome as
`-100`, different accounting than this section describes. The `gated` reason is likewise
0 — `filling_quality` is already null wherever `with_endo` is false, so the GT is
self-consistent and the gate is redundant with the null rule rather than additional to
it.

Two v7.1 GT corrections to inherit rather than rediscover: **a summary adjective no longer
empties an arch a named tooth stands in** (`3a03ca0`; validate fully-edentulous arches
9 → 3), and **a root remnant is a tooth that is still there** (`82db0f5`; 18 remnant teeth
on validate, 238 on training now read present). Both change `eruption_state` targets
materially.

### 3.2b Resolution is **not** the constraint — measured, and it is a null **[v]**

Before booking arm 2, the cheapest competing explanation for H1 was tested directly: is
the feature absent from what the model can resolve, or present and under-sampled? Those
have different fixes — one is training, the other is `create_tooth_detail.py` — and no
LoRA substitutes for pixels that were never rendered.

`code/zoom_probe.py` + `jobs/zoom_probe.sh`, job 554815, 16 min on one H100. Same teeth,
same prompt, same call builder, same server; **only the image differs**. `base` is what
the pipeline sends (`window_scale 1.25`, `out_w 1344`, ~1,722 vision tokens for the 3×3
grid). `zoom` is a tighter crop at twice the pixels (`1.0` / `2688`), putting the same
anatomy on ~4× the patches. 28 teeth: 20 `post_and_core` positives and **8 root-treated
controls without one**, from the same cases.

| condition | n | TP | FN | FP | TN | recall | precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 28 | 6 | 14 | 1 | 7 | 0.30 | 0.86 |
| zoom | 28 | 9 | 11 | 3 | 5 | **0.45** | **0.75** |

| base | zoom | teeth | |
|---|---|---:|---|
| wrong | right | 7 | zoom fixed it |
| right | right | 7 | both right |
| wrong | wrong | **8** | both wrong |
| right | **wrong** | **6** | **zoom broke it** |

**7 fixed against 6 broken — exact McNemar p = 1.000.** There is no effect. What moved
instead is the **assertion rate: 25% → 43%**. At 4× the patches the model says `true`
more often, which raises recall and drops precision, and the two flip counts cancel.

**That is R1 reproduced in miniature, on the intervention nobody suspected of it.** The
extra pixels did not make the feature legible; they made the model more willing to claim
it. Three consequences:

- **Re-rendering will not fix `post_and_core`**, and arm 2 should not be expected to move
  it. The 8 teeth wrong under both conditions stay wrong.
- **H1-as-resolution is closed** for this field, which leaves "not recoverable from these
  renders at all" and H2 (decision threshold) live. The assertion-rate shift looks like
  the latter, so **arm 3 is a real control here, not a formality** (§2).
- **The 8 controls are what made this readable.** Without them the same run reads as
  "recall 0.30 → 0.45, resolution confirmed" — which is precisely the mistake §6's R1
  exists to prevent, and it would have cost a re-render of the pool plus an arm.

Two limits worth carrying: n = 28 on one field and one model, so this closes *resolution
for post_and_core*, not resolution generally; and base precision reads 0.86 here against
0.03 in §5.0 only because this set is enriched 20:8, so the two are not comparable.
`zoom_probe.py` is kept rather than deleted — the same rig answers the same question for
`crown`, `fillings` and `caries`, which are also weak in §5.0.

### 3.3 `visual_evidence` — written by a teacher, the same way the few-shot arm did it

**[v]** every `visual_evidence` in the generated GT is `""`, while `schema.json`'s rule
requires it to lead every block and every other field to be consistent with it. Something
has to write it, and an earlier draft of this section proposed using the baseline model's
own prose. **That is superseded**: the string is drafted by a **teacher model given the
image and the already-established answer**, exactly as
`build_fewshot_exemplars.py --draft-visual-evidence` already does it.

**The property that makes this safe is that the label does not come from the teacher.**
Every decision field is settled from the report and `facts.structured` before the teacher
is called; the teacher is shown the finished answer and told, in `DRAFT_SYSTEM`, that the
findings *"are ALREADY ESTABLISHED … you must not re-judge them, soften them, or
contradict them"*. It writes only what in this image supports them. Letting it diagnose
instead would turn a wrong answer into confidently-worded evidence, which is worse than
no evidence at all.

`build_sft_targets.py` **imports** `DRAFT_SYSTEM` / `DRAFT_USER_TEMPLATE` from
`build_fewshot_exemplars.py` rather than restating them, for the same reason §3.4 imports
the prompt builders: two copies of a prompt drift, and the drift is invisible.

**The risk this carries is real and was measured, so screening is mandatory, not
optional.** A stronger teacher cites cues the student cannot resolve — faint periapical
lucency, subtle trabecular texture — and evidence built on an invisible cue teaches the
student to assert what it cannot verify, which *is* the over-calling under study.
`STAGE=perceive` measured only **~40% cue confirmation on the four tooth-class exemplar
files against ~85% on sinus**: tooth prose is exactly the prose that could not be earned.
Two mitigations, both already built:

- `DRAFT_SYSTEM` orders the teacher to write for **"a reader with WEAKER eyes than
  yours"** — every named feature LOCATABLE, COARSE and RELATIONAL, no fine texture, no
  hedging.
- `--check-perceivable` asks the **student** whether it can see each cue the prose names.
  At exemplar scale a human then rewrote the failures; at 1,850 rows that does not scale,
  so the check runs on every row and a row that fails is **regenerated at a coarser level
  or has its evidence dropped and counted** — never silently kept.

**Both open items are now settled, and the teacher pass has run. [v]**

**The teacher is `Qwen3.5-27B`** (dense, 52 GB, one H100), chosen by audition rather
than by size: `code/pipeline/infer/pool_infer.sh` ran it and the student over the *same* 120-case
payload, the full tooth call, scored by `structured_findings_evaluation.py`. It beat the student on every
axis with signal — `fillings` 0.17/0.06 → 0.38/0.29, `restoration` 0.24/0.33 →
0.36/0.50, `endodontic` 0.32/0.42 → 0.44/0.48, `post_and_core` 0.05/0.17 → 0.09/0.29.
**Precision and recall rose together**, which is not what an operating-point shift looks
like (§3.2b is the contrast): capacity bought discrimination. Cross-family candidates
(Lingshu-32B, MedGemma-27B) were not auditioned — the 27B cleared the bar, is
same-family, and every alternative adds schema-compliance and overlay-blindness risk on
top.

**Detection and description are different jobs**, and the audition bounds only the
first. A model that cannot reliably *find* a post and core can still write what one
looks like, because that is knowledge of the feature class rather than a reading of this
picture, and the label never comes from the teacher. So `post_and_core` is drafted with
everything else rather than withheld.

**The screen ran, and it is much healthier than the exemplar-scale number predicted.**
`check_evidence_perceivable.py`, 9,114 strings put to the **student**: **67% of claimed
features confirmed, 69% of strings kept**, against the ~40% `STAGE=perceive` measured on
four tooth-class exemplar files. `DRAFT_SYSTEM`'s *"write for a reader with WEAKER eyes
than yours"* is doing real work at scale.

| fact | confirmed | rows kept |
|---|---:|---:|
| endodontic_treatment | 0.83 | 86% |
| mandible_canal | 0.83 | 84% |
| bone_quality | 0.80 | 82% |
| morphology | 0.72 | 76% |
| **eruption** | **0.48** | 51% |
| **periodontal_status** | **0.46** | 47% |

**The failures do not track finding-difficulty.** The prediction was that
`post_and_core` would be the invention hotspot; it is not. `morphology` — which carries
`post_and_core`, `caries`, `crown`, `fillings`, `endo` in one string — is 0.72, *better*
than `eruption` and `periodontal_status`. Those two are where the prose is a spatial
**relation** between structures ("level with the occlusal surfaces of 34 and 36", bone
height against root length) rather than the appearance of one thing. A bright object in
a canal is confirmable; a relative height read across three tiles is not. Worth
carrying: it is relational description, not rare findings, that the student cannot
verify.

*Limitation of the design as built:* evidence is one string **per fact**, not per field,
so `post_and_core` cannot be separated from the rest of `morphology`. The per-field
comparison this section pre-registered is not answerable; 0.72 is the finest grain
available.

**The loss weight is 0.04, and the number is arithmetic. [v]** `visual_evidence` is
supervised **per row** — where the screen kept the prose — and masked where it did not
and wherever the string is empty. Failed prose *stays in the target*, so the decision
fields are conditioned on it exactly as at inference; it simply carries no gradient.

Why not zero, and why not one: surviving evidence is ~3.4 strings × ~68 tokens ≈ **230
prose tokens per row**, against ~10 tokens of supervised decision content — about
**23:1**. At full weight ~96% of the gradient teaches radiology prose, which is a
*language* task, which hands **arm 3** an advantage that has nothing to do with H1b and
corrupts the one contrast the experiment exists for. Zero leaves the train/inference
mismatch open, since `visual_evidence` leads every block and every decision token is
conditioned on it. 0.04 makes the two contribute comparable loss mass.

**The teacher's first pass was thrown away, and the reason is worth keeping.**
`qa_pairs.jsonl` stores image paths *relative to the project dir* so one payload
resolves on the host and at `/project` in the container; `build_captioned_image_blocks()`
resolves them against the **working directory** and silently drops what it cannot find.
The job set no working directory, so the teacher wrote **9,301 fluent, specific-sounding
strings having never seen an image** — memory dressed as observation, manufactured by
the harness rather than the model. It surfaced only because the screen refuses to judge
perceivability with no image and therefore made *zero* requests — and then reported
"9301 string(s) kept", because an unjudged string counted as neither pass nor fail.
Both are fixed (`--base-dir`; `NoImage`; unjudged now means **drop**; the screen refuses
to write a filtered dir above a 5% failure rate). The lesson generalises: **a silent
degradation in the harness is indistinguishable from a model result**, and the only
defence is a component that refuses to proceed rather than one that copes.

**It remains the overlay-mention channel.** Per `6e3f323`, whether the answer cites the
red outline is the cheapest acceptance test for perceptual health (§5.1 item 4). With
teacher prose that diagnostic still works, but it now measures whether the student
*learned* to cite the overlay rather than whether it spontaneously does — so the baseline's
rate has to be recorded before any arm trains, or the comparison loses its zero point.

### 3.4 Prompt parity

`build_sft_targets.py` renders prompts through **imported**
`run_vqa_inference.build_call_prompt()` + `build_captioned_image_blocks()` +
`TOOTH_USER_TEMPLATE` / `TOOTH_SYSTEM` — never reimplemented — then through the model's
`chat_template.jinja`.

**The acceptance test has run, it failed the first time, and what it caught is worth
more than the check. [v]** `code/train/check_prompt_parity.py` + `code/train/prompt_parity.sh`, job
555208 then 555214, ~90 s of a30 each: render a call through the training path and
through `POST /tokenize` on a live server, require **identical `input_ids`**. A token-id
diff, not a text diff — `enable_thinking: False`, `add_generation_prompt` and the
caption/image interleave order all render as prompts that look right.

**The interleave was wrong, and it was wrong at serving time, not in the collator.**
Host 6,126 tokens against 6,128 served, diverging at the first token of the user turn:
the host opens with `Image 1 (tooth_46_composite): …`, the server opens with the image.
vLLM resolves a chat-template **content format** per server by inspecting the template's
Jinja AST, guesses `string` for Qwen3.5's, and in that mode
`_get_full_multimodal_text_prompt` does not render the content parts in order — it
**hoists every image placeholder to the front of the turn** and joins the text parts
after it with `\n`. The two extra tokens are those newlines, which appear in no message
anywhere.

So the MedThinkVQA-style caption-before-image design — the reason
`build_captioned_image_blocks` exists, *"so the model knows what it's looking at before
it sees the pixels"* — **has never reached the model, in any run in this repo**, the
§5.0 baseline included. It also undercuts §1's prefill note further than that note
assumed: with the image hoisted to position one, nothing after the system turn was ever
a common prefix, so `--enable-prefix-caching` had even less to hit.

**Decision (2026-08-14): the collator matches the server, not the intent.**
`sft_prompt.CONTENT_FORMAT = "string"` reproduces the hoist exactly, and the constant
carries the reasoning at the point someone would change it. Training against the prompt
the model actually receives costs nothing and keeps every scored arm comparable;
`--chat-template-content-format openai` would restore the intended order but changes the
prompt every run in this repo has been sent, so it invalidates §5.0 and everything scored
against it. Making the caption-first design real is a separate measured experiment, under
§1's rule for the prefill reorder — and it is now a *cheap* one, since the check that
would confirm it is written.

**[v] Parity holds on both call shapes**, one server boot: tooth 46 (lower molar, carries
`mandible_canal`, 16 supervised fields) **6,128 = 6,128**, tooth 12 (upper, no canal fact,
11 supervised) **5,223 = 5,223**, with the 1,764-token image expansion and the call's
`json_schema` confirmed alongside. The real tokenizer against §4.4's ~6,200 arithmetic
is close enough that `MAX_CONCURRENCY=auto`'s ÷6,200 divisor stands.

One number worth carrying out of the rig: **the a30 loads `Qwen3.5-9B-AWQ` in ~80 s**,
against the ~10 min the submission budget assumes. Different filesystem, so it is an A/B
anchor for §7's open A10G timing question rather than an answer to it.

### 3.5 Sample counts

One sample = one tooth call. Pool from `outputs/gt_quality_training/case_ranking_*.csv`
filtered by `audit_report_facts.py` (exit-1 gate) — **[v]** 582 rows, `score_asis ≥ 0.80`
= 230 cases.

| Set | Cases | Tooth calls (≈19.5/case) |
|---|---:|---:|
| **Narrow** (audit-clean, `≥0.90`) | ~45 | **~880** |
| **Wide** (audit-clean, `≥0.80`) | ~180 | **~3,500** |
| Held-out (**whole cases**, never teeth within a case) | 25 | ~490 |

Arm 1 trains on narrow; arms 2 and 3 on wide. **Every one of those cases is a `training`
case** — the held-out 25 included. Nothing here is drawn from validate, and
`build_sft_targets.py` refuses a validate case id rather than relying on this table
being read correctly (§0).

**Do not rebalance toward positives** — the natural prior is the clinical prior, and
inflating positives re-creates the over-calling being removed. But cap negatives at a
**6:1** per-field ratio, because a ~92%-negative boolean under token CE in a
248,320-vocab will happily learn "emit `false`" — which is arm 0 wearing a costume.

**[v] Implemented (`28669b7`) by MASKING negatives, not by dropping calls**, and the
difference matters. This section used to say "cap negative-only *calls*"; that cannot
produce a per-field ratio and does damage on the way. The ratios differ wildly per field
— `is_remnant` 73:1 against `with_fillings` 7.7:1 — while a call carries every field at
once, so dropping enough calls to fix one deletes another's positives with them. And
dropping the *all-negative* calls specifically is worse than useless: a tooth with no
findings is exactly what the model must learn to call normal, and removing those **is**
rebalancing toward positives, which this same paragraph forbids.

Masked instead, per field, down to `cap × positives`: every call stays, every positive
stays, the decoded shape is untouched, only the loss changes. Seeded (`--seed 42`)
because which negatives are masked must be **identical across arms** — arm 2 and arm 3
have to see the same supervision or the contrast is not the contrast. Enums are left
alone; the rule is stated for booleans and "negative" is undefined for `eruption_state`.
A field with zero positives has all of its negatives masked and says so, which is a
second net under `furcation_involvement` independent of the refusal list.

Every supervised boolean now sits at exactly 6.0:1 (`adjacent_to_teeth` was already
3.4:1 and untouched).

## 4. Environments — two of them, one toolchain

Still two environments, because training needs autograd and serving needs vLLM and no
single install gives both. But they are no longer allowed to disagree about CUDA: a
conda env on **cu128** for training and merging, a **cu128** container for serving, and
`Qwen3.5-9B-AWQ` as the only checkpoint either of them loads. The earlier draft split
them across cu130 and cu128 on the argument that training never runs on Grand Challenge;
that is true and beside the point. When an arm behaves differently after merging, the
question "is this the adapter or the toolchain?" should have one obvious answer.

### 4.1 Training: `cbct_sft_cu128` — built, not cloned **[v]**

**[v] transformers 4.x has no `qwen3_5`.** `transformers/models/qwen3_5/` existed in
exactly one env on the account: `~/miniconda3/envs/chandra-ocr_hf` — torch 2.12.0+cu130,
transformers 5.9.0, accelerate 1.13.0. `olmocr` (4.57.3) and `tmd_ocr_cluster` (4.56.1)
have `qwen3_vl` but not `qwen3_5`. **[v] `peft`, `trl`, `datasets`, `bitsandbytes`,
`flash_attn`, `autoawq` were installed in none of the seven envs.**

An earlier draft of this section argued for cloning that cu130 env, on the grounds that
training never runs on Grand Challenge so its driver does not constrain it. **That is
overruled: this experiment uses the cu128 / vLLM 0.19.0 submission environment and
`Qwen3.5-9B-AWQ`, and nothing else.** The reasoning behind the clone was not wrong, it
was beside the point — the value of one toolchain across training, merging and serving
is that a numerical surprise has one place to come from, and that is worth more than a
newer CUDA on a box whose driver (580) runs both.

So the env is **built from scratch on cu128**, with the cu130 clone kept only as a
version reference (`env/sft_reference_cu130.txt`, a `pip freeze` of it):

```bash
conda create -n cbct_sft_cu128 python=3.12
conda activate cbct_sft_cu128
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.9.0 accelerate==1.13.0 peft==0.20.0 datasets==5.0.1 \
            flash-linear-attention==0.5.2 pillow
pip freeze > env/sft_requirements.txt
```

**[v]** installed and verified: **torch 2.11.0+cu128** (`torch.version.cuda == 12.8`),
transformers 5.9.0, peft 0.20.0, datasets 5.0.1, accelerate 1.13.0, fla 0.5.2. 2.11.0 is
the newest cu128 wheel PyTorch publishes.

Conda, not a container: **[v]** there is no `docker`/`podman`/`enroot`/`unsquashfs` on the
login node, so no `.sqsh` can be built or even inspected here; and the project convention
is already "host env for everything that is not the vLLM server".

**The blocking unknown is settled. [v]** `peft` 0.20.0 works against transformers 5.9.0,
and every arm regex hits: loaded on the **meta device** (`init_empty_weights` +
`AutoModelForImageTextToText.from_config`, seconds, no GPU, no shard read),
`code/train/lora_arms.py` reports 286,720 / 7,986,688 / 7,864,320 / 31,946,752 against the
declared counts, on both the bf16 and the AWQ checkpoint. That check is retained as the
gate `train_vision_lora.py` runs before touching a GPU, not thrown away as scaffolding.

### 4.2 Serving: vLLM 0.19.0 / cu128, unchanged

`~/containers/vllm019_cu128.sqsh` (**[v]** 18.9 GiB, built Aug 12) — `nvidia/cuda:12.8.1-devel`
+ `pip install vllm==0.19.0`, because the official `vllm/vllm-openai` images v0.13–v0.19
all ship CUDA 12.9 and declare `cuda>=12.9`, which driver 570 refuses.

**transformers 4.x is sufficient at serve time because vLLM ships its own implementation
and its own config class** — `registry.py:560` maps `Qwen3_5ForConditionalGeneration`, and
`transformers_utils/config.py:132` registers `qwen3_5="Qwen3_5Config"`. So the training env
and the serving env never have to agree on a transformers version. They only have to agree
on the *weights*.

Every job still defaults to `extraction.sqsh` and `QWEN_MODEL_NAME=Qwen3.5-9B`; the new
container is used by env override, as `docs/workflow.md` documents.

### 4.3 The quantization gap, and how to close it

**[v] `Qwen3.5-9B-AWQ` quantizes almost nothing.**
`modules_to_not_convert: ["visual","linear_attn","self_attn","model.layers.0.","mtp"]` —
only `model.language_model.layers.{1..31}.mlp.{gate,up,down}_proj` are 4-bit (279
quantized tensors of 961; 333 visual + 333 language bf16 + 16 other).

**[v] The vision tower and every `self_attn` are bit-identical to the bf16 checkpoint** —
`model.visual.blocks.0.attn.qkv.weight` hashes to the same sha256 in both, and
`layers.3.self_attn.q_proj.weight` compares equal elementwise.

Three consequences:

1. **A vision LoRA merges into the AWQ checkpoint exactly.** The merge target is
   untouched by quantization, so there is no numerical question about the merge itself.
   The trainable artifact and the deployable artifact become the same model — which was a
   flagged unknown in the v6.9 draft and is now a settled fact.
2. **Whether you can train against AWQ directly has a specific answer, and it is not
   "no". [v]** `transformers/quantizers/quantizer_awq.py` gates it explicitly:

   ```python
   @property
   def is_trainable(self):
       return version.parse(importlib.metadata.version("gptqmodel")) >= version.parse("5.0.0")
   ```

   So the blocker is a missing dependency, not a missing derivative — `gptqmodel>=5.0.0`
   supplies backward-capable kernels and transformers then marks the model trainable.
   That matters because **none of the three arms touches a quantized weight**: `visual`
   and `self_attn` are both in `modules_to_not_convert`, so the LoRA sits on bf16
   `nn.Linear` throughout. The only question is whether gradients can *traverse* the
   4-bit language MLPs on the way back to the vision tower.

   That was the stage-B plan: install `gptqmodel`, load AWQ, run one backward. It was
   attempted on 2026-08-14 and **abandoned — see below.**
3. ~~Two fallbacks if that backward does not work.~~ **Superseded 2026-08-14: training
   is bf16, deployment is AWQ, and the transfer gate is live.** See §4.3b.

### 4.3b [v] Why training does not load AWQ at all (2026-08-14)

`gptqmodel` was installed, and the AWQ checkpoint did load and did report
`is_trainable=True` with a trainable kernel. It is still unusable here, for a reason
that has nothing to do with quantization arithmetic:

> **`gptqmodel` monkey-patches Triton's autotuner globally at import**
> (`TritonPatch.apply()` in `gptqmodel/__init__.py`, no opt-out), and Qwen3.5's
> GatedDeltaNet layers run on `fla`'s Triton kernels, whose `CachedAutotuner` subclass
> then dies on a `_cache_lock` that only gptqmodel's patched `__init__` sets. Every
> forward pass hits it.

It is not enough to stop loading an AWQ checkpoint: **peft imports gptqmodel during
`get_peft_model`** — `dispatch_awq()` calls `is_gptqmodel_awq_layer()` per target module
— so merely having it *installed* broke a bf16 run too (job 555440). It is uninstalled,
and `env/sft_requirements.txt` records it as a do-not-install, because "the model is
AWQ, install gptqmodel to load it" is the obvious move that walks straight back in.

Three lesser traps found on the way, all recorded in commit messages `824a923`,
`270e5a7`, `f71fe00`, and worth knowing before anyone retries this:

- `modules_to_not_convert` means **substring** to vLLM and **prefix-or-suffix** to
  transformers. The shipped list is written for vLLM, so under transformers *none* of
  its five patterns fire and it tries to 4-bit all 287 linears. It surfaces as Marlin
  complaining `out_features 4304 must be divisible by 64` — 4304 being the *vision*
  `intermediate_size`, a tower that was never meant to be converted, which reads as a
  kernel problem and invites a useless backend override.
- Kernel selection is **per device**. On the A100 `auto` chose `AwqMarlinLinear`, whose
  `SUPPORTS_TRAINING` is `False`. It failed loudly only because Marlin JIT-compiles CUDA
  C++ and there is no nvcc; a prebuilt Marlin would have run and returned **no
  gradient**, which is exactly R3's failure signature.
- `gptqmodel 7.3.2` cannot be imported next to transformers 5.9.0 (`solar_open2.py`
  wants a `masking_utils` symbol 5.9.0 lacks, eagerly, for an unused architecture).
  7.3.1 predates the file.

**So: train the adapter on `models/Qwen3.5-9B` (bf16), merge into `Qwen3.5-9B-AWQ`.**
This is safe for the *merge* and unmeasured for the *forward*:

- **[v] The merge is exact.** All **142** tensors the three arms target are
  bit-identical between the two checkpoints — merger 2, vision blocks 108, lm self_attn
  32; zero missing, zero differing, compared elementwise rather than by hash. AWQ
  quantizes only `mlp.{down,up,gate}_proj` in layers 1–31 (93 modules, 279 tensors) and
  copies everything else verbatim.
- **[?] The forward is not.** Training runs bf16 MLPs where deployment runs 4-bit ones.
  The **transfer gate is therefore mandatory, not a formality**: run the merged arm-2
  checkpoint on bf16 and on AWQ over the same cases and compare. Given `6e3f323`,
  quantization can change this model's behaviour materially — do not assume transfer.

A side effect worth keeping: dropping AWQ also drops transformers' forced downcast
(`bfloat16 is not supported for AWQ CUDA kernels yet. Casting to float16`), so the run
is genuinely bf16 as §4.4 assumes.

**AWQ passes the overlay test** where int4 AutoRound fails it. **[v]**
`logs/aksssr_v6_554172.log:291`, `A008/tooth_11`: *"The red outline in all axial, coronal,
and sagittal views clearly delineates the tooth structure…"*, describing a normal incisor.
AWQ also fixes both of int4's packaging quirks — it ships `preprocessor_config.json` and a
plain `Qwen2Tokenizer` rather than the transformers-5 `TokenizersBackend`. The earlier
gap here — that run was cancelled mid-first-case, leaving no scored AWQ result — is
**closed as of 2026-08-13 [v]**: the validate-40 arm completed (40/40, `MAX_CONCURRENCY=40`,
17 min 19 s) and a bf16 arm on byte-identical images puts the quantisation cost at nothing
measurable (mean F1 0.572 AWQ vs 0.552 bf16 on `SUMMARY`, 8 axis wins each). `CLAUDE.md`
now names `Qwen3.5-9B-AWQ` as the submission model, matching what these arms ran.

### 4.4 Memory, 1×A100, one tooth call

**[v]** 1,764 vision tokens (1343×1356, patch 16, merge 2) + ~3,500 text tokens
(`questions` 4,224 chars + `guidance` + `output_schema` 3,063 + `TOOTH_SYSTEM` +
`HOW_TO_READ_A_TOOTH` + caption) + 400–900 target ≈ **6,200 tokens**. Uniform across every
sample — no length bucketing needed, which is one more thing the tooth-only scope buys.

| Term | Size |
|---|---|
| frozen base bf16 | **19.31 GB** (9.653 B params) |
| LoRA + grads + AdamW + fp32 master | ~0.13 GB |
| **logits, naive** | 3.08 GB fwd, **~9 GB** peak with fp32 CE + grad — 6,200 × **248,320**. *This is what OOMs the job, not the weights.* |
| **logits with `logits_to_keep`** | **~0.9 GB** — slice hidden states to the ~600 non-`-100` positions *before* `lm_head`. Free and mandatory. |
| activations, both towers grad-checkpointed | ~2.7 GB |
| GatedDeltaNet fp32 scan | **2–5 GB** (~0.5 GB with `flash-linear-attention`) |
| **total** | **≈ 25–28 GB — fits a 40 GB card** |

`modeling_qwen3_5.py:217` sets `is_fast_path_available` from `causal_conv1d` **and**
`chunk_gated_delta_rule` together, and backprop through 24 pure-PyTorch chunked
linear-attention layers is the least predictable term in the budget.

**[v] Only one of the two installed, and it is the one that matters.**
`flash-linear-attention` 0.5.2 is in; `causal_conv1d` will not build here — its
`setup.py` probes `nvcc`, the login node has no CUDA toolkit, and it dies on
`NameError: bare_metal_version`. That is less damaging than the paragraph above implies,
because line 422 binds the kernels *independently* of the combined flag:

```python
self.causal_conv1d_fn      = causal_conv1d_fn                                   # -> None
self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule
```

So the expensive chunked scan already runs on the `fla` kernel and only the depthwise
convolution — the cheap term — falls back to torch. Chasing `causal_conv1d` means
installing a CUDA toolkit to speed up a conv; leave it unless stage B's measured step
time says otherwise. Fallback if the memory budget is still missed:
`--qos=a100-sxm4-80gb`.

Settings: gradient checkpointing on both towers; `batch_size=1`, `grad_accum=16`;
`max_length=8192` with any longer call **dropped and counted, never truncated** (a
truncated call loses its `output_schema` and teaches a wrong shape); LR 1e-4, cosine, 3%
warmup, `lora_alpha=2r`, dropout 0.05, bf16; **arm 3 uses the identical schedule**. Load
with `dtype=bfloat16, low_cpu_mem_usage=True, device_map={"":0}`.

Wide set 3,500 × 2 epochs / 16 = **438 optimizer steps**; at 25–40 s/step (arithmetic, not
measurement — stage B measures it) that is **3–5 h**. One 8 h job per arm.

**[v] Measured at stage B (job 555445, 1×a100-sxm4-80gb):** 78 optimizer steps in
2,808 s = **36.0 s/step**, inside the predicted 25–40 s and near its top. Stage D
therefore lands at 438 × 36 s ≈ **4.4 h**, inside one 8 h job. Measured on an SXM4 80 GB
card, whose memory bandwidth is roughly 2× the PCIe 40 GB card §4.5 books — treat 36 s
as an **optimistic bound** for a 40 GB run, not a transferable number.

**[?] The memory budget above is wrong, and by more than a rounding.** The stage-B probe
reported **peak 60.28 GiB** against the 25–28 GB predicted here — with `batch_size=1`,
one row, gradient checkpointing on (`Qwen3_5VisionBlock` subclasses
`GradientCheckpointingLayer`, so the vision tower *is* covered) and no optimizer states
yet. The training loop's own peak was never printed, so the number that matters for
stage D is still unmeasured; the smoke run survived only because it was on an 80 GB
card. **A 40 GB a100 would not have survived the probe.** Add a peak-memory print to the
training path before booking stage D's card.

### 4.5 Resource ask — **needs sign-off before any `sbatch`**

**[v]** `sacctmgr` grants QOS `a100`, `a100-sxm4-80gb`, `a100_long`, `h100`, `b200`;
`sinfo` shows 8×A100-80GB, 4×H100, 8×B200 nodes with **1.9 TB node RAM**. The `--mem=32G`
and 40 GB conventions in `jobs/` are habit, not limits.

| Stage | Partition / QOS | Ask |
|---|---|---|
| render, targets, arm 0, scoring | `--partition=cpu --qos=cpu` | 8 cpus, 32G, 2 h. **Not the login node** — the per-user cgroup caps at 8 G and rendering is OOM-killed there. |
| overfit sanity | `--qos=a100 --gres=gpu:a100:1` | 8 cpus, 64G, 2 h |
| arm 1 train | same | 6 h |
| arms 2, 3 train | same | 8 h each |
| DeltaNet fallback | `--qos=a100-sxm4-80gb` | as above |
| inference per arm | `--qos=a100 --gres=gpu:a100:1` | 8 cpus, 32G, 8 h |
| RadFact judge | `code/eval/judge_server.sh` | unchanged |

**Do not go multi-GPU.** The entitlement exists; it buys nothing at batch 1 / 8 M
trainable params and adds a class of failure to an experiment whose entire value is a
clean 2-vs-3 contrast.

**[v] Stage B runs on the a100 (40 GB), and the queue is the reason** (`611fec3`). The
h100's 80 GB would take §4.4's memory question off stage B's list entirely — 25–28 GB
budgeted against a 40 GB card with the GatedDeltaNet backward as the least predictable
term, and an OOM there costs a whole allocation to learn what a bigger card measures for
free. What overruled it on 2026-08-14 was arithmetic of a different kind: **all four
h100s were held by jobs with 22 h to 2 days left, estimated start ~46 h out, against ~2 h
for an a100**. A two-hour job is not worth a two-day wait to de-risk a term the run
reports anyway.

So the memory question comes *back* onto stage B's list deliberately, and the run answers
it either way — `peak GPU memory` prints on success, and an OOM is itself the measurement
that §4.4's arithmetic was optimistic. The fallback is one flag:
`--qos=a100-sxm4-80gb --gres=gpu:a100-sxm4-80gb:1`. The step time it measures is now
honest for stage D's a100 ask (open item 5) rather than optimistic on a card stage D will
not get — which is the one thing the queue bought.

## 5. Evaluation

Train on `training`, score on `validate` (40 cases, 93 generated `_gt.json`). Per §0,
validate is read **once per arm, at the end**. Every mid-experiment number — the stage-C
gate, ΔAUROC, whether arm 4 is worth running — comes off the 25 held-out *training*
cases. An arm whose hyperparameters were chosen by looking at validate has no held-out
set left, and the paired CI in §5.1 would be measuring the choosing, not the arm.

### 5.0 The baseline, printed **[v]**

`outputs/aksssr_v7_validate/`, AWQ arm, 40/40 cases, surveyed against the generated GT
(`survey/structured_findings_evaluation_awq_20260813.txt`). This is the row every arm is a delta from, so
it is written down rather than cited — an arm that "improved endodontic" is meaningless
against a number nobody can see.

| axis | N | PRED prec/rec | SUMMARY prec/rec |
|---|---:|---|---|
| absent teeth | 232 | 0.50 / 0.74 | **0.53 / 0.77** |
| impaction | 33 | 0.39 / 0.88 | 0.41 / 0.88 |
| endodontic | 90 | 0.29 / 0.52 | 0.31 / 0.52 |
| **post_and_core** | 17 | 0.05 / 0.06 | **0.03 / 0.18** |
| crown | 49 | 0.14 / 0.39 | 0.11 / 0.22 |
| fillings | 74 | 0.00 / 0.00 | 0.16 / 0.28 |
| restoration | 125 | 0.24 / 0.53 | 0.24 / 0.45 |
| implants | 32 | 0.11 / 0.06 | 0.11 / 0.06 |
| canal-adjacent teeth | 26 | 0.50 / 0.15 | 0.57 / 1.00 |

Accuracy axes: periodontal resorption 0.89, sinus content 0.94, canal position 0.87,
sinus mucosa 0.81, maxilla scope 0.82, atrophy 0.78, condyle scope 0.55, bone quality
0.52, sinus scope 0.52.

Three things to carry forward from it:

- **Precision is the problem, not recall.** Impaction recalls 0.88 at precision 0.41;
  restoration 0.53 at 0.24. The model finds things and is wrong about them — which is
  the over-calling H1 is about, stated in the baseline's own numbers.
- **`fillings` PRED is 0.00/0.00 and SUMMARY is 0.16/0.28**, so `postprocess_pred.py`'s
  cross-source vote is doing all the work on that axis. An arm that improves the raw
  read there will not show up in SUMMARY unless the vote is re-examined too.
- **`post_and_core` at 0.03/0.18 is the weakest axis in the arm**, and per §3.2 it is
  also the one with no inter-reader ceiling. Movement there is the least interpretable
  movement available.

### 5.1 The metric hierarchy is inverted from the v6.9 draft

The few-shot probe's durable lesson: survey recall moved 0.40 → 0.28 while RadFact logical
recall did not move at all. So:

1. **Primary — `official_ranking.py` Final Score**, with RadFact logical precision and
   recall reported separately, **paired per case with a confidence interval**. The few-shot
   arm's CI (−0.038 to +0.050 on 24 cases) is the precedent: on 40 cases, an effect below
   about +0.03 is not detectable, and the plan must say so before the numbers arrive rather
   than after.
2. **Diagnostic — `structured_findings_evaluation.py`**, per field, per call group, `PRED` beside `SUMMARY`.
   This is where a perceptual gain should show up first and most specifically, on the
   `[detail]` group. It explains *what changed*; it does not decide whether the arm won.
3. **Arm 0 comparison** — claim counts per field, matched. Any precision gain that arm 0
   reproduces is a threshold shift.
4. **Overlay-mention rate** — the fraction of `visual_evidence` strings citing the coloured
   outline, per arm. Cheap, and it is the one check that caught the int4 failure.
5. **Held-out ΔAUROC** on the 25 held-out training cases, per boolean field. A threshold
   shift slides along one ROC curve; a perceptual gain moves the curve. Report
   Δprecision-at-matched-recall alongside raw precision.

**Do not read a per-field number without its support.** Inter-reader coverage Jaccard is
32.3% overall — endodontic 62.4%, unerupted 64.0%, fillings 28.4%, crowns 20.5%, caries
13.5%, post_and_core 4.2%; presence κ 0.865.

**These are not ceilings, and an earlier draft of this line treated them as such.** They
measure how often two readers chose to *write about* the same tooth, and reports state
findings by exception, so a reader's silence lowers this number without disagreeing with
anything (§3.2). Presence κ 0.865 is the one figure here that is a genuine agreement
statistic, because presence is the one axis both readers enumerate exhaustively. What
bounds a per-field result is its **positive count** — `with_post_and_core` has 21 across
8 cases, `with_root_fracture` 4 across 4 — and that is what the table in §3.2 reports.

### 5.2 Success, and what falsifies H1

**Win (pre-registered):** arm 2 improves **Final Score** over baseline by an amount whose
paired 95% CI excludes zero, **and** beats arm 0 at matched claim counts, **and** beats
arm 3, **and** shows the gain concentrated in the `[detail]` call group of `structured_findings_evaluation`.

**H1 is falsified by any of:** arm 3 ≥ arm 2 at 1.5%-matched parameters; arm 0 matching arm
2 at the same claim count; precision rising while held-out AUROC does not; or the gain
appearing only in mask-corroborated fields (`eruption_state`) and not in report-derived
ones — which would mean the model learned to read the segmentation overlay rather than the
anatomy. **`NO_FACTS=1` already exists and changes the *pixels*, not just the captions**;
re-render validate with it and re-run base + arm 2. If arm 2's gain vanishes without the
overlay, it learned the overlay.

### 5.3 Commands

```bash
CB=~/miniconda3/envs/cbct_base/bin/python3
SIF=$HOME/containers/vllm019_cu128.sqsh

# payload — one shared build, replayed by every arm (cpu partition)
$CB code/pipeline/preprocess/build_vqa_pairs.py --schema schema/schema.json \
    --images-dir outputs/vsft_shared_validate/images \
    --project-dir . --out outputs/vsft_shared_validate/qa_pairs.jsonl
sha256sum schema/schema.json > outputs/vsft_shared_validate/SCHEMA.sha256

# arm 0 — no GPU
$CB code/eval/rate_matched_null.py --pred-dir outputs/vsft_base_validate/predictions \
    --match-claims outputs/vsft_arm2_validate/predictions --seed 42 \
    --out-dir outputs/vsft_arm0_validate/predictions

# inference — only QWEN_MODEL_NAME changes per arm
SIF_PATH=$SIF QWEN_MODEL_NAME=Qwen3.5-9B-AWQ           RUN_NAME=vsft_base sbatch code/pipeline/aksssr_pipeline.sh validate
SIF_PATH=$SIF QWEN_MODEL_NAME=Qwen3.5-9B-AWQ-vsft-arm2 RUN_NAME=vsft_arm2 sbatch code/pipeline/aksssr_pipeline.sh validate
# ... arm1, arm3 likewise

# CPU scoring — survey_facts is stage 3 of postprocess_now.sh now
for a in base arm0 arm1 arm2 arm3; do code/pipeline/postprocess/postprocess_now.sh outputs/vsft_${a}_validate; done

# official ranking — the primary metric
sbatch --partition=gpu --qos=a100 --gres=gpu:a100:1 code/eval/judge_server.sh
tail -f logs/judge_server_*.log                  # wait for "[PASS] Judge ready"
for a in base arm0 arm1 arm2 arm3; do code/eval/eval_now.sh validate outputs/vsft_${a}_validate/synthesized_reports; done
scancel -n judge
```

**Pin the payload by content hash, not mtime** — **[v]** every file under `code/` and
`schema/` carries an mtime from a checkout with no content change, which is exactly the
false alarm the few-shot probe had to route around with `RESUME=1`.

**Read nothing until coverage reconciles.** Per arm: `called` / `unanswered` / `no_image`,
plus **`n_facts_returned` vs `n_facts_asked` per call** — the open "first complete object"
defect stores 1 fact where 5 were asked and records the call as *answered*.

## 6. Risks and guards

| # | Risk | Guard |
|---|---|---|
| **R1** | **The gain is an operating-point shift, not perception.** Observed in the few-shot arm; SFT on a ~92%-negative label set does it for free. | Arm 0 at matched claim counts; held-out ΔAUROC; Δprecision-at-matched-recall; Final Score as primary rather than survey precision. |
| **R2** | **SFT improves JSON compliance and the coverage change masquerades as clinical improvement.** Few-shot showed 31 unterminated objects vs 1. | **DONE, `aa48c4b`, before any baseline.** Confirmed `[v]` rather than assumed: vLLM's request models are pydantic `extra="allow"`, `guided_json` is deprecated-and-forwarded in 0.11.2 and **absent entirely from 0.22.0** (the version in `extraction.sqsh`), so the key was accepted, ignored and dropped — the schema really has been prompt-only for every run in this repo, and `normalize_pred.py` has been repairing violations the sampler should never have emitted. Now sent as OpenAI-standard `response_format`, which fails loudly at both ends. The tail-starts-with-`,` truncation now raises `IncompleteObject` and `infer_call` resends, never stores short. |
| **R3** | **Train/serve quantization gap** — the adapter may be fit against bf16 language MLPs and deployed against 4-bit ones, and `6e3f323` shows quantization can change this model's behaviour materially. | §4.3, and the risk is now **conditional on a stage-B check**: if `gptqmodel>=5.0.0` lets gradients traverse the quantized MLPs, training happens on the AWQ checkpoint itself and this risk disappears. Otherwise the dequantized-AWQ shadow, and only then bf16 + the explicit transfer gate. Do not assume transfer. |
| **R4** | **Overfitting to the extractor's style**, and **leakage of the held-out set**. Training labels come from a Qwen3-14B extraction. | Hold out **whole cases**, from `training` (§0). Score on the official metric against the radiologists' actual prose, which is a different source from the training labels. **No validate case ever enters training** — `build_sft_targets.py` refuses a validate case id at load time and prints the count refused, rather than trusting the pool file. |
| **R5** | **Learning the overlay rather than the anatomy.** The composite draws the target tooth in red and the canal in yellow, and `create_panoramic.py` filters outlines by `facts.structured.teeth_present`. | The `NO_FACTS=1` ablation in §5.2. ~7 min render + one inference job. |
| **R6** | **Label ceiling.** | §5.1. `post_and_core`, caries, root_fracture, periapical refused up front with the numbers as the reason. |
| **R7** | Catastrophic forgetting of report prose. | Mostly insulated — `synthesize_report.py` is deterministic templates, not an LLM. Still: zero loss on free text, ≤2 epochs, and BLEU/METEOR non-inferiority. |
| **R8** | Train/serve **prompt** skew (chat template, `enable_thinking`, interleave order). | **CLOSED `[v]`, and it fired.** §3.4's token-id diff caught vLLM hoisting the image to the front of every user turn — a two-token, wrong-order skew that would have sat under every training sample and under the 2-vs-3 contrast. Now 6,128 = 6,128 on tooth 46 and 5,223 = 5,223 on tooth 12. Re-run `code/train/prompt_parity.sh` after any change to the template, the server flags, or `build_user_blocks`. |
| **R9** | ~~`peft` × transformers 5.9.0 — the one blocking unknown.~~ | **CLOSED `[v]`.** peft 0.20.0 works against transformers 5.9.0; all four arms hit their declared counts on the meta device, on bf16 and AWQ alike. `code/train/lora_arms.py` is kept as `train_vision_lora.py`'s pre-flight gate. |
| **R10** | GatedDeltaNet backward in pure torch: unpredictable memory, slow. | **Mostly covered `[v]`.** `flash-linear-attention` 0.5.2 installed, and `modeling_qwen3_5.py:422` binds its chunked-scan kernel independently of `is_fast_path_available`, so the expensive term is already fast. `causal_conv1d` will not build without `nvcc` and covers only the depthwise conv — left out. Fallback `a100-sxm4-80gb`. |
| **R11** | **Image staleness.** Every generator fix invalidates earlier PNGs and mtime is not a valid test. | Re-render the pool; write the generator's git commit + `schema.json` sha256 into the caption sidecar and key the skip on that. |
| **R12** | `--export=ALL,VAR=a,b,c` — `--export` is itself comma-separated; Slurm reads `VAR=a`. Cost the few-shot probe a silent one-case run. | Case lists via a file; **print the count** at job start. |

## 7. Staging

| Stage | What | Where | Cost |
|---|---|---|---|
| **A0** | ~~AWQ validate-40 run~~ **DONE 2026-08-13** — 40/40 predictions, summaries and reports, batched at `MAX_CONCURRENCY=40`, 17 min 19 s wall clock, zero parse errors. Scored by `structured_findings_evaluation.py`, and its bf16 counterpart was run on identical images to remove the schema+weights confound: **AWQ costs nothing measurable** (N-weighted mean F1 0.572 AWQ vs 0.552 bf16, 8 axis wins each). Both arms live in `outputs/aksssr_v7_validate/`. **Still open:** timing one case under the A10G envelope (`--gpu-memory-utilization 0.55` on a 40 GB card, or `--qos=a30` for the real 24 GB card) against the 15-min budget — the 17-min figure is A100 throughput and does not transfer. | 1×A100 | run done; A10G timing outstanding |
| ~~**A1**~~ | ~~env + param check + structured decoding + comma-tail~~ **DONE 2026-08-13** (`aa48c4b`, `e42972a`, `e594329`, `51154c1`): cu128 env built, R9 closed, R2 both halves landed, GT absent-tooth rule implemented and both splits re-expanded. | login node | **0 GPU**, done |
| **A2** | ~~pool selection, re-render, shared payload, hash-pin~~ **DONE 2026-08-13** (`79f44d8`, `635fd21`): balanced top-30/prefix pool — wide 120 / narrow 60 / held-out 24, all 25/25/25/25 — 144 cases re-rendered tooth-only in 15.5 min (144/144, 0 fail, 3,064 composites), payload built, schema pinned `8a91c0a7…`. The v6 composites could **not** be reused: `83c95bd` changed the canal from outlined to a translucent fill three days after they were written (R11). | `--partition=cpu` | **0 GPU**, done |
| **A2b** | ~~is `post_and_core` resolution-limited?~~ **DONE 2026-08-13** (`b05c6ca`, job 554815, 16 min): **no** — §3.2b. Closes the cheapest competing explanation for H1 before an arm was booked, and cost less than the re-render it ruled out. | 1×H100 | done |
| **A2c** | ~~teacher audition + draft + screen~~ **DONE 2026-08-14**. Audition (`a5ac89c`): 27B beats the student on every axis with signal, precision *and* recall together. Draft (`8a8773f`, 36 min, H100): 9,114 evidence strings over 1,826 tooth blocks. Screen (`2b5807d`, 15 min): 67% of features confirmed, 69% of strings kept. One pass thrown away and redone — see §3.3's relative-path incident. | 1×H100 | done |
| ~~**A3**~~ | ~~`build_sft_targets.py` + provenance audit~~ **DONE 2026-08-14** (`a7bd191`, `28669b7`, `8277c03`): `sft_wide.jsonl`, 1,850 rows, evidence supervised per row at weight 0.04, 6:1 cap applied, **15,407 of 38,086 field slots supervised (40.5%)**. | login node | **0 GPU**, done |
| **A4** | **arm 0** rate-matched null. **Blocked, and structurally so:** it suppresses baseline claims *to match arm 2's per-field counts*, so it cannot run before D produces them. It is still mandatory — it just runs after arm 2, not before. | login node | **0 GPU** |
| **B** | ~~overfit sanity + step time~~ **DONE 2026-08-14** (job 555445, 1×a100-sxm4-80gb, 48 min). **The rig runs**: 200 calls × 6 epochs, 78 steps, loss 0.647 → 0.4855, `grad_norm` 0.43–3.19, no NaN, **zero dropped rows**, adapter written. **Step time 36.0 s** — closes open item 5, puts stage D at ≈4.4 h. **Gradients reach the arm**: 110/110 `lora_B` at step 0, 110/110 `lora_A` after one step, including `visual.blocks.0`. Two caveats, both now open items: the loss did **not** reach ~0 (item 13 — probably the wrong gate for a vision-only arm), and peak training memory was never printed while the probe showed 60.28 GiB against a 25–28 GB budget (item 12). Took **seven submissions**: six died in the AWQ/gptqmodel load path (§4.3b) and one on a probe that read the wrong matrix (item 3). | 1×A100 80 GB, 48 min | rig sound, step time measured |
| **C** | **arm 1** (projector, 0.29 M) on narrow → merge → transfer gate (§4.3) → serve → score | 1×A100, ~6 h + ~4 h | first real number |
| | **← DECISION GATE →** | | |
| **D** | **arm 2 + arm 3** on wide, in that order | 2×(8 h + inference) | **the result** |
| **E** | `NO_FACTS=1` ablation on the winner; contingent arm 4 | as needed | interpretation |

**The gate is not "did arm 1 win".** Arm 1 failing is a result. Proceed to D unless **B**
failed (the rig cannot overfit 200 calls — fix the rig, do not book D), or arm 1's held-out
field accuracy is indistinguishable from baseline **and** arm 0 already reproduces whatever
precision it got. In that case H1 is looking dead and D is re-scoped: run **arm 3 first**,
with arm 2 as its control — the same two jobs, read the other way round.

## 8. Files

### New

| Path | Purpose |
|---|---|
| `code/train/lora_arms.py` | **DONE `e42972a`.** The §2 arm table as code — anchored target patterns, per-rank counts, `assert_arm()`, and a standalone meta-device check. Imported by `train_vision_lora.py` so four parameter counts exist in one place. |
| `code/train/build_sft_targets.py` | tooth-composite rows of `qa_pairs.jsonl` + `{case}_gt.json` → `sft/*.jsonl`. Prompt via **imported** `run_vqa_inference.build_call_prompt`; loss mask from `_derivation` (`unstated` → `-100`) **plus null-GT → `-100`**, which after `51154c1` is where absent-tooth positions are handled, plus the `with_endo` gate; **refuses any case id found under `dataset/validate` and prints the count** (§0); refuses a payload older than `schema.json` by content hash. |
| `code/train/sft_prompt.py` | **DONE `[v]`.** One SFT row → the exact messages the pipeline sends, via **imported** `build_call_prompt`. Also the one place that knows how the SERVER assembles a multimodal turn (`CONTENT_FORMAT`, §3.4) — the collator and the parity check render through the same `render_chat_text()`, so nothing can build a training prompt a second way. Resolves image paths against the project root and **raises** on a missing image rather than silently sending a text-only call (§3.3). |
| `code/train/check_prompt_parity.py` | **DONE `[v]`.** §3.4's acceptance test as three comparisons — **A** chat text host-vs-`/tokenize` by token id, **B** the `<\|image_pad\|>` run against the grid arithmetic, **C** `json_schema` present. On a mismatch it reconstructs the other content format and names the cause rather than printing a token window. |
| `code/train/prompt_parity.sh` | **DONE `[v]`.** Boots the student with the pipeline's own serving flags and checks both call shapes behind one weight load. **On the a30 by design**: it generates nothing, so throughput is irrelevant and 11.53 GiB fits 24 GiB — it runs in ~90 s while the A100/H100 pool is queued. |
| `code/train/train_vision_lora.py` | **DONE `5021472`.** dataset → collator → `peft` + `Trainer`, driving the loss itself (`loss_on_batch`) because the built-in one is an unweighted mean and §3.3's 0.04 has to survive to the objective. `logits_to_keep=n_target+1` slices hidden states before `lm_head` (§4.4's OOM term) and **asserts the slice is the one the labels assume** rather than trusting the convention. Refuses to start unless `lora_arms.assert_arm()` matches the arm's declared count. |
| `code/train/sft_collator.py` | **DONE `5021472`.** One SFT row → `input_ids`, per-token labels, per-token weights. Only the **values** of supervised fields carry loss — not braces, field names, commas or `<\|im_end\|>`: the shape is enforced at inference by `response_format` (R2), so teaching it again would spend gradient (most of it on arm 3's side of a 1.5%-matched contrast) on the one thing already guaranteed, and would break the 0.04 arithmetic by multiplying the decision side by field-name length. Auditable with **no GPU and no model load** — `--show` prints which tokens carry loss and at what weight, so a mask that swallows a field is visible before an 8 h job rather than inferred from a loss curve. |
| `code/train/vision_sft.sh` | **DONE `5021472`, `a4a636f`, `611fec3`.** `STAGE=probe\|smoke\|train`; `STAGE=b` chains probe→smoke behind **one** weight load, and stops if the probe fails because a smoke run under a dead backward measures nothing. `ARM` is declared rather than defaulted for `STAGE=train`, so neither arm can run under the other's name. |
| `code/train/merge_vision_lora.py` | fp32 merge → bf16 cast → write into a copy of the **AWQ** checkpoint (vision tensors are bit-identical, so only `model.visual.*` changes) → copy every sidecar incl. `chat_template.jinja`, `preprocessor_config.json` → verify 20 calls against adapter-applied inference. |
| `code/sft_holdout_metrics.py` | Held-out field accuracy / **AUROC** / precision-at-matched-recall, plus the **overlay-mention rate**. What the decision gate reads. |
| `code/eval/rate_matched_null.py` | Arm 0 — suppress baseline claims to a target rate by a non-perceptual ordering; write a prediction dir the CPU path scores unchanged. |
| `code/train/select_sft_pool.py` | **DONE `79f44d8`.** Ranking + audit → narrow/wide/held-out case lists. `--per-prefix-top` is the balanced mode; held-out is stratified regardless, and drawn from the same per-prefix band as training so the gate is not easier than the task. Asserts no leak and no validate id on exit. |
| `code/train/draft_evidence.py` | **DONE `8a8773f`, `d652c1f`, `6d2c19f`.** The teacher pass at pool scale. Imports `DRAFT_SYSTEM`/`DRAFT_USER_TEMPLATE`; sends only supervised fields, never nulls, never refused ones; `--base-dir` resolves image paths; raises `NoImage` rather than drafting blind; appends every answer to `drafts.jsonl` and resumes from it. |
| `code/train/check_evidence_perceivable.py` | **DONE `2b5807d`, `6d2c19f`.** §3.3's screen at pool scale, put to the **student**. Unjudged means **drop**, not keep, and it refuses to write filtered evidence above `--max-failure-rate`. Same append-and-resume rule. |
| `code/pipeline/infer/pool_infer.sh` | **DONE `a5ac89c`, `d7fb8ac`.** One model, one existing payload, the full tooth call, scored per field — the teacher audition and the student's own baseline from one command. |
| `code/train/draft_evidence.sh`, `code/train/screen_evidence.sh` | **DONE.** The two GPU passes; the screen chains straight into `build_sft_targets.py` so the audit describes the targets that would really be trained on. |
| `env/sft_requirements.txt` | **DONE `e42972a`.** `pip freeze` of `cbct_sft_cu128` — the env training actually runs in. |
| `env/sft_reference_cu130.txt` | **DONE `e42972a`.** `pip freeze` of the cu130 clone, kept only so the cu128 build could pin the same transformers/peft/datasets/accelerate versions. Not an env anything runs in; the clone itself is deleted. |

### Edited — small, shared-code, **before** the baseline is regenerated

| Path | Change |
|---|---|
| `code/pipeline/infer/run_vqa_inference.py` | (a) **batch the per-tooth calls** — **done earlier.** (b) `parse_json`: a discarded tail beginning `,` is an incomplete call — **done `aa48c4b`**, raises `IncompleteObject`, `infer_call` resends. (c) replace the dead `guided_json` with `response_format` — **done `aa48c4b`**, with `STRUCTURED_OUTPUTS=0` to reproduce a pre-fix run. |
| `code/ground_truth/parse_reports_to_gt.py` | **done `51154c1`** — the absent-tooth null rule of §3.2. Not in the original file list; it turned out to be where the loss mask actually belongs. |
| `code/pipeline/aksssr_pipeline.sh` | **Partly done `e594329`**: step-6a vLLM/render overlap, 1 s `/health` polling, `MAX_CONCURRENCY=auto` from the reported KV pool. **Still to do:** `IMAGES_DIR` / `QA_JSONL` env overrides so every arm replays one payload (~4 lines). `QWEN_MODEL_NAME` already exists — no change needed for model swapping. |

### Reused unchanged

`build_vqa_pairs.py`, `create_tooth_detail.py`, `postprocess_pred.py`,
`synthesize_report.py`, **`structured_findings_evaluation.py`**, `survey_findings.py` (its extractors),
`official_ranking.py`, `per_dataset_breakdown.py`, `parse_reports_to_gt.py`,
`audit_report_facts.py`, `build_triage_sheet.py`, `survey_gt_quality.py`,
`code/pipeline/postprocess/postprocess_now.sh`, `code/eval/eval_now.sh`, `code/eval/judge_server.sh`, `schema/schema.json`.

## 9. Open items

1. ~~**`peft` × transformers 5.9.0**~~ — **CLOSED [v]**, §4.1. peft 0.20.0 against
   transformers 5.9.0, all four arms verified on the meta device.
2. ~~**No scored AWQ baseline exists.**~~ — **CLOSED 2026-08-13**. The earlier note
   (jobs 554171/554172 cancelled mid-first-case, zero predictions) was superseded by a
   completed run: 40/40 predictions, summaries and reports in
   `outputs/aksssr_v7_validate/predictions_awq/`, surveyed against the generated GT, plus
   a bf16 arm on byte-identical images so the comparison isolates the weights. AWQ shows
   no measurable fact-level cost (mean F1 0.572 vs bf16 0.552 on `SUMMARY`, 8 axis wins
   each) — see docs/workflow.md, "bf16 vs AWQ, same schema, same pixels". The doc mismatch that
   survived this item is also resolved: **`vllm019_cu128.sqsh` + `Qwen3.5-9B-AWQ` is the
   settled submission pairing, and AWQ is the SFT target**; `CLAUDE.md` now says so, and
   `Qwen3.5-9B-int4-AutoRound` is recorded there as measured-and-rejected.
3. ~~**Whether gradients traverse AWQ's quantized MLPs with `gptqmodel>=5.0.0`
   installed.**~~ — **RETIRED AS MOOT 2026-08-14, not answered.** §4.3b: `gptqmodel`
   cannot be used with this architecture at all (it patches Triton's autotuner globally
   and kills `fla`'s GatedDeltaNet kernels), so the AWQ backward was never reached and
   the question no longer has a consumer. What *was* settled is the question that
   replaced it: **gradients reach the arm on bf16** — job 555445, 110/110 `lora_B`
   non-zero at step 0 and 110/110 `lora_A` after one optimizer step, including
   `visual.blocks.0`, the furthest point the signal has to travel. **The older "does a
   bf16-trained adapter transfer to AWQ" question is therefore back, and is now
   mandatory** (§4.3b).
   *Method note:* the first probe read `lora_A` and reported "0 non-zero" as R3 failing.
   LoRA initialises B to zeros, so `dL/dA ∝ Bᵀ = 0` — every `lora_A` gradient is exactly
   zero at step 0 on every model, quantized or not. `.grad` being present but zero, not
   `None`, was the tell. Probe a LoRA at `lora_B`, or after a step.
4. **Container internals** — no `unsquashfs`/`squashfuse` on the login node, so the exact
   torch/transformers inside `vllm019_cu128.sqsh` are known only from the build recipe and
   runtime log lines. Irrelevant under this plan; training runs in conda.
5. ~~**Step time for the GatedDeltaNet backward at ~6,200 tokens.**~~ — **CLOSED
   2026-08-14 [v]**, §4.4. Measured **36.0 s/step** (2,808 s / 78 steps, job 555445),
   inside the 25–40 s arithmetic. Stage D ≈ 4.4 h. Caveat recorded in §4.4: measured on
   an 80 GB SXM4 card, so it is an optimistic bound for the 40 GB PCIe card §4.5 books.
6. **How much the four throughput changes buy together** — batching, the step-6a overlap,
   1 s polling and `auto` concurrency. Measure on A0 before deciding how much of the
   15-minute budget SFT has to fit into.
7. **A cu129 container as an alternative base.** Driver 570 runs any CUDA 12.x through
   minor-version compatibility, so the refusal is `NVIDIA_REQUIRE_CUDA=cuda>=12.9` being
   checked by the container runtime, not a capability limit — bypassable with
   `NVIDIA_DISABLE_REQUIRE=1`. It buys no newer vLLM (0.20.0+ is CUDA 13), and the
   failure mode is a container that will not start on the leaderboard. **[?]**
8. **`vllm019_cu128.sqsh` does not run on the B200s. [v]** sm_100 trips a Blackwell
   kernel-library mismatch inside the image (`quack.layout_utils` →
   `cutlass.cute.core has no attribute 'ThrMma'`) and vLLM dies at engine start in ~90 s
   (job 554979). Treat s0-n03's 8 cards as unavailable to this project. A100 and H100
   both work; the a30 is the only 24 GB card and holds the 9B student but not a 27B.
9. ~~**`train_vision_lora.py` and its collator do not exist.**~~ — **CLOSED 2026-08-14**
   (`5021472`). `code/train/sft_collator.py` turns §3's field-level decisions into `-100` spans
   and a per-token weight vector; `code/train/train_vision_lora.py` drives the loss itself so
   0.04 survives to the objective, slices with `logits_to_keep` and asserts the slice
   matches the labels, and calls `lora_arms.assert_arm()` before touching a GPU. The
   collator audits with **no GPU** (`--show`), which is the property that keeps a bad
   mask out of an 8 h job.
10. ~~**§3.4's acceptance test has not been run.**~~ — **CLOSED 2026-08-14 [v]**, §3.4.
   It ran, it failed, and the failure was in the pipeline rather than the collator:
   vLLM's `string` content format hoists the image to the front of the user turn, so the
   caption-before-image design has never reached the model. The collator now matches the
   server (`sft_prompt.CONTENT_FORMAT`), and parity is exact on both call shapes. The
   argument for running it before any training was §3.3's relative-path incident; the
   argument for keeping it is that it found a second one.
11. **The evidence weight of 0.04 is a hyperparameter**, so per §0 it is tuned — if at
   all — on the 24 held-out *training* cases, never on validate. An A/B against
   `--evidence-weight 0` is the obvious ablation and costs one extra training run.
12. **Peak GPU memory during training is unmeasured**, and the one figure we have
   (60.28 GiB, stage-B probe) is more than double §4.4's 25–28 GB budget. Stage D is
   booked on a 40 GB card that would not have survived the probe. Print the peak from
   the training loop, then either re-book the card or find the missing term. §4.4.
13. **§7's stage-B gate is not satisfiable by a vision-only arm, as written.** It asks
   for train loss → ~0 on 200 calls; arm 2 got 0.647 → 0.4855. Arm 2 trains 7.99 M
   vision parameters while the entire language model that emits the target text is
   frozen, so a vision adapter changes what the model *sees*, not what it can *say* —
   driving cross-entropy on text targets to ~0 may be out of reach by construction
   rather than by defect. Everything else about the run is healthy: monotone decrease,
   `grad_norm` 0.43–3.19, no NaN, zero dropped rows, gradients confirmed at every one of
   the 110 modules. **Decide what stage B's pass condition should be for arms 1 and 2
   before reading their results.** Arm 3 trains language parameters and should overfit
   far harder on the same 200 calls, which makes it the cheap empirical test of whether
   0.49 is a floor or a failure.
