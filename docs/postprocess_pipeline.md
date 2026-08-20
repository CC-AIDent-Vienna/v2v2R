# Post-processing and report synthesis

Everything downstream of the VLM: `postprocess_pred.py` → `source_rules.py` →
`synthesize_report.py`. The ten source rules are the substance — they decide
which SOURCE each finding comes from, rather than voting between every read that
can assert it.

**The arm this describes** is arm 6 — the LoRA-SFT arm of
`docs/vision_sft_plan_light.md` §8.3 — on validate-40, with the rules on. It scores
**0.4658** official, the current best, and it is the first model trained on the
nine `global` arch calls per case. The scored run is
`outputs/aksssr_v7_trained_arm6_validate/`; the earlier
`outputs/training_results/vsft_arm6/val_arm6/` holds the same predictions read
by an older postprocess and scores 0.4557 — see *Report level* below.

> **Validate-40 only.** Arm 6's inference on the 582-case training split **has
> not been run**, so every training-split section below is empty and says so.
> The sections that once held those numbers were measured on a superseded arm
> and have been removed rather than left to be mistaken for this one's.
> `git log -- docs/postprocess_pipeline.md` has them.

Fact-level
numbers come from `structured_findings_evaluation.py` / `compare_sources.py` against the generated
ground truth, report-level numbers from `official_ranking.py` against the
reference reports.

**Read it in this order.** Each part is the evidence for the next:

| | | |
|---|---|---|
| **1** | *Per-source accuracy* | which source knows what — one row per finding, one column per source. Every rule is chosen off this table. |
| **2** | *The result* | what the rules did, at fact level and at report level. |
| **3** | *The rules* | the ten source rules plus the maxilla FOV gate, each with the measurement that chose it and what it gives up. |
| **4** | *Reference* | how to run it, what reaches the report, where the artifacts are, and what not to trust. |

> **Removed 2026-08-16: the 2026-08-07 half.** This document used to open with
> how the **0.3193** baseline was built on the v6.4 arm — its frozen input, its
> hour-by-hour timeline, its per-sub-dataset table and its `survey_findings.py`
> `REPORT_GT` survey. That arm is superseded, its label is superseded, and it
> was scored under the `gpt-4o` judge rather than the local `qwen3-14b-text`
> one, so **no number in it was comparable to anything here**. What was
> load-bearing survived in place: the silencing rule and the ≥0.35 precision bar
> it produced (*What reaches the report*). The facts-file provenance went with
> it — it belongs to the facts pool, not to postprocessing;
> `code/competition/competition_sim.sh`'s header is where that lives.
> `git log -- docs/postprocess_pipeline.md` has the rest.

## 1. Per-source accuracy — which source knows what

The pipeline's own instruments treat it as one voice: `PRED` (all reads
unioned) against `SUMMARY` (after the vote). That collapse hides which *read*
earned a number, and therefore which source a rule should trust.
`code/eval/compare_sources.py` splits it — one row per finding, one column per
source: the mask-derived **facts** (see below), the three image reads (**3d**,
**panoramic**, **detail**, plus **sinus**), then `PRED` and `SUMMARY` as
before. It reuses `pred_claims()`'s existing per-source tags, so the split
cannot drift from what the pipeline consumes, and its `PRED`/`SUMMARY` columns
reproduce `structured_findings_evaluation.py`'s OVERALL table exactly — which is the check that
it is the same measurement, only disaggregated.

```bash
python3 code/eval/compare_sources.py outputs/aksssr_v7_trained_arm6_validate --split validate
# -> <run>/survey/source_compare_<stamp>.{txt,json}
```

**The `facts` column is `dataset/$SPLIT/facts/<case>.json`, and it is a function
of the segmentation mask** — `extract_facts.py` over the mask reproduces the
upstream file byte for byte, and no report was ever read to build it
(`code/competition/competition_sim.sh`'s header has the provenance and the check). Two
consequences every rule rests on: scoring it against report-derived ground
truth is not circular, and the competition container can compute it from a CBCT
with no report in the room.

The facts file already had one consumer before any of this —
`validate_summary_with_facts.py`, a drop-only gate that removes summary claims
the mask contradicts and adds nothing. It is **not in the current path**;
`postprocess_now.sh` runs `postprocess_pred.py --facts-dir` → `source_rules.py`,
and the gate is reached only by the older `code/arms/postprocess_val_now.sh`. It is
named below because every rule here argues against its asymmetry.

Measured on **arm 6**, against the generated ground truth, on **validate-40**.
The 582-case training split is *not* measured — see *The same table on the
training split*, which is empty for that reason.

**Why an SFT arm is a fair place to measure this.** Arm 6's *reads* are its own;
a different model would move every VLM column here. The `facts` column would
not — it is a function of the segmentation mask, so it is identical across arms
by construction, and that invariance is the check that a re-measurement is
sound: **any future table where `facts` moves has a bug in it, not a finding.**
Every rule below rests either on that column or on a comparison made WITHIN one
run, so no conclusion inherits this arm's read quality. The absolute VLM numbers
do: where a rule's margin depends on how badly a read performs, a stronger model
narrows it — which is exactly what happened to **fillings**, whose source sat at
0.39 precision when the rule was sized and reads **0.31** here.

Two conventions govern every number in the table — recall is scored over what
each source was SHOWN, and GT silence is not charged as a false positive on the
absence axis. Both can invert a verdict; they are stated under
*Two scoring rules the numbers depend on*, after the table.

### The table

**Validate-40, arm 6, with the source rules on.** Every cell is a straight read
of `compare_sources.py`; nothing here is lifted from a rule section or derived
by hand.

| axis | N | facts | 3d | panoramic | detail | SUMMARY | status |
|---|---|---|---|---|---|---|---|
| absent teeth | 232 | **0.77/0.85** ᵃ | 0.74/0.94 | 0.73/0.67 | — | **0.83/0.85** | on |
| impaction | 33 | — | 0.62/0.48 | 0.67/0.06 | **0.79/0.82** ᵇ | **0.78/0.85** | on |
| endodontic | 90 | — | — | 0.52/0.13 | **0.85/0.74** | **0.88/0.64** | on (teeth only) ᵈ |
| post_and_core | 17 | — | — | — | 0.08/0.09 | 0.06/0.12 | **off** |
| crown | 49 | **0.33/0.29** | — | — | 0.32/0.32 | **0.29/0.31** | **off** |
| fillings | 74 | — | — | — | 0.31/0.26 | **0.30/0.24** | on ᵉ |
| caries | 26 | — | — | 0.00/0.00 | 0.14/0.05 | **0.14/0.04** | **off** ˡ |
| root remnants | 13 | — | — | — | 0.05/0.17 ᵐ | **0.05/0.08** | **off** ˡ |
| implants | 32 | **0.68/0.53** | — | 0.44/0.12 | — | **0.68/0.53** | on |
| fixed bridges (cases) | 11 | **1.00/0.55** | — | 1.00/0.09 | — | **1.00/0.55** ᶠ | on |
| canal-adjacent | 26 | 0.69/0.69 | **0.70/0.27** | — | 0.17/0.32 | **0.69/0.69** | on |
| canal position (sides) | 63 | — | 0.84 | — | — | **0.86** ⁱ | on |
| maxilla scope (cases) | 40 | **0.85** | 0.82 | — | — | **0.88** ⁿ | on |
| atrophy (arches) | 80 | — | 0.76 | — | — | **0.78** | on |
| sinus mucosa (sides) | 80 | — | — | — | — | **0.85** ʲ | on ᵏ |
| sinus content (sides) | 80 | — | — | — | — | **0.95** ʲ | on ᵏ |
| sinus scope (sides) | 80 | — | — | — | — | **0.73** ʲ | on ᵏ |
| periodontal resorpt. (arches) | 80 | — | — | 0.90 | — | 0.90 | **off** |
| bone quality (arches) | 80 | — | — | 0.66 | — | 0.62 | **off** |

**status** = does this finding reach the synthesized report today. `off` means
its renderer in `synthesize_report.py` returns empty —
`render_restoration_summary` for crown and post-and-core (its fillings half now
emits), `render_morphology_findings`, `render_root_remnants` and
`render_periodontal_bone_resorption`. Read the accuracy of an `off` row as
diagnostic only: it is measured on summary content that no report text is built
from, so improving it changes nothing until the renderer is turned back on.

**Every row below 0.2 precision is `off`**, and no `on` row sits below 0.29 —
which is the silencing rule —
*a finding reaches the report only if it is measurably better than silence* —
still holding after a per-source split it was not derived from. Caries and root
remnants are the two worst rows in the table and were silenced on different
evidence entirely (per case, against the reference reports); this is the first
time either has been scored per source against the generated GT, and it agrees.

**The rule reads on precision, and two `off` rows print accuracy instead** —
`periodontal resorpt.` at 0.90 and `bone quality` at 0.66 — so nothing in this
table shows why *they* are silenced, and they look like the obvious candidates
to switch back on. They are not: scored on the abnormal class both collapse,
and periodontal resorption lands **below** the always-`none` constant. The proof
is *Why `periodontal resorpt.` and `bone quality` stay silenced*, immediately
below.

**The panoramic's implants row was the uncomfortable one and no longer is.**
Trained on the arch calls, it reads 0.44/0.12 where an untrained model was wrong
every time it spoke — but the mask beside it still scores 0.68/0.53 on both
halves, so the rule that sources implants from the mask is untouched by the
improvement.

— means the source is *structurally* silent, not that it claimed nothing:
v7.1's arch vocabulary cannot tell a crown from a large filling (both fold to
`restoration`), a mask knows nothing about root canals, and the composite is
never asked about implants. Every such blank is one line in `SPEAKS` with its
reason, printed under the table.

ᵃ The raw `teeth_absent` list. With the FOV gate of *THE RULE — absent teeth*
it is **0.83/0.85**, which is what the pipeline should consume; the raw number
is kept in this row because the column is a source, not a rule.

ᵇ **Available, not consumed.** The composite carries no `impaction` fact — that
left the schema in v6.3 — but it carries `tooth_{fdi}_eruption.eruption_state`,
which is half of the schema's own `_definitions.impacted`. This cell is that
half applied on its own, and `compare_sources.py` derives it rather than
reading it from `pred_claims`, because printing "—" would assert the composite
*cannot* speak when the truth is that nothing asks it. It is the only derived
cell in the table, and it is a floor: the other half of the rule
(`orientation`) lives on the 3D fact, and combining the two is
*THE RULE — impaction* below.

ⁿ **The rule is not in `source_rules.py`** — it fires far earlier, in
`postprocess_pred.py`'s `drop_excluded_maxilla` gate, which decides whether the
maxilla arch is reported at all. `facts` 0.85 against the read's 0.82 is why the
gate reads the facts; SUMMARY 0.88 is the gate's own score. See *THE RULE —
maxilla FOV scope*.

ʲ Read from the **sinus crops**, a source with no column in this table (the
five-source view is `compare_sources.py`'s own output). Scored on the abnormal
class like the rows around it: across 38 answered sides the model claims
`thickening` **0 times** against 6 in the reference, and `fluid`/`mixed` **0
times** against 2. Its enum accuracies — 0.85 and 0.95 — are therefore the
always-normal and always-air constants, neither above nor below. `scope`'s 0.73
is a third kind of nothing; *The same view for `atrophy` and the three sinus
rows* decomposes all three.

ᵏ `on`, and gated on the maxilla FOV: **where `fov.maxilla == "excluded"` the
sinus is not mentioned at all.** That invariant already holds — across the 14
excluded cases the pipeline answers **0** sinus sides, because no maxillary
bone means no segmented sinus, hence no crop and no call — so recording it
fixes existing behaviour rather than changing it. It is worth fixing: the
reference states 28 sinus sides in those same cases, so the gate is a real
recall cost knowingly accepted, on the same principle as the FOV gate in
*THE RULE — absent teeth* — a structure outside the volume is not imaged, and
not-imaged is not a finding.

ˡ **Both were silenced before this table existed, and it confirms the call.**
Their renderers — `render_morphology_findings` for caries and
`render_root_remnants` — return empty, so these are diagnostic rows measured on
summary content no report text is built from. Scored per source:

| | GT | prec/rec |
|---|---|---|
| caries, panoramic `defect` | 26 | 0.00/0.00 |
| caries, composite `with_caries` | 26 | 0.14/0.05 |
| root remnants, composite `is_remnant` | 13 | 0.05/0.17 |

**Root remnants are the lowest-precision finding row in the table** — the same
conclusion the silencing measurement reached from the other direction (107
claims against 10 cases whose reports mention a remnant), reproduced against a
different label. **Caries has no source left that speaks**: the panoramic's
`defect` claims are now wrong every time, and the composite recovers 5% of the
lesions. Neither axis has a mask source — a carious lesion is missing mineral
rather than a label, and `extract_facts.py` either drops a fragment under
`TOOTH_MIN_MM3` or records it as a present tooth, so neither is a remnant claim.

ᵐ **The recall gap is coverage, not accuracy.** The composite is expanded only
for teeth the segmentation places, and a remnant is exactly the case where it
often does not, so the source reads 0.17 recall over the crops it got and 0.08
over all 13 GT remnants. That gap is the one thing that would improve if
remnants were ever worth reporting.

ⁱ Validate is a friendly split for this axis: always-lingual scores 0.873 here
against **0.806** over the 794 training sides. Expect ~0.81, and treat the
margin over shipping as provisional until the composite is scored on training
inference. See *THE RULE — canal location*.

ᶠ **The rule is per ARCH, the row is per case.** It needed a new fact:
`audit_facts.py --derive-bridge-arches`, which writes
`structured.bridge_arches` for `apply_bridges` to source from, and has been run
on validate and on all 582 training facts files. Per-arch is what the mask
settles; per-case is all the pipeline carried before. No span is claimed; see
the rule.

The `periodontal resorpt.` and `bone quality` rows are per ARCH (80 = 40 cases ×
2) and print **accuracy**, which is the wrong lens for an enum dominated by its
negative class — they are the always-normal constant wearing a respectable
number. *Why `periodontal resorpt.` and `bone quality` stay silenced* is the
proof.

The `restoration` row was dropped from this table: it is the roll-up
crown ∪ fillings, so it restated two rows that are now separately decided.
`compare_sources.py` still computes it, because it is the only row in which the
arch survey's `restoration` claims are scored at all.

Endodontic's SUMMARY recall (0.64) sits below the composite column's (0.74)
because a SUMMARY figure is scored over every GT positive, while a source figure
is scored over the positions that source was shown. Same claims, same errors,
wider denominator — see the rule.

ᵉ **On, since 2026-08-16.** Turning it on was not a flag flip:
`render_restoration_summary` silenced crown, post-and-core and fillings as one
unit, so the renderer had to be split, and the fillings sentence is gated on the
summary carrying `source_rules` — it is worth emitting only at the composite's
precision, never at the arch survey's aliased 0.17, which is below every other
reported finding. **The margin has narrowed**: the rule was sized when the
composite read 0.39 and it reads 0.31 here, so what the rule buys is about a
third smaller than when it was chosen. It still holds — nothing else asserts
fillings better — but it should be re-read on the training split before being
trusted at the old strength. See *THE RULE — fillings*.

ᵈ **teeth only** = the report names *which* teeth are root-canal treated and
says nothing about *how*. `render_endodontic_summary` emits one sentence
("Endodontic treatment involving teeth 16, 26, 36.") and deliberately drops
`endodontic_summary.quality_groups` — adequate / inadequate / overfilled /
discontinuous — which stays in the summary JSON and never reaches the report.
So the finding is half-rendered: presence on, quality off. The accuracy in this
row scores presence only, which is the half that ships.

Four readings beyond absence, none of them acted on here:

- **The mask wins wherever it can speak at all** — implants (0.68/0.53 against
  the read's 0.44/0.12), bridges (1.00/0.55 against 1.00/0.09, all of it
  recall), canal-adjacent (0.69/0.69 against PRED's 0.22/0.46), maxilla scope
  (0.85 against 0.82). Its crown list at 0.33/0.29 is *also* level with the
  composite's 0.32/0.32 — a sharper number than the 0.45/0.24 the same list
  scores against the reference reports, and a margin thin enough that
  *THE RULE — crown* should not be read as settled.
- **The detail crop is the best read** on every axis it and another source both
  answer — endodontic 0.85/0.74 against the arch's 0.52/0.13, impaction
  0.79/0.82 against the 3D fact's 0.62/0.48 on half the rule. The arch survey
  earns its place through coverage, not accuracy.
- **PRED can be beaten by one of its own inputs.** On impaction the union
  scores 0.62/0.48 while the composite alone scores 0.79/0.82 — the pipeline
  does not read the composite here, so the better source sits unused in the
  prediction file. That is the pattern the whole per-source split exists to
  expose, and it is what *THE RULE — impaction* acts on.
- **Training on the arch calls moved the reads, not the mask.** `3d` improves on
  every axis it speaks to — impaction precision 0.62, canal-adjacent 0.70,
  atrophy 0.76 — and `panoramic` implants left 0.00/0.00 behind. `facts` is
  unchanged to two decimals on every row, as it must be.

#### Why `periodontal resorpt.` and `bone quality` stay silenced

These two rows print 0.90 and 0.66 in the table above, which are the two
healthiest-looking `off` rows in the whole document and the obvious candidates
for someone to switch back on. **They are the always-normal constant wearing an
accuracy score**, and this is the measurement that says so. Both are enums
dominated by their negative class, so accuracy is the wrong lens; scored on the
**abnormal class**, which is the only class a report sentence would ever be
built from:

**periodontal resorption — 40 cases x 2 arches = 80 arch slots**

| run | claims abnormal | right | precision | GT abnormal | found | recall | acc |
|---|---|---|---|---|---|---|---|
| baseline (AWQ, untrained) | 5 | 0 | **0.00** | 6 | 0 | 0.00 | 0.86 |
| **arm 6** | **2** | **0** | **0.00** | 6 | 0 | 0.00 | 0.90 |

**bone quality — same 80 arch slots**

| run | claims abnormal | right | precision | GT abnormal | found | recall | acc |
|---|---|---|---|---|---|---|---|
| baseline (AWQ, untrained) | 6 | 2 | 0.33 | 27 | 2 | 0.07 | 0.64 |
| **arm 6** | **2** | **1** | 0.50 | 27 | 1 | **0.04** | 0.66 |

**Against the constants**, which is the comparison that settles it:

- periodontal GT is 74 `none` / 4 `severe` / 2 `mild`, so **always-`none` scores
  0.925**. Arm 6 scores **0.90**. It is the only row in the table *below* its
  own trivial constant — answering "no resorption" without opening the image
  would beat it. It also emitted one `moderate`, a value that occurs **zero
  times** in the ground truth.
- bone quality GT is 53 absent / 27 present, so **always-`false` scores 0.662**.
  Arm 6 scores **0.66** — the constant to two decimals, reached by answering
  `false` on 78 of 80 arches. Recall on the abnormal class is **1 of 27**.

This is the same disease footnote ʲ documents for sinus mucosa and content, and
these two rows belong in that company: an enum accuracy that is exactly the
majority-class answer, neither above nor below it. The difference is that those
two were already understood; these two look respectable in the accuracy column
and are not.

**Training does not fix it, and it is not a class-balance artefact.** Both
fields were supervised in arm 6 — `periodontal_bone_resorption_{arch}.extent`
and `bone_quality_{arch}.present`, 558 slots each — and bone quality arrived
close to balanced (263 true / 295 false mandible, 200 / 358 maxilla), so §3.5's
6:1 negative cap never bit. A field that is balanced in training and still
answered `false` 78 times out of 80 at inference is most likely **not legible on
the panoramic at this resolution**, rather than mis-weighted. What arm 6 did
change is the operating point: across both axes it makes **4** abnormal claims
where the baseline made **11**, and its accuracy rose purely by moving closer to
the constant.

So the silencing rule — *a finding reaches the report only if it is measurably
better than silence* — does not merely tolerate these two rows being `off`. For
periodontal resorption, silence is **literally more accurate** than the read.
Turning either renderer back on would need a source that does not exist yet, not
a threshold change.

#### The same view for `atrophy` and the three sinus rows

The two rows above are silenced. These four are **`on`**, and the abnormal-class
view separates them sharply: one is a genuine read, two are constants, and one
is not a scorable axis at all.

**`atrophy` — the real one.** 40 cases x 2 arches = 80 arch slots, all answered.
`present` is "at least one stretch of 3+ missing teeth"; `atrophy` is reduced
bone height, gated on it.

| run | field | claims | right | precision | GT+ | recall | acc |
|---|---|---|---|---|---|---|---|
| baseline (AWQ, untrained) | `present` | 30 | 21 | 0.70 | 32 | 0.66 | 0.75 |
| **arm 6** | `present` | 23 | 19 | **0.83** | 32 | **0.59** | **0.79** |
| baseline (AWQ, untrained) | `atrophy` | 17 | 11 | 0.65 | 20 | 0.55 | 0.77 |
| **arm 6** | `atrophy` | 23 | 13 | 0.57 | 22 | **0.59** | 0.76 |

Always-negative scores **0.600** on `present` and **0.725** on `atrophy`. Arm 6
beats both (0.79, 0.76) and beats the untrained baseline's precision on
`present` (0.83 vs 0.70) while claiming fewer times. This row earns its `on`,
and it is the clearest single thing training the arch calls bought.

**`sinus mucosa` and `sinus content` — constants, exactly.** Only 40 of 80 sides
are answered at all; the crop exists only where the sinus is in the FOV.

| axis | run | answered | claims abnormal | right | GT+ | acc | always-normal |
|---|---|---|---|---|---|---|---|
| mucosa (`thickening`) | baseline | 38 | 2 | 0 | 6 | 0.79 | **0.842** |
| | **arm 6** | 40 | **0** | 0 | 6 | **0.85** | |
| content (`fluid`/`mixed`) | baseline | 38 | 0 | 0 | 2 | 0.95 | **0.947** |
| | **arm 6** | 40 | **0** | 0 | 2 | **0.95** | |

Arm 6 claims the abnormal class **zero times** on either axis, so 0.85 and 0.95
are the always-normal and always-air constants — confirming footnote ʲ, and
showing that training the arch calls changed nothing here. 6 thickenings and 2
fluid/mixed findings are missed in full. Unlike the two silenced rows these are
`on`, but what they emit is the normal sentence, so they add no false claims
either.

**`sinus scope` — not a scorable axis.** The ground truth is
`partially_included` on **80 of 80** sides, and that is not an observation:
`_resolve_scope` in `parse_reports_to_gt.py` returns `partially_included` as its
**fallback**, taking `fully_included` only when the report text matches
`_FULL_INCLUSION` and `not_included` only where allowed.

| run | answered | correct | acc | what it predicted |
|---|---|---|---|---|
| baseline (AWQ, untrained) | 38 | 23 | 0.61 | 15 `fully_included`, 23 `partially_included` |
| **arm 6** | 38 | **38** | **1.00** | **38 `partially_included`** |

Both numbers are the same fact about which constant the model settled on. Arm 6
was anchored onto the one the GT happens to emit and scores 1.00 against a label
with no information in it. **That is worth doing** — a model that settles on the
wrong constant emits 38 wrong sentences — but it demonstrates no ability to read
sinus extent.

**And the SUMMARY shortfall is now entirely postprocess's own derivation, not a
discarded read.** SUMMARY answers 52 sides and gets **38** right — exactly the
38 the model read, so **no correct read is dropped any more**. The 14 misses are
postprocess **deriving** a scope from the maxilla FOV gate (footnote ᵏ) on sides
the model never answered. That derivation is mask-based, which everywhere else
in this document is the source we trust over a read — so "fixing" it to match
the GT would mean deleting a mask signal in favour of a code default. **Leave it
alone.** The axis needs a ground truth before it needs a rule.

Two correct reads *were* being discarded until 2026-08-17, both of them A018's,
and *THE RULE — maxilla FOV scope* is what recovered them: the arch was being
deleted on the model's word, taking its two perfectly good sinus reads with it.
That is the 0.72 → 0.73 on this row and the 0.84 → 0.85 on mucosa.

**What did not move**: the detail crop still wins every axis where it and
another source both speak, the mask still wins everything it can speak on except
crown, and no `off` row crosses the silencing threshold.

### The same table on the training split — 582 cases

**Empty: arm 6 has no training-split inference.** `outputs/training_results/vsft_arm6/`
holds `val_arm6/` and the 22-case `ho_awq_arm6/` only, so the 582-case version of
the table above does not exist for this arm and nothing in this document is
measured on more than 40 cases.

What that costs is specific, and it is the same list every time:

- **the distribution.** Validate is balanced 10/10/10/10; training is **69% P**
  (402 of 582; A 15%, F 9%, S 7%). Every number above is measured on a quarter
  as much P as the pipeline will actually meet.
- **the thin margins.** `crown` (mask 0.33/0.29 against the composite's
  0.32/0.32) and the impaction C-vs-E choice both sit inside the noise of 40
  cases and can only be settled on the larger split.
- **`canal position`.** Always-lingual scores 0.873 on validate against 0.806
  over the 794 training sides, so the rule's margin over the constant is
  measured on the friendlier of the two splits.
- **the abnormal-class sub-analyses** below — atrophy, the three sinus rows,
  periodontal and bone quality — are 80 arch slots each.

Running it is one inference job plus `compare_sources.py --split training`; the
CPU half is seconds. Until then, treat every figure here as validate-40.

### Two scoring rules the numbers depend on

These are how the table above is scored, not incidental caveats: the second one
changed which source wins the absence axis.

**Recall is scored over what the source was shown.** The sources do not cover
the same positions — the 3D renders answer four wisdom slots, the composites
are expanded only for teeth the segmentation places — so an unrestricted
denominator reports a difference in *coverage* as a difference in *accuracy*.
Each source carries a domain, printed as `dom`. Precision is never restricted:
a claim is right or wrong wherever it was made.

**GT silence is not a false positive, on the absence axis only.** No reference
report enumerates all 32 positions; the generated GT answers a mean of **16.9
of 32** per case, and a position no report mentions is *unanswered*, not
"present". Charged over all 32, every source loses ~0.25 precision to the
label's silence — and it lands hardest on whichever source claims most, which
penalises the mask precisely for being complete. Absence is therefore scored
only over positions the GT settles. Deliberately not applied to the finding
axes, where an arch read of `unremarkable` *is* an answer of "nothing there".

Both effects are large enough to invert a verdict, and the second one did:
before restricting to the answered positions, facts read 0.55/0.85 and the
prediction 0.50/0.72 — the same ordering, but at a precision that made the mask
look untrustworthy in absolute terms.

## 2. The result

### Fact level — measured after implementation

All ten rules were **recorded 2026-08-15/16 and IMPLEMENTED 2026-08-16** in
`code/pipeline/postprocess/source_rules.py`, applied as a post-pass by `postprocess_pred.py
--facts-dir` and on by default in `postprocess_now.sh` (`NO_SOURCE_RULES=1`
turns them off). *The rules* states each one, the measurement behind it and
what it gives up.

Arm 6, same 40 cases, same predictions, `compare_sources.py`'s SUMMARY column,
one run with `--facts-dir` and one without — so `pre-rules` and `with the rules`
differ by the rules and nothing else:

| axis | pre-rules | with the rules |
|---|---|---|
| absent teeth | 0.82/0.75 | **0.83/0.85** |
| impaction | 0.67/0.48 | **0.78/0.85** |
| endodontic | 0.75/0.66 | **0.88/0.64** |
| fillings | 0.21/0.35 | **0.30/0.24** |
| crown | 0.30/0.20 | **0.29/0.31** |
| implants | 0.44/0.12 | **0.68/0.53** |
| canal-adjacent | 0.67/0.46 | **0.69/0.69** |
| canal position | 0.83 | **0.86** |
| atrophy | 0.76 | **0.78** |
| fixed bridges | 1.00/0.09 | **1.00/0.55** |
| maxilla scope | 0.85 | **0.88** |

**Nine of the eleven are a clear gain, and the two that are not are known.**
`fillings` buys 0.09 precision with 0.11 recall — the one rule that removes
correct claims, and it is meant to. `crown` loses 0.01 precision for 0.11
recall, which is the margin *THE RULE — crown* already calls the thinnest in the
set; on this arm the mask (0.33/0.29) and the composite (0.32/0.32) are level,
so the rule is no longer choosing a better source, just a different one.

**Implants and bridges are where the rules do their real work**, and both are
almost entirely recall: 0.12 → 0.53 and 0.09 → 0.55. Fixed bridges lands on the
facts column exactly, because the rule is a SOURCE and not a gate. It was
written as a gate first, and that was wrong in a way worth recording: dropping
the VLM's wrong-arch bridges left **4 of the 6** mask-confirmed cases reporting
no bridge at all. Gating protects precision by deleting the finding — the same
trap as the old gate's drop-only asymmetry, walked into while implementing the
rule that exists to break it.

**Absence gains recall, not precision** (0.75 → 0.85 at a flat 0.82/0.83), which
is the FOV-gated mask list reaching positions no read answered.

**Verification.** Without `--facts-dir` the summaries are byte-identical to what
the pre-rules code produced, so nothing here can affect an arm that does not ask
for the rules — re-checked on 2026-08-17 when the maxilla FOV gate was added,
by rebuilding all 40 summaries against the previous commit. And no per-tooth
finding survives on a mask-absent position across the 40 cases — see the
gate-ordering note in `source_rules.apply`.

### Report level — 0.4658

The table above is fact level; this is the official score of the reports those
facts produced. Validate-40, judge `qwen3-14b-text` via `judge_server.sh`,
scored 2026-08-17 22:52 into `outputs/aksssr_v7_trained_arm6_validate/`.

| | arm 6 | arm 6, 07:12 |
|---|---|---|
| **Final Score** | **0.4658** | 0.4557 |
| Clinical (RadFact logical F1) | 0.5139 | 0.5014 |
| RadFact precision | 0.5306 | 0.5409 |
| RadFact recall | 0.4982 | 0.4674 |
| BLEU-4 (corpus) | 0.1626 | 0.1629 |
| METEOR | 0.3848 | 0.3830 |
| Captioning (mean of the two) | 0.2737 | 0.2730 |

**The second column is the same model on the same predictions.** Arm 6 was
inferred once; `outputs/training_results/vsft_arm6/val_arm6/predictions/` and
`outputs/aksssr_v7_trained_arm6_validate/predictions/` are byte-identical. What
moved between 07:12 and 22:52 is postprocess — the condyle, maxilla-FOV and
atrophy commits of that day — and it is worth **+0.0101**, all of it clinical.
Quote 0.4658; `val_arm6/` no longer even holds the reports it scored.

**Precision and recall are level** — 0.53 against 0.50 — which is the operating
point the source rules were chosen for: they add claims the reads missed
(implants, bridges, FOV-gated absences) and remove claims the reads invented
(the arch survey's fillings, the endodontic arch half), rather than trading one
for the other. The last postprocess round spent 0.010 of precision to buy 0.031
of recall, which moved F1 the right way and tightened the gap.

One artefact in this run, working against it: case `S0021` produced a report
with **zero parseable phrases**, so `official_ranking.py` forced its precision
to 0.0 rather than scoring it as an unremarkable report. One case in 40.

#### What the rules are worth at report level — not measured on this arm

**Empty.** The fact-level gain is in the table above; the report-level gain is
not, because scoring the pre-rules arm needs a second judge run
(`NO_SOURCE_RULES=1 postprocess_now.sh` then `eval_now.sh`) and that run has not
happened for arm 6. `NO_RADFACT=1` would give the caption half free and
deterministically, and the caption half is the half that has never moved.

#### Per sub-dataset (official convention: `0.8 × Clinical + 0.2 × mean(cBLEU-4, METEOR)`)

| dataset | clinical | cBLEU-4 | METEOR | Caption | RF-P | RF-R | **Final** |
|---|---|---|---|---|---|---|---|
| A | 0.4784 | 0.0826 | 0.2554 | 0.1709 | 0.5116 | 0.4492 | **0.4169** |
| F | 0.4043 | 0.2137 | 0.4832 | 0.3440 | 0.2960 | 0.6376 | **0.3922** |
| P | 0.5659 | 0.1447 | 0.3771 | 0.2503 | 0.6168 | 0.5227 | **0.5028** |
| S | 0.4948 | 0.2072 | 0.4234 | 0.3149 | 0.6978 | 0.3833 | **0.4588** |
| **ALL** | 0.5139 | 0.1626 | 0.3848 | 0.2700 | 0.5306 | 0.4982 | **0.4651** |

(`per_dataset_breakdown.py --corpus-bleu` recomputes cBLEU per group, so its ALL
lands at 0.4651 against the official 0.4658; the difference is the BLEU
denominator, not the arm.)

**P is still the best prefix, by a distance.** 0.5028 against A's 0.4169 — a
reversal worth stating plainly, because P was the weakest prefix in every arm
before the source rules and it is 69% of the training distribution. The rules
are mask-sourced and P's cases are the ones where the reads were furthest off,
so the source swap has the most to correct there.

**The four prefixes fail in opposite directions**, which the Final column hides:
S and P are precision-heavy (0.69, 0.62) and recall-poor, F is the mirror —
recall 0.58, precision **0.32**, the worst of the four by far, and for the
reason recorded in *Caveats*: F is the one prefix the template still badly
overshoots. A is weak on both halves of the caption score, which is a length and
vocabulary mismatch rather than a clinical one.

### What it did to the report text

The rules are fact-level; this is what they did to the prose. Arm 6's 40
predictions rendered twice through **current** code — once with `--facts-dir`
and once without — so unlike the fact-level table this comparison has no
template change mixed into it:

| | pre-rules | with the rules |
|---|---|---|
| sentences | 408 | **461** |
| normal / scope statements | 274 (67.2%) | **275 (59.7%)** |
| positive findings | 134 | **186** |

**More text, and a larger share of it says something.** The assert-nothing half
does not move at all — 274 sentences against 275 — while positive findings go up
by **52**: implants the mask places and the reads missed, bridges in the right
arch, fillings restored to the report, absences the FOV gate keeps. That is the
trade the rules were chosen for, showing up in the prose.

(A sentence is counted as normal/scope if it states inclusion in the volume,
normal aeration, a regular canal course, or an explicit negation; everything
else counts as a positive finding. The heuristic is crude but applied
identically to both columns.)

#### Two sentences changed, on corpus evidence

**`"No signs of bone atrophy."` — removed entirely.** Across the 40 validate
references the negative occurs **zero** times, in any wording, while 24 of the
40 mention atrophy positively. Radiologists state this finding when it is there
and say nothing when it is not, so the sentence could only ever have matched a
reference by accident. It also asserted the one thing the pipeline cannot see:
the model calls atrophy on 2 of 22 abnormal arches and **0 of 19** partially
edentulous ones. `THE RULE — alveolar bone atrophy` states the positive only
where the mask says the arch carries an edentulous region; everywhere else the
renderer is silent. (Arm 6 reads that axis far better than the arm the sentence
was cut on — 23 claims, 13 right, recall 0.59 — but the corpus argument is about
the *negative*, which no reference states at all, so it is unaffected.)

**`"No definite osteolytic or osteocondensing lesions."` — once per report,
closing the mandible.** It used to close each arch, which put **64 copies into
40 reports** — 13% of all generated text restating one claim. Unlike the
atrophy negative this IS a corpus form, and the most frequent sentence of any
kind in the training reports (70 occurrences, 208 of 933 reports), so it is
kept — but it is a whole-jaw negation there, not an arch-scoped one. The
mandible carries it because that arch is in the volume in every case in this
dataset and the maxilla is not. **64 → 39** (39 rather than 40 because
`render_bone_quality` stays silent when the model claimed a lesion, on which
see its own comment). The maxilla's `bone_quality` fact is still built and
still in the summary JSON; only the sentence is gone.

This overrides a decision recorded at the top of `synthesize_report.py` —
*"each arch's bone_quality closes that arch's own section"* — deliberately, and
the override is noted in the code beside it.

#### What is still constant

Two blocks of assert-nothing text remain, and both differ from the atrophy
negative in the way that matters — they are corpus forms, so they plausibly
earn n-gram credit rather than merely padding. Counted over arm 6's 40 reports:

| sentence | n | why it is near-constant |
|---|---|---|
| `"{Right/Left} mandibular canal with a regular course, predominantly lingual."` | 80 | `regular course` is unconditional in the template and `lingual` is now the prior — see *THE RULE — canal location* |
| scope statements | 156 | one per arch per report, plus sinus and condyle scope |
| `"No definite osteolytic or osteocondensing lesions."` | 39 | the model claims abnormal bone once in 80 arches |
| `"Maxillary sinuses: ... normally aerated."` | 20 | the model has never claimed thickening or fluid |

Roughly **160 of the 275** normal sentences would be emitted identically with no
VLM at all. Whether to cut them is a question about BLEU/METEOR and RadFact
credit for correct negatives, not about accuracy, and it wants the same
grep-the-references check the atrophy negative got before anything else is
touched.

## 3. The rules

### The ten rules, and what changed in the argument

The map. Each row has a section below with the measurement that chose it; the
two number columns are `compare_sources.py`'s SUMMARY on **arm 6**, with and
without `--facts-dir`, which is what the rules are worth on the shipping arm.

| rule | source it moves to | pre-rules | with the rule | `RULES` switch | reaches the report |
|---|---|---|---|---|---|
| absent teeth | mask list, FOV-gated | 0.82/0.75 | **0.83/0.85** | `absent_teeth` | yes |
| impaction | composite eruption + 3D orientation | 0.67/0.48 | **0.78/0.85** | `impaction` | yes |
| endodontic | composite only, drop the arch | 0.75/0.66 | **0.88/0.64** | `endodontic` | yes, teeth only |
| fillings | composite only, drop the arch alias | 0.21/0.35 | **0.30/0.24** | `fillings` | yes — renderer re-enabled, gated on the rule |
| crown | mask crown list | 0.30/0.20 | **0.29/0.31** | `crown` | no, deliberately |
| implants | mask positions, VLM clauses | 0.44/0.12 | **0.68/0.53** | `implants` | yes |
| canal-adjacent | mask `ian_close_teeth` | 0.67/0.46 | **0.69/0.69** | `canal_adjacent` | yes |
| canal location | composite unanimous, else lingual | 0.83 | **0.86** | `canal_location` | yes |
| alveolar atrophy | mask edentulous region | 0.76 | **0.78** | `atrophy` | yes |
| fixed bridges | mask bridge label, per arch | 1.00/0.09 | **1.00/0.55** | `bridges` | yes |
| *(maxilla FOV scope)* | `facts.fov.maxilla` | 0.85 | **0.88** | — (not in `RULES`) | yes |

All ten switches are `True` in `source_rules.RULES` and each can be turned off
on its own; the names match these sections so a survey diff traces back to a
paragraph. The eleventh row is not a source rule — it lives in
`postprocess_pred.py`'s arch gate and rides on the same `--facts-dir` flag.

**The measurement that CHOSE each rule is in its own section and was made on an
earlier arm.** The columns here re-score the same rules on arm 6; they do not
re-derive them. Where the two disagree the section says so — `crown` is the one
that has actually reversed, and `fillings` is the one whose margin shrank.

**The fillings renderer was split with them.** `render_restoration_summary`
silenced crown, post-and-core and fillings as one unit; it now emits fillings
and still silences the other two. The sentence is **gated on the summary
carrying `source_rules`** and fails closed — without the rule it would carry the
arch survey's aliased claims, 161 against the composite's 38 at 0.17 precision,
so a summary built without `--facts-dir` renders no filling sentence at all.
See *THE RULE — fillings*.

**The argument that changed.** The old gate used the file **only to drop**
claims it contradicted, justified by a crown list at 0.45 precision / 0.24 recall
— trustworthy for "there is nothing there", untrustworthy for "there is
something there". Eight of the rules above break that asymmetry, because it does
not hold on their axes: absence is 0.77/0.85, implants 0.68/0.53, bridges
1.00/0.55, canal-adjacency 0.69/0.69, maxilla scope 0.85 accuracy. Where the
mask is the better witness in both directions it may **add** what the reads
missed, not only veto what they invented. The asymmetry stays axis-by-axis;
nothing here licenses trusting the facts file in general, and `THE RULE — crown`
keeps a field silenced precisely because 0.33 does not clear the bar.

**Three of the ten are not source swaps at all** — canal location, alveolar
atrophy and, in effect, the arch half of fillings replace a VLM read with a
constant or a precondition, because the read carries no signal. That is a
different finding from "the mask is better", and it is the one most likely to
generalise to axes not yet examined.

### THE RULE — absent teeth

**What the rule is for: separating *absent* from *unassessable*.**
`facts.structured.teeth_absent` is the complement of `teeth_present`, and a
segmentation cannot tell those two apart. When the mask carries no tooth at a
position, exactly one of two things happened:

1. **the tooth is missing** — extracted, agenetic, or replaced by an implant,
   which is a finding and belongs in the report; or
2. **the position was never imaged** — the maxillary bone lies outside the
   volume, so the crowns above it were never in the field of view. Not imaged
   is not a finding, and the reference reports do not make one out of it.

Left ungated, `teeth_absent` reports (2) as if it were (1) and asserts sixteen
upper absences on a scan that stops at the palate. **Filtering the
unassessable positions out of the list is the whole of this rule** — everything
below is which field decides, and what the reports say in each case.

**The list.** For each case, over the 32 permanent FDIs:

```
absent(case) = facts.structured.teeth_absent
             ∩ permanent FDIs (11-48; deciduous 5x-8x are never scored)
             − maxillary FDIs (11-28)   if facts.structured.fov.maxilla == "excluded"
```

Edge cases, each with a defined answer rather than an implicit one:

| situation | frequency (validate-40 / all 622) | answer |
|---|---|---|
| `fov.maxilla == "excluded"` | 14 / 354 | **gate** — drop all sixteen upper positions |
| `fov.maxilla == "partial"` | 20 / 213 | **no gate** — gating partial collapses recall 0.86 → 0.66 |
| `fov` key missing from the facts file | 6 / 55 | **no gate** — a missing key is not evidence of exclusion |
| facts file missing entirely | 0 / 0 | fall back to today's derived list; never emit an empty list silently |
| mandible FOV | — | no gate exists; `fov` carries `condyles`, which says nothing about teeth |

`fov.maxilla` is mask-derived (`extract_facts.py` measures maxillary bone
against the volume's superior edge, `audit_facts.py` rewrites it to `excluded`
below `MIN_ARCH_BONE_MM3`), so the gate needs no VLM input and no run to
schedule. Gating on the model's own `maxilla_scope == not_included` instead
scores identically on validate-40 — the two signals select the same 14 cases —
but it is the model's answer to a question about the acquisition, and *THE RULE
— maxilla FOV scope* is why that answer is no longer trusted anywhere.

Two scripts produce every number below, both facts-and-reports only — no
inference run, so both splits are in scope: `code/studies/survey_upper_mentions.py`
reads the report text, `code/studies/absent_fov_gate_evidence.py` scores the candidate
gates. Neither writes anything.

#### First, a scoring hazard: the GT cannot referee this rule unaided

`parse_reports_to_gt` fills the positions a report leaves unstated from
`presence_enumerated` — enumerate absences and the rest become present. For a
maxilla the report declares **unassessable** that fill-in fires on a report that
enumerated nothing at all: P076's maxilla block carries `teeth_present: []`,
`teeth_absent: []`, `presence_enumerated: "absent"`, and sixteen upper positions
enter the GT as **present** on the strength of a report whose only upper sentence
is *"Maxilla partially included in the acquisition, not assessable."*

**141 arch blocks are in that state** (125 of them with `fov.maxilla ==
excluded`), and across the excluded cases **2072 of the 2491 GT-answered upper
positions — 83% — were never stated by any report.** They are exactly the
positions this rule argues about, so every number below is given twice:

| answered set | what counts as answered |
|---|---|
| **GT answered set** | every position `{case}_gt.json` settles, fill-in included |
| **report-settled** | only positions a report **names**, plus an arch it calls completely edentulous (`alveolar_atrophy == "fully_edentulous"`, which is how *"Completely edentulous maxilla"* is recorded). A maxilla called unassessable settles nothing |

The ground truth is **not** changed by any of this — `report_answered()` in
`code/studies/absent_fov_gate_evidence.py` re-derives the stricter set at scoring time
from `{case}_report_facts.json`, which is the extraction of the report text, not
the filled-in GT.

#### The evidence — what the reports actually say when the maxilla is excluded

The cleanest evidence touches no GT at all: read the 1000 reference reports and
ask whether the upper arch is mentioned. `python3 code/studies/survey_upper_mentions.py`,
classifying each report by whether it names an FDI in 11–18 / 21–28:

| `fov.maxilla` | reports | names upper teeth | arch statement only | silent on the upper arch | upper FDIs named |
|---|---|---|---|---|---|
| **`excluded`** | 588 | **58 (10%)** | **454 (77%)** | 76 (13%) | 235 |
| `partial` | 350 | 196 (56%) | 148 (42%) | 6 (2%) | 875 |
| no `maxilla` key | 62 | 50 (81%) | 12 (19%) | 0 | 358 |

The 454 arch-only reports say what P063 says and nothing more — *"Maxilla: The
maxilla is partially included in the acquisition."* Their maxilla sentences
bucket as **373 not included / not assessable**, 119 partially included, 1
edentulous. By case rather than by report, and asking specifically about
absence:

| `fov.maxilla` | cases | any report names an upper tooth | any report calls an upper tooth absent |
|---|---|---|---|
| **`excluded`** | 354 | 53 (**15%**) | 18 (**5%**) |
| `partial` | 213 | 143 (67%) | 55 (26%) |
| no `maxilla` key | 55 | 49 (89%) | 32 (58%) |

And of the 235 upper FDIs that *are* named under exclusion, only **42 (18%)**
sit in a sentence carrying absence vocabulary — the rest are restorations,
crowns, endodontic notes. **So the answer to "do the reports mention upper teeth
when the maxilla is out of the volume" is: 15% of cases mention one at all, 5%
call one absent.** A list that asserts sixteen upper absences in those cases is
not disagreeing with the radiologist, it is answering a question the radiologist
declined to answer.

Where the mask and the radiologist disagree about the scope, the mentions
concentrate — which is the honest limit of the gate:

| the report's own scope sentence, in the 354 `excluded` cases | cases | names upper teeth | names one absent |
|---|---|---|---|
| "maxilla not included / not assessable" | 170 | 13 (8%) | 6 (4%) |
| no scope sentence at all | 95 | 8 (8%) | 4 (4%) |
| **"maxilla partially included"** | 89 | **32 (36%)** | 8 (9%) |

#### The evidence — the same question scored against the GT, both ways

| answered set | `fov.maxilla` | upper positions answered | of those, absent | mask tp | mask fp | precision |
|---|---|---|---|---|---|---|
| GT answered set | **`excluded`** | 2491 | 120 (4.8%) | 63 | 831 | **0.07** |
| GT answered set | not excluded | 2671 | 627 (23.5%) | 529 | 248 | 0.68 |
| **report-settled** | **`excluded`** | **419** | 104 (24.8%) | 56 | 66 | **0.46** |
| **report-settled** | not excluded | 2150 | 627 (29.2%) | 529 | 113 | **0.82** |

**The 0.07-vs-0.68 gap was mostly the fill-in.** Corrected, the mask's upper
absence claims are 0.46 precise under exclusion against 0.82 elsewhere — still
clearly worse, still worth acting on, but roughly a halving rather than a
tenfold collapse. The "4.8% of answered upper positions are called absent"
statistic goes with it: on report-settled positions the share is 24.8% against
29.2%, i.e. **when a radiologist does commit to an upper position in an excluded
volume, they call it absent about as often as anywhere else.** What changes
under exclusion is not the rate, it is the volume: 419 upper positions settled
over 354 cases (1.2 per case) against 2150 over 268 (8.0 per case).

#### The three cases this rule was checked against

**1 — maxilla excluded, mask carries no upper tooth at all. Gate: correct a
priori, and there is no report evidence either way.** Seven cases (P063, P076,
P140, P330, P370, P505, P512) putting 112 upper positions in `teeth_absent`.
Between them the reports settle **one** upper position — P330's tooth 26, a
restoration, i.e. present — and call none absent. The 64 "present" labels this
bucket used to be scored against were all fill-in. So the gate is kept here on
the a priori argument alone: a report that says *"the maxilla is partially
included in the acquisition"* and stops has not licensed sixteen absence claims.

**2 — maxilla excluded, mask carries some upper teeth. Gate anyway.**
Report-settled, over the 354 excluded cases:

| upper positions the mask calls absent | cases | claims | settled | of those, absent | tp | fp | precision |
|---|---|---|---|---|---|---|---|
| 16 — mask empty (case 1) | 7 | 112 | 1 | 0 | 0 | 1 | — |
| 9–15 | 102 | 1212 | 60 | 24 | 21 | 17 | 0.55 |
| 5–8 | 83 | 543 | 122 | 12 | 10 | 40 | 0.20 |
| 1–4 | 61 | 159 | 169 | 40 | 25 | 8 | 0.76 |
| 0 — mask carries the whole upper arch | 101 | 0 | 67 | 28 | — | — | — |

The proposal that a partially-populated upper arch means the absences it does
report are real is **not supported**: the stratum precisions are 0.55 / 0.20 /
0.76 on 38, 50 and 33 scored claims — small, non-monotonic, and only the
1–4 stratum reaches the 0.82 that upper claims score outside an excluded
maxilla. Note also the last row: **101 of 354 excluded cases carry the whole
upper arch in the mask**, so exclusion does not imply an empty absence list at
all, and the gate has to be stated on `fov.maxilla` rather than on the shape of
the list.

**3 — maxilla included or partial, all sixteen upper positions absent. Do not
gate: that is edentulism.** Nine cases. Report-settled, six of them settle
nothing upper; **A018 and A023 settle all sixteen as absent** — both reports
state complete maxillary edentulism outright — and S0023 settles 16 with 10
absent. Gating this bucket costs F1 **0.858 → 0.850** on 622 cases and
**0.870 → 0.828** on validate-40. It is the one configuration where the mask's
empty upper arch is a finding, and A018 is the same case *THE RULE — maxilla
FOV scope* exists to stop the pipeline from deleting.

#### Candidates, scored on all 622 cases

Facts-only lists, so both splits are usable. **The report-settled column is the
one to read**; the GT column is kept beside it to show what the fill-in was
doing.

| candidate | report-settled prec / rec / F1 | GT answered set prec / rec / F1 |
|---|---|---|
| **`teeth_absent` − UPPER if `fov.maxilla == excluded`** (shipped) | 0.857 / 0.860 / **0.858** | 0.805 / 0.845 / 0.824 |
| ... and also if the mask carries no upper tooth (case 3 gated too) | 0.857 / 0.843 / 0.850 | 0.818 / 0.829 / 0.824 |
| − UPPER only if excluded **and** the mask carries no upper tooth (case 1) | 0.839 / 0.881 / 0.860 | 0.637 / 0.869 / 0.735 |
| − UPPER if excluded, except where ≤4 upper positions are absent | 0.856 / 0.869 / **0.863** | 0.787 / 0.855 / 0.820 |
| − UPPER unless `fov.maxilla == included` (gate `partial` too) | 0.868 / 0.655 / 0.746 | 0.853 / 0.645 / 0.734 |
| `teeth_absent`, ungated | 0.839 / 0.881 / 0.860 | 0.626 / 0.869 / 0.728 |

**At fact level the gate is a wash, and that has to be said plainly.** Shipped
0.858 against ungated 0.860: it trades 56 true positives for 66 false ones on
the positions the reports actually settle. The 831-false-positive figure the GT
column reports is 765 fill-in labels plus those 66. The ≤4-stratum variant
scores 0.863, ahead by 0.005 on 33 scored claims and exactly tied with shipped
on validate-40 (0.870) — noise, bought with a second condition on the rule, so
it is not adopted.

**What the fact-level score cannot see is why the gate stays.** Of the 2026
upper positions the mask calls absent in excluded cases, only 122 are ever
settled by a report; the other **1904 land on positions no reference report
discusses**. Fact-level scoring drops them as unanswered — RadFact does not. It
scores every *sentence* the pipeline emits against the reference, so an ungated
maxilla emits *"Absence of teeth 11, 12, 13 …"* on a mandible-only scan and is
charged for all of it. The gate's value is in report precision on positions the
fact-level view scores as free, plus the a priori point that a structure outside
the volume is not imaged and not-imaged is not a finding.

Gating `partial` remains wrong for the same reason it always was: recall
0.860 → 0.655. Partial means *some* upper teeth are outside the volume, and the
mask is right about the ones inside it.

**The gate.** A per-tooth finding asserted at an FDI in `absent(case)` is
dropped, whatever source claimed it — nothing can be restored, root-filled,
carious or periodontally involved if it is not there.

Suppressed at an absent FDI:

- caries / defect, fillings, crowns, post-and-core
- endodontic treatment and filling quality
- per-tooth periodontal status and per-tooth bone quality
- canal adjacency for that tooth

**Exempt — never suppressed**, because each is a real finding *at a position
with no tooth*, and suppressing them is how a gate turns into a bug:

- **implants** — an implant sits exactly where a tooth is missing
- **bridge spans and pontics** — a pontic spans an absent position by definition
- **root remnants** — the tooth is legitimately reported absent while a
  fragment remains in the bone and is still visible; a remnant FDI keeps its
  other findings too
- **impacted / unerupted teeth** — present but not erupted; the mask may miss
  them, and 12 of this list's 39 false positives on validate-40 are wisdom
  positions
- **arch- and volume-level facts** — alveolar bone atrophy, per-arch bone
  quality, maxilla scope, sinus findings, canal position

**Where it applies.** The rule is a post-pass, so `postprocess_pred.py`'s own
derivation still runs first and `source_rules` writes over it. Three sites, one
list:

| site | what it does | under the rule |
|---|---|---|
| `postprocess_pred.collect_arch_absent()` | derives absence from `detected == "no_image"` (0.50 prec), then `eruption_state`, then the arch map | unchanged, then **overwritten** by `apply_absent` with `absent(case)`, `source: "facts.teeth_absent, FOV-gated"` |
| `postprocess_pred.toothless_fdis()` | gates on that derived list | unchanged, then `_gate_on_absent` re-gates on `absent(case)`, which is the stricter pass |
| `validate_summary_with_facts.py` `absent` rule | gated on raw `teeth_absent`, ungated | **not in the path** — that gate is reached only by `code/arms/postprocess_val_now.sh`, the older arm-building script |

**The gate runs last, and that ordering is load-bearing.** It sat inside
`apply_absent` first — i.e. before the rules that re-source the per-tooth
findings — so a filling the gate had just dropped at an absent position was
written straight back by `apply_restorations`, and A019's maxilla reported
*"Absence of 14, 15, …"* and *"composite restoration(s) on teeth 15 and 27"* in
the same paragraph. A gate is only a gate if nothing writes after it.

The exemptions are not new: `toothless_fdis()`'s docstring already listed root
remnants, impaction, implants, bridge spans, atrophy and bone quality as
deliberately not suppressed, and the old gate exempted implants and pontics "by
construction". `GATE_EXEMPT` adopts both verbatim and changes only which list
they read.

Both directions of the facts file are now in use, which is the change of
position worth being explicit about. The old gate used it **only to drop** claims
it contradicted, and justified that asymmetry with a crown list at 0.45 precision
/ 0.24 recall — trustworthy for "nothing there", untrustworthy for "something
there". Absence does not behave like crowns: at 0.77/0.85 it is the *better*
witness in both directions, so the asymmetry has no basis on this axis and the
mask may **add** absences the reads missed, not only veto ones they invented.
The rule stays axis-by-axis; nothing here licenses trusting the facts file
generally.

The gate itself was already the old `absent` rule, and its justification there
was made a priori ("tautological — the tooth was never segmented"). What is new
is that the reported absence list changes source, that the FOV filter states
which of the two meanings a mask silence carries, and that the gate sites now
run off one corrected list instead of three different ones.

#### The rest of the evidence — where the list itself comes from

Absence is the axis with the most sources and the most ways to go wrong, so it
carries more working than any other rule. Three parts remain: what each source
scores, where the shipped pipeline's absence actually came from, and why that
is not the facts list. All three are validate-40, the split with predictions.

#### The evidence — the mask beats every read

| source | dom | claims | tp | fp | fn | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| **facts** (mask) | 232 | 255 | 197 | 58 | 35 | 0.77 | **0.85** | **0.81** |
| 3d (wisdom slots only) | 48 | 62 | 45 | 17 | 3 | 0.73 | 0.94 | 0.82 |
| panoramic (arch survey) | 216 | 207 | 159 | 48 | 57 | 0.77 | 0.74 | 0.75 |
| PRED (all reads) | 232 | 220 | 167 | 53 | 65 | 0.76 | 0.72 | 0.74 |
| SUMMARY (derived, *not* a vote — see below) | 232 | 228 | 190 | 38 | 42 | **0.83** | 0.82 | **0.83** |

**The mask dominates the raw prediction outright** — better on precision *and*
recall at once (+0.01 / +0.13), so there is no operating point at which the
VLM's absence read is preferable. That direction is not a surprise: absence is
a segmentation question, and the mask is what every image in this pipeline was
rendered *from*. The reads are guessing at something `facts.json` already knows.

The `3d` column is the strongest single read (F1 0.82) but speaks for 4 of 32
positions; the domain column is what stops that from being read as a general
result.

Unioning the panoramic read back into the gated list buys 2 true positives for
6 false ones (0.79/0.86, F1 0.82); intersecting with it costs 40 true positives
(0.82/0.68); the read alone is 0.77/0.69. Nothing beat the facts list.

#### Where the SUMMARY column's absence actually comes from

`SUMMARY` beats both the mask and every read on this axis (0.83/0.82), which
looks like the cross-source vote doing unusually good work. It is not a vote.
Absence is not asked anywhere — schema v6.1 dropped `absent_teeth_{arch}` — so
`postprocess_pred.collect_arch_absent()` *derives* it, and the chain runs
through a signal the `PRED` column cannot see:

1. `run_vqa_inference.model_absent_teeth()` reads
   `dental_arch_findings_{arch}` — the panoramic read — into a set of
   model-declared-absent FDIs.
2. Every tooth is then stamped with a `detected` flag: `yes` if a composite
   call was actually made, `no` if the panoramic declared it absent and no call
   exists, `no_image` if no composite was ever generated.
3. `collect_arch_absent()` derives absence in strength order —
   `no_image` first, then the tooth's own `eruption_state`, then the arch map
   last.

`pred_absent()`, which feeds the `PRED` column, reads only the *global* dict:
the arch findings map and the four wisdom `eruption_state`s. It never looks at
`teeth[*].detected`. So `SUMMARY` is not `PRED` reconciled — **it has an input
`PRED` does not**, and that is the whole 0.74 → 0.83 F1 gap.

Measured over the 40 cases, on GT-answered positions:

| rule | claims | tp | precision |
|---|---|---|---|
| `detected == "no"` | 181 | 150 | **0.83** |
| `detected == "no_image"` | 80 | 40 | 0.50 |
| `eruption_state == "absent"` | 0 | — | — |
| arch findings map alone | 0 | — | — |

The two live rules are **not** a model source and a mask source. Checked
against the rendered PNGs, **all 261 teeth under both flags have no composite
crop** — the flags differ only in whether the panoramic read *also* called the
tooth absent, because `run_vqa_inference` assigns `no` before `no_image` and
the first loop consumes the agreeing teeth. So the derived absence is one
mask-side signal split by panoramic confirmation: confirmed 0.83, unconfirmed
0.50. The per-tooth `eruption_state` and the arch map contribute nothing on
this run — everything they would catch is already caught upstream.

#### The pipeline's mask signal is not `facts.teeth_absent`

They are close enough to be mistaken for each other and they are not the same set:

| | teeth |
|---|---|
| in both | 346 |
| **no composite generated, but NOT in `facts.teeth_absent`** | **153** |
| in `facts.teeth_absent`, but a composite was generated | 12 |

The two sets disagree in **20 of 40 cases**. "No crop was produced" is a
*rendering* outcome — `create_tooth_detail.py` declining to emit a PNG — while
`facts.structured.teeth_absent` is the *segmentation* fact, the complement of
`teeth_present`. The 153 over-claims are teeth the mask carries and the
renderer skipped, and they are where the precision goes: the unconfirmed bucket
sits at 0.50 against the facts list's 0.77.

**This sharpens the decision rather than weakening it.** The pipeline is
already leaning on a mask-side signal for absence and getting a good number out
of it; it is simply leaning on the *wrong* one — a proxy for the segmentation
rather than the segmentation, and a proxy with no FOV information in it at all.
Sourcing absence from `facts.teeth_absent` replaces a 0.50-precision proxy in
the unconfirmed half, and the panoramic confirmation that produces the 0.83
bucket can stay exactly as it is.

**What remains is mostly not the list's fault.** On validate-40 the gated
list's 39 false positives split wisdom 12 / upper 15 / lower 12, and every one
of them is a position where the GT says a tooth is there and the segmentation
says it is not — and the segmentation is what every image was rendered from, so
the VLM saw no tooth either. The 35 false negatives (wisdom 3 / upper 12 /
lower 20) are the real misses. Chasing precision past 0.83 on this axis means
arguing with the label.

### THE RULE — impaction

**Scope: 18, 28, 38, 48 only.** Nothing else is scored on this axis — all 33
impaction positives in the generated GT are wisdom teeth — and no other FDI has
a fact that could answer it.

**The definition** is the schema's own `_definitions.impacted`, unchanged:

```
impacted = (eruption_state ∈ {complete_bony_inclusion, partially_erupted})
           OR (orientation ∉ {normal, null})
absent   -> no claim
```

**What changes is where the two inputs come from.** The schema tells the model
to derive `impacted` itself ("DERIVED from the two answers above … do not judge
it separately"), and the pipeline takes that bool at face value. It should
derive the bool in postprocess instead, from:

| input | source | why |
|---|---|---|
| `eruption_state` | `tooth_{fdi}_eruption` (**composite crop**), falling back to the 3D wisdom fact where no composite exists | the composite is the better read — see below |
| `orientation` | the 3D wisdom fact (`lower_left_wisdom_tooth.orientation` etc.) | the composite carries no orientation field at all |

Scored on the 66 wisdom slots where the GT commits to `impacted` (33 true /
33 false); a rule that makes no call counts as "not reported", which is what
the report shows:

| rule | tp | fp | fn | tn | prec | rec | F1 | acc |
|---|---|---|---|---|---|---|---|---|
| **C — derive; composite `eruption_state` preferred, 3D `orientation`** | 25 | 9 | 8 | 24 | **0.74** | **0.76** | **0.75** | **0.74** |
| E — derive from the composite's `eruption_state` alone | 20 | 5 | 13 | 28 | 0.80 | 0.61 | 0.69 | 0.73 |
| B — derive from the 3D fact's own `eruption_state` + `orientation` | 17 | 8 | 16 | 25 | 0.68 | 0.52 | 0.59 | 0.64 |
| A — the 3D fact's `impacted` bool, as the pipeline uses it today | 14 | 7 | 19 | 26 | 0.67 | 0.42 | 0.52 | 0.61 |

**F1 0.52 → 0.75, for +11 true positives against +2 false ones.** Two separate
effects, and it is worth keeping them apart:

1. **The model does not reliably apply the rule it was given.** B beats A
   (0.59 vs 0.52) using *the same fact's own fields* — re-deriving `impacted`
   from the `eruption_state` and `orientation` the model wrote is better than
   the `impacted` it wrote alongside them. A derived field is only as good as
   the model's arithmetic on its own answers, and this is what
   "field ORDER in `object_fields` is load-bearing" is protecting against;
   ordering makes the derivation *possible*, it does not make it *right*.
2. **The composite is the better read of eruption**, which is the same result
   the source table shows on every other axis the composite can see:

   | eruption_state, excluding `absent` slots | 3D wisdom fact | composite crop |
   |---|---|---|
   | exact enum vs GT | 0.24 | **0.42** |
   | binary (unerupted vs fully erupted) — what the rule actually uses | 0.43 | **0.64** |

   The rule needs only the binary partition, which is why the composite helps
   more here than its enum accuracy suggests.

The 3D fact still supplies `orientation` — the composite has no such field —
and supplies `eruption_state` for the **95 of 160** slots with no composite at
all, so both sources stay in the derivation. Adding the panoramic as a
tiebreak changes nothing (its 3 impaction claims are a subset of the 3D
fact's 32), and suppressing claims where the mask calls the tooth absent also
changes nothing on these slots.

**Caveat on size.** 66 committed slots, 33 of them positive. C over A is +11
true positives, which is a real difference; C over E is 5 true positives
against 4 false ones, which is not — do not read the ordering of C and E as
settled.

### THE RULE — endodontic treatment

**Use the composite. Do not union.**

```
endodontic(case) = { fdi : tooth_{fdi}_morphology.with_endo is true }
```

The panoramic's `root_canal_treatment` value is **not** read on this axis.
`filling_quality` continues to be carried in the summary JSON and not rendered
(see *status*, below).

All five combination strategies, scored over the same 90 GT positives so the
denominators are comparable:

| strategy | claims | tp | fp | prec | rec | F1 |
|---|---|---|---|---|---|---|
| **composite only** | 78 | 60 | 18 | **0.77** | 0.67 | **0.71** |
| composite where cropped, panoramic elsewhere | 78 | 60 | 18 | 0.77 | 0.67 | 0.71 |
| union of both — **what ships today** | 100 | 62 | 38 | 0.62 | 0.69 | 0.65 |
| both must agree | 24 | 22 | 2 | 0.92 | 0.24 | 0.39 |
| panoramic only | 46 | 24 | 22 | 0.52 | 0.27 | 0.35 |

**What the union actually buys: +2 true positives for +20 false ones.** The 22
claims the panoramic contributes beyond the composite are right 2 times out of
22 — marginal precision **0.09**. Its headline 0.52 looks tolerable only because
22 of its 24 correct claims are ones the composite already made, so on this axis
it is very nearly a duplicate of a better read plus noise.

**The impaction-style fallback is a no-op here**, and that is the interesting
part: "composite where cropped, panoramic elsewhere" scores *identically* to
composite-only, because the panoramic never once catches an endodontic tooth
that had no crop. An endodontically treated tooth is present and therefore
segmented and therefore cropped, so there is no coverage gap for the arch
survey to fill. It is not extending reach, only contradicting a better-placed
read at positions already answered.

**Why the shipped number looks like a recall loss and is not.** In the source
table the composite reads 0.77/0.77 and the summary 0.62/0.69, which invites
the reading that reconciliation costs recall. Recall there is domain-scoped —
the composite's 0.77 is over the 78 GT positives it was shown, the summary's
0.69 over all 90. On one denominator the union is 2 recall points *ahead*
(0.67 → 0.69). The precision collapse is the real effect; the recall difference
is the denominator.

**Cost of adopting it.** Endodontic is `on`, so this is 2 true positives out of
the report text against 20 false ones — the only rule here that removes
correct claims, which is why it wants `NO_RADFACT=1` confirmation rather than
fact-level numbers alone. It needs no new fact and no re-run: it is one source
dropped from an existing union, and `CROSS_VALIDATE_ENDODONTIC` is not the
switch — that gate votes between the two sources, where this removes one.

### THE RULE — fillings

**Keep the composite's answer exactly as it is. Do not fold the arch in, do not
merge the restoration types, and turn the renderer back on.**

```
fillings(case) = { fdi : tooth_{fdi}_morphology.with_fillings is true }
```

**0.39 precision / 0.20 recall.** What ships today is 0.17/0.36, and the gap is
one line of aliasing.

**The pipeline already merges restoration types into fillings, and that is the
whole precision problem.** `ARCH_VALUE_ALIASES` maps the arch survey's
`restoration` onto the internal token `filling`, so every tooth the panoramic
calls restored is filed as a filling:

| | claims | tp | fp | prec | rec |
|---|---|---|---|---|---|
| composite `with_fillings` alone | 38 | 15 | 23 | **0.39** | 0.20 |
| + arch `restoration` folded in — **ships today** | 161 | 27 | 134 | 0.17 | 0.36 |

**+12 true positives for +111 false ones.** The fold is unsound by construction:
v7.1's arch vocabulary has no filling value *because* a crown and a large
filling are one bright capped tooth at that resolution, so `restoration` is
precisely the claim that cannot be resolved into a filling — and the alias
resolves it anyway.

**Merging the composite's own types is no better**, measured three ways against
the GT filling list:

| prediction relabelled as "fillings" | claims | tp | prec | rec |
|---|---|---|---|---|
| composite `with_fillings` only | 38 | 15 | **0.39** | 0.22 |
| crown + fillings | 63 | **15** | 0.24 | 0.22 |
| crown + fillings + post-and-core | 79 | **15** | 0.19 | 0.22 |

**True positives do not move — 15 in all three.** Of 74 GT fillings the
composite calls 0 of them "crown" and 0 "post_and_core"; it either says filling
(15) or says nothing (59). There is no misfiled filling to recover, so every
tooth merged in is a false positive and nothing else changes. The confusion in
this family runs the other way — 4 of 49 GT crowns are called fillings and 6
post-and-core — which is why merging raises *crown* recall (0.21 → 0.47) and
never fillings'. The dominant error is silence, not mislabelling, and no
relabelling reaches it.

**Turn it on.** At 0.39 precision it clears the ≥0.35 bar that every other
reported finding meets (see *The silencing rule, and the ≥0.35 bar*), where the shipped 0.17 does not — the
finding was silenced as part of a renderer whose other two members are genuinely
below the bar, not on its own evidence.

Implementation note: `render_restoration_summary` silenced crown, post-and-core
and fillings **as one unit**, so turning fillings on meant splitting that
renderer rather than flipping a flag. Done — crown (0.28/0.10) and post-and-core
(0.07/0.24) stay silenced, and the fillings sentence is gated on the summary
carrying `source_rules`, because rendering it *without* the rule is what puts
the 134 false claims back.

### Crown categories — the ground truth

A crown is four different findings wearing one word, and the pipeline scores
them as one. The GT split, over the 49 crown positions in validate-40 (14 of
the 40 cases carry any crown at all):

| category | GT crowns | what the GT field is |
|---|---|---|
| **endo + crown** | **24** | in `with_full_crown` and in `with_endo` |
| **other crown** | **16** | in `with_full_crown`, no other tag |
| **implant + crown** | 5 (+1 also in a bridge) | in `with_full_crown` and at an implant FDI |
| **crown in bridge** | 3 (+1 also on an implant) | in `with_full_crown` and named in a bridge span |

Nearly a clean partition: 16 positions carry no tag, 32 carry exactly one, and
**1** carries two (an implant abutment inside a bridge).

**Endo+crown is half of all crowns**, and that is the opening. Endodontic is the
composite's *best* axis (0.77/0.77) and crown one of its worst (0.32/0.21), so
24 of the 49 crowns sit on teeth the pipeline already identifies well.

**The implant row has two possible denominators and they are not the same
question.** Implant crowns are carried by `implants[].with_crown`, a different
field from the per-tooth crown fact:

| | |
|---|---|
| implants with `with_crown: true` | **11** |
| of those, also in `tooth_{fdi}_morphology.with_full_crown` | **6** |

Five implant-supported crowns are invisible to the per-tooth fact, because an
implant position usually has no natural tooth and therefore no
`tooth_{fdi}_morphology` block to carry the flag. Scoring the implant row
against 11 measures whether the pipeline finds implant crowns; scoring against
6 measures whether the crown fact finds them, which it structurally cannot do
for the other 5.

The same asymmetry applies to bridges, smaller: the GT enumerates
`abutment_teeth`, `pontic_teeth` and `implant_supported_teeth` separately, and a
pontic is a crown-shaped unit with no tooth beneath it, so no per-tooth crown
fact will ever claim one.

**Open, before the prediction side of this table is built:** which denominator
the implant row uses (11 or 6), and whether the bridge row counts abutments
only (3) or abutments plus pontics.

### THE RULE — crown

**Source it from the mask. Leave it silenced.**

```
crown(case) = facts.structured.crowns
```

**0.33/0.29**, against 0.32/0.21 for the composite and 0.28/0.10 for what
postprocess produced before the rule. The mask is the best of the three on both
axes at once, and it is the only source here that needs no VLM read.

> **THE MARGIN HAS SINCE CLOSED, AND THE RULE IS NOT SETTLED.** On arm 6 the
> composite reads **0.32/0.32** against the mask's unchanged 0.33/0.29 — level
> on precision, ahead on recall — so the mask is no longer the better source on
> both axes at once, and applying the rule takes the summary from 0.30/0.20 to
> 0.29/0.31: 0.11 recall bought with 0.01 precision, on the axis with the
> tightest precision bar in the document. 49 validate positions cannot separate
> two sources this close, and a larger split has previously reversed this axis
> outright. Nothing that ships is affected — the crown renderer is silenced
> either way — but `RULES["crown"] = False`, or a re-point at the composite, is
> a live option and should be decided on the training split before crown is
> ever un-silenced.

**`status` stays `off`.** At 0.33 precision this is below the ≥0.35 floor every
reported finding meets, so the rule improves the summary JSON and reaches no
report text. That is deliberate and it is not pointless: `crown` is an input to
the bridge/implant reasoning, and the summary JSON is the instrument that would
later show an improvement worth turning on.

Note this rule reverses the direction of the old gate's `crown` rule rather than
enabling it. That rule *dropped* a composite crown claim the mask did not
confirm, was left **off**, and was a no-op — 472 drops that changed no report
text, because crowns are silenced downstream either way. Sourcing the field from the mask outright makes the drop
direction redundant, and does the half a drop could never do: the mask's 29
crowns that no composite claimed are added rather than discarded.

The drop-only gate quoted this list at "0.45 precision / 0.24 recall" against
the reference reports. Scored against the generated ground truth it is 0.33/0.29 —
a different label, not a changed list, and the reason the number moved is worth
keeping in view when comparing the two sections.

**Crowns are not one finding.** The 49 GT crown positions split four ways, and
the categories behave differently enough that a single row understates what is
knowable — see *Crown categories* below.

### THE RULE — implants

**Positions come from the mask. The VLM keeps the two clauses the mask cannot
carry.**

```
implant positions(case) = facts.structured.implants
per-implant clauses     = the VLM's with_crown / osseointegration_status,
                          carried over where its fdi_number matches a mask position
```

**0.68 precision / 0.53 recall, against 0.00/0.00 for what ships.** This is the
largest single gap on the whole table, and the only `on` row whose source is
wrong every time it speaks.

**The VLM does not find implants. It finds their absence.** The survey's
quadrant counts read 0.91 and 0.97, which looks like a working detector and is
not — 129 of 146 quadrants contain no implant at all:

| quadrant implant count | right / scored | acc |
|---|---|---|
| overall | 127 / 146 | 0.87 |
| where the GT count is **0** | 125 / 129 | **0.97** |
| where the GT count is **> 0** | 2 / 17 | **0.12** |

So the counts and the `implants` list at 0.00/0.00 are not in tension: they are
the same failure, one of them disguised by the base rate. The `bridges` rows
behave identically — exact-span tp 0 / fp 5 / fn 16, and only **2 of 16** GT
bridges are even *overlapped* by a predicted span, while `present` scores
0.33/0.12 and 1.00/0.12.

**And the wrong positions are not near-misses.** Of the 18 false-positive
implant positions, 14 sit more than two positions from the nearest true
implant, 3 are off by one, and 1 is in an arch with no implants at all. This is
not FDI drift that a neighbour-tolerant match would forgive; the model puts
implants in the wrong region.

**Why this is not the old gate's `implant` rule, which was measured harmful.**
That rule *dropped* a predicted implant the mask did not confirm, and it cost
F1 0.474 → 0.372, which is why it was left off. Dropping cannot repair a source whose failure is
misplacement — it removes the 18 wrong positions and cannot supply the 32 true
ones the mask already has. Sourcing inverts that: the mask's positions are used,
and the drop rule becomes redundant. The same asymmetry break as absence and
crown.

**What the mask cannot carry.** `facts.structured.implants` is a bare FDI list,
while `render_implants` writes two optional clauses — *"restored with an
implant-supported crown"* (`with_crown`) and the osseointegration clause
(`well` / `poor`). Neither is derivable from a segmentation, so they stay with
the VLM and ride along wherever its `fdi_number` matches a mask position. Where
the VLM named no implant at that position, the sentence is the bare form,
which is the corpus's own most common phrasing.

**Caveat on the number.** 0.68/0.53 is flat per-FDI matching against the
generated GT. `survey_findings.py` scores implants under *slot* semantics
instead — one slot per implant, satisfied by a claim anywhere in its region —
because reports place implants by region as often as by tooth ("in the
1.5-1.6-1.7 region"). Flat matching is the stricter of the two and is what the
0.68 is measured under; the slot number would be no lower.

**Instrument fix that went with this.** `survey_facts._hashable` keyed list
items on `fdi` / `tooth` / `position`, none of which is the name the schema
gives an implant — it is `fdi_number`. Every implant therefore fell through to
the whole-dict comparison the function exists to avoid, and the same implant
failed to match itself as soon as `location` was worded differently or
`osseointegration_status` was null. Now keyed on `fdi_number` too, and bridges
on their ordered `(span_start, span_end)`. **It changes no number in this
document** — re-scored, implants are still 0 tp / 18 fp / 32 fn — but it was
capping both rows at zero regardless of what the pipeline did, so no
improvement to either could ever have shown up.

### THE RULE — canal-adjacent teeth

**Take the list from the mask. Read neither VLM source.**

```
canal-adjacent(case) = facts.structured.ian_close_teeth
```

**0.69 precision / 0.69 recall**, against 0.58/0.54 for what ships and
0.40/0.31 and 0.18/0.36 for the two reads that feed it.

| source | claims | tp | fp | fn | prec | rec |
|---|---|---|---|---|---|---|
| **facts `ian_close_teeth`** | 26 | 18 | 8 | 8 | **0.69** | **0.69** |
| 3D canal trace (`mandible_canal_{side}.adjacent_teeth`) | 20 | 8 | 12 | 18 | 0.40 | 0.31 |
| composite (`tooth_{fdi}_mandible_canal`) | 51 | 9 | 42 | 16 | 0.18 | 0.36 |
| both reads unioned | 66 | 14 | 52 | 12 | 0.21 | 0.54 |
| after the current vote — **ships today** | 24 | 14 | 10 | 12 | 0.58 | 0.54 |

The mask claims 26 teeth and gets 18 right; the two reads together claim 66 and
get 14 right. This is the one axis where the mask is not merely better but
*measured on the same quantity the model is asked to eyeball* — proximity of a
root apex to the canal is a distance between two segmented structures, so the
mask computes what the VLM estimates by eye.

**The composite is the weakest source here, which is worth noting because it is
the strongest almost everywhere else.** 51 claims for 9 hits: asked tooth by
tooth "is this root near the canal", it says yes far too readily, and the
schema's own `tooth_{fdi}_mandible_canal` is restricted to 36-38/46-48 precisely
because that is where the question is meaningful. Its 0.18 is the lowest
precision of any source on any axis in this document.

**Relation to the old gate's `canal` rule, which was on.** That rule dropped a
predicted adjacency not in `ian_close_teeth`, and was measured at precision
0.50 → 0.75 for unchanged recall — the drop direction of exactly this list,
earning its keep before anything was re-sourced. Sourcing takes
the other direction as well: the 8 teeth the mask names and both reads missed
are added rather than left out. It also subsumes the drop direction entirely,
since a list sourced from `ian_close_teeth` cannot contain anything
`ian_close_teeth` would drop.

**This one reaches the report.** `render_canal` appends
`_adjacent_teeth_clause(entry)` to the canal sentence — the corpus frame
*"Right mandibular canal with a regular course, predominantly lingual"* plus the
adjacency clause — so `status` is `on` and the change is visible in the text of
every case with a mandibular canal.

### THE RULE — fixed bridges

**Presence and arch come from the mask's bridge label. No span is claimed.**

```
bridge_arches(case) = the arch each connected component of mask label 8 sits in
```

**Precision 1.00 / recall 0.44 per arch** — 7 arches named, **0 of them wrong**,
9 missed — against 0.67/0.18 per *case* for what ships. Per-arch is also
strictly more than the pipeline carries today, whose facts-side input is a bare
`bridge_present` bool for the whole mouth.

| case | mask-derived | reference |
|---|---|---|
| A041, F014, S0000 | maxilla / both / mandible | **match** |
| A037, F067, P014 | one arch | both arches — one missed |
| A022, F003, F015, F043, P345 | — | **no label 8 in the mask at all** |

**The ceiling is the segmentation, not the derivation.** Five of the eleven
bridge cases carry no bridge label, so nothing downstream can recover them.
That is also why `facts.bridge_present` reads 1.00/0.55 — it is exactly "label
8 exists in this case", and the arch split inherits its recall.

**Why no span, having tried.** Label 8 marks the PONTIC, not the abutments, so
the component's extent is the middle of the bridge and never its ends. Closing
it by dilation lands within about one position and not reliably on it: F067's
pontics touch 14 and 16 where the report says *"splinted crowns from 1.4 to
1.7"*; P014's touch 32 and 42 where the report says *"from 3.3 to 4.3"*.
Absorbing the abutment CROWN label (9) first resolves two more cases and then
overshoots on F014. The error is ±1 with no consistent sign, so an exact span
would be wrong more often than right. Scored exactly, mask spans and VLM spans
both round to zero — the difference is that the arch is never in doubt.

That is not a loss against the current pipeline, which claims spans and gets
**0 of 16** exactly right with only **2 of 16** even overlapped. It is also
consistent with `survey_findings.py` scoring bridges per case on presence
alone, because the reports describe spans too variably to match tooth for tooth
(*"circular fixed prosthetic rehabilitation"*, *"crowns splinted as a bridge"*).

**Implemented.** `audit_facts.py --derive-bridge-arches` writes
`structured.bridge_arches` into every audited facts file, and
`source_rules.apply_bridges` sources the summary from it. **The field is written
even when empty**, so "audited, no bridge label" is distinguishable from "never
audited" — while it was not, P397's false-positive bridge survived, because the
rule could not tell a case with no bridge from a case nobody had looked at.

**The sentence names no span**: `"Prosthetic bridge exists."` The mask's label
marks the pontic and not the abutments, and the VLM's spans are worse than
nothing here — 0 of 16 exact against the reference, and 0 of 2 among the spans
that survive a per-arch gate (A041 says 22-23 against a reference 21-23;
S0000 says 33-37 against 31-42 and 43-45). Carrying a surviving span would
import a known-wrong one to gain nothing measurable. It is behind its own flag because it is **the first thing
that tool adds rather than removes**, against a docstring that promised it only
ever deletes what the mask cannot support.

Two mechanics worth keeping: the arch comes from whatever the dilated pontic
touches — a tooth label settles it, else the jawbone label under it — and where
an implant-borne bridge stands clear of both (F014, S0000), the fallback is
whichever jawbone centroid is nearer. That fallback resolved both correctly and
is the only geometric inference in the tool.

Run on `dataset/training/facts` as well as validate — all 582 files carry
`structured.bridge_arches`. On 122 training bridge cases the rule scores
**0.81/0.79 per case** against the panoramic's 0.73/0.20: the 1.00 precision
validate showed was 7 arches and did not survive, but the recall is four times
the read's, which is the result that matters.

### THE RULE — alveolar bone atrophy

**State atrophy when the mask says the arch is fully edentulous. Otherwise say
nothing. Do not read the model's `atrophy` at all.**

```
atrophy(arch) = every position in that arch is in absent(case)      # THE RULE -- absent teeth
```

**Precision 1.00 / recall 0.23, F1 0.37**, against 0.40/0.09 and F1 0.15 today.

| strategy | claims | tp | fp | prec | rec | F1 |
|---|---|---|---|---|---|---|
| **mask says fully edentulous → state atrophy** | 5 | 5 | **0** | **1.00** | **0.23** | **0.37** |
| gate the model's `atrophy` on mask edentulism | 2 | 2 | 0 | 1.00 | 0.09 | 0.17 |
| gate the model's `atrophy` on its own `fully_edentulous` | 3 | 2 | 1 | 0.67 | 0.09 | 0.16 |
| the model's `atrophy`, ungated — **ships today** | 5 | 2 | 3 | 0.40 | 0.09 | 0.15 |

**The model's judgement is dropped entirely, and that is what makes it work.**
Gating its `atrophy` keeps only 2 of the 5 arches the prior gets right, because
it declines to call atrophy on 3 edentulous jaws that have it. Consulting it
strictly loses information here.

**Why an edentulous jaw is enough on its own.** The alveolar process exists to
hold teeth and resorbs once they are gone, so "no teeth in this jaw" implies
bone loss as a matter of anatomy rather than of image reading. The data agrees
without exception on this split: 5 fully edentulous arches by the mask, 5 with
atrophy in the reference, **0 false positives**.

**`present` is deliberately not a rule.** It is a precondition — *"at least one
stretch of 3+ missing teeth in a row"* — not a finding, and the report never
states it; only `atrophy` reaches the text. It is also pure arithmetic over the
absence list (deriving it scores 1.00/0.66 against the model's 1.00/0.16), so if
it is ever needed it should be computed, never asked.

**What this gives up, stated plainly.** Recall is capped at 0.23 because 17 of
the 22 atrophic arches are only PARTIALLY edentulous, and the rule is silent
there by construction. That is not a limitation to be fixed later by tuning: on
partially edentulous arches the model scores **0 of 19** on atrophy, so the
alternative to silence is not recall, it is noise. Nothing in the mask
distinguishes a resorbed partial ridge from an intact one — bone height in a
span is a genuine imaging judgement, and this is the one axis in this document
where the source that ought to answer it simply cannot.

Where the mask and the reference disagree about edentulism (mask-only 2,
GT-only 2, agreeing 3), the two mask-only arches both carry atrophy in the
reference anyway, which is why precision stays at 1.00 rather than falling to
3/5.

### THE RULE — canal location

**Take the composite's answer when its per-tooth values agree. Otherwise say
"lingual".**

```
canal location(side) = the composite's value at 36-38 / 46-48 when all the
                       answered ones agree, else "lingual"
```

**0.87 against 0.83 shipping** — and the 3D canal fact, which is what ships, is
not read at all.

**This ties the constant on validate and is chosen anyway.** Always-lingual also
scores 0.87 here, and the unanimity rule recovers none of the 8 buccal sides it
misses. The reason to prefer it is that it *can* move and a constant cannot:
where the composite agrees on buccal, the rule says buccal. On this split it
never does, so the choice rests on the training distribution rather than on this
measurement — see the prior below.

The ground truth is **55 lingual / 8 buccal** over 63 sides, so a constant is a
strong answer and every read is below it:

| strategy | accuracy over all 63 sides |
|---|---|
| **composite unanimous, else lingual — THE RULE** | **0.87** |
| constant: always lingual | 0.87 |
| composite majority, 3D as fallback | 0.86 |
| the 3D canal fact alone | 0.83 |
| **3D, else lingual — ships today** | **0.83** |
| composite majority alone (16 sides unanswered) | 0.62 |

**No strategy recovers a single buccal side.** The composite-unanimous variant
ties the constant at 55/63 and gets none of the 8 buccal cases the constant
misses — it only avoids losing lingual ones. *Canal position always stated*
already rested on the first half of this ("the VLM scored 0 of 5 on the buccal
sides and had already answered lingual on four of them", measured on 70 sides
against the reports); the same holds on the generated GT at 63 sides. Neither source has ever identified a
buccal canal in this split.

**On the per-tooth values, since they were the reason to look.** The composite
answers `location` at 36-38/46-48, so a side can carry up to three values, and
they agree: **45 of 47 sides are unanimous**, the other 2 split 2-1. Deciding
between them is therefore not the problem — coverage is (47 of 63 sides get any
value), and neither is worth solving, because the unanimity is partly the same
lingual prior expressed three times.

**THE PRIOR IS SET FROM TRAINING, AND VALIDATE IS THE FRIENDLIER SPLIT.**
Over the training ground truth, first reader per case, the same scoring
convention:

| split | cases stating a location | sides | lingual | buccal | always-lingual |
|---|---|---|---|---|---|
| **training** | 298 of 582 | 794 | 640 | **154** | **0.806** |
| validate | 19 of 40 | 63 | 55 | 8 | 0.873 |

So the expected accuracy of the lingual fallback is about **0.81**, not the 0.87
this split gives, and the rule's apparent +0.04 over shipping is provisional:
the 3D read scores 0.83 on validate and has not been scored on training at all.
Buccal is **19%** of training sides against 13% here, which is both why the
margin may shrink and why a rule that *can* answer buccal is worth keeping over
one that cannot.

Deriving the prior from validate would be tuning on the eval split, which
`docs/vision_sft_plan_light.md` §9 calls the rule most tempting and least worthwhile
to bend. 0.806 is the number to quote; 0.873 is what this draw happens to give.

**Not yet evaluated properly, and known to be so.** The composite's per-tooth
`location` has never been scored against the 794 training sides, only the 63
here. That measurement needs training-split inference, which does not exist for
this arm — `outputs/aksssr_v6_training/predictions` is empty. **Re-run this
comparison when it does**; if the composite still recovers no buccal side over
154 of them, the rule collapses to the constant and `location` should leave the
schema.

**What this gives up until then.** In practice a buccal canal stays
unreportable, because the composite has never once agreed on one: 0 of 8 here.
If `location` is to earn its place it needs a source that can see the buccal
case, not a better vote among two that cannot.

**Schema consequence, worth noticing but not acted on here.**
`tooth_{fdi}_mandible_canal` carries exactly two answerable fields —
`location` and `adjacent_to_teeth`. This rule retires the first, and *THE RULE
— canal-adjacent teeth* sources the second from the mask, so nothing downstream
would read that fact at all. It is asked at six positions per case. The same
applies to `mandible_canal_{side}`'s own two fields, except that its presence is
still what tells postprocess a canal exists to describe — and the mask's canal
labels (3 and 4) would answer that better.

### THE RULE — maxilla FOV scope (2026-08-17)

**Not one of the ten**, and not in `source_rules.py`: it fires far earlier, in
`postprocess_pred.py`'s `drop_excluded_maxilla` gate, which runs before any
builder and decides whether the arch exists at all. But it answers the same
question — *which source decides* — so it belongs here.

**The rule.** With a facts file, `maxilla_scope.maxilla_included` is taken from
`facts.structured.fov.maxilla` and the model's own answer is overwritten:

```
fov.maxilla == "excluded"      -> not_included : null every maxilla bone fact,
                                  drop tooth_11..28 (unchanged behaviour)
fov.maxilla == "partial"       -> included     : keep the whole arch
no maxilla key in the fov dict -> included     : neither FOV condition fired,
                                  so the maxilla is in the volume whole
no fov block / no facts file   -> the model's answer decides, exactly as before
```

`scope_source` records which branch spoke (`facts_fov_maxilla_excluded`,
`facts_fov_maxilla_in_volume`), so no rewrite is silent.

**Why the facts.** "Is the maxilla in this volume" is a property of the
acquisition, not a clinical judgement, and the mask answers it directly:
`extract_facts.py` measures `mask == MAXILLA` against the volume's superior
edge, `audit_facts.py` rewrites the key to `excluded` below
`MIN_ARCH_BONE_MM3`. The VLM is answering the same question off one 3D render,
and the failure is asymmetric — a render can look bare when the mask is not.

**What it moves, validate-40 arm 6.** The model and the facts disagree on 4 of
40 cases. Three are `model included / fov excluded` and were already caught by
the `coverage` sidecar, which is the same mask measurement under another name;
for those, only the `scope_source` label changes. The fourth is the one the old
gate acted on:

| case | fov | model | old report | new report |
|---|---|---|---|---|
| A018 | `partial` | `not_included` | `Maxilla: not included in the scan volume.` | `Maxilla: partially included in the scan volume. Complete maxillary edentulism with marked atrophy.` + the sinus line |

A018's mask holds maxillary bone and its reference report carries a
four-sentence maxilla paragraph — completely edentulous, severe atrophy in all
sectors, normally pneumatized sinuses, no osteolytic lesions. The arch was
being deleted, and three of those four findings with it, because one render
looked empty. Nothing else in the split changes: the other 39 summaries differ
only in that label, and with no `--facts-dir` the output is byte-identical to
the pre-change code.

#### Surveyed against the ground truth

Scored against the consensus `{case}_gt.json`, whose `maxilla_scope` is binary
(`included|not_included`) and `null` where no report states the scan's extent —
those cases are dropped, not counted as negatives. Positive class is
`not_included`, the destructive answer, the one that deletes an arch.

**validate-40, out of sample** — GT 29 included / 11 not_included:

| source | acc | prec | rec |
|---|---|---|---|
| **facts `fov.maxilla`** | **0.875** | 0.714 | 0.909 |
| `coverage` sidecar | 0.875 | 0.714 | 0.909 |
| pred arm 6 | 0.825 | 0.667 | 0.727 |
| pred arm 5 | 0.850 | 0.667 | 0.909 |
| pred AWQ base | 0.850 | 0.667 | 0.909 |
| pred v6.9 | 0.825 | 0.625 | 0.909 |

The facts and the `coverage` sidecar score identically on every case — they are
the same mask measurement under two names, so the rule adds no new witness, it
just lets the existing one answer in both directions.

**The gate, which is what actually deletes an arch** — old (`coverage` OR the
model) against new (facts first): **0.850 → 0.875 on all three model arms**,
each moving exactly one case, every time in the new gate's favour. It is a
*different* case per arm — A018 (arm 6), S0037 (arm 5), S0017 (AWQ base) — so
this is not an A018 quirk: every arm invents one `not_included` on a maxilla the
mask can see, and no two arms invent the same one.

**training-550, and it is a tie** — 32 of the 582 cases carry no GT answer. The
facts score 0.789 and the arm-5 predictions score 0.789; the gate moves 24 cases
and goes **12–12**, trading recall 0.904 → 0.860 for precision 0.731 → 0.749.
Those predictions are **in sample** — arm 5 was LoRA-trained on these cases'
targets, which are derived from these GT files — so this is the model at its
most flattered and it still only draws. It is not evidence the model generalises;
validate is the only split that speaks to that.

**Which bucket is weak, by facts value:**

| `fov.maxilla` | GT included | GT not_included | the rule is right | mean upper teeth in mask |
|---|---|---|---|---|
| `excluded` → not_included | 78 | 233 | **0.749** | 8.5 (incl) / 5.4 (not) |
| `partial` → included | 155 | 36 | **0.812** | 11.9 / 10.7 |
| no maxilla key → included | 46 | 2 | **0.958** | 11.5 / 7.0 |

The two buckets this rule protects are the reliable ones (0.81, 0.96). The weak
one is `excluded` at 0.75 — 78 training cases whose mask holds no maxillary bone
but whose report still calls the maxilla included, and they average 8.5 upper
teeth in the mask, the same population `audit_facts.py`'s own table describes.
That bucket was already the gate before this rule, through `coverage`; the
change neither improves nor worsens it.

#### What it does to absent teeth: nothing, and that is the interesting part

`structured_findings_evaluation.py` against the same ground truth, summary column:

| split | absent teeth, before | after |
|---|---|---|
| validate-40 | 0.69/0.85 | **0.69/0.85** |
| training-582 | 0.60/0.85 | **0.60/0.85** |

Identical, because *THE RULE — absent teeth* is gated on
`fov.maxilla == "excluded"` — the **same field** this rule now reads. The
absent-teeth list was already the acquisition's answer; only the scope sentence
was the model's. What the change does is stop the two from contradicting each
other.

**And they did contradict.** A018's old summary listed all sixteen upper
absences, correctly, inside a `maxilla` section whose scope said `not_included`
— so the renderer deleted the arch and the list never reached the report. One
arch in 40 on validate, **24 in 582** on training.

**At the report level the trade is not free**, and it goes the other way on
training. Counting the claims that become reachable:

| split | arches un-deleted | claims | TP | FP | not asserted by any report |
|---|---|---|---|---|---|
| validate | 1 (A018) | 16 | **16** | 0 | 0 |
| training | 24 | 160 | 6 | 67 | 87 |

A018 is a whole-mouth-empty mask — no upper *and* no lower tooth labels — and
its reference does assert all sixteen upper absences. The 24 training arches are
the opposite: 22 of them have 6–15 upper teeth in the mask, the reference
asserts **no** upper absence in 23 of 24, and the "absences" are simply
positions the segmentation never labelled.

**That failure is the absence rule's, not this one's.** Those claims are emitted
for every maxilla that is not `excluded` — which is what the 0.60 precision on
the training row already measures. The old scope gate happened to mute them in
24 cases by deleting the arch on the model's word, for a reason that had nothing
to do with absence. It was a lucky mute, not a filter.

**The narrow fix was tried and rejected.** Claim upper absences under `partial`
only when the mask holds *no* upper tooth at all (which keeps A018 and drops the
24): validate 0.69/0.85 → **0.69/0.67**, training 0.60/0.85 → **0.66/0.73** —
F1 0.703 → 0.693. Precision buys less than the recall it costs, the same shape
the absent-teeth rule's own docstring records for the blunter version
("extending it to `partial` collapses recall from 0.85 to 0.53"). Left alone.

**And it does not yield to the obvious fix.** Sparing the `excluded` bucket when
the mask still holds upper teeth — `fov == excluded AND n_upper >= K -> included`
— is worse at every K tried: training accuracy falls monotonically from 0.789
(current) to 0.785 at K=14, 0.736 at K=10, 0.631 at K=1, because the
GT-`included` cases are spread across the whole tooth-count range rather than
concentrated at the top of it. The tooth count carries no separating signal here.
Improving that bucket needs a measurement the facts file does not currently
carry — how much maxillary bone, not whether any.

### THE RULE — condyle scope (2026-08-17)

**The schema is untouched.** Predictions keep answering
`fully_included|partially_included|not_included`; everything below happens in
postprocess, so no re-inference is needed and no stored prediction changes
meaning.

**Part 1 — binary, and one value for both sides**
(`postprocess_pred.merge_condyle_scopes`). `CONDYLE_SCOPE_BINARY` folds
`fully_/partially_included` → `included` on the way into the summary, and the
two sides are merged with *any side `not_included` wins*.

Why binary: the three-way read carries no signal. Over validate-40 arm 6 its
per-class precision equals the class prior to three decimals — `not_included`
42/70 = 0.600 against a 0.600 base rate, `partially_included` 4/10 = 0.400
against 0.400. And `fully_included` is not a real state in this corpus: 3 of
2734 GT sides, never once in validate-40 or training-582, while the model
emitted it 6 times on training and all 6 were wrong.

Why merge: 971 of the 978 reference reports that mention the condyles use no
laterality word, only 3 in 1000 assert a different scope per side, and every
case-level consensus GT (622 cases) is symmetric — while arm 6 contradicts
itself between its two renders in 10 of 40 validate cases.

**Part 2 — the mask decides where it speaks**
(`source_rules.apply_condyle_fov`, RULES key `condyle_fov`):

```
fov.condyles == "excluded"  -> not_included   (the direct measurement)
fov.maxilla  == "excluded"  -> not_included   (no maxilla in the volume,
                                               so no condyles either)
neither key present         -> the merged read stands
```

**What each variant scores** (binary, GT `partially_included` folded to
`included`; validate-40 / 22 SFT-heldout / training-582):

| variant | validate | heldout | training |
|---|---|---|---|
| pre-rule, per side, no merge | 0.575 | 0.568 | 0.716 |
| merged reads alone (no facts) | 0.600 | 0.591 | 0.716 |
| facts alone, silence read as `included` | 0.525 | 0.545 | 0.699 |
| **THIS RULE** — facts win where they speak, else merged | **0.600** | **0.636** | **0.720** |
| *constant `not_included`* | *0.600* | *0.636* | *0.722* |

**The mask is not simply "more accurate".** Read as a complete binary answer it
is the *worst* variant, because it only ever writes `excluded` — never `partial`
or `included` — and the key is absent in 7 of 40 validate cases and 63 of 582
training. Its silence is not evidence of inclusion. Used only where it claims
something, it is the best source on the axis: on validate it speaks for 66 of
80 sides and the reads keep the other 14.

**The maxilla clause earns 0 sides today** and is kept as a guard:
`fov.maxilla == excluded` implies `fov.condyles == excluded` in 14/14 validate
cases, 4/4 heldout, 331 of 340 training. In the 9 training cases where they
disagree it overrides the more direct measurement — the accepted cost.

**What reaches the report.** One sentence for both sides, replacing the per-side
pair: `Condyles not included in the scan volume.` /
`Condyles included in the scan volume.` On validate-40 the result is the first
sentence in all 40 cases. Captioning (BLEU/METEOR, `NO_RADFACT=1`) moves
0.27238 → 0.27366 corpus (METEOR +0.0051, corpus BLEU-4 −0.0025 — collapsing two
sentences into one costs 4-grams and gains content words); 10 of 40 cases move,
6 up and 4 down.

**What none of it buys.** Every winning variant lands on `not_included` for all
40 validate cases: 0.600 *is* the majority-class score, all 32 `included` sides
are missed, and on training the constant (0.722) still edges the rule (0.720).
This buys a coherent source hierarchy and a bilaterally consistent statement,
not signal.

**Surveys.** The axis reads under a `scope2` kind (`survey_facts._cast`) so a
folded summary scores against three-way GT files and raw predictions without
re-deriving 1367 GT files. Per-axis on purpose — sinus scope still uses the
graded vocabulary for real.

## 4. Reference

### Running it

`official_ranking.py` runs on the login node, not via `sbatch` — `conda` is not
on `PATH` in a non-interactive shell, so the batch path fails its `nltk`
preflight, and compute nodes are not guaranteed outbound HTTPS.

```bash
ssh hpc
source ~/miniconda3/etc/profile.d/conda.sh && conda activate cbct_base
cd ~/project_ToothFairy4

# postprocess + synthesize + survey, from an existing predictions/ dir.
# --facts-dir (hence the source rules) is on by default.
SPLIT=validate code/pipeline/postprocess/postprocess_now.sh outputs/training_results/vsft_arm6/val_arm6
NO_SOURCE_RULES=1 code/pipeline/postprocess/postprocess_now.sh <run_dir>   # rebuild the pre-rules arm

# free, deterministic, and the right way to choose between arms
NO_RADFACT=1 OUT_DIR=<run_dir>/rank_norad_ \
  bash code/eval/evaluation.sh validate <run_dir>/synthesized_reports

# the official number: persistent qwen3-14b judge on an A100
sbatch --partition=gpu --qos=a100 --gres=gpu:a100:1 code/eval/judge_server.sh
OUT_DIR=<run_dir>/official_ranking_rules \
  code/eval/eval_now.sh validate <run_dir>/synthesized_reports
scancel -n judge
```

### What reaches the report — `synthesize_report.py`

Still deterministic string templates, never an LLM. A template cannot invent a
*new* clinical error on top of whatever the VQA already got wrong, and that
property was not given up.

#### The silencing rule, and the ≥0.35 bar

The rule: **a finding reaches the report only if it is measurably better than
silence.** Each silenced renderer returns empty and keeps its sentence verbatim in
its docstring, so re-enabling is restoring a function body.

Its operational form is the **≥0.35 precision bar**: every finding that reaches
the report text scores at least 0.35 precision, or is an accuracy measure at
0.46–0.86. Everything below the bar is summary-only. That threshold was set on
the v6.4 arm against the reference reports and is the one number carried over
from the removed 2026-08-07 section; *The rules* are held to it against the
generated ground truth instead, which is a different label and a stricter one on
several axes.

Three findings were silenced under it, measured per case against the reference
reports. Caries and root remnants now also have per-source rows in *The table*,
scored against the generated ground truth — a different label reaching the same
verdict:

| silenced | the measurement |
|---|---|
| **root remnants** | 107 teeth claimed across 35 of 40 cases; the references mention a remnant in **10**; **82 of the 107 claims sit in cases whose reports never raise the subject**. Largest sentence class in the report — 107 of 681 sentences, 15.7% — each carrying a second claim in its resorption clause, so ≈200 claims, most of them false. Single-source (`morphology.is_remnant`), no cross-source vote since v6.1 retired the arch-level fact. |
| **caries** | 6 cases claimed, references mention caries in 9, **1 overlaps**. With caries gone `render_morphology_findings` emits nothing at all. |
| **periodontal resorption** | 7 cases claimed, references discuss it in 12, **3 overlap** — and those 3 must still match `extent` and the bone-loss pattern, so 3 is a ceiling, not a score. |

Silenced earlier still, and left that way: crown / post-and-core /
fillings, bone quality, intrasinusal teeth, endodontic filling quality, impaction
angle, root fracture.

All of them **stay in the summary JSON**. Killing the fact would remove the
instrument that would later show an improvement.

#### Canal position always stated

The sentence became `"Right mandibular canal with a regular course, predominantly
lingual."` — the corpus form verbatim. Two axes share it and the references keep
them apart: *regular course* is the anomaly judgement, *predominantly lingual* is
the buccolingual position. schema v6.4's `location` covers only the second, so the
regular-course frame stays unconditional and the position rides beside it.

An unread location defaults to lingual. That is a prior, not a guess: of the 70
sides the validate references place buccolingually, **61 are lingual**, 5 buccal
and 4 central (which the enum cannot express). The VLM scored **0 of 5** on the
buccal sides and had already answered "lingual" on four of them, so the default
costs nothing it was getting right and recovers the 4 sides it left unanswered —
57/66 → 60/66 correct.

### Update log

| date | what |
|---|---|
| **2026-08-15/16** | `compare_sources.py` written; the per-source work begins. `audit_facts.py --derive-bridge-arches` implemented and run; `survey_facts._hashable` and `survey_findings.composite_coverage` fixed |
| **2026-08-16** | facts provenance re-measured — the pool is `extract_facts.py` over the upstream masks, byte-identical on re-run; the competition path takes upstream facts and audits them first |
| **2026-08-16** | all ten source rules implemented in `code/pipeline/postprocess/source_rules.py`; the fillings renderer split out of `render_restoration_summary` and gated on `source_rules` (the gate its docstring promised had never been implemented, so `NO_SOURCE_RULES=1` would have rendered the aliased 161 claims) |
| **2026-08-16** | document restructured into the four parts above; the 2026-08-07 half and the facts-file provenance section removed |
| **2026-08-16** | `caries` and `root_remnant` added to `CATEGORIES` — `survey_findings` keeps a `SCORED_CATEGORIES` subset, since `REPORT_GT` hand-codes neither |
| **2026-08-17 07:12** | official ranking of **arm 6** → **0.4557**, the new best; clinical F1 0.5014 with precision and recall both up on the previous arm, captioning flat |
| **2026-08-17** | the abnormal-class view added for the six arch/side rows. `periodontal resorpt.` and `bone quality` are at or below their trivial constants — the proof they stay silenced. `atrophy` is the opposite and earns its `on`. `sinus mucosa`/`content` are the always-normal constants exactly. **Sinus `scope` measures nothing**: its GT is `_resolve_scope`'s fallback on 80/80 sides, so PRED 1.00 is a constant matching a constant |
| **2026-08-17** | **THE RULE — maxilla FOV scope**: the arch gate reads `facts.fov.maxilla` instead of the model's answer. Gate accuracy 0.850 → 0.875 on validate, on all three model arms tested; a 12–12 tie on 550 training cases. It also recovered the only two correct sinus reads postprocess was discarding |
| **2026-08-17** | **THE RULE — absent teeth re-stated as a filter on *unassessable* positions**, and its FOV gate re-measured on all 622 cases. Two findings. (i) **The GT cannot referee this rule unaided**: `presence_enumerated` fill-in turns 2072 of the 2491 GT-answered upper positions in excluded cases into labels no report ever stated (141 arch blocks name no tooth yet carry a `presence_enumerated` verdict), which is where the old 0.07-vs-0.68 precision gap came from. Scored on report-settled positions only, the gap is 0.46 vs 0.82 and the gate is a fact-level **wash** — F1 0.858 shipped against 0.860 ungated, 56 true positives traded for 66 false ones. (ii) **The report survey is the evidence that stands**: with the maxilla excluded, 77% of reports make only an arch statement and 13% are silent; 15% of cases name an upper tooth at all and **5% call one absent**, against 26% where the maxilla is imaged. The gate stays — 1904 of the mask's 2026 upper absence claims land on positions no report settles, which fact-level scoring drops and RadFact charges. Narrower gates rejected: case-1-only (0.860, no report evidence in that bucket), ≤4 stratum (0.863, noise on 33 claims), gating an edentulous imaged maxilla (0.850, deletes A018). **No code change** — `source_rules.absent_list` already implements the winner. New: `code/studies/survey_upper_mentions.py`, `code/studies/absent_fov_gate_evidence.py` |
| **2026-08-17** | **document re-based on arm 6.** Every table re-measured on `vsft_arm6/val_arm6`; the superseded arm's numbers removed rather than kept beside them; every training-split section emptied, because arm 6 has no training-split inference |
| **2026-08-17 22:52** | **official ranking of arm 6 re-run after the day's postprocess work → 0.4658**, the final number. Same checkpoint, byte-identical predictions; the condyle, maxilla-FOV and atrophy commits are worth **+0.0101**, all clinical — F1 0.5014 → 0.5139, recall +0.031 against precision −0.010, captioning flat. Scored into `outputs/aksssr_v7_trained_arm6_validate/`, which is where the reports now live |

### Where the artifacts live

Arm 6 lives in **two** directories, and the split matters when reading numbers
back. Training artifacts and the first scoring pass are in
`outputs/training_results/vsft_arm6/` — scored in place rather than copied into
an `aksssr_*` dir. `postprocess_now.sh` **cannot infer the split from
`val_arm6`**: the name matches neither `*validate*` nor `*train*`, so it exits
rather than guessing. Pass `SPLIT=validate`. The **final** summaries, reports and
ranking are in `outputs/aksssr_v7_trained_arm6_validate/`, from the 22:52 re-run.

| what | path |
|---|---|
| adapter, 1-epoch and 2-epoch checkpoints, `train_meta.json` | `vsft_arm6/weights/` |
| validate-40 predictions (identical in both dirs) | `vsft_arm6/val_arm6/predictions/` |
| **final summaries and reports** | `aksssr_v7_trained_arm6_validate/{summaries,synthesized_reports}/` |
| **official ranking — 0.4658 (the number)** | `aksssr_v7_trained_arm6_validate/official_ranking/official_ranking_20260817_225249.json` |
| superseded ranking — 0.4557 | `vsft_arm6/val_arm6/official_ranking/official_ranking_20260817_071205.json` |
| **source comparison** | `val_arm6/survey/source_compare_20260817_063758.{txt,json}` |
| fact-level survey, with the rules | `val_arm6/survey/structured_findings_evaluation_20260817_063256.{txt,json}` |
| fact-level survey, as scored by `pool_infer.sh` | `val_arm6/survey/survey_facts.{txt,json}` |
| held-out 22 (tooth-calls-only payload) | `ho_awq_arm6/` |
| the merged servable checkpoint | `models/Qwen3.5-9B-AWQ-dental-cbct-sft` (promoted by rename 2026-08-17) |

The same 40 cases are also mirrored at `outputs/aksssr_v7_trained_arm6_validate/`,
which is where `postprocess_now.sh` and the surveys have been re-run since; it
carries the same `official_ranking/` and the later `survey_facts_*` stamps.
The ranking's per-case `results.jsonl` is what `per_dataset_breakdown.py
--corpus-bleu` reads.

**Not present, and every "empty" section above points at it:** arm 6 has no
training-split inference. There is no `train_arm6/`, so no 582-case source
table, no per-rule confirmation at scale, and no pre-rules report-level
comparison on this arm.

`outputs/` is gitignored (commit `3f8d256`), so these live on the cluster only —
this document and the code are the record that git carries.

**One instrument fix is worth knowing about.** `survey_findings.composite_coverage()`
keyed on `tooth_{fdi}_restoration`, a fact v6.4 retired in favour of
`tooth_{fdi}_morphology`, so on every v6.4+ run it reported *no* tooth as having
been cropped. That fed one line of `survey_findings.py`'s output — "…of which N
got no composite crop at all" — which was counting teeth that had one. It now
reads newest-fact-first with the old name as fallback, the same pattern
`pred_claims` already used. It changes no score, only that count and the
`detail` domains in *The table*.

### Caveats

**The RadFact judge is not deterministic.** On 28 byte-identical reports scored
twice, BLEU-4 matched exactly on all 28 while `clinical_score` differed on **24**,
per-case sd 0.0625, max |Δ| 0.2136 — at `temperature=0`. That puts roughly **±0.02**
around any 40-case clinical score, the same magnitude as most arm differences.
Choose between arms with BLEU/METEOR (`NO_RADFACT=1`, free and deterministic) and
the survey; spend RadFact on the final number, and score twice when the expected
difference is small.

**The validate split no longer flatters the score — it now costs it.** Validate
is balanced 10/10/10/10 while training is **69% P** (402 of 582; A 14.6%,
F 9.1%, S 7.2%). Weighting arm 6's per-prefix per-case means by that mix gives
**0.4422** against the balanced **0.4099** — a **bonus of 0.032**, where every
earlier arm took a penalty. The premise inverted with the result: P was the
weakest prefix in every arm before the source rules and is now the strongest by
0.09, so a P-heavy distribution helps this arm. The remaining exposure is F, at
9% of training and the worst prefix here.

**The template is calibrated to the wrong sub-dataset — specifically to F.**
Sentences per case against the references, on arm 6's reports:

| | A | F | P | S |
|---|---|---|---|---|
| generated | 13.4 | 13.0 | 9.1 | 10.6 |
| reference | 13.6 | 7.5 | 8.2 | 9.2 |
| overshoot | 0.99× | **1.73×** | 1.11× | 1.15× |
| RadFact precision | 0.5322 | **0.3182** | 0.6213 | 0.6918 |

A now matches its reference length exactly and the other three are within 15%,
except F: at 1.73× it carries by far the worst precision, which is the same
relationship every earlier arm showed. **Per-prefix length calibration is an
F-only fix**, and it still needs no new information — the pipeline already
knows the prefix.

**A mask-only baseline was not far behind — restated, not re-measured.**
`outputs/no_VLM` — template reports built from masks and facts with no VLM at
all — scored 0.2761 under `gpt-4o` against the v6.4 arm's 0.2880 (per-case
means). Both numbers belong to the removed section: different arm, different
judge, different eval dates. The point survives the arithmetic and the source
rules sharpen it, since seven of the ten now take the finding from the mask
rather than from a read. **Worth a same-day paired run under the current judge
before claiming the VLM earns its cost** — that run has not happened.

### TODO

**Item 1: run arm 6 on the training split.** Everything else on this list is
downstream of it. It is one inference job; the CPU half — `postprocess_now.sh`,
`compare_sources.py --split training`, `structured_findings_evaluation.py` — is seconds. Until it
exists, every number in this document is 40 cases.

What it would settle, in the order the margins deserve:

| | validate says | what training would decide |
|---|---|---|
| **crown** | mask 0.33/0.29 vs composite 0.32/0.32 — level | whether `RULES["crown"]` should point at the composite or be turned off. Nothing that ships changes either way; the renderer is silenced |
| **fillings** | composite 0.31, down from the 0.39 the rule was sized on | whether the rule still buys what it costs — it is the only rule that removes correct claims |
| **canal location** | 3D 0.84, constant **0.873**, rule 0.86 | the read is already at or below its constant on the friendly split. The COMPOSITE's per-tooth `location` has still never been scored at scale, and that is what says whether `location` should leave the schema |
| **impaction C vs E** | 5 true positives against 4 false on 66 slots | the closest call in the set, and `compare_sources` does not produce the per-strategy table — it was computed by hand |
| **the abnormal-class rows** | 80 arch slots each | atrophy, the three sinus rows, periodontal and bone quality are all measured on 80 slots |

**Item 2: score a pre-rules arm 6 at report level.** The fact-level gain is
measured (*Fact level*); the report-level gain is not, on this arm. One
`NO_SOURCE_RULES=1 postprocess_now.sh` plus one `eval_now.sh` against the same
judge, on the same evening, or the comparison is worthless — see the judge
caveat.

**Item 3: `THE RULE — maxilla FOV scope` is the one rule whose training
evidence exists and disagrees.** Its scope decision is a 12–12 tie on 550
training cases against a clear win on validate, and the absence content it
un-gates is measurably bad there. That section has the numbers; it is flagged
here because it is the only place the two splits are known to point in
different directions.

**Nothing is outstanding in code.** All ten rules are implemented, the fillings
renderer is split and gated, and the maxilla FOV gate reads the facts.
