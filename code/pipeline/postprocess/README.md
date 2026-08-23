# `code/pipeline/postprocess/` — predictions → summary → report

Everything downstream of the VLM. Per-fact JSON answers in, one free-text
radiology report out.

The substance is the **source rules**: they decide which SOURCE each finding
comes from, rather than voting between every read that can assert it. They are
worth **+0.06 official** on the shipping arm, and they are the reason this
directory is more than a formatter.

`docs/postprocess.md` is the evidence — every number, every measurement, every
rule that was tried and rejected. This file is the map.

## The files

| | |
|---|---|
| `normalize_pred.py` | repairs schema-violating VLM output against `schema/schema.json` — wrong enum values, scalars where a list belongs, near-miss key names |
| `postprocess_pred.py` | `{case}_pred.json` → `{case}_summary.json`. Classification, grouping and cross-source reconciliation |
| `source_rules.py` | the eleven source rules, applied as a post-pass over the finished summary |
| `synthesize_report.py` | `{case}_summary.json` → `{case}.txt`. **Deterministic string templates, never an LLM** |
| `rules_config.py` | the binding table: which config key sets which module constant. Not logic — plumbing |

Everything is flat-imported by name (`import source_rules`), never as a package.
See `code/_repo.py`.

## Running it

Normally you don't call these directly — `scripts/run_infer.sh` with
`STAGE=post` drives all three in the right order, which is where the mistakes
otherwise happen (surveying summaries that predate the change being measured, or
reading a report rendered from a stale summary).

```bash
# the CPU tuning loop: predictions -> summaries -> reports -> survey.
# No GPU, no model, seconds per case. Runs on the login node, no queue.
STAGE=post RUN_NAME=aksssr_v7_trained_arm6 scripts/run_infer.sh validate

# a different arm
STAGE=post POSTPROCESS_CONFIG=configs/postprocess/no_source_rules.yaml \
    scripts/run_infer.sh validate

# by hand, if you must -- PASS THE SAME CONFIG TO BOTH
python3 code/pipeline/postprocess/postprocess_pred.py \
    --config configs/postprocess/default.yaml \
    --pred-dir <run>/predictions --out-dir <run>/summaries \
    --facts-dir dataset/validate/facts --schema schema/schema.json
python3 code/pipeline/postprocess/synthesize_report.py \
    --config configs/postprocess/default.yaml \
    --summary-dir <run>/summaries --out-dir <run>/synthesized_reports
```

**`--facts-dir` is what turns the rules on.** Without it every rule is a no-op
and the summaries are the pre-2026-08-16 shape. Nothing else in the output says
so, which is why `infer.py` warns.

## The rules are a config file

Which rules and gates run is `configs/postprocess/<arm>.yaml`, not a constant
edited in these modules. An experiment is a new config file; the code stays put.

```bash
# check an arm before spending a run on it -- '*' marks every non-default key
python3 code/pipeline/postprocess/postprocess_pred.py --print-config \
    --config configs/postprocess/my_arm.yaml --pred-dir x --out-dir y
```

`configs/postprocess/README.md` has the how-to. Two things to know here:

- **An unknown key is an error, not a warning.** `filling:` for `fillings:`
  fails at load with the offending path. A misspelled override that runs clean
  and scores like the default is the failure mode the binding table exists to
  prevent.
- **The default is not a special case.** `default.yaml` restates the constants
  these modules carry, so a run with it and a run with no config produce
  identical summaries. `--print-config` verifies that and exits non-zero if the
  two ever drift.

Every summary records what built it under `postprocess_config` — the config path
and only the keys that differ from `default.yaml`. `outputs/` is gitignored and
job logs rotate, so this is the only durable answer to "which arm produced this".

## The eleven rules

Each moves one finding to the source that actually knows it. Numbers are
precision/recall on validate-40, arm 6, with and without the rules — the full
argument for each is a section of `docs/postprocess.md` §3 with the same name.

| rule | source it moves to | pre-rules | with the rule | in the report |
|---|---|---|---|---|
| absent teeth | mask list, FOV-gated | 0.82/0.75 | **0.83/0.85** | yes |
| impaction | composite eruption + 3D orientation | 0.67/0.48 | **0.78/0.85** | yes |
| endodontic | composite only, drop the arch | 0.75/0.66 | **0.88/0.64** | yes, teeth only |
| fillings | composite only, drop the arch alias | 0.21/0.35 | **0.30/0.24** | yes, gated on the rule |
| crown | mask crown list | 0.30/0.20 | **0.29/0.31** | no, deliberately |
| implants | mask positions, VLM clauses | 0.44/0.12 | **0.68/0.53** | yes |
| canal-adjacent | mask `ian_close_teeth` | 0.67/0.46 | **0.69/0.69** | yes |
| canal location | composite unanimous, else lingual | 0.83 | **0.86** | yes |
| alveolar atrophy | mask edentulous region | 0.76 | **0.78** | yes |
| fixed bridges | mask bridge label, per arch | 1.00/0.09 | **1.00/0.55** | yes |
| condyle scope | `facts.fov.condyles`, where it speaks | 0.575 | **0.600** | yes |

**A twelfth finding rides the same flag and is not in `RULES`:** the maxilla's
FOV scope (0.85 → 0.88). `facts.fov.maxilla` overrides the model's own
`maxilla_scope` answer in *both* directions, so the arch is emptied when the mask
has no maxilla and KEPT when it has one, whatever the render looked like. It
lives in `postprocess_pred.py`'s arch gate, under `maxilla_fov` in the config.
This is why `NO_SOURCE_RULES=1` (which drops `--facts-dir`) and a rules-off
config are **different ablations** — the first takes this with it, the second
does not.

**Where the work actually happens: implants and bridges**, both almost entirely
recall — 0.12 → 0.53 and 0.09 → 0.55.

**A rule is a SOURCE, not a gate.** Bridges was written as a gate first and it
was wrong: dropping the VLM's wrong-arch bridges left 4 of the 6 mask-confirmed
cases reporting no bridge at all. Where the mask is the better witness in *both*
directions, it may add what the reads missed, not only veto what they invented.
That asymmetry break holds axis by axis and nothing licenses trusting the facts
file in general — `crown` keeps a field silenced precisely because 0.33 does not
clear the bar.

## Things that will bite you

**The gate runs last, and that ordering is load-bearing.** `_gate_on_absent`
used to run inside `apply_absent`, i.e. before the rules that re-source
per-tooth findings — so a filling the gate had just dropped at an absent
position was written straight back, and one case reported "Absence of 14, 15…"
and "composite restoration(s) on teeth 15 and 27" in the same paragraph. A gate
is only a gate if nothing writes after it.

**A rule that re-sources a field must be able to write it where the old source
was silent.** Guarding on `if key in block` was written three separate times and
was wrong all three: 55 of 80 arch blocks carry no `impacted_teeth` key at all,
so the guard made the rule able to remove an impaction and never to add one.
Same trap cost 11 of 25 mask implants.

**Field ORDER in the schema is load-bearing upstream.** vLLM decodes with
`guided_json`, which follows `object_fields` order, so a field derived from
another (`impacted` from `eruption_state` + `orientation`) must come after them.
That is `schema/schema.json`'s problem, but it is why these modules can trust the
derived fields at all.

**The prompt's words are not the code's tokens.** v7.1's arch vocabulary says
`restoration`/`defect`/`unremarkable`; `ARCH_VALUE_ALIASES` folds those to
`filling`/`caries`/`normal` on the way in, so ~20 behaviours keyed on the old
strings never had to change and stored v6.x predictions still parse. Change the
schema's wording freely; **add an alias rather than renaming those tokens.**

**`synthesize_report.py` stays a template.** This is a project decision, not an
implementation detail: a template cannot introduce a *new* clinical error on top
of whatever the VQA already got wrong. Do not make it call a model.

## What reaches the report

**A finding reaches the report only if it is measurably better than silence** —
operationally, a **≥0.35 precision bar**. Everything below it is summary-only.

Silenced and staying that way: root remnants (107 teeth claimed, references
mention 10), caries (6 claimed, 1 overlap), periodontal resorption (7 claimed, 3
overlap), crown, post-and-core, bone quality, intrasinusal teeth, endodontic
filling quality, impaction angle, root fracture.

Two rules about them, both deliberate:

- **Every silenced fact stays in the summary JSON.** Killing the fact would
  remove the instrument that would later show an improvement.
- **Each silenced renderer keeps its sentence verbatim in its docstring**, so
  re-enabling one is restoring a function body, not reconstructing a template.

## Verifying a change

There is no test suite. Verification here is:

1. `STAGE=post scripts/run_infer.sh <split>` on a run dir, then **diff the survey against the
   previous one** — it writes `survey/survey_facts_<stamp>.{txt,json}` without
   overwriting and diffs automatically. `outputs/` is gitignored, so `git diff`
   cannot show you what a change did to the output; the survey diff can.
2. `NO_RADFACT=1` scoring — free, deterministic, and the right way to choose
   between arms.
3. RadFact only for the final number. **The judge is not deterministic**: on 28
   byte-identical reports, `clinical_score` differed on 24 at `temperature=0`,
   which puts roughly ±0.02 around any 40-case clinical score — the same
   magnitude as most arm differences. Score twice when the expected difference
   is small.

## Caveats worth carrying

- **Every number here is validate-40.** Arm 6 has not been run on the 582-case
  training split, so no rule is confirmed at scale, and two of them (`crown`,
  impaction C-vs-E) rest on margins inside the noise of 40 cases.
- **The template overshoots on F** — 1.73× the reference sentence count, and by
  far the worst RadFact precision (0.3182). Per-prefix length calibration is an
  F-only fix and needs no new information; the pipeline already knows the prefix.

`docs/postprocess.md` §4 has the rest, and the TODO list of what a training-split
run would settle.
