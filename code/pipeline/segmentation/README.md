# Segmentation — not in this repository

This directory is deliberately empty. It marks stage 1 of the challenge
pipeline, which this repository does not implement.

The ToothFairy4 / ODIN2026 task splits in two. Segmentation turns the CBCT
volume into a labelled mask; report generation turns that mask plus the volume
into radiology prose. **This repo is the second half only.** Plug your own
segmenter in here, or supply a mask and facts file produced elsewhere.

## The interface

Everything downstream needs exactly two files per case:

```
mask.nii.gz     FDI-labelled segmentation: per-tooth labels plus jaw
                structures (mandible, maxilla, mandibular canal, sinuses).

facts.json      {"structured": {"teeth_present": [11, 12, 13, ...],
                                "teeth_absent":  [16, 25, ...]}}
```

`teeth_present` is the load-bearing field. The renderers draw and label **only**
the teeth it names, and `create_tooth_detail.py` builds one composite crop per
present tooth. A segmentation false positive that reaches this list becomes a
tooth in the report; one that never reaches it is never asked about. That is why
the field is an input to *rendering* and not merely metadata.

Everything else in `facts.json` is optional as input to the pipeline. Two fields
are not optional as input to the **generators**, though, and no segmenter
produces them:

| field | who needs it |
|---|---|
| `fov.maxilla: "excluded"` | the maxilla arch gate — see `../postprocess/source_rules.py` |
| `bridge_arches` | the bridge source rule, same file |

In the competition container these were written by a facts *audit* step that
corrected a handed `facts.json` against the mask. That audit, and the
mask-derived facts extractor it worked from, belong to the submission path and
are not part of this release. If you supply your own facts, either write those
two fields yourself or accept the defaults postprocess falls back to.

## Where the mask is actually read

Nothing here reads it — these do:

- `../preprocess/create_panoramic.py` — curved reconstruction, and the tooth
  outlines, filtered by `teeth_present`
- `../preprocess/create_3d_renders.py` — surface renders of jaws, teeth, canal
  and sinuses
- `../preprocess/create_tooth_detail.py` — the per-tooth composite crops
- `../preprocess/create_sinus_detail.py` — the maxillary sinus views
