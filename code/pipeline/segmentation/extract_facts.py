#!/usr/bin/env python3
"""
Extract structured clinical facts from predicted ToothFairy FDI masks.  (v3)

v3 changes:
  - Orientation is derived from ANATOMY, not the header: the maxilla sits above
    the mandible, so their centroids define the superior-inferior axis and sign.
    This is robust to identity/flipped/honest headers alike (no DICOMOrient).
  - FOV (maxilla partial / condyles excluded) tests the anatomically-correct
    superior boundary.
  - When the maxilla is partially imaged, per-tooth UPPER absences are suppressed
    (a tooth outside the FOV is not "absent", just un-imaged) -> big precision win.
    Toggle with --keep-upper-absences.

Label scheme: teeth 11-48, mandible 1, maxilla 2, IAC 3/4, sinus 5/6,
bridge 8, crown 9, implant 10, pulp 50, incisive/lingual 51/52/53.
"""
import os, json, argparse, glob
from pathlib import Path
import numpy as np
from scipy import ndimage

# SimpleITK preferred, nibabel accepted -- and the fallback is not a nicety.
# This is the ONLY module in the codebase that imports SimpleITK; every other
# generator reads masks with nibabel, so a container built for the pipeline is
# not guaranteed to carry it, and infer.py now calls this script
# on the critical path of a 15-minute budget. A missing import there is a
# failed submission, not a slow one.
#
# THE TWO READERS MUST AGREE ON AXIS ORDER, and they do not by default:
# SimpleITK's GetArrayFromImage is (z, y, x) while GetSpacing() is (x, y, z),
# and extract() below relies on exactly that pairing (see `samp`). nibabel
# gives (i, j, k) for both, so the array is transposed to match rather than the
# spacing being re-ordered -- one convention, chosen by the caller that already
# depends on it.
try:
    import SimpleITK as sitk
except ImportError:                                          # pragma: no cover
    sitk = None
    import nibabel as nib


def read_mask(path):
    """(array in z,y,x order, spacing in x,y,z order) -- either backend."""
    if sitk is not None:
        img = sitk.ReadImage(str(path))
        return sitk.GetArrayFromImage(img).astype(np.int16), img.GetSpacing()
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj).astype(np.int16).T      # (i,j,k) -> (k,j,i)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return arr, zooms

MANDIBLE, MAXILLA = 1, 2
IAC = {"left": 3, "right": 4}
BRIDGE, CROWN, IMPLANT = 8, 9, 10
FDI_TEETH = [q * 10 + i for q in (1, 2, 3, 4) for i in range(1, 9)]
UPPER = [t for t in FDI_TEETH if t < 30]
LOWER = [t for t in FDI_TEETH if t >= 30]
POSTERIOR_LOWER = [t for t in LOWER if t % 10 >= 6]

# spatial order of teeth along each arch (right molars -> across midline -> left molars)
ARCH_UPPER = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
ARCH_LOWER = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]

TOOTH_MIN_MM3 = 50.0
RESTO_MIN_MM3 = 20.0
IAN_NEAR_MM = 2.0
SUP_MARGIN = 2
EMIT_NEGATIVES = True
KEEP_UPPER_ABSENCES = True     # emitting true absences helps RadFact; suppression hurt


def largest_cc_size(mask):
    """Voxel count of the biggest connected component.

    np.bincount replaces ndimage.sum(np.ones_like(lab), ...): same counts, but
    without allocating a full-volume array per call. This runs once per tooth
    (32x per case), so the allocation dominated fact extraction on slow CPUs.
    """
    lab, n = ndimage.label(mask)
    if n == 0:
        return 0
    counts = np.bincount(lab.ravel())
    counts[0] = 0                      # background
    return int(counts.max())


def centroid(mask):
    idx = np.argwhere(mask)
    return idx.mean(0) if len(idx) else None


def nearest_tooth(c, tc):
    if c is None or not tc:
        return None
    return min(tc, key=lambda t: np.linalg.norm(tc[t] - c))


def _estimate_slot_centroids(tc, arch):
    """Estimate physical centroids of ABSENT slots in an arch by interpolating /
    extrapolating from present neighbours along the arch order."""
    present = [(i, t) for i, t in enumerate(arch) if t in tc]
    if len(present) < 2:
        return {}
    est = {}
    for i, t in enumerate(arch):
        if t in tc:
            continue
        left = [(pi, pt) for pi, pt in present if pi < i]
        right = [(pi, pt) for pi, pt in present if pi > i]
        if left and right:
            (li, lt), (ri, rt) = left[-1], right[0]
            f = (i - li) / (ri - li)
            est[t] = tc[lt] * (1 - f) + tc[rt] * f
        elif len(left) >= 2:
            (l2i, l2t), (l1i, l1t) = left[-2], left[-1]
            step = (tc[l1t] - tc[l2t]) / max(l1i - l2i, 1)
            est[t] = tc[l1t] + step * (i - l1i)
        elif len(right) >= 2:
            (r1i, r1t), (r2i, r2t) = right[0], right[1]
            step = (tc[r2t] - tc[r1t]) / max(r2i - r1i, 1)
            est[t] = tc[r1t] + step * (i - r1i)
        elif left:
            est[t] = tc[left[-1][1]]
        elif right:
            est[t] = tc[right[0][1]]
    return est


def nearest_absent_slot(c, tc, absent):
    """Assign an implant to the ABSENT tooth slot physically closest to its
    centroid, using slot positions interpolated from present teeth. Geometric —
    respects left/right and does not cross the midline via FDI arithmetic."""
    if c is None or not tc:
        return None
    est = {}
    est.update(_estimate_slot_centroids(tc, ARCH_UPPER))
    est.update(_estimate_slot_centroids(tc, ARCH_LOWER))
    cand = {t: p for t, p in est.items() if t in absent}
    if not cand:                              # fall back to nearest present tooth's quadrant
        ref = nearest_tooth(c, tc)
        return ref
    return min(cand, key=lambda t: np.linalg.norm(cand[t] - c))


def superior_axis_end(mask):
    """(axis, end): array axis along S-I, and +1 if higher index = superior.
    From maxilla-vs-mandible centroids -> header independent."""
    ci = centroid(mask == MANDIBLE)
    cs = centroid(mask == MAXILLA)
    if ci is None or cs is None:
        return 0, +1
    diff = cs - ci                    # mandible -> maxilla points superior
    axis = int(np.argmax(np.abs(diff)))
    return axis, (+1 if diff[axis] > 0 else -1)


def touches_superior(m, axis, end, margin=SUP_MARGIN):
    idx = [slice(None)] * m.ndim
    idx[axis] = slice(m.shape[axis] - margin, None) if end > 0 else slice(0, margin)
    return bool(m[tuple(idx)].any())


def extract(mask, spacing_xyz):
    samp = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    voxvol = float(np.prod(spacing_xyz))
    facts, phrases = {}, []

    # ---- dentition presence (phrases emitted later, after FOV is known) ----
    present, absent, tc = [], [], {}
    for t in FDI_TEETH:
        m = mask == t
        if m.any() and largest_cc_size(m) * voxvol >= TOOTH_MIN_MM3:
            present.append(t); tc[t] = centroid(m)
        else:
            absent.append(t)
    facts["teeth_present"], facts["teeth_absent"] = present, absent
    n_upper = sum(1 for t in UPPER if t in present)

    # ---- FOV from anatomy ----
    si_axis, si_end = superior_axis_end(mask)
    facts["fov"] = {}
    max_m, mand_m = mask == MAXILLA, mask == MANDIBLE
    maxilla_partial = bool(max_m.any() and
                           (touches_superior(max_m, si_axis, si_end) or n_upper < 4))
    condyles_excluded = bool(mand_m.any() and touches_superior(mand_m, si_axis, si_end))
    if maxilla_partial:
        facts["fov"]["maxilla"] = "partial"
    if condyles_excluded:
        facts["fov"]["condyles"] = "excluded"
    suppress_upper = maxilla_partial and not KEEP_UPPER_ABSENCES

    # ---- dentition phrases (gate upper absences when maxilla partial) ----
    for t in absent:
        if t < 30 and suppress_upper:
            continue                    # upper tooth may be un-imaged, not absent
        phrases.append(f"Absence of tooth {t}.")
    for q, name in [(1, "first"), (2, "second"), (3, "third"), (4, "fourth")]:
        if q in (1, 2) and suppress_upper:
            continue
        qt = [t for t in FDI_TEETH if t // 10 == q]
        miss = [t for t in qt if t in absent]
        if len(miss) == len(qt):
            phrases.append(f"Complete edentulism of the {name} quadrant.")
        elif len(miss) >= 5:
            phrases.append(f"Partial edentulism of the {name} quadrant.")

    # ---- canals ----
    facts["canals"] = {}
    for side, lid in IAC.items():
        if (mask == lid).any():
            facts["canals"][side] = "present"
            phrases.append(f"{side.capitalize()} mandibular canal with a regular course.")

    # ---- crowns (dedup) ----
    facts["crowns"] = []
    cm = mask == CROWN
    if cm.any():
        lab, n = ndimage.label(cm); seen = set()
        for i in range(1, n + 1):
            cc = lab == i
            if cc.sum() * voxvol < RESTO_MIN_MM3:
                continue
            tid = nearest_tooth(centroid(cc), tc)
            if tid and tid not in seen:
                seen.add(tid); facts["crowns"].append(tid)
                phrases.append(f"Prosthetic crown on tooth {tid}.")

    # ---- implants (merge fragments, dedup slots) ----
    facts["implants"] = []
    im = mask == IMPLANT
    if im.any():
        im = ndimage.binary_closing(im, iterations=2)
        lab, n = ndimage.label(im); seen = set()
        for i in range(1, n + 1):
            cc = lab == i
            if cc.sum() * voxvol < RESTO_MIN_MM3:
                continue
            pos = nearest_absent_slot(centroid(cc), tc, absent)
            key = pos if pos is not None else f"cc{i}"
            if key in seen:
                continue
            seen.add(key); facts["implants"].append(pos)
            phrases.append(f"Presence of an endosseous implant in position {pos}."
                           if pos else "Presence of an endosseous implant.")

    facts["bridge_present"] = bool((mask == BRIDGE).any()
                                   and (mask == BRIDGE).sum() * voxvol >= RESTO_MIN_MM3)
    if facts["bridge_present"]:
        phrases.append("A fixed prosthetic bridge is present.")

    # ---- IAN proximity ----
    iac = np.isin(mask, list(IAC.values()))
    facts["ian_close_teeth"] = []
    if iac.any():
        # The EDT is only ever read at posterior lower teeth, so compute it on
        # the bounding box of (canals + those teeth) instead of the whole
        # volume. Distances inside the box are identical because the box
        # contains every source voxel (it is built from the canals).
        roi = iac.copy()
        for t in POSTERIOR_LOWER:
            if t in tc:
                roi |= (mask == t)
        idx = np.argwhere(roi)
        if idx.size:
            lo = idx.min(0)
            hi = idx.max(0) + 1
            pad = int(np.ceil(IAN_NEAR_MM / min(samp))) + 2
            lo = np.maximum(lo - pad, 0)
            hi = np.minimum(hi + pad, mask.shape)
            sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
            sub_iac = iac[sl]
            sub_mask = mask[sl]
            edt = ndimage.distance_transform_edt(~sub_iac, sampling=samp)
            for t in POSTERIOR_LOWER:
                if t not in tc:
                    continue
                m = (sub_mask == t)
                if m.any() and edt[m].min() <= IAN_NEAR_MM:
                    facts["ian_close_teeth"].append(t)
                    phrases.append(f"Tooth {t} is in close proximity to the mandibular canal.")

    # ---- FOV phrases ----
    if maxilla_partial:
        phrases.append("The maxilla is partially included in the acquisition.")
    if condyles_excluded:
        phrases.append("The mandibular condyles are excluded from the acquisition and not visible.")

    if EMIT_NEGATIVES:
        phrases.append("No definite osteolytic or osteosclerotic lesions.")

    facts["present_label_ids"] = sorted(np.unique(mask).tolist())
    facts["si_axis"], facts["si_end"] = si_axis, si_end     # for debugging
    return facts, phrases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--no-negatives", action="store_true")
    ap.add_argument("--suppress-upper-absences", action="store_true",
                    help="drop per-tooth upper absences when maxilla is partial "
                         "(off by default — suppression was found to hurt the score)")
    args = ap.parse_args()
    global EMIT_NEGATIVES, KEEP_UPPER_ABSENCES
    EMIT_NEGATIVES = not args.no_negatives
    KEEP_UPPER_ABSENCES = not args.suppress_upper_absences

    out = Path(args.output); (out / "facts").mkdir(parents=True, exist_ok=True)
    masks = sorted(glob.glob(os.path.join(args.input, "*.nii.gz")))
    print(f"extracting facts from {len(masks)} masks")
    with open(out / "facts_all.jsonl", "w") as jl:
        for mp in masks:
            arr, spacing = read_mask(mp)
            case_id = Path(mp).name.replace(".nii.gz", "")
            facts, phrases = extract(arr, spacing)
            rec = {"case_id": case_id, "structured": facts, "phrases": phrases}
            (out / "facts" / f"{case_id}.json").write_text(json.dumps(rec, indent=2))
            jl.write(json.dumps(rec) + "\n")
            print(f"  {case_id}: {len(facts['teeth_present'])} teeth, "
                  f"{len(phrases)} phrases, si_axis={facts['si_axis']}({facts['si_end']:+d}), "
                  f"fov={facts['fov']}")
    print("=== facts done ===")


if __name__ == "__main__":
    main()