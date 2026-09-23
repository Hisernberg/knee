"""Probability maps -> disjoint instance masks."""
import cv2, numpy as np
from scipy import ndimage as ndi


def instances(prob, edge=None, t=0.5, te=1.1, merge=0, min_area=0, min_score=0.0, hyst=None):
    """prob/edge: float32 HxW.  Returns (labels int32 HxW, scores list)."""
    fg = prob > t
    if hyst is not None:  # hysteresis: keep low-threshold regions that contain a high-threshold seed
        lab, n = ndi.label(fg)
        if n:
            mx = ndi.maximum(prob, lab, np.arange(1, n + 1))
            keep = np.zeros(n + 1, bool); keep[1:] = mx > hyst
            fg = keep[lab]
    if edge is not None and te < 1:
        core = fg & (edge < te)
    else:
        core = fg
    lab, n = ndi.label(core, structure=np.ones((3, 3)))
    if n == 0:
        return lab, []
    if edge is not None and te < 1:  # grow cores back into the removed edge pixels
        dist, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
        lab = np.where(fg, lab[iy, ix], 0)
    if merge > 0:  # group fragments closer than `merge` px into one instance
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * merge + 1, 2 * merge + 1))
        grown = cv2.dilate((lab > 0).astype(np.uint8), k)
        glab, gn = ndi.label(grown, structure=np.ones((3, 3)))
        lab = np.where(lab > 0, glab, 0)
    ids = np.unique(lab); ids = ids[ids > 0]
    if len(ids) == 0:
        return lab, []
    areas = ndi.sum(np.ones_like(prob), lab, ids)
    means = ndi.mean(prob, lab, ids)
    out = np.zeros_like(lab); scores = []
    j = 0
    for i, a, m in zip(ids, areas, means):
        if a < min_area or m < min_score:
            continue
        j += 1; out[lab == i] = j; scores.append(float(m))
    return out, scores
