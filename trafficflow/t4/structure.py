"""Corridor structure helpers: every measured link is a screenline between zones k and k+1."""
from __future__ import annotations

import numpy as np

from .data import Panel


def link_segment(P: Panel) -> np.ndarray:
    """Segment index k for every incidence link (paths o<=k<d cross it); -1 if not a pure screenline."""
    seg = np.full(len(P.link_ids), -1)
    for i in range(len(P.link_ids)):
        cols = np.flatnonzero(P.A[i] > 0)
        if len(cols) == 0:
            continue
        k = P.orig[cols].max()
        ideal = (P.orig <= k) & (P.dest > k)
        if np.array_equal(ideal, P.A[i] > 0):
            seg[i] = k
    return seg


def segment_matrix(P: Panel) -> np.ndarray:
    """S (nz-1, n_paths): S[k, p] = 1 if path p crosses segment k."""
    nz = len(P.zones)
    k = np.arange(nz - 1)[:, None]
    return ((P.orig[None, :] <= k) & (P.dest[None, :] > k)).astype(float)


def segment_counts(P: Panel, split: str):
    """Per-segment list of measured counts (in link order)."""
    seg = link_segment(P)
    c = P.counts[split]
    out = {}
    for i in np.flatnonzero(np.isfinite(c)):
        out.setdefault(int(seg[i]), []).append(c[i])
    return out
