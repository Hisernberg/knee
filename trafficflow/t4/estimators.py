"""Task 4 estimators, all expressed on the collapsed screenline operator.

Each estimator takes a PanelSplit context and returns a non-negative path-flow vector
in the official path order.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Panel, load
from .solvers import kl_fit, l2_fit, ridge_nn
from .structure import segment_counts, segment_matrix


@dataclass
class Ctx:
    P: Panel
    split: str
    S: np.ndarray          # (n_seg, n_paths)
    c: np.ndarray          # per-segment count of this split
    nk: np.ndarray         # measured links per segment
    b: np.ndarray          # this split's released weak prior
    c_train: np.ndarray    # per-segment train counts (official baseline input)


def context(panel: str, split: str) -> Ctx:
    P = load(panel)
    S = segment_matrix(P)
    ns = S.shape[0]

    def segc(s):
        sc = segment_counts(P, s)
        c = np.array([sc[k][0] for k in range(ns)])
        nk = np.array([len(sc[k]) for k in range(ns)], float)
        # the release is exactly consistent inside a segment; guard anyway
        assert all(np.ptp(sc[k]) <= 1e-6 * max(1.0, abs(sc[k][0])) for k in range(ns))
        return c, nk

    c, nk = segc(split)
    ctr, _ = segc("train")
    return Ctx(P, split, S, c, nk, P.prior[split], ctr)


# ----------------------------------------------------------------------------- estimators

def est_prior(x: Ctx):
    return x.b.copy()


def est_official_baseline(x: Ctx):
    """Repository baseline: NNLS lambda=0.05 on TRAIN counts with the split prior."""
    return ridge_nn(x.S, x.nk, x.c_train, x.b, 0.05)


def make_nnls(lam: float):
    def f(x: Ctx):
        return ridge_nn(x.S, x.nk, x.c, x.b, lam)
    f.__name__ = f"nnls_split_lam{lam:g}"
    return f


def est_l2proj(x: Ctx):
    """Euclidean projection of the split prior on {A f = c, f >= 0}  (recommended)."""
    return l2_fit(x.S, x.c, x.b)


def est_klproj(x: Ctx):
    """Max-entropy (KL) projection: f = b * exp(S^T mu)."""
    return kl_fit(x.S, x.c, x.b)


def est_chi2proj(x: Ctx):
    """Relative (chi-square) projection: f = max(0, b (1 + S^T mu))."""
    return l2_fit(x.S, x.c, x.b, w=x.b)


def est_scaled_l2proj(x: Ctx):
    """Global scale s = argmin ||S s b - c|| first, then Euclidean projection of s*b."""
    Sb = x.S @ x.b
    s = float((x.nk * Sb) @ x.c / ((x.nk * Sb) @ Sb))
    return l2_fit(x.S, x.c, s * x.b)


def make_blend(fa, fb, a: float):
    def f(x: Ctx):
        return a * fa(x) + (1 - a) * fb(x)
    f.__name__ = f"blend_{fa.__name__}_{a:g}_{fb.__name__}"
    return f


ESTIMATORS = {
    "prior_b": est_prior,
    "official_baseline(train c, lam .05)": est_official_baseline,
    "nnls_split_lam0.05": make_nnls(0.05),
    "nnls_split_lam0.5": make_nnls(0.5),
    "nnls_split_lam5": make_nnls(5.0),
    "nnls_split_lam20": make_nnls(20.0),
    "l2proj": est_l2proj,
    "klproj": est_klproj,
    "chi2proj": est_chi2proj,
    "scaled_l2proj": est_scaled_l2proj,
    "blend_0.9l2+0.1kl": make_blend(est_l2proj, est_klproj, 0.9),
}
