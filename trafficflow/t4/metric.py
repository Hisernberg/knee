"""Exact Task 4 metric (docs/SCORING_SPEC.md) for one panel, given a truth vector."""
from __future__ import annotations

import numpy as np


def attraction(f: np.ndarray, dest: np.ndarray, nz: int) -> np.ndarray:
    a = np.bincount(dest, weights=f, minlength=nz)
    return a / max(a.sum(), 1e-9)


def s_link(Am: np.ndarray, c: np.ndarray, f: np.ndarray) -> float:
    return max(0.0, 1.0 - np.abs(Am @ f - c).sum() / max(c.sum(), 1e-9))


def score(f, fstar, b, Am, c, dest, nz) -> dict:
    s_od = max(0.0, 1.0 - np.abs(f - fstar).sum() / max(fstar.sum(), 1e-9))
    sl = s_link(Am, c, f)
    dhat = np.abs(f - b).sum()
    dstar = np.abs(fstar - b).sum()
    if dstar <= 1e-9:
        s_dev = 1.0 if dhat <= 1e-9 else 0.0
    else:
        s_dev = float(np.exp(-abs(dhat / dstar - 1.0)))
    s_attr = max(0.0, 1.0 - 0.5 * np.abs(attraction(f, dest, nz) - attraction(fstar, dest, nz)).sum())
    tot = 0.45 * s_od + 0.25 * sl + 0.15 * s_dev + 0.15 * s_attr
    return {"S_od": s_od, "S_link": sl, "S_dev": s_dev, "S_attr": s_attr, "S_ODME": tot,
            "Dhat_over_Dstar": dhat / max(dstar, 1e-9)}
