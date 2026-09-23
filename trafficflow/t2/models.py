"""LightGBM cell classifiers for Task 2 with IoU-aware decoding.

Two models, pooled over the 8 scored panels:

* ongoing: P(queued) per (window, link, step) on cells near queue activity;
* onset:   P(queued at T+30) per (window, link); steps 1-5 are left empty
  because the selector makes the onset horizon empty before T+30.

Decoding turns probabilities into the set that maximises expected IoU,
approximated by E|S n Y| / E|S u Y| = sum_S p / (|S| + sum_notS p), searched
over top-m sets (ongoing) or top-m / contiguous ranges (onset).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import K, PANELS8, FEAT, WORK

NONFEAT = {"w", "link", "y", "panel", "fold"}
PANEL_CODE = {p: i for i, p in enumerate(PANELS8)}

PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=31, min_data_in_leaf=200,
              feature_fraction=0.7, bagging_fraction=0.5, bagging_freq=1, lambda_l2=1.0,
              max_bin=63, num_threads=2, verbose=-1, seed=0)
P_BIG = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, num_threads=2, verbose=-1, seed=0)
P_HUGE = {**P_BIG, "num_leaves": 127, "min_data_in_leaf": 50}
# name -> (params, rounds). Final: onset "p1" (+ onset prior), ongoing "p2" (+ window weights)
CFG = {"fast": (PARAMS, 250), "p1": (P_BIG, 400), "p2": (P_HUGE, 600)}


def load_train(panels=PANELS8):
    Xs, Ms = [], []
    for p in panels:
        X = pd.read_parquet(FEAT / f"feat_{p}.parquet")
        M = pd.read_parquet(FEAT / f"meta_{p}.parquet")
        X["pcode"] = np.int16(PANEL_CODE[p])
        Xs.append(X); Ms.append(M)
    return Xs, pd.concat(Ms, ignore_index=True)


def load_test(split, panels=PANELS8):
    Xs, Ms = [], []
    for p in panels:
        X = pd.read_parquet(FEAT / f"feat_{p}_{split}.parquet")
        M = pd.read_parquet(FEAT / f"meta_{p}_{split}.parquet")
        X["pcode"] = np.int16(PANEL_CODE[p])
        Xs.append(X); Ms.append(M)
    return pd.concat(Xs, ignore_index=True), pd.concat(Ms, ignore_index=True)


def feat_cols(X, drop=()):
    return [c for c in X.columns if c not in NONFEAT and c not in set(drop)]


# ----------------------------------------------------------------------------
# decoding
def eiou_topm(p: np.ndarray, max_m: int | None = None) -> tuple[np.ndarray, float]:
    """Indices of the top-m set maximising sum_S p / (|S| + sum_notS p)."""
    o = np.argsort(-p)
    cp = np.cumsum(p[o]); tot = p.sum()
    m = np.arange(1, len(p) + 1)
    r = cp / (m + tot - cp)
    if max_m:
        r = r[:max_m]
    j = int(np.argmax(r))
    return o[: j + 1], float(r[j])


def eiou_range(p: np.ndarray, max_len: int = 60) -> tuple[int, int, float]:
    """Contiguous [a, b] maximising sum_S p / (|S| + sum_notS p)."""
    L = len(p)
    C = np.concatenate([[0.0], np.cumsum(p)]); tot = C[-1]
    best = (0, 0, -1.0)
    for ln in range(1, min(max_len, L) + 1):
        s = C[ln:] - C[:-ln]
        r = s / (ln + tot - s)
        a = int(np.argmax(r))
        if r[a] > best[2]:
            best = (a, a + ln - 1, float(r[a]))
    return best
