"""Better train truth at missing cells (label reconstruction for Task 2).

The train truth used so far fills cells without an observation (~17%) by time
interpolation first. At the first queued slot of a new queue that is badly
biased: a masked-cell experiment on onset slots (cells with an observation,
hidden and re-filled) recovers only 24% of the queued cells with time-first
interpolation, 52% with space-first and 48% with their average (accuracy
0.79 / 0.82 / 0.86). Queue onset is sharp in time but smooth in space.

``Imputer`` learns P(queued) of a cell from its space-time neighbours
(t +-1,2 on the link, links +-1,2 at t, the four diagonals), trained on
observed cells (hidden) of half of the train days and checked on the other
half, and ``fixed_truth`` rebuilds the queue truth: observation where
available, imputer (p >= 0.5) where missing.

    python -m trafficflow.t2.truthfix fit      # fit + masked check
    python -m trafficflow.t2.truthfix relabel  # WORK/ds_<panel>_y2.npz for every panel
"""
from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..data import SLOTS, load
from .core import K, PANELS8, TRAIN_END, WORK, statics

OFFS = [(-1, 0), (1, 0), (-2, 0), (2, 0), (0, -1), (0, 1), (0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]
NAMES = [f"r_t{dt}_l{dl}" for dt, dl in OFFS]
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, num_threads=2, verbose=-1, seed=0)


def neighbour_features(R: np.ndarray, t: np.ndarray, l: np.ndarray) -> np.ndarray:
    """R [T, L] speed/v_cut ratio (NaN missing); cells (t, l) -> [n, 12+2]."""
    Tn, L = R.shape
    X = np.full((len(t), len(OFFS) + 2), np.nan, np.float32)
    for j, (dt, dl) in enumerate(OFFS):
        tt = t + dt; ll = l + dl
        ok = (tt >= 0) & (tt < Tn) & (ll >= 0) & (ll < L)
        X[ok, j] = R[tt[ok], ll[ok]]
    X[:, -2] = np.nanmean(X[:, 0:2], 1)           # time average
    X[:, -1] = np.nanmean(X[:, 4:6], 1)           # space average
    return X


def ratio(panel: str) -> np.ndarray:
    d = load(panel)
    R = d["tspeed"][:TRAIN_END] / statics(panel)["vcut"]
    del d
    return R.astype(np.float32)


def sample_cells(R: np.ndarray, day_ok: np.ndarray, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Observed cells near congestion (some neighbour within 1 slot/link below 1.3)."""
    Tn, L = R.shape
    near = np.zeros_like(R, bool)
    slow = np.nan_to_num(R, nan=9.0) <= 1.3
    for dt in (-1, 0, 1):
        for dl in (-1, 0, 1):
            near |= np.roll(np.roll(slow, dt, 0), dl, 1)
    cand = near & ~np.isnan(R)
    cand &= np.repeat(day_ok, SLOTS)[:Tn, None]
    t, l = np.nonzero(cand)
    if len(t) > n:
        i = rng.choice(len(t), n, replace=False); t, l = t[i], l[i]
    return t, l


def fit(n_per_panel: int = 150_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    Xs, ys, te = [], [], []
    for p in PANELS8:
        R = ratio(p)
        nd = R.shape[0] // SLOTS
        day_tr = (np.arange(nd) % 2) == 0
        t, l = sample_cells(R, day_tr, n_per_panel, rng)
        Xs.append(neighbour_features(R, t, l)); ys.append((R[t, l] <= 1).astype(np.float32))
        del R
    X = np.vstack(Xs); y = np.concatenate(ys)
    m = lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=NAMES + ["r_tavg", "r_savg"]), 300)
    m.save_model(str(WORK / "truthfix_imputer.txt"))
    return m


def check(m: lgb.Booster):
    """Masked check on the onset / ongoing slots of fill_eval (odd days only)."""
    rows = []
    for p in PANELS8:
        R = ratio(p); nd = R.shape[0] // SLOTS
        z = np.load(WORK / f"ds_{p}.npz", allow_pickle=True)
        for cond, steps in (("queue_onset", [6]), ("queue_ongoing", [1, 3, 6])):
            sel = np.flatnonzero((z["w_condition"] == cond) & (z["w_src"] == "sim"))
            T = z["w_T"][sel].astype(int); y = z["y"][sel]
            for j, t0 in enumerate(T):
                if (t0 // SLOTS) % 2 == 0:
                    continue
                for k in steps:
                    s = t0 + k; site = np.flatnonzero(y[j, k - 1])
                    if not len(site):
                        continue
                    ls = np.arange(max(0, site.min() - 4), min(R.shape[1], site.max() + 5))
                    ls = ls[~np.isnan(R[s, ls])]
                    X = neighbour_features(R, np.full(len(ls), s), ls)
                    q = R[s, ls] <= 1
                    pm = m.predict(X, num_threads=2) >= 0.5
                    tv = X[:, -2] <= 1; sv = X[:, -1] <= 1
                    both = ~np.isnan(X[:, -2]) & ~np.isnan(X[:, -1])
                    for a, b, c, d_, e in zip(q[both], pm[both], tv[both], sv[both], ((X[:, -2] + X[:, -1]) / 2 <= 1)[both]):
                        rows.append((cond, a, b, c, d_, e))
        del R
    r = pd.DataFrame(rows, columns=["cond", "q", "imp", "time", "space", "avg"])
    for cond, g in r.groupby("cond"):
        print(cond, "cells", len(g), "queued share %.3f" % g.q.mean())
        for mth in ("time", "space", "avg", "imp"):
            print("   %-5s acc %.3f recall %.3f spec %.3f" % (mth, (g[mth] == g.q).mean(), g[g.q][mth].mean(),
                                                           1 - g[~g.q][mth].mean()))


def fixed_truth(panel: str, m: lgb.Booster) -> np.ndarray:
    """[TRAIN_END, L] bool queue truth: observation where available, imputer elsewhere."""
    R = ratio(panel)
    Q = np.nan_to_num(R, nan=9.0) <= 1.0
    t, l = np.nonzero(np.isnan(R))
    for c0 in range(0, len(t), 2_000_000):
        tt, ll = t[c0:c0 + 2_000_000], l[c0:c0 + 2_000_000]
        X = neighbour_features(R, tt, ll)
        allnan = np.isnan(X).all(1)
        p = m.predict(X, num_threads=2)
        Q[tt, ll] = (p >= 0.5) & ~allnan
    return Q


def hybrid_truth(panel: str, m: lgb.Booster, dt: int = 1, dl: int = 2) -> np.ndarray:
    """Old (time-first) truth everywhere, except missing cells with an observed
    queued cell within +-dt slots and +-dl links, where the imputer decides.
    The imputer alone creates scattered false queue cells in free flow (at
    threshold 0.5 over millions of missing cells), which breaks the selector
    reproduction; restricting it to congestion neighbourhoods keeps the
    onset-slot fix without them."""
    R = ratio(panel)
    z = np.load(WORK / f"ds_{panel}.npz", allow_pickle=True)
    L = int(z["L"])
    Q = np.unpackbits(z["qtrue"], axis=1, count=L).astype(bool)[:R.shape[0]]
    obsq = np.nan_to_num(R, nan=9.0) <= 1.0
    near = np.zeros_like(obsq)
    for a in range(-dt, dt + 1):
        for b in range(-dl, dl + 1):
            near |= np.roll(np.roll(obsq, a, 0), b, 1)
    t, l = np.nonzero(np.isnan(R) & near)
    for c0 in range(0, len(t), 2_000_000):
        tt, ll = t[c0:c0 + 2_000_000], l[c0:c0 + 2_000_000]
        Q[tt, ll] = m.predict(neighbour_features(R, tt, ll), num_threads=2) >= 0.5
    return Q


def relabel(m: lgb.Booster, mode: str = "imputer"):
    for p in PANELS8:
        Q = fixed_truth(p, m) if mode == "imputer" else hybrid_truth(p, m)
        z = np.load(WORK / f"ds_{p}.npz", allow_pickle=True)
        T = z["w_T"].astype(np.int64)
        fut = T[:, None] + np.arange(1, K + 1)[None, :]
        y2 = Q[fut]
        y1 = z["y"]
        np.savez_compressed(WORK / f"ds_{p}_y2.npz", y=y2, qtrue=np.packbits(Q, axis=1), L=Q.shape[1])
        on = z["w_condition"] == "queue_onset"
        print(p, "cells changed: onset %.4f ongoing %.4f | onset T+30 queued cells old %.0f new %.0f" % (
            (y1[on] != y2[on]).mean(), (y1[~on] != y2[~on]).mean(), y1[on][:, K - 1].sum(), y2[on][:, K - 1].sum()),
            flush=True)
        del Q


def make_tables(src_dir: str = "/home/user/work/t2/feat_v3", dst_dir: str = "/home/user/work/t2/feat_v3y"):
    """Onset-only feature tables relabelled with the imputed T+30 truth.
    Onset windows whose imputed T+30 truth is empty are artifacts of the old
    truth (the real selector would not draw them) and are dropped."""
    import shutil
    from pathlib import Path
    src, dst = Path(src_dir), Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for p in PANELS8:
        y2 = np.load(WORK / f"ds_{p}_y2.npz")["y"]
        M = pd.read_parquet(src / f"meta_{p}.parquet")
        on = M.condition == "queue_onset"
        empty = on & ~y2[M.w.to_numpy(), K - 1].any(1)
        M2 = M[~empty]
        ids = M2.w[M2.condition == "queue_onset"].tolist()
        X = pd.read_parquet(src / f"feat_{p}.parquet", filters=[("w", "in", ids)])
        X["y"] = y2[X.w.to_numpy(), K - 1, X.link.to_numpy().astype(int)].astype(X.y.dtype)
        X.to_parquet(dst / f"feat_{p}.parquet"); M2.to_parquet(dst / f"meta_{p}.parquet")
        for split in ("train", "validation", "private"):
            for kind in ("feat", "meta"):
                shutil.copy(src / f"{kind}_{p}_{split}.parquet", dst / f"{kind}_{p}_{split}.parquet")
        print(p, "onset windows kept", int((M2.condition == "queue_onset").sum()), "dropped", int(empty.sum()),
              "| sim dropped", int((empty & (M.src == "sim")).sum()), "rows", len(X), "pos %.4f" % X.y.mean(), flush=True)

if __name__ == "__main__":
    if sys.argv[1] == "fit":
        m = fit()
        check(m)
    elif sys.argv[1] == "relabel":
        relabel(lgb.Booster(model_file=str(WORK / "truthfix_imputer.txt")),
                sys.argv[2] if len(sys.argv) > 2 else "imputer")
    elif sys.argv[1] == "tables":
        make_tables()

