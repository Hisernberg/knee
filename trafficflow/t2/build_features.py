"""Compute and cache the cell feature tables for the Task 2 models.

Per panel, writes ``WORK/feat_<panel>.parquet`` with one row per
(window, link, step) for

* ongoing windows: every step, restricted to cells "near" queue activity
  (queued / slow in the history, or a queue within 12 km downstream, or a
  historically congested cell); all other cells are predicted 0;
* onset windows: step 6 only, every link.

Train windows (sim/off/cand; cand ongoing capped per panel) use the time-of-day
profile learned without their own fold (4 interleaved week folds), so CV is
honest; validation/private windows use the full-train profile and the released
window history only.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

from ..data import SLOTS
from .baselines import NFOLD, fold_of
from .core import K, PANELS8, FEAT, WORK, direction, official_history, official_windows, statics
from .dataset import load_ds
from ..data import load
from .extra import MaskedView, extra_features
from .features import _nearest_dist, expand_steps, profiles, window_link_features
from .physics import lwr_features, onset_physics

PHYSICS = os.environ.get("T2_PHYSICS", "0") == "1"   # v3 feature set (physics + LWR)
NORMDIR = os.environ.get("T2_NORMDIR", "0") == "1"   # v4: traffic-direction normalised features
ONLY = os.environ.get("T2_ONLY")                     # build one condition only (e.g. queue_onset)
CONDS = (ONLY,) if ONLY else ("queue_onset", "queue_ongoing")

MAX_CAND_ONGOING = 1200
CHUNK = 150


def near_mask(df: pd.DataFrame) -> np.ndarray:
    return ((df.ddn_now.to_numpy() <= 12) | (df.dup_now.to_numpy() <= 3) | (df.r_min.to_numpy() <= 1.5)
            | (df.r_last.to_numpy() <= 1.6) | (df.pq_T.to_numpy() >= 0.02)
            | (df.ddn_60.to_numpy() <= 12))


def _rows(panel, hs, hf, he, hp, T, cond, prof, mv=None, y=None, wkey=None):
    """Feature rows for a batch of windows of one condition. With T2_NORMDIR=1
    the link axis of W/S panels is reversed before computing features, so every
    directional feature means the same thing on every panel; the ``link``
    column always holds the original link index."""
    L = hs.shape[2]
    rev = NORMDIR and direction(panel) < 0
    if rev:
        hs, hf, he, hp = (a[:, :, ::-1] for a in (hs, hf, he, hp))
        prof = {k: v[..., ::-1] for k, v in prof.items()}
        mv = mv.flipped() if mv is not None else None
        y = y[:, :, ::-1] if y is not None else None
        panel = panel + "@rev"
    df, step = window_link_features(panel, hs, hf, he, hp, T, prof)
    n = hs.shape[0]
    if mv is not None:
        st = statics(panel)
        F, st2 = extra_features(mv, T, df.r_last.to_numpy().reshape(n, L))
        qn = F["r_now"] <= 1.0
        F["ddn_T"] = _nearest_dist(qn, st["mp"], "dn")
        F["dup_T"] = _nearest_dist(qn, st["mp"], "up")
        F["c_qT"] = np.repeat(qn.mean(1, keepdims=True), L, 1)
        for k_, v in F.items():
            df[k_] = np.asarray(v, np.float32).reshape(-1)
        step.update(st2)
        if PHYSICS:
            if cond == "queue_onset":
                P = onset_physics(panel, hs, hf, he, r_now=F["r_now"], f_now=F["fT"])
                for k_, v in P.items():
                    df[k_] = v.reshape(-1)
            else:
                f_last = df.f_last.to_numpy().reshape(n, L)
                f_now = np.where(np.isnan(F["fT"]), f_last, F["fT"]) * st["cap"].astype(np.float32)
                Fl, Sl = lwr_features(panel, hs, hf, F["r_now"], f_now)
                for k_, v in Fl.items():
                    df[k_] = v.reshape(-1)
                step.update(Sl)
    df["w"] = np.repeat(np.asarray(wkey), L)
    lk = np.arange(L)[::-1] if rev else np.arange(L)
    df["link"] = np.tile(lk, n).astype(np.int16)
    if cond == "queue_onset":
        X = expand_steps(df, step, steps=[K])
        if y is not None:
            X["y"] = y[:, K - 1].reshape(-1)
    else:
        m = near_mask(df)
        X = expand_steps(df, step)
        X = X[np.tile(m, K)]
        if y is not None:
            X["y"] = np.concatenate([y[:, k].reshape(-1)[m] for k in range(K)])
    return X


def build_train(panel: str, seed: int = 0) -> pd.DataFrame:
    W, A = load_ds(panel)
    rng = np.random.default_rng(seed)
    W["fold"] = fold_of(W["T"])
    keep = W.src.isin(["sim", "off"]) | (W.condition == "queue_onset")
    cand_og = W.index[(W.src == "cand") & (W.condition == "queue_ongoing")].to_numpy()
    if len(cand_og) > MAX_CAND_ONGOING:
        cand_og = rng.choice(cand_og, MAX_CAND_ONGOING, replace=False)
    keep[cand_og] = True
    W = W[keep]
    nd = A["qtrue"].shape[0] // SLOTS
    dayfold = fold_of(np.arange(nd) * SLOTS)
    mv = masked_view(panel)
    out = []
    for f in range(NFOLD):
        prof = profiles(A["qtrue"], None, dayfold != f)
        for cond in CONDS:
            ids = W.index[(W.fold == f) & (W.condition == cond)].to_numpy()
            for c0 in range(0, len(ids), CHUNK):
                b = ids[c0:c0 + CHUNK]
                X = _rows(panel, A["hs"][b], A["hf"][b], A["he"][b], A["hp"][b], W.loc[b, "T"].to_numpy(),
                          cond, prof, mv=mv, y=A["y"][b], wkey=b)
                out.append(X)
    X = pd.concat(out, ignore_index=True)
    X["panel"] = panel
    meta = W[["T", "condition", "src", "draws", "fold"]].copy()
    meta["panel"] = panel
    meta.index.name = "w"
    return X, meta.reset_index()


def masked_view(panel: str) -> MaskedView:
    d = load(panel)
    st = statics(panel)
    mv = MaskedView({k: d[k] for k in ("speed", "flow", "elig")}, st["vcut"], st["cap"])
    del d
    return mv


def build_test(panel: str, split: str, mv: MaskedView | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    W, A = load_ds(panel)
    prof = profiles(A["qtrue"], None, np.ones(A["qtrue"].shape[0] // SLOTS, bool))
    wi = official_windows(panel, split)
    hist = official_history(panel, split)
    out = []
    for cond in CONDS:
        sub = wi[wi.condition == cond]
        if not len(sub):
            continue
        hs = np.stack([hist[w]["speed"] for w in sub.window_id]); hf = np.stack([hist[w]["flow"] for w in sub.window_id])
        he = np.stack([hist[w]["elig"] for w in sub.window_id]); hp = np.stack([hist[w]["pct"] for w in sub.window_id])
        out.append(_rows(panel, hs, hf, he, hp, sub["T"].to_numpy(), cond, prof, mv=mv, wkey=sub.index.to_numpy()))
    X = pd.concat(out, ignore_index=True)
    X["panel"] = panel
    meta = wi[["window_id", "T", "condition"]].copy(); meta["panel"] = panel
    meta.index.name = "w"
    return X, meta.reset_index()


def main(panels):
    for p in panels:
        t = time.time()
        X, meta = build_train(p)
        X.to_parquet(FEAT / f"feat_{p}.parquet"); meta.to_parquet(FEAT / f"meta_{p}.parquet")
        mv = masked_view(p)
        for split in ("train", "validation", "private"):
            Xt, mt = build_test(p, split, mv)
            Xt.to_parquet(FEAT / f"feat_{p}_{split}.parquet"); mt.to_parquet(FEAT / f"meta_{p}_{split}.parquet")
        print(p, X.shape, "pos", round(float(X.y.mean()), 4), f"{time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or PANELS8)
