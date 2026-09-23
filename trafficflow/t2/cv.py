"""Cross-validated evaluation of the Task 2 models on simulated train windows.

4 interleaved-week folds. For each fold the models are trained on windows of
the other folds (all sources) and scored on the held-out fold's selector-drawn
windows (``sim``) and on the official train windows (``off``), with the
official window -> panel&condition -> panel -> family aggregation.

    PYTHONPATH=/home/user/knee python -m trafficflow.t2.cv
"""
from __future__ import annotations

import json
import sys
import time

import gc
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from .baselines import NFOLD
from .core import K, PANELS8, FEAT, WORK, aggregate, iou
from .dataset import load_ds
from .models import CFG, NONFEAT, PARAMS, eiou_range, eiou_topm


TAG = os.environ.get("T2_TAG", "v2")


class Rows:
    """Feature rows of one condition as a float32 matrix plus row metadata."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __len__(self):
        return len(self.y)


def gather(cond: str, cand_frac: float = 1.0, seed: int = 0, panels=PANELS8, drop=()):
    """All train feature rows of one condition as one preallocated float32
    matrix (two passes over the parquet files, column groups at a time, so the
    peak memory is ~ the final matrix). ``cand_frac`` subsamples the extra
    (non-selector) candidate windows."""
    import pyarrow.parquet as pq
    rng = np.random.default_rng(seed)
    plan, Ms = [], []
    cols = None
    for p in panels:
        M = pd.read_parquet(FEAT / f"meta_{p}.parquet")
        M["gw"] = PANELS8.index(p) * 100000 + M.w.astype(np.int64)
        keep = (M.condition == cond).to_numpy().copy()
        if cand_frac < 1:
            keep &= (M.src != "cand").to_numpy() | (rng.random(len(M)) < cand_frac)
        M = M[keep]
        pf = pq.ParquetFile(FEAT / f"feat_{p}.parquet")
        if cols is None:
            names = pf.schema_arrow.names
            cols = [c for c in names if c not in NONFEAT and c not in set(drop)] + ["pcode"]
        w = pf.read(columns=["w"]).column("w").to_numpy()
        mask = np.isin(w, M.w.to_numpy())
        plan.append((p, pf, mask, M))
        Ms.append(M)
    n = sum(int(m.sum()) for _, _, m, _ in plan)
    X = np.empty((n, len(cols)), np.float32)
    meta = {k: np.empty(n, dt) for k, dt in (("gw", np.int64), ("k", np.int8), ("link", np.int16),
                                             ("y", np.int8), ("fold", np.int8), ("ev", bool))}
    r0 = 0
    for p, pf, mask, M in plan:
        m = int(mask.sum()); sl = slice(r0, r0 + m)
        feat = [c for c in cols if c != "pcode"]
        for j0 in range(0, len(feat), 16):
            cs = feat[j0:j0 + 16]
            t = pf.read(columns=cs)
            for c in cs:
                X[sl, cols.index(c)] = t.column(c).to_numpy()[mask]
            del t
        X[sl, cols.index("pcode")] = np.float32(PANELS8.index(p))
        t = pf.read(columns=["w", "k", "link", "y"])
        w = t.column("w").to_numpy()[mask]
        fm = M.set_index("w")
        meta["gw"][sl] = PANELS8.index(p) * 100000 + w.astype(np.int64)
        meta["k"][sl] = t.column("k").to_numpy()[mask]
        meta["link"][sl] = t.column("link").to_numpy()[mask]
        meta["y"][sl] = t.column("y").to_numpy()[mask]
        meta["fold"][sl] = fm.fold.reindex(w).to_numpy()
        meta["ev"][sl] = fm.src.reindex(w).isin(["sim", "off"]).to_numpy()
        del t
        r0 += m
    R = Rows(X=X, cols=cols, **meta)
    return R, pd.concat(Ms, ignore_index=True)


def window_weights(gw: np.ndarray) -> np.ndarray:
    """Row weights giving every window the same total weight (mean 1)."""
    _, inv, cnt = np.unique(gw, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    return (w / w.mean()).astype(np.float32)


def oof(R: Rows, params, rounds, return_models=False, weighted=False):
    """Out-of-fold probabilities for rows of eval windows (one binned Dataset,
    fold subsets share its bins to bound memory)."""
    p = np.full(len(R), np.nan, np.float32)
    ev_idx = np.flatnonzero(R.ev)
    X_ev = R.X[ev_idx]  # keep only the rows we predict; the binned Dataset holds the rest
    full = lgb.Dataset(R.X, R.y.astype(np.float32), feature_name=list(R.cols), free_raw_data=True,
                       weight=window_weights(R.gw) if weighted else None,
                       params={"max_bin": params.get("max_bin", 255), "verbose": -1}).construct()
    R.X = None  # release the raw matrix (callers must not reuse R.X after oof)
    gc.collect()
    models = []
    for f in range(NFOLD):
        tr = np.where(R.fold != f)[0]
        te_ev = R.fold[ev_idx] == f
        m = lgb.train(params, full.subset(tr), rounds)
        p[ev_idx[te_ev]] = m.predict(X_ev[te_ev], num_threads=2)
        m.free_dataset()  # a Booster keeps its (binned) training subset alive otherwise
        if return_models:
            models.append(m)
        del m
        gc.collect()
    return (p, models) if return_models else p


def truth_lookup():
    Y = {}
    for p in PANELS8:
        z = np.load(WORK / f"ds_{p}.npz", allow_pickle=True)
        Y[p] = (None, z["y"], None)
    return Y


def score_windows(R, p, M, Y, cond, decoders):
    """Per-window IoU for each decoder: f(p, k, link, L) -> bool mask over the window's rows."""
    rows = []
    ev = M[(M.condition == cond) & M.src.isin(["sim", "off"])].drop_duplicates("gw").set_index("gw")
    sel_rows = np.where(np.isin(R.gw, ev.index.to_numpy()))[0]
    order = sel_rows[np.argsort(R.gw[sel_rows], kind="stable")]
    bounds = np.flatnonzero(np.diff(R.gw[order])) + 1
    for chunk in np.split(order, bounds):
        gw = int(R.gw[chunk[0]])
        mrow = ev.loc[gw]
        _, y, _ = Y[mrow.panel]
        yt = y[int(mrow.w)]
        L = yt.shape[1]
        kk = R.k[chunk].astype(int) - 1; ll = R.link[chunk].astype(int); pp = p[chunk]
        res = dict(panel=mrow.panel, condition=cond, src=mrow.src, gw=gw)
        for name, fn in decoders.items():
            sel = fn(pp, kk, ll, L)
            pred = np.zeros_like(yt)
            pred[kk[sel], ll[sel]] = True
            res[name] = iou(pred, yt)
        rows.append(res)
    return pd.DataFrame(rows)


def dec_thr(t):
    return lambda pp, kk, ll, L: pp >= t


def dec_topm(pp, kk, ll, L):
    s = np.zeros(len(pp), bool)
    idx, _ = eiou_topm(pp)
    s[idx] = True
    return s


def dec_range(pp, kk, ll, L):
    """onset: contiguous link range at step 6."""
    full = np.zeros(L)
    full[ll] = pp
    a, b, _ = eiou_range(full)
    return (ll >= a) & (ll <= b)


def summarize(df, names):
    out = {}
    for src in ("sim", "off"):
        d = df[df.src == src]
        for n in names:
            out[f"{src}:{n}"] = round(aggregate(d.rename(columns={n: "iou"}))[d.condition.iloc[0]], 4)
    return out


def main():
    """python -m trafficflow.t2.cv {queue_onset|queue_ongoing|both} [CFG] [--weighted] [--oprior]

    CFG in models.CFG (fast, p1, p2). Final settings: onset ``p1 --oprior``
    (sim 0.713), ongoing ``p2 --weighted`` (sim 0.877)."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="both")
    ap.add_argument("cfg", nargs="?", default="fast")
    ap.add_argument("--weighted", action="store_true")
    ap.add_argument("--oprior", action="store_true")
    a = ap.parse_args()
    params, rounds = CFG[a.cfg]
    tag = f"{TAG}_{a.cfg}{'_w' if a.weighted else ''}{'_op' if a.oprior else ''}"
    Y = truth_lookup()
    res = {}
    for cond in (["queue_onset", "queue_ongoing"] if a.which == "both" else [a.which]):
        t = time.time()
        R, M = gather(cond, cand_frac=0.5 if cond == "queue_ongoing" else 1.0)
        if a.oprior and cond == "queue_onset":
            from .oprior import add_to_rows
            R = add_to_rows(R, M, PANELS8)
        meta = dict(gw=R.gw, k=R.k, link=R.link, y=R.y, ev=R.ev)
        p = oof(R, params, rounds, weighted=a.weighted)
        ev = meta["ev"]
        pd.DataFrame({k: v[ev] for k, v in meta.items() if k != "ev"} | {"p": p[ev]}).to_parquet(
            WORK / f"oof_{cond}_{tag}.parquet")
        M.to_parquet(WORK / f"oofmeta_{cond}_{tag}.parquet")
        decs = {"topm": dec_topm}
        for t_ in (0.3, 0.4, 0.5):
            decs[f"thr{t_}"] = dec_thr(t_)
        if cond == "queue_onset":
            decs["range"] = dec_range
        df = score_windows(R, p, M, Y, cond, decs)
        df.to_parquet(WORK / f"cv_{cond}_{tag}.parquet")
        res[cond] = summarize(df, list(decs))
        print(cond, tag, res[cond], f"{time.time()-t:.0f}s", flush=True)
    return res


if __name__ == "__main__":
    main()
