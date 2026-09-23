"""Final Task 2 pipeline: train on all train windows, predict the official
validation + private windows, decode, write a template-exact CSV.

    PYTHONPATH=/home/user/knee OMP_NUM_THREADS=2 python -m trafficflow.t2.pipeline NAME \
        [--onset-cfg fast|p1|p2] [--ongoing-cfg fast|p1|p2] [--onset-weighted] [--ongoing-weighted]
        [--onset-dec topm|range|thrX] [--ongoing-dec topm|thrX]
"""
from __future__ import annotations

import argparse
import gc
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from .core import FEAT, K, PANELS8, WORK, statics
from .cv import NONFEAT, gather, window_weights
from .models import CFG, PARAMS, eiou_range, eiou_topm
from .oprior import COLS as OP_COLS, OnsetPrior, add_to_rows
from .submit import check, write




def train_full(cond: str, rounds: int, params=PARAMS, cand_frac: float = 1.0, weighted=False, oprior=False, seed=0):
    R, M = gather(cond, cand_frac=cand_frac)
    if oprior:
        R = add_to_rows(R, M, PANELS8)
    ds = lgb.Dataset(R.X, R.y.astype(np.float32), feature_name=list(R.cols),
                     weight=window_weights(R.gw) if weighted else None,
                     params={"verbose": -1, "max_bin": params.get("max_bin", 255)}).construct()
    cols = list(R.cols)
    del R, M
    gc.collect()
    m = lgb.train({**params, "seed": seed}, ds, rounds)
    m.free_dataset()
    return m, cols


def load_split(split: str, cond: str, cols):
    Xs = []
    for p in PANELS8:
        X = pd.read_parquet(FEAT / f"feat_{p}_{split}.parquet")
        M = pd.read_parquet(FEAT / f"meta_{p}_{split}.parquet").set_index("w")
        X["pcode"] = np.float32(PANELS8.index(p))
        X["window_id"] = X.w.map(M.window_id)
        X = X[X.w.map(M.condition) == cond].reset_index(drop=True)
        if any(c in cols for c in OP_COLS):
            ws = np.unique(X.w)
            f = OnsetPrior(p).features(M.loc[ws, "T"].to_numpy(), None)
            pos = {w: i for i, w in enumerate(ws)}
            wi = X.w.map(pos).to_numpy(); li = X.link.to_numpy().astype(int)
            for c in OP_COLS:
                X[c] = f[c][wi, li]
        Xs.append(X)
    return pd.concat(Xs, ignore_index=True)


def decode(dec: str, p, k, link, L):
    if dec == "topm":
        s = np.zeros(len(p), bool); s[eiou_topm(p)[0]] = True; return s
    if dec == "range":
        full = np.zeros(L); full[link] = p
        a, b, _ = eiou_range(full); return (link >= a) & (link <= b)
    if dec.startswith("thr"):
        return p >= float(dec[3:])
    raise ValueError(dec)


def predict(models: dict, decs: dict, splits=("validation", "private")) -> tuple[dict, pd.DataFrame]:
    preds = {}
    probs = []
    for split in splits:
        for cond, (m, cols) in models.items():
            X = load_split(split, cond, cols)
            X["p"] = m.predict(X[cols].to_numpy(np.float32), num_threads=2)
            probs.append(X[["window_id", "panel", "k", "link", "p"]].assign(split=split, condition=cond))
            for wid, g in X.groupby("window_id"):
                panel = g.panel.iloc[0]
                L = statics(panel)["L"]
                kk = g.k.to_numpy().astype(int) - 1; ll = g.link.to_numpy().astype(int)
                sel = decode(decs[cond], g.p.to_numpy(), kk, ll, L)
                P = np.zeros((K, L), bool)
                P[kk[sel], ll[sel]] = True
                preds[wid] = P
    return preds, pd.concat(probs, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--onset-dec", default="topm")
    ap.add_argument("--ongoing-dec", default="topm")
    ap.add_argument("--onset-cfg", default="p1", choices=list(CFG))
    ap.add_argument("--ongoing-cfg", default="p2", choices=list(CFG))
    ap.add_argument("--ongoing-cand-frac", type=float, default=0.5)
    ap.add_argument("--onset-weighted", action="store_true", help="equal total weight per window")
    ap.add_argument("--ongoing-weighted", action="store_true", help="equal total weight per window")
    ap.add_argument("--no-oprior", action="store_true", help="drop the onset location-prior features")
    a = ap.parse_args()
    (po, ro), (pg, rg) = CFG[a.onset_cfg], CFG[a.ongoing_cfg]
    models = {"queue_onset": train_full("queue_onset", ro, po, weighted=a.onset_weighted, oprior=not a.no_oprior),
              "queue_ongoing": train_full("queue_ongoing", rg, pg, cand_frac=a.ongoing_cand_frac,
                                          weighted=a.ongoing_weighted)}
    for c, (m, cols) in models.items():
        m.save_model(str(WORK / f"model_{a.name}_{c}.txt"))
    preds, probs = predict(models, {"queue_onset": a.onset_dec, "queue_ongoing": a.ongoing_dec})
    probs.to_parquet(WORK / f"probs_{a.name}.parquet")
    out = WORK / f"{a.name}.csv"
    write(preds, out)
    print(json.dumps(check(out)))


if __name__ == "__main__":
    main()
