"""Task 1 pooled pipeline: features -> pooled LightGBM -> predictions.

Stages (python -m trafficflow.t1_pipeline <stage> ...):
  feat    per panel: simulate the val/private blackouts on train, sample training
          cells (regular targets + dark cells), extract features for them and for
          every validation/private target cell; write parquet under WORK/feat.
  train   pooled models on all panels: speed, flow (regular), gspeed, gflow (dark
          cells); `--holdout` keeps train days >= HOLD out for scoring.
  predict write WORK/pred/state_<tag>.parquet keyed like the Task 1 templates.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import PANELS, REL, SLOTS, SPLIT_DAYS
from .t1 import Panel

WORK = Path(os.environ.get("TFB_WORK", "/home/user/work/t1"))
HOLD = 243  # first holdout day of train (last 30 train days)
N_REG = 300_000  # regular training cells per panel
REGIME_NAME = {1: "R1", 2: "R2", 3: "R3"}


def _meta(P: Panel, tt, ll, kind):
    return pd.DataFrame({"panel": P.panel, "t": tt.astype(np.int32), "j": ll.astype(np.int16), "kind": kind,
                         "day": (tt // SLOTS).astype(np.int16),
                         "treg": P.target[tt, ll].astype(np.int8),
                         "y_speed": P.tspeed[tt, ll], "y_flow": P.tflow[tt, ll]})


def feat_panel(panel: str, seed: int = 0):
    out = WORK / "feat"; out.mkdir(parents=True, exist_ok=True)
    if (out / f"{panel}_test.parquet").exists():
        return
    t0 = time.time()
    P = Panel(panel)
    origins = P.select_origins(0, SPLIT_DAYS["train"][1], spacing=36)
    P.apply_blackouts(origins)
    rng = np.random.default_rng(seed)
    ntr = SPLIT_DAYS["train"][1] * SLOTS
    # regular targets (train, not dark)
    tt, ll = np.nonzero((P.target[:ntr] > 0) & ~P.dark[:ntr, None])
    s = rng.choice(len(tt), min(N_REG, len(tt)), replace=False)
    tt, ll = tt[s], ll[s]
    # all train holdout targets are kept for scoring
    ht, hl = np.nonzero((P.target[HOLD * SLOTS:ntr] > 0) & ~P.dark[HOLD * SLOTS:ntr, None])
    ht = ht + HOLD * SLOTS
    s = rng.choice(len(ht), min(150_000, len(ht)), replace=False); ht, hl = ht[s], hl[s]
    keep = tt < HOLD * SLOTS
    tt, ll = np.concatenate([tt[keep], ht]), np.concatenate([ll[keep], hl])
    # dark cells in train (eligible), sampled
    dk = np.nonzero(P.dark[:ntr])[0]
    dt_, dl_ = np.nonzero(P.elig[dk]); dt_ = dk[dt_]
    s = rng.choice(len(dt_), min(400_000, len(dt_)), replace=False); dt_, dl_ = dt_[s], dl_[s]
    kinds = np.array(["reg"] * len(tt) + ["dark"] * len(dt_))
    tt, ll = np.concatenate([tt, dt_]), np.concatenate([ll, dl_])
    parts = []
    for i in range(0, len(tt), 200_000):
        t2, l2 = tt[i:i + 200_000], ll[i:i + 200_000]
        m = _meta(P, t2, l2, "reg"); m["kind"] = kinds[i:i + 200_000]
        f = P.features(t2, l2)
        f64 = f.select_dtypes("float64").columns; f[f64] = f[f64].astype(np.float32)
        parts.append(pd.concat([m, f], axis=1))
    pd.concat(parts, ignore_index=True).to_parquet(out / f"{panel}_train.parquet")
    del parts; gc.collect()
    # test targets: validation + private
    a = SPLIT_DAYS["validation"][0] * SLOTS
    tt, ll = np.nonzero(P.target[a:] > 0); tt = tt + a
    parts = []
    for i in range(0, len(tt), 200_000):
        t2, l2 = tt[i:i + 200_000], ll[i:i + 200_000]
        m = _meta(P, t2, l2, "test")
        m["kind"] = np.where(P.dark[t2], "dark", "reg")
        m["link_id"] = P.links[l2]
        f = P.features(t2, l2)
        f64 = f.select_dtypes("float64").columns; f[f64] = f[f64].astype(np.float32)
        parts.append(pd.concat([m, f], axis=1))
    pd.concat(parts, ignore_index=True).to_parquet(out / f"{panel}_test.parquet")
    print(f"{panel}: feat done in {time.time() - t0:.0f}s, origins={len(origins)}", flush=True)


NON_FEAT = {"panel", "t", "j", "kind", "day", "treg", "y_speed", "y_flow", "link_id", "y_dens"}


CAP = {"reg": (int(os.environ.get("TFB_NREG", 150_000)), 100_000), "dark": (int(os.environ.get("TFB_NDARK", 150_000)), 60_000)}  # (train rows, holdout rows) per panel


def load_train(panels, kind, seed=0):
    rng = np.random.default_rng(seed)
    dfs = []
    for p in panels:
        d = pd.read_parquet(WORK / "feat" / f"{p}_train.parquet")
        d = d[d.kind == kind]
        tr = np.nonzero((d.day < HOLD).to_numpy())[0]; ho = np.nonzero((d.day >= HOLD).to_numpy())[0]
        ctr, cho = CAP[kind]
        tr = rng.choice(tr, min(ctr, len(tr)), replace=False); ho = rng.choice(ho, min(cho, len(ho)), replace=False)
        d = d.iloc[np.sort(np.concatenate([tr, ho]))]
        f64 = d.select_dtypes("float64").columns
        d[f64] = d[f64].astype(np.float32)
        dfs.append(d)
        del d; gc.collect()
    d = pd.concat(dfs, ignore_index=True)
    d["panel_id"] = d.panel.map({p: i for i, p in enumerate(PANELS)}).astype("int16")
    d["y_dens"] = d.y_flow / np.maximum(d.y_speed, 1.0)
    return d


def base_of(d, c):
    if c == "dens":
        return base_of(d, "flow") / np.maximum(base_of(d, "speed"), 1.0)
    return d[f"li_{c}"].fillna(d[f"h_{c}"]).to_numpy()


PARAMS = dict(objective="regression", learning_rate=float(os.environ.get("TFB_LR", 0.1)), num_leaves=255,
              min_data_in_leaf=100, feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1, lambda_l2=2.0, max_bin=63,
              num_threads=int(os.environ.get("TFB_THREADS", "3")), verbose=-1)


def train(panels, holdout: bool, tag: str, rounds: dict | None = None):
    mdir = WORK / "models" / tag; mdir.mkdir(parents=True, exist_ok=True)
    report = {}
    for kind, targets in (("reg", ("speed", "flow", "dens")), ("dark", ("speed", "flow", "dens"))):
        d = load_train(panels, kind)
        feats = [c for c in d.columns if c not in NON_FEAT]
        tr = d.day < HOLD if holdout else np.ones(len(d), bool)
        va = d.day >= HOLD
        for c in targets:
            name = f"{kind}_{c}"
            if (mdir / f"{name}.txt").exists() and (not holdout or (mdir / f"hold_{name}.npy").exists()):
                print("skip", name, flush=True)
                continue
            y = d[f"y_{c}"].to_numpy() - base_of(d, c)
            ok = np.isfinite(y)
            w = (d.length * d.lanes).to_numpy() if c == "dens" else None
            dtr = lgb.Dataset(d.loc[tr & ok, feats], y[tr & ok], weight=None if w is None else w[tr & ok],
                              categorical_feature=["panel_id"], free_raw_data=True)
            p = dict(PARAMS)
            if c == "dens":
                p.update(objective="huber", alpha=float(os.environ.get("TFB_HUBER", 1.0)))
            if kind == "dark":
                p.update(num_leaves=63, min_data_in_leaf=200, learning_rate=0.05)
            t0 = time.time()
            if holdout:
                dva = lgb.Dataset(d.loc[va & ok, feats], y[va & ok], weight=None if w is None else w[va & ok], reference=dtr)
                m = lgb.train(p, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False),
                                                                        lgb.log_evaluation(250)])
                pred = m.predict(d.loc[va, feats], num_iteration=m.best_iteration) + base_of(d[va], c)
                rm = float(np.sqrt(np.nanmean((pred - d.loc[va, f"y_{c}"].to_numpy()) ** 2)))
                report[name] = dict(best_iter=m.best_iteration, rmse=rm)
                np.save(mdir / f"hold_{name}.npy", pred)
            else:
                n = (rounds or {}).get(name, 1500)
                m = lgb.train(p, dtr, n)
            m.save_model(str(mdir / f"{name}.txt"))
            print(name, report.get(name), f"{time.time() - t0:.0f}s", flush=True)
        del d; gc.collect()
    json.dump(report, open(mdir / "report.json", "w"), indent=1)
    return report


def predict(panels, tag: str):
    mdir = WORK / "models" / tag
    out = WORK / "pred"; out.mkdir(parents=True, exist_ok=True)
    models = {n: lgb.Booster(model_file=str(mdir / f"{n}.txt")) for n in
              ("reg_speed", "reg_flow", "reg_dens", "dark_speed", "dark_flow", "dark_dens")}
    frames = []
    for p in panels:
        d = pd.read_parquet(WORK / "feat" / f"{p}_test.parquet")
        d["panel_id"] = np.int16(PANELS.index(p))
        feats = models["reg_speed"].feature_name()
        sp = np.empty(len(d)); fl = np.empty(len(d)); dn = np.empty(len(d))
        for kind in ("reg", "dark"):
            m = (d.kind == kind).to_numpy()
            if m.any():
                X = d.loc[m, feats]
                sp[m] = models[f"{kind}_speed"].predict(X) + base_of(d[m], "speed")
                fl[m] = models[f"{kind}_flow"].predict(X) + base_of(d[m], "flow")
                dn[m] = models[f"{kind}_dens"].predict(X) + base_of(d[m], "dens")
        frames.append(pd.DataFrame({"panel": p, "t": d.t, "link_id": d.link_id, "regime": d.treg,
                                    "kind": d.kind, "speed": sp, "flow_lane": fl, "dens_lane": dn,
                                    "lanes": d.lanes}))
    res = pd.concat(frames, ignore_index=True)
    res.to_parquet(out / f"state_{tag}.parquet")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--panels", nargs="*", default=PANELS)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args()
    if a.stage == "feat":
        for p in a.panels:
            feat_panel(p)
            gc.collect()
    elif a.stage == "train":
        print(train(a.panels, a.holdout, a.tag))
    elif a.stage == "predict":
        predict(a.panels, a.tag)
