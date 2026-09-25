"""Shift-robust Task 2 file: reuse / train models, blend ongoing probabilities,
decode, write the template-exact CSV.

    python -m trafficflow.t2.robust_pipeline NAME --onset-model lgb_v3 \
        --ongoing "lgb_v3:1.0" [--ongoing "train:noloc:p2:w:noop:0.5" ...] [--gate THR:IDX]

Each ``--onset`` / ``--ongoing`` spec is either ``<saved model name>:<weight>``
(loads WORK/model_<name>_<condition>.txt) or
``train:<variant>:<cfg>:<w|nw>:<op|noop>:<weight>[:<seed>]`` (trains on all
train windows with robust.VARIANTS[variant] dropped, optionally with the onset
location prior, and saves it). Probabilities of a condition are the weighted
mean of its components. Feature tables come from $T2_FEAT (feat_v3 has every
v2 column, so v2 models can be mixed in). ``--gate THR:IDX`` uses component IDX alone on windows whose
recurrence (robust.recurrence, data <= T only) is below THR.
"""
from __future__ import annotations

import argparse
import gc
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from .core import K, WORK, statics
from .cv import gather, window_weights
from .models import CFG, eiou_topm
from .pipeline import load_split
from .robust import VARIANTS, recurrence
from .submit import check, write


def train_variant(variant: str, cfg: str, weighted: bool, cond="queue_ongoing", seed=0, oprior=False):
    params, rounds = CFG[cfg]
    R, M = gather(cond, cand_frac=0.5 if cond == "queue_ongoing" else 1.0, drop=VARIANTS[variant])
    if oprior:
        from .core import PANELS8
        from .oprior import add_to_rows
        R = add_to_rows(R, M, PANELS8)
    ds = lgb.Dataset(R.X, R.y.astype(np.float32), feature_name=list(R.cols),
                     weight=window_weights(R.gw) if weighted else None,
                     params={"verbose": -1, "max_bin": params.get("max_bin", 255)}).construct()
    del R, M
    gc.collect()
    m = lgb.train({**params, "seed": seed}, ds, rounds)
    m.free_dataset()
    return m


def probs(m: lgb.Booster, split: str, cond: str) -> pd.DataFrame:
    cols = m.feature_name()
    X = load_split(split, cond, cols)
    X["p"] = m.predict(X[cols].to_numpy(np.float32), num_threads=2)
    return X[["window_id", "panel", "k", "link", "p"]]


def decode_topm(P: pd.DataFrame) -> dict:
    out = {}
    for wid, g in P.groupby("window_id"):
        L = statics(g.panel.iloc[0])["L"]
        kk = g.k.to_numpy().astype(int) - 1; ll = g.link.to_numpy().astype(int)
        idx, _ = eiou_topm(g.p.to_numpy())
        A = np.zeros((K, L), bool); A[kk[idx], ll[idx]] = True
        out[wid] = A
    return out


def components(specs: list[str], cond: str, name: str) -> list[tuple[lgb.Booster, float]]:
    """Parse model specs: ``<saved>:<weight>`` loads WORK/model_<saved>_<cond>.txt;
    ``train:<variant>:<cfg>:<w|nw>:<op|noop>:<weight>[:<seed>]`` trains on all
    train windows and saves WORK/model_<name>_<variant>[_s<seed>]_<cond>.txt."""
    out = []
    for spec in specs:
        f = spec.split(":")
        if f[0] == "train":
            variant, cfg, w, op, wt = f[1:6]
            seed = int(f[6]) if len(f) > 6 else 0
            m = train_variant(variant, cfg, w == "w", cond=cond, seed=seed, oprior=(op == "op"))
            m.save_model(str(WORK / f"model_{name}_{variant}{f'_s{seed}' if seed else ''}_{cond}.txt"))
        else:
            m = lgb.Booster(model_file=str(WORK / f"model_{f[0]}_{cond}.txt")); wt = f[1]
        out.append((m, float(wt)))
    return out


def blend_probs(comps, split: str, cond: str) -> pd.DataFrame:
    key = ["window_id", "panel", "k", "link"]
    parts = [probs(m, split, cond) for m, _ in comps]
    B = parts[0][key].copy()
    for i, q in enumerate(parts):
        B = B.merge(q.rename(columns={"p": f"p{i}"}), on=key, how="left")
    wsum = sum(w for _, w in comps)
    B["p"] = sum(w * B[f"p{i}"] for i, (_, w) in enumerate(comps)) / wsum
    return B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--onset-model", default=None, help="saved onset model name (weight 1)")
    ap.add_argument("--onset", action="append", default=[])
    ap.add_argument("--ongoing", action="append", required=True)
    ap.add_argument("--gate", default=None)
    a = ap.parse_args()
    on_specs = a.onset + ([f"{a.onset_model}:1.0"] if a.onset_model else [])
    onset = components(on_specs, "queue_onset", a.name)
    comps = components(a.ongoing, "queue_ongoing", a.name)
    preds, allp = {}, []
    key = ["window_id", "panel", "k", "link"]
    for split in ("validation", "private"):
        Po = blend_probs(onset, split, "queue_onset")
        preds.update(decode_topm(Po[key + ["p"]]))
        B = blend_probs(comps, split, "queue_ongoing")
        if a.gate:
            thr, gi = a.gate.split(":")
            rec = recurrence(split)
            low = B.window_id.map(rec).to_numpy() < float(thr)
            B["p"] = np.where(low, B[f"p{int(gi)}"], B["p"])
        preds.update(decode_topm(B[key + ["p"]]))
        allp.append(pd.concat([Po[key + ["p"]].assign(condition="queue_onset"),
                               B[key + ["p"]].assign(condition="queue_ongoing")]).assign(split=split))
    pd.concat(allp).to_parquet(WORK / f"probs_{a.name}.parquet")
    out = WORK / f"{a.name}.csv"
    write(preds, out)
    print(json.dumps(check(out)))


if __name__ == "__main__":
    main()
