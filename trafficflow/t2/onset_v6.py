"""lgb_v6: v5 with the onset part retrained on corrected (hybrid) T+30 labels.

Onset: mean of four LightGBM onset models (v3 physics features + location
prior, seeds 0/1/2; v2 features + prior), p1 config, trained on every train
onset window with the hybrid truth (truthfix.py), top-m expected-IoU decoding
at T+30 only. Ongoing: the lgb_v5 rows, copied unchanged.

    T2_WORK=... T2_FEAT=... [T2_TRUTH=y2] python -m trafficflow.t2.onset_v6 OUT.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .core import K, PANELS8, WORK, official_windows, statics, template
from .models import eiou_topm
from .robust_pipeline import probs, train_variant
from .submit import check

SPECS = [("on_v3", 0), ("on_v3", 1), ("on_v3", 2), ("on_v2", 0)]
V5 = Path("/home/user/work/t2/lgb_v5.csv")


def main(out: str):
    import lightgbm as lgb
    models = []
    for variant, seed in SPECS:
        f = WORK / f"model_v6_{variant}_s{seed}_queue_onset.txt"
        if f.exists():                      # reuse a finished training run
            m = lgb.Booster(model_file=str(f))
        else:
            m = train_variant(variant, "p1", False, cond="queue_onset", seed=seed, oprior=True)
            m.save_model(str(f))
        models.append(m)
    key = ["window_id", "panel", "k", "link"]
    preds = {}
    allp = []
    for split in ("validation", "private"):
        parts = [probs(m, split, "queue_onset") for m in models]
        B = parts[0][key].copy()
        B["p"] = np.mean([q.p.to_numpy() for q in parts], 0)
        for q in parts[1:]:
            assert (q[key].to_numpy() == B[key].to_numpy()).all()
        allp.append(B.assign(split=split))
        for wid, g in B.groupby("window_id"):
            L = statics(g.panel.iloc[0])["L"]
            ll = g.link.to_numpy().astype(int)
            A = np.zeros(L, bool); A[ll[eiou_topm(g.p.to_numpy())[0]]] = True
            preds[wid] = A
    pd.concat(allp).to_parquet(WORK / "probs_v6_onset.parquet")
    # v5 file with the onset rows replaced
    sub = pd.read_csv(V5, dtype=str)
    W = pd.concat([official_windows(p, s) for p in PANELS8 for s in ("validation", "private")]).set_index("window_id")
    onset = sub.window_id.map(W.condition).to_numpy() == "queue_onset"
    Tw = sub.window_id.map(W["T"]).to_numpy()
    from ..data import tindex
    k = tindex(sub.timestamp) - Tw - 1
    q = sub.queue_pred.astype(int).to_numpy().copy()
    for i in np.flatnonzero(onset):
        wid = sub.window_id.iat[i]
        lid = statics(W.loc[wid, "panel"])["lid"][sub.link_id.iat[i]]
        q[i] = int(k[i] == K - 1 and preds[wid][lid])
    sub["queue_pred"] = q
    sub.to_csv(out, index=False)
    print(json.dumps(check(Path(out))))


if __name__ == "__main__":
    main(sys.argv[1])
