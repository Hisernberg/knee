"""lgb_v7: lgb_v6 onset rows unchanged + ongoing retrained on the hybrid truth.

Ongoing: the v5 recipe (0.35 LWR-all + 0.35 LWR-noloc + 0.15 v2-all + 0.15
v2-noloc; p2 config, equal window weights, half of the extra candidate
windows) trained on the windows re-drawn by the selector on the hybrid truth
with hybrid labels (WORK=/home/user/work/t2h, FEAT=.../t2h/feat_og). Top-m
expected-IoU decoding.

    T2_WORK=/home/user/work/t2h T2_FEAT=/home/user/work/t2h/feat_og \
        python -m trafficflow.t2.ongoing_v7 /home/user/work/t2/lgb_v7.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..data import tindex
from .core import K, PANELS8, WORK, official_windows, statics
from .robust_pipeline import blend_probs, decode_topm, train_variant
from .submit import check

RECIPE = [("og_v3", 0.35), ("og_v3_noloc", 0.35), ("og_v2", 0.15), ("og_v2_noloc", 0.15)]
V6 = Path("/home/user/work/t2/lgb_v6.csv")


def main(out: str):
    comps = []
    for variant, w in RECIPE:
        f = WORK / f"model_v7_{variant}_queue_ongoing.txt"
        if f.exists():
            m = lgb.Booster(model_file=str(f))
        else:
            m = train_variant(variant, "p2", True, cond="queue_ongoing")
            m.save_model(str(f))
        comps.append((m, w))
        print("model ready", variant, flush=True)
    preds, allp = {}, []
    for split in ("validation", "private"):
        B = blend_probs(comps, split, "queue_ongoing")
        preds.update(decode_topm(B[["window_id", "panel", "k", "link", "p"]]))
        allp.append(B.assign(split=split))
    pd.concat(allp).to_parquet(WORK / "probs_v7_ongoing.parquet")
    sub = pd.read_csv(V6, dtype=str)
    Wd = pd.concat([official_windows(p, s) for p in PANELS8 for s in ("validation", "private")]).set_index("window_id")
    og = sub.window_id.map(Wd.condition).to_numpy() == "queue_ongoing"
    k = tindex(sub.timestamp) - sub.window_id.map(Wd["T"]).to_numpy() - 1
    q = sub.queue_pred.astype(int).to_numpy().copy()
    for i in np.flatnonzero(og):
        wid = sub.window_id.iat[i]
        lid = statics(Wd.loc[wid, "panel"])["lid"][sub.link_id.iat[i]]
        q[i] = int(preds[wid][k[i], lid])
    sub["queue_pred"] = q
    sub.to_csv(out, index=False)
    print(json.dumps(check(Path(out))))


if __name__ == "__main__":
    main(sys.argv[1])
