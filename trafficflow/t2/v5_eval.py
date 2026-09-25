"""Per-slice CV comparison of OOF probability blends (v4 vs v5 candidates).

A blend spec is {"queue_onset": {oof_file: weight, ...}, "queue_ongoing": {...}}.
Reported per condition: sim / off (official aggregation) and the plain mean
IoU on the non-recurrent subsets (recurrence < 0.05 / < 0.2; ongoing uses
robust.recurrence, onset the truth-based onset_recurrence); overall = mean of
the two condition aggregates (every panel has both conditions).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import WORK, aggregate
from .robust import meta_index, onset_recurrence, recurrence, truth_lookup_y, window_scores

KEY = ["gw", "k", "link"]
_cache: dict = {}


def _oof(f):
    if f not in _cache:
        _cache[f] = pd.read_parquet(WORK / f)[KEY + ["p"]]
    return _cache[f]


def blend_oof(spec: dict) -> pd.DataFrame:
    items = list(spec.items())
    B = _oof(items[0][0])[KEY].copy()
    p = np.zeros(len(B))
    wsum = 0.0
    for f, w in items:
        O = B.merge(_oof(f), on=KEY, how="left")
        assert len(O) == len(B) and O.p.notna().all(), f
        p += w * O.p.to_numpy(); wsum += w
    return B.assign(p=p / wsum)


def evaluate(spec: dict) -> dict:
    out = {}
    Y = truth_lookup_y()
    for cond, s in spec.items():
        rec = recurrence("train") if cond == "queue_ongoing" else onset_recurrence()
        df = window_scores(blend_oof(s), meta_index(cond), Y).assign(recur=lambda d: d.gw.map(rec))
        sim = df[df.src == "sim"]
        c = "on" if cond == "queue_onset" else "og"
        out[f"{c}_sim"] = aggregate(sim)[cond]
        out[f"{c}_off"] = aggregate(df[df.src == "off"])[cond]
        for t in (0.05, 0.2):
            out[f"{c}_rec<{t}"] = float(sim[sim.recur < t].iou.mean())
    if "on_sim" in out and "og_sim" in out:
        out["overall_sim"] = (out["on_sim"] + out["og_sim"]) / 2
        out["overall_off"] = (out["on_off"] + out["og_off"]) / 2
    return {k: round(v, 4) for k, v in out.items()}


V4 = {"queue_onset": {"oof_queue_onset_rob_op_p1.parquet": 1.0},
      "queue_ongoing": {"oof_queue_ongoing_p2w.parquet": 0.5, "oof_queue_ongoing_rob_noloc_p2_w.parquet": 0.5}}
