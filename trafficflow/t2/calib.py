"""Onset decoding calibration: a logit bias b before top-m expected-IoU decoding.

The question is whether the official first-slot (T+30) queue blocks are larger or
smaller than the ones our labels teach. The v6 onset models are trained on the
hybrid truth, so on CV their optimum sits near b = 0 by construction. The public
LB decides which way the official truth leans: b > 0 means bigger sets.

    T2_WORK=/home/user/work/t2h T2_FEAT=/home/user/work/t2h/feat \
        python -m trafficflow.t2.calib cv                  # CV curve, hybrid and old truth
    ... python -m trafficflow.t2.calib build B OUT.csv     # lgb_v6 with the onset re-decoded at bias B
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .core import K, PANELS8, WORK, aggregate, iou, official_windows, statics
from .models import eiou_topm
from .robust import meta_index, onset_recurrence
from .submit import check

OOFS = ["oof_queue_onset_rob_on_v3_p1_op_all_new.parquet", "oof_queue_onset_rob_on_v3_p1_op_s1_all_new.parquet",
        "oof_queue_onset_rob_on_v3_p1_op_s2_all_new.parquet", "oof_queue_onset_rob_on_v2_p1_op_all_new.parquet"]
BIASES = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
V6 = Path("/home/user/work/t2/lgb_v6.csv")


def shift(p: np.ndarray, b: float) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + b)))


def load_oof(files=OOFS) -> pd.DataFrame:
    """Mean probability of the v6 onset models (same rows in every OOF file)."""
    key = ["gw", "k", "link"]
    base, ps = None, []
    for f in files:
        O = pd.read_parquet(WORK / f)
        if base is None:
            base = O[key].copy()
        else:
            assert (O[key].to_numpy() == base[key].to_numpy()).all(), f
        ps.append(O.p.to_numpy())
    return base.assign(p=np.mean(ps, 0))


def truths() -> dict:
    """Re-drawn-window truths [n, K, L] per panel: hybrid (y) and old (y_old),
    onset steps 1..K-1 empty (the organizer confirms onset truth is at T+30 only)."""
    out = {"hybrid": {}, "old": {}}
    for p in PANELS8:
        z = np.load(WORK / f"ds_{p}.npz", allow_pickle=True)
        on = z["w_condition"] == "queue_onset"
        for name, key in (("hybrid", "y"), ("old", "y_old")):
            y = z[key].copy()
            y[on, :K - 1] = False
            out[name][p] = y
    return out


def decoders(names=None) -> pd.DataFrame:
    """The site-committing decoders of onset_decode.py on the v6 hybrid OOF, under both truths."""
    from .onset_decode import DECODERS
    decs = {k: v for k, v in DECODERS.items() if names is None or k in names}
    O = load_oof()
    Mi = meta_index("queue_onset")
    O = O[O.gw.map(Mi.src).isin(["sim", "off"]).to_numpy()].sort_values("gw", kind="stable")
    Y = truths()
    rec = onset_recurrence()
    g = O.gw.to_numpy(); cut = np.flatnonzero(np.diff(g)) + 1
    kk = O.k.to_numpy().astype(int) - 1; ll = O.link.to_numpy().astype(int); pp = O.p.to_numpy()
    rows = []
    for idx in np.split(np.arange(len(O)), cut):
        gw = int(g[idx[0]]); m = Mi.loc[gw]
        for name, dec in decs.items():
            sel = dec(pp[idx])
            for tn in ("hybrid", "old"):
                yt = Y[tn][m.panel][int(m.w)]
                P = np.zeros_like(yt); P[kk[idx][sel], ll[idx][sel]] = True
                rows.append(dict(gw=gw, panel=m.panel, condition=m.condition, src=m.src, dec=name, truth=tn,
                                 iou=iou(P, yt), npred=len(sel)))
    df = pd.DataFrame(rows).assign(recur=lambda d: d.gw.map(rec))
    res = []
    for (name, tn), d in df.groupby(["dec", "truth"], sort=False):
        s = d[d.src == "sim"]
        res.append(dict(dec=name, truth=tn, sim=aggregate(s)["queue_onset"], off=aggregate(d[d.src == "off"])["queue_onset"],
                        rec005=s[s.recur < 0.05].iou.mean(), rec02=s[s.recur < 0.2].iou.mean(), npred=s.npred.mean()))
    return pd.DataFrame(res).round(4)


def curve(biases=BIASES) -> pd.DataFrame:
    O = load_oof()
    Mi = meta_index("queue_onset")
    O = O[O.gw.map(Mi.src).isin(["sim", "off"]).to_numpy()].sort_values("gw", kind="stable")
    Y = truths()
    rec = onset_recurrence()
    g = O.gw.to_numpy(); cut = np.flatnonzero(np.diff(g)) + 1
    kk = O.k.to_numpy().astype(int) - 1; ll = O.link.to_numpy().astype(int); pp = O.p.to_numpy()
    rows = []
    for idx in np.split(np.arange(len(O)), cut):
        gw = int(g[idx[0]]); m = Mi.loc[gw]
        for b in biases:
            sel = eiou_topm(shift(pp[idx], b))[0]
            for name in ("hybrid", "old"):
                yt = Y[name][m.panel][int(m.w)]
                P = np.zeros_like(yt); P[kk[idx][sel], ll[idx][sel]] = True
                rows.append(dict(gw=gw, panel=m.panel, condition=m.condition, src=m.src, b=b, truth=name,
                                 iou=iou(P, yt), npred=len(sel), ntrue=int(yt.sum())))
    df = pd.DataFrame(rows).assign(recur=lambda d: d.gw.map(rec))
    res = []
    for (b, name), d in df.groupby(["b", "truth"]):
        s = d[d.src == "sim"]
        res.append(dict(b=b, truth=name, sim=aggregate(s)["queue_onset"], off=aggregate(d[d.src == "off"])["queue_onset"],
                        rec005=s[s.recur < 0.05].iou.mean(), rec02=s[s.recur < 0.2].iou.mean(),
                        npred=s.npred.mean(), ntrue=s.ntrue.mean()))
    return pd.DataFrame(res).round(4), df


V5_ONGOING = {"oof_queue_ongoing_rob_og_v3_p2_w.parquet": 0.35, "oof_queue_ongoing_rob_og_v3_noloc_p2_w.parquet": 0.35,
              "oof_queue_ongoing_p2w.parquet": 0.15, "oof_queue_ongoing_rob_noloc_p2_w.parquet": 0.15}


def curve_ongoing(biases=(-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)) -> pd.DataFrame:
    """Same bias curve for the v5 ongoing blend (original windows, old truth; run with
    T2_WORK=/home/user/work/t2 T2_FEAT=/home/user/work/t2/feat_v3)."""
    from .robust import recurrence, truth_lookup_y, window_scores
    from .v5_eval import blend_oof
    O = blend_oof(V5_ONGOING)
    Mi = meta_index("queue_ongoing")
    O = O[O.gw.map(Mi.src).isin(["sim", "off"]).to_numpy()]
    Y = truth_lookup_y()
    rec = recurrence("train")
    res = []
    for b in biases:
        df = window_scores(O.assign(p=shift(O.p.to_numpy(), b)), Mi, Y).assign(recur=lambda d: d.gw.map(rec))
        s = df[df.src == "sim"]
        res.append(dict(b=b, sim=aggregate(s)["queue_ongoing"], off=aggregate(df[df.src == "off"])["queue_ongoing"],
                        rec005=s[s.recur < 0.05].iou.mean(), rec02=s[s.recur < 0.2].iou.mean()))
        print(res[-1], flush=True)
    return pd.DataFrame(res).round(4)


def build(b: float, out: str, probs: Path = WORK / "probs_v6_onset.parquet", base: Path = V6, dec: str | None = None):
    """base file with the onset rows re-decoded from the saved val/private probabilities
    at bias b (top-m), or with an onset_decode.py decoder ``dec`` (bias still applied)."""
    from .onset_decode import DECODERS
    decode = (lambda p: eiou_topm(p)[0]) if dec is None else DECODERS[dec]
    B = pd.read_parquet(probs)
    preds = {}
    for wid, g in B.groupby("window_id"):
        A = np.zeros(statics(g.panel.iloc[0])["L"], bool)
        A[g.link.to_numpy().astype(int)[decode(shift(g.p.to_numpy(), b))]] = True
        preds[wid] = A
    sub = pd.read_csv(base, dtype=str)
    W = pd.concat([official_windows(p, s) for p in PANELS8 for s in ("validation", "private")]).set_index("window_id")
    onset = sub.window_id.map(W.condition).to_numpy() == "queue_onset"
    from ..data import tindex
    k = tindex(sub.timestamp) - sub.window_id.map(W["T"]).to_numpy() - 1
    q = sub.queue_pred.astype(int).to_numpy().copy()
    for i in np.flatnonzero(onset):
        wid = sub.window_id.iat[i]
        lid = statics(W.loc[wid, "panel"])["lid"][sub.link_id.iat[i]]
        q[i] = int(k[i] == K - 1 and preds[wid][lid])
    old = sub.queue_pred.astype(int).to_numpy()
    sub["queue_pred"] = q
    sub.to_csv(out, index=False)
    print(json.dumps({"bias": b, "decoder": dec or "topm", "onset_cells": int(q[onset].sum()), "onset_cells_base": int(old[onset].sum()),
                      "changed_cells": int((q != old).sum()), "ongoing_changed": int((q != old)[~onset].sum()),
                      "check": check(Path(out))}))


if __name__ == "__main__":
    if sys.argv[1] == "cv":
        res, df = curve()
        df.to_parquet(WORK / "calib_onset_windows.parquet")
        print(res.to_string(index=False))
    elif sys.argv[1] == "decoders":
        print(decoders().to_string(index=False))
    elif sys.argv[1] == "cv_ongoing":
        print(curve_ongoing().to_string(index=False))
    elif sys.argv[1] == "build":
        build(float(sys.argv[2]), sys.argv[3], dec=sys.argv[4] if len(sys.argv) > 4 else None)
