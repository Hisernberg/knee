"""Ongoing label fix: score old-label vs hybrid-label model blends under the
old and the hybrid (truthfix) truth, on the original and on the re-drawn
(hybrid-truth selector) windows.

Blend = v5 ongoing recipe weights: 0.35 LWR-all + 0.35 LWR-noloc + 0.15 v2-all
+ 0.15 v2-noloc, here in the fast config (see TASK2_ANALYSIS.md section 14).

    PYTHONPATH=/home/user/knee python -m trafficflow.t2.og_labelfix_eval
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .core import K, PANELS8, aggregate, iou
from .models import eiou_topm

T2 = "/home/user/work/t2"; T2H = "/home/user/work/t2h"
W = {"og_v3": 0.35, "og_v3_noloc": 0.35, "og_v2": 0.15, "og_v2_noloc": 0.15}
SETS = {
    "original": dict(work=T2, feat=f"{T2}/feat_v3",
                     old={"og_v3": "rob_og_v3_fast_w", "og_v3_noloc": "rob_og_v3_noloc_fast_w",
                          "og_v2": "rob_base_fast_w", "og_v2_noloc": "rob_noloc_fast_w"},
                     new={v: f"rob_{v}_fast_w_orig_y2" for v in W}),
    "redrawn": dict(work=T2H, feat=f"{T2H}/feat_og",
                    old={v: f"rob_{v}_fast_w_t2h_old" for v in W},
                    new={v: f"rob_{v}_fast_w_t2h_new" for v in W}),
}


def _blend(work, names):
    B = None
    for v, f in names.items():
        x = pd.read_parquet(f"{work}/oof_queue_ongoing_{f}.parquet")[["gw", "k", "link", "p"]]
        x["p"] *= W[v]
        B = x if B is None else B.merge(x, on=["gw", "k", "link"], suffixes=("", "_b"))
        if "p_b" in B:
            B["p"] = B.p + B.pop("p_b")
    return B


def _meta(feat):
    Ms = []
    for p in PANELS8:
        M = pd.read_parquet(f"{feat}/meta_{p}.parquet")
        M["gw"] = PANELS8.index(p) * 100000 + M.w.astype(np.int64)
        Ms.append(M[M.condition == "queue_ongoing"])
    return pd.concat(Ms).drop_duplicates("gw").set_index("gw")


def _recurrence(feat):
    out = []
    for p in PANELS8:
        M = pd.read_parquet(f"{feat}/meta_{p}.parquet")
        ids = M.w[M.condition == "queue_ongoing"].tolist()
        X = pd.read_parquet(f"{feat}/feat_{p}.parquet", columns=["w", "r_last", "pq_T"],
                            filters=[("w", "in", ids), ("k", "==", 1.0)])
        r = X[X.r_last <= 1.0].groupby("w").pq_T.mean().reindex(ids).fillna(0.0)
        r.index = PANELS8.index(p) * 100000 + r.index.astype(np.int64)
        out.append(r)
    return pd.concat(out)


def truths(set_name):
    """{'old': {panel: y}, 'hybrid': {panel: y}} for the set's windows."""
    out = {"old": {}, "hybrid": {}}
    for p in PANELS8:
        if set_name == "original":
            z = np.load(f"{T2}/ds_{p}.npz", allow_pickle=True)
            out["old"][p] = z["y"]; out["hybrid"][p] = np.load(f"{T2}/ds_{p}_y2.npz")["y"]
        else:
            z = np.load(f"{T2H}/ds_{p}.npz", allow_pickle=True)
            out["old"][p] = z["y_old"]; out["hybrid"][p] = z["y"]
    return out


def score(B, Mi, Ys, rec):
    B = B.sort_values("gw", kind="stable")
    g = B.gw.to_numpy(); cut = np.flatnonzero(np.diff(g)) + 1
    kk = B.k.to_numpy().astype(int) - 1; ll = B.link.to_numpy().astype(int); pp = B.p.to_numpy()
    rows = []
    for idx in np.split(np.arange(len(B)), cut):
        gw = int(g[idx[0]]); m = Mi.loc[gw]
        if m.src not in ("sim", "off"):
            continue
        sel = eiou_topm(pp[idx])[0]
        r = dict(gw=gw, panel=m.panel, condition="queue_ongoing", src=m.src, recur=rec.get(gw, np.nan))
        for tname, Y in Ys.items():
            yt = Y[m.panel][int(m.w)]
            P = np.zeros_like(yt); P[kk[idx][sel], ll[idx][sel]] = True
            r[tname] = iou(P, yt)
        rows.append(r)
    return pd.DataFrame(rows)


def summary(df, col):
    s = df[df.src == "sim"]
    return {"sim": aggregate(s.rename(columns={col: "iou"}))["queue_ongoing"],
            "off": aggregate(df[df.src == "off"].rename(columns={col: "iou"}))["queue_ongoing"],
            "rec<0.05": s[s.recur < 0.05][col].mean(), "rec<0.2": s[s.recur < 0.2][col].mean(), "n_sim": len(s)}


def bootstrap(dfo, dfn, col, n=400, seed=0):
    a = dfo[dfo.src == "sim"].set_index("gw")[[col, "panel", "condition"]]
    b = dfn[dfn.src == "sim"].set_index("gw")[[col]]
    d = a.join(b, rsuffix="_n").dropna()
    rng = np.random.default_rng(seed); diffs = []
    for _ in range(n):
        x = d.iloc[rng.integers(0, len(d), len(d))]
        diffs.append(aggregate(x.assign(iou=x[f"{col}_n"]))["queue_ongoing"] - aggregate(x.assign(iou=x[col]))["queue_ongoing"])
    full = aggregate(d.assign(iou=d[f"{col}_n"]))["queue_ongoing"] - aggregate(d.assign(iou=d[col]))["queue_ongoing"]
    return full, float(np.std(diffs)), int((d[f"{col}_n"] > d[col] + 1e-9).sum()), int((d[f"{col}_n"] < d[col] - 1e-9).sum())


def main():
    rows = []
    for sname, cfg in SETS.items():
        try:
            Bo = _blend(cfg["work"], cfg["old"]); Bn = _blend(cfg["work"], cfg["new"])
        except FileNotFoundError as e:
            print(sname, "missing", e); continue
        Mi = _meta(cfg["feat"]); rec = _recurrence(cfg["feat"]); Ys = truths(sname)
        dfo = score(Bo, Mi, Ys, rec); dfn = score(Bn, Mi, Ys, rec)
        dfo.to_parquet(f"{T2}/og_labelfix_{sname}_old.parquet"); dfn.to_parquet(f"{T2}/og_labelfix_{sname}_new.parquet")
        for t in ("old", "hybrid"):
            so, sn = summary(dfo, t), summary(dfn, t)
            full, se, better, worse = bootstrap(dfo, dfn, t)
            rows.append(dict(windows=sname, truth=t, old_labels_sim=so["sim"], new_labels_sim=sn["sim"],
                             delta=full, se=se, better=better, worse=worse,
                             off=f"{so['off']:.4f} -> {sn['off']:.4f}",
                             rec005=f"{so['rec<0.05']:.3f} -> {sn['rec<0.05']:.3f}",
                             rec02=f"{so['rec<0.2']:.3f} -> {sn['rec<0.2']:.3f}", n_sim=so["n_sim"]))
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(out.round(4).to_string())
    return out


if __name__ == "__main__":
    main()
