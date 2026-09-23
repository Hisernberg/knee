"""Offline decoder comparison on saved out-of-fold probabilities.

    PYTHONPATH=/home/user/knee python -m trafficflow.t2.decode_tune queue_onset v2
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .core import K, PANELS8, WORK, aggregate, iou
from .models import eiou_range, eiou_topm


def load_oof(cond, tag):
    O = pd.read_parquet(WORK / f"oof_{cond}_{tag}.parquet")
    M = pd.read_parquet(WORK / f"oofmeta_{cond}_{tag}.parquet").drop_duplicates("gw").set_index("gw")
    Y = {p: np.load(WORK / f"ds_{p}.npz", allow_pickle=True)["y"] for p in PANELS8}
    return O, M, Y


def windows(O, M, Y):
    O = O.sort_values("gw", kind="stable")
    g = O.gw.to_numpy()
    bounds = np.flatnonzero(np.diff(g)) + 1
    for idx in np.split(np.arange(len(O)), bounds):
        gw = int(g[idx[0]])
        m = M.loc[gw]
        yt = Y[m.panel][int(m.w)]
        yield m, yt, O.k.to_numpy()[idx].astype(int) - 1, O.link.to_numpy()[idx].astype(int), O.p.to_numpy()[idx]


def closing(pred: np.ndarray, gap: int = 1) -> np.ndarray:
    """Fill spatial gaps of <= gap links between predicted cells at each step."""
    out = pred.copy()
    L = pred.shape[1]
    for k in range(pred.shape[0]):
        idx = np.flatnonzero(pred[k])
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= gap + 1:
                out[k, a:b] = True
    return out


def evaluate(cond, tag, decoders: dict) -> pd.DataFrame:
    O, M, Y = load_oof(cond, tag)
    rows = []
    for m, yt, kk, ll, pp in windows(O, M, Y):
        r = dict(panel=m.panel, condition=cond, src=m.src)
        for name, fn in decoders.items():
            pred = np.zeros_like(yt)
            sel = fn(pp, kk, ll, yt.shape[1])
            if isinstance(sel, np.ndarray) and sel.ndim == 2:
                pred = sel
            else:
                pred[kk[sel], ll[sel]] = True
            r[name] = iou(pred, yt)
        rows.append(r)
    return pd.DataFrame(rows)


def topm(power=1.0, scale=1.0, cap=None):
    def f(pp, kk, ll, L):
        q = np.clip(scale * pp ** power, 0, 1)
        s = np.zeros(len(pp), bool)
        s[eiou_topm(q, cap)[0]] = True
        return s
    return f


def topm_close(gap=1):
    def f(pp, kk, ll, L):
        s = np.zeros(len(pp), bool); s[eiou_topm(pp)[0]] = True
        P = np.zeros((K, L), bool); P[kk[s], ll[s]] = True
        return closing(P, gap)
    return f


def thr(t):
    return lambda pp, kk, ll, L: pp >= t


def main():
    cond = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else "v2"
    decs = {"topm": topm(), "topm_p0.8": topm(0.8), "topm_p1.25": topm(1.25), "topm_s1.2": topm(1, 1.2),
            "topm_s0.8": topm(1, 0.8), "topm_close1": topm_close(1)}
    for t in (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5):
        decs[f"thr{t}"] = thr(t)
    df = evaluate(cond, tag, decs)
    for src in ("sim", "off"):
        d = df[df.src == src]
        print(src, {n: round(aggregate(d.rename(columns={n: "iou"}))[cond], 4) for n in decs})
    return df


if __name__ == "__main__":
    main()
