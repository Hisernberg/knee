"""Set-level decoders for onset windows (all cells at T+30), evaluated on OOF
probabilities.

The truth of an onset window is usually one queued block (sometimes two) at a
bottleneck. A pointwise model spreads probability over the candidate sites;
top-m then mixes sites. These decoders commit to sites instead:

* ``topm``        expected-IoU top-m over all links (baseline);
* ``range``       best contiguous link range under the surrogate;
* ``anchor_up``   argmax link plus the best contiguous extension upstream;
* ``site``        clusters of links with p >= ``lo`` (gaps <= 1 link); keep the
                  cluster with the largest mass and run top-m inside it;
* ``site2``       like ``site`` but a second cluster is added when its mass
                  is at least ``ratio`` times the first.

    python -m trafficflow.t2.onset_decode OOF_FILE
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .core import K, WORK, aggregate, iou
from .models import eiou_range, eiou_topm


def _clusters(p: np.ndarray, lo: float, gap: int = 1):
    idx = np.flatnonzero(p >= lo)
    if not len(idx):
        return [np.array([int(np.argmax(p))])]
    cut = np.flatnonzero(np.diff(idx) > gap + 1)
    return np.split(idx, cut + 1)


def dec_topm(p):
    return eiou_topm(p)[0]


def dec_range(p):
    a, b, _ = eiou_range(p)
    return np.arange(a, b + 1)


def dec_anchor_up(p, max_len=40):
    b = int(np.argmax(p))
    C = np.concatenate([[0.0], np.cumsum(p)]); tot = C[-1]
    best, ba = -1, b
    for a in range(max(0, b - max_len), b + 1):
        s = C[b + 1] - C[a]; n = b - a + 1
        r = s / (n + tot - s)
        if r > best:
            best, ba = r, a
    return np.arange(ba, b + 1)


def dec_site(p, lo=0.05, ratio=None):
    cl = _clusters(p, lo)
    mass = np.array([p[c].sum() for c in cl])
    order = np.argsort(-mass)
    keep = [cl[order[0]]]
    if ratio is not None and len(cl) > 1 and mass[order[1]] >= ratio * mass[order[0]]:
        keep.append(cl[order[1]])
    idx = np.concatenate(keep)
    q = np.zeros_like(p); q[idx] = p[idx]
    sel = eiou_topm(q)[0]
    sel = sel[q[sel] > 0]
    return sel if len(sel) else np.array([int(np.argmax(p))])


DECODERS = {
    "topm": dec_topm,
    "range": dec_range,
    "anchor_up": dec_anchor_up,
    "site_lo.05": lambda p: dec_site(p, 0.05),
    "site_lo.1": lambda p: dec_site(p, 0.10),
    "site_lo.2": lambda p: dec_site(p, 0.20),
    "site2_lo.1_r.5": lambda p: dec_site(p, 0.10, 0.5),
    "site2_lo.1_r.3": lambda p: dec_site(p, 0.10, 0.3),
    "site2_lo.05_r.5": lambda p: dec_site(p, 0.05, 0.5),
}


def evaluate(oof_file: str, decoders=DECODERS, rec=None) -> pd.DataFrame:
    from .robust import meta_index, truth_lookup_y
    O = pd.read_parquet(WORK / oof_file).sort_values("gw", kind="stable")
    Mi = meta_index("queue_onset"); Y = truth_lookup_y()
    g = O.gw.to_numpy(); b = np.flatnonzero(np.diff(g)) + 1
    ll_all = O.link.to_numpy().astype(int); pp_all = O.p.to_numpy()
    rows = []
    for idx in np.split(np.arange(len(O)), b):
        gw = int(g[idx[0]]); m = Mi.loc[gw]
        yt = Y[m.panel][int(m.w)]
        L = yt.shape[1]
        p = np.zeros(L); p[ll_all[idx]] = pp_all[idx]
        r = dict(gw=gw, panel=m.panel, condition="queue_onset", src=m.src)
        for name, fn in decoders.items():
            P = np.zeros_like(yt); P[K - 1, fn(p)] = True
            r[name] = iou(P, yt)
        rows.append(r)
    df = pd.DataFrame(rows)
    if rec is not None:
        df["recur"] = df.gw.map(rec)
    return df


def summarize(df: pd.DataFrame, names) -> pd.DataFrame:
    out = {}
    for n in names:
        s = df[df.src == "sim"]
        d = {"sim": aggregate(s.rename(columns={n: "iou"}))["queue_onset"],
             "off": aggregate(df[df.src == "off"].rename(columns={n: "iou"}))["queue_onset"]}
        if "recur" in df:
            for t in (0.05, 0.2):
                d[f"recur<{t}"] = float(s[s.recur < t][n].mean())
        out[n] = {k: round(v, 4) for k, v in d.items()}
    return pd.DataFrame(out).T


if __name__ == "__main__":
    from .robust import onset_recurrence
    df = evaluate(sys.argv[1], rec=onset_recurrence())
    print(summarize(df, list(DECODERS)).to_string())
