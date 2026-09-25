"""Stage-2 onset model: re-score each link from the stage-1 probability profile
of its window (block shape), in traffic direction.

Stage 1 = OOF probabilities of the onset models for every onset window (train
candidates and evaluation windows; ``robust cv ... `` with T2_OOF_ALL=1).
Stage 2 features per (window, link), with offsets in traffic direction
(+ = downstream): p, p at offsets -4..+4, window max / sum / rank, offset and
distance (km) to the window argmax, probability mass 0.5/1/2 km downstream and
upstream, head / tail proxies, link length, relative position, panel. Trained
with the same 4 week-folds (OOF stacking), scored with top-m decoding.

    python -m trafficflow.t2.stack OOF_ALL_FILE [OOF_ALL_FILE ...]
"""
from __future__ import annotations

import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from .baselines import fold_of
from .core import K, PANELS8, WORK, aggregate, direction, iou, statics
from .models import eiou_topm
from .robust import meta_index, onset_recurrence, truth_lookup_y

P2 = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8,
          bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=int(os.environ.get("T2_THREADS", "2")),
          verbose=-1, seed=0)


def _shift(a, o):
    L = a.shape[1]
    idx = np.clip(np.arange(L) + o, 0, L - 1)
    out = a[:, idx].copy()
    if o > 0:
        out[:, L - o:] = 0.0
    elif o < 0:
        out[:, :(-o)] = 0.0
    return out


def panel_features(p: np.ndarray, panel: str) -> dict:
    """p [n, L] stage-1 probabilities in original link order -> features [n, L]."""
    rev = direction(panel) < 0
    st = statics(panel + "@rev" if rev else panel)
    q = p[:, ::-1] if rev else p
    n, L = q.shape
    mp = st["mp"]; ln = st["length"]
    F = {"s_p": q}
    for o in (-4, -3, -2, -1, 1, 2, 3, 4):
        F[f"s_p_o{o}"] = _shift(q, o)
    mx = q.max(1, keepdims=True); sm = q.sum(1, keepdims=True)
    F["s_pmax"] = np.repeat(mx, L, 1); F["s_psum"] = np.repeat(sm, L, 1)
    F["s_prel"] = q / np.maximum(mx, 1e-6)
    F["s_rank"] = np.argsort(np.argsort(-q, 1), 1).astype(np.float32)
    am = q.argmax(1)
    F["s_off_am"] = (np.arange(L)[None, :] - am[:, None]).astype(np.float32)
    F["s_km_am"] = (mp[None, :] - mp[am][:, None]).astype(np.float32)
    C = np.concatenate([np.zeros((n, 1)), np.cumsum(q, 1)], 1)
    for km in (0.5, 1.0, 2.0):
        hi = np.searchsorted(mp, mp + km, side="right")          # links with mp <= mp_l + km
        lo = np.searchsorted(mp, mp - km - 1e-9, side="left")     # links with mp >= mp_l - km
        F[f"s_dn{km}"] = C[:, hi] - C[:, np.arange(L) + 1]
        F[f"s_up{km}"] = C[:, np.arange(L)] - C[:, lo]
    F["s_head"] = q * (1 - _shift(q, 1)); F["s_tail"] = q * (1 - _shift(q, -1))
    if CLUSTER:
        for o in (-8, -6, 6, 8):
            F[f"s_p_o{o}"] = _shift(q, o)
        names = ["s_cl_mass", "s_cl_max", "s_cl_len", "s_cl_pos_dn", "s_cl_pos_up", "s_cl_km_dn", "s_cl_km_up",
                 "s_cl_top", "s_n_cl"]
        G = {k: np.zeros((n, L), np.float32) for k in names}
        for i in range(n):
            idx = np.flatnonzero(q[i] >= 0.2)
            if not len(idx):
                continue
            cut = np.flatnonzero(np.diff(idx) > 2)
            cls = np.split(idx, cut + 1)
            masses = [q[i, c[0]:c[-1] + 1].sum() for c in cls]
            top = int(np.argmax(masses))
            G["s_n_cl"][i] = len(cls)
            for j, c in enumerate(cls):
                a, b = c[0], c[-1]
                sl = slice(a, b + 1)
                G["s_cl_mass"][i, sl] = masses[j]; G["s_cl_max"][i, sl] = q[i, sl].max()
                G["s_cl_len"][i, sl] = b - a + 1
                G["s_cl_pos_dn"][i, sl] = b - np.arange(a, b + 1); G["s_cl_pos_up"][i, sl] = np.arange(a, b + 1) - a
                G["s_cl_km_dn"][i, sl] = mp[b] - mp[a:b + 1]; G["s_cl_km_up"][i, sl] = mp[a:b + 1] - mp[a]
                G["s_cl_top"][i, sl] = float(j == top)
        F.update(G)
    F["s_len"] = np.repeat(ln[None], n, 0); F["s_relpos"] = np.repeat((np.arange(L) / L)[None], n, 0)
    F["s_pcode"] = np.full((n, L), PANELS8.index(panel), np.float32)
    if rev:
        F = {k: v[:, ::-1] for k, v in F.items()}
    return {k: np.asarray(v, np.float32) for k, v in F.items()}


def build(files: list[str]) -> tuple[pd.DataFrame, list[str]]:
    O = None
    for f in files:
        x = pd.read_parquet(WORK / f)[["gw", "k", "link", "y", "p"]]
        O = x if O is None else O.merge(x[["gw", "k", "link", "p"]], on=["gw", "k", "link"], suffixes=("", "_b"))
        if "p_b" in O:
            O["p"] = O["p"] + O.pop("p_b")
    O["p"] /= len(files)
    Mi = meta_index("queue_onset")
    parts = []
    for p in PANELS8:
        pc = PANELS8.index(p)
        sub = O[O.gw // 100000 == pc]
        gws = np.unique(sub.gw.to_numpy())
        L = statics(p)["L"]
        pos = {g: i for i, g in enumerate(gws)}
        wi = sub.gw.map(pos).to_numpy(); li = sub.link.to_numpy().astype(int)
        Pm = np.zeros((len(gws), L), np.float32); Ym = np.zeros((len(gws), L), np.int8)
        Pm[wi, li] = sub.p.to_numpy(); Ym[wi, li] = sub.y.to_numpy()
        F = panel_features(Pm, p)
        d = pd.DataFrame({k: v.reshape(-1) for k, v in F.items()})
        d["gw"] = np.repeat(gws, L); d["link"] = np.tile(np.arange(L), len(gws)); d["y"] = Ym.reshape(-1)
        parts.append(d)
    X = pd.concat(parts, ignore_index=True)
    m = Mi.reindex(X.gw.to_numpy())
    X["fold"] = fold_of(m["T"].to_numpy()); X["src"] = m.src.to_numpy()
    if EXTRA:
        X = add_extra(X, Mi)
    cols = [c for c in X.columns if c.startswith("s_") or c.startswith("x_")]
    return X, cols


EXTRA = os.environ.get("T2_STACK_EXTRA", "0") == "1"
CLUSTER = os.environ.get("T2_STACK_CLUSTER", "0") == "1"
XCOLS = ["r_now", "f_last", "ph_fcap_ext", "ph_k_ext", "ph_r_ext", "pq_k", "rq7_k", "rT_ok"]


def add_extra(X: pd.DataFrame, Mi: pd.DataFrame) -> pd.DataFrame:
    """Non-directional stage-1 inputs at the link, plus the onset location prior."""
    from .core import FEAT
    from .oprior import OnsetPrior
    parts = []
    for p in PANELS8:
        pc = PANELS8.index(p)
        sub = X[X.gw // 100000 == pc]
        ws = np.unique(sub.gw.to_numpy() % 100000).tolist()
        F = pd.read_parquet(FEAT / f"feat_{p}.parquet", columns=["w", "link"] + XCOLS, filters=[("w", "in", ws)])
        F = F.drop_duplicates(["w", "link"])
        F["gw"] = pc * 100000 + F.w.astype(np.int64)
        F = F.drop(columns="w").rename(columns={c: "x_" + c for c in XCOLS})
        sub = sub.merge(F, on=["gw", "link"], how="left")
        gws = np.unique(sub.gw.to_numpy())
        f = OnsetPrior(p).features(Mi.loc[gws, "T"].to_numpy(), Mi.loc[gws, "fold"].to_numpy()
                                   if "fold" in Mi else fold_of(Mi.loc[gws, "T"].to_numpy()))
        pos = {g: i for i, g in enumerate(gws)}
        sub["x_op"] = f["op_tod90"][sub.gw.map(pos).to_numpy(), sub.link.to_numpy().astype(int)]
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def cv(X: pd.DataFrame, cols, params=P2, rounds=300) -> np.ndarray:
    p = np.full(len(X), np.nan, np.float32)
    for f in range(4):
        tr = (X.fold != f).to_numpy(); te = (X.fold == f).to_numpy()
        m = lgb.train(params, lgb.Dataset(X.loc[tr, cols].to_numpy(np.float32), X.y[tr].to_numpy(np.float32)), rounds)
        p[te] = m.predict(X.loc[te, cols].to_numpy(np.float32), num_threads=P2["num_threads"])
    return p


def score(X: pd.DataFrame, pcols: dict) -> pd.DataFrame:
    Mi = meta_index("queue_onset"); Y = truth_lookup_y(); rec = onset_recurrence()
    E = X[X.src.isin(["sim", "off"])]
    rows = []
    for gw, g in E.groupby("gw"):
        m = Mi.loc[gw]; yt = Y[m.panel][int(m.w)]
        r = dict(gw=gw, panel=m.panel, condition="queue_onset", src=m.src, recur=rec.get(gw, np.nan))
        ll = g.link.to_numpy()
        for name, c in pcols.items():
            pp = g[c].to_numpy()
            P = np.zeros_like(yt); P[K - 1, ll[eiou_topm(pp)[0]]] = True
            r[name] = iou(P, yt)
        rows.append(r)
    return pd.DataFrame(rows)


def summarize(df, names):
    out = {}
    s = df[df.src == "sim"]
    for n in names:
        out[n] = {"sim": aggregate(s.rename(columns={n: "iou"}))["queue_onset"],
                  "off": aggregate(df[df.src == "off"].rename(columns={n: "iou"}))["queue_onset"],
                  "rec<0.05": s[s.recur < 0.05][n].mean(), "rec<0.2": s[s.recur < 0.2][n].mean()}
    return pd.DataFrame(out).T.round(4)


if __name__ == "__main__":
    X, cols = build(sys.argv[1:])
    X["p1"] = X["s_p"]
    rounds = int(os.environ.get("T2_STACK_ROUNDS", "300"))
    leaves = int(os.environ.get("T2_STACK_LEAVES", "31"))
    seed = int(os.environ.get("T2_STACK_SEED", "0"))
    X["p2"] = cv(X, cols, {**P2, "num_leaves": leaves, "seed": seed}, rounds)
    for w in (0.5, 0.7):
        X[f"b{w}"] = w * X.p2 + (1 - w) * X.p1
    X[["gw", "link", "y", "fold", "src", "p1", "p2"]].to_parquet(
        WORK / f"stack_onset_oof{os.environ.get('T2_STACK_TAG', '')}.parquet")
    df = score(X, {"stage1": "p1", "stage2": "p2", "blend0.5": "b0.5", "blend0.7": "b0.7"})
    print(summarize(df, ["stage1", "stage2", "blend0.5", "blend0.7"]).to_string())
