"""Stage-2 stacking for the ongoing model on the v5 stage-1 probability field
(TASK2_ANALYSIS.md section 16).

Stage 1 = the v5 ongoing blend (0.35 LWR-all + 0.35 LWR-noloc + 0.15 v2-all +
0.15 v2-noloc; p2 config, window weights, old labels). Its 4-fold OOF
probabilities exist for the 3,041 ongoing evaluation windows (2,999 sim + 40
off; ``robust cv`` without T2_OOF_ALL) of the original window set.

Stage 2 = LightGBM per candidate cell (window, step k, link), trained on the
old labels with the same 4 week folds (OOF stacking: the stage-2 model for a
fold never sees that fold's windows; the stage-1 OOF probabilities it reads
never came from a model that saw their own window). Features (``features``),
all in traffic direction (the link axis of the W/S panels is reversed, see
``core.direction``; + = downstream):

* the field: p; p at link offsets -4..+4 (same step); p at steps k-1 / k+1
  (same link and +-1 link); step sum / max, window sum, step-sum growth
  (vs step 1 and vs step k-1), rank and p / step max; probability mass
  0.5 / 1 / 2 km downstream and upstream;
* predicted block geometry at step k (runs of p >= 0.5): signed distance
  (links and km) to the tail of the block containing the link or the next one
  downstream, and to the head of the block containing it or the next one
  upstream; block length (links, km) and mass; number of blocks; tail / head
  movement vs step k-1 and vs the observed block at the origin;
* the observed queue (data <= T only): queue indicator at the origin
  (``r_now <= 1``; slot T where visible, else the last history value), the
  observed block's tail / head distances, the observed tail and head movement
  over the last 20 and 35 min (filled history ratio at T-20 / T-35 min =
  ``r_last - d_r15`` / ``r_last - d_r30``) and the tail extrapolated
  linearly to T+5k;
* window context: recurrence (``robust.recurrence``), queued links at the end
  of the history and at the origin, their 15 / 60-min trend, weekend flag,
  time of day; per link r_now, r_last and the 15-min ratio change;
* static: step k, link length, relative position, panel code.

Variants: ``--comp`` adds the loc / noloc component probabilities (stage-1
disagreement), ``--loc`` adds the time-of-day prior pq_k, ``--weighted``
gives every window equal total weight, ``--drop`` removes columns or prefixes
(``x_*``). The adopted recipe (``dyn``, section 16) is
``--seeds 0,1,2 --drop 'x_rec,x_tod,x_wkend,s_pcode,s_relpos'`` at w = 0.8.

Evaluation (``OngoingEval``): top-m expected-IoU decoding per window, scored
under the old truth (``ds_<p>.npz["y"]``, the labels of every model so far)
and the hybrid truth (``ds_<p>_y2.npz["y"]``, truthfix) on the original
windows; official aggregation for sim / off; plain means for the recurrence
slices; paired bootstrap by window (``OnsetEval.compare``).

    T2_WORK=/home/user/work/t2 T2_FEAT=/home/user/work/t2/feat_v3 \
    python -m trafficflow.t2.og_stack TAG [--seeds 0,1,2] [--rounds 300] [--leaves 31] [--min-data 100]
        [--weighted] [--comp] [--loc] [--drop COLS]
    python -m trafficflow.t2.og_stack table16 [W]          # section 16 table from the saved OOFs
    python -m trafficflow.t2.og_stack diag TAG W           # folds, panels, confidence, size, growth, calibration
    python -m trafficflow.t2.og_stack bias TAG W           # decoder check: logit bias before top-m
    python -m trafficflow.t2.og_stack watch W TAG [TAG...] # non-recurrent / D7_I10_W slices

Writes WORK/ogstack_oof_<TAG>.parquet (gw, k, link, y, fold, src, p1, p2).
"""
from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .core import FEAT, K, PANELS8, WORK, direction, statics
from .models import eiou_topm
from .onset_eval import OnsetEval
from .robust import meta_index, recurrence

V5 = {"oof_queue_ongoing_rob_og_v3_p2_w.parquet": 0.35, "oof_queue_ongoing_rob_og_v3_noloc_p2_w.parquet": 0.35,
      "oof_queue_ongoing_p2w.parquet": 0.15, "oof_queue_ongoing_rob_noloc_p2_w.parquet": 0.15}
# component groups for --comp: models with / without the location & time-of-day priors
LOC_FILES = ("oof_queue_ongoing_rob_og_v3_p2_w.parquet", "oof_queue_ongoing_p2w.parquet")
NOLOC_FILES = ("oof_queue_ongoing_rob_og_v3_noloc_p2_w.parquet", "oof_queue_ongoing_rob_noloc_p2_w.parquet")
KEY = ["gw", "k", "link"]
TRUTH_KEYS = {"old": ("ds_{p}.npz", "y"), "hybrid": ("ds_{p}_y2.npz", "y")}
THREADS = int(os.environ.get("T2_THREADS", "2"))
P2 = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8,
          bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=THREADS, verbose=-1, seed=0)
# per-row columns of the feature tables used as stage-2 context
CTX = ["r_now", "r_last", "d_r15", "d_r30", "c_q_now", "c_qT", "c_q_trend", "c_q_trend60", "tod", "dow", "pq_k"]


# ----------------------------------------------------------------------------
# stage 1
def stage1(spec: dict = V5, comp: bool = False) -> pd.DataFrame:
    """Weighted mean of OOF files (same rows): gw, k, link, y, p (+ p_loc, p_noloc)."""
    B, P, Pl, Pn = None, 0.0, 0.0, 0.0
    for f, w in spec.items():
        x = pd.read_parquet(WORK / f)
        if B is None:
            B = x[KEY + ["y"]].copy()
        else:
            assert (x[KEY].to_numpy() == B[KEY].to_numpy()).all(), f
        p = x.p.to_numpy(np.float64)
        P = P + w * p
        if f in LOC_FILES:
            Pl = Pl + w * p
        if f in NOLOC_FILES:
            Pn = Pn + w * p
    wsum = sum(spec.values())
    B["p"] = P / wsum
    if comp:
        B["p_loc"] = Pl / sum(w for f, w in spec.items() if f in LOC_FILES)
        B["p_noloc"] = Pn / sum(w for f, w in spec.items() if f in NOLOC_FILES)
    return B


# ----------------------------------------------------------------------------
# evaluation
class OngoingEval(OnsetEval):
    """``OnsetEval`` for the ongoing windows of the original set: rows = the v5
    OOF rows (every step, candidate cells); truths old / hybrid."""

    def __init__(self, truths=("old", "hybrid"), work: Path = WORK):
        self.work = Path(work)
        self.truths = tuple(truths)
        self.rows = pd.read_parquet(self.work / next(iter(V5)), columns=KEY)
        self.meta = meta_index("queue_ongoing")
        gw = self.rows.gw.to_numpy()
        kk = self.rows.k.to_numpy().astype(np.int64) - 1
        ll = self.rows.link.to_numpy().astype(np.int64)
        truth = {t: {p: np.load(self.work / TRUTH_KEYS[t][0].format(p=p))[TRUTH_KEYS[t][1]] for p in PANELS8}
                 for t in self.truths}
        rec = recurrence("train")
        order = np.argsort(gw, kind="stable")
        W = []
        for idx in np.split(order, np.flatnonzero(np.diff(gw[order])) + 1):
            g = int(gw[idx[0]]); m = self.meta.loc[g]
            if m.src not in ("sim", "off"):
                continue
            w = dict(gw=g, panel=m.panel, src=m.src, idx=idx, kk=kk[idx], ll=ll[idx])
            for t in self.truths:
                y = truth[t][m.panel][int(m.w)].astype(bool)
                w[f"y_{t}"] = y; w[f"n_{t}"] = int(y.sum())
            W.append(w)
        self.win = W
        info = pd.DataFrame([{k: w[k] for k in ("gw", "panel", "src")} for w in W])
        info["pi"] = info.panel.map({p: i for i, p in enumerate(PANELS8)})
        for t in self.truths:
            info[f"n_{t}"] = [w[f"n_{t}"] for w in W]
        info["rec"] = info.gw.map(rec).to_numpy()
        info["fold"] = self.meta.fold.reindex(info.gw.to_numpy()).to_numpy()
        self.info = info

    def decode(self, p: np.ndarray) -> list[np.ndarray]:
        p = np.asarray(p, np.float64)
        assert len(p) == len(self.rows), (len(p), len(self.rows))
        return [eiou_topm(p[w["idx"]])[0] for w in self.win]

    def window_iou(self, p: np.ndarray) -> pd.DataFrame:
        sets = self.decode(p)
        out = {t: np.empty(len(self.win)) for t in self.truths}
        for i, (w, s) in enumerate(zip(self.win, sets)):
            k, l = w["kk"][s], w["ll"][s]
            for t in self.truths:
                inter = int(w[f"y_{t}"][k, l].sum()); union = len(s) + w[f"n_{t}"] - inter
                out[t][i] = 1.0 if union == 0 else inter / union
        return self.info.assign(**{f"iou_{t}": out[t] for t in self.truths}, m=[len(s) for s in sets])

    def align(self, df: pd.DataFrame, col: str = "p") -> np.ndarray:
        m = self.rows.merge(df[KEY + [col]], on=KEY, how="left")
        assert len(m) == len(self.rows) and m[col].notna().all(), "rows missing"
        return m[col].to_numpy(np.float64)

    def growth(self, p: np.ndarray, truth: str = "old") -> pd.DataFrame:
        """Mean predicted / true / correct cells per step on the sim windows, by recurrence slice."""
        sets = self.decode(p)
        rows = []
        for w, s in zip(self.win, sets):
            if w["src"] != "sim":
                continue
            y = w[f"y_{truth}"]
            npred = np.bincount(w["kk"][s], minlength=K)
            hit = np.bincount(w["kk"][s], weights=y[w["kk"][s], w["ll"][s]], minlength=K)
            rows.append(np.r_[npred, y.sum(1), hit])
        A = pd.DataFrame(rows, columns=[f"{c}{k}" for c in ("pred", "true", "hit") for k in range(1, K + 1)])
        rec = self.info.rec[self.info.src == "sim"].to_numpy()
        out = {}
        for name, m in (("all", np.ones(len(A), bool)), ("rec<0.2", rec < 0.2), ("rec<0.05", rec < 0.05)):
            out[name] = A[m].mean()
        return pd.DataFrame(out).T.round(2)


# ----------------------------------------------------------------------------
# features
def _sh(a: np.ndarray, o: int) -> np.ndarray:
    """Value at link l+o along the last axis, 0 beyond the ends."""
    if o == 0:
        return a
    out = np.zeros_like(a)
    if o > 0:
        out[..., :-o] = a[..., o:]
    else:
        out[..., -o:] = a[..., :o]
    return out


def _blocks(B: np.ndarray):
    """B [..., L] bool in traffic direction. Per link: tail index of the block
    containing it or the next block downstream (L if none), head index of the
    block containing it or the next block upstream (-1 if none), the head of
    the tail-associated block, and the number of blocks."""
    L = B.shape[-1]
    idx = np.arange(L)
    z = np.zeros_like(B[..., :1])
    start = B & ~np.concatenate([z, B[..., :-1]], -1)
    end = B & ~np.concatenate([B[..., 1:], z], -1)
    last_start = np.maximum.accumulate(np.where(start, idx, -1), -1)
    next_start = np.flip(np.minimum.accumulate(np.flip(np.where(start, idx, L), -1), -1), -1)
    last_end = np.maximum.accumulate(np.where(end, idx, -1), -1)
    next_end = np.flip(np.minimum.accumulate(np.flip(np.where(end, idx, L), -1), -1), -1)
    tail = np.where(B, last_start, next_start)
    head = np.where(B, next_end, last_end)
    hot = np.take_along_axis(next_end, np.minimum(tail, L - 1), -1)
    return tail, head, hot, start.sum(-1)


def _geom(B: np.ndarray, st: dict, prefix: str, p: np.ndarray | None = None) -> tuple[dict, np.ndarray, np.ndarray]:
    """Block geometry features [..., L] and the tail (upstream end, km) / head
    (downstream end, km) position of the associated blocks."""
    L = B.shape[-1]
    mp, up = st["mp"], st["mp"] - st["length"]
    xm = (mp + up) / 2
    tail, head, hot, nb = _blocks(B)
    ht = tail < L; hh = head >= 0
    tc = np.minimum(tail, L - 1); hc = np.maximum(head, 0); hoc = np.minimum(hot, L - 1)
    li = np.arange(L)
    F = {f"{prefix}bt": np.where(ht, li - tail, np.nan), f"{prefix}bt_km": np.where(ht, xm - up[tc], np.nan),
         f"{prefix}bh": np.where(hh, head - li, np.nan), f"{prefix}bh_km": np.where(hh, mp[hc] - xm, np.nan),
         f"{prefix}blen": np.where(ht, hot - tail + 1, np.nan), f"{prefix}blen_km": np.where(ht, mp[hoc] - up[tc], np.nan),
         f"{prefix}nblk": np.broadcast_to(nb[..., None], B.shape).astype(np.float32)}
    if p is not None:
        C = np.concatenate([np.zeros_like(p[..., :1]), np.cumsum(p, -1)], -1)
        F[f"{prefix}bmass"] = np.where(ht, np.take_along_axis(C, hoc + 1, -1) - np.take_along_axis(C, tc, -1), np.nan)
    return F, np.where(ht, up[tc], np.nan), np.where(hh, mp[hc], np.nan)


def features(P: np.ndarray, panel: str, ctx: dict, W: dict) -> dict:
    """Stage-2 features of one panel.

    P [n, K, L] stage-1 field (original link order, 0 off the candidate cells).
    ctx: per-link context [n, L] (r_now, r_last, d_r15, d_r30; NaN off the
    candidate links) and per-step context [n, K, L] (pq_k, optional).
    W: per-window context [n] (rec, nq_now, nqT, qtrend, qtrend60, tod, wkend).
    Returns {name: [n, K, L] float32} in the original link order."""
    rev = direction(panel) < 0
    st = statics(panel + "@rev" if rev else panel)
    fl = (lambda a: a[..., ::-1]) if rev else (lambda a: a)
    q = fl(P).astype(np.float32)
    n, Kk, L = q.shape
    mp = st["mp"]; up = mp - st["length"]; xm = (mp + up) / 2
    F = {"s_p": q}
    for o in (-4, -3, -2, -1, 1, 2, 3, 4):
        F[f"s_p_o{o}"] = _sh(q, o)
    nanK = np.full((n, 1, L), np.nan, np.float32)
    pm1 = np.concatenate([nanK, q[:, :-1]], 1); pp1 = np.concatenate([q[:, 1:], nanK], 1)
    F["s_p_km1"] = pm1; F["s_p_kp1"] = pp1
    for o in (-1, 1):
        F[f"s_p_km1_o{o}"] = _sh(pm1, o); F[f"s_p_kp1_o{o}"] = _sh(pp1, o)
    ks = q.sum(2, keepdims=True); km = q.max(2, keepdims=True)
    q0 = np.nan_to_num(fl(ctx["r_now"]), nan=9.0) <= 1.0                 # observed queue at the origin
    n0 = q0.sum(1).astype(np.float32)[:, None, None]
    ks_prev = np.concatenate([n0, ks[:, :-1]], 1)
    F["s_ksum"] = np.broadcast_to(ks, q.shape); F["s_kmax"] = np.broadcast_to(km, q.shape)
    F["s_wsum"] = np.broadcast_to(ks.sum(1, keepdims=True), q.shape)
    F["s_ksum_r1"] = np.broadcast_to(ks / np.maximum(ks[:, :1], 1e-3), q.shape)
    F["s_ksum_d"] = np.broadcast_to(ks - ks_prev, q.shape)
    F["s_ksum_r0"] = np.broadcast_to(ks / np.maximum(n0, 1.0), q.shape)
    F["s_prel"] = q / np.maximum(km, 1e-6)
    F["s_rank"] = np.argsort(np.argsort(-q, 2, kind="stable"), 2).astype(np.float32)
    C = np.concatenate([np.zeros((n, Kk, 1), np.float32), np.cumsum(q, 2)], 2)
    li = np.arange(L)
    for d in (0.5, 1.0, 2.0):
        hi = np.searchsorted(mp, mp + d, side="right"); lo = np.searchsorted(mp, mp - d - 1e-9, side="left")
        F[f"s_dn{d}"] = C[:, :, hi] - C[:, :, li + 1]
        F[f"s_up{d}"] = C[:, :, li] - C[:, :, lo]
    # predicted block geometry per step, and the observed block at the origin / 20 / 35 min before
    G, tpos, hpos = _geom(q >= 0.5, st, "s_", q)
    F.update(G)
    G0, t0, h0 = _geom(q0, st, "o_")
    for k_, v in G0.items():
        if k_ != "o_nblk":
            F[k_] = np.broadcast_to(v[:, None], q.shape)
    F["o_q0"] = np.broadcast_to(q0[:, None].astype(np.float32), q.shape)
    F["o_n0"] = np.broadcast_to(n0, q.shape)
    tprev = np.concatenate([t0[:, None], tpos[:, :-1]], 1); hprev = np.concatenate([h0[:, None], hpos[:, :-1]], 1)
    F["s_dtail_prev"] = tpos - tprev; F["s_dtail_obs"] = tpos - t0[:, None]
    F["s_dhead_prev"] = hpos - hprev; F["s_dhead_obs"] = hpos - h0[:, None]
    r_last = fl(ctx["r_last"])
    dt = np.arange(1, Kk + 1, dtype=np.float32)[None, :, None] * 5.0
    for tag, col, mins in (("20", "d_r15", 20.0), ("35", "d_r30", 35.0)):
        qh = np.nan_to_num(r_last - fl(ctx[col]), nan=9.0) <= 1.0
        _, th, hh_ = _geom(qh, st, "h_")
        vt = (t0 - th) / mins; vh = (h0 - hh_) / mins                     # km per minute (+ = downstream)
        F[f"o_vtail{tag}"] = np.broadcast_to(vt[:, None], q.shape); F[f"o_vhead{tag}"] = np.broadcast_to(vh[:, None], q.shape)
        if tag == "20":
            F["o_text"] = xm - (t0[:, None] + np.nan_to_num(vt, nan=0.0)[:, None] * dt)
    for c in ("r_now", "r_last", "d_r15"):
        F[f"x_{c}"] = np.broadcast_to(fl(ctx[c])[:, None], q.shape)
    if "pq_k" in ctx:
        F["x_pq_k"] = fl(ctx["pq_k"])
    for c in ("p_loc", "p_noloc"):
        if c in ctx:
            a = fl(ctx[c]).astype(np.float32)
            F[f"s_{c}"] = a
            if c == "p_noloc":
                F["s_dloc"] = a - fl(ctx["p_loc"]).astype(np.float32)
    for c, v in W.items():
        F[f"x_{c}"] = np.broadcast_to(np.asarray(v, np.float32)[:, None, None], q.shape)
    F["s_k"] = np.broadcast_to(np.arange(1, Kk + 1, dtype=np.float32)[None, :, None], q.shape)
    F["s_len"] = np.broadcast_to(st["length"].astype(np.float32), q.shape)
    F["s_relpos"] = np.broadcast_to((li / L).astype(np.float32), q.shape)
    F["s_pcode"] = np.full(q.shape, PANELS8.index(panel), np.float32)
    return {k_: np.asarray(fl(v) if v.shape[-1] == L else v, np.float32) for k_, v in F.items()}


def window_context(Fr: pd.DataFrame, rec: pd.Series, key: str, L: int) -> pd.DataFrame:
    """Per-window context from the rows of the feature table (one row per window)."""
    g = Fr.groupby(key, sort=False).first()
    return pd.DataFrame({"rec": rec.reindex(g.index).fillna(0.0).to_numpy(), "nq_now": g.c_q_now * L, "nqT": g.c_qT * L,
                         "qtrend": g.c_q_trend * L, "qtrend60": g.c_q_trend60 * L, "tod": g.tod,
                         "wkend": (g.dow >= 5).astype(np.float32)}, index=g.index)


def panel_rows(R: pd.DataFrame, Fr: pd.DataFrame, panel: str, rec: pd.Series, key: str, use_loc: bool,
               cols: list[str] | None = None) -> pd.DataFrame:
    """Features for the rows R (key, k, link, p[, p_loc, p_noloc]) of one panel.
    Fr: feature-table rows of the same windows (key, k, link, CTX). Returns a
    frame aligned to R with the stage-2 columns (all of them, or ``cols``)."""
    L = statics(panel)["L"]
    wins = pd.Index(pd.unique(R[key]))
    wi = wins.get_indexer(R[key]); ki = R.k.to_numpy().astype(int) - 1; li = R.link.to_numpy().astype(int)
    n = len(wins)
    P = np.zeros((n, K, L), np.float32); P[wi, ki, li] = R.p.to_numpy()
    fw = wins.get_indexer(Fr[key]); assert (fw >= 0).all()
    fk = Fr.k.to_numpy().astype(int) - 1; flk = Fr.link.to_numpy().astype(int)
    ctx = {}
    for c in ("r_now", "r_last", "d_r15", "d_r30"):
        a = np.full((n, L), np.nan, np.float32); a[fw, flk] = Fr[c].to_numpy(); ctx[c] = a
    if use_loc:
        a = np.zeros((n, K, L), np.float32); a[fw, fk, flk] = Fr.pq_k.to_numpy(); ctx["pq_k"] = a
    for c in ("p_loc", "p_noloc"):
        if c in R:
            a = np.zeros((n, K, L), np.float32); a[wi, ki, li] = R[c].to_numpy(); ctx[c] = a
    Wc = window_context(Fr, rec, key, L).reindex(wins)
    F = features(P, panel, ctx, {c: Wc[c].to_numpy() for c in Wc.columns})
    names = cols or list(F)
    return pd.DataFrame({c: F[c][wi, ki, li] for c in names}, index=R.index)


def read_ctx(panel: str, split: str, ws: list, use_loc: bool) -> pd.DataFrame:
    """Context rows of windows ``ws``: split "cv" = the CV training table (fold-excluded
    profiles), else feat_<panel>_<split> (validation / private / the official train windows)."""
    cols = [c for c in CTX if use_loc or c != "pq_k"]
    f = FEAT / (f"feat_{panel}.parquet" if split == "cv" else f"feat_{panel}_{split}.parquet")
    return pd.read_parquet(f, columns=["w", "k", "link"] + cols, filters=[("w", "in", ws)])


def build(S: pd.DataFrame, use_loc=False, drop=()) -> tuple[pd.DataFrame, list[str]]:
    """Stage-2 training frame from the stage-1 OOF rows S (gw, k, link, y, p[, comps])."""
    Mi = meta_index("queue_ongoing")
    rec = recurrence("train")
    parts = []
    for p in PANELS8:
        pc = PANELS8.index(p)
        R = S[S.gw // 100000 == pc]
        ws = np.unique(R.gw.to_numpy() % 100000).tolist()
        Fr = read_ctx(p, "cv", ws, use_loc)
        Fr["gw"] = pc * 100000 + Fr.w.astype(np.int64)
        X = panel_rows(R, Fr, p, rec, "gw", use_loc)
        parts.append(pd.concat([R[KEY + ["y", "p"]], X], axis=1))
        del Fr; gc.collect()
    X = pd.concat(parts).sort_index()                          # original row order
    m = Mi.reindex(X.gw.to_numpy())
    X["fold"] = m.fold.to_numpy(); X["src"] = m.src.to_numpy()
    cols = [c for c in X.columns if c.split("_")[0] in ("s", "o", "x") and not dropped(c, drop)]
    return X, cols


def dropped(c: str, drop) -> bool:
    """``drop`` items are column names or prefixes ending in '*'."""
    return any(c == d or (d.endswith("*") and c.startswith(d[:-1])) for d in drop)


def window_weights(gw: np.ndarray) -> np.ndarray:
    _, inv, cnt = np.unique(gw, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    return (w / w.mean()).astype(np.float32)


def cv(X: pd.DataFrame, cols, params, rounds, weighted=False) -> np.ndarray:
    """OOF stage-2 probabilities (4 week folds; one binned Dataset, fold subsets)."""
    A = X[cols].to_numpy(np.float32)
    ds = lgb.Dataset(A, X.y.to_numpy(np.float32), feature_name=list(cols), free_raw_data=False,
                     weight=window_weights(X.gw.to_numpy()) if weighted else None,
                     params={"verbose": -1, "max_bin": 255}).construct()
    fold = X.fold.to_numpy()
    p = np.full(len(X), np.nan)
    for f in range(4):
        m = lgb.train(params, ds.subset(np.flatnonzero(fold != f)), rounds)
        te = fold == f
        p[te] = m.predict(A[te], num_threads=THREADS)
        del m; gc.collect()
    return p


def stage2_oof(tag: str, seeds=(0,), rounds=300, leaves=31, min_data=100, lr=0.05, weighted=False, comp=False,
               loc=False, drop=()):
    S = stage1(V5, comp=comp)
    X, cols = build(S, use_loc=loc, drop=drop)
    del S; gc.collect()
    p2 = np.zeros(len(X))
    for s in seeds:
        t = time.time()
        p2 += cv(X, cols, {**P2, "num_leaves": leaves, "min_data_in_leaf": min_data, "learning_rate": lr, "seed": s},
                 rounds, weighted)
        print(f"  seed {s}: {time.time() - t:.0f}s", flush=True)
    X["p1"] = X["p"]; X["p2"] = p2 / len(seeds)
    X[KEY + ["y", "fold", "src", "p1", "p2"]].to_parquet(WORK / f"ogstack_oof_{tag}.parquet")
    return X, cols


def saved(tag: str, E: OngoingEval) -> tuple[np.ndarray, np.ndarray]:
    X = pd.read_parquet(WORK / f"ogstack_oof_{tag}.parquet")
    return E.align(X, "p1"), E.align(X, "p2")


WEIGHTS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)


def report(E: OngoingEval, p1: np.ndarray, p2: np.ndarray, weights=WEIGHTS, n_boot=2000) -> pd.DataFrame:
    from .onset_eval import table
    from .stack_v8 import nested_weight
    V = {"v5": p1}
    for w in weights[1:]:
        V[f"w{w}"] = w * p2 + (1 - w) * p1
    t = table(E, V, ref="v5", n_boot=n_boot)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    print(t.round(4).T.to_string(), flush=True)
    for truth in E.truths:
        nest, chosen = nested_weight(E, p1, p2, weights, truth=truth)
        c = E.compare(nest, E.window_iou(p1), n_boot=n_boot)
        print(f"nested weight (chosen on {truth})", chosen,
              {t_: {k: round(v, 4) for k, v in c[t_].items()} for t_ in E.truths}, flush=True)
    return t


def shift(p: np.ndarray, b: float) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + b)))


def surrogate(E: OngoingEval, p: np.ndarray) -> np.ndarray:
    """Per evaluation window: the decoder's expected-IoU surrogate (stage-1 confidence)."""
    return np.array([eiou_topm(p[w["idx"]])[1] for w in E.win])


def iso_step_oof(E: OngoingEval, p1: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-step isotonic calibration of stage 1, fitted out of fold (baseline for stage 2)."""
    from sklearn.isotonic import IsotonicRegression
    fold = E.meta.fold.reindex(E.rows.gw.to_numpy()).to_numpy(); k = E.rows.k.to_numpy()
    out = np.empty_like(p1)
    for f in range(4):
        for kk in range(1, K + 1):
            tr = (fold != f) & (k == kk); te = (fold == f) & (k == kk)
            ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p1[tr], y[tr])
            out[te] = ir.predict(p1[te])
    return out


def diag(tag: str, w: float) -> None:
    """Where the gain of ``w * p2 + (1 - w) * p1`` over v5 comes from, and simple calibration baselines."""
    E = OngoingEval()
    X = pd.read_parquet(WORK / f"ogstack_oof_{tag}.parquet")
    p1, p2 = E.align(X, "p1"), E.align(X, "p2"); y = E.align(X.assign(y=X.y.astype(float)), "y")
    pb = w * p2 + (1 - w) * p1
    a, b = E.window_iou(pb), E.window_iou(p1)
    d = a.assign(d_old=a.iou_old - b.iou_old, d_hyb=a.iou_hybrid - b.iou_hybrid, v5=b.iou_old,
                 sur=surrogate(E, p1))
    s = d[d.src == "sim"]
    pd.set_option("display.width", 250)
    print("Δ by fold (official aggregation within the fold):",
          {f: round(E.agg(g, "d_old"), 4) for f, g in s.groupby("fold")}, "hybrid:",
          {f: round(E.agg(g, "d_hyb"), 4) for f, g in s.groupby("fold")})
    print("Δ by panel (plain mean):\n" + s.groupby("panel")[["d_old", "d_hyb"]].mean().round(4).T.to_string())
    s = s.assign(sur_b=pd.cut(s.sur, [0, 0.6, 0.8, 0.9, 0.95, 1.0]),
                 size_b=pd.cut(s.n_old, [-1, 6, 12, 24, 48, 96, 10 ** 6]))
    for c in ("sur_b", "size_b"):
        print(s.groupby(c, observed=True).agg(n=("d_old", "size"), v5=("v5", "mean"), d_old=("d_old", "mean"),
                                              d_hyb=("d_hyb", "mean")).round(4).to_string())
    P = pd.read_parquet(WORK / "probs_lgb_v5.parquet")
    P = P[P.condition == "queue_ongoing"]
    sv = P.groupby("window_id").p.apply(lambda g: eiou_topm(g.to_numpy())[1])
    sv = pd.DataFrame({"sur": sv, "split": np.where(sv.index.str.contains("_validation_"), "validation", "private")})
    print("stage-1 surrogate: CV sim mean", round(float(s.sur.mean()), 3), "| validation / private:",
          sv.groupby("split").sur.mean().round(3).to_dict())
    print(sv.assign(b=pd.cut(sv.sur, [0, 0.6, 0.8, 0.9, 0.95, 1.0])).groupby(["b", "split"], observed=True)
          .size().unstack().to_string())
    print("growth (mean cells per step, old truth), v5:\n" + E.growth(p1).to_string())
    print(f"growth, stacked w={w}:\n" + E.growth(pb).to_string())
    V = {"v5": p1, f"stack w={w}": pb, "iso per step (OOF)": iso_step_oof(E, p1, y)}
    for bb in (-0.25, 0.25, 0.5):
        V[f"v5 bias {bb:+}"] = shift(p1, bb)
    for bb in (-0.25, 0.25, 0.5):
        V[f"stack bias {bb:+}"] = shift(pb, bb)
    from .onset_eval import table
    t = table(E, V, ref="v5")
    print(t[[c for c in t.columns if c.split(":")[1] in ("sim", "off", "recur<0.05", "recur<0.2", "Δsim", "se")]]
          .round(4).to_string())


def watch(E: OngoingEval, V: dict, ref: str = "v5") -> pd.DataFrame:
    """Per variant: Δ vs ``ref`` (plain means, old truth) on the non-recurrent sim windows,
    overall and on D7_I10_W (the panel whose March ongoing windows are all non-recurrent)."""
    D = {k: E.window_iou(v) for k, v in V.items()}
    b = D[ref]
    s = (b.src == "sim").to_numpy(); r = b.rec.to_numpy(); w = (b.panel == "D7_I10_W").to_numpy()
    rows = []
    for k, a in D.items():
        if k == ref:
            continue
        d = (a.iou_old - b.iou_old).to_numpy()
        rows.append({"variant": k, "rec<0.05": d[s & (r < 0.05)].mean(), "rec<0.05 excl. I10_W": d[s & (r < 0.05) & ~w].mean(),
                     "I10_W rec<0.05": d[s & (r < 0.05) & w].mean(), "I10_W rec<0.2": d[s & (r < 0.2) & w].mean(),
                     "I10_W all": d[s & w].mean(), "n I10_W rec<0.05": int((s & (r < 0.05) & w).sum())})
    return pd.DataFrame(rows).set_index("variant").round(4)


TAGS16 = {"base (3 seeds)": "base_s012", "base, 1 seed": "base_s0", "window weights": "w_s0", "+ loc/noloc components": "comp_s0",
          "+ time-of-day prior pq_k": "loc_s0", "no panel code": "nopcode_s0", "no context (x_*, panel, position)": "noctx_s0",
          "field only (no o_*, x_*)": "field_s0", "**dyn** (3 seeds)": "dyn_s012", "no context (3 seeds)": "noctx_s012",
          "no context, 63 leaves / 500 rounds": "noctx_big_s0"}


def table16(w: float = 0.8, n_boot: int = 2000) -> pd.DataFrame:
    """Section 16 table: every saved stage-2 OOF at blend weight ``w`` vs v5 (both truths), the nested
    estimate (weight chosen on three folds, old truth) and the D7_I10_W non-recurrent watch column."""
    from .stack_v8 import nested_weight
    E = OngoingEval()
    rows = []
    p1 = None
    for name, tag in TAGS16.items():
        if not (WORK / f"ogstack_oof_{tag}.parquet").exists():
            continue
        a, b = saved(tag, E)
        if p1 is None:
            p1 = a
            r0 = E.evaluate(p1)
            rows.append({"variant": "v5 (stage 1)", **{f"{t}:{c}": r0[t][c] for t in E.truths
                                                       for c in ("sim", "off", "recur<0.05", "recur<0.2")}})
        assert np.allclose(a, p1)
        pb = w * b + (1 - w) * a
        r = E.evaluate(pb); c = E.compare(pb, p1, n_boot=n_boot)
        nest, chosen = nested_weight(E, a, b, WEIGHTS, truth="old")
        cn = E.compare(nest, E.window_iou(p1), n_boot=n_boot)
        wt = watch(E, {"v5": p1, name: pb})
        rows.append({"variant": name, **{f"{t}:{k}": r[t][k] for t in E.truths for k in ("sim", "off", "recur<0.05", "recur<0.2")},
                     **{f"{t}:Δsim": c[t]["d_sim"] for t in E.truths}, **{f"{t}:se": c[t]["se_sim"] for t in E.truths},
                     "old:Δoff": c["old"]["d_off"], "old:Δ<0.05": c["old"]["d_recur<0.05"], "old:Δ<0.2": c["old"]["d_recur<0.2"],
                     "nested old Δ": cn["old"]["d_sim"], "nested old se": cn["old"]["se_sim"],
                     "nested hybrid Δ": cn["hybrid"]["d_sim"], "nested w": "/".join(str(chosen[f]) for f in range(4)),
                     "I10_W rec<0.05 Δ": wt.loc[name, "I10_W rec<0.05"]})
    t = pd.DataFrame(rows).set_index("variant")
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
    print(t.round(4).to_string(), flush=True)
    return t


def bias_curve(tag: str, w: float, biases=(-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)) -> pd.DataFrame:
    """Decoder check: logit bias b before top-m decoding, for v5 and the stacked blend (Δ vs plain v5)."""
    E = OngoingEval()
    p1, p2 = saved(tag, E)
    pb = w * p2 + (1 - w) * p1
    d0 = E.window_iou(p1)
    rows = []
    for name, p in (("v5", p1), (f"stack w={w}", pb)):
        for b in biases:
            d = E.window_iou(shift(p, b)); s = E.summary(d); c = E.compare(d, d0, n_boot=500)
            rows.append({"probs": name, "b": b, **{f"{t}:sim": s[t]["sim"] for t in E.truths},
                         **{f"{t}:Δ": c[t]["d_sim"] for t in E.truths}, "old:off": s["old"]["off"],
                         "old:rec<0.05": s["old"]["recur<0.05"], "old:rec<0.2": s["old"]["recur<0.2"],
                         "cells/window": float(d[d.src == "sim"].m.mean())})
    t = pd.DataFrame(rows)
    print(t.round(4).to_string(index=False), flush=True)
    return t


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "table16":
        table16(float(sys.argv[2]) if len(sys.argv) > 2 else 0.8)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "bias":
        bias_curve(sys.argv[2], float(sys.argv[3]))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "diag":
        diag(sys.argv[2], float(sys.argv[3]))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "watch":         # watch W TAG [TAG ...]
        E = OngoingEval(); w = float(sys.argv[2])
        V = {"v5": None}
        for tag in sys.argv[3:]:
            p1, p2 = saved(tag, E)
            V["v5"] = p1 if V["v5"] is None else V["v5"]
            V[tag] = w * p2 + (1 - w) * p1
        print(watch(E, V).to_string())
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--min-data", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--weighted", action="store_true")
    ap.add_argument("--comp", action="store_true")
    ap.add_argument("--loc", action="store_true")
    ap.add_argument("--drop", default="", help="comma list of stage-2 columns or prefixes (ending in *) to drop")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    t = time.time()
    if not a.report_only:
        drop = [c for c in a.drop.split(",") if c]
        X, cols = stage2_oof(a.tag, [int(s) for s in a.seeds.split(",")], a.rounds, a.leaves, a.min_data, a.lr,
                             a.weighted, a.comp, a.loc, drop=tuple(drop))
        print(a.tag, f"{len(cols)} stage-2 features, {len(X)} rows, {time.time() - t:.0f}s", cols, flush=True)
        del X; gc.collect()
    E = OngoingEval()
    p1, p2 = saved(a.tag, E)
    report(E, p1, p2)


if __name__ == "__main__":
    main()
