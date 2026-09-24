"""Shift-robustness experiments for the ongoing (and onset) models.

Motivation (public LB, March = validation): ongoing IoU 0.812 vs CV 0.877.
The March ongoing windows contain many more queues at places where train
rarely queues at that time of day (incident-like): 12.5% of windows have a
queued-link mean train queue probability ("recur") < 0.05, against 2.6% of the
train CV windows, and the CV model scores only ~0.56 on those.

``recur`` of a window = mean over the links queued at the end of the history
(filled last observation) of the train time-of-day queue probability at T
(same weekday, profile learned without the window's fold). It uses only data
<= T, so it can also gate models at test time.

    python -m trafficflow.t2.robust cv VARIANT [CFG] [--weighted]
    python -m trafficflow.t2.robust blend
"""
from __future__ import annotations

import argparse
import gc
import os
import time

import numpy as np
import pandas as pd

from .core import FEAT, K, PANELS8, WORK, aggregate, iou
from .cv import gather, oof, truth_lookup
from .models import CFG, eiou_topm

LOC = ["pq_T", "pq_Tm30", "pq_up2", "pq_dn2", "pq_k", "pq_all_k", "rq7_T", "rq7_k"]
ID = ["pcode", "relpos", "dist_end", "vf", "lanes", "cap_lane", "length", "ramps_on", "ramps_off"]
TIME = ["tod", "dow"]
EARLY = ["early_rmin", "early_qfrac", "early_nq"]
VARIANTS = {
    "base": [],
    "op": [],                        # onset: + location prior (oprior.py)
    "noloc": LOC,
    "noloc_notime": LOC + TIME,
    "dyn": LOC + ID + TIME,          # dynamics only
    "dyn_noearly": LOC + ID + TIME + EARLY,
}
PH = ["ph_fcap_slope", "ph_fcap_ext", "ph_fcap_max3", "ph_k_ratio", "ph_k_slope", "ph_k_ext", "ph_k_jam",
      "ph_r_slope", "ph_r_ext", "ph_r_min3", "ph_r_margin_min", "ph_cap_dn1", "ph_cap_up1", "ph_lanes_dn",
      "ph_lanes_up", "ph_capmin_dn", "ph_dr_dn", "ph_dr_up", "ph_dr_dnup3", "ph_d08_dn", "ph_d08_up", "ph_d07_dn",
      "ph_d07_up", "ph_qobs", "ph_qobs_age", "ph_dq_dn", "ph_dq_up", "ph_c_qobs", "ph_fup_cap", "ph_fup_slope"]
LW = ["lw_s_tail", "lw_s_emp", "lw_s_head", "lw_qu_cap", "lw_qq_cap", "lw_ku_kc", "lw_kq_kc", "lw_xrel_tail",
      "lw_xrel_head", "lw_blen", "lw_inblk", "lw_tail_k", "lw_tail_emp_k", "lw_head_k"]
VARIANTS.update({
    # feat_v3 tables (T2_FEAT=.../feat_v3): onset rows carry PH, ongoing rows carry LW
    "on_v2": PH + LW, "on_v3": LW,
    "og_v2": PH + LW, "og_v3": PH, "og_v3_noloc": PH + LOC,
})
SHIFT_THR = (0.05, 0.2)


# ----------------------------------------------------------------------------
def recurrence(split: str = "train") -> pd.Series:
    """Window recurrence for ongoing windows, indexed by gw (train) or window_id."""
    out = []
    for p in PANELS8:
        pc = PANELS8.index(p)
        if split == "train":
            M = pd.read_parquet(FEAT / f"meta_{p}.parquet")
            ids = M.w[M.condition == "queue_ongoing"].tolist()
            X = pd.read_parquet(FEAT / f"feat_{p}.parquet", columns=["w", "r_last", "pq_T"],
                                filters=[("w", "in", ids), ("k", "==", 1.0)])
        else:
            M = pd.read_parquet(FEAT / f"meta_{p}_{split}.parquet")
            ids = M.w[M.condition == "queue_ongoing"].tolist()
            X = pd.read_parquet(FEAT / f"feat_{p}_{split}.parquet", columns=["w", "r_last", "pq_T"],
                                filters=[("w", "in", ids), ("k", "==", 1.0)])
        q = X[X.r_last <= 1.0]
        r = q.groupby("w").pq_T.mean().reindex(ids).fillna(0.0)
        if split == "train":
            r.index = pc * 100000 + r.index.astype(np.int64)
        else:
            r.index = M.set_index("w").window_id.reindex(r.index).to_numpy()
        out.append(r)
    return pd.concat(out)


def window_scores(O: pd.DataFrame, Mi: pd.DataFrame, Y: dict, dec=None) -> pd.DataFrame:
    """Per-window IoU (top-m decoding) from OOF rows gw,k,link,p."""
    O = O.sort_values("gw", kind="stable")
    g = O.gw.to_numpy(); b = np.flatnonzero(np.diff(g)) + 1
    kk_all = O.k.to_numpy().astype(int) - 1; ll_all = O.link.to_numpy().astype(int); pp_all = O.p.to_numpy()
    rows = []
    for idx in np.split(np.arange(len(O)), b):
        gw = int(g[idx[0]]); m = Mi.loc[gw]
        yt = Y[m.panel][int(m.w)]
        pp = pp_all[idx]
        sel, r = eiou_topm(pp) if dec is None else dec(pp)
        P = np.zeros_like(yt); P[kk_all[idx][sel], ll_all[idx][sel]] = True
        rows.append(dict(gw=gw, panel=m.panel, condition=m.condition, src=m.src, iou=iou(P, yt), eiou=r))
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, rec: pd.Series, cond: str) -> dict:
    df = df.assign(recur=df.gw.map(rec))
    s = df[df.src == "sim"]
    out = {"sim": aggregate(s)[cond], "off": aggregate(df[df.src == "off"])[cond]}
    for t in SHIFT_THR:
        x = s[s.recur < t]
        out[f"recur<{t}"] = float(x.iou.mean())
        out[f"n<{t}"] = len(x)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def run_cv(variant: str, cfg: str, weighted: bool, cond: str = "queue_ongoing", oprior: bool = False):
    params, rounds = CFG[cfg]
    seed = int(os.environ.get("T2_SEED", "0"))
    params = {**params, "seed": seed}
    tag = f"rob_{variant}_{cfg}{'_w' if weighted else ''}{'_op' if oprior else ''}{f'_s{seed}' if seed else ''}"
    t = time.time()
    R, M = gather(cond, cand_frac=0.5 if cond == "queue_ongoing" else 1.0, drop=VARIANTS[variant])
    if cond == "queue_onset" and (variant.startswith("op") or oprior):
        from .oprior import add_to_rows
        R = add_to_rows(R, M, PANELS8)
    meta = dict(gw=R.gw, k=R.k, link=R.link, y=R.y)
    pred_all = os.environ.get("T2_OOF_ALL") == "1"
    ev = np.ones(len(R), bool) if pred_all else R.ev.copy()
    if pred_all:
        tag += "_all"
    tag += os.environ.get("T2_TAGX", "")
    p = oof(R, params, rounds, weighted=weighted, pred_all=pred_all)
    O = pd.DataFrame({k: v[ev] for k, v in meta.items()} | {"p": p[ev]})
    O.to_parquet(WORK / f"oof_{cond}_{tag}.parquet")
    del R; gc.collect()
    Mi = M.drop_duplicates("gw").set_index("gw")
    O = O[O.gw.map(Mi.src).isin(["sim", "off"]).to_numpy()]
    df = window_scores(O, Mi, truth_lookup_y())
    rec = recurrence("train") if cond == "queue_ongoing" else onset_recurrence()
    res = report(df, rec, cond)
    print(tag, res, f"{time.time()-t:.0f}s", flush=True)
    return res


_Y = None


def truth_lookup_y():
    """Window truths [n, K, L] per panel. T2_TRUTH=y2 switches to the imputed
    truth (truthfix.py) with onset windows' steps 1-5 set empty (the organizer
    confirms the onset horizon is empty before T+30)."""
    global _Y
    if _Y is None:
        _Y = {p: load_truth(p) for p in PANELS8}
    return _Y


def load_truth(p: str) -> np.ndarray:
    z = np.load(WORK / f"ds_{p}.npz", allow_pickle=True)
    if os.environ.get("T2_TRUTH") != "y2":
        return z["y"]
    y2 = np.load(WORK / f"ds_{p}_y2.npz")["y"].copy()
    on = z["w_condition"] == "queue_onset"
    y2[on, :K - 1] = False
    return y2


def meta_index(cond="queue_ongoing"):
    Ms = []
    for p in PANELS8:
        M = pd.read_parquet(FEAT / f"meta_{p}.parquet")
        M["gw"] = PANELS8.index(p) * 100000 + M.w.astype(np.int64)
        Ms.append(M[M.condition == cond])
    return pd.concat(Ms).drop_duplicates("gw").set_index("gw")


def blend(files: list[str], weights_grid=None, cond="queue_ongoing", gate: tuple | None = None):
    """Evaluate convex blends of saved OOF probability files (same rows).
    gate=(i, thr): use file i alone on windows with recur < thr."""
    Os = [pd.read_parquet(WORK / f) for f in files]
    key = ["gw", "k", "link"]
    base = Os[0][key + ["y"]].copy()
    for i, O in enumerate(Os):
        base = base.merge(O[key + ["p"]].rename(columns={"p": f"p{i}"}), on=key, how="inner")
    Mi = meta_index(cond)
    Y = truth_lookup_y()
    rec = recurrence("train")
    res = {}
    for w in (weights_grid or [[1.0] + [0.0] * (len(Os) - 1)]):
        p = sum(wi * base[f"p{i}"].to_numpy() for i, wi in enumerate(w))
        if gate is not None:
            gi, thr = gate
            low = base.gw.map(rec).to_numpy() < thr
            p = np.where(low, base[f"p{gi}"].to_numpy(), p)
        df = window_scores(base.assign(p=p), Mi, Y)
        res[str(w)] = report(df, rec, cond)
        print(files, w, gate, res[str(w)], flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("variant", nargs="?", default="base")
    ap.add_argument("cfg", nargs="?", default="fast")
    ap.add_argument("--weighted", action="store_true")
    ap.add_argument("--cond", default="queue_ongoing")
    ap.add_argument("--oprior", action="store_true")
    a = ap.parse_args()
    if a.mode == "cv":
        run_cv(a.variant, a.cfg, a.weighted, a.cond, a.oprior)



def onset_recurrence() -> pd.Series:
    """Evaluation-only onset shift proxy (uses truth): mean train time-of-day
    queue probability at T+30 (fold-excluded profile) over the links that truly
    queue at T+30."""
    out = []
    for p in PANELS8:
        pc = PANELS8.index(p)
        M = pd.read_parquet(FEAT / f"meta_{p}.parquet")
        ids = M.w[M.condition == "queue_onset"].tolist()
        X = pd.read_parquet(FEAT / f"feat_{p}.parquet", columns=["w", "pq_k", "y"], filters=[("w", "in", ids)])
        r = X[X.y == 1].groupby("w").pq_k.mean().reindex(ids).fillna(0.0)
        r.index = pc * 100000 + r.index.astype(np.int64)
        out.append(r)
    return pd.concat(out)


def onset_compare(files: list[str], thr=(0.05, 0.2)):
    Mi = meta_index("queue_onset")
    Y = truth_lookup_y()
    rec = onset_recurrence()
    for f in files:
        O = pd.read_parquet(WORK / f)
        df = window_scores(O, Mi, Y).assign(recur=lambda d: d.gw.map(rec))
        s = df[df.src == "sim"]
        res = {"sim": round(aggregate(s)["queue_onset"], 4), "off": round(aggregate(df[df.src == "off"])["queue_onset"], 4)}
        for t in thr:
            x = s[s.recur < t]
            res[f"recur<{t}"] = round(float(x.iou.mean()), 4); res[f"n<{t}"] = len(x)
        print(f, res, flush=True)


if __name__ == "__main__":
    main()
