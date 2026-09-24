"""Task 2 core: panel statics, queue truth, the window selector, IoU scoring.

Everything works on the dense [T, L] cache of ``trafficflow.data`` whose link
axis is the topology order_index, which on every scored panel is monotone in
milepost (upstream -> downstream in the direction of travel).

Window selector (reverse-engineered from the official train windows, reproduces
~84% of them exactly and the rest to within one 5-minute slot, the residual
being threshold noise in our approximate truth):

* candidate origin T on the 5-min grid; history = slots [T-12, T), horizon =
  slots T+1..T+6 (the origin slot T itself is neither);
* history coverage = share of eligible cells in the history >= 0.7;
* condition from the visible history: ``ongoing`` when >= 2 queued eligible
  observations and some link has >= 2, else ``onset``;
* at least one truly queued cell in the horizon;
* persistence IoU of the *true* state at slot T repeated over the horizon
  against the true horizon <= 0.9;
* chronological greedy, 5 windows per condition, every pair of selected origins
  (any condition) at least 360 min apart.

Consequence: an onset window is the first origin whose horizon touches a new
queue, so its truth is (almost always) confined to the last step T+30.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import REL, SLOTS, SPLIT_DAYS, T0, load, network, tindex

PANELS8 = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W",
           "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S"]
FAMILY = {p: p.rsplit("_", 1)[0] for p in PANELS8}
WORK = Path(os.environ.get("T2_WORK", "/home/user/work/t2"))
FEAT = Path(os.environ.get("T2_FEAT", str(WORK / "feat_v2")))  # feature tables
H = 12          # history slots
K = 6           # horizon steps
SPACING = 72    # 360 min in slots
TRAIN_END = SPLIT_DAYS["train"][1] * SLOTS


# ----------------------------------------------------------------------------
# statics
@lru_cache(None)
def statics(panel: str) -> dict:
    """Per-link static arrays in cache (spatial) order. ``<panel>@rev`` gives
    the same panel with the link axis reversed (mileposts recomputed), used to
    put every panel in traffic direction (index increasing downstream)."""
    if panel.endswith("@rev"):
        base = statics(panel[:-4])
        out = dict(base)
        for k, v in base.items():
            if isinstance(v, np.ndarray) and v.shape == (base["L"],):
                out[k] = v[::-1].copy()
        out["links"] = base["links"][::-1]
        out["lid"] = {l: i for i, l in enumerate(out["links"])}
        out["mp"] = np.cumsum(out["length"])
        return out
    links = [str(x) for x in network(panel)["links"]]
    nd = REL / "corridors" / panel / "network"
    lk = pd.read_csv(nd / "links.csv", dtype={"link_id": str}).drop_duplicates("link_id").set_index("link_id").reindex(links)
    fd = pd.read_csv(nd / "fd_parameters.csv", dtype={"link_id": str}).drop_duplicates("link_id").set_index("link_id").reindex(links)
    vf = lk.free_speed_kmh.to_numpy(np.float64)
    vcut = 0.60 * vf
    if "v_cut" in fd.columns:
        vc = fd.v_cut.to_numpy(np.float64)
        vcut = np.where(np.isfinite(vc), vc, vcut)
    length = fd.length_km.to_numpy(np.float64)
    # milepost of the link's downstream end = cumulative length (matches the
    # released milepost to rounding; order_index is monotone in milepost)
    mp = np.cumsum(length)
    ramps = pd.read_csv(nd / "ramp_attachment_map.csv", dtype=str)
    on = np.zeros(len(links)); off = np.zeros(len(links))
    lid = {l: i for i, l in enumerate(links)}
    for r in ramps.itertuples():
        i = lid.get(r.nearest_mainline_link_id)
        if i is None:
            continue
        if str(r.ramp_type).upper() in ("OR", "ON"):
            on[i] += 1
        else:
            off[i] += 1
    return dict(links=links, lid=lid, L=len(links), vf=vf, vcut=vcut.astype(np.float32),
                length=length, mp=mp, lanes=fd.lanes.to_numpy(np.float64),
                cap=fd.capacity_vph.to_numpy(np.float64), kc=fd.critical_density.to_numpy(np.float64),
                kjam=fd.k_jam.to_numpy(np.float64), on=on, off=off)


@lru_cache(None)
def direction(panel: str) -> int:
    """+1 if the link index increases downstream (E/N panels), -1 if link i+1
    is the incoming (upstream) link of link i (W/S panels)."""
    net = network(panel); topo = net["topo"]; links = net["links"]
    nxt = {l: (str(o).split(";") if isinstance(o, str) else []) for l, o in zip(topo.link_id.astype(str), topo.outgoing_link_ids)}
    prv = {l: (str(o).split(";") if isinstance(o, str) else []) for l, o in zip(topo.link_id.astype(str), topo.incoming_link_ids)}
    fwd = sum(links[i + 1] in nxt[links[i]] for i in range(len(links) - 1))
    bwd = sum(links[i + 1] in prv[links[i]] for i in range(len(links) - 1))
    return 1 if fwd >= bwd else -1


def fill_truth(S: np.ndarray, tlim: int = 3, slim: int = 2) -> np.ndarray:
    """Approximate the underlying speed on train from the unmasked observation:
    linear interpolation in time (gaps <= tlim), then in space (<= slim links),
    then a wider time fill; anything left stays NaN (treated as not queued)."""
    f = pd.DataFrame(S).interpolate(limit=tlim, limit_area="inside").to_numpy(np.float32)
    f = pd.DataFrame(f.T).interpolate(limit=slim, limit_area="inside").to_numpy(np.float32).T
    f = pd.DataFrame(f).interpolate(limit=12, limit_area="inside").to_numpy(np.float32)
    return f


# ----------------------------------------------------------------------------
# panel data for Task 2
class PanelT2:
    """Holds what Task 2 needs from one panel, in float32/bool [T, L]."""

    def __init__(self, panel: str, with_truth: bool = True):
        self.panel = panel
        st = statics(panel)
        self.st = st
        d = load(panel)
        self.L = st["L"]
        self.vcut = st["vcut"]
        self.elig = d["elig"]
        self.pct = d["pct"]
        # observation layer as a Task 2 history would show it: unmasked on
        # train (window histories are never Task-1 masked), masked view elsewhere
        sp = d["speed"].copy(); fl = d["flow"].copy()
        sp[:TRAIN_END] = d["tspeed"][:TRAIN_END]
        fl[:TRAIN_END] = d["tflow"][:TRAIN_END]
        self.speed, self.flow = sp, fl
        if with_truth:
            tf = fill_truth(d["tspeed"][:TRAIN_END])
            self.tfill = tf
            self.qtrue = np.zeros((TRAIN_END, self.L), bool)
            self.qtrue[:] = tf <= self.vcut
        del d

    # ------------------------------------------------------------------
    def selector_stats(self, t0: int = 12, t1: int | None = None) -> pd.DataFrame:
        """Per-origin selector quantities over the train period (vectorised)."""
        t1 = t1 or TRAIN_END - K - 1
        E = self.elig[:TRAIN_END].astype(np.int32)
        Qo = ((self.speed[:TRAIN_END] <= self.vcut) & self.elig[:TRAIN_END]).astype(np.int32)
        Qt = self.qtrue
        cE = np.vstack([np.zeros((1, self.L), np.int32), np.cumsum(E, 0)])
        cQ = np.vstack([np.zeros((1, self.L), np.int32), np.cumsum(Qo, 0)])
        T = np.arange(t0, t1)
        cov = (cE[T] - cE[T - H]).sum(1) / (H * self.L)
        perlink = cQ[T] - cQ[T - H]
        nq = perlink.sum(1); maxl = perlink.max(1)
        ongoing = (nq >= 2) & (maxl >= 2)
        cq = Qt.sum(1)
        nfut = np.zeros(len(T), np.int64); inter = np.zeros(len(T)); union = np.zeros(len(T))
        nstep = np.zeros((len(T), K), np.int64)
        QT = Qt[T]
        for k in range(1, K + 1):
            Qk = Qt[T + k]
            nstep[:, k - 1] = Qk.sum(1)
            inter += (QT & Qk).sum(1); union += (QT | Qk).sum(1)
        nfut = nstep.sum(1)
        piou = np.where(union == 0, 1.0, inter / np.maximum(union, 1))
        df = pd.DataFrame(dict(T=T, cov=cov, nq=nq, maxl=maxl, ongoing=ongoing, nfut=nfut, piou=piou, nT=QT.sum(1)))
        for k in range(K):
            df[f"n{k+1}"] = nstep[:, k]
        df["cand"] = (df["cov"] >= 0.7) & (df.nfut > 0) & (df.piou <= 0.9)
        return df

    @staticmethod
    def greedy(stats: pd.DataFrame, start: int, end: int, quota: int = 5) -> list[tuple[int, str]]:
        c = stats[(stats.cand) & (stats["T"] >= start) & (stats["T"] < end)]
        acc = []; cnt = {True: 0, False: 0}
        for T, og in zip(c["T"].to_numpy(), c.ongoing.to_numpy()):
            if cnt[og] >= quota:
                continue
            if all(abs(T - a) >= SPACING for a, _ in acc):
                acc.append((int(T), bool(og))); cnt[og] += 1
                if cnt[True] >= quota and cnt[False] >= quota:
                    break
        return [(T, "queue_ongoing" if og else "queue_onset") for T, og in acc]


def simulate_windows(P: PanelT2, stats: pd.DataFrame, days: range, horizon_days: int = 40) -> pd.DataFrame:
    """Run the official selector from every start day in ``days`` (as if a split
    began that day) and return the unique windows with how often each was drawn."""
    out = {}
    for s in days:
        start = s * SLOTS + H
        end = min(TRAIN_END - K - 1, (s + horizon_days) * SLOTS)
        for T, c in PanelT2.greedy(stats, start, end):
            key = (T, c)
            out[key] = out.get(key, 0) + 1
    w = pd.DataFrame([(T, c, n) for (T, c), n in out.items()], columns=["T", "condition", "draws"])
    return w.sort_values("T").reset_index(drop=True)


# ----------------------------------------------------------------------------
# official windows
def official_windows(panel: str, split: str) -> pd.DataFrame:
    w = pd.read_csv(REL / "task2" / panel / split / "window_index.csv")
    w["T"] = tindex(w.forecast_origin)
    return w


def official_history(panel: str, split: str) -> dict:
    """window_id -> dict(speed, flow, pct, elig) arrays [H, L] from the release."""
    st = statics(panel)
    h = pd.read_parquet(REL / "task2" / panel / split / "window_history.parquet")
    w = official_windows(panel, split).set_index("window_id")
    out = {}
    for wid, g in h.groupby("window_id", sort=False):
        T = int(w.loc[wid, "T"])
        ti = tindex(g.timestamp) - (T - H)
        li = g.link_id.map(st["lid"]).to_numpy()
        a = {k: np.full((H, st["L"]), np.nan, np.float32) for k in ("speed", "flow")}
        a["speed"][ti, li] = g.speed_kmh.to_numpy(np.float32)
        a["flow"][ti, li] = g.flow_vph.to_numpy(np.float32)
        pct = np.zeros((H, st["L"]), np.int16); pct[ti, li] = g.pct_observed.to_numpy()
        el = np.zeros((H, st["L"]), bool); el[ti, li] = g.is_score_eligible.astype(bool).to_numpy()
        a["pct"] = pct; a["elig"] = el
        out[wid] = a
    return out


def template(panel: str, split: str) -> pd.DataFrame:
    return pd.read_csv(REL / "task2" / panel / split / "sample_submission_queue.csv", dtype=str,
                       usecols=["window_id", "timestamp", "link_id"])


# ----------------------------------------------------------------------------
# scoring
def iou(pred: np.ndarray, true: np.ndarray) -> float:
    u = np.logical_or(pred, true).sum()
    return 1.0 if u == 0 else float(np.logical_and(pred, true).sum() / u)


def aggregate(df: pd.DataFrame, col: str = "iou") -> dict:
    """Official aggregation: window -> panel&condition -> panel -> family -> mean.
    ``df`` needs panel, condition and ``col``. Returns overall + per condition."""
    pc = df.groupby(["panel", "condition"])[col].mean().reset_index()
    pm = pc.groupby("panel")[col].mean()
    fam = pm.groupby(pm.index.map(FAMILY)).mean()
    res = {"overall": float(fam.mean())}
    for c in ("queue_onset", "queue_ongoing"):
        x = pc[pc.condition == c].set_index("panel")[col]
        res[c] = float(x.groupby(x.index.map(FAMILY)).mean().mean()) if len(x) else float("nan")
    return res
