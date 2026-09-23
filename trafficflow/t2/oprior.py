"""Onset location prior ("where does the first queue appear at this time of day").

For a window with origin T, the share of reference onset windows of the same
panel (train, first-of-run onset candidates, excluding the window's own fold
in CV) whose T+30 truth contains link l, restricted to origins within +-90 /
+-60 min time of day, same weekday class, or any time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import SLOTS
from .baselines import fold_of
from .core import FEAT, K, WORK

COLS = ["op_tod90", "op_tod60", "op_tod90_wk", "op_all", "op_tod90_nb"]


class OnsetPrior:
    def __init__(self, panel: str):
        z = np.load(WORK / f"ds_{panel}.npz", allow_pickle=True)
        src = z["w_src"]; cond = z["w_condition"]
        ref = np.flatnonzero((src == "cand") & (cond == "queue_onset"))
        self.T = z["w_T"][ref].astype(np.int64)
        self.y6 = z["y"][ref, K - 1].astype(np.float32)
        self.fold = fold_of(self.T)
        self.tod = self.T % SLOTS
        self.wk = ((self.T // SLOTS + 5) % 7) >= 5

    def features(self, T: np.ndarray, fold: np.ndarray | None) -> dict:
        """Per window [n, L] arrays. fold=None -> use every reference window."""
        T = np.asarray(T, np.int64)
        tod = T % SLOTS; wk = ((T // SLOTS + 5) % 7) >= 5
        n = len(T); L = self.y6.shape[1]
        out = {c: np.zeros((n, L), np.float32) for c in COLS}
        for i in range(n):
            ok = np.ones(len(self.T), bool) if fold is None else (self.fold != fold[i])
            dt = np.abs(self.tod - tod[i])
            for name, m in (("op_tod90", ok & (dt <= 18)), ("op_tod60", ok & (dt <= 12)),
                            ("op_tod90_wk", ok & (dt <= 18) & (self.wk == wk[i])), ("op_all", ok)):
                out[name][i] = self.y6[m].mean(0) if m.sum() >= 3 else np.nan
        x = out["op_tod90"]
        out["op_tod90_nb"] = np.nanmax(np.stack([x, np.roll(x, 1, 1), np.roll(x, -1, 1)]), 0)
        return out


def add_to_rows(R, M, panels):
    """Append onset-prior columns to a cv.Rows of onset rows (in place)."""
    from .core import PANELS8
    extra = np.full((len(R), len(COLS)), np.nan, np.float32)
    Mi = M.drop_duplicates("gw").set_index("gw")
    for p in panels:
        pc = PANELS8.index(p)
        rows = np.flatnonzero(R.gw // 100000 == pc)
        if not len(rows):
            continue
        op = OnsetPrior(p)
        gws = np.unique(R.gw[rows])
        feats = op.features(Mi.loc[gws, "T"].to_numpy(), Mi.loc[gws, "fold"].to_numpy())
        pos = {g: i for i, g in enumerate(gws)}
        wi = np.array([pos[g] for g in R.gw[rows]])
        li = R.link[rows].astype(int)
        for j, c in enumerate(COLS):
            extra[rows, j] = feats[c][wi, li]
    R.X = np.hstack([R.X, extra])
    R.cols = list(R.cols) + COLS
    return R
