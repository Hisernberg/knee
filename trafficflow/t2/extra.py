"""Features from released data at or before the origin T beyond the 60-min
window history (allowed by the organizer's Task 2 data rule, forum #742068:
"a submission may use released data with a timestamp at or before T").

Source is the masked mainline view of every split (Task-1 blanks and the Task-2
horizon/buffer blanks are simply missing), which is what a participant has for
validation/private; train windows use the same masked view so the train and
test feature distributions match.

* slot T itself (the origin slot, neither history nor horizon);
* earlier the same day: the 2 hours before the history;
* the previous 7 days at the same time of day (recent-days queue climatology,
  which tracks month-to-month changes in which bottlenecks are active).
"""
from __future__ import annotations

import numpy as np

from ..data import SLOTS
from .core import H, K


class MaskedView:
    def __init__(self, d: dict, vcut: np.ndarray, cap: np.ndarray):
        self.speed = d["speed"]; self.flow = d["flow"]; self.elig = d["elig"]
        self.vcut = vcut.astype(np.float32); self.cap = cap.astype(np.float32)
        ok = self.elig & ~np.isnan(self.speed)
        self.ok = ok
        self.q = ok & (self.speed <= self.vcut)


def extra_features(mv: MaskedView, T: np.ndarray, hist_last_r: np.ndarray) -> tuple[dict, dict]:
    """Shared [n, L] features and per-step [n, K, L] features."""
    T = np.asarray(T, np.int64)
    n = len(T); L = mv.speed.shape[1]
    F = {}
    sT = mv.speed[T]
    rT = sT / mv.vcut
    F["rT"] = rT
    F["fT"] = mv.flow[T] / mv.cap
    F["rT_ok"] = (~np.isnan(rT)).astype(np.float32)
    r_now = np.where(np.isnan(rT), hist_last_r, rT)
    F["r_now"] = r_now
    F["dT_last"] = rT - hist_last_r
    # neighbours of the origin-slot value (spatially filled by r_now)
    for o in (-3, -2, -1, 1, 2, 3):
        idx = np.clip(np.arange(L) + o, 0, L - 1)
        F[f"rnow_o{o}"] = r_now[:, idx]
    # earlier the same day: [T-36, T-12)
    idx = T[:, None] + np.arange(-3 * H, -H)[None, :]
    idx = np.maximum(idx, 0)
    okb = mv.ok[idx]; qb = mv.q[idx]
    with np.errstate(all="ignore"):
        rb = np.where(okb, mv.speed[idx] / mv.vcut, np.nan)
        F["early_rmin"] = np.nanmin(rb, 1)
        F["early_qfrac"] = qb.sum(1) / np.maximum(okb.sum(1), 1)
    F["early_nq"] = qb.sum(1).astype(np.float32)
    # recent days climatology (previous 7 days, +-2 slots)
    day = T // SLOTS; tod = T % SLOTS
    offs = np.arange(-2, 3)
    step = {}
    def clim(shift):
        acc_q = np.zeros((n, L), np.float32); acc_o = np.zeros((n, L), np.float32)
        for j in range(1, 8):
            base = (day - j) * SLOTS + tod + shift
            ii = base[:, None] + offs[None, :]
            valid = (day - j) >= 0
            ii = np.clip(ii, 0, mv.speed.shape[0] - 1)
            qo = mv.q[ii].sum(1).astype(np.float32); oo = mv.ok[ii].sum(1).astype(np.float32)
            acc_q += np.where(valid[:, None], qo, 0); acc_o += np.where(valid[:, None], oo, 0)
        return np.where(acc_o > 0, acc_q / np.maximum(acc_o, 1), np.nan).astype(np.float32)
    F["rq7_T"] = clim(0)
    step["rq7_k"] = np.stack([clim(k) for k in range(1, K + 1)], 1)
    return F, step
