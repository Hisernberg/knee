"""Rule-based ongoing forecaster: persistence + kinematic-wave extrapolation.

Queue blocks (maximal runs of queued links in the filled last-slot state) are
matched between 15 min ago and now by overlap; each block's upstream tail and
downstream head are extrapolated linearly in milepost (tail speed clipped to
[-vmax, vmax] km/h), and the block is drawn at every horizon step.
"""
from __future__ import annotations

import numpy as np

from .core import K
from .features import _ffill_time, _fill_space


def _runs(q: np.ndarray):
    idx = np.flatnonzero(q)
    if not len(idx):
        return []
    cut = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[cut + 1]]; ends = np.r_[idx[cut], idx[-1]]
    return list(zip(starts, ends))


def predict_kinematic(hs: np.ndarray, vcut: np.ndarray, mp: np.ndarray, lag: int = 3,
                      vmax: float = 25.0, gain: float = 1.0, rT: np.ndarray | None = None) -> np.ndarray:
    """hs [n, 12, L] history speeds -> bool [n, K, L]."""
    n, h, L = hs.shape
    rf, _ = _ffill_time(hs / vcut)
    rf = _fill_space(rf)
    now = rf[:, -1].copy()
    if rT is not None:
        now = np.where(np.isnan(rT), now, rT)
    out = np.zeros((n, K, L), bool)
    up = mp - np.r_[mp[0], np.diff(mp)]  # upstream end of each link
    for i in range(n):
        qn = now[i] <= 1.0; qp = rf[i, -1 - lag] <= 1.0
        bn = _runs(qn); bp = _runs(qp)
        dt_h = (lag + (1 if rT is not None else 0)) * 5 / 60.0
        for a, b in bn:
            # matching previous block: largest overlap (else a new block -> no motion info)
            best = None; ov = 0
            for c, d in bp:
                o = min(b, d) - max(a, c) + 1
                if o > ov:
                    ov, best = o, (c, d)
            if best is None:
                v_tail = 0.0; v_head = 0.0
            else:
                v_tail = (up[best[0]] - up[a]) / dt_h      # >0: tail moving upstream (growing)
                v_head = (mp[b] - mp[best[1]]) / dt_h      # >0: head moving downstream
            v_tail = float(np.clip(gain * v_tail, -vmax, vmax)); v_head = float(np.clip(gain * v_head, -vmax, vmax))
            for k in range(1, K + 1):
                t_h = (k + (0 if rT is not None else 1)) * 5 / 60.0
                x_tail = up[a] - v_tail * t_h
                x_head = mp[b] + v_head * t_h
                if x_head <= x_tail:
                    continue
                sel = (mp > x_tail + 1e-6) & (up < x_head - 1e-6)
                out[i, k - 1] |= sel
    return out


def main():
    """Score persistence variants and the kinematic rule on simulated/official
    train ongoing windows (origin-slot values from the masked view)."""
    import pandas as pd

    from ..data import load
    from .baselines import last_valid, pred_persistence, pred_persistence_fill
    from .core import PANELS8, aggregate, iou, statics
    from .dataset import load_ds

    rows = []
    for p in PANELS8:
        W, A = load_ds(p); st = statics(p)
        ev = W[W.src.isin(["sim", "off"]) & (W.condition == "queue_ongoing")]
        idx = ev.index.to_numpy(); T = ev["T"].to_numpy()
        d = load(p); rT = d["speed"][T] / st["vcut"]; del d
        hs = A["hs"][idx]; y = A["y"][idx]
        nowr = np.where(np.isnan(rT), last_valid(hs) / st["vcut"], rT)
        preds = {"persistence": pred_persistence(hs, A["he"][idx], st["vcut"]),
                 "persistence_fill": pred_persistence_fill(hs, st["vcut"]),
                 "persistence_fill+T": np.repeat((nowr <= 1)[:, None, :], K, 1),
                 "kinematic": predict_kinematic(hs, st["vcut"], st["mp"]),
                 "kinematic+T": predict_kinematic(hs, st["vcut"], st["mp"], rT=rT),
                 "kinematic_g0.5+T": predict_kinematic(hs, st["vcut"], st["mp"], gain=0.5, rT=rT)}
        for j, i in enumerate(idx):
            r = dict(panel=p, condition="queue_ongoing", src=W.src[i])
            for n, P in preds.items():
                r[n] = iou(P[j], y[j])
            rows.append(r)
    df = pd.DataFrame(rows)
    names = [c for c in df.columns if c not in ("panel", "condition", "src")]
    for src in ("sim", "off"):
        d = df[df.src == src]
        print(src, {n: round(aggregate(d.rename(columns={n: "iou"}))["queue_ongoing"], 4) for n in names})


if __name__ == "__main__":
    main()
