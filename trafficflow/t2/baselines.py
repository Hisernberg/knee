"""Rule-based Task 2 predictors evaluated on the simulated train windows.

* ``persistence``        the official baseline: eligible observation at T-1.
* ``persistence_fill``   last non-null observation per link inside the history.
* ``onset_range_prior``  onset only: the single contiguous link range at step
                         T+30 that maximises mean IoU over training onset
                         windows of the same panel within +-90 min time of day
                         (other folds only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import SLOTS
from .core import K, PANELS8, aggregate, iou, statics
from .dataset import load_ds

NFOLD = 4


def fold_of(T):
    return (np.asarray(T) // SLOTS // 7) % NFOLD


def last_valid(hs: np.ndarray) -> np.ndarray:
    """[n, H, L] -> [n, L] last non-NaN value along H (NaN if none)."""
    ok = ~np.isnan(hs)
    idx = np.where(ok, np.arange(hs.shape[1])[None, :, None], -1).max(1)
    out = np.take_along_axis(hs, np.maximum(idx, 0)[:, None, :], 1)[:, 0]
    out[idx < 0] = np.nan
    return out


def pred_persistence(hs, he, vcut):
    q = (hs[:, -1] <= vcut) & he[:, -1]
    return np.repeat(q[:, None, :], K, 1)


def pred_persistence_fill(hs, vcut):
    q = last_valid(hs) <= vcut
    return np.repeat(q[:, None, :], K, 1)


def best_range(Y: np.ndarray, w: np.ndarray | None = None) -> tuple[int, int, float]:
    """Contiguous [a, b] maximising the (weighted) mean IoU against rows of Y."""
    n, L = Y.shape
    w = np.ones(n) if w is None else w
    C = np.hstack([np.zeros((n, 1)), np.cumsum(Y, 1)])  # n x L+1
    ny = Y.sum(1)
    best = (0, 0, -1.0)
    for a in range(L):
        b = np.arange(a, L)
        inter = C[:, b + 1] - C[:, [a]]            # n x len(b)
        union = (b - a + 1)[None, :] + ny[:, None] - inter
        m = (w[:, None] * inter / union).sum(0) / w.sum()
        j = int(np.argmax(m))
        if m[j] > best[2]:
            best = (a, int(b[j]), float(m[j]))
    return best


def onset_range_prior(W: pd.DataFrame, A: dict, eval_idx: np.ndarray, tod_win: int = 18) -> np.ndarray:
    """Predictions [len(eval_idx), K, L] for onset windows (step 6 only)."""
    L = A["y"].shape[2]
    tr = W[(W.src == "cand") & (W.condition == "queue_onset")]
    tod = (W["T"].to_numpy() % SLOTS)
    fold = fold_of(W["T"].to_numpy())
    out = np.zeros((len(eval_idx), K, L), bool)
    for j, i in enumerate(eval_idx):
        nb = tr[(fold[tr.index] != fold[i]) & (np.abs(tod[tr.index] - tod[i]) <= tod_win)]
        if len(nb) < 3:
            nb = tr[fold[tr.index] != fold[i]]
        Y = A["y"][nb.index.to_numpy(), K - 1]
        a, b, _ = best_range(Y)
        out[j, K - 1, a:b + 1] = True
    return out


def main():
    rows = []
    for p in PANELS8:
        W, A = load_ds(p)
        vcut = statics(p)["vcut"]
        ev = W[W.src.isin(["sim", "off"])]
        idx = ev.index.to_numpy()
        preds = {"persistence": pred_persistence(A["hs"][idx], A["he"][idx], vcut),
                 "persistence_fill": pred_persistence_fill(A["hs"][idx], vcut)}
        on = ev.condition.to_numpy() == "queue_onset"
        pr = np.zeros_like(preds["persistence"])
        pr[on] = onset_range_prior(W, A, idx[on])
        pr[~on] = preds["persistence_fill"][~on]
        preds["range_prior+pers_fill"] = pr
        for name, P in preds.items():
            for j, i in enumerate(idx):
                rows.append(dict(panel=p, src=W.src[i], condition=W.condition[i], method=name,
                                 iou=iou(P[j], A["y"][i])))
        print(p, flush=True)
    df = pd.DataFrame(rows)
    for (src, m), g in df.groupby(["src", "method"]):
        print(src, m, {k: round(v, 4) for k, v in aggregate(g).items()})
    return df



# ----------------------------------------------------------------------------
# test-time predictions for the official validation / private windows
def predict_rules(split: str) -> dict:
    """{method: {window_id: bool[K, L]}} for the rule-based methods."""
    from .core import official_history, official_windows
    out = {"persistence": {}, "persistence_fill": {}, "range_prior+pers_fill": {}}
    for p in PANELS8:
        W, A = load_ds(p)
        vcut = statics(p)["vcut"]
        tr = W[(W.src == "cand") & (W.condition == "queue_onset")]
        trtod = W.loc[tr.index, "T"].to_numpy() % SLOTS
        wi = official_windows(p, split)
        hist = official_history(p, split)
        for r in wi.itertuples():
            h = hist[r.window_id]
            hs = h["speed"][None]; he = h["elig"][None]
            out["persistence"][r.window_id] = pred_persistence(hs, he, vcut)[0]
            pf = pred_persistence_fill(hs, vcut)[0]
            out["persistence_fill"][r.window_id] = pf
            if r.condition == "queue_onset":
                nb = tr.index[np.abs(trtod - r.T % SLOTS) <= 18].to_numpy()
                if len(nb) < 3:
                    nb = tr.index.to_numpy()
                a, b, _ = best_range(A["y"][nb, K - 1])
                pr = np.zeros_like(pf); pr[K - 1, a:b + 1] = True
            else:
                pr = pf
            out["range_prior+pers_fill"][r.window_id] = pr
    return out


if __name__ == "__main__":
    main()
