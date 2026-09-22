"""Metric helpers: masked per-target ROC-AUC, macro average, rank-based ensembling."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from .schema import TARGETS


def per_target_auc(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """AUC per target, skipping NaN labels; NaN if a target lacks both classes."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    out: dict[str, float] = {}
    for j, name in enumerate(TARGETS[: y_true.shape[1]]):
        m = np.isfinite(y_true[:, j])
        yt = y_true[m, j]
        if m.sum() == 0 or len(np.unique(yt > 0.5)) < 2:
            out[name] = float("nan")
            continue
        out[name] = float(roc_auc_score((yt > 0.5).astype(int), y_pred[m, j]))
    return out


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray, strict: bool = False) -> float:
    """Competition metric: mean of per-target AUCs. With strict=True any undefined target yields NaN
    (the competition never silently drops a target); with strict=False undefined targets are skipped."""
    aucs = per_target_auc(y_true, y_pred)
    vals = np.array(list(aucs.values()), dtype=np.float64)
    if strict and not np.isfinite(vals).all():
        return float("nan")
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else float("nan")


def to_rank(x: np.ndarray) -> np.ndarray:
    """Column-wise average-rank transform to (0,1]. Permutation invariant, tie-safe (unlike argsort-argsort)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return rankdata(x, method="average") / len(x)
    return np.stack([rankdata(x[:, j], method="average") / x.shape[0] for j in range(x.shape[1])], axis=1)


def rank_average(preds: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Weighted mean of per-column average ranks; robust to differently calibrated arms."""
    if not preds:
        raise ValueError("no predictions to average")
    w = np.ones(len(preds)) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return sum(wi * to_rank(p) for wi, p in zip(w, preds, strict=True))


def prob_average(preds: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    w = np.ones(len(preds)) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return sum(wi * np.asarray(p, dtype=np.float64) for wi, p in zip(w, preds, strict=True))
