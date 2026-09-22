"""Group-aware, multilabel-stratified K-fold assignment (no external dependency).

Greedy iterative stratification at the group level (Sechidis et al. 2011 style): groups are assigned to
the fold that most needs the rarest label they carry, with fold-size balancing as tie-breaker.
Missing labels (NaN) are ignored in the counts. `groups` defaults to the study id; supply verified
patient ids when available (DICOM PatientID is stripped/unreliable in this competition).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import ID_COL, TARGETS


def stratified_group_kfold(df: pd.DataFrame, n_folds: int = 5, seed: int = 42,
                           group_col: str | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = df[ID_COL].astype(str).to_numpy()
    groups = df[group_col].astype(str).to_numpy() if group_col and group_col in df.columns else ids
    y = df[[t for t in TARGETS if t in df.columns]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    # group-level positive counts (unknown -> 0 positives, but counted separately as "labelled")
    uniq, inv = np.unique(groups, return_inverse=True)
    G = len(uniq)
    pos = np.zeros((G, y.shape[1]))
    size = np.zeros(G)
    for gi in range(G):
        m = inv == gi
        pos[gi] = np.nansum(y[m], axis=0)
        size[gi] = m.sum()
    desired_pos = pos.sum(0) / n_folds
    desired_size = size.sum() / n_folds
    fold_pos = np.zeros((n_folds, y.shape[1]))
    fold_size = np.zeros(n_folds)
    assign = -np.ones(G, dtype=int)
    order = rng.permutation(G)
    remaining = set(order.tolist())
    label_total = pos.sum(0)
    while remaining:
        # rarest label with remaining positives
        rem_idx = np.array(sorted(remaining))
        rem_pos = pos[rem_idx].sum(0)
        cand_labels = np.where(rem_pos > 0)[0]
        if len(cand_labels) == 0:  # only negatives left: balance sizes
            for gi in rem_idx:
                f = int(np.argmin(fold_size))
                assign[gi], fold_size[f] = f, fold_size[f] + size[gi]
                fold_pos[f] += pos[gi]
            break
        lab = cand_labels[np.argmin(label_total[cand_labels] + 1e-9 * rng.random(len(cand_labels)))]
        # groups carrying this label
        carriers = rem_idx[pos[rem_idx, lab] > 0]
        gi = carriers[rng.integers(len(carriers))]
        need = desired_pos[lab] - fold_pos[:, lab]
        best = np.where(need == need.max())[0]
        if len(best) > 1:
            best = best[np.argsort(fold_size[best])[:1]]
        f = int(best[0])
        assign[gi] = f
        fold_pos[f] += pos[gi]
        fold_size[f] += size[gi]
        remaining.discard(int(gi))
        label_total[lab] -= pos[gi, lab]
    out = df[[ID_COL]].copy()
    out["group"] = groups
    out["fold"] = assign[inv]
    return out


def fold_audit(df: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    """Per-fold positive/labelled counts per target, to check every fold has both classes."""
    m = df.merge(folds[[ID_COL, "fold"]], on=ID_COL)
    rows = []
    for f, g in m.groupby("fold"):
        r = {"fold": f, "n": len(g)}
        for t in TARGETS:
            if t in g.columns:
                col = pd.to_numeric(g[t], errors="coerce")
                r[f"{t}:pos"] = int((col > 0.5).sum())
                r[f"{t}:lab"] = int(col.notna().sum())
        rows.append(r)
    return pd.DataFrame(rows)
