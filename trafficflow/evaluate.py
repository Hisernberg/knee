"""Local scorers on the train split.

s_state   -- exact reimplementation of score_task1 (per regime, then mean).
s_lwr_proxy -- Task 3 conservation proxy. The organizer boundary flux is
    projected onto the published observations, so the official right-hand side
    is (to ~5%) the observed dN itself. Hence
        S_LWR ~= 1 - sum|dN_sub - dN_obs| / sum|dN_obs|
    over valid transitions: consecutive eligible cells on regime days, with the
    ramp-validity filter of Mode A applied at the interval start.
"""
from __future__ import annotations

import numpy as np

from .data import SLOTS, SPLIT_DAYS

SPEED_W, FLOW_W = 0.54, 0.46


def split_slice(split: str) -> slice:
    a, b = SPLIT_DAYS[split]
    return slice(a * SLOTS, b * SLOTS)


def regime_rows(d: dict, split: str, r: int, days_range=None) -> np.ndarray:
    a, b = SPLIT_DAYS[split]
    days = np.where(d["regime"][a:b] == r)[0]
    if days_range is not None:
        days = days[(days >= days_range[0]) & (days < days_range[1])]
    m = np.zeros((b - a) * SLOTS, bool)
    for dd in days:
        m[dd * SLOTS:(dd + 1) * SLOTS] = True
    return m


def lanes(d: dict) -> np.ndarray:
    return d["net"]["fd"].lanes.to_numpy(np.float64)


def s_state(d: dict, ps: np.ndarray, pq: np.ndarray, split: str = "train", detail=False):
    sl = split_slice(split)
    tg = d["target"][sl]
    ts, tq = d["tspeed"][sl], d["tflow"][sl]
    ln = lanes(d)[None, :]
    out = {}
    for r in (1, 2, 3):
        m = tg == r
        if not m.any():
            continue
        es = (ps[m] - ts[m]).astype(np.float64)
        eq = ((pq - tq) / ln)[m].astype(np.float64)
        rs, rq = np.sqrt(np.mean(es ** 2)), np.sqrt(np.mean(eq ** 2))
        out[r] = dict(rmse_s=rs, rmse_q=rq, S=SPEED_W * max(0, 1 - rs / 25) + FLOW_W * max(0, 1 - rq / 600))
    S = float(np.mean([v["S"] for v in out.values()]))
    return (S, out) if detail else S


def link_ramp_validity(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """[T, L] on/off ramp validity as the Task 3 evaluator sees it."""
    topo = d["net"]["topo"]; links = d["links"]; lid = {l: i for i, l in enumerate(links)}
    ramps = d["net"]["ramps"]; rv = d["rvalid"]
    T = rv.shape[0]; L = len(links)
    on_ok = np.ones((T, L), bool); off_ok = np.ones((T, L), bool)
    on_att = set(topo.loc[topo.on_ramp_link_ids.fillna("").astype(str).str.len() > 0, "link_id"])
    off_att = set(topo.loc[topo.off_ramp_link_ids.fillna("").astype(str).str.len() > 0, "link_id"])
    anyon = np.zeros((T, L), bool); anyoff = np.zeros((T, L), bool)
    for j, row in ramps.reset_index(drop=True).iterrows():
        li = lid.get(row.nearest_mainline_link_id)
        if li is None:
            continue
        typ = str(row.ramp_type).upper()
        if typ in ("OR", "ON"):
            anyon[:, li] |= rv[:, j]
        elif typ in ("FR", "OFF"):
            anyoff[:, li] |= rv[:, j]
    for l in on_att:
        if l in lid: on_ok[:, lid[l]] = anyon[:, lid[l]]
    for l in off_att:
        if l in lid: off_ok[:, lid[l]] = anyoff[:, lid[l]]
    return on_ok, off_ok


def s_lwr_proxy(d: dict, ps: np.ndarray, pq: np.ndarray, split: str = "train", detail=False, rv=None, days_range=None):
    sl = split_slice(split)
    tg = d["target"][sl] > 0
    el = d["elig"][sl]
    ts, tq = d["tspeed"][sl].astype(np.float64), d["tflow"][sl].astype(np.float64)
    Lk = d["net"]["fd"].length_km.to_numpy(np.float64)[None, :]
    if rv is None:
        on_ok, off_ok = link_ramp_validity(d)
        rv = on_ok[sl] & off_ok[sl]
    Nt = tq / np.maximum(ts, 1) * Lk
    sv = np.where(tg, ps, ts); sq = np.where(tg, pq, tq)
    Ns = sq / np.maximum(sv, 1) * Lk
    out = {}
    for r in (1, 2, 3):
        rows = regime_rows(d, split, r, days_range)
        if not rows.any():
            continue
        ok = el[:-1] & el[1:] & rows[:-1, None] & rows[1:, None] & rv[:-1]
        dNt = (Nt[1:] - Nt[:-1])[ok]; dNs = (Ns[1:] - Ns[:-1])[ok]
        e = np.abs(dNs - dNt).sum() / np.abs(dNt).sum()
        out[r] = 1 - min(1.0, e)
    S = float(np.mean(list(out.values())))
    return (S, out) if detail else S
