"""Per (window, link, step) features for the Task 2 cell classifiers.

Everything is computed from the 60-minute window history only, plus static
network data and time-of-day profiles learned on train truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import SLOTS
from .core import H, K, statics

BIG = 30.0  # km cap for distances


# ----------------------------------------------------------------------------
# profiles
def profiles(qtrue: np.ndarray, speed_ratio: np.ndarray | None, day_mask: np.ndarray) -> dict:
    """Time-of-day queue probability per link, by day of week and overall.

    qtrue [Ttrain, L] bool; day_mask [ndays] bool selects the days to learn from.
    Returns pq_dow [7, 288, L] float32 (shrunk to the all-day profile) and pq [288, L].
    """
    nd = qtrue.shape[0] // SLOTS
    L = qtrue.shape[1]
    Q = qtrue[: nd * SLOTS].reshape(nd, SLOTS, L).astype(np.float32)
    days = np.where(day_mask[:nd])[0]
    dow = (days + 5) % 7  # 2030-06-01 is a Saturday -> 5 (Mon=0)
    allp = Q[days].mean(0)
    pq_dow = np.zeros((7, SLOTS, L), np.float32)
    alpha = 5.0
    for d in range(7):
        sel = days[dow == d]
        pq_dow[d] = (Q[sel].sum(0) + alpha * allp) / (len(sel) + alpha)
    # light temporal smoothing (+-1 slot)
    def sm(x):
        return (np.roll(x, 1, -2) + 2 * x + np.roll(x, -1, -2)) / 4
    return dict(pq=sm(allp), pq_dow=sm(pq_dow))


# ----------------------------------------------------------------------------
def _ffill_time(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[n, H, L] forward fill along H; returns filled array and age of last obs."""
    n, h, L = x.shape
    ok = ~np.isnan(x)
    idx = np.where(ok, np.arange(h)[None, :, None], -1)
    idx = np.maximum.accumulate(idx, axis=1)
    f = np.take_along_axis(x, np.maximum(idx, 0), 1)
    f[idx < 0] = np.nan
    age = (np.arange(h)[None, :, None] - idx).astype(np.float32)
    age[idx < 0] = np.nan
    return f, age


def _fill_space(x: np.ndarray) -> np.ndarray:
    """[..., L] fill NaN along links by linear interpolation / edge hold."""
    sh = x.shape
    y = x.reshape(-1, sh[-1])
    out = pd.DataFrame(y.T).interpolate(limit_direction="both").to_numpy(np.float32).T
    return out.reshape(sh)


def _shift(a: np.ndarray, o: int) -> np.ndarray:
    """value at link l+o (edge-held); a [..., L]."""
    if o == 0:
        return a
    L = a.shape[-1]
    idx = np.clip(np.arange(L) + o, 0, L - 1)
    return a[..., idx]


def _nearest_dist(q: np.ndarray, mp: np.ndarray, direction: str) -> np.ndarray:
    """q [n, L] bool. Distance (km) from each link to the nearest True link
    downstream (index >= l) or upstream (index <= l); BIG if none."""
    n, L = q.shape
    if direction == "dn":
        # running "next True" position scanning from the end
        nxt = np.full((n, L), np.nan)
        cur = np.full(n, np.nan)
        for l in range(L - 1, -1, -1):
            cur = np.where(q[:, l], mp[l], cur)
            nxt[:, l] = cur
        d = nxt - mp[None, :]
    else:
        prv = np.full((n, L), np.nan)
        cur = np.full(n, np.nan)
        for l in range(L):
            cur = np.where(q[:, l], mp[l], cur)
            prv[:, l] = cur
        d = mp[None, :] - prv
    d = np.where(np.isnan(d), BIG, np.minimum(d, BIG))
    return d.astype(np.float32)


def window_link_features(panel: str, hs, hf, he, hp, T, prof: dict) -> tuple[pd.DataFrame, dict]:
    """Features shared by all steps: DataFrame with n*L rows (window-major),
    plus per-step extras returned in a dict of [n, K, L] arrays."""
    st = statics(panel)
    L = st["L"]; n = hs.shape[0]
    vcut = st["vcut"].astype(np.float32); mp = st["mp"]; length = st["length"]
    cap = st["cap"].astype(np.float32); lanes = st["lanes"].astype(np.float32)
    r = hs / vcut
    rf, age = _ffill_time(r)
    ff, _ = _ffill_time(hf)
    # links never observed in the hour: fill spatially from neighbours at each slot
    rfs = _fill_space(rf)
    ffs = _fill_space(ff)
    with np.errstate(all="ignore"):
        r_m3 = np.nanmean(r[:, -3:], 1); r_m12 = np.nanmean(r, 1)
        r_min = np.nanmin(r, 1); r_max = np.nanmax(r, 1)
        f_m3 = np.nanmean(hf[:, -3:], 1)
    nq12 = np.nansum(r <= 1, 1).astype(np.float32)
    nq3 = np.nansum(r[:, -3:] <= 1, 1).astype(np.float32)
    nobs = (~np.isnan(r)).sum(1).astype(np.float32)
    last = rfs[:, -1]
    q_now = last <= 1.0
    q_15 = rfs[:, -4] <= 1.0
    q_30 = rfs[:, -7] <= 1.0
    q_60 = rfs[:, 0] <= 1.0
    F = {}
    F["r_last"] = last
    F["r_last_raw"] = rf[:, -1]
    F["age"] = np.nan_to_num(age[:, -1], nan=H)
    F["r_m3"] = r_m3; F["r_m12"] = r_m12; F["r_min"] = r_min; F["r_max"] = r_max
    F["d_r15"] = last - rfs[:, -4]; F["d_r30"] = last - rfs[:, -7]; F["d_r55"] = last - rfs[:, 0]
    F["nq12"] = nq12; F["nq3"] = nq3; F["nobs"] = nobs
    F["f_last"] = ffs[:, -1] / cap; F["f_m3"] = f_m3 / cap
    F["d_f15"] = (ffs[:, -1] - ffs[:, -4]) / cap
    F["dens"] = (ffs[:, -1] / np.maximum(rfs[:, -1] * vcut, 5.0)) / lanes
    for o in (-6, -4, -3, -2, -1, 1, 2, 3, 4, 6):
        F[f"r_o{o}"] = _shift(last, o)
    for o in (-2, -1, 1, 2):
        F[f"dr15_o{o}"] = _shift(F["d_r15"], o)
        F[f"f_o{o}"] = _shift(F["f_last"], o)
    # windowed spatial minima of the current ratio, downstream / upstream by km
    bands = ((0, 1), (1, 3), (3, 6), (6, 12))
    for side, sgn in (("dn", 1), ("up", -1)):
        ms = {b: np.full((n, L), 9.0, np.float32) for b in bands}
        for o in range(1, L):
            idx = np.arange(L) + sgn * o
            okl = (idx >= 0) & (idx < L)
            if not okl.any():
                break
            idc = np.clip(idx, 0, L - 1)
            dist = np.where(okl, np.abs(mp[idc] - mp), np.inf)
            if np.all(dist > bands[-1][1]):
                break
            v = last[:, idc]
            for lo, hi in bands:
                sel = (dist > lo) & (dist <= hi)
                if sel.any():
                    ms[(lo, hi)] = np.where(sel[None, :], np.minimum(ms[(lo, hi)], v), ms[(lo, hi)])
        for lo, hi in bands:
            F[f"rmin_{side}_{lo}_{hi}"] = ms[(lo, hi)]
    # queue geometry now / 15 / 30 / 60 min ago
    for tag, q in (("now", q_now), ("15", q_15), ("30", q_30), ("60", q_60)):
        F[f"ddn_{tag}"] = _nearest_dist(q, mp, "dn")
        F[f"dup_{tag}"] = _nearest_dist(q, mp, "up")
    F["ddn_rate15"] = F["ddn_15"] - F["ddn_now"]
    F["ddn_rate30"] = (F["ddn_30"] - F["ddn_now"]) / 2
    F["dup_rate15"] = F["dup_15"] - F["dup_now"]
    F["qhist_frac"] = np.nanmean(np.where(np.isnan(r), np.nan, (r <= 1).astype(np.float32)), 1)
    # corridor-level context (broadcast)
    tot_now = q_now.sum(1, keepdims=True).astype(np.float32)
    tot_15 = q_15.sum(1, keepdims=True).astype(np.float32)
    tot_60 = q_60.sum(1, keepdims=True).astype(np.float32)
    F["c_q_now"] = np.repeat(tot_now / L, L, 1)
    F["c_q_trend"] = np.repeat((tot_now - tot_15) / L, L, 1)
    F["c_q_trend60"] = np.repeat((tot_now - tot_60) / L, L, 1)
    F["c_rmin"] = np.repeat(np.nanmin(last, 1, keepdims=True), L, 1)
    # static
    F["vf"] = np.repeat(st["vf"][None].astype(np.float32), n, 0)
    F["lanes"] = np.repeat(lanes[None], n, 0)
    F["cap_lane"] = np.repeat((cap / lanes)[None], n, 0)
    F["length"] = np.repeat(length[None].astype(np.float32), n, 0)
    F["relpos"] = np.repeat((np.arange(L) / L)[None].astype(np.float32), n, 0)
    F["dist_end"] = np.repeat((mp[-1] - mp)[None].astype(np.float32), n, 0)
    on_c = np.convolve(st["on"], np.ones(5), "same"); off_c = np.convolve(st["off"], np.ones(5), "same")
    F["ramps_on"] = np.repeat(on_c[None].astype(np.float32), n, 0)
    F["ramps_off"] = np.repeat(off_c[None].astype(np.float32), n, 0)
    # time
    tod = (np.asarray(T) % SLOTS).astype(np.int64)
    dow = ((np.asarray(T) // SLOTS + 5) % 7).astype(np.int64)
    F["tod"] = np.repeat((tod / 12.0)[:, None].astype(np.float32), L, 1)
    F["dow"] = np.repeat(dow[:, None].astype(np.float32), L, 1)
    # profiles at the origin and per step
    pq_dow = prof["pq_dow"]
    F["pq_T"] = pq_dow[dow, tod]
    F["pq_Tm30"] = pq_dow[dow, (tod - 6) % SLOTS]
    F["pq_up2"] = _shift(F["pq_T"], -2); F["pq_dn2"] = _shift(F["pq_T"], 2)
    step = {}
    step["pq_k"] = np.stack([pq_dow[dow, (tod + k) % SLOTS] for k in range(1, K + 1)], 1)
    step["pq_all_k"] = np.stack([prof["pq"][(tod + k) % SLOTS] for k in range(1, K + 1)], 1)
    df = pd.DataFrame({k: np.asarray(v, np.float32).reshape(-1) for k, v in F.items()})
    return df, step


def expand_steps(df: pd.DataFrame, step: dict, steps=range(1, K + 1)) -> pd.DataFrame:
    """Stack the shared features for each requested step (step-major)."""
    parts = []
    for k in steps:
        d = df.copy()
        d["k"] = np.float32(k)
        for name, arr in step.items():
            d[name] = arr[:, k - 1].reshape(-1)
        d["ddn_proj"] = d["ddn_now"] - d["ddn_rate15"] * k / 3.0
        parts.append(d)
    return pd.concat(parts, ignore_index=True)
