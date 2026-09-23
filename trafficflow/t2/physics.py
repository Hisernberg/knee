"""Traffic-flow physics features for the Task 2 cell models (data <= T only).

``onset_physics``  per link, from the 60-min history (+ origin slot where
    visible): flow/capacity and density/critical-density levels, 30-min linear
    trends and their extrapolation to T+30, speed margin to v_cut and its
    extrapolation, bottleneck signature (capacity/lane drop to the
    neighbours, downstream-minus-upstream speed ratio), distance to the nearest
    link already below 0.8 / 0.7 free speed, and the (at most one per link)
    queued observations an onset history may contain.

``lwr_features``  per link and step, for queues visible at the origin: queue
    blocks (runs of links with speed <= v_cut) with tail/head mileposts;
    Rankine-Hugoniot tail speed s = (q_q - q_u) / (k_q - k_u) from the arriving
    state upstream of the tail and the queue state behind it; head speed from
    the state downstream of the head; empirical tail speed from matching the
    blocks 20 min earlier; predicted tail/head at T+5k and the signed distance
    of each link to them.

Mileposts increase downstream; speeds are dx/dt in km/h (negative = moving
upstream, i.e. a growing queue tail).
"""
from __future__ import annotations

import numpy as np

from .core import H, K, statics
from .features import _ffill_time, _fill_space, _nearest_dist, _shift


def _slope(x: np.ndarray, n: int = 6) -> np.ndarray:
    """Least-squares slope per slot over the last n slots; x [N, H, L]."""
    y = x[:, -n:]
    t = np.arange(n, dtype=np.float32) - (n - 1) / 2
    return np.einsum("t,ntl->nl", t, y - y.mean(1, keepdims=True)) / (t ** 2).sum()


def onset_physics(panel: str, hs, hf, he, r_now: np.ndarray | None = None, f_now: np.ndarray | None = None) -> dict:
    st = statics(panel)
    L = st["L"]; n = hs.shape[0]
    vcut = st["vcut"].astype(np.float32); vf = st["vf"].astype(np.float32)
    cap = st["cap"].astype(np.float32); kc = st["kc"].astype(np.float32); kjam = st["kjam"].astype(np.float32)
    lanes = st["lanes"].astype(np.float32); mp = st["mp"]
    r = hs / vcut
    rfs = _fill_space(_ffill_time(r)[0])
    ffs = _fill_space(_ffill_time(hf)[0])
    fc = ffs / cap
    k = ffs / np.maximum(rfs * vcut, 3.0)          # veh/km, total over lanes
    kr = k / kc
    F = {}
    last_r = rfs[:, -1] if r_now is None else np.where(np.isnan(r_now), rfs[:, -1], r_now)
    last_f = fc[:, -1] if f_now is None else np.where(np.isnan(f_now), fc[:, -1], f_now)
    F["ph_fcap_slope"] = _slope(fc)
    F["ph_fcap_ext"] = last_f + F["ph_fcap_slope"] * 7
    fc3 = (fc[:, 2:] + fc[:, 1:-1] + fc[:, :-2]) / 3
    F["ph_fcap_max3"] = fc3.max(1)
    F["ph_k_ratio"] = kr[:, -1]
    F["ph_k_slope"] = _slope(kr)
    F["ph_k_ext"] = kr[:, -1] + F["ph_k_slope"] * 7
    F["ph_k_jam"] = k[:, -1] / kjam
    F["ph_r_slope"] = _slope(rfs)
    F["ph_r_ext"] = last_r + F["ph_r_slope"] * 7
    F["ph_r_min3"] = rfs[:, -3:].min(1)
    F["ph_r_margin_min"] = (rfs.min(1) - 1.0)
    # bottleneck signature (static + speed differences)
    capn = np.r_[cap[1:], cap[-1]]; capp = np.r_[cap[0], cap[:-1]]
    F["ph_cap_dn1"] = np.repeat((capn / cap)[None], n, 0)
    F["ph_cap_up1"] = np.repeat((capp / cap)[None], n, 0)
    lanen = np.r_[lanes[1:], lanes[-1]]; lanep = np.r_[lanes[0], lanes[:-1]]
    F["ph_lanes_dn"] = np.repeat((lanen - lanes)[None], n, 0)
    F["ph_lanes_up"] = np.repeat((lanes - lanep)[None], n, 0)
    capmin = cap.copy()
    for o in range(1, 12):
        idx = np.clip(np.arange(L) + o, 0, L - 1)
        ok = (mp[idx] - mp) <= 1.5
        capmin = np.where(ok, np.minimum(capmin, cap[idx]), capmin)
    F["ph_capmin_dn"] = np.repeat((capmin / cap)[None], n, 0)
    F["ph_dr_dn"] = _shift(last_r, 1) - last_r
    F["ph_dr_up"] = last_r - _shift(last_r, -1)
    F["ph_dr_dnup3"] = _shift(rfs[:, -3:].mean(1), 2) - _shift(rfs[:, -3:].mean(1), -2)
    # proximity to already-slow links
    for tag, thr in (("08", 0.8), ("07", 0.7)):
        slow = last_r * vcut <= thr * vf
        F[f"ph_d{tag}_dn"] = _nearest_dist(slow, mp, "dn")
        F[f"ph_d{tag}_up"] = _nearest_dist(slow, mp, "up")
    # queued observations inside the (onset) history
    qh = (np.nan_to_num(r, nan=9.0) <= 1.0) & he
    anyq = qh.any(1)
    last_idx = np.where(qh, np.arange(H)[None, :, None], -1).max(1)
    F["ph_qobs"] = anyq.astype(np.float32)
    F["ph_qobs_age"] = np.where(last_idx >= 0, H - 1 - last_idx, H).astype(np.float32)
    F["ph_dq_dn"] = _nearest_dist(anyq, mp, "dn")
    F["ph_dq_up"] = _nearest_dist(anyq, mp, "up")
    F["ph_c_qobs"] = np.repeat(qh.sum((1, 2))[:, None].astype(np.float32), L, 1)
    # arriving demand vs local capacity
    fup = (_shift(ffs[:, -1], -1) + _shift(ffs[:, -1], -2) + _shift(ffs[:, -1], -3)) / 3
    F["ph_fup_cap"] = fup / cap
    fup_s = (_shift(ffs, -1) + _shift(ffs, -2) + _shift(ffs, -3)) / 3
    F["ph_fup_slope"] = _slope(fup_s / cap)
    return {k_: np.asarray(v, np.float32) for k_, v in F.items()}


# ----------------------------------------------------------------------------
def _runs(q: np.ndarray):
    idx = np.flatnonzero(q)
    if not len(idx):
        return []
    cut = np.flatnonzero(np.diff(idx) > 1)
    return list(zip(np.r_[idx[0], idx[cut + 1]], np.r_[idx[cut], idx[-1]]))


def lwr_features(panel: str, hs, hf, r_now: np.ndarray, f_now: np.ndarray, lag: int = 4) -> tuple[dict, dict]:
    """Shared [n, L] and per-step [n, K, L] shockwave features.

    r_now / f_now: speed ratio and flow at the origin slot T, already filled
    with the last history value where T is not visible (f_now in vph)."""
    st = statics(panel)
    L = st["L"]; n = hs.shape[0]
    vcut = st["vcut"].astype(np.float64); cap = st["cap"]; kc = st["kc"]
    mp = st["mp"]; up = mp - st["length"]; xm = (mp + up) / 2
    rfs = _fill_space(_ffill_time(hs / st["vcut"])[0]).astype(np.float64)
    names = ["lw_s_tail", "lw_s_emp", "lw_s_head", "lw_qu_cap", "lw_qq_cap", "lw_ku_kc", "lw_kq_kc",
             "lw_xrel_tail", "lw_xrel_head", "lw_blen", "lw_inblk"]
    F = {k: np.full((n, L), np.nan, np.float32) for k in names}
    S = {k: np.full((n, K, L), np.nan, np.float32) for k in ("lw_tail_k", "lw_tail_emp_k", "lw_head_k")}
    dts = np.arange(1, K + 1) * 5 / 60.0            # hours after T
    dlag = lag * 5 / 60.0
    for i in range(n):
        rn = r_now[i].astype(np.float64); fn = f_now[i].astype(np.float64)
        v = rn * vcut
        blocks = _runs(rn <= 1.0)
        if not blocks:
            continue
        prev = _runs(rfs[i, -lag] <= 1.0)
        # per block physics
        info = []
        for a, b in blocks:
            ua = list(range(max(0, a - 3), a))
            qa = list(range(a, min(b, a + 2) + 1))
            qb = list(range(max(a, b - 2), b + 1))
            db = list(range(b + 1, min(L, b + 4)))
            def state(ix):
                if not ix:
                    return np.nan, np.nan
                q = np.nanmean(fn[ix]); vv = np.nanmean(v[ix])
                return q, q / max(vv, 3.0)
            qu, ku = state(ua); qq, kq = state(qa); qh, kh = state(qb); qd, kd = state(db)
            s_tail = (qq - qu) / (kq - ku) if np.isfinite(kq - ku) and (kq - ku) > 5 else np.nan
            s_head = (qd - qh) / (kd - kh) if np.isfinite(kd - kh) and abs(kd - kh) > 5 else np.nan
            s_tail = float(np.clip(s_tail, -40, 40)) if np.isfinite(s_tail) else np.nan
            s_head = float(np.clip(s_head, -40, 40)) if np.isfinite(s_head) else np.nan
            # empirical tail speed from the best-overlapping block `lag` slots earlier
            ov, best = 0, None
            for c, d in prev:
                o = min(b, d) - max(a, c) + 1
                if o > ov:
                    ov, best = o, (c, d)
            s_emp = (up[a] - up[best[0]]) / dlag if best is not None else 0.0
            s_emp = float(np.clip(s_emp, -40, 40))
            info.append((a, b, s_tail, s_emp, s_head,
                         qu / cap[a] if np.isfinite(qu) else np.nan, qq / cap[a] if np.isfinite(qq) else np.nan,
                         ku / kc[a] if np.isfinite(ku) else np.nan, kq / kc[a] if np.isfinite(kq) else np.nan))
        heads = np.array([b for _, b, *_ in info]); tails = np.array([a for a, *_ in info])
        for l in range(L):
            # tail features: containing block, else nearest block downstream
            j = np.flatnonzero(heads >= l)
            if len(j):
                j = j[0]
                a, b, s_tail, s_emp, s_head, qu_c, qq_c, ku_c, kq_c = info[j]
                if up[a] - xm[l] <= 12:
                    x_t = up[a]
                    F["lw_s_tail"][i, l] = s_tail; F["lw_s_emp"][i, l] = s_emp
                    F["lw_qu_cap"][i, l] = qu_c; F["lw_qq_cap"][i, l] = qq_c
                    F["lw_ku_kc"][i, l] = ku_c; F["lw_kq_kc"][i, l] = kq_c
                    F["lw_xrel_tail"][i, l] = xm[l] - x_t
                    F["lw_blen"][i, l] = mp[b] - up[a]
                    F["lw_inblk"][i, l] = float(a <= l <= b)
                    st_ = s_tail if np.isfinite(s_tail) else s_emp
                    S["lw_tail_k"][i, :, l] = xm[l] - (x_t + st_ * dts)
                    S["lw_tail_emp_k"][i, :, l] = xm[l] - (x_t + s_emp * dts)
            # head features: containing block, else nearest block upstream
            j = np.flatnonzero(tails <= l)
            if len(j):
                j = j[-1]
                a, b, s_tail, s_emp, s_head = info[j][:5]
                if xm[l] - mp[b] <= 5:
                    x_h = mp[b]
                    F["lw_s_head"][i, l] = s_head
                    F["lw_xrel_head"][i, l] = x_h - xm[l]
                    sh = s_head if np.isfinite(s_head) else 0.0
                    S["lw_head_k"][i, :, l] = (x_h + sh * dts) - xm[l]
    return F, S
