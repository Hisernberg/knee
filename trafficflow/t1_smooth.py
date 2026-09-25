"""dN-aware temporal smoothing of Task 1 density predictions (Task 3 LWR post-processing).

The LWR proxy charges, for a run of consecutive target cells t = a..b on one link,
    |e_a| + sum_i |e_{i+1} - e_i| + |e_b|,     e_i = N_sub,i - N_true,i,
where the first/last terms come from the transitions to the (known) neighbour cells.
Small predicted increments inside a run are mostly model jitter, and their L1-optimal
value is 0; the boundary increments are already calibrated. The adopted smoother is
therefore 1-D total-variation denoising of the per-lane density k = q/v inside every
run (a soft threshold on the increments, relative to the run's density level), with
optional weak L1 anchors to the observed neighbour densities (`smooth_cells`, spec
`DEFAULT`). `apply_k` moves (v, q) so that q/v equals the smoothed density, with a
speed share a_in inside the density gate (as t1_holdout.reconcile) and a_out outside.

A quadratic (Whittaker) smoother `smooth_runs` is kept for the record; it is worse
(see docs/traffic/T3_SMOOTHING.md).

All arrays are dense [T, L] windows in any fixed link order (one column = one link).
"""
from __future__ import annotations

import numpy as np

# holdout-selected setting (docs/traffic/T3_SMOOTHING.md): relative TV thresholds per cell category
DEFAULT = dict(free=0.0075, free_a=0.001, gate=0.02, gate_a=0.005, dark=0.05, dark_a=0.0,
               a_in=0.75, a_out=0.0, clip=0.5, iters=300)


def runs(tgt: np.ndarray):
    """Maximal runs of True along axis 0, per column: (col, start_row, length)."""
    T, L = tgt.shape
    m = np.zeros((T + 2, L), np.int8)
    m[1:-1] = tgt
    dm = np.diff(m, axis=0).T  # [L, T+1]
    sj, sr = np.nonzero(dm == 1)
    ej, er = np.nonzero(dm == -1)
    assert np.array_equal(sj, ej)
    return sj, sr, er - sr


def tv_rows(Y: np.ndarray, lam: np.ndarray, iters: int = 300, lam_l=None, lam_r=None, x_l=None, x_r=None) -> np.ndarray:
    """Row-wise 1-D total-variation denoising with optional L1 anchors:
        argmin_x 0.5||x - y||^2 + lam sum|x_{i+1} - x_i| + lam_l |x_1 - x_l| + lam_r |x_r - x_n|.

    Y [m, n]; lam, lam_l, lam_r, x_l, x_r [m] (lam_l/lam_r None or x NaN: no anchor).
    Projected gradient on the dual (step 1/4); Z[:, k] is the dual of the k-th difference of
    [x_l, x_1..x_n, x_r], so x = y - Z[:, :-1] + Z[:, 1:]."""
    m, n = Y.shape
    zero = np.zeros(m)
    bl = zero if lam_l is None else np.where(np.isfinite(x_l), lam_l, 0.0)
    br = zero if lam_r is None else np.where(np.isfinite(x_r), lam_r, 0.0)
    if n < 2 and not (bl.any() or br.any()):
        return Y.copy()
    B = np.concatenate([bl[:, None], np.repeat(lam[:, None], n - 1, 1), br[:, None]], 1)
    xl = np.nan_to_num(zero if x_l is None else x_l)[:, None]
    xr = np.nan_to_num(zero if x_r is None else x_r)[:, None]
    Z = np.zeros((m, n + 1))
    for _ in range(iters):
        X = Y - Z[:, :-1] + Z[:, 1:]
        E = np.concatenate([xl, X, xr], 1)
        Z = np.clip(Z + 0.25 * (E[:, 1:] - E[:, :-1]), -B, B)
    return Y - Z[:, :-1] + Z[:, 1:]


def tv_runs(K: np.ndarray, tgt: np.ndarray, tau: float, min_len: int = 2, max_len: int | None = None,
            iters: int = 300, kobs: np.ndarray | None = None, tau_a: float = 0.0,
            only: np.ndarray | None = None) -> np.ndarray:
    """TV smoothing inside each run of target cells, threshold lam = tau * mean(run density).

    With kobs and tau_a > 0 the observed neighbour densities enter as L1 anchors
    (lam_a = tau_a * mean(run density)) and isolated cells are included. `only` [T, L]:
    process only runs containing at least one True cell (speed-up; others unchanged)."""
    out = K.copy()
    T = tgt.shape[0]
    sj, sr, n = runs(tgt)
    anch = kobs is not None and tau_a > 0
    sel = n >= (min_len if anch else max(min_len, 2))
    if max_len is not None:
        sel &= n <= max_len
    if only is not None:
        cs = np.concatenate([np.zeros((1, only.shape[1]), np.int64), np.cumsum(only, 0)])
        sel &= (cs[sr + n, sj] - cs[sr, sj]) > 0
    for ln in np.unique(n[sel]):
        g = sel & (n == ln)
        rows = sr[g][:, None] + np.arange(ln)[None, :]
        cols = sj[g][:, None]
        X = K[rows, cols].astype(np.float64)
        lev = np.abs(X.mean(1))
        if anch:
            s0, e0, j0 = sr[g], sr[g] + ln, sj[g]
            xl = np.where(s0 >= 1, kobs[np.maximum(s0 - 1, 0), j0], np.nan)
            xr = np.where(e0 < T, kobs[np.minimum(e0, T - 1), j0], np.nan)
            out[rows, cols] = tv_rows(X, tau * lev, iters, tau_a * lev, tau_a * lev, xl, xr)
        else:
            out[rows, cols] = tv_rows(X, tau * lev, iters)
    return out


def smooth_runs(K: np.ndarray, tgt: np.ndarray, kobs: np.ndarray, lam_in: float = 1.0, lam_a: float = 0.0,
                min_len: int = 1, max_len: int | None = None) -> np.ndarray:
    """Quadratic (Whittaker) smoothing of the density inside each run of target cells (not adopted).

    For a run k_1..k_n with observed neighbour densities x_0 / x_{n+1} (NaN if not observed),
    minimise sum_i (k'_i - k_i)^2 + lam_in sum_i (k'_{i+1} - k'_i)^2
             + lam_a [(k'_1 - x_0)^2 + (k'_n - x_{n+1})^2]      (missing anchors dropped)."""
    T, L = tgt.shape
    out = K.copy()
    sj, sr, n = runs(tgt)
    er = sr + n
    xl = np.where(sr >= 1, kobs[np.maximum(sr - 1, 0), sj], np.nan)
    xr = np.where(er < T, kobs[np.minimum(er, T - 1), sj], np.nan)
    hl, hr = np.isfinite(xl) & (lam_a > 0), np.isfinite(xr) & (lam_a > 0)
    sel = n >= min_len
    if max_len is not None:
        sel &= n <= max_len
    for ln in np.unique(n[sel]):
        g = sel & (n == ln)
        for bl in (False, True):
            for br in (False, True):
                gg = g & (hl == bl) & (hr == br)
                if not gg.any() or (ln == 1 and not (bl or br)):
                    continue
                A = np.eye(ln)
                if ln > 1 and lam_in > 0:
                    D = np.diff(np.eye(ln), axis=0)
                    A += lam_in * D.T @ D
                A[0, 0] += lam_a * bl
                A[ln - 1, ln - 1] += lam_a * br
                rows = sr[gg][:, None] + np.arange(ln)[None, :]
                cols = sj[gg][:, None]
                R = K[rows, cols].astype(np.float64)
                if bl:
                    R[:, 0] += lam_a * xl[gg]
                if br:
                    R[:, -1] += lam_a * xr[gg]
                out[rows, cols] = R @ np.linalg.inv(A).T
    return out


def apply_k(v: np.ndarray, q: np.ndarray, k_new: np.ndarray, gate: np.ndarray, a_in: float = 0.75,
            a_out: float | None = 0.0, clip: float = 0.5):
    """Move (v, q) so that q/max(v,1) = k_new: v <- v r^-a, q <- q r^(1-a), r = k_new / k_old.

    a_in inside the gate (dense traffic), a_out outside it; a_out=None leaves cells outside
    the gate unchanged. |log r| is clipped to `clip`."""
    k_old = q / np.maximum(v, 1.0)
    lr = np.log(np.maximum(k_new, 1e-3)) - np.log(np.maximum(k_old, 1e-3))
    lr = np.clip(np.nan_to_num(lr), -clip, clip)
    a = np.where(gate, a_in, 0.0 if a_out is None else a_out)
    if a_out is None:
        lr = np.where(gate, lr, 0.0)
    return v * np.exp(-a * lr), q * np.exp((1 - a) * lr)


def parse(spec: str | None) -> dict:
    """'default' -> DEFAULT; 'free=0.0075,gate=0.02,...' -> DEFAULT updated with those keys
    (a_out=none -> None)."""
    out = dict(DEFAULT)
    if spec and spec.lower() != "default":
        for kv in spec.split(","):
            if kv:
                k, v = kv.split("=")
                if k not in DEFAULT:
                    raise KeyError(f"unknown smoothing key {k!r}; valid: {sorted(DEFAULT)}")
                out[k] = None if v.lower() == "none" else (int(v) if k == "iters" else float(v))
    return out


def smooth_cells(v, q, r, c, shape, kobs, in_gate, in_dark, **spec):
    """TV-smooth the density of post-processed (v, q_lane) at target cells (r, c) of a dense window.

    v, q, in_gate, in_dark: 1-D over the target cells; r, c: their row / column in the window
    of `shape` [T, L]; kobs: dense observed per-lane density (NaN where unobserved or target).
    Each cell takes the TV result of its category (blackout row > density gate > free flow)."""
    p = dict(DEFAULT); p.update(spec)
    tgt = np.zeros(shape, bool); tgt[r, c] = True
    K = np.full(shape, np.nan); K[r, c] = q / np.maximum(v, 1.0)
    k0 = K[r, c]
    cat = np.where(in_dark, 2, np.where(in_gate, 1, 0))
    kn = k0.copy()
    for ci, name in enumerate(("free", "gate", "dark")):
        m = cat == ci
        tau, ta = p[name], p[f"{name}_a"]
        if not m.any() or (tau <= 0 and ta <= 0):
            continue
        only = np.zeros(shape, bool); only[r[m], c[m]] = True
        kn[m] = tv_runs(K, tgt, tau, min_len=1, iters=int(p["iters"]), kobs=kobs, tau_a=ta, only=only)[r[m], c[m]]
    return apply_k(v, q, kn, in_gate, p["a_in"], p["a_out"], p["clip"])


def smooth_frame(pr, v, q, in_gate, spec: dict):
    """smooth_cells on a Task 1 prediction frame (make_submission.state_frame, --smooth).

    pr: rows with panel, t (slot index), link_id, kind ('dark' = blackout row); v, q (per lane),
    in_gate: post-processed values / density gate per row. Observed neighbour densities come
    from the panel cache's masked view (the release), excluding Task 1 target cells."""
    from .data import build_panel, network
    v, q = np.array(v, np.float64), np.array(q, np.float64)
    in_gate = np.asarray(in_gate, bool)
    for p, idx in pr.groupby("panel", sort=False).indices.items():
        z = np.load(build_panel(p), allow_pickle=True)
        links = [str(x) for x in z["links"]]
        col = {l: i for i, l in enumerate(links)}
        t = pr["t"].to_numpy()[idx].astype(np.int64)
        c = np.array([col[l] for l in pr["link_id"].to_numpy()[idx]])
        t0, t1 = int(t.min()) - 1, int(t.max()) + 2
        xs = z["speed"][t0:t1].astype(np.float64); xf = z["flow"][t0:t1].astype(np.float64)
        tg = z["target"][t0:t1]
        lanes = network(p)["fd"].lanes.to_numpy(np.float64)
        obs = np.isfinite(xs) & np.isfinite(xf) & (tg == 0)
        kobs = np.where(obs, xf / lanes[None, :] / np.maximum(xs, 1), np.nan)
        dark = pr["kind"].to_numpy()[idx] == "dark"
        v[idx], q[idx] = smooth_cells(v[idx], q[idx], t - t0, c, xs.shape, kobs, in_gate[idx], dark, **spec)
        del z, xs, xf, tg, obs, kobs
    return v, q
