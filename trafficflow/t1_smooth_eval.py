"""Holdout evaluation of the dN-aware smoothing (trafficflow.t1_smooth).

python -m trafficflow.t1_smooth_eval cache <panels>   hold3 predictions on the full-coverage holdout
                                                        (realistic blackouts, same protocol as the hold3
                                                        J check) -> WORK/smooth/hold3_<panel>.npz
python -m trafficflow.t1_smooth_eval share <panels>   LWR error share by cell type (baseline) and the
                                                        change with the adopted smoothing
python -m trafficflow.t1_smooth_eval grid <panels>    J / S_state / LWR per panel: smoothing variants and
                                                        the gate x recon-split grid (with / without smoothing)
Panels default to the four holdout panels; results go to WORK/smooth/*.csv.

Scoring is exact on the holdout window (verified against evaluate.s_lwr_proxy): only transitions
that touch a predicted cell carry error, so the numerator is summed over those only.
"""
from __future__ import annotations

import gc
import sys
import time

import numpy as np

from .data import PANELS, SLOTS
from .t1_holdout import reconcile
from .t1_pipeline import HOLD, WORK
from .t1_smooth import DEFAULT, apply_k, runs, smooth_cells, smooth_runs

NTR_DAY = 273
HOLD_PANELS = ["D12_I5_S", "D7_I10_W", "D7_I405_S", "D12_I405_N"]


def cache(panel: str, out=None):
    import lightgbm as lgb
    from .evaluate import link_ramp_validity
    from .t1 import Panel
    from .t1_pipeline import add_fd, base_of
    out = WORK / "smooth" if out is None else out
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"hold3_{panel}.npz"
    if f.exists():
        return
    t0 = time.time()
    M = {f"{k}_{c}": lgb.Booster(model_file=str(WORK / "models" / "hold3" / f"{k}_{c}.txt"))
         for k in ("reg", "dark") for c in ("speed", "flow", "dens")}
    P = Panel(panel)
    orig = P.select_origins(0, NTR_DAY, spacing=36); ho = [o for o in orig if o[0] >= HOLD * SLOTS]
    keep = ho[::max(1, len(ho) // 10)][:10]
    P.apply_blackouts([o for o in orig if o[0] < HOLD * SLOTS] + keep)
    ntr = NTR_DAY * SLOTS
    tt, ll = np.nonzero(P.target[HOLD * SLOTS:ntr] > 0); tt = tt + HOLD * SLOTS
    X = P.features(tt, ll); X["panel_id"] = np.int16(PANELS.index(panel)); X["panel"] = panel; X["j"] = ll
    X = add_fd(X)
    dark = P.dark[tt]; pr = {}
    for c in ("speed", "flow", "dens"):
        y = np.full(len(tt), np.nan)
        for kind, m in (("reg", ~dark), ("dark", dark)):
            if m.any():
                b = M[f"{kind}_{c}"]
                y[m] = b.predict(X.loc[m, b.feature_name()], num_threads=2) + base_of(X[m], c)
        pr[c] = y
    del X
    on, off = link_ramp_validity(P.raw); rv = (on & off)[:, P.order]
    a0 = (HOLD - 1) * SLOTS
    w = slice(a0, ntr)
    np.savez_compressed(
        f, tt=tt, ll=ll, speed=pr["speed"], flow=pr["flow"], dens=pr["dens"], a0=a0,
        keep=np.array([o[0] for o in keep]),
        xs=P.X["speed"][w], xq=P.X["flow"][w], dark=P.dark[w], ts=P.tspeed[w], tq=P.tflow[w],
        elig=P.elig[w], target=P.target[w], rv=rv[w], regime_day=P.regime_day,
        lanes=P.lanes, length=P.length, vf=P.vf, w=P.w, kj=P.kj_l, order=P.order)
    print(panel, f"{len(tt)} cells, dark {dark.sum()}, {time.time() - t0:.0f}s", flush=True)


class Hold:
    """Cached holdout panel: predictions at target cells + everything the scorers need."""

    def __init__(self, panel: str):
        z = np.load(WORK / "smooth" / f"hold3_{panel}.npz")
        self.panel = panel
        a0 = int(z["a0"])
        self.r, self.c = z["tt"] - a0, z["ll"]
        self.pr = {k: z[k] for k in ("speed", "flow", "dens")}
        self.lanes = z["lanes"].astype(np.float64); self.length = z["length"].astype(np.float64)
        self.vf = z["vf"].astype(np.float64)
        ts = z["ts"].astype(np.float64); tq = z["tq"].astype(np.float64)
        self.W, self.L = ts.shape
        self.ys, self.yq = ts[self.r, self.c], tq[self.r, self.c]
        self.Nt = tq * self.lanes / np.maximum(ts, 1) * self.length
        rowday = (a0 + np.arange(self.W)) // SLOTS
        reg = z["regime_day"][rowday]
        self.creg = reg[self.r]
        el, rv = z["elig"], z["rv"]
        S = np.zeros((self.W, self.L), bool); S[self.r, self.c] = True
        self.S = S
        self.dark_row = z["dark"]
        self.tr = {}
        for g in (1, 2, 3):
            rows = (reg == g) & (rowday >= HOLD)
            if not rows.any():
                continue
            ok = el[:-1] & el[1:] & rows[:-1, None] & rows[1:, None] & rv[:-1]
            den = np.abs(self.Nt[1:] - self.Nt[:-1])[ok].sum()
            ti, tj = np.nonzero(ok & (S[:-1] | S[1:]))
            self.tr[g] = (ti, tj, den)
        xs, xq = z["xs"].astype(np.float64), z["xq"].astype(np.float64)
        obs = np.isfinite(xs) & np.isfinite(xq) & (z["target"] == 0)
        self.kobs = np.where(obs, xq / np.maximum(xs, 1), np.nan)
        self.cvf = self.vf[self.c]

    # ------------------------------------------------------------------ post-processing
    def base(self, a=0.75, gate=0.6):
        """Adopted post-processing (make_submission --recon-a 0.75 --gate 0.6), before clipping."""
        s, f, k = self.pr["speed"], self.pr["flow"], self.pr["dens"]
        v, q = reconcile(s, f, k, a)
        g = s < gate * self.cvf
        return np.where(g, v, s), np.where(g, q, f), g

    def clip(self, v, q):
        ln = self.lanes[self.c]
        return np.clip(v, 3.0, 130.0), np.maximum(q * ln, 60.0) / ln

    # ------------------------------------------------------------------ scorers
    def lwr(self, v, q, detail=False):
        Ns = self.Nt.copy()
        Ns[self.r, self.c] = q * self.lanes[self.c] / np.maximum(v, 1) * self.length[self.c]
        per = {}
        for g, (ti, tj, den) in self.tr.items():
            e = np.abs((Ns[ti + 1, tj] - Ns[ti, tj]) - (self.Nt[ti + 1, tj] - self.Nt[ti, tj])).sum()
            per[g] = 1 - min(1.0, e / den)
        s = float(np.mean(list(per.values())))
        return (s, per) if detail else s

    def s_state(self, v, q):
        st = []
        for g in (1, 2, 3):
            m = self.creg == g
            if not m.any():
                continue
            rs = np.sqrt(np.mean((v[m] - self.ys[m]) ** 2)); rq = np.sqrt(np.mean((q[m] - self.yq[m]) ** 2))
            st.append(0.54 * max(0, 1 - rs / 25) + 0.46 * max(0, 1 - rq / 600))
        return float(np.mean(st))

    def score(self, v, q):
        v, q = self.clip(v, q)
        s, lw = self.s_state(v, q), self.lwr(v, q)
        return dict(S_state=s, LWR=lw, J=0.35 * s + 0.10 * lw)

    # ------------------------------------------------------------------ smoothing
    def smooth(self, v, q, in_gate, **spec):
        """TV smoothing (t1_smooth.smooth_cells, the make_submission --smooth path)."""
        return smooth_cells(v, q, self.r, self.c, (self.W, self.L), self.kobs, in_gate, self.dark_row[self.r], **spec)

    def smooth_quad(self, v, q, in_gate, lam_in=0.1, lam_a=0.0, a_in=0.75, a_out=0.0):
        """Quadratic smoother (not adopted), for the record."""
        K = np.full((self.W, self.L), np.nan); K[self.r, self.c] = q / np.maximum(v, 1)
        kn = smooth_runs(K, self.S, self.kobs, lam_in, lam_a)[self.r, self.c]
        return apply_k(v, q, kn, in_gate, a_in, a_out)


# ---------------------------------------------------------------------- analyses
def share(panel: str, spec: dict | None = None, quiet: bool = False):
    """Contribution to (1 - LWR) by cell type (blackout rows vs regular rows by run length, boundary vs
    inner transitions, free vs density-gate cells) for the adopted base, or base + smoothing `spec`."""
    H = Hold(panel)
    v, q, g = H.base()
    if spec is not None:
        v, q = H.smooth(v, q, g, **spec)
    v, q = H.clip(v, q)
    Ns = H.Nt.copy()
    Ns[H.r, H.c] = q * H.lanes[H.c] / np.maximum(v, 1) * H.length[H.c]
    RL = np.zeros((H.W, H.L), np.int32)
    sj, sr, n = runs(H.S)
    for j, s, ln in zip(sj, sr, n):
        RL[s:s + ln, j] = ln
    dark = np.repeat(H.dark_row[:, None], H.L, 1)
    cong = np.zeros((H.W, H.L), bool); cong[H.r, H.c] = g
    tot = {}
    ng = len(H.tr)
    for gi, (ti, tj, den) in H.tr.items():
        e = np.abs((Ns[ti + 1, tj] - Ns[ti, tj]) - (H.Nt[ti + 1, tj] - H.Nt[ti, tj])) / den / ng
        a_t, b_t = H.S[ti, tj], H.S[ti + 1, tj]
        dk = dark[ti, tj] | dark[ti + 1, tj]
        rl = np.maximum(RL[ti, tj], RL[ti + 1, tj])
        inner = a_t & b_t
        cg = cong[ti, tj] | cong[ti + 1, tj]
        cat = np.where(dk, "blackout", np.where(rl == 1, "isolated", np.where(rl <= 3, "run2-3", "run4+")))
        pos = np.where(inner, "inner", "boundary")
        for cc in np.unique(cat):
            for pp in ("boundary", "inner"):
                for gg in (False, True):
                    m = (cat == cc) & (pos == pp) & (cg == gg)
                    key = (cc, pp, "gate" if gg else "free")
                    tot[key] = tot.get(key, 0.0) + e[m].sum()
    loss = sum(tot.values())
    if not quiet:
        print(f"{panel}: 1-LWR = {loss:.4f}")
        for k in sorted(tot, key=lambda x: -tot[x]):
            print(f"  {k[0]:9s} {k[1]:8s} {k[2]:5s} {tot[k]:.4f}  share {tot[k] / loss:.3f}")
    return tot, loss


def share_table(panels):
    """Baseline error shares and the loss removed by the adopted smoothing, per category."""
    import pandas as pd
    rows = []
    for p in panels:
        for name, spec in (("base", None), ("TV", dict(DEFAULT))):
            tot, loss = share(p, spec, quiet=True)
            for k, x in tot.items():
                rows.append(dict(panel=p, variant=name, cat=k[0], pos=k[1], traffic=k[2], loss=x))
    d = pd.DataFrame(rows)
    d.to_csv(WORK / "smooth" / "share.csv", index=False)
    b = d[d.variant == "base"]; t = d[d.variant == "TV"]
    tb = b.groupby("panel").loss.sum()
    for by in (["cat"], ["cat", "pos"], ["traffic"]):
        sb = b.groupby(by + ["panel"]).loss.sum().unstack("panel")[panels]
        st = t.groupby(by + ["panel"]).loss.sum().unstack("panel")[panels]
        print(f"--- share of baseline (1 - LWR) by {by}"); print((sb / tb[panels]).round(3).to_string())
        print(f"--- change in (1 - LWR) with the adopted TV by {by} (x 1e4)"); print(((st - sb) * 1e4).round(1).to_string())
    return d


OFF = dict(free=0.0, free_a=0.0, gate=0.0, gate_a=0.0, dark=0.0, dark_a=0.0)
VARIANTS = {  # name -> (post-processing gate, a, smoothing spec or None, quadratic kwargs or None)
    "base (gate 0.6, a 0.75)": (0.6, 0.75, None, None),
    "quad lam_in 0.1": (0.6, 0.75, None, dict(lam_in=0.1)),
    "quad lam_in 0.1, anchors 0.25": (0.6, 0.75, None, dict(lam_in=0.1, lam_a=0.25)),
    "TV free 0.0075": (0.6, 0.75, OFF | dict(free=0.0075), None),
    "TV gate 0.02": (0.6, 0.75, OFF | dict(gate=0.02), None),
    "TV dark 0.05": (0.6, 0.75, OFF | dict(dark=0.05), None),
    "TV free 0.0075 + gate 0.02": (0.6, 0.75, OFF | dict(free=0.0075, gate=0.02), None),
    "TV adopted (DEFAULT)": (0.6, 0.75, dict(DEFAULT), None),
    "TV adopted, a_out 1 (speed)": (0.6, 0.75, DEFAULT | dict(a_out=1.0), None),
    "TV adopted, a_out none (gate only)": (0.6, 0.75, DEFAULT | dict(a_out=None), None),
}
for _g in (0.5, 0.6, 0.7, 0.8):
    for _a in (0.75, 1.0):
        VARIANTS[f"gate {_g}, a {_a}"] = (_g, _a, None, None)
        VARIANTS[f"gate {_g}, a {_a} + TV"] = (_g, _a, dict(DEFAULT), None)


def grid(panels, variants: dict = VARIANTS, out: str | None = "variants.csv"):
    """J / S_state / LWR per panel for every variant; deltas vs the first variant (the adopted base)."""
    import pandas as pd
    rows = []
    for p in panels:
        H = Hold(p)
        for name, (gt, a, spec, quad) in variants.items():
            v, q, g = H.base(a, gt)
            if spec is not None:
                v, q = H.smooth(v, q, g, **spec)
            elif quad is not None:
                v, q = H.smooth_quad(v, q, g, **quad)
            rows.append(dict(panel=p, variant=name, **H.score(v, q)))
        del H; gc.collect()
    d = pd.DataFrame(rows)
    ref = d[d.variant == next(iter(variants))].set_index("panel")
    for c in ("J", "S_state", "LWR"):
        d["d" + c] = d[c] - d.panel.map(ref[c])
    if out:
        d.to_csv(WORK / "smooth" / out, index=False)
    for c, sc in (("J", 5), ("S_state", 5), ("LWR", 4)):
        piv = d.pivot_table(index="variant", columns="panel", values=c, sort=False)[panels]
        dp = d.pivot_table(index="variant", columns="panel", values="d" + c, sort=False)[panels]
        piv["mean d" + c] = dp.mean(axis=1)
        if c == "J":
            piv["n_up"] = (dp > 0).sum(axis=1)
        print(f"--- {c}"); print(piv.round(sc).to_string(), flush=True)
    return d


if __name__ == "__main__":
    stage, panels = sys.argv[1], sys.argv[2:] or HOLD_PANELS
    if stage == "cache":
        for p in panels:
            cache(p); gc.collect()
    elif stage == "share":
        share_table(panels)
    elif stage == "grid":
        grid(panels)
