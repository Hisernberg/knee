"""Task 1: masked state reconstruction with spatio-temporal gradient boosting.

Links are put in milepost order so that j-1/j+1 are physical neighbours. Every
feature is read from the masked view (what the release publishes), so a model
trained on train targets sees exactly the evidence it will see on validation /
private.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from .data import REL, SLOTS, T0, load

CH = ("speed", "flow", "occ")


def mileposts(panel: str, links) -> np.ndarray:
    f = sorted(glob.glob(str(REL / "corridors" / panel / "train" / "mainline_states_masked" / "**" / "*.parquet"),
                         recursive=True))[0]
    mp = pd.read_parquet(f, columns=["link_id", "milepost"]).drop_duplicates("link_id").set_index("link_id").milepost
    return mp.reindex(list(links)).to_numpy(np.float64)


def ffill_idx(ok: np.ndarray) -> np.ndarray:
    """For each row, index of last True at or before it along axis 0 (-1 if none)."""
    T, L = ok.shape
    idx = np.where(ok, np.arange(T)[:, None], -1)
    return np.maximum.accumulate(idx, axis=0)


def bfill_idx(ok: np.ndarray) -> np.ndarray:
    T, L = ok.shape
    idx = np.where(ok, np.arange(T)[:, None], T)
    return np.minimum.accumulate(idx[::-1], axis=0)[::-1]


class Panel:
    def __init__(self, panel: str):
        d = load(panel)
        self.panel = panel
        mp = mileposts(panel, d["links"])
        o = np.argsort(mp)
        self.order = o
        self.links = np.asarray(d["links"])[o]
        self.mp = mp[o]
        fd = d["net"]["fd"].iloc[o]
        self.lanes = fd.lanes.to_numpy(np.float32)
        self.length = fd.length_km.to_numpy(np.float32)
        self.vf = fd.free_speed_kmh.to_numpy(np.float32)
        self.cap = fd.capacity_vph.to_numpy(np.float32)
        self.T, self.L = d["speed"].shape
        # observed channels (masked view). flow & occ per lane.
        self.X = {
            "speed": d["speed"][:, o],
            "flow": d["flow"][:, o] / self.lanes[None, :],
            "occ": d["occ"][:, o],
        }
        self.pct = d["pct"][:, o]
        self.elig = d["elig"][:, o]
        self.target = d["target"][:, o]
        self.regime_day = d["regime"]
        self.tspeed = d["tspeed"][:, o]
        self.tflow = d["tflow"][:, o] / self.lanes[None, :]
        self.raw = d  # keep for evaluators (original link order)
        t = np.arange(self.T)
        self.tod = (t % SLOTS).astype(np.int16)
        self.wd = ((t // SLOTS + T0.weekday()) % 7).astype(np.int8)
        self._hist()
        self._interp()

    def _hist(self):
        key = self.wd.astype(np.int32) * SLOTS + self.tod
        self.H = {}
        for c in CH:
            X = self.X[c]
            ok = np.isfinite(X)
            s = np.zeros((7 * SLOTS, self.L)); n = np.zeros((7 * SLOTS, self.L))
            np.add.at(s, key, np.where(ok, X, 0)); np.add.at(n, key, ok)
            prof = (s / np.maximum(n, 1)).astype(np.float32)
            # light smoothing over tod
            prof = prof.reshape(7, SLOTS, self.L)
            prof = (np.roll(prof, 1, 1) + 2 * prof + np.roll(prof, -1, 1)) / 4
            self.H[c] = prof.reshape(7 * SLOTS, self.L)
        self.key = key

    def _interp(self):
        ok = np.isfinite(self.X["speed"]) & np.isfinite(self.X["flow"])
        self.ok = ok
        self.fi = ffill_idx(ok)
        self.bi = bfill_idx(ok)
        # corridor-wide blackout rows (Task 2 horizon + buffer): no value anywhere
        dark = ok.sum(1) == 0
        self.dark = dark
        pos = np.zeros(self.T, np.int16); run = 0
        for t in range(self.T):
            run = run + 1 if dark[t] else 0
            pos[t] = run
        self.dark_pos = pos
        rem = np.zeros(self.T, np.int16); run = 0
        for t in range(self.T - 1, -1, -1):
            run = run + 1 if dark[t] else 0
            rem[t] = run
        self.dark_rem = rem

    # ----------------------------------------------------------- blackouts
    def select_origins(self, day_lo: int, day_hi: int, spacing: int = 72):
        """Replicate the Task 2 window selector on unmasked train truth.

        onset: earliest origin whose 60-min history shows no established queue
        and whose 30-min horizon has one; ongoing: established queue in history
        and a queue in the horizon, persistence IoU <= 0.9.
        """
        vcut = 0.6 * self.vf[None, :]
        q = (self.tspeed <= vcut) & np.isfinite(self.tspeed)
        out = []
        last = {"queue_onset": -10**9, "queue_ongoing": -10**9}
        for day in range(day_lo, day_hi):
            for T in range(day * SLOTS + 13, (day + 1) * SLOTS - 19):
                hist = q[T - 12:T + 1]
                hor = q[T + 1:T + 7]
                if not hor.any():
                    continue
                established = (hist.sum(0) >= 2).any()
                cond = "queue_ongoing" if established else "queue_onset"
                if T - last[cond] < spacing:
                    continue
                if cond == "queue_ongoing":
                    pers = np.repeat(q[T][None, :], 6, 0)
                    inter = (pers & hor).sum(); union = (pers | hor).sum()
                    if union and inter / union > 0.9:
                        continue
                out.append((T, cond)); last[cond] = T
        return out

    def apply_blackouts(self, origins):
        """Mimic the val/private release around each origin T: rows T-12..T-1
        fully observed (no Task 1 targets there), rows T+1..T+18 dark."""
        for c in CH:
            X = self.X[c]
            tr = {"speed": self.tspeed, "flow": self.tflow}.get(c)
            for T, _ in origins:
                if tr is not None:
                    h = slice(T - 12, T)
                    m = self.elig[h] & np.isnan(X[h])
                    X[h] = np.where(m, tr[h], X[h])
                X[T + 1:T + 19] = np.nan
        if "occ" in self.X:  # occupancy truth is not cached; history occ stays masked at old targets
            pass
        self._hist_keep = True
        self._interp()

    # ------------------------------------------------------------------ features
    def features(self, ti: np.ndarray, lj: np.ndarray) -> pd.DataFrame:
        T, L = self.T, self.L
        F = {}
        key = self.key[ti]
        for c in CH:
            F[f"h_{c}"] = self.H[c][key, lj]

        def get(c, dt, dl):
            t2 = ti + dt; l2 = lj + dl
            valid = (t2 >= 0) & (t2 < T) & (l2 >= 0) & (l2 < L)
            t2c = np.clip(t2, 0, T - 1); l2c = np.clip(l2, 0, L - 1)
            v = self.X[c][t2c, l2c].astype(np.float32)
            v[~valid] = np.nan
            return v, t2c, l2c, valid

        # own link, temporal neighbours
        for dt in (-6, -4, -3, -2, -1, 1, 2, 3, 4, 6):
            for c in CH:
                v, *_ = get(c, dt, 0)
                F[f"o_{c}_{dt}"] = v
        for dt in (-1, 1):
            _, t2, l2, valid = get("speed", dt, 0)
            F[f"o_pct_{dt}"] = np.where(valid, self.pct[t2, l2], -1).astype(np.float32)
        # prev / next observed (any distance) and linear interpolation
        fi = self.fi[np.clip(ti - 1, 0, T - 1), lj]; fi = np.where(ti - 1 >= 0, fi, -1)
        bi = self.bi[np.clip(ti + 1, 0, T - 1), lj]; bi = np.where(ti + 1 < T, bi, T)
        gp = (ti - fi).astype(np.float32); gn = (bi - ti).astype(np.float32)
        gp[fi < 0] = np.nan; gn[bi >= T] = np.nan
        F["gap_prev"] = gp; F["gap_next"] = gn
        for c in CH:
            pv = np.where(fi >= 0, self.X[c][np.clip(fi, 0, T - 1), lj], np.nan)
            nv = np.where(bi < T, self.X[c][np.clip(bi, 0, T - 1), lj], np.nan)
            F[f"pv_{c}"] = pv; F[f"nv_{c}"] = nv
            w = np.where(np.isfinite(gp) & np.isfinite(gn), gp / (gp + gn), np.nan)
            li = pv + w * (nv - pv)
            li = np.where(np.isfinite(li), li, np.where(np.isfinite(pv), pv, nv))
            F[f"li_{c}"] = li
            # deviation of neighbours from hist, re-applied to own hist
            hp = np.where(fi >= 0, self.H[c][self.key[np.clip(fi, 0, T - 1)], lj], np.nan)
            hn = np.where(bi < T, self.H[c][self.key[np.clip(bi, 0, T - 1)], lj], np.nan)
            F[f"lidev_{c}"] = F[f"h_{c}"] + np.nanmean(np.stack([pv - hp, nv - hn]), 0)
        # spatial neighbours
        for dl in (-3, -2, -1, 1, 2, 3):
            dts = (-1, 0, 1) if abs(dl) <= 2 else (0,)
            for dt in dts:
                for c in CH:
                    v, t2, l2, valid = get(c, dt, dl)
                    F[f"n{dl}_{c}_{dt}"] = v
            for c in ("speed", "flow"):
                v, t2, l2, valid = get(c, 0, dl)
                hv = np.where(valid, self.H[c][self.key[t2], l2], np.nan)
                F[f"n{dl}_{c}_dev"] = v - hv
            F[f"n{dl}_dist"] = np.where((lj + dl >= 0) & (lj + dl < L),
                                        self.mp[np.clip(lj + dl, 0, L - 1)] - self.mp[lj], np.nan).astype(np.float32)
        # nearest available spatial values (skip missing) upstream / downstream at time t
        for c in ("speed", "flow"):
            up = np.full(len(ti), np.nan, np.float32); dn = np.full(len(ti), np.nan, np.float32)
            for dl in (1, 2, 3, 4, 5):
                v, *_ = get(c, 0, -dl); up = np.where(np.isnan(up), v, up)
                v, *_ = get(c, 0, dl); dn = np.where(np.isnan(dn), v, dn)
            F[f"near_up_{c}"] = up; F[f"near_dn_{c}"] = dn
        # static / calendar
        F["tod"] = self.tod[ti].astype(np.float32)
        F["wd"] = self.wd[ti].astype(np.float32)
        F["lanes"] = self.lanes[lj]; F["length"] = self.length[lj]
        F["vf"] = self.vf[lj]; F["cap_lane"] = self.cap[lj] / self.lanes[lj]
        F["mp_frac"] = (self.mp[lj] / max(self.mp.max(), 1e-6)).astype(np.float32)
        F["regime"] = self.regime_day[ti // SLOTS].astype(np.float32)
        F["dark_pos"] = self.dark_pos[ti].astype(np.float32)
        F["dark_rem"] = self.dark_rem[ti].astype(np.float32)
        # neighbour links' last / next observed values (for long gaps)
        for dl in (-2, -1, 1, 2):
            l2 = np.clip(lj + dl, 0, L - 1); inb = (lj + dl >= 0) & (lj + dl < L)
            f2 = self.fi[np.clip(ti - 1, 0, T - 1), l2]; b2 = self.bi[np.clip(ti + 1, 0, T - 1), l2]
            for c in ("speed", "flow"):
                pv = np.where(inb & (f2 >= 0), self.X[c][np.clip(f2, 0, T - 1), l2], np.nan)
                nv = np.where(inb & (b2 < T), self.X[c][np.clip(b2, 0, T - 1), l2], np.nan)
                F[f"n{dl}_pv_{c}"] = pv.astype(np.float32); F[f"n{dl}_nv_{c}"] = nv.astype(np.float32)
        # local observed density around the day
        return pd.DataFrame(F)

    def target_cells(self, day_lo: int, day_hi: int):
        sl = slice(day_lo * SLOTS, day_hi * SLOTS)
        tt, ll = np.nonzero(self.target[sl] > 0)
        return tt + day_lo * SLOTS, ll
