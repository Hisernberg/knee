"""Full-coverage holdout evaluation of S_state and the Task 3 LWR proxy.

Recreates the simulated blackouts of the feature stage (same selector, same
origins), predicts every train target cell on days >= HOLD with a model tag, and
scores both S_state (per regime, as the official scorer) and the LWR proxy for
several (v, q) post-processing variants.
"""
from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np

from .data import PANELS, SLOTS, SPLIT_DAYS
from .evaluate import link_ramp_validity, s_lwr_proxy
from .t1 import Panel
from .t1_holdout import reconcile
from .t1_pipeline import HOLD, WORK, base_of


def predict_cells(P, models, tt, ll):
    X = P.features(tt, ll)
    X["panel_id"] = np.int16(PANELS.index(P.panel))
    out = {}
    dark = P.dark[tt]
    for c in ("speed", "flow", "dens"):
        y = np.empty(len(tt))
        for kind, m in (("reg", ~dark), ("dark", dark)):
            if m.any():
                b = models[f"{kind}_{c}"]
                y[m] = b.predict(X.loc[m, b.feature_name()]) + base_of(X[m], c)
        out[c] = y
    return out


def evaluate(panel: str, tag: str):
    P = Panel(panel)
    orig = P.select_origins(0, SPLIT_DAYS["train"][1], spacing=36)
    # realistic density on the scored holdout days: ~10 windows / month like validation
    ho = [o for o in orig if o[0] >= HOLD * SLOTS]
    keep = ho[::max(1, len(ho) // 10)][:10]
    P.apply_blackouts([o for o in orig if o[0] < HOLD * SLOTS] + keep)
    models = {f"{k}_{c}": lgb.Booster(model_file=str(WORK / "models" / tag / f"{k}_{c}.txt"))
              for k in ("reg", "dark") for c in ("speed", "flow", "dens")}
    ntr = SPLIT_DAYS["train"][1] * SLOTS
    tt, ll = np.nonzero(P.target[HOLD * SLOTS:ntr] > 0); tt = tt + HOLD * SLOTS
    pr = predict_cells(P, models, tt, ll)
    d = P.raw
    on, off = link_ramp_validity(d); rv = (on & off)[:ntr]
    oj = P.order[ll]
    lanes = P.lanes[ll]
    reg_t = P.regime_day[tt // SLOTS]
    dark = P.dark[tt]
    variants = {"raw": (pr["speed"], pr["flow"])}
    for a in (0.0, 0.25, 0.5):
        variants[f"recon_a{a}"] = reconcile(pr["speed"], pr["flow"], pr["dens"], a)
        v, q = variants[f"recon_a{a}"]
        variants[f"recon_a{a}_darkonly"] = (np.where(dark, v, pr["speed"]), np.where(dark, q, pr["flow"]))
    res = {}
    for name, (v, q) in variants.items():
        S = d["tspeed"][:ntr].copy(); Q = d["tflow"][:ntr].copy()
        S[tt, oj] = v; Q[tt, oj] = q * lanes
        lw, per = s_lwr_proxy(d, S, Q, rv=rv, days_range=(HOLD, SPLIT_DAYS["train"][1]), detail=True)
        st = []
        for r in (1, 2, 3):
            m = reg_t == r
            if not m.any():
                continue
            rs = np.sqrt(np.mean((v[m] - P.tspeed[tt[m], ll[m]]) ** 2))
            rq = np.sqrt(np.mean((q[m] - P.tflow[tt[m], ll[m]]) ** 2))
            st.append(0.54 * max(0, 1 - rs / 25) + 0.46 * max(0, 1 - rq / 600))
        s1 = float(np.mean(st))
        res[name] = (s1, lw)
        print(f"{panel} {name:22s} S_state {s1:.5f} LWR {lw:.4f} {[round(float(x), 3) for x in per.values()]} "
              f"J={0.35 * s1 + 0.10 * lw:.5f}", flush=True)
    return res


if __name__ == "__main__":
    tag = sys.argv[1]
    for p in sys.argv[2:]:
        evaluate(p, tag)
