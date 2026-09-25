"""Score a pooled Task 1 holdout run (train days >= HOLD).

Regular and dark holdout cells are sampled at different rates, so the dark
share is re-weighted to what validation/private actually contain (DARK_SHARE of
all targets, measured on the release). Reports S_state per panel and regime, the
family-level mean, and error breakdown by traffic state.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from .data import PANELS, FAMILY
from .t1_pipeline import HOLD, WORK, load_train

# measured share of Task 1 targets inside blackout rows (validation+private, per panel)
DARK_SHARE = {"D7_I10_E": 0.0233, "D7_I10_W": 0.0187, "D7_I210_E": 0.0209, "D7_I210_W": 0.0187,
              "D7_I405_N": 0.0241, "D7_I405_S": 0.0207, "D12_I5_N": 0.0179, "D12_I5_S": 0.0165,
              "D12_I405_N": 0.0, "D12_I405_S": 0.0}


def reconcile(speed, flow, dens, a=0.25, clip=0.5):
    """Move (v, q) so that q/v matches the density model; a = share taken by speed."""
    lr = np.log(np.maximum(dens, 1e-3)) - np.log(np.maximum(flow, 1e-3) / np.maximum(speed, 1.0))
    lr = np.clip(lr, -clip, clip)
    return speed * np.exp(-a * lr), flow * np.exp((1 - a) * lr)


def score(tag: str, a=0.25):
    sf = WORK / "models" / tag / "seed.json"  # seed-ensemble member (t1_pipeline TFB_SEED); absent = seed 0
    seed = json.load(open(sf))["seed"] if sf.exists() else 0
    rows = []
    for kind in ("reg", "dark"):
        d = load_train(PANELS, kind, seed=seed)
        d = d[d.day >= HOLD].reset_index(drop=True)
        pr = {c: np.load(WORK / "models" / tag / f"hold_{kind}_{c}.npy") for c in ("speed", "flow", "dens")}
        d["p_speed"], d["p_flow"], d["p_dens"] = pr["speed"], pr["flow"], pr["dens"]
        d["kind"] = kind
        rows.append(d[["panel", "kind", "regime", "t", "j", "y_speed", "y_flow", "p_speed", "p_flow", "p_dens", "vf", "lanes", "length"]])
    d = pd.concat(rows, ignore_index=True)
    d["r_speed"], d["r_flow"] = reconcile(d.p_speed.values, d.p_flow.values, d.p_dens.values, a)
    out = []
    for p, g in d.groupby("panel"):
        ds = DARK_SHARE[p]
        for variant, cs, cq in (("raw", "p_speed", "p_flow"), ("recon", "r_speed", "r_flow")):
            res = {}
            for kind in ("reg", "dark"):
                x = g[g.kind == kind]
                res[kind] = (np.mean((x[cs] - x.y_speed) ** 2), np.mean((x[cq] - x.y_flow) ** 2)) if len(x) else (0, 0)
            ms = (1 - ds) * res["reg"][0] + ds * res["dark"][0]
            mq = (1 - ds) * res["reg"][1] + ds * res["dark"][1]
            s = 0.54 * (1 - np.sqrt(ms) / 25) + 0.46 * (1 - np.sqrt(mq) / 600)
            out.append(dict(panel=p, variant=variant, rmse_s_reg=np.sqrt(res["reg"][0]), rmse_q_reg=np.sqrt(res["reg"][1]),
                            rmse_s_dark=np.sqrt(res["dark"][0]), rmse_q_dark=np.sqrt(res["dark"][1]), S_state=s))
    o = pd.DataFrame(out)
    o["family"] = o.panel.map(FAMILY)
    fam = o.groupby(["variant", "family"]).S_state.mean().groupby("variant").mean()
    print(o.round(4).to_string())
    print("family-mean S_state:", fam.round(5).to_dict())
    # error by traffic state (regular cells)
    x = d[d.kind == "reg"].copy()
    x["state"] = pd.cut(x.y_speed / x.vf, [0, 0.6, 0.85, 0.95, 2], labels=["queued", "slow", "transition", "free"])
    x["se"] = (x.p_speed - x.y_speed) ** 2; x["qe"] = (x.p_flow - x.y_flow) ** 2
    b = x.groupby("state", observed=True).agg(n=("se", "size"), mse_s=("se", "mean"), mse_q=("qe", "mean"))
    b["share_sse_s"] = b.n * b.mse_s / (b.n * b.mse_s).sum(); b["share_sse_q"] = b.n * b.mse_q / (b.n * b.mse_q).sum()
    b["frac"] = b.n / b.n.sum()
    print(b.round(4).to_string())
    return o


if __name__ == "__main__":
    score(sys.argv[1] if len(sys.argv) > 1 else "hold1")
