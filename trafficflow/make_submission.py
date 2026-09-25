"""Build a full Kaggle upload from the per-task prediction artefacts.

python -m trafficflow.make_submission --state-tag v1 --queue /home/user/work/t2/<f>.csv \
       --odme /home/user/work/t4/t4_l2proj.csv --out /home/user/work/subs/<name>.csv [--only state|queue|odme]

--only keeps one task and zeroes the others, which isolates that task's score on
the leaderboard (a missing task scores exactly 0).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import PANELS, REL, tindex
from .submit import assemble
from .t1_holdout import reconcile
from .t1_pipeline import WORK


def fd_per_cell(pr: pd.DataFrame):
    """Per-lane triangular FD parameters (v_f, w, k_jam) for every predicted cell."""
    rows = []
    for p in PANELS:
        fd = pd.read_csv(REL / "corridors" / p / "network" / "fd_parameters.csv", dtype={"link_id": str})
        fd = fd.drop_duplicates("link_id")
        cap_l = fd.capacity_vph / fd.lanes
        kj_l = fd.k_jam / fd.lanes
        kc_l = cap_l / fd.free_speed_kmh.clip(lower=1)
        rows.append(pd.DataFrame({"panel": p, "link_id": fd.link_id, "vf": fd.free_speed_kmh,
                                  "w": cap_l / (kj_l - kc_l).clip(lower=1e-3), "kj": kj_l}))
    m = pr[["panel", "link_id"]].merge(pd.concat(rows), on=["panel", "link_id"], how="left")
    return m.vf.to_numpy(), m.w.to_numpy(), m.kj.to_numpy()


def state_frame(tag: str, a: float = 0.25, gate: float | None = None, fdband=None,
                smooth: str | None = None) -> pd.DataFrame:
    """Task 1 rows. gate: reconcile (v, q) to the density model only where the
    predicted speed is below gate*v_f (dense traffic, where the density model is
    better than q/v); fdband=(lo, hi): inside lo..hi*v_f use the FD congested-branch
    density w*kj/(v+w) instead of the density model. smooth: TV smoothing of the
    density inside runs of target cells (t1_smooth; 'default' or 'key=value,...'),
    applied after the gate and before clipping; None (default) leaves it off."""
    pr = pd.read_parquet(WORK / "pred" / f"state_{tag}.parquet")
    v0, q0 = pr.speed.values, pr.flow_lane.values
    if a is not None and "dens_lane" in pr:
        k = pr.dens_lane.values.copy()
        if gate is not None or fdband is not None:
            vf, w, kj = fd_per_cell(pr)
            if fdband is not None:
                m = (v0 >= fdband[0] * vf) & (v0 < fdband[1] * vf)
                k[m] = w[m] * kj[m] / (v0[m] + w[m])
        v, q = reconcile(v0, q0, k, a)
        if gate is not None:
            g = v0 < gate * vf
            v, q = np.where(g, v, v0), np.where(g, q, q0)
    else:
        v, q = v0, q0
    if smooth is not None:
        from .t1_smooth import parse, smooth_frame
        spec = parse(smooth)
        if gate is not None and a is not None and "dens_lane" in pr:
            sg = v0 < gate * vf
        else:
            sg = np.zeros(len(pr), bool)
        print(f"state smoothing {spec} ({sg.sum()} gated cells)", flush=True)
        v, q = smooth_frame(pr, v, q, sg, spec)
    pr["speed_kmh"] = np.clip(v, 3.0, 130.0)
    pr["flow_vph"] = np.clip(q * pr.lanes.values, 60.0, None)
    out = []
    for p in PANELS:
        for split in ("validation", "private"):
            t = pd.read_csv(REL / "task1" / p / split / "sample_submission_state.csv", dtype=str)
            t["t"] = tindex(t.timestamp)
            m = t.drop(columns=["speed_kmh", "flow_vph"]).merge(
                pr.loc[pr.panel == p, ["t", "link_id", "speed_kmh", "flow_vph"]], on=["t", "link_id"], how="left")
            out.append(m.drop(columns=["t"]))
    return pd.concat(out, ignore_index=True)


def queue_frame(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"window_id": str, "timestamp": str, "link_id": str})


def odme_frame(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"panel": str, "departure_time": str, "path_id": str})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-tag")
    ap.add_argument("--recon-a", type=float, default=0.25)
    ap.add_argument("--gate", type=float, default=None, help="reconcile only where speed < gate*v_f")
    ap.add_argument("--fdband", type=float, nargs=2, default=None, help="use FD density for lo..hi*v_f")
    ap.add_argument("--smooth", default=None,
                    help="TV smoothing of the density inside runs of target cells (trafficflow.t1_smooth): "
                         "'default' or 'free=0.0075,free_a=0.001,gate=0.02,gate_a=0.005,dark=0.05'; off if omitted")
    ap.add_argument("--queue")
    ap.add_argument("--odme")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", choices=["state", "queue", "odme"])
    ap.add_argument("--zero", nargs="*", default=[], choices=["state", "queue", "odme"],
                    help="zero these tasks (a zeroed task scores exactly 0 on the leaderboard)")
    ap.add_argument("--note", default="")
    ap.add_argument("--force", action="store_true", help="write even if checks fail (deliberate probes only)")
    a = ap.parse_args()
    st = state_frame(a.state_tag, a.recon_a, a.gate, a.fdband, a.smooth)
    q = queue_frame(a.queue)
    o = odme_frame(a.odme)
    force = a.force
    zero = set(a.zero) | ({"state", "queue", "odme"} - {a.only} if a.only else set())
    if "state" in zero:
        st = st.assign(speed_kmh=0.0, flow_vph=0.0); force = True
    if "queue" in zero:
        q = q.assign(queue_pred=0); force = True
    if "odme" in zero:
        o = o.assign(path_flow=0.0); force = True
    assemble(st, q, o, Path(a.out), force=force, note=a.note)
