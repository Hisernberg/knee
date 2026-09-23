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


def state_frame(tag: str, a: float = 0.25) -> pd.DataFrame:
    pr = pd.read_parquet(WORK / "pred" / f"state_{tag}.parquet")
    if a is not None and "dens_lane" in pr:
        v, q = reconcile(pr.speed.values, pr.flow_lane.values, pr.dens_lane.values, a)
    else:
        v, q = pr.speed.values, pr.flow_lane.values
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
    ap.add_argument("--queue")
    ap.add_argument("--odme")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", choices=["state", "queue", "odme"])
    ap.add_argument("--zero", nargs="*", default=[], choices=["state", "queue", "odme"],
                    help="zero these tasks (a zeroed task scores exactly 0 on the leaderboard)")
    ap.add_argument("--note", default="")
    ap.add_argument("--force", action="store_true", help="write even if checks fail (deliberate probes only)")
    a = ap.parse_args()
    st = state_frame(a.state_tag, a.recon_a)
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
