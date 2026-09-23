"""Dense (time x link) tensors for every TrafficFlowBench panel.

One station per link on every panel, so a panel is a set of float32 arrays of
shape [T, L] where T = 334 days * 288 slots covers train+validation+private and
L is the link count in corridor order (topology order_index). Everything the
models need is precomputed once and stored as a compressed npz per panel.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REL = Path(os.environ.get("TFB_REL", "/home/user/data/kaggle_public"))
CACHE = Path(os.environ.get("TFB_CACHE", "/home/user/cache"))
PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N",
          "D7_I405_S", "D12_I5_N", "D12_I5_S", "D12_I405_N", "D12_I405_S"]
FAMILY = {p: p.rsplit("_", 1)[0] for p in PANELS}
SPLITS = ("train", "validation", "private")
T0 = pd.Timestamp("2030-06-01T00:00:00Z")
NDAYS = 334
SLOTS = 288
T = NDAYS * SLOTS
SPLIT_DAYS = {"train": (0, 273), "validation": (273, 304), "private": (304, 334)}
REGIME_ID = {"R1": 1, "R2": 2, "R3": 3}


def tindex(ts) -> np.ndarray:
    """Timestamp text/series -> integer slot index from T0."""
    t = pd.to_datetime(pd.Series(ts), utc=True)
    return ((t - T0).dt.total_seconds() // 300).astype(np.int64).to_numpy()


def network(panel: str) -> dict:
    nd = REL / "corridors" / panel / "network"
    topo = pd.read_csv(nd / "lwr_mainline_topology.csv", dtype={"link_id": str})
    topo = topo.sort_values("order_index").reset_index(drop=True)
    fd = pd.read_csv(nd / "fd_parameters.csv", dtype={"link_id": str}).drop_duplicates("link_id").set_index("link_id")
    links = topo.link_id.astype(str).tolist()
    fd = fd.reindex(links)
    return {"links": links, "topo": topo, "fd": fd,
            "ramps": pd.read_csv(nd / "ramp_attachment_map.csv", dtype=str)}


def _read(paths, cols):
    return pd.concat([pd.read_parquet(p, columns=cols) for p in paths], ignore_index=True)


def build_panel(panel: str) -> Path:
    out = CACHE / f"{panel}.npz"
    if out.exists():
        return out
    net = network(panel)
    links = net["links"]
    L = len(links)
    lidx = {l: i for i, l in enumerate(links)}
    A = {k: np.full((T, L), np.nan, np.float32) for k in ("speed", "flow", "occ", "tspeed", "tflow")}
    pct = np.full((T, L), -1, np.int8)
    elig = np.zeros((T, L), bool)
    regime = np.zeros(NDAYS, np.int8)
    target = np.zeros((T, L), np.int8)  # regime id at Task 1 target cells
    cols = ["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy", "pct_observed", "is_score_eligible"]
    base = REL / "corridors" / panel
    for split in SPLITS:
        for reg in ("R1", "R2", "R3"):
            ps = sorted((base / split / "mainline_states_masked" / f"mask_regime={reg}").glob("*.parquet"))
            if not ps:
                continue
            d = _read(ps, cols)
            ti = tindex(d.timestamp); li = d.link_id.astype(str).map(lidx).to_numpy()
            A["speed"][ti, li] = d.speed_kmh.to_numpy(np.float32)
            A["flow"][ti, li] = d.flow_vph.to_numpy(np.float32)
            A["occ"][ti, li] = d.occupancy.to_numpy(np.float32)
            pct[ti, li] = d.pct_observed.to_numpy(np.int8)
            elig[ti, li] = d.is_score_eligible.astype(bool).to_numpy()
            regime[np.unique(ti // SLOTS)] = REGIME_ID[reg]
        tp = REL / "task1" / panel / split / "sample_submission_state.csv"
        t = pd.read_csv(tp, usecols=["timestamp", "link_id", "mask_regime"], dtype=str)
        target[tindex(t.timestamp), t.link_id.map(lidx).to_numpy()] = t.mask_regime.map(REGIME_ID).to_numpy(np.int8)
    ps = sorted((base / "train" / "mainline_states").glob("**/*.parquet"))
    d = _read(ps, ["timestamp", "link_id", "speed_kmh", "flow_vph"])
    ti = tindex(d.timestamp); li = d.link_id.astype(str).map(lidx).to_numpy()
    A["tspeed"][ti, li] = d.speed_kmh.to_numpy(np.float32)
    A["tflow"][ti, li] = d.flow_vph.to_numpy(np.float32)
    # ramps: per ramp link, flow + validity
    ramps = net["ramps"]
    ridx = {r: i for i, r in enumerate(ramps.ramp_link_id)}
    rflow = np.full((T, len(ramps)), np.nan, np.float32)
    rvalid = np.zeros((T, len(ramps)), bool)
    for split in SPLITS:
        ps = sorted((base / split / "ramp_states").glob("**/*.parquet"))
        d = _read(ps, ["timestamp", "ramp_link_id", "flow_vph", "pct_observed", "is_score_eligible"])
        ti = tindex(d.timestamp); ri = d.ramp_link_id.astype(str).map(ridx).to_numpy()
        ok = ~pd.isna(ri)
        ti, ri, d = ti[ok], ri[ok].astype(int), d[ok]
        rflow[ti, ri] = d.flow_vph.to_numpy(np.float32)
        rvalid[ti, ri] = (d.is_score_eligible.astype(bool) & (d.pct_observed >= 75) & d.flow_vph.notna()).to_numpy()
    np.savez_compressed(out, **A, pct=pct, elig=elig, regime=regime, target=target,
                        rflow=rflow, rvalid=rvalid, links=np.array(links), ramps=ramps.ramp_link_id.to_numpy())
    return out


def load(panel: str) -> dict:
    z = np.load(build_panel(panel), allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d["net"] = network(panel)
    d["panel"] = panel
    return d


if __name__ == "__main__":
    import sys
    from multiprocessing import Pool
    panels = sys.argv[1:] or PANELS
    with Pool(2) as p:
        for r in p.imap_unordered(build_panel, panels):
            print(r, flush=True)
