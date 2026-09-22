"""Task 4 (ODME) data access: incidence operator, counts, priors per panel/split.

Everything here mirrors src/task4/build_task4_odme_artifacts.py of the official
repository (same path order, same link order, same "measured" rule), so an
estimate produced from these arrays can be written straight into a submission.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REL = Path(os.environ.get("TFB_REL", "/home/user/data/kaggle_public"))
PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N",
          "D7_I405_S", "D12_I5_N", "D12_I5_S", "D12_I405_N", "D12_I405_S"]
FAMILY = {p: p.rsplit("_", 1)[0] for p in PANELS}
SPLITS = ("train", "validation", "private")


@dataclass
class Panel:
    panel: str
    path_ids: list
    link_ids: list            # all links of the incidence (official order)
    A: np.ndarray             # (n_links, n_paths) 0/1
    paths: pd.DataFrame       # path_id, origin_zone, destination_zone, link_seq
    orig: np.ndarray          # origin zone index per path
    dest: np.ndarray          # destination zone index per path
    zones: list               # zone names sorted by number
    counts: dict = field(default_factory=dict)    # split -> full-length counts (nan = unmeasured)
    prior: dict = field(default_factory=dict)     # split -> prior path flow
    dep: dict = field(default_factory=dict)       # split -> departure_time token

    def measured(self, split: str) -> np.ndarray:
        return np.isfinite(self.counts[split])

    def Am(self, split: str) -> np.ndarray:
        return self.A[self.measured(split)]

    def cm(self, split: str) -> np.ndarray:
        c = self.counts[split]
        return c[np.isfinite(c)]


def _zone_num(z: str) -> int:
    return int(str(z).lstrip("Z"))


@lru_cache(maxsize=None)
def load(panel: str) -> Panel:
    net = REL / "corridors" / panel / "network"
    paths = pd.read_csv(net / "path_set.csv", dtype=str)
    inc = pd.read_csv(net / "path_link_incidence.csv", dtype=str)
    path_ids = paths.path_id.tolist()
    link_ids = inc.link_id.drop_duplicates().tolist()
    pidx = {x: i for i, x in enumerate(path_ids)}
    lidx = {x: i for i, x in enumerate(link_ids)}
    A = np.zeros((len(link_ids), len(path_ids)))
    A[inc.link_id.map(lidx).to_numpy(), inc.path_id.map(pidx).to_numpy()] = 1.0
    zones = sorted(set(paths.origin_zone) | set(paths.destination_zone), key=_zone_num)
    zidx = {z: i for i, z in enumerate(zones)}
    P = Panel(panel, path_ids, link_ids, A, paths,
              paths.origin_zone.map(zidx).to_numpy(), paths.destination_zone.map(zidx).to_numpy(), zones)
    for s in SPLITS:
        d = REL / "task4" / panel / s
        c = pd.read_csv(d / "synthetic_link_counts.csv", dtype={"link_id": str})
        P.counts[s] = c.set_index("link_id")["count"].reindex(link_ids).to_numpy(dtype=float)
        b = pd.read_csv(d / "synthetic_weak_prior.csv", dtype={"path_id": str})
        P.prior[s] = b.set_index("path_id")["path_flow"].reindex(path_ids).fillna(0.0).to_numpy(dtype=float)
        tmpl = pd.read_csv(d / "sample_submission_path_flow.csv", dtype=str, nrows=1)
        P.dep[s] = str(tmpl.departure_time.iloc[0])
    return P


def submission_frame(P: Panel, split: str, f: np.ndarray) -> pd.DataFrame:
    tmpl = pd.read_csv(REL / "task4" / P.panel / split / "sample_submission_path_flow.csv", dtype=str)
    val = pd.Series(np.asarray(f, float), index=P.path_ids)
    out = tmpl[["panel", "departure_time", "path_id", "origin_zone", "destination_zone"]].copy()
    out["path_flow"] = val.reindex(out.path_id).to_numpy()
    assert np.isfinite(out.path_flow).all() and (out.path_flow >= 0).all()
    return out
