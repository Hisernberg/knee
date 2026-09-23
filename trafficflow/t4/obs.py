"""PM-window (15:00-19:00 UTC-hour tokens) means of released mainline and ramp flows."""
from __future__ import annotations

import glob
from functools import lru_cache

import numpy as np
import pandas as pd

from .data import REL

PM = (15, 19)


def _pm(frame: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(frame.timestamp, utc=True)
    h = ts.dt.hour
    return frame[(h >= PM[0]) & (h < PM[1])]


@lru_cache(maxsize=None)
def mainline_pm(panel: str, split: str, unmasked: bool = False) -> pd.DataFrame:
    """Per-link PM stats: mean flow over all released non-null values, n rows."""
    base = REL / "corridors" / panel / split
    pat = "mainline_states/**/*.parquet" if unmasked else "mainline_states_masked/**/*.parquet"
    parts = []
    for x in sorted(glob.glob(str(base / pat), recursive=True)):
        d = pd.read_parquet(x, columns=["timestamp", "link_id", "flow_vph", "pct_observed"])
        d = _pm(d)
        parts.append(d[["link_id", "flow_vph", "pct_observed"]])
    D = pd.concat(parts, ignore_index=True)
    D["link_id"] = D.link_id.astype(str)
    D = D.dropna(subset=["flow_vph"])
    g = D.groupby("link_id").flow_vph
    out = pd.DataFrame({"mean": g.mean(), "median": g.median(), "n": g.size()})
    e = D[D.pct_observed >= 75].groupby("link_id").flow_vph
    out["mean_elig"] = e.mean()
    return out


@lru_cache(maxsize=None)
def ramp_pm(panel: str, split: str) -> pd.DataFrame:
    base = REL / "corridors" / panel / split
    parts = []
    for x in sorted(glob.glob(str(base / "ramp_states/**/*.parquet"), recursive=True)):
        d = pd.read_parquet(x, columns=["timestamp", "ramp_link_id", "ramp_type", "flow_vph", "pct_observed"])
        d = _pm(d)
        parts.append(d)
    D = pd.concat(parts, ignore_index=True).dropna(subset=["flow_vph"])
    D["ramp_link_id"] = D.ramp_link_id.astype(str)
    g = D.groupby("ramp_link_id").flow_vph
    out = pd.DataFrame({"mean": g.mean(), "n": g.size(), "type": D.groupby("ramp_link_id").ramp_type.first()})
    e = D[D.pct_observed >= 75].groupby("ramp_link_id").flow_vph
    out["mean_elig"] = e.mean()
    return out
