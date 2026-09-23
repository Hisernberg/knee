"""Write Task 2 submission files aligned to the released templates.

``write(preds, path)`` takes {window_id: bool[6, L]} (links in cache order) and
emits window_id,timestamp,link_id,queue_pred for exactly the rows of the
validation and private sample_submission_queue.csv of the 8 scored panels,
with the timestamp text copied verbatim.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data import tindex
from .core import PANELS8, official_windows, statics, template

SPLITS = ("validation", "private")


def write(preds: dict, path: Path, splits=SPLITS) -> pd.DataFrame:
    parts = []
    for p in PANELS8:
        lid = statics(p)["lid"]
        for sp in splits:
            tp = template(p, sp)
            w = official_windows(p, sp).set_index("window_id")["T"]
            k = tindex(tp.timestamp) - w.loc[tp.window_id].to_numpy() - 1
            assert k.min() >= 0 and k.max() <= 5, (p, sp)
            li = tp.link_id.map(lid).to_numpy()
            q = np.zeros(len(tp), np.int8)
            for wid, idx in tp.groupby("window_id").indices.items():
                P = preds.get(wid)
                if P is None:
                    continue
                q[idx] = P[k[idx], li[idx]].astype(np.int8)
            tp = tp.copy(); tp["queue_pred"] = q
            parts.append(tp)
    out = pd.concat(parts, ignore_index=True)
    assert not out.duplicated(["window_id", "timestamp", "link_id"]).any()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def check(path: Path) -> dict:
    """Row-exactness check against the templates."""
    sub = pd.read_csv(path, dtype=str)
    tp = pd.concat([template(p, s) for p in PANELS8 for s in SPLITS], ignore_index=True)
    key = ["window_id", "timestamp", "link_id"]
    m = tp.merge(sub, on=key, how="outer", indicator=True)
    return dict(rows=len(sub), template_rows=len(tp), missing=int((m._merge == "left_only").sum()),
                extra=int((m._merge == "right_only").sum()),
                binary=bool(sub.queue_pred.isin(["0", "1"]).all()),
                pos_rate=float((sub.queue_pred == "1").mean()))
