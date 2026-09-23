"""Build and cache per-panel Task 2 window datasets from the train split.

For every scored panel this writes ``WORK/ds_<panel>.npz`` holding

* ``sim``   windows drawn by the reproduced official selector started on every
            train day (the evaluation distribution);
* ``cand``  a superset for training: every onset candidate that is the first of
            its run (the selector can only ever draw those) and every second
            ongoing candidate origin;
* ``off``   the 10 official train windows;

each with the history block exactly as a released window history would show it
([n, 12, L] speed / flow / eligibility / pct_observed) and the approximate truth
([n, 6, L] bool). The per-slot train truth is stored packed for the profiles.

    PYTHONPATH=/home/user/knee python -m trafficflow.t2.dataset [PANEL ...]
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from ..data import SLOTS
from .core import (H, K, PANELS8, TRAIN_END, WORK, PanelT2, official_windows,
                   simulate_windows)


def _blocks(P: PanelT2, T: np.ndarray):
    idx = T[:, None] + np.arange(-H, 0)[None, :]
    fut = T[:, None] + np.arange(1, K + 1)[None, :]
    return dict(hs=P.speed[idx], hf=P.flow[idx], he=P.elig[idx], hp=P.pct[idx].astype(np.int8),
                y=P.qtrue[fut])


def build(panel: str, force: bool = False):
    out = WORK / f"ds_{panel}.npz"
    if out.exists() and not force:
        return out
    t = time.time()
    P = PanelT2(panel)
    st = P.selector_stats()
    sim = simulate_windows(P, st, range(0, 266))
    sim["src"] = "sim"
    off = official_windows(panel, "train")[["T", "condition"]].copy()
    off["draws"] = 1; off["src"] = "off"
    # training superset
    s = st.set_index("T")
    cand = s[s.cand]
    prev_cand = s.cand.shift(1, fill_value=False)
    on_first = cand[(~cand.ongoing) & (~prev_cand.loc[cand.index])]
    og = cand[cand.ongoing]
    og = og[(og.index % 2) == 0]
    c = pd.concat([pd.DataFrame({"T": on_first.index, "condition": "queue_onset"}),
                   pd.DataFrame({"T": og.index, "condition": "queue_ongoing"})])
    c["draws"] = 1; c["src"] = "cand"
    W = pd.concat([sim, off, c], ignore_index=True)
    W["T"] = W["T"].astype(np.int64)
    W = W.merge(st[["T", "cov", "nq", "maxl", "piou", "nfut", "nT"] + [f"n{k}" for k in range(1, K + 1)]], on="T", how="left")
    B = _blocks(P, W["T"].to_numpy())
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **B, qtrue=np.packbits(P.qtrue, axis=1), L=P.L,
                        **{f"w_{c}": W[c].to_numpy() for c in W.columns})
    print(panel, "windows", W.groupby(["src", "condition"]).size().to_dict(), f"{time.time()-t:.0f}s", flush=True)
    return out


def load_ds(panel: str) -> tuple[pd.DataFrame, dict]:
    z = np.load(WORK / f"ds_{panel}.npz", allow_pickle=True)
    W = pd.DataFrame({k[2:]: z[k] for k in z.files if k.startswith("w_")})
    W["panel"] = panel
    L = int(z["L"])
    A = {k: z[k] for k in ("hs", "hf", "he", "hp", "y")}
    A["qtrue"] = np.unpackbits(z["qtrue"], axis=1, count=L).astype(bool)
    return W, A


if __name__ == "__main__":
    for p in (sys.argv[1:] or PANELS8):
        build(p, force=True)
