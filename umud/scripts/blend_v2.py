"""Robust per-target blend of the pipeline with a reference submission (v2).

    python scripts/blend_v2.py --pipeline A.csv --ref vera.csv --features features.csv --groups groups.npy \
        --w 0.56 0.27 0.75 --pa-offset 1.6 [--clip-pa 6 --clip-fl 15 --clip-mt 3] --out out.csv

For each target t: out = ref + w_t * clip(pipe - ref, -c_t, c_t)  (no clip = plain linear blend), then
cine-loop smoothing (0.6 toward the 5-frame median) and the two public anchors pinned. Rows where the
pipeline failed (features ok == 0) use the reference only. The PA offset is added to the pipeline PA first.
Fixes relative to the day-1..3 blends: failures no longer blend prior constants; optional outlier clipping.
"""
import argparse

import numpy as np
import pandas as pd

T = ("pa_deg", "fl_mm", "mt_mm")
RANGES = {"pa_deg": (5.0, 45.0), "fl_mm": (30.0, 200.0), "mt_mm": (10.0, 50.0)}
ANCHORS = {"IMG_00001.tif": (17.334, 79.423, 21.778), "IMG_00002.tif": (12.876, 69.424, 15.478)}

ap = argparse.ArgumentParser()
ap.add_argument("--pipeline", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--features", required=True)
ap.add_argument("--groups", required=True, help=".npy of video-group ids in image_id order")
ap.add_argument("--w", type=float, nargs=3, required=True, metavar=("PA", "FL", "MT"))
ap.add_argument("--pa-offset", type=float, default=1.6)
ap.add_argument("--clip-pa", type=float, default=None)
ap.add_argument("--clip-fl", type=float, default=None)
ap.add_argument("--clip-mt", type=float, default=None)
ap.add_argument("--fl-tail", type=float, nargs=2, default=None, metavar=("THRESH", "W"),
                help="piecewise FL blend: weight w_FL for |pipe-ref| <= THRESH mm, weight W beyond it")
ap.add_argument("--mt-offset", type=float, default=0.0, help="mm added to the final MT (before smoothing)")
ap.add_argument("--mt-scale", type=float, default=1.0, help="global factor on the final MT")
ap.add_argument("--fl-scale", type=float, default=1.0, help="global factor on the final FL")
ap.add_argument("--alpha", type=float, default=0.6, help="cine-loop smoothing strength")
ap.add_argument("--out", required=True)
a = ap.parse_args()

p = pd.read_csv(a.pipeline).set_index("image_id")
r = pd.read_csv(a.ref).set_index("image_id").loc[p.index]
feat = pd.read_csv(a.features).set_index("image_id").loc[p.index]
g = pd.Series(np.load(a.groups), index=p.index)
p = p.copy()
p["pa_deg"] += a.pa_offset
failed = feat["ok"] != 1
p.loc[failed, list(T)] = r.loc[failed, list(T)]
clips = {"pa_deg": a.clip_pa, "fl_mm": a.clip_fl, "mt_mm": a.clip_mt}
out = pd.DataFrame(index=p.index)
for t, w in zip(T, a.w):
    d = p[t] - r[t]
    if clips[t] is not None:
        d = d.clip(-clips[t], clips[t])
    if t == "fl_mm" and a.fl_tail is not None:
        th, w2 = a.fl_tail
        core = d.clip(-th, th)
        out[t] = r[t] + w * core + w2 * (d - core)  # continuous; slope w inside, w2 in the tail
        continue
    out[t] = r[t] + w * d
out["mt_mm"] = out["mt_mm"] * a.mt_scale + a.mt_offset
out["fl_mm"] *= a.fl_scale
for t in T:
    out[t] = (1 - a.alpha) * out[t] + a.alpha * out.groupby(g)[t].transform("median")
for k, v in ANCHORS.items():
    out.loc[k] = v
for t, (lo, hi) in RANGES.items():
    out[t] = out[t].clip(lo, hi)
out.round(3).reset_index().to_csv(a.out, index=False)
print(f"wrote {a.out}: {len(out)} rows, {int(failed.sum())} pipeline failures -> ref")
