"""Validate the pipeline on the UMUD "Expert Analysed Benchmarks" (OSF xbawc, 35 images, up to 7 raters).

    python scripts/osf_benchmark.py --bench <dir with im_XX_arch.tif + Results_*.xlsx> --weights <dir with apo.pt/fasc.pt>
Writes <bench>/pipeline_features.csv and prints per-target MAE / bias vs the rater mean, the DLTrack
baseline and the inter-rater spread, on the competition metric.
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from umud import seg  # noqa: E402
from umud.geometry import analyse  # noqa: E402

TAU = {"pa": 6.0, "fl": 12.0, "mt": 3.0}

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True)
ap.add_argument("--weights", required=True)
a = ap.parse_args()
bench = Path(a.bench)
x = pd.read_excel(glob.glob(str(bench / "Results_*.xlsx"))[0], sheet_name="Manual_architecture")
for t in ("MT", "FL", "PA"):
    cols = [f"R{i}_{t}" for i in range(1, 8)]
    r = x[cols].astype(float)
    med = r.median(axis=1)
    r = r.where((r - med.values[:, None]).abs() < 0.5 * med.values[:, None])  # drop entry errors (e.g. 80 mm MT)
    x[f"GT_{t}"] = r.mean(axis=1)
    x[f"SD_{t}"] = r.std(axis=1)
    # leave-one-rater-out: a single rater vs the mean of the others (human-level error)
    x[f"HUM_{t}"] = [np.nanmean([abs(r.iloc[i, j] - r.iloc[i].drop(r.columns[j]).mean())
                                 for j in range(r.shape[1]) if np.isfinite(r.iloc[i, j])]) for i in range(len(r))]

models = {}
for k in ("apo", "fasc"):
    m = seg.make_model(None)
    m.load_state_dict(torch.load(Path(a.weights) / f"{k}.pt", map_location="cpu"))
    models[k] = m.eval()
rows = []
for r in x.itertuples():
    g = seg.read_gray(str(bench / f"{r.ImageID}.tif"))
    probs = {k: seg.predict(m, g, "cpu") for k, m in models.items()}
    f = analyse(probs["apo"], probs["fasc"], r.Scale_pixel_per_cm / 10.0)
    f["ImageID"] = r.ImageID
    rows.append(f)
F = pd.DataFrame(rows).merge(x, on="ImageID")
F.to_csv(bench / "pipeline_features.csv", index=False)


def rep(name, est, t):
    e = (F[est] - F[f"GT_{t.upper()}"]).dropna()
    return f"{name:14s} n={len(e):2d} MAE={e.abs().mean():6.2f} bias={e.mean():+6.2f} norm={e.abs().mean() / TAU[t]:.3f}"


for t, cands in {"mt": ["mt_inner", "mt_inner_vert", "mt_center"], "pa": ["pa_wmed", "pa_med", "pa_top5", "pa_orient"],
                 "fl": ["fl_med", "fl_wmed", "fl_top5"]}.items():
    print(f"== {t.upper()}  human (1 rater vs rest) MAE={F[f'HUM_{t.upper()}'].mean():.2f}")
    print("  " + rep("DLTrack", f"DLTrack_{t.upper()}", t))
    for c in cands:
        print("  " + rep(c, c, t))
