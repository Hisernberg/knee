"""Compare extracted geometry features with the two public anchors and a reference submission.

The reference (e.g. a public LB-scored CSV) is only a diagnostic: it tells us whether an estimator
tracks something sensible and whether it is biased. Usage:

    python scripts/diagnose.py --features work/features.csv --ref subs/s00_vera_public.csv
"""
import argparse

import numpy as np
import pandas as pd

ANCHORS = {"IMG_00001.tif": (17.334, 79.423, 21.778), "IMG_00002.tif": (12.876, 69.424, 15.478)}

ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True)
ap.add_argument("--ref", required=True)
a = ap.parse_args()
f = pd.read_csv(a.features).set_index("image_id")
r = pd.read_csv(a.ref).set_index("image_id").loc[f.index]
print("ok rate", f.ok.mean(), "| n_fasc median", f.get("n_fasc_kept", pd.Series(dtype=float)).median())
groups = {"pa_deg": [c for c in f.columns if c.startswith("pa_")],
          "fl_mm": [c for c in f.columns if c.startswith("fl_")],
          "mt_mm": [c for c in f.columns if c.startswith("mt_")]}
for tgt, cols in groups.items():
    print(f"\n== {tgt}: ref mean {r[tgt].mean():.2f}")
    for c in cols:
        v = f[c]
        ok = v.notna()
        if ok.sum() < 10:
            continue
        d = v[ok] - r[tgt][ok]
        corr = np.corrcoef(v[ok], r[tgt][ok])[0, 1]
        # best linear map to ref (diagnostic only)
        print(f"  {c:16s} n={ok.sum():3d} mean={v[ok].mean():7.2f} bias={d.mean():+6.2f} "
              f"MAD={d.abs().mean():6.2f} MAD_debiased={(d - d.median()).abs().mean():6.2f} corr={corr:.3f}")
print("\nanchors:")
for k, (pa, fl, mt) in ANCHORS.items():
    row = f.loc[k]
    print(k, "GT", pa, fl, mt, "|", {c: round(row[c], 2) for c in
          ["pa_wmed", "pa_orient", "fl_wmed", "mt_inner", "mt_center"] if c in row})
