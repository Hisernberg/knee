"""Write Task 4 candidate files (validation + private rows, all 10 panels).

    python -m trafficflow.t4.make_submissions            # writes /home/user/work/t4/*.csv

Columns: panel,departure_time,path_id,origin_zone,destination_zone,path_flow
Rows follow each split's sample_submission_path_flow.csv (departure_time copied).
Nothing is submitted anywhere.
"""
from __future__ import annotations

import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np
import pandas as pd

from .data import PANELS, submission_frame
from .estimators import context, est_l2proj, est_official_baseline, make_nnls
from .metric import s_link

CANDIDATES = {
    "t4_l2proj": est_l2proj,                       # primary
    "t4_nnls_split_lam0.05": make_nnls(0.05),      # official objective on the split's own counts
    "t4_ref_official_baseline": est_official_baseline,  # calibration reference only (expected ~0.836)
}


def build(name: str, fn, out_dir: Path) -> pd.DataFrame:
    parts, diag = [], []
    for split in ("validation", "private"):
        for panel in PANELS:
            x = context(panel, split)
            f = np.asarray(fn(x), float)
            f = np.where(np.abs(f) < 1e-9, 0.0, f)
            assert np.isfinite(f).all() and (f >= 0).all(), (name, panel, split)
            parts.append(submission_frame(x.P, split, f))
            diag.append({"split": split, "panel": panel, "S_link": s_link(x.P.Am(split), x.P.cm(split), f),
                         "sum_f": f.sum(), "zero_frac": float((f == 0).mean()),
                         "max_rel_seg_resid": float(np.max(np.abs(x.S @ f - x.c) / x.c))})
    sub = pd.concat(parts, ignore_index=True)
    assert not sub.duplicated(["panel", "departure_time", "path_id"]).any()
    out = out_dir / f"{name}.csv"
    sub.to_csv(out, index=False, float_format="%.10g")
    d = pd.DataFrame(diag)
    print(f"{name}: {len(sub):,} rows -> {out}; min S_link={d.S_link.min():.6f}, "
          f"mean zero_frac={d.zero_frac.mean():.3f}, max seg resid={d.max_rel_seg_resid.max():.2e}")
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("/home/user/work/t4"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in CANDIDATES.items():
        build(name, fn, a.out_dir)


if __name__ == "__main__":
    main()
