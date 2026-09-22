"""Probability maps -> per-image geometry features -> submission.csv.

    python -m umud.predict features --probs <seg out dir> --out work/features.csv
    python -m umud.predict submit --features work/features.csv --config configs/final.json --out submission.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from umud.geometry import analyse

RANGES = {"pa_deg": (5.0, 45.0), "fl_mm": (30.0, 200.0), "mt_mm": (10.0, 50.0)}


def extract_features(seg_dir: Path) -> pd.DataFrame:
    crops = pd.read_csv(seg_dir / "crops.csv")
    rows = []
    for r in crops.itertuples():
        ap = cv2.imread(str(seg_dir / "probs" / f"{r.image_id}_apo.png"), cv2.IMREAD_GRAYSCALE)
        fp = cv2.imread(str(seg_dir / "probs" / f"{r.image_id}_fasc.png"), cv2.IMREAD_GRAYSCALE)
        f = analyse(ap, fp, r.px_per_mm)
        f.update(image_id=r.image_id, px_per_mm=r.px_per_mm, family=r.family)
        rows.append(f)
    return pd.DataFrame(rows)


def video_groups(ids: list[str], data_dir: Path | None = None, thr: float = 0.08) -> np.ndarray:
    """Group consecutive test frames that come from the same cine loop (thumbnail correlation)."""
    from PIL import Image
    g = [0]
    prev = None
    for i, name in enumerate(ids):
        p = data_dir / name
        a = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
        cur = (a.shape, cv2.resize(a, (64, 48), interpolation=cv2.INTER_AREA).ravel())
        if prev is not None:
            same = prev[0] == cur[0] and 1 - np.corrcoef(prev[1], cur[1])[0, 1] < thr
            g.append(g[-1] + (0 if same else 1))
        prev = cur
    return np.asarray(g)


def make_submission(feat: pd.DataFrame, cfg: dict, groups: np.ndarray | None = None) -> pd.DataFrame:
    """Turn features into predictions following an explicit, documented config."""
    f = feat.copy()
    pri = cfg.get("prior", {"pa_deg": 17.0, "fl_mm": 80.0, "mt_mm": 21.0})
    pa = f[cfg.get("pa_col", "pa_wmed")].astype(float)
    mt = f[cfg.get("mt_col", "mt_inner")].astype(float)
    mt = mt * cfg.get("mt_scale", 1.0) + cfg.get("mt_offset", 0.0)
    pa = pa * cfg.get("pa_scale", 1.0) + cfg.get("pa_offset", 0.0)
    pa = pa.fillna(pri["pa_deg"]).clip(*RANGES["pa_deg"])
    mt = mt.fillna(pri["mt_mm"]).clip(*RANGES["mt_mm"])
    fl_geo = f[cfg.get("fl_col", "fl_wmed")].astype(float) * cfg.get("fl_scale", 1.0)
    # trigonometric FL with aponeurosis divergence ignored: MT / sin(PA)
    fl_trig = mt / np.sin(np.radians(pa)) * cfg.get("fl_trig_scale", 1.0)
    wg = cfg.get("fl_geo_weight", 0.5)
    fl = np.where(fl_geo.notna(), wg * fl_geo + (1 - wg) * fl_trig, fl_trig)
    fl = pd.Series(fl, index=f.index).clip(*RANGES["fl_mm"])
    out = pd.DataFrame({"image_id": f.image_id, "pa_deg": pa, "fl_mm": fl, "mt_mm": mt})
    if groups is not None and cfg.get("temporal_alpha", 0) > 0:
        a = cfg["temporal_alpha"]
        for c in ("pa_deg", "fl_mm", "mt_mm"):
            med = out.groupby(groups)[c].transform("median")
            out[c] = (1 - a) * out[c] + a * med
    for c, (lo, hi) in RANGES.items():
        out[c] = out[c].clip(lo, hi).round(3)
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("features")
    a1.add_argument("--probs", required=True)
    a1.add_argument("--out", required=True)
    a2 = sub.add_parser("submit")
    a2.add_argument("--features", required=True)
    a2.add_argument("--config", required=True)
    a2.add_argument("--test-dir", default=None)
    a2.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "features":
        df = extract_features(Path(a.probs))
        df.to_csv(a.out, index=False)
        print(df.describe().T.to_string())
    else:
        feat = pd.read_csv(a.features)
        cfg = json.loads(Path(a.config).read_text())
        groups = video_groups(list(feat.image_id), Path(a.test_dir)) if a.test_dir else None
        sub_df = make_submission(feat, cfg, groups)
        sub_df.to_csv(a.out, index=False)
        print(sub_df.describe().T.to_string())


if __name__ == "__main__":
    main()
