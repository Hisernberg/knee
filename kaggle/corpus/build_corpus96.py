#!/usr/bin/env python3
"""Kaggle CPU kernel: pre-decode RSNA Knee training studies into a dense 96-slice, 336 px, 140 mm corpus.

Geometry follows the strongest public Raptor family (dreaddevelopment): five anatomical slots filled from
the study's series (fluid-sensitive preference per slot), slices sampled evenly across 2-98 % of each
series, per-slot 2-98 percentile normalisation over the sampled slices, 140 mm centre crop using the DICOM
pixel spacing, resize to 336 px, uint8. Slot widths 26/22/18/12/18 = 96 (v9 finespacing scaled to 96).

Runs on a CPU-only Kaggle session (no GPU quota). Studies are split into N_PARTS contiguous parts so each
part's output (~10.8 MB/study) stays under the 20 GB notebook-output limit. Outputs per part:
  vols_part{K}.npy   (n, 96, 336, 336) uint8      masks_part{K}.npy (n, 96) uint8
  ids_part{K}.npy    (n,) StudyInstanceUID         meta_part{K}.json  chosen series per slot, failures, timing
Set PART / N_PARTS through environment variables (defaults 0 / 3) or edit the constants below.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

PART = int(os.environ.get("CORPUS_PART", "0"))
N_PARTS = int(os.environ.get("CORPUS_N_PARTS", "3"))
IMG = 336
CROP_MM = 140.0
SPAN = (0.02, 0.98)
SLOTS = [("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18), ("Coronal", 0, 12), ("Axial", -1, 18)]
MAXS = sum(s[2] for s in SLOTS)
WORKERS = max(1, min(4, os.cpu_count() or 1))
TIME_BUDGET_S = float(os.environ.get("CORPUS_BUDGET_S", str(11.2 * 3600)))
LIMIT = int(os.environ.get("CORPUS_LIMIT", "0"))  # >0: only this many studies (smoke test)


def find_root() -> Path:
    for p in ("/kaggle/input/competitions/rsna-knee-abnormality-detection", "/kaggle/input/rsna-knee-abnormality-detection"):
        if Path(p, "train_series.csv").is_file():
            return Path(p)
    env = os.environ.get("RSNA_ROOT")
    if env and Path(env, "train_series.csv").is_file():
        return Path(env)
    raise FileNotFoundError("competition data not mounted")


def order_series(sdir: Path):
    """(files sorted along the slice normal, per-file spacing, median spacing)."""
    import pydicom

    recs, spacings = [], []
    for f in sorted(sdir.glob("*.dcm")):
        try:
            h = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            iop = getattr(h, "ImageOrientationPatient", None)
            ipp = getattr(h, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                r, c = np.asarray(iop[:3], float), np.asarray(iop[3:], float)
                pos = float(np.dot(np.asarray(ipp[:3], float), np.cross(r, c)))
            else:
                pos = float(getattr(h, "InstanceNumber", 0) or 0)
            ps = getattr(h, "PixelSpacing", None)
            ps = float(ps[0]) if ps is not None else 0.0
            if ps > 0:
                spacings.append(ps)
            recs.append((pos, str(f), ps))
        except Exception:
            recs.append((0.0, str(f), 0.0))
    recs.sort(key=lambda t: t[0])
    med = float(np.median(spacings)) if spacings else 0.5
    return [(f, ps if ps > 0 else med) for _, f, ps in recs], med


def read_px(path: str) -> np.ndarray:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut

    d = pydicom.dcmread(path, force=True)
    a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if a.ndim == 3:  # multi-frame or RGB: take the first frame / channel
        a = a[0] if a.shape[0] < a.shape[-1] else a[..., 0]
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def crop_resize(a: np.ndarray, ps: float) -> np.ndarray:
    import cv2

    h, w = a.shape
    c = int(round(CROP_MM / max(ps, 1e-3)))
    c = max(16, min(c, h, w))
    y0, x0 = (h - c) // 2, (w - c) // 2
    return cv2.resize(a[y0:y0 + c, x0:x0 + c], (IMG, IMG), interpolation=cv2.INTER_AREA)


def pick(rows, plane, fluid, used):
    cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def build_study(args):
    uid, rows, series_root = args
    t0 = time.time()
    vol = np.zeros((MAXS, IMG, IMG), np.uint8)
    mask = np.zeros(MAXS, np.uint8)
    chosen, used, idx, failures = {}, set(), 0, []
    for plane, fluid, k in SLOTS:
        r = pick(rows, plane, fluid, used)
        if r is None:
            idx += k
            continue
        sid = r["SeriesInstanceUID"]
        used.add(sid)
        try:
            files, med = order_series(Path(series_root) / uid / sid)
            n = len(files)
            if n == 0:
                idx += k
                continue
            lo = int(n * SPAN[0])
            hi = max(int(n * SPAN[1]) - 1, lo)
            picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else np.zeros(k, int)
            arrs, pss = [], []
            for p in picks:
                f, ps = files[min(int(p), n - 1)]
                try:
                    arrs.append(read_px(f))
                    pss.append(ps)
                except Exception as exc:  # noqa: BLE001
                    arrs.append(None)
                    pss.append(med)
                    failures.append(f"{sid}:{Path(f).name}:{type(exc).__name__}")
            valid = [a for a in arrs if a is not None]
            if valid:
                allpx = np.concatenate([a.ravel() for a in valid])
                loq, hiq = np.percentile(allpx, [2.0, 98.0])
            else:
                loq, hiq = 0.0, 1.0
            for a, ps in zip(arrs, pss):
                if a is not None:
                    aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
                    vol[idx] = (crop_resize(aw, ps) * 255).astype(np.uint8)
                    mask[idx] = 1
                idx += 1
            chosen[f"{plane}_{fluid}_{k}"] = {"series": sid, "n_files": n, "median_spacing_mm": med}
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{sid}:series:{type(exc).__name__}:{str(exc)[:120]}")
            idx += k
    return uid, vol, mask, {"slots": chosen, "failures": failures, "seconds": round(time.time() - t0, 2)}


def main():
    t0 = time.time()
    root = find_root()
    out_dir = Path(os.environ.get("CORPUS_OUT", "/kaggle/working"))
    out_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(root / "train.csv", dtype={"StudyInstanceUID": str}, usecols=["StudyInstanceUID"])
    series = pd.read_csv(root / "train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    uids = sorted(train["StudyInstanceUID"].unique().tolist())
    per_part = -(-len(uids) // N_PARTS)
    part_uids = uids[PART * per_part:(PART + 1) * per_part]
    if LIMIT:
        part_uids = part_uids[:LIMIT]
    rows = {k: g.to_dict("records") for k, g in series.groupby("StudyInstanceUID")}
    series_root = root / "train_series"
    n = len(part_uids)
    print(f"part {PART}/{N_PARTS}: {n} studies | workers {WORKERS} | out {out_dir}", flush=True)
    vols = np.lib.format.open_memmap(out_dir / f"vols_part{PART}.npy", mode="w+", dtype=np.uint8, shape=(n, MAXS, IMG, IMG))
    masks = np.zeros((n, MAXS), np.uint8)
    meta = {"geometry": {"slots": SLOTS, "span": SPAN, "crop_mm": CROP_MM, "img": IMG, "norm": "per-slot 2-98 pct"},
            "part": PART, "n_parts": N_PARTS, "studies": {}, "incomplete": []}
    pos = {u: i for i, u in enumerate(part_uids)}
    done = 0
    with ProcessPoolExecutor(WORKERS) as ex:
        futures = {ex.submit(build_study, (u, rows.get(u, []), str(series_root))): u for u in part_uids}
        for fut in as_completed(futures):
            u = futures[fut]
            try:
                uid, vol, mask, info = fut.result()
                vols[pos[uid]] = vol
                masks[pos[uid]] = mask
                meta["studies"][uid] = info
            except Exception as exc:  # noqa: BLE001
                meta["incomplete"].append({"uid": u, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            done += 1
            if done % 50 == 0 or done == n:
                el = time.time() - t0
                print(f"{done}/{n} | {el / 60:.1f} min | {el / done:.2f} s/study | eta {(n - done) * el / done / 60:.1f} min", flush=True)
                vols.flush()
            if time.time() - t0 > TIME_BUDGET_S:
                print("time budget reached; stopping early (remaining studies marked incomplete)", flush=True)
                for f2, u2 in futures.items():
                    if not f2.done():
                        f2.cancel()
                        meta["incomplete"].append({"uid": u2, "error": "time budget"})
                break
    vols.flush()
    del vols
    np.save(out_dir / f"masks_part{PART}.npy", masks)
    np.save(out_dir / f"ids_part{PART}.npy", np.asarray(part_uids))
    meta["elapsed_seconds"] = round(time.time() - t0, 1)
    meta["n_complete"] = len(meta["studies"])
    (out_dir / f"meta_part{PART}.json").write_text(json.dumps(meta, indent=1))
    print(f"DONE part {PART}: {meta['n_complete']}/{n} studies, {meta['elapsed_seconds'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
