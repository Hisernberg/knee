"""Robust DICOM series reading for knee MRI.

Design (from audits of public pipelines):
  * order slices by projecting ImagePositionPatient onto the slice normal (from ImageOrientationPatient);
    fall back to InstanceNumber, then filename
  * apply RescaleSlope/Intercept, invert MONOCHROME1
  * per-series percentile normalisation (robust to scanner/vendor intensity scales)
  * keep row/column PixelSpacing separately; explicit policy when spacing is missing
  * never crash a study: unusable files are skipped and reported; an empty series returns None
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:  # optional at import time so that pure-numpy consumers can import the module
    import pydicom
    from pydicom.errors import InvalidDicomError
except Exception:  # pragma: no cover
    pydicom = None
    InvalidDicomError = Exception

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*Invalid value for VR.*")


@dataclass
class SeriesVolume:
    """A decoded, ordered, normalised series: `pixels` is uint8 [N, H, W]."""

    pixels: np.ndarray
    spacing: tuple[float, float]  # (row, col) in mm
    slice_spacing: float
    n_files: int
    n_decoded: int
    order_method: str
    events: list[str] = field(default_factory=list)

    @property
    def n_slices(self) -> int:
        return int(self.pixels.shape[0])


def _to_float_list(v, n):
    try:
        out = [float(x) for x in v]
        return out if len(out) == n else None
    except Exception:
        return None


def _pixel_array(ds) -> np.ndarray | None:
    try:
        arr = ds.pixel_array
    except Exception as e:  # decoder missing / corrupt pixel data
        log.debug("pixel decode failed: %s", e)
        return None
    if arr.ndim == 3:
        # multi-frame single file or RGB; keep grayscale by averaging channels if colour, else first frame set
        if arr.shape[-1] in (3, 4) and getattr(ds, "SamplesPerPixel", 1) > 1:
            arr = arr[..., :3].mean(-1)
        else:
            return None  # multi-frame volumes are not expected in this dataset; handled upstream
    arr = arr.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    if slope != 1.0 or intercept != 0.0:
        arr = arr * slope + intercept
    if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).upper() == "MONOCHROME1":
        arr = arr.max() - arr
    return arr


def normalise_volume(vol: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    """Percentile clip per series then scale to uint8 [0,255]."""
    vol = vol.astype(np.float32)
    lo, hi = np.percentile(vol, [lo_pct, hi_pct])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(vol.min()), float(vol.max())
        if hi <= lo:
            return np.zeros(vol.shape, dtype=np.uint8)
    vol = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
    return (vol * 255.0 + 0.5).astype(np.uint8)


def read_series(series_dir: str | Path, normalise: bool = True) -> SeriesVolume | None:
    """Decode every .dcm in `series_dir` into an ordered volume. Returns None if nothing decodable."""
    if pydicom is None:
        raise RuntimeError("pydicom is required to read DICOM series")
    series_dir = Path(series_dir)
    files = sorted(p for p in series_dir.iterdir() if p.is_file() and not p.name.startswith("."))
    slices, keys_pos, keys_inst, events = [], [], [], []
    spacing_rc: tuple[float, float] | None = None
    normal = None
    shape_ref = None
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), force=True)
        except (InvalidDicomError, Exception) as e:
            events.append(f"unreadable:{f.name}:{type(e).__name__}")
            continue
        arr = _pixel_array(ds)
        if arr is None:
            events.append(f"undecodable:{f.name}")
            continue
        if shape_ref is None:
            shape_ref = arr.shape
        elif arr.shape != shape_ref:
            events.append(f"shape_mismatch:{f.name}:{arr.shape}")
            continue
        ipp = _to_float_list(getattr(ds, "ImagePositionPatient", None) or [], 3)
        iop = _to_float_list(getattr(ds, "ImageOrientationPatient", None) or [], 6)
        if normal is None and iop is not None:
            r, c = np.array(iop[:3]), np.array(iop[3:])
            normal = np.cross(r, c)
        pos_key = float(np.dot(normal, ipp)) if (normal is not None and ipp is not None) else None
        inst = getattr(ds, "InstanceNumber", None)
        inst_key = float(inst) if inst is not None and str(inst).strip() != "" else None
        if spacing_rc is None:
            ps = _to_float_list(getattr(ds, "PixelSpacing", None) or [], 2)
            if ps is not None and all(v > 0 for v in ps):
                spacing_rc = (ps[0], ps[1])
        slices.append(arr)
        keys_pos.append(pos_key)
        keys_inst.append(inst_key)
    if not slices:
        return None
    n = len(slices)
    if all(k is not None for k in keys_pos) and len(set(keys_pos)) == n:
        order = np.argsort(np.array(keys_pos, dtype=np.float64), kind="stable")
        method = "position"
    elif all(k is not None for k in keys_inst) and len(set(keys_inst)) == n:
        order = np.argsort(np.array(keys_inst, dtype=np.float64), kind="stable")
        method = "instance_number"
    else:
        order = np.arange(n)
        method = "filename"
    vol = np.stack([slices[i] for i in order], axis=0)
    slice_spacing = 0.0
    if method == "position" and n > 1:
        ks = np.sort(np.array(keys_pos, dtype=np.float64))
        slice_spacing = float(np.median(np.diff(ks)))
    if spacing_rc is None:
        events.append("missing_pixel_spacing:assume_isotropic_1mm")
        spacing_rc = (1.0, 1.0)
    if normalise:
        vol = normalise_volume(vol)
    return SeriesVolume(vol, spacing_rc, slice_spacing, len(files), n, method, events)


def resize_physical_square(img: np.ndarray, spacing: tuple[float, float], size: int,
                           fov_mm: float | None = None) -> np.ndarray:
    """Resample a 2D slice to `size`x`size` while preserving physical aspect ratio.

    If `fov_mm` is given, a centred square field of view of that size (mm) is cropped/padded first,
    so anatomy has a consistent physical scale across scanners. Otherwise the whole image is used
    with aspect-preserving padding.
    """
    import cv2

    h, w = img.shape[:2]
    sr, sc = spacing
    ph, pw = h * sr, w * sc
    if fov_mm is None:
        fov = max(ph, pw)
    else:
        fov = float(fov_mm)
    # target pixel extents of the square FOV in the source image
    hh = int(round(fov / sr))
    ww = int(round(fov / sc))
    # crop or pad to (hh, ww) centred
    def _crop_pad(a: np.ndarray, th: int, tw: int) -> np.ndarray:
        H, W = a.shape[:2]
        top = max((H - th) // 2, 0)
        left = max((W - tw) // 2, 0)
        a = a[top : top + th, left : left + tw]
        H2, W2 = a.shape[:2]
        pad_t, pad_l = (th - H2) // 2, (tw - W2) // 2
        pad_b, pad_r = th - H2 - pad_t, tw - W2 - pad_l
        if pad_t or pad_b or pad_l or pad_r:
            a = np.pad(a, ((pad_t, pad_b), (pad_l, pad_r)), mode="constant")
        return a

    sq = _crop_pad(img, hh, ww)
    interp = cv2.INTER_AREA if sq.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(sq, (size, size), interpolation=interp)
