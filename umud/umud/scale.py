"""Pixel-to-millimetre scale and field-of-view detection for the UMUD test devices.

The per-device rules follow the tick-mark parser published by AmbrosM
("UMUD Quick and Dirty", Kaggle, 2026), re-implemented with graceful fallbacks
instead of assertions so that unseen devices degrade to an estimate rather than
crashing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_RIGHT_TICK_FOV = {
    7: (142, 91, 1058), 8: (163, 91, 1037), 9: (211, 91, 989), 10: (249, 91, 951),
    11: (282, 91, 918), 12: (308, 91, 892), 13: (331, 91, 869), 14: (349, 91, 851),
    15: (142, 91, 1058),
}


@dataclass
class Scale:
    px_per_mm: float
    l: int
    t: int
    r: int
    b: int
    family: str

    @property
    def mm_per_px(self) -> float:
        return 1.0 / self.px_per_mm


def _col(a: np.ndarray, x: int) -> np.ndarray:
    c = a[:, x]
    return c.mean(axis=-1) if c.ndim == 2 else c.astype(float)


def detect_scale(a: np.ndarray, suffix: str) -> Scale:
    """Return scale and the bounding box (l, t, r, b) of the B-mode image area."""
    h, w = a.shape[:2]
    suffix = suffix.lower().lstrip(".")
    rgb = a if a.ndim == 3 else np.repeat(a[..., None], 3, axis=2)
    try:
        if suffix == "png":  # Philips Lumify
            c6, c9 = _col(rgb, 6), _col(rgb, 9)
            first = int(np.argmax(c6 > 50))
            sec_minor = 150 + int(np.argmax(c6[150:] > 50))
            sec_major = 150 + int(np.argmax(c9[150:] > 50))
            last = h - 1 - int(np.argmax(c6[::-1] > 50))
            if (sec_major - first) < 3 * (sec_minor - first):
                px_cm = float(sec_major - first)
            else:
                px_cm = float(sec_minor - first)
            hw = w // 2
            w2 = int(np.argmin(rgb[:, hw:].sum(axis=(0, 2))))
            return Scale(px_cm / 10, hw - w2, first, hw + w2, last, "lumify_png")
        if (h, w) == (800, 1200):
            if (rgb[87, 1147:1157] == 175).all() or rgb[87, 1147:1157].mean() > 150:
                col = _col(rgb, 1150)
                first = int(np.argmax(col > 50))
                sec = first + 20 + int(np.argmax(col[first + 20:] > 50))
                last = h - 1 - int(np.argmax(col[::-1] > 50))
                n = int(round((last - first) / max(sec - first, 1)))
                px_cm = (last - first) / n * 2 if n <= 14 else (last - first) / 3
                l, t, r = _RIGHT_TICK_FOV.get(n, (142, 91, 1058))
                return Scale(px_cm / 10, l, t, r, last, f"juniper_right_{n}")
            if (rgb[42, 67:74, 0] > 115).all():
                return Scale((783 - 42) / 50, 171, 42, 1029, 798, "left_ticks_800")
        if (h, w) == (644, 1088):
            return Scale(630.5 / 50, 140, 0, 947, 643, "telemed_644")
        if h in (512, 513):
            return Scale((442 - 53) / 50, 0, 0, w, h - 10, "h512")
        if h == 853:
            g = a if a.ndim == 2 else a.mean(axis=-1)
            if g[-5, 44] == 170 and g[-5, 879] == 170:
                return Scale((879 - 44) / 50, 0, 0, w, h, "gray853_b")
            return Scale((934 - 100) / 50, 0, 0, w, h, "gray853_a")
    except Exception:  # pragma: no cover - defensive fallback
        pass
    return Scale(14.82, 0, 0, w, h, "unknown")
