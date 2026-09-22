"""Muscle-architecture geometry from aponeurosis / fascicle probability maps.

All inputs are on the B-mode crop (see scale.detect_scale). Every function returns
several estimator variants (e.g. MT between inner edges vs. band centrelines) so
that the final choice can be validated rather than hard-coded.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


# --------------------------------------------------------------------------- aponeuroses
def _band_profiles(mask: np.ndarray, prob: np.ndarray, lab: np.ndarray, k: int):
    """Per-column upper edge, lower edge and prob-weighted centre of component k."""
    ys, xs = np.where(lab == k)
    W = mask.shape[1]
    top = np.full(W, np.nan)
    bot = np.full(W, np.nan)
    cen = np.full(W, np.nan)
    order = np.argsort(xs, kind="stable")
    xs, ys = xs[order], ys[order]
    bounds = np.flatnonzero(np.diff(xs)) + 1
    for seg_x, seg_y in zip(np.split(xs, bounds), np.split(ys, bounds)):
        x = seg_x[0]
        top[x] = seg_y.min()
        bot[x] = seg_y.max()
        w = prob[seg_y, x]
        cen[x] = float((seg_y * w).sum() / max(w.sum(), 1e-6))
    return top, bot, cen


def _fit(x: np.ndarray, y: np.ndarray, deg: int = 1):
    ok = np.isfinite(y)
    if ok.sum() < 10:
        return None
    xs, ys = x[ok], y[ok]
    # robust: two passes dropping the worst 10% residuals
    c = np.polyfit(xs, ys, deg)
    for _ in range(2):
        r = np.abs(np.polyval(c, xs) - ys)
        keep = r <= np.percentile(r, 90) + 1e-6
        if keep.sum() >= 10:
            c = np.polyfit(xs[keep], ys[keep], deg)
    return c


def find_aponeuroses(apo_prob: np.ndarray, px_per_mm: float, thr: float = 0.5):
    """Return dict with superficial/deep line fits (edges + centre) or None."""
    H, W = apo_prob.shape
    m = (apo_prob >= thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    cands = []
    for k in range(1, n):
        span = st[k, cv2.CC_STAT_WIDTH]
        if span < 0.2 * W or st[k, cv2.CC_STAT_AREA] < 0.002 * H * W:
            continue
        cands.append(k)
    if len(cands) < 2:
        return None
    x = np.arange(W, dtype=float)
    bands = []
    for k in cands:
        top, bot, cenl = _band_profiles(m, apo_prob, lab, k)
        c_top, c_bot, c_cen = _fit(x, top), _fit(x, bot), _fit(x, cenl)
        if c_cen is None:
            continue
        bands.append(dict(k=k, top=c_top, bot=c_bot, cen=c_cen, span=st[k, cv2.CC_STAT_WIDTH],
                          ymid=float(np.polyval(c_cen, W / 2)),
                          mass=float(apo_prob[lab == k].sum())))
    bands.sort(key=lambda b: b["ymid"])
    if len(bands) < 2:
        return None
    # superficial: upper-most substantial band; deep: strongest band >= 5 mm below it
    sup = bands[0]
    below = [b for b in bands[1:] if b["ymid"] - sup["ymid"] >= 5 * px_per_mm]
    if not below:
        return None
    deep = max(below, key=lambda b: b["span"] * 1.0 + 0.0 * b["mass"])
    return dict(sup=sup, deep=deep, n_bands=len(bands), W=W, H=H)


def _perp_dist(c_deep, c_sup, xs):
    """Distance from points on the deep line to the superficial line, measured perpendicular."""
    out = []
    a, b = c_sup[0], c_sup[1]  # y = a x + b
    for x0 in xs:
        y0 = np.polyval(c_deep, x0)
        out.append(abs(a * x0 - y0 + b) / math.sqrt(a * a + 1))
    return np.asarray(out)


def muscle_thickness(apo: dict, px_per_mm: float) -> dict:
    W = apo["W"]
    xs = np.array([0.25, 0.5, 0.75]) * W
    s, d = apo["sup"], apo["deep"]
    res = {}
    for name, cs, cd in (("inner", s["bot"], d["top"]), ("center", s["cen"], d["cen"]), ("outer", s["top"], d["bot"])):
        if cs is None or cd is None:
            res[f"mt_{name}"] = np.nan
            continue
        res[f"mt_{name}"] = float(_perp_dist(cd, cs, xs).mean() / px_per_mm)
        res[f"mt_{name}_vert"] = float(np.mean(np.polyval(cd, xs) - np.polyval(cs, xs)) / px_per_mm)
    res["sup_thick"] = float(np.mean(np.polyval(s["bot"], xs) - np.polyval(s["top"], xs)) / px_per_mm) if s["bot"] is not None and s["top"] is not None else np.nan
    res["deep_thick"] = float(np.mean(np.polyval(d["bot"], xs) - np.polyval(d["top"], xs)) / px_per_mm) if d["bot"] is not None and d["top"] is not None else np.nan
    res["deep_angle"] = float(math.degrees(math.atan(d["cen"][0])))
    res["sup_angle"] = float(math.degrees(math.atan(s["cen"][0])))
    return res


# --------------------------------------------------------------------------- fascicles
def _angle_between(a_deg: float, b_deg: float) -> float:
    d = abs(a_deg - b_deg) % 180
    return min(d, 180 - d)


def _intersect(m, c, line):
    """Intersection x of y = m x + c with y = line[0] x + line[1]."""
    den = m - line[0]
    if abs(den) < 1e-9:
        return None
    return (line[1] - c) / den


def fascicles(fasc_prob: np.ndarray, apo: dict, px_per_mm: float, thr: float = 0.35,
              min_len_mm: float = 3.0) -> dict:
    H, W = fasc_prob.shape
    s, d = apo["sup"], apo["deep"]
    xs = np.arange(W)
    y_sup = np.polyval(s["bot"] if s["bot"] is not None else s["cen"], xs)
    y_deep = np.polyval(d["top"] if d["top"] is not None else d["cen"], xs)
    yy = np.arange(H)[:, None]
    margin = 0.5 * px_per_mm
    region = (yy > y_sup[None] + margin) & (yy < y_deep[None] - margin)
    p = np.where(region, fasc_prob, 0).astype(np.float32)
    m = (p >= thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    deep_ang = math.degrees(math.atan(d["cen"][0]))
    sup_line = s["bot"] if s["bot"] is not None else s["cen"]
    deep_line = d["top"] if d["top"] is not None else d["cen"]
    recs = []
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < 10:
            continue
        ys, xk = np.where(lab == k)
        pts = np.stack([xk, ys], 1).astype(float)
        w = p[ys, xk]
        mu = (pts * w[:, None]).sum(0) / w.sum()
        q = pts - mu
        cov = (q * w[:, None]).T @ q / w.sum()
        ev, vec = np.linalg.eigh(cov)
        v = vec[:, -1]
        length = 4 * math.sqrt(max(ev[-1], 0))  # ~ extent of a uniform segment
        if length < min_len_mm * px_per_mm:
            continue
        elong = math.sqrt(max(ev[-1], 1e-9) / max(ev[0], 1e-9))
        if elong < 3:
            continue
        ang = math.degrees(math.atan2(v[1], v[0]))
        pa = _angle_between(ang, deep_ang)
        if not 2 <= pa <= 60:
            continue
        slope = v[1] / v[0] if abs(v[0]) > 1e-9 else 1e9
        c0 = mu[1] - slope * mu[0]
        xu, xl = _intersect(slope, c0, sup_line), _intersect(slope, c0, deep_line)
        fl = np.nan
        if xu is not None and xl is not None:
            yu, yl = slope * xu + c0, slope * xl + c0
            fl = math.hypot(xu - xl, yu - yl) / px_per_mm
        recs.append((pa, fl, length, float(w.mean()), ang))
    out = dict(n_fasc=len(recs))
    if not recs:
        return out
    R = np.array(recs)
    pa, fl, ln, conf = R[:, 0], R[:, 1], R[:, 2], R[:, 3]
    wt = ln * conf
    # direction sanity: fascicles should share one orientation sign relative to the deep apo
    rel = ((R[:, 4] - deep_ang + 90) % 180) - 90
    sign = np.sign(np.average(np.sign(rel), weights=wt))
    keep = np.sign(rel) == sign if sign != 0 else np.ones(len(R), bool)
    if keep.sum() >= 1:
        pa, fl, ln, wt = pa[keep], fl[keep], ln[keep], wt[keep]
    out["pa_med"] = float(np.median(pa))
    out["pa_wmean"] = float(np.average(pa, weights=wt))
    out["pa_wmed"] = _wmedian(pa, wt)
    top = np.argsort(-wt)[:5]
    out["pa_top5"] = float(np.median(pa[top]))
    ok = np.isfinite(fl) & (fl > 10) & (fl < 300)
    if ok.any():
        out["fl_med"] = float(np.median(fl[ok]))
        out["fl_wmed"] = _wmedian(fl[ok], wt[ok])
        t5 = [i for i in top if ok[i]]
        out["fl_top5"] = float(np.median(fl[t5])) if t5 else np.nan
    out["n_fasc_kept"] = int(len(pa))
    out["fasc_sign"] = float(sign)
    return out


def _wmedian(v, w):
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return float(v[o][np.searchsorted(cw, cw[-1] / 2)])


def orientation_pa(fasc_prob: np.ndarray, apo: dict, px_per_mm: float) -> float:
    """PA from the dominant structure-tensor orientation of the fascicle map inside the muscle."""
    H, W = fasc_prob.shape
    s, d = apo["sup"], apo["deep"]
    xs = np.arange(W)
    y_sup = np.polyval(s["cen"], xs)
    y_deep = np.polyval(d["cen"], xs)
    yy = np.arange(H)[:, None]
    region = (yy > y_sup[None] + px_per_mm) & (yy < y_deep[None] - px_per_mm)
    f = fasc_prob.astype(np.float32) / 255.0 if fasc_prob.dtype == np.uint8 else fasc_prob.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    sig = max(1.0, 1.5 * px_per_mm)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sig)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sig)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sig)
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2  # line direction
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)
    w = (coh * f)[region]
    if w.sum() <= 0:
        return np.nan
    ang = np.degrees(theta[region])
    deep_ang = math.degrees(math.atan(d["cen"][0]))
    rel = ((ang - deep_ang + 90) % 180) - 90
    hist, edges = np.histogram(rel, bins=180, range=(-90, 90), weights=w)
    hist = np.convolve(hist, np.ones(5) / 5, mode="same")
    i = int(np.argmax(hist))
    return float(abs(0.5 * (edges[i] + edges[i + 1])))


def analyse(apo_prob: np.ndarray, fasc_prob: np.ndarray, px_per_mm: float) -> dict:
    ap = apo_prob.astype(np.float32) / 255.0 if apo_prob.dtype == np.uint8 else apo_prob
    fp = fasc_prob.astype(np.float32) / 255.0 if fasc_prob.dtype == np.uint8 else fasc_prob
    apo = find_aponeuroses(ap, px_per_mm)
    if apo is None:
        apo = find_aponeuroses(ap, px_per_mm, thr=0.3)
    if apo is None:
        return dict(ok=0)
    out = dict(ok=1, n_bands=apo["n_bands"])
    out.update(muscle_thickness(apo, px_per_mm))
    out.update(fascicles(fp, apo, px_per_mm))
    out["pa_orient"] = orientation_pa(fp, apo, px_per_mm)
    return out
