"""Study preparation: DICOM series -> 2.5D context windows at a fixed resolution.

Each series contributes up to `max_centers` window centres spread uniformly over its slices; each window
stacks `ctx` adjacent slices (edge-clamped) as channels. Series boundaries are explicit: windows never
cross series. Per-window metadata (series index, plane, fluid-sensitive, fat-suppressed, relative depth)
is retained for the model's geometry/protocol embeddings and attention.

Train: prepare once to an on-disk cache (uint8 .npz per study, small and fast to read).
Test : the same function is used for streaming inference (no cache).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .dicom import normalise_volume, read_series, resize_physical_square
from .schema import ID_COL, PLANE_TO_IDX, SERIES_COL, UNKNOWN_PLANE_IDX

log = logging.getLogger(__name__)


@dataclass
class PrepConfig:
    size: int = 320
    ctx: int = 3  # adjacent slices per window (odd)
    max_centers: int = 24  # windows per series (uniform over depth)
    max_series: int = 8  # series per study kept (priority ordering below)
    fov_mm: float | None = None  # None: aspect-preserving full FOV; e.g. 140.0 for a fixed physical FOV
    lo_pct: float = 0.5  # per-series intensity percentile clip (public strong recipe: 2 / 98)
    hi_pct: float = 99.5
    span_lo: float = 0.0  # fraction of the series depth to sample centres from (public strong recipe: 0.15-0.85)
    span_hi: float = 1.0

    def key(self) -> str:
        return (f"s{self.size}_c{self.ctx}_k{self.max_centers}_n{self.max_series}_f{self.fov_mm or 0}"
                f"_p{self.lo_pct}-{self.hi_pct}_z{self.span_lo}-{self.span_hi}")


@dataclass
class StudyWindows:
    study_id: str
    tiles: np.ndarray  # uint8 [W, ctx, size, size]
    series_idx: np.ndarray  # int16 [W] index into series metadata arrays (0..S-1)
    depth: np.ndarray  # float32 [W] centre position in (0,1)
    plane: np.ndarray  # int8 [S]
    fluid: np.ndarray  # int8 [S]
    fat: np.ndarray  # int8 [S]
    n_slices: np.ndarray  # int16 [S]
    events: list[str] = field(default_factory=list)

    @property
    def n_windows(self) -> int:
        return int(self.tiles.shape[0])

    def save(self, path: Path) -> None:
        np.savez(path, tiles=self.tiles, series_idx=self.series_idx, depth=self.depth, plane=self.plane,
                 fluid=self.fluid, fat=self.fat, n_slices=self.n_slices,
                 events=np.array(self.events, dtype=object), study_id=np.array(self.study_id))

    @classmethod
    def load(cls, path: Path) -> StudyWindows:
        z = np.load(path, allow_pickle=True)
        return cls(str(z["study_id"]), z["tiles"], z["series_idx"], z["depth"], z["plane"], z["fluid"],
                   z["fat"], z["n_slices"], list(z["events"]))


def _centers(n: int, k: int, span_lo: float = 0.0, span_hi: float = 1.0) -> np.ndarray:
    """k centre indices spread uniformly over the [span_lo, span_hi] fraction of n slices."""
    lo, hi = span_lo * (n - 1), span_hi * (n - 1)
    n_in = int(round(hi - lo)) + 1
    if n_in <= k:
        return np.unique(np.round(np.linspace(lo, hi, max(n_in, 1))).astype(int))
    return np.round(np.linspace(lo, hi, k)).astype(int)


def windows_from_volume(vol: np.ndarray, centers: np.ndarray, ctx: int) -> np.ndarray:
    n = vol.shape[0]
    half = ctx // 2
    idx = np.clip(centers[:, None] + np.arange(-half, half + 1)[None, :], 0, n - 1)
    return vol[idx]  # [W, ctx, H, W]


def series_priority(meta: pd.DataFrame) -> pd.DataFrame:
    """Order series so the most diagnostically useful come first when a study exceeds max_series:
    fluid-sensitive fat-suppressed sagittal/coronal first, then others; stable within groups."""
    m = meta.copy()
    plane = m["Anatomical_Plane"].astype(str).str.strip().str.title()
    fluid = pd.to_numeric(m.get("Fluid_Sensitive", 0), errors="coerce").fillna(0).astype(int)
    fat = pd.to_numeric(m.get("Fat_Suppression", 0), errors="coerce").fillna(0).astype(int)
    plane_rank = plane.map({"Sagittal": 0, "Coronal": 1, "Axial": 2}).fillna(3).astype(int)
    m["_prio"] = plane_rank * 10 - fluid * 4 - fat * 2
    m["_plane_idx"] = plane.map(PLANE_TO_IDX).fillna(UNKNOWN_PLANE_IDX).astype(int)
    m["_fluid"], m["_fat"] = fluid, fat
    return m.sort_values(["_prio", SERIES_COL], kind="stable").reset_index(drop=True)


def prepare_study(root: Path, split: str, study_id: str, series_meta: pd.DataFrame, cfg: PrepConfig,
                  rng: np.random.Generator | None = None) -> StudyWindows:
    """Decode all (available) series of a study into windows. Never raises for bad series; records events."""
    meta = series_priority(series_meta[series_meta[ID_COL] == study_id])
    tiles, sidx, depth, plane, fluid, fat, nsl, events = [], [], [], [], [], [], [], []
    s_out = 0
    for _, row in meta.iterrows():
        if s_out >= cfg.max_series:
            events.append(f"series_dropped_over_max:{row[SERIES_COL]}")
            continue
        sdir = root / f"{split}_series" / study_id / str(row[SERIES_COL])
        if not sdir.is_dir():
            events.append(f"series_dir_missing:{row[SERIES_COL]}")
            continue
        try:
            vol = read_series(sdir, normalise=False)
        except Exception as e:  # pragma: no cover - defensive
            events.append(f"series_read_error:{row[SERIES_COL]}:{type(e).__name__}")
            vol = None
        if vol is None or vol.n_slices == 0:
            events.append(f"series_unusable:{row[SERIES_COL]}")
            continue
        events.extend(f"{row[SERIES_COL]}:{e}" for e in vol.events)
        pixels = normalise_volume(vol.pixels, cfg.lo_pct, cfg.hi_pct)
        px = np.stack([resize_physical_square(s, vol.spacing, cfg.size, cfg.fov_mm) for s in pixels])
        n = px.shape[0]
        centers = _centers(n, cfg.max_centers, cfg.span_lo, cfg.span_hi)
        if rng is not None and n > cfg.max_centers:  # train-time jitter of centre positions
            step = max((n - 1) / max(cfg.max_centers - 1, 1), 1.0)
            centers = np.clip(centers + rng.integers(-int(step // 2), int(step // 2) + 1, len(centers)), 0, n - 1)
        tiles.append(windows_from_volume(px, centers, cfg.ctx))
        sidx.append(np.full(len(centers), s_out, dtype=np.int16))
        depth.append(((centers + 0.5) / n).astype(np.float32))
        plane.append(int(row["_plane_idx"]))
        fluid.append(int(row["_fluid"]))
        fat.append(int(row["_fat"]))
        nsl.append(n)
        s_out += 1
    if not tiles:
        return StudyWindows(study_id, np.zeros((0, cfg.ctx, cfg.size, cfg.size), np.uint8),
                            np.zeros(0, np.int16), np.zeros(0, np.float32), np.zeros(0, np.int8),
                            np.zeros(0, np.int8), np.zeros(0, np.int8), np.zeros(0, np.int16), events)
    return StudyWindows(study_id, np.concatenate(tiles), np.concatenate(sidx), np.concatenate(depth),
                        np.array(plane, np.int8), np.array(fluid, np.int8), np.array(fat, np.int8),
                        np.array(nsl, np.int16), events)


def prepare_split(root: str | Path, split: str, out_dir: str | Path, cfg: PrepConfig,
                  study_ids: list[str] | None = None, workers: int = 1, overwrite: bool = False) -> pd.DataFrame:
    """Prepare every study of a split to `out_dir/<study>.npz`; returns a manifest DataFrame."""
    from concurrent.futures import ProcessPoolExecutor

    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    studies = pd.read_csv(root / f"{split}.csv", encoding="utf-8-sig")
    series = pd.read_csv(root / f"{split}_series.csv", encoding="utf-8-sig")
    studies[ID_COL] = studies[ID_COL].astype(str)
    series[ID_COL] = series[ID_COL].astype(str)
    ids = [str(s) for s in (study_ids or studies[ID_COL].tolist())]
    todo = [s for s in ids if overwrite or not (out_dir / f"{s}.npz").is_file()]
    args = [(root, split, s, series[series[ID_COL] == s], cfg, out_dir) for s in todo]
    if workers > 1 and args:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_prepare_one, args, chunksize=4))
    else:
        for a in args:
            _prepare_one(a)
    rows = []
    for s in ids:
        p = out_dir / f"{s}.npz"
        if p.is_file():
            z = np.load(p, allow_pickle=True)
            rows.append({ID_COL: s, "n_windows": int(z["tiles"].shape[0]), "n_series": int(len(z["plane"])),
                         "n_events": int(len(z["events"]))})
        else:
            rows.append({ID_COL: s, "n_windows": 0, "n_series": 0, "n_events": -1})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    return manifest


def _prepare_one(a):
    root, split, s, meta, cfg, out_dir = a
    sw = prepare_study(root, split, s, meta, cfg)
    sw.save(out_dir / f"{s}.npz")
    return s
