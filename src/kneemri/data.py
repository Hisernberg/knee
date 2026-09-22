"""Datasets and collate for variable-series, variable-window studies."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocess import PrepConfig, StudyWindows, prepare_study
from .schema import ID_COL, N_TARGETS, TARGETS

IMAGENET_MEAN = 0.45
IMAGENET_STD = 0.225


def _augment(tiles: np.ndarray, rng: random.Random, p_flip: float = 0.5) -> np.ndarray:
    """Cheap, label-preserving augmentations on uint8 [W, C, H, W]. Horizontal flip converts a left knee
    into a plausible right knee (medial/lateral semantics preserved anatomically), so it is safe."""
    import cv2

    W, C, H, Wd = tiles.shape
    out = tiles
    if rng.random() < p_flip:
        out = out[..., ::-1]
    # global affine: rotation +-10deg, scale 0.9-1.1, translation +-6%
    if rng.random() < 0.8:
        ang = rng.uniform(-10, 10)
        sc = rng.uniform(0.9, 1.1)
        tx, ty = rng.uniform(-0.06, 0.06) * Wd, rng.uniform(-0.06, 0.06) * H
        M = cv2.getRotationMatrix2D((Wd / 2, H / 2), ang, sc)
        M[:, 2] += (tx, ty)
        flat = np.ascontiguousarray(out).reshape(W * C, H, Wd)
        flat = np.stack([cv2.warpAffine(f, M, (Wd, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                         for f in flat])
        out = flat.reshape(W, C, H, Wd)
    # brightness / contrast jitter
    if rng.random() < 0.5:
        a, b = rng.uniform(0.85, 1.15), rng.uniform(-15, 15)
        out = np.clip(out.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)


class StudyDataset(Dataset):
    """Yields one study per item: windows + per-window protocol metadata + (masked) labels.

    `source` is either a cache directory of .npz files (train) or a competition root for streaming (test).
    """

    def __init__(self, studies: pd.DataFrame, source: str | Path, cfg: PrepConfig, split: str = "train",
                 series: pd.DataFrame | None = None, train: bool = False, max_windows: int | None = None,
                 series_dropout: float = 0.0, seed: int = 0):
        self.df = studies.reset_index(drop=True)
        self.ids = self.df[ID_COL].astype(str).tolist()
        self.source = Path(source)
        self.cfg = cfg
        self.split = split
        self.series = series
        self.train = train
        self.max_windows = max_windows
        self.series_dropout = series_dropout
        self.seed = seed
        self.has_labels = all(t in self.df.columns for t in TARGETS)
        self.streaming = not (self.source / f"{self.ids[0]}.npz").is_file() if self.ids else False

    def __len__(self) -> int:
        return len(self.ids)

    def _load(self, i: int) -> StudyWindows:
        sid = self.ids[i]
        p = self.source / f"{sid}.npz"
        if p.is_file():
            return StudyWindows.load(p)
        if self.series is None:
            raise FileNotFoundError(f"no cache for {sid} and no series metadata for streaming")
        return prepare_study(self.source, self.split, sid, self.series, self.cfg)

    def __getitem__(self, i: int) -> dict:
        sw = self._load(i)
        rng = random.Random(self.seed + i + (random.randint(0, 10**9) if self.train else 0))
        tiles, sidx, depth = sw.tiles, sw.series_idx.astype(np.int64), sw.depth
        plane, fluid, fat = sw.plane.astype(np.int64), sw.fluid.astype(np.int64), sw.fat.astype(np.int64)
        if self.train and self.series_dropout > 0 and len(plane) > 1:
            keep = np.array([rng.random() >= self.series_dropout for _ in range(len(plane))])
            if not keep.any():
                keep[rng.randrange(len(plane))] = True
            wkeep = keep[sidx]
            tiles, sidx, depth = tiles[wkeep], sidx[wkeep], depth[wkeep]
        if self.max_windows and tiles.shape[0] > self.max_windows:
            sel = np.sort(np.array(rng.sample(range(tiles.shape[0]), self.max_windows))) if self.train else \
                np.round(np.linspace(0, tiles.shape[0] - 1, self.max_windows)).astype(int)
            tiles, sidx, depth = tiles[sel], sidx[sel], depth[sel]
        if self.train and tiles.shape[0] > 0:
            tiles = _augment(tiles, rng)
        x = torch.from_numpy(np.ascontiguousarray(tiles)).float().div_(255.0).sub_(IMAGENET_MEAN).div_(IMAGENET_STD)
        item = {
            "id": self.ids[i],
            "tiles": x,  # [W, C, H, W]
            "series_idx": torch.from_numpy(sidx),
            "depth": torch.from_numpy(depth.astype(np.float32)),
            "plane": torch.from_numpy(plane[sidx]) if len(sidx) else torch.zeros(0, dtype=torch.long),
            "fluid": torch.from_numpy(fluid[sidx]) if len(sidx) else torch.zeros(0, dtype=torch.long),
            "fat": torch.from_numpy(fat[sidx]) if len(sidx) else torch.zeros(0, dtype=torch.long),
            "n_events": len(sw.events),
        }
        if self.has_labels:
            y = self.df.loc[i, TARGETS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            item["labels"] = torch.from_numpy(y)
        else:
            item["labels"] = torch.full((N_TARGETS,), float("nan"))
        return item


def collate_studies(items: list[dict]) -> dict:
    """Pad variable window counts: tiles [B, Wmax, C, H, W] + mask [B, Wmax] (True = real window)."""
    B = len(items)
    wmax = max(1, max(it["tiles"].shape[0] for it in items))
    c, h, w = items[0]["tiles"].shape[1:] if items[0]["tiles"].shape[0] else (3, 1, 1)
    for it in items:
        if it["tiles"].shape[0]:
            c, h, w = it["tiles"].shape[1:]
            break
    tiles = torch.zeros(B, wmax, c, h, w)
    mask = torch.zeros(B, wmax, dtype=torch.bool)
    series_idx = torch.zeros(B, wmax, dtype=torch.long)
    depth = torch.zeros(B, wmax)
    plane = torch.zeros(B, wmax, dtype=torch.long)
    fluid = torch.zeros(B, wmax, dtype=torch.long)
    fat = torch.zeros(B, wmax, dtype=torch.long)
    labels = torch.stack([it["labels"] for it in items])
    for b, it in enumerate(items):
        n = it["tiles"].shape[0]
        if n == 0:
            continue
        tiles[b, :n] = it["tiles"]
        mask[b, :n] = True
        series_idx[b, :n] = it["series_idx"]
        depth[b, :n] = it["depth"]
        plane[b, :n] = it["plane"]
        fluid[b, :n] = it["fluid"]
        fat[b, :n] = it["fat"]
    return {"ids": [it["id"] for it in items], "tiles": tiles, "mask": mask, "series_idx": series_idx,
            "depth": depth, "plane": plane, "fluid": fluid, "fat": fat, "labels": labels}
