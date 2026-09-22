"""Synthetic competition directory generator (for CPU tests and notebook dry-runs).

Creates the exact on-disk layout of the real competition:
  root/train.csv, train_series.csv, train_series/<study>/<series>/<sop>.dcm,
  root/test.csv,  test_series.csv,  test_series/<study>/<series>/<sop>.dcm, sample_submission.csv
Images are synthetic knee-like phantoms whose lesion brightness/size correlates with the labels, so a
model can demonstrably learn (AUC > 0.5) on the fake data. Never used for real training.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import ID_COL, SERIES_COL, TARGETS

PLANE_CHOICES = ["Sagittal", "Coronal", "Axial"]


def _write_dicom(path: Path, pixels: np.ndarray, study_uid: str, series_uid: str, sop_uid: str,
                 instance: int, plane: str, spacing=(0.4, 0.4), thickness=3.0, monochrome1=False,
                 drop_position=False):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = pydicom.uid.MRImageStorage
    ds.SOPInstanceUID = generate_uid()
    ds.StudyInstanceUID = _uid_for(study_uid)
    ds.SeriesInstanceUID = _uid_for(series_uid)
    ds.Modality = "MR"
    ds.PatientID = "anon"
    ds.InstanceNumber = instance
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME1" if monochrome1 else "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.PixelSpacing = [str(spacing[0]), str(spacing[1])]
    ds.SliceThickness = str(thickness)
    ds.SpacingBetweenSlices = str(thickness)
    ds.RescaleSlope, ds.RescaleIntercept = 1, 0
    orient = {"Sagittal": [0, 1, 0, 0, 0, -1], "Coronal": [1, 0, 0, 0, 0, -1], "Axial": [1, 0, 0, 0, 1, 0]}[plane]
    ds.ImageOrientationPatient = [str(v) for v in orient]
    if not drop_position:
        normal = np.cross(orient[:3], orient[3:])
        pos = normal * instance * thickness
        ds.ImagePositionPatient = [f"{v:.3f}" for v in pos]
    px = pixels.astype(np.uint16)
    if monochrome1:
        px = px.max() - px
    ds.PixelData = px.tobytes()
    try:
        ds.save_as(str(path), enforce_file_format=True)  # pydicom >= 3
    except TypeError:  # pragma: no cover - pydicom 2.x
        ds.is_little_endian, ds.is_implicit_VR = True, False
        ds.save_as(str(path), write_like_original=False)


def _uid_for(name: str) -> str:
    """Deterministic, VR-UI-valid UID derived from a readable name."""
    import hashlib

    digest = int(hashlib.md5(name.encode()).hexdigest()[:20], 16)
    return f"1.2.826.0.1.3680043.8.498.{digest}"[:64]


def _phantom(size: int, n_slices: int, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Knee-like phantom: two 'bones' + joint space; lesions appear as bright blobs proportional to labels."""
    yy, xx = np.mgrid[0:size, 0:size]
    base = np.zeros((n_slices, size, size), dtype=np.float32)
    cy, cx = size / 2, size / 2
    for z in range(n_slices):
        zf = (z - n_slices / 2) / max(n_slices, 1)
        femur = ((yy - cy + size * 0.25) ** 2 + (xx - cx) ** 2) < (size * 0.22 * (1 - zf**2)) ** 2
        tibia = ((yy - cy - size * 0.25) ** 2 + (xx - cx) ** 2) < (size * 0.22 * (1 - zf**2)) ** 2
        img = 900 * femur.astype(np.float32) + 850 * tibia.astype(np.float32) + 300
        # per-label lesion blob at a label-specific location; brightness scales with label value
        for j, v in enumerate(labels):
            if v > 0.5:
                ang = 2 * np.pi * j / len(labels)
                ly, lx = cy + size * 0.32 * np.sin(ang), cx + size * 0.32 * np.cos(ang)
                r = size * (0.04 + 0.03 * rng.random())
                blob = ((yy - ly) ** 2 + (xx - lx) ** 2) < r**2
                img += 1500 * blob * (0.6 + 0.4 * rng.random())
        img += rng.normal(0, 60, img.shape).astype(np.float32)
        base[z] = np.clip(img, 0, 4000)
    return base


def make_synthetic_competition(root: str | Path, n_train: int = 24, n_test: int = 8, size: int = 64,
                               seed: int = 0, slices=(6, 12), series_per_study=(2, 4),
                               missing_label_frac: float = 0.15) -> Path:
    root = Path(root)
    rng = np.random.default_rng(seed)
    random.seed(seed)
    prevalence = np.linspace(0.15, 0.55, len(TARGETS))

    def build(split: str, n: int, labelled: bool):
        rows, srows = [], []
        for i in range(n):
            study = f"{split}_study_{i:04d}"
            labels = (rng.random(len(TARGETS)) < prevalence).astype(np.float32)
            ns = rng.integers(series_per_study[0], series_per_study[1] + 1)
            for s in range(ns):
                series = f"{study}_series_{s}"
                plane = PLANE_CHOICES[s % 3]
                fluid = int(rng.random() < 0.6)
                fat = int(rng.random() < 0.5)
                nz = int(rng.integers(slices[0], slices[1] + 1))
                vol = _phantom(size, nz, labels, rng)
                sdir = root / f"{split}_series" / study / series
                sdir.mkdir(parents=True, exist_ok=True)
                for z in range(nz):
                    _write_dicom(sdir / f"{series}_{z:03d}.dcm", vol[z], study, series, f"{series}_{z}",
                                 instance=z + 1, plane=plane, monochrome1=(s == 1 and i % 5 == 0),
                                 drop_position=(s == 2 and i % 7 == 0))
                srows.append({ID_COL: study, SERIES_COL: series, "Anatomical_Plane": plane,
                              "Fluid_Sensitive": fluid, "Fat_Suppression": fat})
            row = {ID_COL: study}
            if labelled:
                for j, t in enumerate(TARGETS):
                    row[t] = float(labels[j]) if rng.random() > missing_label_frac else np.nan
            rows.append(row)
        return pd.DataFrame(rows), pd.DataFrame(srows)

    tr, trs = build("train", n_train, True)
    te, tes = build("test", n_test, False)
    root.mkdir(parents=True, exist_ok=True)
    tr.to_csv(root / "train.csv", index=False)
    trs.to_csv(root / "train_series.csv", index=False)
    te[[ID_COL]].to_csv(root / "test.csv", index=False)
    tes.to_csv(root / "test_series.csv", index=False)
    sample = te[[ID_COL]].copy()
    for t in TARGETS:
        sample[t] = 0.5
    sample.to_csv(root / "sample_submission.csv", index=False)
    return root
