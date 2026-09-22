"""Competition data contract: identifiers, target names, roots, submission validation/writing.

Facts verified from public competition code (Sept 2026):
  * id column           : StudyInstanceUID
  * sample_submission   : StudyInstanceUID + the 12 targets below, in this exact order
  * train.csv           : StudyInstanceUID + 12 targets (labels may be missing/NaN for some cells)
  * train_series.csv    : StudyInstanceUID, SeriesInstanceUID, Anatomical_Plane, Fluid_Sensitive, Fat_Suppression
  * test.csv/test_series.csv mirror the train files without labels; test_series/<study>/<series>/<sop>.dcm
  * radiology reports exist for TRAIN only (never at test time)
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

ID_COL = "StudyInstanceUID"
SERIES_COL = "SeriesInstanceUID"
COMPETITION = "rsna-knee-abnormality-detection"

TARGETS: list[str] = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
N_TARGETS = len(TARGETS)

SERIES_META_COLS = ["Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"]
PLANES = ["Sagittal", "Coronal", "Axial"]
PLANE_TO_IDX = {p: i for i, p in enumerate(PLANES)}
UNKNOWN_PLANE_IDX = len(PLANES)


def targets_hash(targets: Sequence[str] = TARGETS) -> str:
    return hashlib.sha256(json.dumps(list(targets)).encode()).hexdigest()[:16]


def find_competition_root(input_root: str | Path = "/kaggle/input", split: str = "test") -> Path:
    """Locate the competition mount. Works for /kaggle/input/<comp> and /kaggle/input/competitions/<comp>
    and for arbitrary local roots that directly contain the CSVs."""
    base = Path(input_root)
    marker = f"{split}.csv"
    candidates = [base, base / COMPETITION, base / "competitions" / COMPETITION]
    for c in candidates:
        if (c / marker).is_file():
            return c
    # last resort: shallow search
    for c in base.glob(f"*/{marker}"):
        return c.parent
    raise FileNotFoundError(f"Could not find {marker} under {base}")


def read_split(root: str | Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (studies_df, series_df) for split in {train, test}."""
    root = Path(root)
    studies = pd.read_csv(root / f"{split}.csv", encoding="utf-8-sig")
    series = pd.read_csv(root / f"{split}_series.csv", encoding="utf-8-sig")
    if ID_COL not in studies.columns:
        raise ValueError(f"{split}.csv missing {ID_COL}")
    missing = {ID_COL, SERIES_COL} - set(series.columns)
    if missing:
        raise ValueError(f"{split}_series.csv missing columns {missing}")
    studies[ID_COL] = studies[ID_COL].astype(str)
    series[ID_COL] = series[ID_COL].astype(str)
    series[SERIES_COL] = series[SERIES_COL].astype(str)
    return studies, series


def verify_sample_submission(root: str | Path) -> list[str]:
    """Assert the sample submission header equals the frozen schema; return the target order."""
    path = Path(root) / "sample_submission.csv"
    header = list(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)
    expected = [ID_COL] + TARGETS
    if header != expected:
        raise ValueError(f"sample_submission.csv header {header} != expected {expected}")
    return TARGETS


def train_prior(train_df: pd.DataFrame) -> np.ndarray:
    """Per-target positive prevalence on the labelled cells (used for studies with no usable images)."""
    prior = np.full(N_TARGETS, 0.5, dtype=np.float64)
    for i, t in enumerate(TARGETS):
        if t in train_df.columns:
            col = pd.to_numeric(train_df[t], errors="coerce")
            if col.notna().any():
                prior[i] = float(col.dropna().mean())
    return prior


def validate_submission(sub: pd.DataFrame, test_ids: Iterable[str]) -> None:
    """Raise if `sub` is not a valid competition submission for exactly `test_ids`."""
    expected_cols = [ID_COL] + TARGETS
    if list(sub.columns) != expected_cols:
        raise ValueError(f"Submission columns {list(sub.columns)} != {expected_cols}")
    ids = [str(x) for x in test_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("test ids contain duplicates")
    sub_ids = sub[ID_COL].astype(str).tolist()
    if len(sub_ids) != len(set(sub_ids)):
        raise ValueError("submission contains duplicate StudyInstanceUID")
    if set(sub_ids) != set(ids):
        missing = set(ids) - set(sub_ids)
        extra = set(sub_ids) - set(ids)
        raise ValueError(f"submission ids mismatch: missing={len(missing)} extra={len(extra)}")
    vals = sub[TARGETS].to_numpy(dtype=np.float64)
    if not np.isfinite(vals).all():
        raise ValueError("submission contains NaN/inf values")
    if (vals < 0).any() or (vals > 1).any():
        raise ValueError("submission values outside [0, 1]")


def make_submission(ids: Sequence[str], probs: np.ndarray) -> pd.DataFrame:
    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape != (len(ids), N_TARGETS):
        raise ValueError(f"probs shape {probs.shape} != ({len(ids)}, {N_TARGETS})")
    probs = np.clip(np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
    sub = pd.DataFrame(probs, columns=TARGETS)
    sub.insert(0, ID_COL, [str(x) for x in ids])
    return sub


def write_submission(sub: pd.DataFrame, path: str | Path, test_ids: Iterable[str] | None = None) -> Path:
    """Validate (against test ids if given) then write with full float precision, atomically."""
    if test_ids is not None:
        validate_submission(sub, test_ids)
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    sub.to_csv(tmp, index=False, float_format="%.10g")
    tmp.replace(path)
    return path
