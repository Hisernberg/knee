"""Offline (Kaggle) inference: stream test DICOMs, run every checkpoint, blend, write submission.csv.

Guarantees:
  * every test study gets a row (studies with no decodable image get the training prior)
  * checkpoint target order is re-mapped by name to the sample-submission order
  * a wall-clock budget stops starting new arms that cannot finish; the primary arm always completes
  * submission is validated against test ids before it is written
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import StudyDataset, collate_studies
from .metrics import prob_average, rank_average
from .models import load_checkpoint
from .preprocess import PrepConfig
from .schema import (ID_COL, TARGETS, find_competition_root, make_submission, read_split, train_prior,
                     verify_sample_submission, write_submission)


@dataclass
class InferConfig:
    input_root: str = "/kaggle/input"
    checkpoints: tuple[str, ...] = ()
    weights: tuple[float, ...] = ()  # per checkpoint; default equal
    out_csv: str = "/kaggle/working/submission.csv"
    tta_flip: bool = True
    blend: str = "prob"  # "prob" within a family; "rank" across heterogeneous families
    max_seconds: float = 8.0 * 3600  # leave margin under the 9h limit
    num_workers: int = 3
    batch_size: int = 1
    prior_csv: str | None = None  # train.csv to fit priors for empty studies (optional)
    limit: int | None = None  # debug: first N studies


def _probs_from_ckpt(model, ck, loader, device, tta_flip: bool):
    from .train import predict

    ids, probs = predict(model, loader, device, torch.float16, tta_flip=tta_flip)
    # re-map checkpoint target order to the competition order by name
    ck_targets = list(ck.get("targets", TARGETS))
    if ck_targets != TARGETS:
        if set(ck_targets) != set(TARGETS):
            raise ValueError(f"checkpoint targets {ck_targets} incompatible with schema {TARGETS}")
        perm = [ck_targets.index(t) for t in TARGETS]
        probs = probs[:, perm]
    return ids, probs


def run_inference(cfg: InferConfig) -> Path:
    t_start = time.time()
    root = find_competition_root(cfg.input_root, "test")
    verify_sample_submission(root)
    test_df, test_series = read_split(root, "test")
    if cfg.limit:
        test_df = test_df.head(cfg.limit)
    ids_all = test_df[ID_COL].astype(str).tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":  # probe the GPU with a real op (P100 + some torch builds report available but fail)
        try:
            torch.ones(2, device=device).sum().item()
        except Exception as e:  # pragma: no cover
            print(f"[infer] GPU probe failed ({e}); falling back to CPU", flush=True)
            device = torch.device("cpu")
    prior = np.full(len(TARGETS), 0.5)
    if cfg.prior_csv and Path(cfg.prior_csv).is_file():
        prior = train_prior(pd.read_csv(cfg.prior_csv))
    arms, weights, receipt = [], [], {"arms": [], "device": device.type, "n_test": len(ids_all)}
    per_arm_time = None
    for k, ck_path in enumerate(cfg.checkpoints):
        elapsed = time.time() - t_start
        if arms and per_arm_time is not None and elapsed + 1.15 * per_arm_time > cfg.max_seconds:
            receipt["arms"].append({"ckpt": ck_path, "status": "skipped_time_budget"})
            print(f"[infer] skipping {ck_path}: time budget", flush=True)
            continue
        t0 = time.time()
        model, ck = load_checkpoint(ck_path, map_location="cpu")
        model.to(device).eval()
        prep = PrepConfig(**ck["prep"]) if "prep" in ck else PrepConfig()
        ds = StudyDataset(test_df, root, prep, split="test", series=test_series, train=False)
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                        collate_fn=collate_studies)
        ids, probs = _probs_from_ckpt(model, ck, dl, device, cfg.tta_flip)
        order = pd.Series(range(len(ids)), index=ids).reindex(ids_all)
        if order.isna().any():
            raise RuntimeError("inference lost studies")
        probs = probs[order.to_numpy(dtype=int)]
        arms.append(probs)
        weights.append(cfg.weights[k] if k < len(cfg.weights) else 1.0)
        per_arm_time = time.time() - t0 if per_arm_time is None else max(per_arm_time, time.time() - t0)
        receipt["arms"].append({"ckpt": ck_path, "status": "ok", "seconds": round(time.time() - t0, 1),
                                "val_macro_auc": ck.get("val_macro_auc")})
        print(f"[infer] arm {k} done in {time.time() - t0:.0f}s", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not arms:
        raise RuntimeError("no checkpoint produced predictions")
    blended = rank_average(arms, weights) if cfg.blend == "rank" else prob_average(arms, weights)
    # studies without any usable window: fall back to the prior (ties are harmless for AUC)
    empty = _empty_studies(test_df, test_series, root)
    for i, sid in enumerate(ids_all):
        if sid in empty:
            blended[i] = prior
    sub = make_submission(ids_all, blended)
    out = write_submission(sub, cfg.out_csv, test_ids=ids_all)
    receipt.update({"seconds_total": round(time.time() - t_start, 1), "empty_studies": sorted(empty),
                    "config": {**asdict(cfg), "checkpoints": list(cfg.checkpoints), "weights": list(cfg.weights)}})
    Path(cfg.out_csv).with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=1))
    print(f"[infer] wrote {out} ({len(sub)} rows) in {time.time() - t_start:.0f}s", flush=True)
    return out


def _empty_studies(test_df: pd.DataFrame, test_series: pd.DataFrame, root: Path) -> set[str]:
    """Studies whose series directories are all missing (cheap check; decode failures are handled by the
    model's empty-batch path which already yields a constant)."""
    empty = set()
    grouped = test_series.groupby(ID_COL)
    for sid in test_df[ID_COL].astype(str):
        if sid not in grouped.groups:
            empty.add(sid)
            continue
        dirs = [root / "test_series" / sid / str(s) for s in grouped.get_group(sid)["SeriesInstanceUID"]]
        if not any(d.is_dir() and any(d.iterdir()) for d in dirs):
            empty.add(sid)
    return empty
