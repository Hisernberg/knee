"""Command line entry point: python -m kneemri <command> [options]."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import fields
from pathlib import Path

import pandas as pd
import yaml

from .folds import fold_audit, stratified_group_kfold
from .infer import InferConfig, run_inference
from .models import ModelConfig
from .preprocess import PrepConfig, prepare_split
from .schema import ID_COL, TARGETS, validate_submission
from .train import TrainConfig, train_fold


def _load_yaml(path: str | None) -> dict:
    if not path:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _apply_overrides(d: dict, overrides: list[str]) -> dict:
    for ov in overrides:
        k, v = ov.split("=", 1)
        cur = d
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = yaml.safe_load(v)
    return d


def _build_train_cfg(d: dict) -> TrainConfig:
    model = ModelConfig(**d.pop("model", {}))
    prep = PrepConfig(**d.pop("prep", {}))
    known = {f.name for f in fields(TrainConfig)}
    return TrainConfig(model=model, prep=prep, **{k: v for k, v in d.items() if k in known})


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="kneemri")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="decode DICOMs to a window cache")
    p.add_argument("--root", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--out", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("-o", "--override", action="append", default=[])

    p = sub.add_parser("folds", help="write stratified group folds")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--group-col", default=None)

    p = sub.add_parser("train", help="train one fold")
    p.add_argument("--config", default=None)
    p.add_argument("-o", "--override", action="append", default=[])

    p = sub.add_parser("infer", help="predict the test split and write submission.csv")
    p.add_argument("--input-root", default="/kaggle/input")
    p.add_argument("--ckpt", action="append", required=True)
    p.add_argument("--weight", action="append", type=float, default=[])
    p.add_argument("--out", default="/kaggle/working/submission.csv")
    p.add_argument("--blend", default="prob", choices=["prob", "rank"])
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--max-seconds", type=float, default=8 * 3600)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--prior-csv", default=None)
    p.add_argument("--limit", type=int, default=None)

    p = sub.add_parser("validate", help="validate a submission.csv against test.csv")
    p.add_argument("--submission", required=True)
    p.add_argument("--test-csv", required=True)

    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        d = _apply_overrides(_load_yaml(a.config), a.override)
        prep = PrepConfig(**d.get("prep", {}))
        ids = None
        if a.limit:
            ids = pd.read_csv(Path(a.root) / f"{a.split}.csv")[ID_COL].astype(str).head(a.limit).tolist()
        m = prepare_split(a.root, a.split, a.out, prep, study_ids=ids, workers=a.workers)
        print(m.describe(include="all").to_string())
    elif a.cmd == "folds":
        df = pd.read_csv(a.train_csv)
        df[ID_COL] = df[ID_COL].astype(str)
        folds = stratified_group_kfold(df, a.n_folds, a.seed, a.group_col)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        folds.to_csv(a.out, index=False)
        audit = fold_audit(df, folds)
        audit.to_csv(Path(a.out).with_name("fold_audit.csv"), index=False)
        print(audit[["fold", "n"] + [f"{t}:pos" for t in TARGETS]].to_string(index=False))
    elif a.cmd == "train":
        d = _apply_overrides(_load_yaml(a.config), a.override)
        res = train_fold(_build_train_cfg(d))
        print(json.dumps({"best_val_macro_auc": res["best_val_macro_auc"], "ckpt": res["ckpt"]}))
    elif a.cmd == "infer":
        cfg = InferConfig(input_root=a.input_root, checkpoints=tuple(a.ckpt), weights=tuple(a.weight),
                          out_csv=a.out, tta_flip=not a.no_tta, blend=a.blend, max_seconds=a.max_seconds,
                          num_workers=a.workers, prior_csv=a.prior_csv, limit=a.limit)
        run_inference(cfg)
    elif a.cmd == "validate":
        sub_df = pd.read_csv(a.submission)
        ids = pd.read_csv(a.test_csv)[ID_COL].astype(str).tolist()
        validate_submission(sub_df, ids)
        print(f"OK: {len(sub_df)} rows, columns = {list(sub_df.columns)}")


if __name__ == "__main__":
    main()
