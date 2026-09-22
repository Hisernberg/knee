#!/usr/bin/env python3
"""Evaluate arms and blends on out-of-fold predictions (macro AUC), to choose the submission variants.

  python scripts/blend_oof.py --runs work/runs --train-csv data/train.csv
Reports per-arm OOF macro AUC (all folds concatenated), equal-weight prob/rank blends of all arms, and a
leave-one-out table. Fitted per-target weights are deliberately NOT searched: public OOF audits at n~4.3k
measured them at -0.0008 vs equal weights (overfit), while adding a decorrelated family gave +0.001.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kneemri.metrics import macro_auc, per_target_auc, prob_average, rank_average  # noqa: E402
from kneemri.schema import ID_COL, TARGETS  # noqa: E402


def load_arm(run_dir: Path) -> pd.DataFrame | None:
    parts = [pd.read_csv(p) for p in sorted(run_dir.glob("fold*/oof.csv"))]
    if not parts:
        return None
    df = pd.concat(parts).drop_duplicates(ID_COL).set_index(ID_COL)
    return df[TARGETS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="work/runs")
    ap.add_argument("--train-csv", required=True)
    a = ap.parse_args()
    train = pd.read_csv(a.train_csv)
    train[ID_COL] = train[ID_COL].astype(str)
    y_all = train.set_index(ID_COL)[TARGETS].apply(pd.to_numeric, errors="coerce")
    arms = {d.name: load_arm(d) for d in sorted(Path(a.runs).iterdir()) if d.is_dir()}
    arms = {k: v for k, v in arms.items() if v is not None}
    if not arms:
        raise SystemExit("no oof.csv found")
    common = sorted(set.intersection(*(set(v.index) for v in arms.values())))
    y = y_all.loc[common].to_numpy(float)
    preds = {k: v.loc[common].to_numpy(float) for k, v in arms.items()}
    report = {"n_studies": len(common), "arms": {}, "blends": {}}
    for k, p in preds.items():
        report["arms"][k] = {"macro_auc": macro_auc(y, p), "per_target": per_target_auc(y, p)}
    names = list(preds)
    P = [preds[k] for k in names]
    report["blends"]["all_prob"] = macro_auc(y, prob_average(P))
    report["blends"]["all_rank"] = macro_auc(y, rank_average(P))
    for k in names:
        rest = [preds[j] for j in names if j != k]
        if rest:
            report["blends"][f"without_{k}_rank"] = macro_auc(y, rank_average(rest))
    print(json.dumps(report, indent=1, default=float))
    Path(a.runs, "oof_blend_report.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
