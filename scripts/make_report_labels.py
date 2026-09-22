#!/usr/bin/env python3
"""Build fold-safe weak-label tables from radiology reports (train only).

  python scripts/make_report_labels.py --train-csv data/train.csv --folds work/folds.csv --out work/report_labels
  [--llm-json work/llm_labels.jsonl]   # optional: merge LLM teacher output (scripts/label_reports_llm.py)

Writes work/report_labels/states.csv (tri-states) and report_labels_fold{k}.csv (soft labels + __w weights
calibrated on the gold rows OUTSIDE fold k), for TrainConfig.aux_labels_csv = "work/report_labels/report_labels_fold{fold}.csv".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kneemri.reports import calibrate_states, label_reports  # noqa: E402
from kneemri.schema import ID_COL, TARGETS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--out", default="work/report_labels")
    ap.add_argument("--llm-json", default=None, help="jsonl with {StudyInstanceUID, labels: {target: -1|0|1}} rows")
    ap.add_argument("--unk-weight", type=float, default=0.25)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(a.train_csv)
    train[ID_COL] = train[ID_COL].astype(str)
    states = label_reports(train, "Report")
    if a.llm_json:  # LLM states override rule states where present
        llm = {}
        for line in Path(a.llm_json).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                llm[str(r[ID_COL])] = r["labels"]
        n = 0
        for i, sid in enumerate(states[ID_COL]):
            if sid in llm:
                for t in TARGETS:
                    if t in llm[sid]:
                        states.loc[i, t] = int(llm[sid][t])
                n += 1
        print(f"merged LLM labels for {n} studies")
    states.to_csv(out / "states.csv", index=False)
    folds = pd.read_csv(a.folds)
    folds[ID_COL] = folds[ID_COL].astype(str)
    gold = train[[ID_COL] + TARGETS]
    for k in sorted(folds["fold"].unique()):
        fit_ids = set(folds.loc[folds.fold != k, ID_COL])
        cal = calibrate_states(states, gold, fit_ids, unk_weight=a.unk_weight)
        cal.to_csv(out / f"report_labels_fold{k}.csv", index=False)
        print(f"fold {k}: calibration", json.dumps({t: {str(s): round(v, 3) for s, v in d.items()}
                                                    for t, d in cal.attrs["calibration"].items()}))
    summary = pd.DataFrame({t: states[t].value_counts().reindex([1, 0, -1]).fillna(0).astype(int) for t in TARGETS}).T
    summary.columns = ["pos", "unmentioned", "neg"]
    print(summary.to_string())


if __name__ == "__main__":
    main()
