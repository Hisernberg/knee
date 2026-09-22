#!/usr/bin/env bash
# Full offline recipe on a GPU box with the competition data at $DATA (train.csv, train_series.csv, train_series/).
# 1) prepare caches  2) folds  3) train 4 arms x 5 folds  4) evaluate OOF blends  5) upload weights  6) submit loop
set -euo pipefail
DATA=${DATA:-data}
export PYTHONPATH=src
python -m kneemri folds --train-csv "$DATA/train.csv" --out work/folds.csv --n-folds 5
python scripts/make_report_labels.py --train-csv "$DATA/train.csv" --folds work/folds.csv --out work/report_labels ${LLM_JSON:+--llm-json "$LLM_JSON"}
python -m kneemri prepare --root "$DATA" --split train --out work/cache_s320_c3_k24_n8 \
  -o prep.size=320 -o prep.ctx=3 -o prep.max_centers=24 -o prep.max_series=8 --workers "${WORKERS:-8}"
python -m kneemri prepare --root "$DATA" --split train --out work/cache_s384_c3_k20_n8_f140 \
  -o prep.size=384 -o prep.ctx=3 -o prep.max_centers=20 -o prep.max_series=8 -o prep.fov_mm=140 -o prep.lo_pct=2 -o prep.hi_pct=98 -o prep.span_lo=0.15 -o prep.span_hi=0.85 --workers "${WORKERS:-8}"
python -m kneemri prepare --root "$DATA" --split train --out work/cache_s320_c5_k24_n8 \
  -o prep.size=320 -o prep.ctx=5 -o prep.max_centers=24 -o prep.max_series=8 --workers "${WORKERS:-8}"
for cfg in configs/arm_a_convnext_tiny_320.yaml configs/arm_b_coatnet0_384.yaml \
           configs/arm_c_effnetv2s_320_asl.yaml configs/arm_d_maxvit_tiny_384.yaml; do
  for fold in 0 1 2 3 4; do
    python -m kneemri train --config "$cfg" -o fold=$fold -o train_csv="$DATA/train.csv"
  done
done
python scripts/blend_oof.py --runs work/runs --train-csv "$DATA/train.csv"
python kaggle/upload_weights.py --owner "${KAGGLE_OWNER:?set KAGGLE_OWNER}" --runs work/runs
python kaggle/submit_loop.py --owner "$KAGGLE_OWNER" --all --max-concurrent 2
