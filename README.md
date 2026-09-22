# RSNA Knee Abnormality Detection — `kneemri`

Study-level detection of twelve knee MRI findings (macro ROC-AUC) for the Kaggle code competition
[`rsna-knee-abnormality-detection`](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).
Weakly supervised (58 of 4,407 training studies are labelled; every study has a multilingual radiology
report), variable multi-series DICOM input, offline notebook submission.

* `docs/COMPETITION_ANALYSIS.md` — data facts, public-solution anatomy, leaderboard state, pitfalls.
* `docs/SOLUTION_DESIGN.md` — the top-5-targeted design and validation protocol.
* `kaggle/README.md` — how submissions work in this code competition and how to run the 5-variant loop.

## Status (2026-09-22)

| Deliverable | State |
|---|---|
| Training + inference pipeline (`src/kneemri`) | complete, 8 tests pass end-to-end on synthetic DICOM data (CPU) |
| Report → weak-label tooling (rules + LLM teacher, fold-safe calibration) | complete; rule labeler measured at 0.781 macro AUC vs gold-58 on the real reports |
| Kaggle notebook builder + 5-submission loop (`kaggle/`) | complete, dry-run tested locally |
| `best_filament_unet.pth` (trained weights) | **not produced**: no GPU and Kaggle is unreachable from the build sandbox |
| `submission.csv` on the leaderboard | **not submitted**: Kaggle API blocked (HTTP 403) from the sandbox |

The checkpoint name `best_filament_unet.pth` is kept as requested (`TrainConfig.ckpt_name`); the model is a
2.5D attention-MIL classifier, not a segmentation UNet, because the competition is study-level
classification scored by AUC.

## Quick start

```bash
pip install -r requirements.txt
export PYTHONPATH=src
python -m pytest tests -q                       # synthetic end-to-end smoke (~2 min on CPU)
```

Real data (place `train.csv`, `train_series.csv`, `train_series/`, `test.csv`, `test_series.csv`,
`sample_submission.csv` under `data/`; competition data must not be committed):

```bash
python -m kneemri folds --train-csv data/train.csv --out work/folds.csv --n-folds 5
python scripts/label_reports_llm.py submit --train-csv data/train.csv      # optional LLM teacher (Anthropic API)
python scripts/label_reports_llm.py collect --out work/llm_labels.jsonl
python scripts/make_report_labels.py --train-csv data/train.csv --folds work/folds.csv \
       --out work/report_labels --llm-json work/llm_labels.jsonl
python -m kneemri prepare --root data --split train --out work/cache_s384_c3_k20_n8_f140 \
       -o prep.size=384 -o prep.fov_mm=140 -o prep.lo_pct=2 -o prep.hi_pct=98 -o prep.span_lo=0.15 -o prep.span_hi=0.85
python -m kneemri train --config configs/arm_b_coatnet0_384.yaml -o fold=0      # repeat folds 0-4, arms A-D
python scripts/blend_oof.py --runs work/runs --train-csv data/train.csv
python kaggle/upload_weights.py --owner <kaggle-user> --runs work/runs
python kaggle/submit_loop.py --owner <kaggle-user> --all                        # 5 notebook submissions
```
`scripts/run_all.sh` chains all of the above.

Local inference / validation without Kaggle:

```bash
python -m kneemri infer --input-root data --ckpt work/runs/arm_b_coatnet0_384/fold0/best_filament_unet.pth \
       --out submission.csv --prior-csv data/train.csv
python -m kneemri validate --submission submission.csv --test-csv data/test.csv
```

## Repository layout

```
src/kneemri/      schema (contract + submission validation), dicom, preprocess, data, folds, models,
                  train, infer, metrics, reports (multilingual weak labels), synthetic (test data), cli
configs/          debug.yaml + four pipeline-diverse arms (ConvNeXt-T 320, CoAtNet-0 384/140 mm,
                  EffNetV2-S 5-slice ASL, MaxViT-T 384/140 mm)
scripts/          run_all.sh, make_report_labels.py, label_reports_llm.py, blend_oof.py
kaggle/           build_notebook.py, variants.yaml (5 submissions), submit_loop.py, upload_weights.py
tests/            schema/metrics, DICOM ordering, end-to-end synthetic pipeline, notebook build+exec, reports
docs/             competition analysis and solution design
```

## Design in one paragraph

DICOM series are decoded once per study, ordered by patient-position projection, normalised per series
(2–98 %), cropped to a fixed 140 mm field of view and resampled; each series contributes up to K 3-slice
windows from the central depth span. A timm encoder (CoAtNet/ConvNeXt/EfficientNet/MaxViT) embeds every
window; protocol embeddings (plane, fluid-sensitive, fat-sat, depth, series slot) are added; an optional
within-series BiGRU and a cross-series transformer mix context; twelve learned finding queries pool the
study with masked attention into twelve logits. Training uses masked, confidence-weighted BCE on gold
labels plus fold-calibrated report labels, fp16 AMP, EMA and cosine schedule. Folds and arms are blended
by equal-weight average ranks; the offline notebook re-validates the submission against `test.csv`.
