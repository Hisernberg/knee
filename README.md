# kraggle-3-in-one — Kaggle competition workspace

One repository, one self-contained folder per competition. Code, configs, docs and submissions of a
competition live only inside its folder; nothing is shared across folders.

| Folder | Competition | Codename / package |
|---|---|---|
| repo root: `src/kneemri`, `configs/`, `docs/`, `kaggle/`, `scripts/`, `tests/` | [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) | `kneemri` (section below) |
| [`umud/`](umud/README.md) | [UMUD Challenge: Muscle Architecture in Ultrasound Data](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data) | `umud` (results: [`umud/RESULTS.md`](umud/RESULTS.md)) |
| `trafficflow/` | traffic-flow forecasting competition | `trafficflow` |
| [`cobalt_heron/`](cobalt_heron/README.md) | filament-segmentation-2026 | `ch` |

Rules: never commit `kaggle.json`, API keys or competition data; Kaggle kernels are prefixed by their
project (`umud-*`, `rsna-knee-*`, `ch-*`, `tfb-*`) so account-level kernel lists stay separable.

---

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
| Account baseline | public LB **0.942 (rank 304)**: `rsna-knee-dinosaur-v5` v1 (fork of the public 0.943 *Speedy Raptors* stack + a weak ConvNeXt arm); v2-v5 crashed on the hidden set (fatal asserts added by the experimental *MAST* rewrite) |
| Extended-Raptor submission notebook (`kaggle/ext/`) | **built and verified locally**: 0.943 anchor unchanged + two unused public Raptor views (v9 finespacing 0.932 solo, v4 widedense 0.927 solo) + five blend variants from one GPU run |
| Kaggle run | **blocked until Saturday 2026-09-26 00:00 UTC**: the account's 30 h weekly GPU quota is exhausted (`kaggle kernels push` refused). Runbook: `docs/EXT_SUBMISSION_PLAN.md` section 5 |
| Expected result | anchor 0.943; main variant 0.943-0.945 (rank ~90-150). Rank 1-3 (0.956+) is not reachable with public assets; see `docs/EXT_SUBMISSION_PLAN.md` |
| Training + inference pipeline (`src/kneemri`) | complete, 8 tests pass end-to-end on synthetic DICOM data (CPU); no trained weights |

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
