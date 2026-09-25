# UMUD — muscle architecture from ultrasound (`umud/`)

Pipeline for the Kaggle competition
[UMUD Challenge: Muscle Architecture in Ultrasound Data](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data):
predict pennation angle (PA, °), fascicle length (FL, mm) and muscle thickness (MT, mm) per image.

Metric: `mean(MAE_PA/6, MAE_FL/12, MAE_MT/3)`, lower is better.

## Status (2026-09-25)
Best public LB **0.32182 (rank 4/293)**: `submissions/d4_S2_mt10.csv`. Full log with every submission: [`RESULTS.md`](RESULTS.md).
Reproduce it with `scripts/blend_v2.py --w 0.56 0.27 1.0 --clip-mt 4`, using the pipeline predictions in
`submissions/s17_Ahyb_pipeline.csv` and the public Vera CSV as reference. The pipeline alone (`configs/a_raw.json`, re-runnable
end-to-end via `kaggle/build_kernel.py --mode submit`) is the prize-eligible part. Blends with the hard-coded Vera CSV are not.

Layout: `umud/` package (scale, seg, geometry, predict) · `scripts/` (blends, OSF benchmark, diagnostics) ·
`configs/` · `kaggle/` (kernel builder) · `submissions/` (every submitted CSV) · `RESULTS.md`.

## Method
1. **Scale / FOV** (`umud/scale.py`): per-device tick-mark parser → px/mm and B-mode crop box.
2. **Segmentation** (`umud/seg.py`): two U-Nets (ResNet-34 encoder, ImageNet init) trained on the
   provided aponeurosis (1,048) and fascicle (2,761) masks; letterboxed 512×768 input, hflip TTA.
   Masks stored inverted in the training set are auto-corrected.
3. **Geometry** (`umud/geometry.py`): aponeurosis bands → robust line fits (upper edge, lower edge,
   centre); MT = perpendicular inner-edge distance averaged at 25/50/75 % of the width; fascicle
   fragments → weighted PCA lines → PA vs. the deep aponeurosis and FL by linear extrapolation to both
   aponeuroses (length-weighted median); structure-tensor PA as a second estimator.
4. **Submission** (`umud/predict.py`): documented config (`configs/*.json`) chooses estimators,
   blends geometric and trigonometric FL (MT / sin PA), applies cine-loop temporal smoothing
   (the test set has 28 five-frame clips) and clips to the physiological ranges given by the hosts.

## Run
```bash
pip install -r umud/requirements.txt
# GPU training + test inference (≈1 h on a Kaggle T4). Or build a Kaggle kernel:
python umud/kaggle/build_kernel.py --user <kaggle-user> --out build/kernel && kaggle kernels push -p build/kernel
python umud/umud/seg.py --data <competition dir> --out work/seg
cd umud && python -m umud.predict features --probs ../work/seg --out ../work/features.csv
python -m umud.predict submit --features ../work/features.csv --config configs/final.json \
    --test-dir <competition dir>/test_images_v2/test_set_v2 --out ../work/submission.csv
```
Never commit `kaggle.json` or competition data.
