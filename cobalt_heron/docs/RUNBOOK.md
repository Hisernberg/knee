# Runbook — cobalt-heron

All commands are run from `cobalt_heron/` with `export PYTHONPATH=.`.
Environment variables: `CH_DATA` (competition root, the folder holding `train/` and `test/`) and `CH_WORK` (cache and model dir, default `/home/user/work`).

## 0. Setup
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install segmentation-models-pytorch timm pycocotools opencv-python-headless scikit-learn pandas scipy kaggle
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json   # never commit it
kaggle competitions download filament-segmentation-2026 -p /home/user/data/fil && (cd /home/user/data/fil && unzip -q *.zip)
```

## 1. Cache + folds (~3 min on 4 cores)
```bash
python scripts/prep.py            # $CH_WORK/c1024/*.npz, test1024/*.npy, folds.json (month-grouped, 5 folds)
```

## 2. Train (~2.3 s/step on 4 CPU cores at bs 8 × 384 crops)
```bash
python -m ch.train --fold 0  --steps 1100 --out $CH_WORK/models/r18_f0.pt          # OOF model
python -m ch.train --fold -1 --steps 1100 --seed 3 --out $CH_WORK/models/r18_full_s3.pt   # all data
```
To run on Kaggle CPU in parallel (12 h limit, internet on):
```bash
python kaggle/build_unet_kernel.py ch-unet-r34-full "--fold -1 --enc resnet34 --steps 2500 --seed 1"
kaggle kernels push -p kaggle/unet/ch-unet-r34-full
kaggle kernels status kragglenote2forwork/ch-unet-r34-full
kaggle kernels output kragglenote2forwork/ch-unet-r34-full -p $CH_WORK/kout/r34   # model.pt + testprob.tar
```

## 3. OOF maps, tuning, learned filter
```bash
python -m ch.infer --models $CH_WORK/models/r18_f0.pt --src $CH_WORK/c1024 --stems $CH_WORK/folds.json:0 --out $CH_WORK/oof/r18_f0
python -m ch.tune  --probs $CH_WORK/oof/r18_f0 --grid '{"t":[0.5,0.6],"te":[1.1],"merge":[4,8],"min_area":[120,200],"min_score":[0,0.7]}'
python -m ch.eval2048 --probs $CH_WORK/oof/r18_f0 --cfg '{"t":0.5,"te":1.1,"merge":4,"min_area":120,"min_score":0.7}'
python -m ch.filt  --probs $CH_WORK/oof/r18_f0 --cfg '{"t":0.5,"te":1.1,"merge":4,"min_area":30,"min_score":0}' --save $CH_WORK/models/filt_r18f0.pkl
```

## 4. Test maps + submission
```bash
python -m ch.infer --models $CH_WORK/models/r18_f0.pt --src $CH_WORK/test1024 --out $CH_WORK/test/r18_f0
python -m ch.submit --probs $CH_WORK/test/r18_f0 [$CH_WORK/test/other ...] --filt $CH_WORK/models/filt_r18f0.pkl --thr 0.35 --out subs/sXX.csv
kaggle competitions submit filament-segmentation-2026 -f subs/sXX.csv -m "cobalt-heron sXX: <change> (OOF ...)"
kaggle competitions submissions filament-segmentation-2026 | head
```
`--probs` takes several directories, and their maps are averaged; that is how model ensembling works.
Log every submission in `docs/SUBMISSIONS.md`.

## 5. Daily loop (5 submissions, reset 00:00 UTC)
1. Collect any finished Kaggle kernel outputs and refresh the OOF maps.
2. Re-tune post-processing and the filter on all available OOF folds.
3. Submit the best new OOF configuration first, then ensembles and variants.
4. Record the LB, update `docs/SUBMISSIONS.md`, commit, and push.
