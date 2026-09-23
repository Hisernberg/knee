# cobalt-heron — solar filament instance segmentation

Codename for the Kaggle community competition **`filament-segmentation-2026`**. The task is instance segmentation of solar filaments in GONG H-alpha full-disk images. Each image is 2048×2048, and the metric is Panoptic Quality (PQ).

Everything here runs on **CPU**: 4 cores and 15 GB RAM locally, or Kaggle CPU kernels. The account's Kaggle GPU quota is exhausted until the weekly reset, and no GPU is needed.

| Doc | What it covers |
|---|---|
| `docs/ANALYSIS.md` | Metric replica, data facts, public-notebook audit, and the leaderboard-leak caveat |
| `docs/PLAN.md` | Roadmap to the top of the LB, experiment queue, and daily 5-submission policy |
| `docs/RUNBOOK.md` | Exact commands: prep, train, OOF, tune, filter, submit, Kaggle kernels |
| `docs/SUBMISSIONS.md` | Every submission with its config, OOF score, and public LB score |

## Layout
```
ch/data.py         COCO parsing, per-annotator records, paths (CH_DATA / CH_WORK env vars)
ch/metric.py       exact replica of the organizer's pooled PQ (IoU>0.5, all hits, per annotator record)
ch/feats.py        3-channel input: raw, CLAHE, background-flattened
ch/train.py        smp U-Net (resnet18/34...), 1024px cache, random 384 crops, soft-consensus + edge targets
ch/infer.py        flip-TTA inference -> float16 prob/edge maps
ch/post.py         prob -> disjoint instances (threshold, hysteresis, edge split, fragment merge, area/score filter)
ch/tune.py         post-processing grid search against the exact PQ on OOF maps (1024 proxy)
ch/eval2048.py     exact 2048 PQ of the full submit path
ch/inst_feats.py   per-instance features + OOF match targets
ch/filt.py         learned keep/drop filter (GBM) with grouped CV, saves filt_*.pkl
ch/submit.py       test maps -> 2048 instances -> submission.csv (optional learned filter)
scripts/prep.py    build 1024 cache, soft masks, month-grouped folds
scripts/cmp_subs.py compare two submissions (PQ of A vs B)
kaggle/build_unet_kernel.py  package ch/ into a self-contained Kaggle kernel (train full data + test maps)
kaggle/yolo/       YOLO-seg GPU kernel (for when GPU quota returns)
```
