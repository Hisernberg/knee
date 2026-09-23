# Submission log — cobalt-heron (filament-segmentation-2026)

| # | date (UTC) | file | change | OOF PQ | public LB |
|---|---|---|---|---|---|
| s01 | 2026-09-22 | s01_r18f0.csv | U-Net R18 (fold-0 model, 1100 steps), CC post-processing t0.5 merge4 area120 score0.7 | 0.382 @2048 | 0.32 |
| s02 | 2026-09-22 | s02_r18f0_filt.csv | s01 + GBM instance filter thr 0.35 | 0.422 @1024 proxy | 0.34 |
| s03 | 2026-09-22 | s03_fuse_ektarr.csv | s02 scores: keep >0.5, or 0.2–0.5 if a public ektarr instance agrees (IoU>0.3) | n/a (unvalidated) | 0.34 |
| s04 | 2026-09-22 | s04_ens2_filt.csv | mean(r18 fold-0, r18 full-data) maps + filter thr 0.35 | n/a (full model has no OOF) | 0.34 |
| s05 | 2026-09-22 | s05_ens2_filt050.csv | s04 with stricter filter thr 0.50 | 0.407 @1024 (thr 0.5 on f0) | 0.33 |
| s06 | 2026-09-23 | s06_k18full.csv | r18 full-data **3000 steps** (Kaggle CPU kernel ch-unet-r18-full) + filter thr 0.35 | n/a (full data) | **0.35** |
| s07 | 2026-09-23 | s07_ens3.csv | mean(r18 f0, r18 full 1100, r18 full 3000) + filter thr 0.35 | n/a | 0.34 |
| s08 | 2026-09-23 | s08_k18full_filtcomb.csv | s06 model + filter retrained on fold-0 + fold-2 OOF (316 imgs), thr 0.40 | 0.413 @1024 (2-fold) | 0.35 |
| s09 | 2026-09-23 | s09_r34full.csv | r34 full-data 2500 steps + filter retrained on 3-fold OOF (466 imgs) thr 0.40 | 0.411 @1024 (3-fold, r18 OOF) | 0.34 |
| s10 | 2026-09-23 | s10_r34_r18full.csv | mean(r34 full 2500, r18 full 3000) + filter thr 0.35 | n/a | 0.35 |

Takeaways from day 1:
- The LB shows only 2 decimals.
- OOF and the LB move in the same direction: the filter helps, and too strict a filter hurts.
- LB ≈ 0.84 × OOF@2048.
- To move the LB we need larger OOF gains (≥ +0.02): more OOF folds for tuning, longer training, bigger encoders.

Account history before cobalt-heron: 0.55 came from the public hdjojo/lamhuy payload (see ANALYSIS.md), and there was an unknown 0.33.

Day 2:
- Training longer (3000 vs 1100 steps) gives +0.01 on the LB.
- Averaging with the weaker short-trained models hurts.
- Next: long-trained models only (r34 2500, r18 folds at 3000+ steps).
- Local long runs die when the container restarts while idle, so all training now runs on Kaggle CPU kernels.
- Day-2 end: plateau at 0.35. The r34 encoder alone is not better than r18 (0.34), and the r34+r18 ensemble ties at 0.35. On OOF, FN dominates (TP 3091 / FP 1418 / FN 2531 at thr 0.40).
- Next levers: 4500-step r18 (ch-unet-r18-full-s5, running), a 2048 crop refiner for SQ, YOLO11-seg on GPU after the Saturday quota reset, and fusing U-Net and YOLO instances.
