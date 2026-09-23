# Plan — cobalt-heron

Goal: the highest honest PQ we can reach, working toward the top of the LB with no test-label leaks, on CPU only.

## Pipeline (v1, working)
1. 1024 cache and soft consensus mask: the mean over annotator records, so a model trained on it predicts *consensus*. This matters because inter-annotator PQ is 0.34.
2. U-Net (ImageNet encoder), 3-channel input (raw / CLAHE / flattened), random 384 crops biased 70% to filaments. BCE+Dice on the soft mask, plus a touching-edge head.
3. Flip-4 TTA → prob maps → connected components, fragment merge (radius 4 at 1024), area and score filter.
4. **Learned instance filter**: a GBM on instance features (area, prob stats, elongation, limb distance, contrast, crowding) predicts the fraction of annotator records each instance will hit. Instances are kept above a threshold chosen on grouped CV, which gives +0.04 OOF PQ.
5. Upsample to 2048 with prob-guided boundaries, then write RLE.

## Experiment queue (ordered by expected gain / CPU cost)
1. Ensemble fold/full models (local and Kaggle CPU kernels in parallel).
2. More OOF folds (1–4), so the filter and post-processing are tuned on 707 images instead of 144.
3. Train longer (3–5k steps) and try resnet34 or efficientnet encoders (Kaggle CPU kernels: 12 h each, several in parallel).
4. Native-resolution refinement: a crop refiner at 2048 around each kept instance, to raise SQ (currently 0.66–0.68).
5. Instance grouping learned from annotators: tune merge radius by direction (along the filament spine).
6. Once the GPU quota returns (Saturday): YOLO11m-seg @1536 fold-0 + full (`kaggle/yolo/`), fused with the U-Net instances.
7. Pseudo-label the test set with the ensemble (self-training). This is legal because it uses no test labels.

## Submission policy (5 per day, resets 00:00 UTC)
- Every submission must carry an OOF-validated change. Log it in `docs/SUBMISSIONS.md` with its OOF and LB.
- Spend at most one slot per day on a pure calibration probe.
- Keep the final 2 selections as the best OOF and the best LB, if they differ.
