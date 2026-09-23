# Competition analysis — cobalt-heron

## Data
- 707 training images (2048×2048 GONG H-alpha JPEGs from 6 sites B/C/L/M/T/U, 2011-2022) and 180 test images with the same year and site mix.
- COCO json with **1154 image records for 707 files**. Each record is one annotator's full set of filaments for that image: 411 files have 1 record, 145 have 2, 151 have 3. There are 37 annotator ids and 8199 polygons, about 7.1 per record.
- Categories are Left, Right, Unidentifiable and Ambiguous (chirality). **The metric ignores the class.**
- Instance areas at 2048: p5 317, median 1228, p95 6947 px.

## Metric (organizer `Self_Evaluation_Notebook`, replicated exactly in `ch/metric.py`)
- Submission columns are `filament_id,segmentation_rle`. The id is `<stem>_<k>`, and the RLE is the pycocotools compressed counts string at 2048×2048.
- **Every annotator record is scored separately** against the same predictions for its image, and the counts are pooled over all records:
  `PQ = sum IoU(TP) / (TP + 0.5 FP + 0.5 FN)`
  - Every (gt, pred) cell with IoU > 0.5 is a TP.
  - A pred column with no hit is an FP, and a gt row with no hit is an FN.
- Because of this, an extra prediction costs 0.5 in the denominator for *each* annotator record of that image. A prediction is worth keeping only if P(hit)·IoU > PQ/2, i.e. P(hit) ≳ 0.3.
- **Inter-annotator PQ on train is only 0.34** (one annotator scored against another). A model predicting the consensus can beat single annotators.
- Duplicate or overlapping predictions could be double-counted as TPs by the hit matrix. That would be a metric exploit, so we **do not** use it: all our masks are disjoint.

## Leaderboard reading (2026-09-22)
- Top of the public LB: 0.62, 0.61, then a large cluster at 0.55–0.56.
- The 0.55 cluster is one file: `hdjojo/solar-filament-seg-inference`, a YOLOv8l-seg at 2048 with private weights. `lamhuy8904/...0-55` hard-codes that file as a base85 payload, and so did the user's earlier 0.55 submission.
- Public notebooks mention that the public MAGFiLO release and its mirrors contain **test annotations**. Honest OOF PQs are about 0.41 (ektarr, timgluhkikh, both on their own folds). Our honest model agrees with the 0.55 file only at PQ 0.44, the same as other honest models (0.47). **The 0.55+ scores may therefore be leak-assisted.** We do not use test labels or MAGFiLO mirrors.
- Our calibration so far: OOF PQ at 2048 of 0.382 → LB 0.32, so LB ≈ 0.84 × OOF. Test images likely carry more annotator records per image.

## Public notebook audit (summary)
| notebook | approach | notes |
|---|---|---|
| hdjojo | YOLOv8l-seg @2048 | the 0.55 file; weights private |
| ektarr | 5-fold YOLO11s-seg + ResNet18 crop refiner, fold fusion | OOF PQ 0.41; weights in kernel output |
| timgluhkikh | YOLO11s detect → U-Net-R34 crop refiner | val PQ 0.41 |
| phuongncn | Mask R-CNN + refiner | "0.70" was under the old Dice metric |
| anthonytherrien | semantic U-Net + connected components | no PQ number |
| pavloivanin / rokaiyasomapti "0.9 / LB 1st" | trains on empty masks | exploited the old Dice-metric bug; meaningless now |
