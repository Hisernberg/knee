# Technique survey (research agent, 2026-09-22): past RSNA winners and this competition's public threads

Summary of the full agent report; every number is quoted from the linked source, [speculation] marks inference.

## Ranked techniques for 0.943 -> 0.955+ (public LB, estimated gains)
| # | Technique | Evidence | Est. gain | Cost |
|---|---|---|---|---|
| 1 | Label engineering: LLM labels with an explicit "not addressed" state, per-finding silence policy (absent-when-silent for Baker's/medial OA; mask or impute for synovitis, PF OA, fracture), soft targets, per-fold calibration on gold-58, then OOF pseudo-label blending / self-distillation and confident-learning removal of report-image conflicts | every 0.947-0.958 single-model poster credits labels (thread 735304: #1 Scott Willis small ResNet 5-fold 0.947; #30 CoAtNet@224 single fold 0.950; laymond 0.938->0.943 label-only; yannmajewski +0.015 label work); LLM vs regex labels 0.878 vs 0.814 gold AUC, synovitis imputed from effusion 0.678->0.790 (thread 733932); lumbar 2nd: confident learning "notable LB"; aneurysm 5th: cleaning +0.008 OOF | +0.005 to +0.012 | low |
| 2 | Input geometry: physical-scale crops per finding (150 mm centre, ~90 mm meniscus crop), sequence-aware slot selection, more slices per slot, spacing-aware neighbours | crop fix +0.0059 on 10/12 labels (735154); 3->9 slices "substantial" (vhittuvalli); aneurysm 6th +0.01 tighter ROI | +0.003 to +0.008 | low-medium |
| 3 | Anatomy-specific ROI / auxiliary localisation heads | aneurysm 1st aux seg +0.026, seg-pretrained backbone +0.108; abdominal 1st aux seg +0.01-0.03 | +0.003 to +0.010 [speculation for knee] | medium |
| 4 | Compact 2.5D CNN at 224-288 with attention-MIL + bi-LSTM head and aux losses, longer training, stronger regularisation (not bigger encoders) | lumbar 1st MIL 0.37->0.35, +LSTM/aux ->0.33; DINOv2-S->B +0.0011 (null, 735154); 25->50 epochs +0.004 (740610) | +0.002 to +0.006 | low |
| 5 | Fine-tuned VLM (Qwen 3.5 class) at 384 as a decorrelated arm | andy2709 single fold 0.950 in 3.5 h; decorrelated arm +0.004 (charlessavas) | +0.003 to +0.006 blended | high |
| 6 | Augmentation/TTA: flips, cutmix/mixup, ShiftScaleRotate, multi-crop and slice-offset TTA | aneurysm 2nd TTA 4x +0.027 public/+0.012 private; tennogh +0.001 TTA | +0.002 to +0.005 | low |
| 7 | Plane-aware fusion (per-plane heads + learned late fusion, label-query transformer over slot/slice tokens) | MRNet: axial PD best for meniscus, coronal T1 for ACL | +0.002 to +0.005 [speculation] | low |
| 8 | 3-5 diverse families, rank averaging | lumbar 3rd 30-model mean -0.0215 logloss; abdominal 2nd no private gain | +0.002 to +0.004 | medium |

Do not spend on: bigger DINOv2/ViT encoders, 3D CNNs, Mamba over slices, blanket imputation of unaddressed labels, blend-weight tuning on the LB, 20+ model ensembles (top-10 teams run in minutes on the efficiency LB).

## This competition's public facts (thread 735304 "best single-model score", and others)
* Scott Willis (#1, 0.958): smallest possible ResNet, single fold 0.938, 5-fold 0.947, "spent a lot of time working around the low-quality labels"; top of the efficiency LB.
* Archit Konde (#30): single-fold 2.5D CoAtNet@224 0.950; OOF predictions closer to the radiologists than the extracted labels -> worked on labels.
* tennogh: 0.942 raw, 0.943 with TTA at 288; "OOF pseudo-labels well correlated with LB"; resolution not a big driver.
* Tom Aindow: DINOv2 0.915 at 392 px, 150 mm crop, random bag of 32 slices; "top results are not big ensembles".
* charlessavas (0.943 team): one decorrelated external arm (corr 0.83) +0.004.
* Host: gold labels come from images with stricter thresholds; image label is authoritative over the report; commercial LLM APIs ruled out, local open-weights LLM fine (733826, 733652, 733965).
* "Not addressed is a label too" (733932): 25.4 % of cells unaddressed; synovitis 84 % unaddressed; silence means absent for Baker's (3 % gold-positive when silent) but unknown for synovitis (34 %).
* No DICOM-metadata shortcut (733517).
* Weak columns on gold-58 for image models: synovitis 0.69, lateral OA 0.78, PF OA 0.82, lateral meniscus 0.86 (740610).

## Past RSNA competitions (components -> stated gains)
* Lumbar 2024 1st: keypoint crop + attention-MIL (0.37->0.35) + bi-LSTM/aux losses/ensemble (->0.33); ConvNeXt-small > base > large; instance-number jitter crucial. 2nd: 27-crop TTA "significant", confident learning on labels, 0.5 label + 0.5 pseudo-label loss. 3rd: 30-model mean -0.0215; data multiplication via L/R flip.
* Abdominal 2023 1st: 96 slices -> 32 triplets at 384, CoaT-Lite + GRU, aux segmentation +0.01-0.03. 2nd: organ-visibility frame sampling "key", crops 0.41->0.37 public. 3rd: masking, custom samplers.
* Cervical 2022 1st: 3D UNet masks from 87 cases, 2.5D + LSTM, 3D CNN "did not work".
* Aneurysm 2025 1st: seg-pretrained encoder (+0.108), aux sphere seg (+0.026), resolution 96->128 (+0.014), flip TTA, EMA. 2nd: corrected annotations +0.032 private, flip-with-label-swap TTA. 5th: brain-crop +0.03-0.05, label cleaning +0.008 OOF. 6th: each of tighter crop, strong aug, cutmix, flip-swap, spacing-aware sampling ~+0.01 CV.
* MRNet: per-slice CNN + max over slices, plane-specific findings, 0.92-0.94 AUC with clean labels.
* AnyMC3D (CVPR 2026, arXiv 2512.12887): query-based attention pooling over slice embeddings 0.962 vs LSTM 0.903; 2D beats 3D.

Key URLs: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304 , /733932 , /735154 , /740610 , /733826 ,
https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution ,
https://github.com/Nischaydnk/RSNA-2023-1st-place-solution , https://github.com/uchiyama33/rsna2025_1st_place ,
https://github.com/vhittuvalli/RSNA-Knee-Abnormality-Detection , https://github.com/TranBaDat2607/RSNA-Knee-Abnormality-Detection , https://arxiv.org/abs/2512.12887
