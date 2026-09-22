# Solution design: RSNA Knee Abnormality Detection (target: top-5 private LB)

This design is derived from the evidence in `COMPETITION_ANALYSIS.md`. Every choice below is tied to a
measured fact from public work (cited there) or to a stated hypothesis with a promotion rule.

## 1. What the leaderboard tells us

| Fact | Consequence for the design |
|---|---|
| Public LB top ≈ 0.955; several teams ≥ 0.95; efficiency-track #1 is also at 0.954 | A 9-hour mega-ensemble is not required. The gap above the public 0.94 stack is a **better core model + better labels**, not more members. |
| Strongest public family: CoAtNet @384, 2.5D triplet windows, per-finding attention-MIL, 140 mm physical crop, 15–85 % depth span, per-series 2–98 % normalisation. Single checkpoint 0.914 LB / 0.912–0.920 gold-58 | Reproduce that recipe as **Arm B** (CoAtNet-0 @384) and keep its exact preprocessing contract. |
| Backbone swaps on the same labels/views were worth only +0.001 LB; a differently-trained family (other labels, slots, crop) was worth +0.007 gold / +0.02 LB | Diversity must come from **label source, series selection, crop and resolution**, not from backbones alone. Arms A–D deliberately differ on those axes. |
| Per-target blend weights fitted on OOF: −0.0008 vs equal weights (n=4,349). Adding a decorrelated family: +0.0010 (95 % CI +0.0001…+0.0019) | Equal-weight rank averaging across families; never fit weights on public LB or on 58 gold studies. |
| Gold-58 macro-AUC SE ≈ 0.02–0.03 | Gold-58 is a regression guard only. Model selection uses **OOF on all 4,407 studies against report labels** (n=4,349 weak + 58 gold), which resolves ±0.001. |
| Remaining headroom is concentrated on Synovitis (0.83), PF OA (0.85), Lateral OA (0.86), Lateral Meniscus (0.88); medial counterparts are 0.97–0.98 | Add axial-plane emphasis (PF OA, synovitis, effusion) and a high-resolution lateral-compartment view; make sure the axial series is never dropped by series selection. |
| Report labels agree with gold at 0.85–0.90 macro; image models beat report labels on Effusion/Synovitis/Contusion/Medial OA | Reports are a noisy teacher: use **calibrated soft labels with confidence weights** and keep the 58 gold rows at high weight; unmentioned findings get low weight, never a hard 0. |
| Canonical anatomical orientation (mirroring right knees) hurt training (−0.097 hold-out on a subset) | Do **not** canonicalise. Use label-preserving horizontal flips as augmentation (a flipped left knee is a plausible right knee). |
| Reports are unavailable at test time; hidden test ≈ 1,300 studies; public LB = 30 % | No text model at inference. Everything text-derived is distilled into the image models through labels. |

## 2. Pipeline

```
train.csv (Report, 58 gold rows)      train_series.csv + DICOM
        │                                       │
        ▼                                       ▼
  report labeler (rules + LLM)         prepare(): decode → order → 2–98 % normalise → 140 mm crop
        │  tri-state per finding                │  → 384 px → K centres in 15–85 % depth → 3-slice windows
        ▼                                       ▼
  fold-safe calibration (gold rows      StudyWindows cache (uint8 .npz per study, ~1–3 MB)
  of the training fold only)                    │
        │  soft label + weight per cell         ▼
        └──────────────► masked, weighted BCE ◄─┘
                               │
                   StudyModel: timm encoder (2.5D) → protocol/geometry embeddings
                   → within-series BiGRU (optional) → cross-series transformer (optional)
                   → 12 finding queries (masked attention pooling) → 12 logits
                               │
                   5 folds × 4 arms → OOF → equal-weight rank blend → Kaggle notebook
```

Key implementation properties (all in `src/kneemri/`):

* **Variable series and windows per study** with explicit padding masks; series boundaries are explicit
  (windows never cross series; the BiGRU runs per series). Any number of series (3–14 in train) works.
* **Physical geometry**: row/column spacing kept separately; fixed 140 mm square FOV so anatomy has a
  constant scale across the 5-continent, multi-vendor data; missing spacing has an explicit logged policy.
* **Never lose a test study**: unusable series are skipped and logged; a study with no usable image gets
  the training prior; the checkpoint's target order is re-mapped by name to the sample-submission order.
* **fp16 autocast on T4** (bf16 is 3× slower on sm_75), sequence head in fp32 (public runs overflowed),
  EMA weights, cosine schedule with warmup, gradient clipping, gradient checkpointing for 384 px arms.
* **Time budget**: inference measures the first arm and refuses to start arms that cannot finish before the
  budget; the primary arm always completes; the receipt lists skipped arms.

## 3. The four arms (pipeline-diverse by construction)

| Arm | Encoder | Resolution / crop | Windows | Series | Labels | Head |
|---|---|---|---|---|---|---|
| A | ConvNeXt-Tiny (in22k) | 320 px, full FOV | 24 centres × 3 slices, 0–100 % | up to 8 | rules+LLM calibrated | BiGRU + 1 transformer layer + finding queries |
| B | CoAtNet-0 | 384 px, 140 mm | 20 centres × 3 slices, 15–85 % | up to 8 | rules+LLM calibrated | BiGRU + 1 transformer layer + finding queries |
| C | EfficientNetV2-S | 320 px, full FOV | 24 centres × **5 slices** | up to 8 | rules+LLM, **ASL (γ⁻=2)** | 2 transformer layers, no GRU |
| D | MaxViT-Tiny 384 | 384 px, 140 mm | 20 centres × 3 slices, 15–85 % | up to 8 | rules+LLM calibrated | 2 transformer layers, no GRU |

Planned fifth arm once A–D are measured (the single lever with evidence behind it): a **second teacher**.
Train Arm B again on LLM-only labels (no rule labels) and on rule-only labels; keep whichever is least
correlated with the rest while ≥ 0.90 OOF macro. Pipeline diversity is what paid publicly; this is the
cheapest way to buy it.

## 4. Validation protocol

1. `stratified_group_kfold` on all 4,407 studies (positives from gold where known, else from report states),
   5 folds. Patient IDs are stripped, so grouping is by study; duplicate-exam leakage cannot be excluded.
2. Report calibration and aux-label tables are built **per fold** from the gold rows outside that fold
   (`report_labels_fold{k}.csv`). Validation-fold studies contribute no training loss.
3. Model selection metric: OOF macro-AUC against the best available label per study (gold if present,
   else calibrated report label), reported alongside gold-58 AUC as a guard.
4. Blends are evaluated on OOF with `scripts/blend_oof.py`: equal-weight prob and rank blends plus a
   leave-one-arm-out table. Weights are not fitted.
5. Promotion rule for any new arm: it must raise the equal-weight OOF blend by ≥ 0.001 (the resolution of
   n≈4.4k) and not lower gold-58 by more than its SE.

## 5. Submission strategy (5 daily submissions)

`kaggle/variants.yaml` defines five notebooks that isolate one question each:

| Variant | Question it answers |
|---|---|
| v1 Arm A 5-fold | single-family control; establishes CV→LB offset |
| v2 A+B rank | does the CoAtNet family add what public work says it should? |
| v3 all arms rank | full ensemble, equal weights |
| v4 all arms prob, no TTA | isolates TTA + rank-vs-prob effects |
| v5 best-two weighted | the OOF-selected pair, to test whether fewer, stronger arms beat the full blend |

Final two selections: the best OOF-supported blend (not the best public-LB score) and the most different
strong variant, because the private LB is 70 % of the test data and public-LB probing overfits.

## 6. Risks and mitigations

* **Prevalence shift** between train, public and private (stated on the data page): rank averaging and
  AUC are invariant to monotone recalibration, so no threshold tuning anywhere.
* **Series-count and protocol variability** in the hidden set: dynamic series handling with protocol
  embeddings; series-dropout augmentation (15–20 %) during training.
* **Runtime**: measured public families need ≈ 100 min for the CoAtNet family on 2×T4 for ≈ 1,300
  studies; four arms × 5 folds with flip TTA is ≈ 5–6 h. The budget guard prevents a timeout; if it
  triggers, drop TTA first (variant v4 shows what it costs).
* **P100 fallback**: `machine_shape` must be `NvidiaTeslaT4`; inference probes the GPU with a real op and
  falls back to CPU rather than crashing.
* **Label leakage**: some public label tables copy the 58 gold labels verbatim; never validate against them.
  This repo builds its own labels from the reports and calibrates per fold.
