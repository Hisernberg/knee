# RSNA Knee Abnormality Detection — deep competition analysis

*Compiled 2026-09-22 from the official data files' metadata (shipped inside public solution repositories),
six public solution repositories, public OOF/leaderboard audits, and web search. Kaggle.com itself was not
reachable from the sandbox that produced this document; every number below is attributed.*

## 1. The task in one paragraph

Predict, per knee MRI **study**, a probability for each of twelve findings; the score is the **mean of the
twelve per-finding ROC-AUCs** (macro AUC). It is a **code competition**: the scored artifact is a Kaggle
notebook version that runs offline (no internet) in ≤ 9 h on 2×T4 and writes `submission.csv`. Public LB
is computed on ≈ 30 % of the hidden test set, the private LB on the other 70 %. Timeline: entry deadline
2026-10-15, final submissions 2026-10-22, $77k prize pool, winners recognised at RSNA 2026.

## 2. Data (verified from the real train/test metadata)

| Item | Value |
|---|---|
| Training studies | **4,407** (`train.csv`) |
| Training series | **24,371** (`train_series.csv`), 819,078 DICOM files |
| Series per study | median 5, range 3–14; **every study has Sagittal, Coronal and Axial** series |
| Slices per series | median 30, range 11–320 |
| Plane distribution (series) | Sagittal 9,864 · Coronal 8,609 · Axial 5,898 |
| Fluid-sensitive / fat-suppressed | 14,010 of 24,371 series each (flags coincide in this release) |
| Fluid-sensitive fat-sat series in all three planes | 3,991 studies |
| **Official labels** | **only 58 studies** have the 12 binary labels; 4,349 have `NaN` |
| **Radiology report** | `train.csv` column `Report`, free text, **present for every training study**, median 977 characters (p90 2,118, max 4,743), **absent at test time** |
| Report languages (keyword heuristic on 4,407 reports) | Dutch ≈ 1,684 · German ≈ 715 · Spanish ≈ 681 · English ≈ 504 · French ≈ 77 · ≈ 746 others incl. Croatian/Serbian, Bulgarian (Cyrillic), Greek, Turkish — at least ten languages and three scripts |
| Test set | public stub: 3 studies (5 series each); hidden rerun ≈ **1,300 studies** (public audits) |
| DICOM | stripped to an allow-list of 86 tags; mixed transfer syntaxes (uncompressed, JPEG Lossless, JPEG 2000, Implicit VR); `PatientID` unreliable; no laterality tag on ~50 % of sites |
| Series metadata columns | `StudyInstanceUID, SeriesInstanceUID, Fluid_Sensitive, Fat_Suppression, Anatomical_Plane` (train and test identical) |

Gold-58 prevalence (n positive / 58): ACL 24 · MCL 9 · Medial Meniscus 26 · Lateral Meniscus 23 · Medial OA 15 ·
Lateral OA 11 · PF OA 21 · Effusion 35 · Synovitis 27 · Baker's 12 · Contusion 19 · Fracture 18.
The data page states prevalence may differ between train, public and private sets.

**Submission format** (`sample_submission.csv`, verified): `StudyInstanceUID` + the twelve targets in the
order `ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis,
Baker's, Contusion, Fracture`, one row per test study, values in [0,1].

### 2.1 What this implies

1. **This is a weak-supervision problem in disguise.** 1.3 % of studies are labelled. Every competitive
   solution derives labels from the reports (rule-based extractors, community LLM label tables such as
   `pilkwang/rsna-knee-llm-labels`, `stevenleehans/rsna-knee-llm-report-labels`) and trains image models
   on them. Public audits measured report-label agreement with gold at **0.85–0.90 macro AUC** (best clean
   public table 0.899); five public tables were found to copy the gold labels verbatim and are useless for
   validation.
2. **Image models beat the reports on several findings** (Effusion 0.95 vs 0.83, Synovitis 0.80 vs 0.69,
   Contusion 0.92 vs 0.82, Medial OA 0.98 vs 0.94 on gold-58): the gold labels are image-based, reports
   under-report incidental findings. Hence report labels must be used as *soft, weighted* targets, and
   "unmentioned" must not be treated as a confident negative.
3. **Validation is the hard part.** Gold-58 has a macro-AUC standard error of ≈ 0.02–0.03, so it can only
   catch large regressions. Public OOF work at n = 4,349 (against the best weak labels) resolves ±0.001 and
   is the only honest way to rank blends.

### 2.2 Our own measurement: a transparent multilingual rule labeler

`kneemri.reports` (regex, negation windows, vocabularies for EN/ES/NL/DE/FR/PT/IT plus anchors for
Croatian/Serbian, Turkish, Bulgarian and Greek) labels all 4,407 reports in 2 s; 97 % of reports get at
least one explicit state. Against gold-58 the raw tri-state scores **0.781 macro AUC** (ACL 0.87, MCL 0.87,
Baker's 0.86, Medial OA 0.84, Lateral OA 0.81, Lateral Meniscus 0.80, Fracture 0.79 … Contusion 0.71,
Synovitis 0.66, Effusion 0.64), and 27–88 % of studies per finding remain "unmentioned". The weakest
findings are exactly those radiologists mention inconsistently (effusion, synovitis, marrow oedema) — the
same findings where public work found image models beat any report source. This confirms that a good LLM
teacher (0.85–0.90 on the same gold set) is worth ≈ +0.07 to +0.12 of label AUC over rules, i.e. **label
quality is the single largest lever in this competition**. `scripts/label_reports_llm.py` implements that teacher
(Batches API, structured JSON, ≈ $0.02/report).

## 3. The public solution landscape (what 0.94 is made of)

Lineage of the top public notebooks: `pilkwang/rsna-knee-baseline-v1` → `prvsiyan` → `mattiaangeli
/bend-the-knee-to-the-dinosaurs` (+ CoAtNet blend from `dreaddevelopment`) → ~10 forks at **0.935–0.941**.
Their anatomy, verified by audits that read the notebooks and logs:

| Stage | Component | View | Labels | Gold-58 AUC |
|---|---|---|---|---|
| 1 | DINOv2-S, 20 members (5 folds × 4 seeds), per-finding window pooling | 6 protocol slots × 12 slices, 336 px, 130 mm crop | rule-based multilingual extractor | 0.840 (OOF) |
| 2 | DINOv3-S ("A5"), 5 folds, blended at 0.45 | 6 slots × 16 slices, 336 px | community LLM labels | — |
| 3 | RadImageNet ResNet-50 frozen + 3 head families + logistic calibrator | 3–4 slots × 8 slices, 224 px | three report-label teachers | 0.854 (OOF) |
| 4 | **CoAtNet-RMLP-2 @384, 2.5D triplets, per-finding attention-MIL ("Raptor" v5/v8/v10 + residual-gated e4/e6/e8)** | 5 slots × 44–64 slices, 336/384 px cache, **140 mm crop, 15–85 % depth span, per-series 2–98 % normalisation** | LLM labels | **0.912–0.920 per checkpoint** |
| out | CoAtNet family 0.60 + chain 0.40, per-target outer weights probed on the public LB | | | |

Measured facts from the audits:

* One CoAtNet checkpoint alone: **0.914 public LB**; the CoAtNet family ≈ 0.92; + DINO/RadImageNet chain
  → 0.937–0.941. A clean re-implementation of the CoAtNet family only (3 public checkpoints + the
  residual-gated arm, equal weights, ≈ 2.5–3 h runtime) scored **0.939** — the chain and the LB-probed
  weights are worth ≤ 0.002.
* Swapping backbones on the same labels and views (Swin-B-384, EfficientNetV2-L-480) moved LB by **+0.001**.
  Public Raptor checkpoints correlate 0.91–0.94 with each other; adding one more is a no-op.
* The only additions that measurably helped were **differently-trained families** (other labels, slot
  layout, crop, resolution): +0.007 gold-58 for the residual-gated CoAtNet; +0.0010 (95 % CI +0.0001…+0.0019)
  OOF for a fourth decorrelated family; equal-weight blending beat the best single family by +0.005.
* Fitting per-target blend weights by 5-fold CV on 4,349 studies: **−0.0008** vs equal weights.
* Canonical anatomical orientation (mirroring right knees so medial is always on one side) — **−0.097**
  hold-out macro on a 2,000-study A/B; the "v5-reverse" window-order arm is a mathematical no-op.
* Remaining headroom on gold-58 for the strong family: Synovitis 0.83, PF OA 0.85, Lateral OA 0.86,
  Lateral Meniscus 0.88 (medial counterparts 0.97–0.98).

## 4. Leaderboard state and what top-5 needs

* Public LB top ≈ **0.955**, several teams ≥ 0.95 (ledger dated 2026-09-12); 0.935 ≈ rank 206 of 2,406.
* The **efficiency-track leader is also at 0.954** ⇒ 0.95+ does not require a 9-hour ensemble.
* Public notebooks plateau at 0.939–0.941 because they recombine the same representation; the gap to
  0.955 is a **better core model trained on better labels**, plus pipeline-diverse arms.
* From 0.941, reaching 0.95 requires closing ≈ 15 % of the remaining AUC gap; 0.96 requires ≈ 32 %.

Our target recipe therefore: (1) LLM-labelled, fold-calibrated soft targets; (2) the exact CoAtNet
preprocessing contract as one arm; (3) three more arms that differ in resolution/crop/context/label
treatment; (4) equal-weight rank blending validated on OOF at n = 4,407; (5) attention to the axial plane
and the lateral compartment where the headroom is. See `SOLUTION_DESIGN.md`.

## 5. Engineering facts that cost other teams GPU-hours

| Pitfall | Evidence | Our handling |
|---|---|---|
| Kaggle **P100** cannot run the current torch build (`cudaErrorNoKernelImageForDevice`, even on a tensor copy); an invalid `machine_shape` silently falls back to P100 | public README | `machine_shape: NvidiaTeslaT4`; inference probes the GPU with a real op |
| **bf16 on T4 is 3.1× slower than fp16** (796 vs 254 ms/step at 384 px) | measured | fp16 autocast, fp32 sequence head (fp16 overflowed there in public runs) |
| **Training on Kaggle is not viable**: 5.4 s/study = 6.5 h/epoch from network storage | measured | train offline on local disk; Kaggle is inference-only |
| Kaggle storage bandwidth ≈ 1 MB/s for random access | measured | compact uint8 window cache, streaming test decode once per study shared by all arms |
| `argsort().argsort()` rank fusion is not tie-safe | audit | average ranks (`scipy.rankdata`) |
| Notebooks with `from __future__` inside an indented block pass `ast.parse` but fail at run time | audit | `compile()` every cell at build time |
| Checkpoint/preprocessing contract mismatches (336 vs 384 cache, 2–98 vs 6–94 span) silently degrade arms | audit | the prep config is embedded in every checkpoint and re-used at inference |
| Missing `PixelSpacing`, single-component spacing, multi-frame files | audit | explicit policies with logged events; row/col spacing kept separately |
| A stale arm in production (0.905 vs 0.921 available) was the largest single gain of one project | audit | checkpoints carry `val_macro_auc`; `blend_oof.py` ranks arms before packaging |

## 6. Prior RSNA competitions (pattern transfer)

RSNA 2022 (cervical spine), 2023 (abdominal trauma), 2024 (lumbar spine degenerative), 2025 (intracranial
aneurysm) were all won by **2.5D CNN/transformer encoders over slice windows + a sequence/attention
aggregator**, frequently two-stage (localise the structure, then classify a crop), with heavy per-family
ensembling and TTA, and with validation designed around the exact metric. The knee task differs in one key
way: it is study-level with no localisation labels, so the "second stage" must be learned attention over
windows (per-finding queries), and the anatomy crop is done with physical millimetres rather than detected
keypoints. The public strong family is exactly that design; our `StudyModel` generalises it to variable
series with protocol embeddings.

## 7. What this session could and could not do

* **Could**: research; write and test the full pipeline end-to-end on synthetic DICOM data (8 tests
  pass on CPU); build the offline Kaggle notebook and the 5-submission loop; measure the rule labeler on the
  real reports and gold labels.
* **Could not**: download the images, train real weights, or submit. The sandbox's egress policy returns
  HTTP 403 for `www.kaggle.com`, `api.kaggle.com` and `huggingface.co`. Nothing in this repository is a
  trained model or a scored submission; the `best_filament_unet.pth` produced by the tests is a
  synthetic-data smoke checkpoint and is not committed.
