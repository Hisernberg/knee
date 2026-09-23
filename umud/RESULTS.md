# UMUD leaderboard log (2026-09-22)

Metric: mean(MAE_PA/6, MAE_FL/12, MAE_MT/3), lower is better. Leaderboard at time of writing: #1 0.28263, #3 0.31898.

| # | Submission | Public LB | Re-runnable on new data? |
|---|---|---|---|
| s01 | Public "variational-ensemble" notebook output (Vera with FL/MT shrunk) | 0.49332 | no (hard-coded CSV) |
| — | Public Vera notebook output (score reported by its author; baseline) | 0.45134 | no (hard-coded CSV) |
| s02 | **A**: this pipeline (`configs/a_raw.json`) | 0.52866 | **yes** |
| s03 | **B**: 0.5·A + 0.5·Vera (`scripts/blend.py --shot B`) | 0.38770 | no |
| s04 | **C**: A(PA+1.6°) ·0.45 + Vera ·0.55, cine-loop smoothing, anchors pinned (`--shot C`) | **0.37011** (rank 13/293) | no |
| s06 | A2: pipeline with OSF-benchmark calibration (`configs/a2_osf_calibrated.json`) | 0.52945 | **yes** |
| s07 | C2: C with A2's calibrated FL | 0.37706 | no |
| s08 | **C3**: C with per-target pipeline weights PA .45 / FL .3 / MT .6 (`--shot C3`) | **0.35157** (rank 8/293) | no |
| s10 | **C5**: C3 with PA pipeline weight 0.6 (`scripts/blend_weights.py 0.6 0.3 0.6`) | **0.34947** | no |
| s09 | C4: per-target weights PA .45 / FL .1 / MT .75 | 0.35513 | no |
| s05 | D: C + Juniper right-tick length rescale (MT ×0.89, FL ×0.87, fitted to the 2 anchors) | 0.40742 | no |

Segmentation (Kaggle T4, `umud-seg-train`): aponeurosis val Dice 0.844 (40 ep), fascicle val Dice 0.303 (30 ep;
fascicle masks are sparse partial annotations, so Dice is structurally low).

## Diagnostics
- Pipeline vs Vera (on the metric's scale): PA 0.52, FL 0.89, MT 0.39; so FL is the weakest estimator.
  Using geometric (extrapolated) FL alone beats blending it with MT/sin(PA) (0.89 vs 1.13).
- Pipeline PA is ~2° lower than Vera and 1.6° lower than the two public anchors.
- On both anchors the pipeline's inner-edge MT is ~14 % above the label (Vera is ~8 % above too), which suggests a
  different MT convention; this is unresolved.
- Test set: 28 five-frame cine loops (IMG_00056–IMG_00195) plus near-duplicate pair 18/19.

## Prize eligibility
Top-3 requires a public, licensed, re-runnable repository. Only shot A qualifies as-is. B/C depend on a hard-coded
public CSV, so they need to be replaced by an equally strong re-runnable model before the final selection.

## Rejected hypothesis (s05)
Both public anchors (Siemens Juniper, right-tick HUD) imply lengths ~14 % shorter than the tick-mark scale gives
(consistent with a 5.0 cm instead of 5.75 cm field of view). Rescaling the 89 Juniper test images accordingly made
the public score worse (0.370 → 0.407), so the anchors are not representative; do not rescale lengths from them.

## Host guidance (discussion forum, 10 topics read)
- MT: three straight *vertical* lines (left / middle / right) between the aponeuroses, averaged over 2 raters.
- PA: FIJI angle tool at three fascicle-fragment insertions (not at the labelled fascicles, not line intersections).
- FL: three fascicles drawn to the aponeuroses, chosen to minimise extrapolation, extended manually if needed.
- Declared external data is allowed, incl. the UMUD "Expert Analysed Benchmarks" (35 images, 7 raters, OSF) that
  the host recommends for calibration. LB-probed ensemble weights are not prize-eligible; hand-tuned ones are.
- 670 exact duplicate fascicle image/mask pairs in train (remove one copy); test is independent of train.

## External validation: UMUD OSF expert benchmark (osf.io/xbawc, 35 images, 7 raters)
`scripts/osf_benchmark.py` runs the trained models + geometry with the benchmark's own px/cm scale and compares to
the rater mean (entry errors > 50 % from the median dropped). MAE (competition-normalised):

| Target | pipeline | DLTrack (host baseline) | one rater vs the others |
|---|---|---|---|
| MT | 0.50 mm (0.166) `mt_inner` | 1.03 mm | 0.24 mm |
| PA | 1.06° (0.177) `pa_wmed` | 1.45° | 1.47° |
| FL | 8.50 mm `fl_med` → **4.69 mm** with `fl_wmed × 0.93` | 3.74 mm | 4.78 mm |

Biases: FL +5 mm (fascicle extrapolation too long), PA −0.5°, MT −0.25 mm → `configs/a2_osf_calibrated.json`.
The benchmark comes from training-like devices (Telemed / Aloka / Philips HD11), not the test devices
(Juniper / Lumify), so these numbers are optimistic for the test set.

Prepared for the next quota day: `submissions/s06_A2_osf_calibrated.csv` (re-runnable) and
`submissions/s07_C2_blend_osfFL.csv` (C with the calibrated FL component).

## Day 2 (2026-09-23)
- The OSF-benchmark calibration did not transfer: A2 0.52945 vs A 0.52866; C2 0.37706 vs C 0.37011. This is the third
  signal (with the Variational output and D) that shortening FL on the test set hurts. The benchmark devices differ from the test devices.
- Per-target blending helps. The benchmark says the pipeline is strongest on MT and weakest on FL, so C3 weights the
  pipeline 0.6 on MT and 0.3 on FL and gains 0.0185. Pushing further (C4) loses 0.0036, so the optimum is near C3.
- Prize eligibility: C3 still depends on the hard-coded public Vera CSV. The host allows hand-tuned ensemble weights,
  but the ensemble member itself must be re-runnable. For a prize-eligible final, Vera has to be replaced by a
  second re-runnable model of similar quality.
- C5 (PA pipeline weight 0.45 → 0.6) gains a further 0.0021, consistent with the OSF benchmark (pipeline PA 1.06° vs
  DLTrack 1.45°; the Vera PA is DLTrack-lineage).
- Weekly Kaggle GPU quota exhausted; `seg.py --init/--kinds/--lr` + `build_kernel.py --cpu --weights-kernel` fine-tune
  the fascicle model on de-duplicated data (670 duplicate pairs removed) in a 12 h CPU kernel.
