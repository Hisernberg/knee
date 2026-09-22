# UMUD leaderboard log (2026-09-22)

Metric: mean(MAE_PA/6, MAE_FL/12, MAE_MT/3), lower is better. Leaderboard at time of writing: #1 0.28263, #3 0.31898.

| # | Submission | Public LB | Re-runnable on new data? |
|---|---|---|---|
| s01 | Public "variational-ensemble" notebook output (Vera with FL/MT shrunk) | 0.49332 | no (hard-coded CSV) |
| — | Public Vera notebook output (score reported by its author; baseline) | 0.45134 | no (hard-coded CSV) |
| s02 | **A**: this pipeline (`configs/a_raw.json`) | 0.52866 | **yes** |
| s03 | **B**: 0.5·A + 0.5·Vera (`scripts/blend.py --shot B`) | 0.38770 | no |
| s04 | **C**: A(PA+1.6°) ·0.45 + Vera ·0.55, cine-loop smoothing, anchors pinned (`--shot C`) | **0.37011** (rank 13/293) | no |

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
