# Experiment log

All numbers are local. S_state uses the official formula on the train holdout (days 243–272). The LWR
proxy is `evaluate.s_lwr_proxy`. J = 0.35·S_state + 0.10·S_LWR is the part of S_total that Task 1
controls; S_FD is about constant.

## Task 1 / Task 3

| # | Setup | S_state | LWR proxy | Note |
|---|---|---|---|---|
| E0 | historical mean (official baseline), D7_I10_W | 0.832 | 0.070 | full train, no blackouts |
| E0 | temporal linear interpolation, D7_I10_W | 0.932 | 0.534 | |
| E1 | per-panel LightGBM residual on interpolation, 500k rows, D7_I10_W | 0.9507 | 0.585 | holdout, no blackouts |
| E2 | + L1 density model, q = v·k̂ | 0.9478 | 0.607 | J +0.0014 |
| E3 | blackout cells, D7_I10_W: interpolation vs gap-LGB | speed RMSE 5.81 → 5.36 | | regular cells about 1.1 |
| P1 | pooled 10-panel LightGBM (150k reg/panel), sampled holdout with dark share reweighted | **0.9385** family mean | | raw |
| P1 | same, reconcile a=0.25 | 0.9348 | | costs mostly dark-cell flow |
| P1-full | full-coverage holdout, realistic blackout density, 4 panels | raw → recon a=0.25 | LWR +0.013 to +0.081 | J +0.0010 to +0.0062 on every panel → **adopt a=0.25** |

Pooled model holdout RMSE:
- regular cells: speed 1.54, flow 30.6/lane, density (Huber) 0.567
- dark cells: speed 6.90, flow 65.9/lane, density 3.21

Error budget (regular cells):
- free flow: 97% of cells, 74% of speed SSE and 91% of flow SSE (noise floor)
- queued: 2% of cells, 22% of speed SSE

## Physics facts
- Congested observations lie on the congested branch of the released triangular FD: median deviation
  0.0%, IQR ±2.4%. In free flow, v ≈ v_f (std 2.5 including the transition zone).
- Observed |ΔN|/N per step is 4.4%. The topology-based flux correlates with ΔN at only 0.06, so the
  organizer's flux must absorb the noise, which is why the proxy is valid.

## Task 2 (agent report, docs/traffic/TASK2_ANALYSIS.md)
CV S_queue on 5,083 selector-replicated windows (official aggregation):

| Method | S_queue (sim) |
|---|---|
| persistence (official) | 0.285 |
| persistence_fill | 0.353 |
| rules | 0.663 |
| lgb_v2 | 0.788 |
| **lgb_v3** | **0.795** (onset 0.713, ongoing 0.877) |

## Task 4 (agent report, docs/traffic/TASK4_ANALYSIS.md)
The L2 projection of the split prior onto the split counts reproduces the published baseline S_ODME
(0.8357 vs 0.8359). Expected S_ODME is about 1.000.

## Expected S_total (local estimates)
| Task | Estimate | Weighted |
|---|---|---|
| state | ~0.935 | 0.327 |
| queue | ~0.75–0.79 | 0.225–0.237 |
| physics | ~(0.98 + 2·0.56)/3 ≈ 0.70 | 0.105 |
| ODME | ~1.0 | 0.200 |
| **total** | | **≈ 0.857–0.869** |

That compares with the best post-rebuild public score of 0.879. Our previous best was 0.80855.

## Leaderboard log (public = validation month, March 2031)

| Date (UTC) | Submission | Contents | Public | Rank | Note |
|---|---|---|---|---|---|
| 2026-09-22 | earlier account subs (v1–v5) | various, not from this pipeline | best 0.80855 | 38th/108 post-rebuild | baseline for comparison |
| 2026-09-23 13:42 | **A** `A_full_t1full1r25_t2lgbv3_t4l2proj.zip` | T1 pooled LGB full1 + density recon a=0.25; T2 lgb_v3; T4 L2 projection | **0.85204** | 21/145 overall, **13/115 post-rebuild** | +0.043 over previous best; about 0.005–0.017 below the local estimate (0.857–0.869); post-rebuild top 0.88153 |
| 2026-09-23 13:47 | **P1** probe_odme_only | T4 only (state, queue zeroed) | 0.19876 | – | **S_ODME = 0.9938.** The Task 4 projection hypothesis holds; the most Task 4 can still add is 0.0012 total |
| 2026-09-23 13:51 | **P2** probe_state_odme | A with queue zeroed | 0.62747 | – | A − P2 = 0.30·S_queue → **S_queue = 0.7486** (CV 0.795); P2 − P1 = 0.35·S_state + 0.15·S_phys = 0.42871 (local 0.432) → **S_phys ≈ 0.67–0.68** |
| 2026-09-23 13:57 | **P3** probe_onset_zeroed | A with onset windows zeroed | 0.74919 | – | onset = 2·(A − P3)/0.30 = **0.686** (CV 0.713); ongoing = **0.812** (CV 0.877) → most of the Task 2 transfer loss is in ongoing windows |

### Decomposition of A (0.85204), exact from the probes
| Task | Weighted | Task score | Local estimate |
|---|---|---|---|
| ODME | 0.19876 | 0.994 | 1.000 |
| queue | 0.22457 | 0.749 (onset 0.686, ongoing 0.812) | 0.795 (0.713 / 0.877) |
| state + physics | 0.42871 | S_state ≈ 0.935–0.94, S_phys ≈ 0.67–0.68 | 0.432 |

Where the headroom is:
- queue: +0.1 S_queue is worth +0.030 total. Ongoing robustness to incidents comes first.
- physics: +0.1 S_phys is worth +0.015. The density model and blackout cells are the levers.
- state: about +0.005 at most.
- ODME: done.

## Gated density reconciliation (2026-09-23)
Per speed band, the L1 density error (D12_I5_S holdout) shows where the density model is better:

| v/v_f | q/v | density model | FD(v) |
|---|---|---|---|
| < 0.4 | 2.99 | **1.50** | 1.77 |
| 0.4–0.6 | 3.40 | 2.82 | **2.64** |
| 0.6–0.8 | **4.83** | 5.10 | 5.50 |
| 0.8–0.9 | **2.85** | 3.73 | 5.94 |

The density model wins only in dense traffic, so reconciliation is gated to predicted v < 0.6·v_f.
Full-coverage holdout, realistic blackouts:

| Panel | J recon-all (A) | J gate 0.6 | Δ |
|---|---|---|---|
| D12_I5_S | 0.38307 | 0.38529 | +0.0022 |
| D7_I10_W | 0.38704 | 0.38767 | +0.0006 |
| D7_I405_S | 0.37741 | 0.37928 | +0.0019 |
| D12_I405_N | 0.39010 | 0.39184 | +0.0017 |

Gating improves J on 4/4 panels (mean +0.0016 total) and is adopted: `make_submission --gate 0.6`.
Adding FD density in 0.4–0.6·v_f changes J by −0.0002 to +0.0005, which is noise, so it is not adopted.
