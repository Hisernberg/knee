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
| 2026-09-23 16:23 | **B1** `B1_gate06_t2v4robust.zip` | A + density reconciliation gated to v<0.6·v_f; T2 lgb_v4_robust (ongoing = 50/50 blend of v3 and the no-location model; onset identical to v3) | **0.85732** | 14/146 overall, **8/116 post-rebuild** | +0.0053 vs A (local expectation +0.002 to +0.004: gating +0.0016, ongoing robustness +0.001 to +0.003). The Task 1/3 and Task 2 parts of the gain are not separated on the LB; post-rebuild top 0.88153 |
| 2026-09-24 12:02 | **S1 C1** `C1_gate06_t2v5.zip` | B1 with queue v5 (279 queue cells) | **0.85975** | – | +0.00243 vs B1 → ΔS_queue = **+0.0081** (CV +0.005). Base → C1 |
| 2026-09-24 12:08 | **S2 D1** `D1_gate06_t2v6.zip` | C1 with v6 onset trained on corrected labels (42 onset cells) | **0.86428** | – | +0.00453 vs C1 → ΔS_queue = +0.0151, **onset +0.030** on March. Base → D1 |
| 2026-09-24 12:11 | **S3 D2** `D2_gate06a75_t2v6.zip` | D1 with gate split a = 0.75 (266,914 dense-traffic state rows) | **0.86492** | 14/158 overall, **8/128 post-rebuild** | +0.00064 vs D1, **exactly the local prediction (+0.0006)**. Base → D2; post-rebuild top 0.88318 |
| 2026-09-24 14:40 | **S4 E1** `E1_fd_gate06a75_t2v6.zip` | D2 with Task 1 state from the FD-feature models `full3` (state rows only) | **0.86591** | 14/160 overall, **8/130 post-rebuild** | +0.00099 vs D2 (local +0.0012). Base → E1; post-rebuild top 0.88318 |
| 2026-09-24 14:42 | **S5 P4** probe `P4_E1_onset_zeroed.zip` | E1 with onset windows zeroed (311 cells) | 0.75669 | – | **onset = 2·(E1 − P4)/0.30 = 0.728** on March (A: 0.686) |
| 2026-09-25 08:18 | **F1** probe `F1_onsetb05.zip` | E1 with the v6 onset re-decoded at logit bias +0.5 (+15 onset cells; 12 in 8 validation windows) | 0.86356 | – | −0.00235 → **March onset −0.016**: larger onset sets hurt |
| 2026-09-25 08:44 | **F2** `F2_onset_site2.zip` | E1 with the onset decoded by `site2_lo.05_r.5` (commit to 1–2 sites; −11 hedge cells, 9 in 7 validation windows) | 0.86418 | 9/140 post-rebuild (E1) | −0.00173 → **March onset −0.012**: fewer hedges hurt too. Base stays E1 |

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

## Task 1/3 v2 round (2026-09-23 evening)
Full-coverage holdout, realistic blackouts, gate 0.6. J = 0.35·S_state + 0.10·S_LWR (the Task 1+3 part of S_total).

| Panel | hold1 (submitted) | hold2 (all new) | + longer blackout speed/flow only | + L1-leaning density only |
|---|---|---|---|---|
| D12_I5_S | **0.38529** | 0.38422 | 0.38530 | 0.38419 |
| D7_I10_W | **0.38767** | 0.38747 | 0.38753 | 0.38760 |
| D7_I405_S | 0.37928 | 0.37900 | **0.37938** | 0.37890 |
| D12_I405_N | 0.39184 | 0.39157 | **0.39195** | 0.39144 |

- The L1-leaning density model (Huber α 0.2, 127 leaves) is worse on 4/4 panels, so it is **rejected**. The near-L2 density (α 1) is better for S_LWR.
- Longer-trained blackout models: mean +0.00002 total, which is noise and not worth a retrain.

**Speed/flow split inside the density gate.** LWR is unchanged, since q/v = k̂ for any a:

| Panel | a = 0.25 (A, B1) | a = 0.75 | a = 1.0 |
|---|---|---|---|
| D12_I5_S | 0.93180 | **0.93429** | 0.93429 |
| D7_I10_W | 0.93996 | 0.94091 | **0.94109** |
| D7_I405_S | 0.91588 | **0.91845** | 0.91819 |
| D12_I405_N | 0.94800 | **0.94934** | 0.94930 |

a = 0.75 improves S_state on 4/4 panels (mean +0.0018, about +0.0006 total). In queues, flow sits near discharge capacity while speed carries the uncertainty. **Adopted** (`--recon-a 0.75 --gate 0.6`).

## Task 2 v5 (2026-09-23 evening, agent)
CV on sim / official windows. Details in docs/traffic/TASK2_ANALYSIS.md section 12.

| Slice | v4 (in B1) | v5 |
|---|---|---|
| overall | 0.7971 / 0.8233 | **0.8022 / 0.8299** |
| onset | 0.7133 | **0.7210** |
| ongoing | 0.8809 | **0.8833** |
| non-recurrent onset / ongoing (recur < 0.05) | 0.275 / 0.588 | **0.295 / 0.599** |

## Candidates for 2026-09-24 (single-factor chain, each passes 65/65 checks)
| File | Differs from | Change | Expected Δ |
|---|---|---|---|
| `C1_gate06_t2v5.zip` | B1 | Task 2 v4 → v5 (279 queue cells) | about +0.0015 (CV × 0.30), possibly more on March |
| `C2_gate06a75_t2v5.zip` | C1 | gate split a 0.25 → 0.75 (266,914 dense-traffic state cells) | about +0.0006 |

## Task 1 with FD (fundamental-diagram) features, hold3 (2026-09-24)
Column-based FD features (`TFB_FD=1`, `t1_pipeline.add_fd`, verified identical to the native path): congested-branch flow implied by speed, speed implied by flow, and q/v density for the interpolated/previous/next values, plus `fd_w`, `fd_kj` and v/v_f.

Holdout RMSE, hold1 → hold3:

| Model | hold1 | hold3 |
|---|---|---|
| speed | 1.543 | 1.515 |
| flow | 30.57 | 30.19 |
| density | 0.567 | 0.541 |
| dark speed | 6.903 | 6.888 |
| dark flow | 65.86 | 65.70 |
| dark density | 3.21 | 3.19 |

Full-coverage holdout, gate 0.6, a = 0.75:

| Panel | J hold1 | J hold3 | Δ | LWR hold1 → hold3 |
|---|---|---|---|---|
| D12_I5_S | 0.38618 | 0.38810 | +0.0019 | 0.5916 → 0.6068 |
| D7_I10_W | 0.38800 | 0.38862 | +0.0006 | 0.5868 → 0.5905 |
| D7_I405_S | 0.38044 | 0.38215 | +0.0017 | 0.5896 → 0.5985 |
| D12_I405_N | 0.39232 | 0.39293 | +0.0006 | 0.6005 → 0.6055 |

Better on 4/4 panels (mean +0.0012 total), with gains from both the speed/flow and the density models. **Adopted.** Full-data fit `full3` is in progress.

## Day summary 2026-09-24: B1 0.85732 → E1 0.86591 (+0.0086)
Every step changed a single, locally validated factor. Each Task 1 delta matched its holdout prediction to within 0.0002.

| Step | Change | LB Δ | Local prediction |
|---|---|---|---|
| S1 | queue v5 (physics + shockwave features) | +0.00243 (ΔS_queue +0.0081) | CV +0.005 × 0.30 |
| S2 | onset v6 (trained on corrected labels) | +0.00453 (onset +0.030) | CV onset +0.005 to +0.012 |
| S3 | gate split a = 0.75 | +0.00064 | +0.0006 |
| S4 | Task 1 FD-feature models | +0.00099 | +0.0012 |

**Decomposition of E1 (0.86591).** Only the gating share of B1 is estimated locally; everything else is exact from probes and single-factor deltas.

| Task | Weighted | Task score |
|---|---|---|
| ODME | 0.19876 | 0.994 |
| Task 1 + 3 | ≈ 0.4319 | 0.4287 (A) + gating 0.0016 + split 0.0006 + FD 0.0010 |
| queue | ≈ 0.2352 | S_queue ≈ 0.784: **onset 0.728 (exact)**, ongoing ≈ 0.840 |

The ongoing retrain on corrected labels (v7) was rejected: +0.0006 in the conservative evaluation, and the non-recurrent slice got worse. The label bias sits at queue formation, not inside established queues.

**Next lever: onset.**
- March onset is 0.728, against about 0.76 in CV (old truth) and about 0.86 (corrected truth), so March onsets transfer worst. Diagnose March onset windows next (site recurrence, time of day, incident-like sites) with history-only features.
- Each +0.1 onset is worth +0.015 total. The gap to the leader is 0.0173.

## Decoding calibration (2026-09-25)
**Onset, v6 hybrid OOF on re-drawn windows** (`python -m trafficflow.t2.calib cv` / `decoders`, T2_WORK=t2h):

| logit bias b | hybrid sim | hybrid off | old sim | old off | rec<0.05 | rec<0.2 | cells/window |
|---|---|---|---|---|---|---|---|
| −0.5 | 0.8563 | 0.8841 | 0.7844 | 0.7943 | 0.394 | 0.691 | 3.92 |
| **0 (v6)** | **0.8589** | 0.8841 | **0.7871** | 0.7943 | 0.412 | 0.704 | 4.00 |
| +0.25 | 0.8594 | 0.8841 | 0.7869 | 0.7943 | 0.418 | 0.707 | 4.04 |
| +0.5 | 0.8588 | 0.8928 | 0.7862 | 0.8038 | 0.419 | 0.707 | 4.07 |
| +1.0 | 0.8554 | 0.8970 | 0.7824 | 0.8088 | 0.434 | 0.713 | 4.16 |

- Site-commit decoders cost −0.001 to −0.002. `site2_lo.05_r.5` is −0.0010 ± 0.0012 hybrid and −0.0009 ± 0.0011 old (paired, 94 windows changed: 44 better, 50 worse).
- The contiguous-range and anchor decoders lose heavily (0.795 and 0.562 hybrid).

**On the LB, both directions lose on March:**
- b = +0.5 adds hedge cells: onset −0.016.
- `site2` removes them: onset −0.012.
- **Conclusion:** the v6 top-m decoder at b = 0 is at the March optimum. Onset gains have to come from better probabilities.
- The added cells were mostly extra sites in uncertain windows (D12_I5_N, D12_I5_S, D7_I405_S). Removing the model's own hedges also hurts, so the hedges it picks do hit.

**Ongoing, v5 blend OOF, original windows, old truth** (`calib cv_ongoing`):

| b | −0.5 | −0.25 | **0** | +0.25 | +0.5 | +1.0 |
|---|---|---|---|---|---|---|
| sim | 0.8797 | 0.8827 | **0.8833** | 0.8820 | 0.8788 | 0.8659 |
| off | 0.8790 | 0.8866 | 0.8906 | 0.8950 | 0.8960 | 0.8889 |

- The official train windows prefer larger sets for both conditions. That preference did not transfer to the LB for onset, so it is an artefact of our truth on those windows (5-minute selector shifts), not a property of the official truth.
- No ongoing bias probe was spent.
