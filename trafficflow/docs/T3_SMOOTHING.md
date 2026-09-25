# Task 3: dN-aware temporal smoothing of the Task 1 densities (2026-09-25)

**Result: adopt.** Total-variation (TV) smoothing of the per-lane density inside every run of
consecutive target cells raises the holdout J on **4/4 panels, mean +0.00062**. It adds LWR +0.0058
and S_state +0.00009, so S_state is not hurt. The parameters were chosen on two panels
(D12_I5_S, D7_I10_W) and confirmed on the other two (D7_I405_S +0.00047, D12_I405_N +0.00065).

Recommended make_submission flags (E1 + smoothing, a single-factor change):

```
python -m trafficflow.make_submission --state-tag full3 --recon-a 0.75 --gate 0.6 \
    --smooth "free=0.0075,free_a=0.001,gate=0.02,gate_a=0.005,dark=0.05" \
    --queue /home/user/work/t2/lgb_v6.csv --odme /home/user/work/t4/t4_l2proj.csv --out <file>.csv
```

- `--smooth default` is equivalent (`t1_smooth.DEFAULT`).
- Without `--smooth` the output is unchanged (the default is off).
- The run passes 65/65 checks on the real `state_full3` predictions.

J = 0.35·S_state + 0.10·S_LWR is the Task 1+3 part of S_total. Past Task 1 deltas matched their holdout
prediction to within 0.0002 on the leaderboard, so expect about **+0.0006 total**.

## Protocol and baseline reproduction
- Holdout: the same full-coverage protocol as the hold3 J check (`scratchpad/j3_eval.py`, EXPERIMENTS.md
  "Task 1 with FD features"):
  - train days 243–272, realistic blackouts (10 kept Task 2 origins, 18 dark rows each);
  - every target cell is predicted with the hold3 FD-feature models (`TFB_FD` column path `add_fd`, 2 threads).
- Predictions and the holdout-window arrays are cached per panel in
  `/home/user/work/t1/smooth/hold3_<panel>.npz` (13–16 MB). Scoring a variant then takes 0.02 s.
- The fast scorer `t1_smooth_eval.Hold` sums |dN_sub − dN_obs| only over transitions that touch a
  predicted cell. It agrees with `evaluate.s_lwr_proxy` to 3e-9.
- Baseline = the adopted post-processing: `reconcile(v, q, k̂, a=0.75)` where v < 0.6·v_f, then the
  make_submission clips. It reproduces the logged hold3 numbers exactly:

| Panel | S_state | LWR | J |
|---|---|---|---|
| D12_I5_S | 0.93549 | 0.6068 | 0.38810 |
| D7_I10_W | 0.94164 | 0.5905 | 0.38862 |
| D7_I405_S | 0.92084 | 0.5985 | 0.38215 |
| D12_I405_N | 0.94965 | 0.6055 | 0.39293 |

## Where the LWR error comes from (baseline)
Each transition's |dN error| is attributed to its target cells:
- **blackout**: either cell is in a dark row;
- **isolated**: a run of 1 target cell;
- **run 2–3 / run 4+**: longer runs of consecutive target cells;
- **boundary** transitions go to a known neighbour; **inner** transitions are target → target.

Numbers are shares of (1 − LWR).

| Category | D12_I5_S | D7_I10_W | D7_I405_S | D12_I405_N |
|---|---|---|---|---|
| blackout rows (boundary + inner) | 0.075 (0.062 + 0.013) | 0.092 (0.077 + 0.015) | 0.064 (0.051 + 0.013) | 0.049 (0.039 + 0.010) |
| isolated cells (2·\|e_t\|) | 0.455 | 0.461 | 0.468 | 0.475 |
| runs of 2–3 (boundary + inner) | 0.372 (0.172 + 0.201) | 0.356 (0.162 + 0.194) | 0.374 (0.170 + 0.204) | 0.378 (0.175 + 0.203) |
| runs of 4+ (boundary + inner) | 0.097 (0.022 + 0.075) | 0.091 (0.020 + 0.071) | 0.094 (0.020 + 0.073) | 0.098 (0.022 + 0.076) |
| of which dense traffic (a cell in the 0.6·v_f gate) | 0.256 | 0.049 | 0.211 | 0.112 |

- Blackout runs carry only **5–9 %** of the LWR error.
- Isolated cells carry about 46–48 %. For them only per-cell L1 accuracy matters, and no smoothing
  can help.
- Inner transitions, the only terms that smoothing targets directly, carry about **29 %**.

Target runs in train are short. On the D12_I5_S holdout, in R3, 37 % of target cells are isolated,
29 % are in runs of 2 and 17 % in runs of 3; only a handful of runs reach 12 cells or more.

## Diagnostic: are the predicted increments too wiggly?
The L1-optimal scale c for the predicted increments minimises Σ|c·dN_pred − dN_obs| per transition
type. c < 1 means that shrinking the increments (smoothing) helps.

| Transition, cells | D12_I5_S | D7_I10_W | D7_I405_S | D12_I405_N |
|---|---|---|---|---|
| inner, free flow | 0.80 | 0.75 | 0.82 | 0.85 |
| inner, gate | 0.65 | 0.72 | 0.54 | 0.43 |
| inner, blackout | 0.43 | 0.29 | 0.33 | 0.05 |
| boundary, free flow | 1.01 | 0.99 | 1.01 | 1.02 |
| boundary, gate | 0.93 | 0.85 | 0.90 | 0.79 |

- Boundary increments are calibrated (c ≈ 1). Pulling run ends toward the observed neighbours
  therefore hurts, and it did in every test.
- Inner increments are over-dispersed. Increments between two predictions are partly model jitter.
- For blackout boundary transitions, c is about 0.1 (Σ could be cut by 20–50 %), but it is not
  implementable: the non-target neighbour inside a dark row is scored with the truth yet is unknown
  to us.

## Method (`trafficflow/t1_smooth.py`)
1. Density per target cell: k = q_lane / max(v, 1), from the post-processed (gated, reconciled) v, q.
2. Maximal runs of consecutive target cells per link are grouped by run length and solved together.
   - For each run, 1-D TV denoising:
     `argmin_x ½‖x − k‖² + λ Σ|x_{i+1} − x_i| + λ_a (|x_1 − k_obs,left| + |k_obs,right − x_n|)`,
     with λ = τ·mean(run k) and λ_a = τ_a·mean(run k).
   - The solver is projected gradient on the dual, 300 iterations, converged to about 1e-6. It is exact
     against the closed form for n = 2 and against a multi-start brute force.
   - This soft-thresholds small within-run increments to 0 and keeps large ones (queue fronts), which
     is the L1-optimal response to symmetric jitter.
   - The L1 anchors move a run end only when it lies outside the observed neighbours' range. They are
     skipped when the neighbour is unobserved (blackout rows, other target cells).
3. Each cell takes the result of its category, with separate (τ, τ_a) for:
   - **blackout rows** (`dark`);
   - **gate** cells (raw speed < gate·v_f);
   - **free** flow.
4. (v, q) are moved to the new density (`apply_k`): v·r^−a, q·r^(1−a), r = k_new/k, |log r| ≤ 0.5.
   - In the gate a_in = 0.75 (the same split as the reconciliation).
   - Outside the gate a_out = 0: flow takes the change, and speed stays near v_f.
   - The make_submission clips follow.

Adopted `DEFAULT`: free τ = 0.0075, τ_a = 0.001; gate τ = 0.02, τ_a = 0.005; dark τ = 0.05, τ_a = 0;
a_in = 0.75, a_out = 0.

## Variants (full-coverage holdout, hold3, gate 0.6, a 0.75)
J per panel, with ΔJ × 1e4 vs the baseline in brackets:

| Variant | D12_I5_S | D7_I10_W | D7_I405_S | D12_I405_N | mean ΔJ | panels up |
|---|---|---|---|---|---|---|
| baseline (gate 0.6, a 0.75) | 0.38810 | 0.38862 | 0.38215 | 0.39293 | – | – |
| quadratic (Whittaker) λ_in 0.1 | 0.38827 (+1.7) | 0.38894 (+3.1) | 0.38231 (+1.7) | 0.39300 (+0.7) | +0.00018 | 4/4 |
| quadratic λ_in 0.1 + quadratic anchors 0.25 | 0.38646 (−16.4) | 0.38821 (−4.1) | 0.38057 (−15.8) | 0.39143 (−15.0) | −0.00128 | 0/4 |
| TV free 0.0075 only | 0.38842 (+3.2) | 0.38915 (+5.2) | 0.38245 (+3.0) | 0.39333 (+4.0) | +0.00039 | 4/4 |
| TV gate 0.02 only | 0.38837 (+2.6) | 0.38864 (+0.2) | 0.38225 (+1.0) | 0.39307 (+1.4) | +0.00013 | 4/4 |
| TV blackout 0.05 only | 0.38808 (−0.2) | 0.38868 (+0.6) | 0.38214 (−0.1) | 0.39294 (+0.1) | +0.00001 | 2/4 |
| TV free 0.0075 + gate 0.02 (no anchors) | 0.38869 (+5.8) | 0.38917 (+5.5) | 0.38255 (+4.0) | 0.39348 (+5.5) | +0.00052 | 4/4 |
| **TV adopted (DEFAULT)** | **0.38877 (+6.7)** | **0.38930 (+6.8)** | **0.38262 (+4.7)** | **0.39358 (+6.5)** | **+0.00062** | **4/4** |
| TV adopted, a_out 1 (move speed in free flow) | 0.38840 (+3.0) | 0.38881 (+1.8) | 0.38237 (+2.2) | 0.39308 (+1.5) | +0.00021 | 4/4 |
| TV adopted, gate cells only (a_out none) | 0.38841 (+3.1) | 0.38866 (+0.4) | 0.38230 (+1.5) | 0.39312 (+1.9) | +0.00017 | 4/4 |

S_state and LWR:

| Variant | S_state D12_I5_S / D7_I10_W / D7_I405_S / D12_I405_N | mean ΔS_state | LWR (same order) | mean ΔLWR |
|---|---|---|---|---|
| baseline | 0.93549 / 0.94164 / 0.92084 / 0.94965 | – | 0.6068 / 0.5905 / 0.5985 / 0.6055 | – |
| quadratic λ_in 0.1 | 0.93547 / 0.94169 / 0.92074 / 0.94948 | −0.00006 | 0.6086 / 0.5934 / 0.6005 / 0.6068 | +0.0020 |
| quadratic + anchors 0.25 | 0.93287 / 0.94123 / 0.91780 / 0.94767 | −0.00201 | 0.5996 / 0.5878 / 0.5934 / 0.5975 | −0.0058 |
| TV free only | 0.93558 / 0.94176 / 0.92091 / 0.94974 | +0.00009 | 0.6097 / 0.5953 / 0.6013 / 0.6092 | +0.0036 |
| TV free + gate | 0.93559 / 0.94176 / 0.92091 / 0.94974 | +0.00009 | 0.6123 / 0.5955 / 0.6023 / 0.6107 | +0.0049 |
| **TV adopted** | **0.93559 / 0.94179 / 0.92087 / 0.94975** | **+0.00009** | **0.6132 / 0.5967 / 0.6032 / 0.6116** | **+0.0058** |
| TV adopted, a_out 1 | 0.93453 / 0.94038 / 0.92015 / 0.94833 | −0.00106 | 0.6132 / 0.5967 / 0.6032 / 0.6116 | +0.0058 |

**What passed and what failed:**
- **TV (adopted): passes.** J is up on 4/4 panels, and S_state is up on 4/4.
- **Quadratic smoothing: rejected.** Only λ_in ≤ 0.1 helps (+0.00018). On the D12_I5_S grid, any
  λ_in ≥ 0.25 loses (λ_in 1 costs −0.0013), and so does any quadratic anchoring (−0.0003 to −0.0049).
  It shrinks every increment, including real ones.
- **Anchoring run ends to the observed neighbours** (quadratic, or L1 with τ_a ≥ 0.01 in free flow):
  **rejected.** It follows from the calibrated boundaries.
- **Blackout-row smoothing:** neutral (+0.00001). It is kept in DEFAULT at τ = 0.05, which was chosen on
  the selection panels; it is harmless.
- **Moving speed instead of flow in free flow (a_out 1): rejected.** It gives the same LWR, but S_state
  drops by 0.001.
- **Other density post-processings, tested as diagnostics on D12_I5_S:** all rejected.
  - Reconciling blackout cells beyond the gate: J −0.0005 to −0.0008.
  - Blending q/v toward the density model outside the gate: w 0.25 +0.00006, w 0.5 −0.0005.
  - A global ±0.5–1 % density scale outside the gate: −0.0003 to −0.0017. There is no L1 bias.

Where the adopted TV's gain comes from, as the change in (1 − LWR) × 1e4 per panel
(D12_I5_S / D7_I10_W / D7_I405_S / D12_I405_N):

| Category | Change |
|---|---|
| inner transitions, runs of 2–3 | −31.6 / −28.3 / −22.0 / −30.2 |
| inner transitions, runs of 4+ | −16.6 / −13.2 / −11.3 / −12.4 |
| run boundaries (anchors) | −9.5 / −10.3 / −7.0 / −11.0 |
| isolated cells (anchors) | −7.5 / −4.8 / −5.1 / −5.4 |
| blackout rows | +1.7 / −6.1 / −0.9 / −2.2 |

## Selection and robustness
- **Selection** (`scratchpad/sm_select.py`): 90 configurations on D12_I5_S + D7_I10_W, each τ/τ_a per
  category from {free τ 0.005, 0.0075, 0.01} × {free τ_a 0, 0.001, 0.0025} × {gate τ 0, 0.01, 0.02} ×
  {gate τ_a 0, 0.005} × {dark τ 0, 0.05}.
  - Every configuration gained, from +3.9e-4 to +6.7e-4 mean ΔJ.
  - DEFAULT was the best mean (+6.74e-4), and the second-best setting tied it.
- **Confirmation panels** (not seen during selection): D7_I405_S +4.7e-4, D12_I405_N +6.5e-4.
- **Sensitivity on all four panels** (`sensitivity.csv`, not used for re-selection):
  - Every TV setting in the neighbourhood (free 0.005–0.01, gate 0.01–0.03, dark 0 or 0.05) gains
    +4.1e-4 to +7.0e-4 **on every panel**.
  - DEFAULT has the best four-panel mean (+6.17e-4).
  - The anchors add +0.9e-4 mean (5.28 → 6.17e-4) and help on 4/4 panels.
- **Test-set behaviour matches the holdout:**

  | | Share of cells changed | Mean relative density change (changed cells) | Mean flow change |
  |---|---|---|---|
  | Holdout | 74–76 % | 0.39–0.43 % | 11–12 veh/h |
  | Test | 76 % | 0.41 % | 11 veh/h |

## Gate threshold × reconciliation split (coordinator request)
Full-coverage holdout, hold3. J per panel, with ΔJ × 1e4 vs the current setting (gate 0.6, a 0.75, no
smoothing) in brackets. "+ TV" uses DEFAULT smoothing, whose gate category follows the same gate.

| Setting | D12_I5_S | D7_I10_W | D7_I405_S | D12_I405_N | mean ΔJ | panels up |
|---|---|---|---|---|---|---|
| gate 0.5, a 0.75 | 0.38780 (−3.0) | 0.38837 (−2.5) | 0.38194 (−2.1) | 0.39299 (+0.6) | −0.00017 | 1/4 |
| gate 0.5, a 0.75 + TV | 0.38847 (+3.6) | 0.38905 (+4.2) | 0.38244 (+2.9) | 0.39365 (+7.2) | +0.00045 | 4/4 |
| gate 0.5, a 1.0 | 0.38786 (−2.4) | 0.38841 (−2.1) | 0.38189 (−2.6) | 0.39299 (+0.6) | −0.00016 | 1/4 |
| gate 0.5, a 1.0 + TV | 0.38852 (+4.2) | 0.38909 (+4.7) | 0.38239 (+2.4) | 0.39365 (+7.2) | +0.00046 | 4/4 |
| **gate 0.6, a 0.75 (current)** | 0.38810 | 0.38862 | 0.38215 | 0.39293 | – | – |
| **gate 0.6, a 0.75 + TV** | 0.38877 (+6.7) | 0.38930 (+6.8) | 0.38262 (+4.7) | 0.39358 (+6.5) | +0.00062 | 4/4 |
| gate 0.6, a 1.0 | 0.38814 (+0.3) | 0.38868 (+0.6) | 0.38202 (−1.3) | 0.39291 (−0.2) | −0.00001 | 2/4 |
| gate 0.6, a 1.0 + TV | 0.38881 (+7.1) | 0.38936 (+7.4) | 0.38249 (+3.4) | 0.39356 (+6.3) | +0.00060 | 4/4 |
| gate 0.7, a 0.75 | 0.38792 (−1.8) | 0.38880 (+1.8) | 0.38218 (+0.3) | 0.39307 (+1.4) | +0.00004 | 3/4 |
| gate 0.7, a 0.75 + TV | 0.38860 (+4.9) | 0.38947 (+8.5) | 0.38266 (+5.1) | 0.39371 (+7.8) | **+0.00066** | 4/4 |
| gate 0.7, a 1.0 | 0.38785 (−2.5) | 0.38886 (+2.4) | 0.38199 (−1.6) | 0.39307 (+1.4) | −0.00001 | 2/4 |
| gate 0.7, a 1.0 + TV | 0.38852 (+4.2) | 0.38953 (+9.1) | 0.38246 (+3.2) | 0.39371 (+7.8) | +0.00061 | 4/4 |
| gate 0.8, a 0.75 | 0.38727 (−8.4) | 0.38866 (+0.4) | 0.38184 (−3.0) | 0.39301 (+0.8) | −0.00025 | 2/4 |
| gate 0.8, a 0.75 + TV | 0.38795 (−1.5) | 0.38934 (+7.1) | 0.38233 (+1.9) | 0.39366 (+7.3) | +0.00037 | 3/4 |
| gate 0.8, a 1.0 | 0.38702 (−10.9) | 0.38865 (+0.3) | 0.38149 (−6.5) | 0.39299 (+0.6) | −0.00041 | 2/4 |
| gate 0.8, a 1.0 + TV | 0.38770 (−4.0) | 0.38933 (+7.0) | 0.38198 (−1.6) | 0.39364 (+7.1) | +0.00021 | 2/4 |

S_state and LWR for the same settings are in `/home/user/work/t1/smooth/variants.csv`
(`python -m trafficflow.t1_smooth_eval grid` prints all three tables). The main facts:
- a = 1.0 without smoothing, S_state vs current: gate 0.5 +0.00008, gate 0.6 −0.00002, gate 0.7 −0.00020.
- gate 0.7 costs S_state −0.00006 and gains LWR +0.0007.
- gate 0.8 costs S_state −0.0007 to −0.0012.

Reading of the gate grid:
- **a = 1.0 does not pass** at any gate: vs current, 1/4, 2/4, 2/4 and 2/4 panels go up. At the same
  gate (with or without TV) it is within ±0.00002 of a = 0.75 at gates 0.5–0.6 and worse at 0.7–0.8.
- **gate 0.5 and 0.8 fail.**
- **gate 0.7 with a 0.75 formally passes** the rule, both alone (3/4, mean +0.00004 vs gate 0.6) and on
  top of TV (gate 0.7 + TV vs gate 0.6 + TV: −1.7 / +1.7 / +0.4 / +1.3 e-4, 3/4, mean +0.00004).
  - This increment is at noise level. D12_I5_S loses 0.00017, and gate 0.65 gives yet another pattern
    (+0.0 / +0.8 / +1.0 / −0.1 e-4 on top of TV).
  - An expected gain of +0.00004 is below the 0.0002 accuracy of the local-to-leaderboard transfer.
- The best combination by the rule is therefore **gate 0.7, a 0.75 + TV** (+0.00066). Its flags:
  `--state-tag full3 --recon-a 0.75 --gate 0.7 --smooth "free=0.0075,free_a=0.001,gate=0.02,gate_a=0.005,dark=0.05"`.
  It also passes 65/65 checks.
- **Recommendation: gate 0.6 + TV** (+0.00062, every panel ≥ +0.00047). It is a single-factor change
  from E1 and keeps the gate that the leaderboard has already validated. Gate 0.7 adds only a
  noise-level expected +0.00004.

## Check on the real test predictions
- The recommended command, run with `--out /home/user/work/subs/tmp_smooth_check.csv`, passed
  **65 checks, 0 failed**:
  - speed 3.00–130.00, mean 111.26;
  - flow 355–11,506 veh/h;
  - no NaN;
  - FD empty-road guard at share 0.0000 everywhere.
- Diff against the E1 upload (`E1_fd_gate06a75_t2v6.zip`): the queue and path_flow columns are
  identical. The 6,735,795 state rows move by a mean |Δv| of 0.0066 km/h and a mean |Δq| of 11.05 veh/h,
  and 75 % of them change.
- The gate-0.7 variant also passed 65/65 checks.
- Both scratch CSVs and their checks.json were deleted afterwards.
- Cost: the smoothing adds about 150 s to `state_frame` for the 6.7 M rows. Peak RSS was 3.8 GB for a
  process that built both the smoothed and unsmoothed frames.

## Reproduce
```
cd /home/user/knee
# hold3 predictions on the full-coverage holdout (about 5 min per panel, 2 threads, about 1.6 GB RSS)
OMP_NUM_THREADS=2 PYTHONPATH=. python -m trafficflow.t1_smooth_eval cache D12_I5_S D7_I10_W D7_I405_S D12_I405_N
# error shares, baseline and with the adopted smoothing (-> WORK/smooth/share.csv)
OMP_NUM_THREADS=2 PYTHONPATH=. python -m trafficflow.t1_smooth_eval share
# variant table + gate x split grid (-> WORK/smooth/variants.csv), about 2 min
OMP_NUM_THREADS=2 PYTHONPATH=. python -m trafficflow.t1_smooth_eval grid
# submission (recommended)
OMP_NUM_THREADS=2 PYTHONPATH=. python -m trafficflow.make_submission --state-tag full3 --recon-a 0.75 --gate 0.6 \
    --smooth "free=0.0075,free_a=0.001,gate=0.02,gate_a=0.005,dark=0.05" \
    --queue /home/user/work/t2/lgb_v6.csv --odme /home/user/work/t4/t4_l2proj.csv --out /home/user/work/subs/<name>.csv
```

Files:
- `trafficflow/t1_smooth.py`: `DEFAULT`, `tv_rows`, `tv_runs`, `smooth_cells`, `smooth_frame`, `apply_k`,
  `parse`, and `smooth_runs` (quadratic, not adopted).
- `trafficflow/t1_smooth_eval.py`: the cache / share / grid stages.
- `trafficflow/make_submission.py`: `--smooth`, off by default.
- Results in `/home/user/work/t1/smooth/`:
  - `variants.csv`, `share.csv`, `sensitivity.csv`, `increment_scale.csv`;
  - `select_D12_I5_S_D7_I10_W.csv` (selection grid);
  - `explore{1,2,3}.csv` (the first D12_I5_S / D7_I10_W explorations).
- The exploration and selection scripts are in the session scratchpad (`sm_explore*.py`, `sm_select.py`,
  `sm_sens.py`).
