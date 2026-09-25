# Task 1: seed ensemble of the LightGBM state models (2026-09-25)

**Result: adopt.** Averaging the raw predictions (speed, flow, density) of the current FD-feature models
(`hold3`/`full3`, seed 0) with a second member trained with another seed (`hold4`/`full4`, `TFB_SEED=1`)
raises the holdout J on **4/4 panels, mean +0.00077**, with the full adopted post-processing (gate 0.6,
a = 0.75, TV smoothing). The second member alone is as good as the first (mean ΔJ +0.00001), so the gain
comes from variance reduction.

Gate: the average must improve J on at least 3 of 4 panels and on the mean. It improves J on 4/4
panels (+0.00063 to +0.00092), so the gate **passes**.

The ensemble state prediction is `/home/user/work/t1/pred/state_ens34.parquet`, the element-wise mean
of `state_full3` and `state_full4`. It is a single-factor change: swap `--state-tag full3` for `ens34`
and keep every other flag of the current best:

```
python -m trafficflow.make_submission --state-tag ens34 --recon-a 0.75 --gate 0.6 --smooth default \
    --queue <current best queue csv> --odme /home/user/work/t4/t4_l2proj.csv --out <file>.csv
```

FULL4_CHECK_PLACEHOLDER

J = 0.35·S_state + 0.10·S_LWR is the Task 1+3 part of S_total. The Task 3 proxy has predicted leaderboard
deltas to within 0.0002, so expect about **+0.0008 total**.

## The seed knob (`TFB_SEED`, trafficflow/t1_pipeline.py)
- `TFB_SEED=s` changes two things:
  - the CAP row sample in `load_train` (rng seed s instead of 0);
  - the LightGBM seeds (`lgb_seeds`): `seed=s`, `data_random_seed=100s+1`, `feature_fraction_seed=100s+2`,
    `bagging_seed=100s+3`.
- The default s = 0 is the old code path bit-for-bit. PARAMS are identical, `load_train` gets seed 0, and
  a test fit with the new PARAMS equals one with the old literal exactly. LightGBM then uses its default
  seeds (1/2/3), as recorded in the hold3/full3 model files.
- Seed 1 keeps 57 % of seed 0's regular training rows and 48 % of its blackout training rows. The
  early-stopping holdout rows overlap by 70 % (regular) and 100 % (blackout: every holdout row is used).
  So the second member is a bagging member, not just a reseed.

hold4 is the hold3 recipe (`TFB_FD=1`, 2 threads, all other knobs at their defaults, as recorded in the
hold3 model files) with `TFB_SEED=1`. Best iteration and RMSE on each fit's own early-stopping sample
(the samples differ, so the RMSEs are not a like-for-like comparison):

| Model | hold3 best iter | hold4 best iter | hold3 RMSE | hold4 RMSE |
|---|---|---|---|---|
| reg_speed | 829 | 715 | 1.515 | 1.513 |
| reg_flow | 944 | 1202 | 30.19 | 30.21 |
| reg_dens | 2986 | 3000 | 0.541 | 0.549 |
| dark_speed | 2992 | 3000 | 6.888 | 6.903 |
| dark_flow | 2969 | 2233 | 65.70 | 66.08 |
| dark_dens | 3000 | 3000 | 3.190 | 3.189 |

## Holdout evaluation
Protocol: the full-coverage holdout of trafficflow/docs/T3_SMOOTHING.md (train days 243–272, realistic
blackouts, every target cell). The steps:
- `t1_smooth_eval preds hold4` rebuilds the features of the cached cells and predicts them with hold4.
- It also recomputes the hold3 predictions from those features. They equal the cache **bit for bit on
  all 4 panels**, so both members are scored on identical cells and features.
- `t1_smooth_eval ens hold3,hold4` scores every subset of members. A subset is the element-wise mean of
  its members' raw speed / flow / density, taken before any post-processing, exactly as `state_ens34`
  is built.
- The hold3 row reproduces the logged baseline exactly.

**Adopted post-processing** (reconcile at v < 0.6·v_f with a = 0.75, then TV smoothing `default`):

| Panel | J hold3 | J hold4 | J mean(3, 4) | ΔJ mean − hold3 | S_state hold3 → mean | LWR hold3 → mean |
|---|---|---|---|---|---|---|
| D12_I5_S | 0.38877 | 0.38894 | **0.38969** | **+0.00091** | 0.93559 → 0.93667 | 0.6132 → 0.6185 |
| D7_I10_W | 0.38930 | 0.38910 | **0.38999** | **+0.00069** | 0.94179 → 0.94227 | 0.5967 → 0.6019 |
| D7_I405_S | 0.38262 | 0.38262 | **0.38347** | **+0.00084** | 0.92087 → 0.92169 | 0.6032 → 0.6087 |
| D12_I405_N | 0.39358 | 0.39363 | **0.39421** | **+0.00064** | 0.94975 → 0.95007 | 0.6116 → 0.6169 |
| mean | 0.38857 | 0.38857 | **0.38934** | **+0.00077 (4/4)** | +0.00068 | +0.0053 |

hold4 alone vs hold3 (ΔJ): D12_I5_S +0.00017, D7_I10_W −0.00020, D7_I405_S −0.00000, D12_I405_N +0.00006,
mean +0.00001. S_state −0.00018 and LWR +0.0007 on average.

**Without the TV smoothing** (gate and split only):

| Panel | J hold3 | J hold4 | J mean(3, 4) | ΔJ mean − hold3 |
|---|---|---|---|---|
| D12_I5_S | 0.38810 | 0.38830 | 0.38929 | +0.00119 |
| D7_I10_W | 0.38862 | 0.38837 | 0.38953 | +0.00091 |
| D7_I405_S | 0.38215 | 0.38213 | 0.38323 | +0.00109 |
| D12_I405_N | 0.39293 | 0.39284 | 0.39383 | +0.00090 |
| mean | | | | +0.00102 (4/4) |

- Averaging and the TV smoothing overlap. Both remove model jitter from the density increments.
- The TV gain drops from +0.00062 on hold3 to +0.00037 on the mean. Stacked, the two still add up:
  hold3 (no TV) → mean + TV is +0.00139.
- The TV thresholds were tuned on hold3 and were not re-tuned for the mean (see below).

**Raw RMSE at the holdout target cells** (all cells, before post-processing; speed km/h, flow veh/h/lane,
density veh/km/lane; mean over the 4 panels):

| | speed | flow | density |
|---|---|---|---|
| hold3 | 1.930 | 31.02 | 0.795 |
| hold4 | 1.941 | 31.08 | 0.793 |
| mean(3, 4) | **1.906** | **30.54** | **0.788** |
| RMS difference hold3 − hold4 | 0.663 | 11.25 | 0.187 |

The two members disagree by about a third of their error (speed 0.66 against 1.93 km/h). For independent
seed noise, the expected MSE reduction is var/2 = (0.66²/2)/2 ≈ 0.11 km²/h². The observed reduction is
1.936² − 1.906² ≈ 0.11, which matches.

**Do the TV thresholds need re-tuning for the mean?** All five thresholds of `t1_smooth.DEFAULT` were
scaled by f. The table shows J on the mean(3, 4) predictions (results in
`/home/user/work/t1/smooth/ens_tv_sens.csv`). To reproduce:

```python
from trafficflow.t1_smooth import DEFAULT
from trafficflow.t1_smooth_eval import Hold
H = Hold("D12_I5_S", ("hold3", "hold4")); v, q, g = H.base()
f = 0.75
spec = DEFAULT | {k: DEFAULT[k] * f for k in ("free", "free_a", "gate", "gate_a", "dark")}
H.score(*H.smooth(v, q, g, **spec))
```

| f | D12_I5_S | D7_I10_W | D7_I405_S | D12_I405_N | mean ΔJ vs f = 1 |
|---|---|---|---|---|---|
| off | 0.38929 | 0.38953 | 0.38323 | 0.39383 | −0.00037 |
| 0.5 | 0.38970 | 0.38994 | 0.38351 | 0.39423 | +0.00001 |
| 0.75 | 0.38972 | 0.38998 | 0.38351 | 0.39425 | +0.00003 |
| **1 (default)** | 0.38969 | 0.38999 | 0.38347 | 0.39421 | 0 |
| 1.25 | 0.38964 | 0.38996 | 0.38340 | 0.39416 | −0.00005 |
| 1.5 | 0.38957 | 0.38991 | 0.38332 | 0.39409 | −0.00011 |

The optimum moves slightly toward weaker smoothing (f = 0.75: +0.00003 on 3/4 panels), as expected for
smoother inputs. On hold3 alone, f = 1 remains the optimum. The difference is noise, so the ensemble
keeps `--smooth default`.

## Is a third seed worth it?
Mean J (adopted post-processing) over all member subsets of size k:

| Panel | k = 1 (mean of hold3, hold4) | k = 2 | projected k = 3 |
|---|---|---|---|
| D12_I5_S | 0.38886 | 0.38969 | 0.38996 |
| D7_I10_W | 0.38920 | 0.38999 | 0.39025 |
| D7_I405_S | 0.38262 | 0.38347 | 0.38375 |
| D12_I405_N | 0.39360 | 0.39421 | 0.39442 |
| mean | 0.38857 | 0.38934 | 0.38959 |

- The fit uses J_M = J_∞ − c/M. With the 1 → 2 gain of +0.00077, it projects **+0.00026** for a third
  member (a third of the 1 → 2 gain) and +0.00013 for a fourth.
- A third seed costs about 2.5–3 h at 2 threads (see Timings).
- +0.00026 is at the resolution of the Task 3 proxy (0.0002). Worth it only when the CPU is otherwise
  idle, e.g. overnight. Run `TFB_SEED=2` → hold5/full5, then `ens hold3,hold4,hold5`. The same stage then
  measures 2 → 3 directly.

## full4 and the test ensemble
- **Rounds.** full4 uses the rule behind full3 (`t1_pipeline.full_rounds`, stage `rounds`): 1.1 × the
  hold best iteration, rounded to 10. A model whose early stopping ran into the 3000-round cap
  (best ≥ 2980) gets 3300.
  - Applied to the hold3 report, the rule reproduces the full3 tree counts exactly (910, 1040, 3300,
    3300, 3270, 3300).
  - From hold4: `{"reg_speed": 790, "reg_flow": 1320, "reg_dens": 3300, "dark_speed": 3300,
    "dark_flow": 2460, "dark_dens": 3300}`.
  - The full4 model files carry these tree counts and seed 1's LightGBM seeds.
- **Ensemble.** `predict --tag full4` writes `state_full4.parquet`. Then
  `ens --tag ens34 --members full3 full4` writes `state_ens34.parquet`:
  - It asserts that both files have the same columns and identical key columns (panel, t, link_id,
    regime, kind, lanes) in the same row order.
  - It averages speed, flow_lane and dens_lane. The schema, dtypes and index are unchanged, so
    make_submission reads it like any other state tag.
- ENS_TEST_PLACEHOLDER
- `seed.json` in a model directory records a non-zero seed (hold4, full4). `t1_holdout.score` reads it so
  that the `hold_*.npy` rows line up with `load_train(..., seed)`. Seed-0 tags have no file and behave
  as before.

## Timings and resources
2 threads (`TFB_THREADS=2`, `TFB_PRED_THREADS=2`, `OMP_NUM_THREADS=2`), on a machine shared with the
Task 2 agent:

| Step | Wall | Peak RSS |
|---|---|---|
| hold4 train (6 models, early stopping) | 75.6 min | 5.70 GB |
| hold4 predictions at the holdout cells (4 panels; includes the hold3 recomputation check, about half the time) | 31 min | 1.31 GB |
| `ens` evaluation (3 subsets × 2 post-processings × 4 panels) | 23 s | 0.32 GB |
| TV sensitivity (2 × 6 variants × 4 panels) | 70 s | < 1 GB |
| full4 train (6 models, fixed rounds) | 70.3 min | 4.76 GB |
| TIMING_PLACEHOLDER

Disk: models hold4 212 MB, full4 199 MB; state_full4 and state_ens34 about 170 MB each; holdout
prediction caches `pred_hold4_<panel>.npz` 22 MB in total. DISK_PLACEHOLDER

## Reproduce
```
cd /home/user/knee
# 1. holdout member with seed 1 (hold3 recipe: TFB_FD=1, defaults otherwise)
TFB_FD=1 TFB_SEED=1 TFB_THREADS=2 PYTHONPATH=. python -m trafficflow.t1_pipeline train --holdout --tag hold4
# 2. its predictions at the cached full-coverage holdout cells (hold3 is recomputed and checked)
PYTHONPATH=. python -m trafficflow.t1_smooth_eval preds hold4
# 3. ensemble evaluation -> /home/user/work/t1/smooth/ens.csv, ens_raw.csv
PYTHONPATH=. python -m trafficflow.t1_smooth_eval ens hold3,hold4
# 4. full-data member, rounds from the hold4 report
TFB_FD=1 TFB_SEED=1 TFB_THREADS=2 PYTHONPATH=. python -m trafficflow.t1_pipeline train --tag full4 \
    --rounds "$(PYTHONPATH=. python -m trafficflow.t1_pipeline rounds --tag hold4)"
TFB_FD=1 TFB_PRED_THREADS=2 PYTHONPATH=. python -m trafficflow.t1_pipeline predict --tag full4
# 5. test ensemble and submission
PYTHONPATH=. python -m trafficflow.t1_pipeline ens --tag ens34 --members full3 full4
PYTHONPATH=. python -m trafficflow.make_submission --state-tag ens34 --recon-a 0.75 --gate 0.6 --smooth default \
    --queue <queue csv> --odme /home/user/work/t4/t4_l2proj.csv --out <file>.csv
```
A third member is the same with `TFB_SEED=2` and tags hold5/full5. Then run
`ens hold3,hold4,hold5` and `ens --tag ens345 --members full3 full4 full5`.
