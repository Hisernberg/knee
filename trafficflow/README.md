# TrafficFlowBench (`trafficflow/`)

Kaggle competition [`2026-ieee-big-data-traffic-flow-bench`](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench)
(2026 IEEE Big Data Cup). Freeway corridors are observed by loop detectors with masked and blacked-out
cells. There are four tasks, scored together:

```
S_total = 0.35 S_state (Task 1) + 0.30 S_queue (Task 2) + 0.15 S_physics (Task 3) + 0.20 S_ODME (Task 4)
```

This folder is self-contained: code, docs and the submission loop of this competition live only here.
The repository also holds other competitions; see the root README.

## Status
| | |
|---|---|
| Best public score | **0.86711** (`G2_onset_v8stack.zip`, 2026-09-25), rank 9 of 140 teams on the rebuilt leaderboard |
| Deadline | 2026-11-07 06:55 UTC; 5 submissions a day |
| Automation | Three daily Routines (00:07, 12:43, 21:13 UTC) run the loop in [`docs/LOOP.md`](docs/LOOP.md) |
| Leaderboard log | [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) (every submission, with its score and what it taught), [`docs/lb_log.csv`](docs/lb_log.csv) |

## Approach
| Task | Method | Code |
|---|---|---|
| 1: state reconstruction | Pooled LightGBM residuals on interpolation baselines, with fundamental-diagram features. Separate "dark" models are trained on simulated 90-min blackouts. Density is reconciled in dense traffic (v < 0.6·v_f, speed/flow split 0.75) | `t1.py`, `t1_pipeline.py`, `t1_holdout.py` |
| 3: physics (LWR / FD) | Scored on the Task 1 answer. Total-variation smoothing of density inside runs of target cells lowers the conservation residual | `t1_smooth.py`, local proxy in `evaluate.py`, `t1_lwr_eval.py`, `t1_smooth_eval.py` |
| 2: queue forecasting | Reproduced window selector. Onset: LightGBM on hybrid-imputed labels with a stage-2 stacking model, top-m expected-IoU decoding at T+30. Ongoing: blend of LWR-shockwave and location-robust models | `t2/` ([`docs/TASK2_ANALYSIS.md`](docs/TASK2_ANALYSIS.md)) |
| 4: ODME | L2 projection of the weak path prior onto the link counts | `t4/` ([`docs/TASK4_ANALYSIS.md`](docs/TASK4_ANALYSIS.md)) |

Why each piece exists is in [`docs/PLAN.md`](docs/PLAN.md) and [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).
The Task 3 smoothing is in [`docs/T3_SMOOTHING.md`](docs/T3_SMOOTHING.md).

## Run it
Commands run from the repository root with `PYTHONPATH=.`. Data and artefacts stay outside the
repository: `/home/user/data/kaggle_public` (release; set `TFB_REL`), `/home/user/cache` (`TFB_CACHE`),
`/home/user/work`.

```bash
pip install -r trafficflow/requirements.txt
kaggle competitions download -c 2026-ieee-big-data-traffic-flow-bench -p /home/user/data   # unzip to kaggle_public/
python3 -m trafficflow.data                                    # dense per-panel caches

# Task 1 (FD features, holdout fit for validation, then a full fit)
TFB_FD=1 python3 -m trafficflow.t1_pipeline feat
TFB_FD=1 python3 -m trafficflow.t1_pipeline train --holdout --tag hold3
TFB_FD=1 python3 -m trafficflow.t1_pipeline train --tag full3 --rounds "$(python3 -m trafficflow.t1_pipeline rounds --tag hold3)"
TFB_FD=1 python3 -m trafficflow.t1_pipeline predict --tag full3

# Task 2 (full chain, about 2 h on 2 threads), then the v8 onset builder (docs/TASK2_ANALYSIS.md, section 15)
bash trafficflow/t2/run_all.sh

# Task 4
python3 -m trafficflow.t4.make_submissions

# Current best submission, and the guarded submit
python3 -m trafficflow.make_submission --state-tag full3 --recon-a 0.75 --gate 0.6 \
  --smooth "free=0.0075,free_a=0.001,gate=0.02,gate_a=0.005,dark=0.05" \
  --queue /home/user/work/t2/lgb_v8_seeds9_stack03.csv --odme /home/user/work/t4/t4_l2proj.csv \
  --out /home/user/work/subs/<ID>.csv --note "<ID>: ..."
python3 -m trafficflow.loop pack /home/user/work/subs/<ID>.csv
python3 -m trafficflow.loop submit /home/user/work/subs/<ID>.zip -m "<ID>: <change>; <local evidence>"
```

`trafficflow.loop` also has `status`, `diff`, `rank` and `backup`. The backup is a private Kaggle
dataset, `tfb-work`.

## Layout
```
trafficflow/
  README.md, requirements.txt
  data.py            release loader and dense per-panel caches
  evaluate.py        exact Task 1 scorer and Task 3 LWR proxy
  t1*.py             Task 1 models, holdout evaluators, Task 3 smoothing
  t2/                Task 2: window selector, datasets, features, models, stacking, builders
  t4/                Task 4: ODME solvers, evidence and simulator
  make_submission.py assemble a full upload from the task artefacts (65 pre-upload checks in submit.py)
  loop.py            daily loop: status / pack / diff / submit / rank / backup
  docs/              LOOP.md (runbook and state), EXPERIMENTS.md, PLAN.md, TASK2_ANALYSIS.md,
                     TASK4_ANALYSIS.md, T3_SMOOTHING.md, lb_log.csv
```

## Rules for this folder
- Never commit `kaggle.json`, API keys or competition data.
- Kaggle kernels and datasets of this project are prefixed `tfb-`.
- A Task 2 forecast at origin T uses only data timestamped ≤ T.
