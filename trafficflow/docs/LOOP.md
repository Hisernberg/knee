# Automated daily loop: runbook and state

Goal: every UTC day, submit up to 5 files, each one locally validated and each changing a single
factor. Submit one at a time, read the score, learn, report. Runs until the deadline, 2026-11-07
06:55 UTC.

Helpers: `trafficflow/loop.py` (`status`, `pack`, `diff`, `submit`, `rank`, `backup`). Run everything
from the repository root with `PYTHONPATH=.`.

**Scope.** The repository (`Hisernberg/kraggle-3-in-one`) holds several competitions, one folder each.
This loop works only on the traffic project:
- everything it creates or edits lives in `trafficflow/`, with docs in `trafficflow/docs/`;
- it never edits another competition's folder or the shared root files;
- data and artefacts stay outside the repository (`/home/user/data`, `/home/user/cache`, `/home/user/work`).

> **Research agents paused (2026-09-25 ~13:00 UTC).** The user stopped two research agents, the
> Task 1 ensemble and the importance-weighted onset training.
> - Do not launch new agents until the user asks again.
> - The Routines still run the submission chain with already-validated candidates, plus light local
>   work (builds, checks, logging, backups).

## Schedule (UTC; Routines fire into this session)
| Time | Job |
|---|---|
| 00:07 | **Daily chain:** submit the queued candidates one at a time, reading each score before the next |
| 12:43 | **Heartbeat:** review finished research, submit newly validated candidates, launch the next experiments |
| 21:13 | **Evening sweep:** spend the remaining quota (validated candidates first, then probes that answer an open question), queue tomorrow's candidates, start overnight jobs |

Background agents and jobs wake the session when they finish, so work continues between firings.

## Every firing, in order
1. `python3 -m trafficflow.loop status`. It reports quota used/left today, pending scores, the best
   file, free disk, credentials and missing critical files. If credentials or files are missing, go
   to **Recovery**.
2. Read **Current best**, **Candidate queue** and **Decision log** below.
3. Review finished agent/job results against the **Gates**. Build each passing candidate on the
   current best:
   - `make_submission ... --out /home/user/work/subs/<ID>.csv`
   - `loop pack` the CSV
   - `loop diff` against the best zip, which must touch only the intended rows.
4. Submit with `python3 -m trafficflow.loop submit <zip> -m "<ID>: <change>; <local evidence>"`.
   Add `--probe` for probes. After each score, apply the adoption rule, then rebuild the next
   candidate on the new best if the best changed.
5. After the firing's last submission, run `python3 -m trafficflow.loop rank`.
6. Update `trafficflow/docs/EXPERIMENTS.md` (LB log rows and what was learned) and this file (best,
   queue, decision log). Commit and push to `claude/focused-allen-uzq8gr`. Pushes update the open draft PR; merging is the user's call.
7. Report in the chat:
   - a table of submission, change, public score, Δ and post-rebuild rank;
   - the best so far and the gap to #1;
   - what we learned, and what comes next.

   Send one push notification a day, once the day's submissions are done.
8. Launch the next experiments: at most 2 LightGBM jobs at once, 2 threads each (4 cores, 15 GB).
   Keep at least 2.5 GB of disk free. Agents never submit or commit.
9. Once a day (evening sweep), back up the artefacts: `python3 -m trafficflow.loop backup`
   (**Backup** below).

## Current best
**`G2_onset_v8stack.zip` = 0.86711** (2026-09-25). #1 post-rebuild is 0.88849.

Build:
```
python3 -m trafficflow.make_submission --state-tag full3 --recon-a 0.75 --gate 0.6 \
  --smooth "free=0.0075,free_a=0.001,gate=0.02,gate_a=0.005,dark=0.05" \
  --queue /home/user/work/t2/lgb_v8_seeds9_stack03.csv --odme /home/user/work/t4/t4_l2proj.csv \
  --out /home/user/work/subs/<ID>.csv --note "<ID>: ..."
python3 -m trafficflow.loop pack /home/user/work/subs/<ID>.csv
```
The build takes about 2.5 min, of which the smoothing is about 150 s.

- **Task 1:** `full3` LightGBM models (FD features, `TFB_FD=1`).
  - Density reconciliation applies only where v < 0.6·v_f, with speed/flow split a = 0.75.
  - Then total-variation (TV) smoothing of the density inside runs of target cells (`trafficflow/t1_smooth.py`, trafficflow/docs/T3_SMOOTHING.md).
- **Task 2:** `lgb_v8_seeds9_stack03` (TASK2_ANALYSIS.md section 15).
  - Onset: 0.7 × stage 1 + 0.3 × stage-2 stacking. Stage 1 is 0.75 × six on_v3 seeds + 0.25 × three on_v2 seeds, all on hybrid labels. Top-m decoding at bias 0; cap any bias at +0.25.
  - Ongoing: the v5 blend, unchanged since v5.
  - Val/private probabilities: `work/t2h/probs_v8_seeds9_stack03_onset.parquet`, `work/t2/probs_lgb_v5.parquet`.
- **Task 4:** L2 projection.

G2 by task: ODME 0.1988 (task score 0.994), Task 1+3 ≈ 0.4325, queue ≈ 0.2358 (onset ≈ 0.732,
ongoing ≈ 0.840).

## Gates
**Candidate:** one locally validated change against the current best.
- **Adopt** as the new best if LB Δ ≥ +0.0005. A Δ between 0 and +0.0005 needs a matching local
  prediction.
- **Task 1/3 gate:** J = 0.35·S_state + 0.10·S_LWR (full-coverage holdout, realistic blackouts,
  `trafficflow/t1_lwr_eval.py`) improves on at least 3 of 4 panels, and the mean improves.
- **Task 2 gate** (since 2026-09-25; TASK2_ANALYSIS.md section 17). All three must hold:
  1. **Plain CV.**
     - The hybrid-truth score is ≥ best − 0.002.
     - The conservative evaluation (re-drawn windows, old truth) is ≥ best − 0.002 and within one paired SE of 0.
     - Onset, ongoing or the non-recurrent slices improve.
  2. **Shift-weighted CV** (`trafficflow/t2/shift_cv.py`: `weights`, `score`). The Δ is positive under both the validation and the private weighting. Reject if either is below −1 SE.
  3. **Footprint check** (`shift_cv.footprint_check`, on both months). A flag (≥ p95) means the change goes to the LB as a probe first and is never adopted directly. A flag on private means don't adopt at all.

  Applied retroactively, this gate makes the right call on all 8 past Task 2 changes. That includes G3 (+0.0065 in plain CV but −0.023 on March), which the footprint check flags.
- **Local gain, LB Δ ≤ 0:** not adopted. It goes on the robust list for the final pick, since there
  are only 40 windows per Task 2 condition and private is a different month (7 incidents vs 5).

**Probe:** measures something, such as one task's score or a calibration direction. Never adopted.

**Never:**
- more than 5 per UTC day, or anything after 23:45 UTC;
- an untested change;
- a blind resubmission after an ERROR.

## Candidate queue
| ID | Change vs best | Local evidence | Status |
|---|---|---|---|
| H1 (26 Sep, 00:07) | G2 + ongoing v10 `og_shrink08_rec05`: stage 2 may only remove v5 cells, only where recurrence ≥ 0.05. Removes 35 validation cells in 10 windows (42 private cells in 16 windows), all of them G3 removals. Build with `--queue /home/user/work/t2/lgb_v10_og_shrink08_rec05.csv` | plain +0.0029 ± 0.0005 old / +0.0023 hybrid; validation-weighted +0.0049 ± 0.0010; private-weighted +0.0035 ± 0.0006; footprint clean on both months | **probe-candidate.** Adopt if Δ ≥ +0.0005. If Δ ≤ −0.0006 (ongoing ≤ −0.004), drop the ongoing stack line entirely, final pick included |
| H2 (26 Sep, 00:07) | G2 with Task 1 state = mean of full3 and full4, the seed ensemble (`--state-tag ens34`; built as `/home/user/work/subs/H2_t1ens34.zip` once the full4 prediction finishes) | holdout J +0.00077, 4/4 panels (+0.00063 to +0.00092; T1_ENSEMBLE.md) | **candidate.** Expected about +0.0008; adopt if Δ ≥ +0.0005 |

## Decision log
| Date | Submission | Public (Δ vs best) | Decision / lesson |
|---|---|---|---|
| 09-25 | F1: onset re-decoded with logit bias +0.5 (+15 cells, 12 in validation) | 0.86356 (−0.00235) | Larger onset sets hurt on March (onset −0.016); the official first-slot blocks are not larger than ours. Keep b = 0 |
| 09-25 | F2: onset site-commit decoder `site2_lo.05_r.5` (−11 hedge cells) | 0.86418 (−0.00173) | Fewer hedges hurt too (onset −0.012). Top-m at b = 0 is optimal on March from both sides; onset gains must come from better probabilities, not decoding |
| 09-25 | **G1: E1 + TV density smoothing inside target runs** (state rows only) | **0.86651 (+0.00060)** | Local J predicted +0.00062. **Adopted: new best.** The Task 3 proxy predicts the LB to within 0.00002 |
| 09-25 | **G2: G1 + onset v8** (stacking 0.3 + 9-seed mix; 15 cells, 11 in 4 validation windows) | **0.86711 (+0.00060)** | CV onset +0.0035 hybrid / +0.0024 old, i.e. about +0.0005 total. **Adopted: new best.** Shape-aware hedges from stage 2 help on March, where F1's blanket bias hurt |
| 09-25 | G3: G2 + ongoing v9 stage-2 stacking (217 ongoing cells, 139 in validation) | 0.86361 (−0.00350) | CV +0.0065 ± 0.0009 (7 SE) but **March ongoing −0.023**. Not adopted, and not on the robust list. The stack extends queues (D7_I10_E +22 to +27 cells per window) and reshuffles small ones (D7_I10_W, D12_I5_N). Train-month growth patterns don't hold in the shifted months. **Lesson: plain CV cannot gate Task 2 ongoing changes; build a shift-weighted CV first** |

## Robust list (final-selection pool)
Empty so far.

## Backlog (ordered by expected gain per effort)
1. **Shift-weighted CV for Task 2** (the gate for every future Task 2 change).
   - Importance weights make train CV windows resemble the validation or the private windows. They come from a classifier on window-level features known at T.
   - It must reproduce today's LB directions before it is trusted: F1 −, F2 −, G2 +, G3 −.
   - Use it for all Task 2 candidates, with a separate April (private) weighting for final selection.
2. Task 1/3 per-cell accuracy. Isolated target cells carry 46–48% of the LWR loss and short runs (2–3 cells) 36–38% (T3_SMOOTHING.md):
   - seed/bagging ensemble of the six Task 1 models (hold fit for the J gate, then a full fit);
   - flow-model capacity, since flow error dominates free-flow density error.
3. Robustness to incidents and non-recurrent queues (private has 7 incidents), for onset and ongoing:
   - incident-signature features: a sudden drop in downstream capacity, or a speed drop that time
     of day doesn't explain;
   - upweighting non-recurrent windows;
   - recent-month features for private built from March's masked view (all ≤ T).
4. Ongoing capacity: all candidate windows, bigger trees.
5. Task 1 transductive fine-tuning on observed cells of the validation/private months.
6. Ongoing label fix v7 as an LB test (low priority: its non-recurrent slice got worse).

Closed:
- Ongoing stage-2 stacking v9 (G3): failed transfer (−0.00350). Revisit only through the shift-weighted CV.
- Onset stacking and seeds: adopted in G2. Seed means saturate at 3; dropping on_v2 (v3x6) is exploratory only.
- Task 3 TV smoothing: adopted in G1 (+0.00060). Gate 0.7 adds only +0.00004 locally (noise); a = 1.0 fails the gate.
- decoder calibration (F1/F2 above);
- ongoing logit bias: CV optimum at b = 0 (0.8833). Only the official train windows prefer larger
  sets, which did not transfer for onset.

## Final selection (5–6 Nov)
- Recommend 2 finals: the best public score, and the most robust (best local validation and
  non-recurrent performance, from the robust list).
- The Kaggle CLI cannot select finals, so ask the user to tick them. If none are ticked, Kaggle
  takes the top 2 public.
- After the deadline, delete the Routines and send the final report.

## Backup (daily, evening sweep)
`python3 -m trafficflow.loop backup [EXTRA_GLOB ...]` creates a new version of the private Kaggle
dataset `kragglenote2forwork/tfb-work`, replacing the old one. The first version (2026-09-25) holds
18 files, 482 MB:
- the best zip;
- `work/t1/pred/state_full3.parquet` and `work/t1/models/full3/`;
- `work/t2/lgb_v6.csv`, `work/t2/probs_lgb_v5.parquet`, `work/t2h/probs_v6_onset.parquet` and
  `work/t2h/model_v6_*`;
- `work/t4/t4_l2proj.csv`;
- `research/lb/lb_with_era.csv`.

## Recovery (fresh container)
1. **Credentials.** If `~/.kaggle/kaggle.json` is missing, use env `KAGGLE_USERNAME`/`KAGGLE_KEY`,
   or copy `/root/.claude/uploads/*/*kaggle.json` to `~/.kaggle/kaggle.json` (chmod 600). If there
   are none: report to the user and pause submissions.
2. **Data.** `kaggle competitions download -c 2026-ieee-big-data-traffic-flow-bench -p /home/user/data`,
   then unzip into `/home/user/data/kaggle_public` (README.md, config, corridors, task1, task2,
   task4, submission_key.csv, sample_submission.csv).
3. **Caches.** `python3 -m trafficflow.data` rebuilds `/home/user/cache/<panel>.npz`.
4. **Artefacts.** `kaggle datasets download kragglenote2forwork/tfb-work -p /home/user/work/restore --unzip`,
   then move each file back to its path in `manifest.json`. Kaggle unpacks the best zip into a
   folder, so re-zip that CSV with `loop pack` (copy the checks.json from git history or rebuild).
   Submissions can resume from here.
5. **Research state (hours, in the background):**
   - Task 2: `trafficflow/t2/run_all.sh`;
   - Task 1: `trafficflow/t1_pipeline.py` stages (`TFB_FD=1`).
