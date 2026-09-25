# TrafficFlowBench (2026 IEEE Big Data Cup): master plan

Kaggle `2026-ieee-big-data-traffic-flow-bench`. Deadline 2026-11-07 06:55 UTC. 5 submissions/day.
Prizes: $1,500 / $1,000 / $500, plus a $500 student/newcomer award. Winners must write a 10-page report
and pass a code reproducibility check.

## Score
```
S_total = 0.35 S_state + 0.30 S_queue + 0.15 S_physics + 0.20 S_ODME
```
- Public LB = the validation month (March 2031). Private LB = April 2031, a different month generated
  independently (5 vs 7 incidents).
- The leaderboard mixes pre-rebuild scores (leak era, before 11 Sep) with post-rebuild ones. The real
  post-rebuild top is **0.879**. Our account is 0.80855 (38th of 108 post-rebuild teams).

## Key insights (each backed by code or evidence in this repo)

1. **Task 3 is an L1 density problem.** The organizer's boundary flux is projected onto the published
   observations, so the conservation RHS equals the observed ΔN (to about 5%). Therefore
   `S_LWR ≈ 1 − Σ|ΔN_sub − ΔN_obs| / Σ|ΔN_obs|`, with `N = (q/v)·L` over every valid 5-minute
   transition. Only target cells carry error.
   - This is scoreable locally (`evaluate.s_lwr_proxy`).
   - The historical mean gets about 0.07, temporal interpolation 0.53, and LightGBM 0.585.
   - Adding an L1 density model and reconciling `q = v·k̂` raises the proxy to 0.607.
2. **Blackouts in validation/private.** Around every Task 2 window the release blanks rows T+1..T+18
   (90 minutes) for every link. Task 1 targets inside those rows (1–6.5% per regime) have no
   same-time evidence, and they sit in congestion by construction.
   - Train has no such blackouts, so we simulate them (replicated window selector) and train a
     dedicated gap model. On one panel, interpolation gives 5.8 km/h RMSE and the gap model 5.36,
     against about 1.1 on regular cells. They dominate the MSE.
3. **Task 4 truth = L2 projection of the split prior onto the split counts** (`trafficflow/docs/TASK4_ANALYSIS.md`).
   - That hypothesis reproduces the published baseline 0.8359 (we get 0.8357) and the val/private
     gap of 0.008.
   - The projection scores about 1.000 against the baseline's 0.836, roughly **+0.033 total**.
4. **Task 2 onset windows: the queue appears only at T+30** (organizer, 21 Sep; conditioning on it is
   allowed). The problem is *which links*. For a window at T, Task 2 may use only data at or before T.
5. **Noise floor.** Measurement noise is about 1.0–1.9 km/h for speed and 3.5–6% for flow per lane,
   which caps S_state around 0.955–0.96. Temporal interpolation already gets 0.932 and LightGBM 0.950
   on the holdout.

## Validation protocol
- Task 1: exact local scorer on the last 30 train days, which hold out regular targets and simulated
  blackout cells.
- Task 3: LWR proxy on the same holdout. The absolute level is optimistic by about 0.045, since a
  perfect answer scores about 0.955 officially.
- Task 2: IoU on selector-replicated train windows plus the official train windows. Labels come from
  the observed speed ≤ 0.6·v_f (a noisy proxy for the underlying state).
- Task 4: simulation under several generation hypotheses. The leaderboard confirms.

## Leaderboard decomposition (probing plan)
Missing tasks score 0, so partial submissions isolate a task exactly:
- ODME-only: `0.20·S_ODME` confirms the Task 4 hypothesis (expect about 0.200).
- queue-only: `0.30·S_queue`.
- state-only: `0.35·S_state + 0.15·S_physics`.

A few probes, each measuring one task exactly, avoid guessing which component moved. Public is only
a sanity check of transfer; decisions are driven by the local validation above, not by public-LB
tuning, because private is a different month.

## Submission plan (5/day, from 2026-09-23)
Day 1:
1. **Full best**: Task 1 pooled LightGBM with the gap model and density reconciliation, Task 2 best
   model, Task 4 L2 projection.
2. **ODME-only probe**: measures S_ODME and confirms the projection hypothesis.
3. **Queue-only probe**: measures S_queue on validation.
4. The remaining submissions are held back until 1–3 are read (learn, then re-plan).

Final selection: the two submissions with the best *local* validation that also transfer on public.
One of them is conservative.
