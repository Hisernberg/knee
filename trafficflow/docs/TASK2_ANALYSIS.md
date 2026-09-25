# Task 2 (online queue-propagation forecasting): analysis and models

Code: `trafficflow/t2/` (run with `PYTHONPATH=/home/user/knee OMP_NUM_THREADS=2`;
`trafficflow/t2/run_all.sh` runs the whole thing).
Candidate files: `/home/user/work/t2/*.csv` (validation + private rows of all 8
scored panels, 174,000 rows, template-exact, timestamps verbatim, binary int).

## 0. Summary

Scores are 4-fold week-blocked CV on windows drawn by our reproduction of the
official selector on train. `sim` = 2,082 onset + 3,001 ongoing windows;
`off` = the 80 official train windows. Official aggregation is used.

| candidate file | onset sim / off | ongoing sim / off | **overall sim / off** |
|---|---:|---:|---:|
| `persistence.csv` (official baseline) | 0.000 / 0.025 | 0.569 / 0.581 | 0.285 / 0.303 |
| `persistence_fill.csv` | 0.000 / 0.025 | 0.705 / 0.749 | 0.353 / 0.387 |
| `range_prior_pers_fill.csv` (rules only) | 0.622 / 0.683 | 0.705 / 0.749 | 0.663 / 0.716 |
| `lgb_v2.csv` (onset p1+prior+weights, ongoing p1+weights) | 0.709 / 0.758 | 0.867 / 0.879 | 0.788 / 0.819 |
| **`lgb_v3.csv`** (onset p1+prior, ongoing p2+weights) | **0.713 / 0.758** | **0.877 / 0.888** | **0.795 / 0.823** |

Recommended: `lgb_v3.csv`. What moves the score, in order:
1. **Onset windows have truth only at `T+30`** (selector artifact, organizer
   confirmed and allowed). Predict nothing at `T+5..T+25` and a few links at
   `T+30`: 0.00 -> 0.62 with a location prior alone, 0.71 with the model.
2. **Missing cells.** Persistence reads the ~25% missing/ineligible `T-5`
   cells as free-flow; last-valid fill adds +0.14 on ongoing windows.
3. **A cell classifier with expected-IoU decoding** adds +0.17 on ongoing
   windows over fill-persistence (+0.14 over the kinematic rule with slot `T`). Bigger trees
   keep helping on ongoing (0.852 -> 0.867 -> 0.877), not on onset.
4. **The `<= T` data rule** (origin slot `T` visible for 50-66% of cells,
   earlier hours, previous days): +0.02-0.03 on the rule baselines, small but
   positive in the model.

## 1. How the windows are selected (reverse-engineered, verified on train)

`core.PanelT2.selector_stats` + `core.PanelT2.greedy` reproduce the official
selector. Rules (all confirmed against the 80 official train windows):

| rule | value |
|---|---|
| history | slots `[T-60, T)` (12 slots). The origin slot `T` is in neither history nor horizon |
| horizon | `T+5 .. T+30` (6 steps) |
| coverage | share of *eligible* cells in the history `>= 0.7` (equals `history_coverage` in `window_index.csv` exactly) |
| condition | `ongoing` iff the history holds `>= 2` queued eligible observations and some link has `>= 2`; else `onset` |
| horizon queue | at least one truly queued cell in the horizon |
| persistence cap | IoU of the **true state at slot T** repeated over the horizon vs the true horizon `<= 0.9` |
| order | chronological greedy from the split start, 5 per condition, **every pair of selected origins (either condition) >= 360 min apart** |

Hypothesis test on the 80 official train windows (approximate truth, see below):
spacing within condition only / persistence from `T-1` observation reproduce
4-6 of 10 windows per panel; global spacing + true-state-at-`T` persistence
reproduces 9,10,6,7,10,9,8,6 of 10, and every remaining mismatch is a
5-minute shift caused by one threshold-noise cell in our approximate truth.

**Consequence (now also announced by the organizer, forum #742350):** an onset
window is the earliest origin whose horizon touches a new queue, so its truth
lies only at `T+30`. Of 2,082 simulated onset windows (8 panels) 98.0% have
nothing queued at steps 1-5 with our approximate truth (93.6-100% per panel;
the organizer confirms 120/120 official windows). Predicting
anything at `T+5..T+25` for an onset window is a pure false positive.

Onset truth is small: median cells at `T+30` = 6 (D7_I10_E), 1 (D7_I10_W),
4 (D7_I210_E), 2 (D7_I210_W), 2 (D7_I405_N, the downstream boundary
bottleneck), 4 (D7_I405_S), 3 (D12_I5_N), 3 (D12_I5_S); on `D7_I405_S` 2
bottlenecks often break down in the same 5 minutes (2 separate runs).

## 2. Labels on train

Truth = underlying speed `<= 0.6 * free_speed` (`links.csv` free speed; equals
`fd_parameters.csv`; no `v_cut` column is released). On train we use the
unmasked observation, NaN cells interpolated (time <= 3 slots, then space <= 2
links, then time <= 12). Measurement noise is ~1.3 km/h (night lag-1
differences / sqrt 2: 0.85-1.32 km/h) and speeds cross the threshold fast: only
0.44-0.47% of queued horizon cells lie within 1 km/h of `v_cut` (0.9-1.0%
within 2 km/h), so noise-induced label flips are negligible; the approximation
error is dominated by the ~17% interpolated cells.

## 3. Data rule used

Organizer (forum #742068): for origin `T` a forecast may use any released data
with timestamp `<= T` (earlier the same day, earlier days of the same split, all
of train; masked mainline states and ramps), nothing after `T`. Features use:
the released 60-min history, the masked view at slot `T` (50-66% of cells
visible), the masked view `[T-180, T-60)`, the masked view of the previous 7
days (recent-days queue climatology) and train-truth time-of-day profiles. No
horizon, buffer or post-buffer value is read.

## 4. Evaluation protocol

`dataset.py` simulates the official selector started on every train day
(days 0..265, as if a split began that day) and keeps the unique windows:
~250-350 onset and ~130-490 ongoing windows per panel (`sim`), plus the 80
official train windows (`off`). Training additionally uses every first-of-run
onset candidate and every second ongoing candidate origin (`cand`).
Cross-validation: 4 interleaved week folds (`(day // 7) % 4`); models and
time-of-day profiles for a fold never see its days. Scores use the official
aggregation (window -> panel & condition -> panel -> family -> mean of 4).

`sim` = 2,082 onset + 3,001 ongoing selector-drawn windows (the reliable
number; per-window IoU sd ~0.3 so the aggregated SE is ~0.005 and paired
method differences are tighter). `off` = the 80 official train windows (40 per
condition; noisy, SE ~0.03-0.05). Overall = mean of the two condition scores
(the aggregation is linear and every panel has both conditions).

## 5. Methods

* `persistence` - official baseline: eligible observation at `T-5` repeated.
  Our file is bit-identical to `src/task2/build_task2_persistence_submission.py`
  on validation.
* `persistence_fill` - last non-null observation of each link within the
  history (any `pct_observed`), repeated. Plain persistence reads the ~25%
  missing/ineligible cells at `T-5` as "not queued".
* `+T` - the same, using the masked-view observation at the origin slot `T`
  where visible (50-66% of cells).
* `kinematic` (ongoing) - queue blocks matched between 15 min earlier and now;
  tail/head extrapolated linearly in milepost (`kinematic.py`).
* `range_prior` (onset) - the contiguous link range at `T+30` maximising mean
  IoU over training onset windows of the same panel within +-90 min time of
  day; steps 1-5 empty (`baselines.py`).
* `lgb` - two LightGBM binary classifiers pooled over the 8 panels
  (`build_features.py`, `cv.py`, `pipeline.py`):
  * onset: one row per (window, link) at `T+30` only; steps 1-5 predicted empty;
  * ongoing: one row per (window, link, step) on cells near queue activity
    (queued/slow in the history, a queue within 12 km downstream, or a
    historically congested cell): ~1/3 of cells, 99.5% of queued cells
    (0.46% of true cells fall outside and are predicted 0).
  Features (`features.py`, `extra.py`, `oprior.py`, ~100 columns): speed/`v_cut`
  ratio (last, mean, min, max, 15/30/55-min change, age of the last
  observation), flow/capacity, density, neighbour ratios at +-1..6 links,
  spatial minima of the ratio in 0-1/1-3/3-6/6-12 km bands downstream and
  upstream, distance to the nearest queued link downstream/upstream now, 15, 30,
  60 min ago and their rates (queue-tail motion), corridor queue share and
  trend, static link data (free speed, lanes, capacity/lane, length, ramps
  within 2 links, relative position, panel code), time of day, weekday, train
  time-of-day queue probability by weekday at `T`, `T-30`, `T+k`; the origin
  slot `T` (ratio, flow, neighbours, queue geometry), the 2 hours before the
  history, the previous-7-days queue frequency at `T` and `T+k`; for onset the
  share of training onset windows (same panel, +-60/+-90 min, weekday class,
  all) whose `T+30` truth contains the link (`oprior.py`, out-of-fold in CV).
  Final parameters: onset lr 0.05, 63 leaves, min_data 100, feature/bagging
  fraction 0.8, 400 rounds; ongoing the same with 127 leaves, min_data 50, 600
  rounds and rows weighted so every window has equal total weight; ongoing
  training uses half of the extra candidate windows (memory).
* Decoding (`models.eiou_topm`): per window pick the top-m cells by probability
  maximising `sum_S p / (|S| + sum_notS p)` (expected-IoU surrogate).

## 6. Cross-validated results

### Rule baselines

| method | onset sim | onset off | ongoing sim | ongoing off | overall sim | overall off |
|---|---:|---:|---:|---:|---:|---:|
| persistence (official) | 0.000 | 0.025 | 0.569 | 0.581 | 0.285 | 0.303 |
| persistence_fill | 0.000 | 0.025 | 0.705 | 0.749 | 0.353 | 0.387 |
| persistence_fill+T | - | - | 0.729 | 0.758 | - | - |
| kinematic | - | - | 0.711 | 0.741 | - | - |
| kinematic+T | - | - | 0.739 | 0.754 | - | - |
| range_prior (onset) + persistence_fill | 0.622 | 0.683 | 0.705 | 0.749 | 0.663 | 0.716 |

(The official persistence scores 0.3028 on the official train windows with
our truth; the organizers quote 0.3017 on validation for 10 corridors and
0.2518 in the README.)

### LightGBM, onset (`T+30` only, top-m decoding)

| variant | sim | off |
|---|---:|---:|
| history-only features (v1), lr .05 / 63 leaves / 400 | 0.706 | 0.725 |
| v2 (+ slot T, earlier today, recent days), lr .08 / 31 leaves / 250 | 0.692 | 0.764 |
| v2, lr .05 / 63 leaves / 400 | 0.709 | 0.764 |
| v2 + onset location prior, lr .05 / 63 / 400 (**final**) | **0.713** | 0.758 |
| v2 + onset prior + window weights | 0.709 | 0.758 |
| v2 + onset prior, lr .05 / 127 leaves / 600 | 0.707 | 0.752 |
| same + window weights | 0.710 | 0.749 |
| average of the last two | 0.709 | 0.752 |

Ablations with the small model (base 0.692 sim): drop slot-T group 0.698, drop
recent-days 0.694, drop earlier-today 0.695, drop panel code/position 0.687,
+window weights 0.699, +onset prior 0.699. The new groups matter little on sim
(within noise) but more on the 40 official windows.

Decoders on the same OOF probabilities (small model): top-m 0.692, top-m with
p^0.8 0.691, threshold 0.25/0.3/0.5 0.681/0.680/0.651, best contiguous range
0.652 (onset often breaks down at two sites at once, e.g. D7_I405_S), gap
closing 0.648.

Error anatomy (small model, sim): the predicted set overlaps the truth in 94%
of onset windows; IoU given a hit is 0.75; knowing the true size and taking
the top-|Y| links would give only 0.727. The residual is the exact extent of
the first queued block, not its location. Per panel (small model):
D7_I210_E 0.78, D7_I405_N 0.76, D7_I10_W 0.75, D7_I210_W 0.72, D7_I10_E 0.69,
D12_I5_N 0.67, D12_I5_S 0.63, D7_I405_S 0.54.

### LightGBM, ongoing

| variant | sim | off |
|---|---:|---:|
| v2, lr .08 / 31 leaves / 250, top-m | 0.852 | 0.869 |
| same, threshold 0.3 / 0.4 / 0.5 | 0.843 / 0.849 / 0.847 | 0.863 / 0.868 / 0.864 |
| same + window weights | 0.853 | 0.877 |
| v2, lr .05 / 63 leaves / 400, window weights | 0.867 | 0.879 |
| v2, lr .05 / 127 leaves / 600, min_data 50, window weights (**final**) | **0.877** | **0.888** |

Per panel (small model, sim): 0.80 (D7_I10_E) to 0.88 (D7_I210_E, D7_I405_N).
Error anatomy: false positives and negatives grow from ~0.6/0.5 cells per
window at `T+5` to ~1.7/1.4 at `T+30`; the per-window mean is pulled down by
small, dissipating queues (windows with <= 6 true cells: IoU 0.31, 12-24
cells: 0.79, > 96 cells: 0.92).

## 7. Validation/private predictions (sanity)

`lgb_v2.csv`: onset windows get cells only at `T+30` (298 cells over 80
windows, 1-9 per window, median 3); ongoing windows ~1,630-1,750 cells per
step. Window-level agreement with `persistence_fill` on ongoing windows 0.74,
with `range_prior` on onset windows 0.76. Predicted onset sites follow the
evidence rather than only the train prior: e.g. `D12_I5_N` mostly links
103-105 (the most congested links in the April observations and among the
top in March, whereas the most congested links on train were 233-239), `D7_I405_S` splits between the two sites
11-15 and 32-39 (both active in March/April).

## 8. Distribution shift to validation/private

Observed queued share per corridor segment by month (masked view, eligible
cells) is stable over the 9 train months but shifts in March/April on some
panels (the splits have their own demand draws and incidents): `D12_I5_N`
segments 7-8 drop from 13-18% to 3-5%, `D12_I5_S` segments 2-4 drop in March,
`D7_I405_S` segment 8 doubles in April, `D7_I210_W` segments 7-9 drop.
Location priors learned on train are therefore a risk; the recent-days
features (previous 7 days, allowed by the `<= T` rule and available for
validation/private from the masked view) and the history/origin-slot evidence
are what let the model move to the currently active bottleneck.

## 9. Caveats

* Train truth is the observed (noisy) speed with ~17% of cells interpolated;
  the official truth is the noise-free state on every cell. The error sits
  mostly at queue edges, so absolute onset scores (tiny truth sets) are the most
  affected; the method ranking should hold.
* Train windows are drawn all year round from a stationary demand; validation
  and private have their own demand draws and incidents (section 8).
  Month-to-month shifts in active bottlenecks are the main risk for onset.
* `off` has only 5 windows per panel and condition (SE ~0.03-0.05), so use the
  `sim` column to compare methods.
* LightGBM runs with 2 threads. The final ongoing model (127 leaves, 600
  rounds) takes ~12 min to train on all windows and ~35 min for 4-fold CV. Peak
  RSS is 3.6 GB in CV and 2.8 GB in the pipeline. The data loader reads column
  groups into one preallocated matrix and frees the raw matrix after binning. An
  earlier version that did not do this reached 5.9 GB and was OOM-killed.

## 10. Possible next steps

* Onset is at a plateau (~0.71) across model sizes and feature groups. It
  finds the right place 94% of the time, so the remaining gap is the exact
  links of the first queued block. Directions: joint (set-level) decoding, for
  example scoring candidate blocks taken from analogous training windows under
  the model marginals, and a physics prior on which link of a bottleneck
  breaks down first.
* Ongoing: capacity is still paying off (0.852 -> 0.867 -> 0.877). Larger
  models or more candidate windows (only half are used, for memory) are the
  cheapest gain. Small dissipating queues (<= 12 true cells) score 0.3-0.5 and
  are the main error.
* Recent-days features could use the whole previous month (for private, all
  of March validation) to track bottleneck shifts better.

## 11. Public LB transfer (March) and shift-robust variant `lgb_v4_robust`

**LB result for `lgb_v3.csv`** (Task 2 isolated by the coordinator's probe submissions,
validation = March): S_queue 0.749 against 0.795 in CV. Onset scored 0.686
(CV 0.713, -0.027) and ongoing 0.812 (CV 0.877, -0.065).

### Diagnosis (data <= T only: histories, masked view, train profiles)

`robust.recurrence` measures how recurrent a window's queue is. It averages
the train time-of-day queue probability at T, for the same weekday and
learned out of fold, over the links queued at the end of the history.

* **The model itself rates the March ongoing windows as harder.** Its
  expected-IoU surrogate (the decoder objective) averages 0.841 on validation
  and 0.853 on private, against 0.882 on the CV windows, where the realised
  IoU was 0.877. Validation is lowest on D12_I5_N (0.74) and D7_I10_W
  (0.74). The surrogate explains about 0.04 of the 0.065 drop. The rest is
  overconfidence on shifted windows.
* **March has far more non-recurrent queues.** 12.5% of validation ongoing
  windows have recurrence < 0.05, against 2.6% in train CV and 2.5% on
  private. These are all 5 D7_I10_W windows: queues at links 24-26, the train
  bottleneck, but at weekdays and times when train almost never queues there.
  Validation also has more weekend windows (35% vs 24% in train) and fewer
  large queues (90th percentile of queued links 38 vs 67). Large queues are
  the easy ones: IoU 0.95 with >= 24 queued links.
* **CV IoU by recurrence** (v3 ongoing model):

  | recurrence | windows | IoU | model surrogate |
  |---|---:|---:|---:|
  | < 0.05 | 78 | 0.562 | 0.691 |
  | 0.05-0.2 | 220 | 0.838 | 0.841 |
  | >= 0.2 | 2,703 | 0.891 | 0.88-0.89 |

  Reweighting these CV scores to the validation mix already predicts 0.847,
  about -0.03 from composition alone. The model is overconfident exactly on
  the non-recurrent windows. There the true queue grows from 9.4 to 16.1
  links between T+5 and T+30, while the prediction grows only from 8.6 to
  10.8. The time-of-day priors say "no queue here", which pulls growth down.
  Rules do worse on these windows: fill-persistence 0.433, kinematic 0.442.
* **"Shift-like" CV subset:** train CV windows with recurrence < 0.2 (298
  windows) or < 0.05 (78). For onset, the proxy uses the truth, so it is for
  evaluation only: the mean train profile at T+30 over the truly queued links
  is < 0.2 (471 windows) or < 0.05 (143).

### Variants (4-fold CV; ongoing windows; shift columns are plain means)

| ongoing variant | sim | off | recur<0.05 | recur<0.2 |
|---|---:|---:|---:|---:|
| fast config (31 leaves, 250 rounds, window weights), all features | 0.853 | 0.877 | 0.522 | 0.731 |
| fast, no location/time-of-day priors (`noloc`: drop pq_*, rq7_*) | 0.848 | 0.868 | 0.541 | 0.727 |
| fast, dynamics only (also drop link identity, tod, weekday) | 0.829 | 0.844 | 0.539 | 0.707 |
| fast, 50/50 blend all + noloc | 0.856 | 0.872 | 0.540 | 0.738 |
| **v3**: p2 config (127 leaves, 600 rounds, weights), all features | 0.877 | 0.888 | 0.562 | 0.766 |
| p2, noloc | 0.877 | 0.890 | 0.586 | 0.768 |
| **v4**: 50/50 blend v3 + p2 noloc | **0.881** | 0.889 | **0.588** | **0.775** |
| v4, noloc alone when recurrence < 0.05 | 0.881 | 0.889 | 0.586 | 0.775 |
| v4, logit +0.5 when recurrence < 0.2 | 0.881 | 0.889 | 0.583 | 0.773 |
| v3, logit +0.5 when recurrence < 0.2 | 0.878 | 0.889 | 0.572 | 0.771 |

What each lever did:
* Dropping only the location priors costs nothing at full capacity and helps
  the non-recurrent windows.
* Dropping the link identity as well hurts everywhere, including the shift
  subset: which bottleneck a queue sits at still matters.
* Blending the two models is the best option on every metric.
* Gating and probability boosts add nothing on top of the blend.

Onset variants on the onset shift subsets (sim / recur<0.05 / recur<0.2):
* fast without prior: 0.692 / 0.240 / 0.512
* fast + prior: 0.699 / 0.265 / 0.533
* p2 + prior: 0.707 / 0.244 / 0.523
* p2 + prior, weighted: 0.710 / 0.249 / 0.530
* **v3 = p1 + prior: 0.713 / 0.275 / 0.543**

The location prior helps on the non-recurrent onsets too, because it
includes all-day and weekday-class variants. So onset stays as in v3.

### Result

`/home/user/work/t2/lgb_v4_robust.csv` (`robust_pipeline.py`):
* **Onset:** the v3 model and decoder, unchanged. The file's onset rows are
  identical to v3.
* **Ongoing:** probability = 0.5 × v3 ongoing model + 0.5 × the no-location-prior
  model (p2, weights). Top-m expected-IoU decoding as before.
* **Coverage:** 174,000 rows, binary, all 160 windows non-empty, onset cells
  only at T+30. Ongoing predictions agree with v3 at a mean window IoU of
  0.948 (minimum 0.727).

| | onset sim / off / shift<0.2 | ongoing sim / off / shift<0.2 / shift<0.05 | overall sim / off |
|---|---|---|---|
| lgb_v3 | 0.713 / 0.758 / 0.543 | 0.877 / 0.888 / 0.766 / 0.562 | 0.795 / 0.823 |
| lgb_v4_robust | 0.713 / 0.758 / 0.543 | 0.881 / 0.889 / 0.775 / 0.588 | 0.797 / 0.823 |

Expected effect on March: a small gain (the recur < 0.05 bucket is 12.5% of
March ongoing windows, +0.026 there, +0.004 elsewhere). Most of the LB gap
comes from the March window mix (small, non-recurrent queues) and is not
recoverable by any variant tested here.

## 12. Physics and shockwave features: `lgb_v5`

**Public LB for `lgb_v4_robust`:** it was submitted together with a Task 1
change. Public 0.85732 vs 0.85204 (+0.0053); Task 2's share was not separated.

### New features (`physics.py`, data <= T only; feature tables `feat_v3/`)

* **Onset rows (`ph_*`, 30 columns).** For each link, from the history and
  the origin slot where visible:
  * flow/capacity: level, 30-min least-squares slope, linear extrapolation to
    T+30, maximum 15-min mean in the hour;
  * density `k = q/v` against critical density (`fd_parameters`): ratio,
    slope, extrapolation to T+30, `k/k_jam`;
  * speed margin to `v_cut`: slope, extrapolation to T+30, 15-min minimum, and
    the minimum over the hour;
  * bottleneck signature: capacity ratio to the downstream and upstream
    neighbour, lane change to each neighbour, minimum capacity within 1.5 km
    downstream relative to the link, and the downstream-minus-upstream speed
    ratio difference (1 link and ±2 links);
  * distance downstream and upstream to the nearest link already below 0.8 and
    below 0.7 of free speed;
  * the at-most-one queued observation per link an onset history may hold:
    presence at the link, its age, the distance to the nearest link with one,
    and the corridor count;
  * arriving demand: mean flow of the 3 upstream links over local capacity,
    and its slope.

  One observation: in D7_I10_E onset windows no link is below 0.8 of free
  speed at T. Queues form from about 0.87 free speed within 35 min, so demand
  against capacity carries most of the signal.
* **Ongoing rows (`lw_*`, 11 shared + 3 per-step columns).**
  * Queue blocks at the origin: runs of links with speed <= `v_cut`, taking the
    slot-T value where visible and the last history value otherwise. Each
    block has a tail (upstream end) and head milepost.
  * Rankine-Hugoniot tail speed `s = (q_q - q_u)/(k_q - k_u)`. The arriving
    state is the mean of the 3 links upstream of the tail; the queue state is
    the first 3 links of the block.
  * Head speed from the state downstream of the head. Empirical tail speed
    from the best-overlapping block 20 min earlier.
  * Per link, from the block containing it or the nearest one downstream
    (<= 12 km; for the head, the nearest one upstream, <= 5 km): signed
    distances to the tail and head, block length, arriving and queue flow over
    capacity, densities over `k_c`.
  * Per step k: signed distance to the predicted tail `x_tail + s*5k min`
    (RH speed, and separately the empirical speed) and to the predicted head.

  Sanity check on 100 D7_I10_E windows: within 3 km of the tail, 81% of
  cells predicted inside the RH-extrapolated queue are queued, against 24% of
  those predicted outside.

### Set-level decoding for onset (the v3 onset OOF)

| decoder | sim | off | recur<0.05 | recur<0.2 |
|---|---:|---:|---:|---:|
| top-m, expected IoU (current) | **0.713** | 0.758 | **0.275** | **0.543** |
| best contiguous range | 0.666 | 0.706 | 0.261 | 0.511 |
| most likely link + best upstream extension | 0.498 | 0.493 | 0.177 | 0.376 |
| most likely site (p >= 0.05 cluster), top-m inside | 0.711 | 0.766 | 0.275 | 0.543 |
| best 1-2 sites (2nd if its mass >= 0.5 × 1st) | 0.714 | 0.762 | 0.270 | 0.542 |

Committing to one site or block does not beat top-m. The queue often extends
downstream of the most likely link, and the model already picks the right
site in most windows. Site confusion on D7_I405_S, D12_I5_S and D12_I5_N is
mostly on the diagonal. The remaining error is the extent within the site:
for example D7_I405_S's 15-link site B scores 0.57 even when the site is
right. Top-m stays.

### CV per slice (4-fold, sim = selector-drawn train windows; recur slices are plain means)

| model / blend | onset sim | onset off | onset rec<0.05 | onset rec<0.2 |
|---|---:|---:|---:|---:|
| v4 onset (v2 features + prior, p1) | 0.7133 | 0.7578 | 0.275 | 0.543 |
| v3 features (+ physics) + prior, p1, seed 0 | 0.7152 | 0.7703 | 0.282 | 0.551 |
| same, seed 1 / seed 2 | 0.7171 / 0.7171 | 0.768 / 0.775 | 0.285 / 0.291 | 0.551 / 0.555 |
| mean of the 3 seeds | 0.7200 | 0.7630 | 0.289 | 0.557 |
| no prior, v3 features | 0.7109 | 0.7638 | 0.239 | 0.524 |
| **v5 onset: 0.75 × 3 seeds + 0.25 × v4 onset** | **0.7210** | **0.7691** | **0.295** | **0.561** |

| model / blend | ongoing sim | ongoing off | ongoing rec<0.05 | ongoing rec<0.2 |
|---|---:|---:|---:|---:|
| fast config, v2 features | 0.8529 | 0.8767 | 0.522 | 0.731 |
| fast config, + LWR | 0.8567 | 0.8725 | 0.546 | 0.741 |
| p2w, v2 features (v3's model) | 0.8772 | 0.8875 | 0.562 | 0.766 |
| p2w, + LWR | 0.8796 | 0.8873 | 0.581 | 0.773 |
| p2w, + LWR, no location priors | 0.8792 | 0.8869 | 0.597 | 0.772 |
| v4 ongoing: 50/50 v2 all + v2 noloc | 0.8809 | 0.8889 | 0.588 | 0.775 |
| 50/50 LWR all + LWR noloc | 0.8820 | 0.8883 | 0.599 | 0.779 |
| 4-way equal | 0.8832 | 0.8902 | 0.598 | 0.781 |
| **v5 ongoing: 0.35 LWR all + 0.35 LWR noloc + 0.15 v2 all + 0.15 v2 noloc** | **0.8833** | **0.8906** | **0.599** | **0.781** |

| | onset sim / off | ongoing sim / off | **overall sim / off** |
|---|---|---|---|
| lgb_v4_robust | 0.7133 / 0.7578 | 0.8809 / 0.8889 | 0.7971 / 0.8233 |
| **lgb_v5** | **0.7210 / 0.7691** | **0.8833 / 0.8906** | **0.8022 / 0.8299** |

Adoption rule: overall sim must be >= v4 - 0.002, and onset or the
non-recurrent slices must improve. v5 gains +0.005 overall sim, with onset
+0.008 (non-recurrent onset +0.018 to +0.020) and ongoing +0.002
(non-recurrent ongoing +0.006 to +0.011). Adopted.

### The file

`/home/user/work/t2/lgb_v5.csv` was written by `robust_pipeline.py` with
`T2_FEAT=feat_v3`. The models were retrained on all train windows:

* **Onset:** mean of three v3-feature onset models (p1 config, location
  prior, seeds 0/1/2) at 0.25 each, plus the v4/v3 onset model at 0.25.
  Top-m expected-IoU decoding, T+30 only.
* **Ongoing:** 0.35 × the LWR model with every feature, 0.35 × the LWR model
  without location priors (both p2 config with window weights), 0.15 × the
  v3 ongoing model, and 0.15 × the v4 no-location-prior model. Top-m
  decoding.

Checks:
* 174,000 rows with the same keys and order as v4 and the templates;
  `queue_pred` is in {0, 1}.
* All 160 windows are non-empty.
* Onset cells appear only at T+30: 309 cells, 1-9 per window, median 4.
  Ongoing has 1,621-1,735 cells per step.
* Mean window agreement with v4 is 0.964 on ongoing windows (minimum 0.80)
  and 0.938 on onset windows. 12 of 80 onset windows changed. Most changes
  add or drop links within the same candidate sites, or add the second site
  (D12_I5_S). One window (D7_I10_W validation 005, 05:25) moved from link 28
  to link 51.
* Peak RSS was 3.2 GB for training and prediction, and 3.15 GB for the
  largest CV run. Training ran with 2 threads.

Reproduce:
```
T2_PHYSICS=1 T2_FEAT=/home/user/work/t2/feat_v3 python -m trafficflow.t2.build_features
T2_FEAT=/home/user/work/t2/feat_v3 python -m trafficflow.t2.robust_pipeline lgb_v5 \
  --onset train:on_v3:p1:nw:op:0.25:0 --onset train:on_v3:p1:nw:op:0.25:1 \
  --onset train:on_v3:p1:nw:op:0.25:2 --onset lgb_v3:0.25 \
  --ongoing train:og_v3:p2:w:noop:0.35 --ongoing train:og_v3_noloc:p2:w:noop:0.35 \
  --ongoing lgb_v3:0.15 --ongoing rob_noloc_p2w:0.15
```
The CV rows above come from `python -m trafficflow.t2.robust cv VARIANT CFG
[--weighted] [--cond queue_onset --oprior]` with `T2_FEAT=.../feat_v3`
(`T2_SEED` sets the seed). `trafficflow.t2.v5_eval` compares the blends, and
`python -m trafficflow.t2.onset_decode OOF` compares the decoders.

## 13. Onset extent: diagnosis, decoders, and the label fix (`lgb_v6`)

### Diagnosis on the v5 onset OOF (old truth, 2,082 sim windows)

**The site is usually right; the boundaries are what go wrong.**
* The predicted block overlaps the true block in 94.4% of windows, and IoU on
  those windows is 0.776.
* When both ends of the block are exact (61.6% of those windows) IoU is
  0.931; otherwise it is 0.527.
* The head is off by at least one link in 22% of them, the tail in 22%.

**Relative to this truth the model over-predicts.** Per window there are 0.68
false-positive links inside the site and 0.12 false negatives, plus 0.17 / 0.13
outside it. Predicted minus true block size:

| predicted − true links | share of windows |
|---|---:|
| 0 | 54% |
| +1 | 24% |
| +2 | 8% |
| +3 or more | 7% |
| negative | 7% |

**Short links are the hardest.**
* Sites with links of 0.2 km or less score IoU 0.61 (0.71 km predicted vs
  0.34 km true).
* Links of 0.45-0.7 km score 0.91.
* By panel (IoU when the site is right), D7_I405_S is weakest at 0.62 and
  D7_I10_W strongest at 0.89. The 07-08 h origins score 0.71-0.72; 14 h
  scores 0.83.

**Physics extent is not predictable from demand.**
* At a given head link the first-slot extent varies by a standard deviation of
  0.5-1.5 links.
* It is nearly uncorrelated with anything observable at T: flow/capacity, its
  slope and extrapolation, upstream demand, density (all |r| <= 0.24 at the
  four main sites).
* That fits the extent being set by when, within the 5-minute slot, the
  breakdown happens, which cannot be observed.
* With the true head known, the site's modal extent gives block IoU 0.893.
* A shockwave-speed extent estimate therefore had nothing to add over the
  per-site empirical extent.

**Two findings that affect other work.**
* **Direction.** On the W/S panels (D7_I10_W, D7_I210_W, D7_I405_S, D12_I5_S)
  link i+1 is the upstream neighbour of link i, not the downstream one. All
  "downstream/upstream" features, and the LWR tail/head features of section 12,
  are mirrored on those panels. The tree compensates with the panel code:
  direction-normalised onset features scored 0.7195 vs 0.7210 for the 4-model
  blend.
* **Label bias.** The train truth is biased at the first queued slot. That
  bias, not the model, explains most of the "over-prediction" above (next
  subsection).

### Decoders and stacking (v5 OOF, old truth, 6-step metric)

| method | onset sim | off | rec<0.05 | rec<0.2 |
|---|---:|---:|---:|---:|
| **top-m (v5)** | **0.7210** | 0.7691 | 0.295 | 0.561 |
| head (from top-m) + empirical extent, contiguous block upstream | 0.6708 | 0.7225 | 0.296 | 0.532 |
| head from the model's head probability + extent | 0.6680 | 0.7138 | 0.282 | 0.521 |
| hybrid: top-m inside the predicted block ± 1 link | 0.7216 | 0.7691 | 0.295 | 0.562 |
| stage-2 stacking on the probability profile (3 seeds, 50/50 with stage 1) | 0.7237 | 0.7710 | 0.311 | 0.576 |

* **Contiguous blocks lose** because true first-slot blocks often skip a link.
  At D7_I405_N link 350 is rarely queued between 349 and 351, and the pointwise
  model already skips it.
* **Library shape-prior decoder.** It computes the posterior over training
  truth shapes of the panel, then takes the Bayes-optimal expected-IoU choice.
  It gains at most +0.002 to +0.004 and loses on the official windows and the
  non-recurrent slice.
* **Stacking is real but small:** +0.0027 ± 0.0016 (paired bootstrap).
* Cluster features and a larger stage-2 model did not help. Adding four more
  stage-1 models gave 0.7213.

None of these clears the +0.005 bar.

### The label fix

**The problem.** The train truth fills missing cells (about 17%, random
detector gaps) by time interpolation first. At the first queued slot of a new
queue the state is sharp in time and smooth in space. A masked test hid
observed cells and re-filled them; queued-cell recall at onset slots:

| fill of a missing cell | accuracy | queued recall | specificity |
|---|---:|---:|---:|
| time interpolation (used so far) | 0.792 | **0.266** | 0.984 |
| space interpolation | 0.819 | 0.550 | 0.916 |
| mean of the two | 0.859 | 0.506 | 0.988 |
| **LightGBM imputer** (`truthfix.py`: neighbours t±1,2, links ±1,2, diagonals) | **0.962** | **0.947** | 0.967 |

**The imputer.**
* On ongoing horizon cells it is as good as time interpolation (0.988 vs
  0.984).
* On observed cells 5-15 min before an onset its specificity is 0.98-0.99.
* It was trained on even train days and checked on odd days.
* The official truth is the complete underlying state, so the old fill's
  missing queue cells were an artifact of our truth only. The model learned
  first-slot blocks 7-17% too small (per panel) and was scored against them.

**Hybrid truth.** The imputer's 1-2% false-positive rate (specificity
0.98-0.99 before onsets), applied to millions of missing cells, scatters false
queue cells through free flow. The window selector's "any queued cell" rule
then breaks: 5/80 official windows reproduced. The hybrid truth keeps the
old fill everywhere except missing cells within ±5 min / ±2 links of an
observed queued cell, where the imputer decides.
* Selector reproduction: 68/80 exact (old truth 66) and 75/80 within 5 min
  (old 78).
* It adds 7-17% queued cells at T+30 in onset windows (D7_I10_E 3,019 → 3,428;
  D7_I405_N 1,397 → 1,631).

**Results.** Same 4-model onset recipe as v5 (v3 physics features seeds 0/1/2
+ v2 features, p1, location prior), old vs hybrid labels:

| evaluation | truth | old labels (v5) | hybrid labels (v6) | Δ sim | off | rec<0.05 | rec<0.2 |
|---|---|---:|---:|---:|---|---|---|
| original sim windows minus 222 whose hybrid T+30 is empty (1,860), actual v5 OOF | hybrid | 0.8510 | 0.8629 | **+0.0119** | 0.9075 → 0.9194 | 0.358 → 0.392 | 0.679 → 0.702 |
| same | old | 0.7750 | 0.7827 | **+0.0077** | 0.7881 → 0.7932 | 0.332 → 0.352 | 0.632 → 0.650 |
| windows re-drawn by the selector on the hybrid truth (2,081) | hybrid | 0.8518 | 0.8589 | **+0.0071 ± 0.0017** | 0.8758 → 0.8841 | 0.406 → 0.412 | 0.689 → 0.704 |
| same | old | 0.7555 | 0.7605 | **+0.0050 ± 0.0018** | 0.7565 → 0.7659 | 0.373 → 0.374 | 0.632 → 0.641 |

* On the re-drawn windows, 110 improve and 46 worsen under the hybrid truth;
  102 and 50 under the old truth.
* Absolute levels depend on the truth. Under the corrected truth onset CV is
  about 0.85-0.86, not 0.72: much of the old "error" was missing truth cells.
  Compare within a row only.
* Ongoing is unchanged, so overall sim rises by half the onset gain, about
  +0.0025 to +0.006.

**Adopted.**
* Onset improves by at least +0.005 in every evaluation. The most conservative
  one (re-drawn windows, old truth) sits exactly at the bar.
* Overall is not lower than v5, and the non-recurrent onset slices improve
  everywhere.

### The file

`/home/user/work/t2/lgb_v6.csv`:
* **Onset:** the mean of four onset models (the v5 recipe), trained on the
  re-drawn windows with hybrid labels. The location prior is built from
  hybrid-truth onsets, and the feature tables (`/home/user/work/t2h/feat`)
  use hybrid-truth time-of-day profiles. Top-m decoding, T+30 only.
* **Ongoing:** rows are the lgb_v5 rows, identical.

Checks:
* 174,000 rows with the same keys and order as v5 and the templates; binary.
* All 160 windows are non-empty.
* 311 onset cells, all at T+30, 1-9 per window (median 4).
* 19 of 80 onset windows changed vs v5. Mostly the site extent moved by one or
  two links, or a second site was added or dropped. D7_I10_W validation 005
  moved back to link 28 (v4's choice).

Reproduce:
```
python -m trafficflow.t2.truthfix fit                 # imputer + masked check
python -m trafficflow.t2.truthfix relabel hybrid      # WORK/ds_<p>_y2.npz (hybrid truth)
T2_WORK=/home/user/work/t2h T2_TRUTHQ=/home/user/work/t2/ds_{panel}_y2.npz python -m trafficflow.t2.dataset
T2_WORK=/home/user/work/t2h T2_PHYSICS=1 T2_ONLY=queue_onset T2_FEAT=/home/user/work/t2h/feat python -m trafficflow.t2.build_features
T2_WORK=/home/user/work/t2h T2_FEAT=/home/user/work/t2h/feat python -m trafficflow.t2.onset_v6 /home/user/work/t2/lgb_v6.csv
```
Other modules for this section:
* `onset_extent.py`: diagnosis, head+extent and library decoders.
* `stack.py`: stage-2 stacking.
* `onset_decode.py`: site decoders.
* `T2_NORMDIR=1` in `build_features.py`: direction-normalised features.

The CV runs use `robust cv on_v3|on_v2 p1 --cond queue_onset --oprior`, with
`T2_OOF_ALL=1` and `T2_TAGX` set.

## 14. Hybrid-truth label fix for the ongoing model: not adopted

**LB context (March, the coordinator's single-factor probes).** The v5 ongoing
blend gave S_queue +0.0081. The v6 onset label fix gave S_queue +0.0151,
which is onset +0.030 on March (CV had said +0.005 to +0.012).

**How much the ongoing labels change.** On the 3,001 original simulated
ongoing windows, the hybrid truth changes 0.23% of horizon cells.
* That is 1.7% of the old queued cells, or 2.6 cells per window against
  about 148 queued.
* Most changes add queue cells (6,690 added vs 1,049 removed), and 79% of
  windows have at least one change.
* Where the changed cells sit: 33% just upstream of a queue block (tail
  side), 26% just downstream (head side), 38% inside blocks (gaps filled),
  2% elsewhere.
* The pattern is the same in growing and dissipating windows: 2.3 vs 2.9
  changed cells per window, about 0.9 on the tail side and 0.6-0.7 on the
  head side.
* For onset the change was 7-17% of the T+30 queue cells, much larger than
  here.

**Evaluation.**
* Setup: the v5 blend recipe (0.35 LWR-all + 0.35 LWR-noloc + 0.15 v2-all +
  0.15 v2-noloc, window weights). Each component was retrained with old or
  with hybrid labels on the windows the selector draws on the hybrid truth,
  4-fold CV.
* Proxy: the fast config (31 leaves, 250 rounds) stood in for p2. Eight
  p2 CV runs (about 40-50 min each on the shared machine) did not fit in the
  time available.
* Labels were swapped on the fly (`T2_RELABEL=<npz>:<key>` in `cv.gather`),
  so no relabelled feature-table copies are kept on disk.

| re-drawn windows (2,081 sim) | old labels | hybrid labels | Δ sim (paired bootstrap) | windows better / worse | off | rec<0.05 | rec<0.2 |
|---|---:|---:|---:|---|---|---|---|
| old truth | 0.8665 | 0.8671 | **+0.0006 ± 0.0006** | 948 / 932 | 0.8784 → 0.8828 | **0.586 → 0.581** | 0.747 → 0.748 |
| hybrid truth | 0.8634 | 0.8666 | **+0.0032 ± 0.0006** | 1,036 / 839 | 0.8810 → 0.8878 | **0.578 → 0.572** | 0.741 → 0.743 |

The first component alone (LWR-all) shows the same pattern: +0.0006 ± 0.0007
under the old truth, +0.0028 ± 0.0006 under the hybrid truth, and recurrence
< 0.05 at −0.006 under both.

**Verdict: not adopted.**
* The rule needs ≥ +0.003 under both truths on the re-drawn windows and no
  worse non-recurrent slice. The hybrid-truth gain passes (+0.0032).
* The old-truth gain (+0.0006) fails, and the recurrence < 0.05 slice worsens
  under both truths (−0.005 / −0.006).
* The original-window rows were stopped once the conservative rows had
  decided the verdict.

**Why the effect is small.** Ongoing labels change on ~1.7% of queued cells,
spread over boundaries and gaps. The old truth already gets most ongoing cells
right: time interpolation recovers 97% of queued cells inside established
queues, against 27% at the first queued slot of a new one. So the bias the fix
removes is concentrated at onset. v6 (onset fix + v5 ongoing) remains the
recommended file.

**Code.**
* `og_labelfix_eval.py`: the evaluation.
* `ongoing_v7.py`: the v7 builder, ready but not run.
* Tables: `/home/user/work/t2h/feat_og` (re-drawn-window ongoing features,
  ~1 GB). Delete it if disk is needed.

## 15. Onset stage-2 stacking and more seeds on the hybrid truth (`lgb_v8_*`)

**Goal.** A single-factor onset change on top of `lgb_v6`. On the public
month, truth-consistent onset changes transfer strongly: the v6 label fix
gave onset +0.030 on March where CV said +0.005 to +0.012 (March onset is
now 0.728).

**Gate** (the coordinator's rule for sending a Task 2 candidate to the
leaderboard):
* onset sim under the hybrid truth improves on v6;
* the conservative evaluation (old truth, re-drawn windows) is not negative;
* the non-recurrent slices (recur < 0.05, recur < 0.2) are not worse by more
  than noise.

### Evaluator (`onset_eval.py`)

`OnsetEval` scores any probability vector aligned to the rows of the v6 onset
OOF files. Those rows cover all 4,583 onset windows × links at T+30
(`T2_OOF_ALL=1`). Scoring uses the 2,081 re-drawn sim windows and the 40
official windows:
* **Decoding:** top-m expected IoU per window.
* **Truths:** `hybrid` (`ds_<p>.npz["y"]`) and `old` (`y_old`). Steps 1-5
  are empty under both, as in `robust.load_truth`.
* **Scores:** official aggregation for sim and off. The recurrence slices are
  plain means. They use `robust.onset_recurrence` on the t2h tables, which is
  based on the hybrid truth, so both truths use the same 144 / 438 windows.
* **`compare(a, b)`:** paired bootstrap of Δ sim by window. It resamples the
  sim windows jointly (2,000 replicates, as `og_labelfix_eval` does). It also
  reports Δ off, Δ and SE of the slices, and the number of windows better /
  worse.
* **`stack_v8.nested_weight`:** picks the blend weight on three week folds
  and scores it on the fourth. This gives an honest estimate of a tuned
  weight.

**Reproducing section 13** (v6 = mean of its four OOF files):

| truth | sim | off | rec<0.05 | rec<0.2 |
|---|---:|---:|---:|---:|
| hybrid (section 13: 0.8589 / 0.8841 / 0.412 / 0.704) | 0.8589 | 0.8841 | 0.412 | 0.704 |
| old, steps 1-5 kept (section 13: 0.7605 / 0.7659 / 0.374 / 0.641) | 0.7605 | 0.7659 | 0.374 | 0.641 |
| **old, steps 1-5 empty (used from here on)** | **0.7871** | **0.7943** | **0.383** | **0.661** |

* Every section 13 number is reproduced exactly.
* **Why the old-truth rows differ.** The section 13 old-truth row kept the
  old truth's steps 1-5. Of the 2,081 re-drawn sim windows (drawn on the
  hybrid truth), 261 have old-truth queue cells at T+5..T+25. A T+30-only
  forecast can never hit those cells. The hybrid truth has such cells in 41
  windows, and those are emptied.
* With steps 1-5 empty (as for the hybrid truth, and as onset truth is
  defined), v6 scores 0.7871 / 0.7943 under the old truth.
* Section 13's slices used the hybrid-truth recurrence for both truths. That
  is kept here.
* `OnsetEval(truths=ALL_TRUTHS)` adds the section 13 convention as `old_e`.

### Stage-2 stacking on the v6 OOF (`stack_v8.py`)

**Setup.**
* This is `stack.py` (section 13) on the t2h OOF.
* Stage 1 is the mean of v6's four OOF files. Stage 2 is LightGBM on the
  window's stage-1 probability profile (26 features, traffic direction).
* It is trained on the hybrid labels of every onset window (cand, sim and
  off), with the same 4 week folds (OOF stacking).
* Final probability = `w × stage 2 + (1 − w) × stage 1`, then top-m decoding.
* Default stage 2: 31 leaves, min_data 100, lr 0.05, 300 rounds.

Stage 2 bagged over 3 seeds, by blend weight (Δ vs v6 ± paired-bootstrap SE):

| w | hybrid sim | Δ | old sim | Δ | hybrid rec<0.05 / rec<0.2 | old rec<0.05 / rec<0.2 |
|---|---:|---:|---:|---:|---|---|
| 0 (v6) | 0.8589 | | 0.7871 | | 0.412 / 0.704 | 0.383 / 0.661 |
| 0.2 | 0.8610 | +0.0022 ± 0.0009 | 0.7878 | +0.0008 ± 0.0008 | 0.433 / 0.713 | 0.394 / 0.667 |
| **0.3** | **0.8620** | **+0.0031 ± 0.0011** | **0.7887** | **+0.0016 ± 0.0011** | **0.431 / 0.713** | **0.393 / 0.667** |
| 0.4 | 0.8616 | +0.0027 ± 0.0013 | 0.7882 | +0.0011 ± 0.0013 | 0.430 / 0.713 | 0.393 / 0.668 |
| 0.5 | 0.8590 | +0.0001 ± 0.0017 | 0.7853 | −0.0017 ± 0.0016 | 0.430 / 0.714 | 0.390 / 0.666 |
| 0.7 | 0.8567 | −0.0022 ± 0.0020 | 0.7828 | −0.0043 ± 0.0020 | 0.429 / 0.710 | 0.389 / 0.662 |
| 1 (stage 2 alone) | 0.8511 | −0.0077 ± 0.0025 | 0.7786 | −0.0085 ± 0.0026 | 0.433 / 0.706 | 0.392 / 0.657 |

Stage-2 variants at w = 0.3, and with the weight chosen out of fold (nested):

| stage 2 | hybrid Δ | old Δ | nested hybrid Δ | nested old Δ | weights chosen per fold |
|---|---:|---:|---:|---:|---|
| 1 seed | +0.0031 ± 0.0012 | +0.0014 ± 0.0011 | +0.0024 ± 0.0014 | +0.0006 ± 0.0013 | 0.4 / 0.3 / 0.3 / 0.3 |
| **3 seeds (bagged)** | **+0.0031 ± 0.0011** | **+0.0016 ± 0.0011** | **+0.0025 ± 0.0013** | **+0.0010 ± 0.0012** | 0.4 / 0.3 / 0.3 / 0.3 |
| 3 seeds, 150 rounds | +0.0029 ± 0.0011 | +0.0018 ± 0.0010 | +0.0029 ± 0.0011 | +0.0018 ± 0.0010 | 0.3 everywhere |
| 3 seeds, 15 leaves, min_data 200 | +0.0023 ± 0.0010 | +0.0012 ± 0.0010 | +0.0013 ± 0.0010 | +0.0004 ± 0.0010 | 0.4 / 0.2 / 0.3 / 0.2 |
| 3 seeds, + link features and location prior (`T2_STACK_EXTRA`) | +0.0018 ± 0.0011 | +0.0008 ± 0.0009 | −0.0004 ± 0.0014 | −0.0018 ± 0.0014 | 0.7 / 0.2 / 0.3 / 0.2 |

**What stage 2 does here.**
* Stage 2 alone loses (−0.008). On v5 / old labels it lost −0.003 and the
  best weight was 0.5; on the hybrid OOF the best weight is 0.3.
* **Calibration.** Stage 1 (the v6 mean) is overconfident at the top
  (predicted 0.991 → observed 0.966) and underconfident in the middle (0.10 →
  0.20, 0.29 → 0.41). Stage 2 is calibrated (0.094 → 0.090, 0.98 → 0.97).
* **The gain is not a calibration shift.** A logit bias on v6 does not
  reproduce it:

  | logit bias b | −0.25 | 0 | +0.25 | +0.5 | +0.75 | +1.0 |
  |---|---:|---:|---:|---:|---:|---:|
  | v6 + b, hybrid Δ | −0.0014 | 0 | +0.0005 | −0.0001 | −0.0024 | −0.0034 |
  | v6 + b, old Δ | −0.0014 | 0 | −0.0001 | −0.0009 | −0.0035 | −0.0047 |
  | stack w=0.3 + b, hybrid Δ | +0.0020 | **+0.0031** | +0.0004 | −0.0016 | −0.0038 | −0.0088 |
  | stack w=0.3 + b, old Δ | +0.0005 | **+0.0016** | −0.0009 | −0.0031 | −0.0055 | −0.0103 |
  | v6 + b, off hybrid / old | 0.8841 / 0.7943 | 0.8841 / 0.7943 | 0.8841 / 0.7943 | 0.8928 / 0.8038 | 0.8945 / 0.8060 | 0.8970 / 0.8088 |
  | stack + b, off hybrid / old | 0.8841 / 0.7943 | 0.8897 / 0.8006 | 0.8925 / 0.8037 | 0.8952 / 0.8068 | 0.9001 / 0.8119 | 0.8995 / 0.8068 |

  (Δ is against plain v6. The sim-window optimum of the stacked blend is at
  b = 0. On the 40 official windows a positive bias helps both, by about
  +0.01.)
* **Where it gains.** The gain comes from windows where stage 1 is unsure;
  confident windows are unchanged. By the window's stage-1 maximum
  probability (sim windows, plain mean Δ):

  | stage-1 max p | windows | v6 IoU (hybrid) | Δ hybrid | Δ old | validation / private windows |
  |---|---:|---:|---:|---:|---|
  | ≤ 0.5 | 64 | 0.238 | +0.051 | +0.025 | 5 / 2 |
  | 0.5-0.8 | 64 | 0.543 | +0.016 | +0.018 | 6 / 3 |
  | 0.8-0.95 | 195 | 0.754 | +0.000 | −0.002 | 4 / 2 |
  | > 0.95 | 1,758 | 0.914 | +0.001 | +0.000 | 25 / 33 |

  Stage 2 mostly extends a low-confidence scatter into the adjacent block
  (e.g. links 104, 107, 153 → 103-107, 153).
* **Validation and private have more of these windows** (the March shift,
  sections 8 and 11). The stage-1 maximum probability averages 0.82 on
  validation and 0.92 on private, against 0.95 on CV sim windows. Reweighting
  the per-bucket gains to the validation mix gives about +0.009 (hybrid) /
  +0.006 (old), and +0.004 for private. These are plain means over few
  windows, so they are only indicative.
* **Stability.**
  * By week fold, Δ hybrid is +0.0008 / +0.0027 / +0.0037 / +0.0050 and Δ old
    is +0.0000 / +0.0016 / +0.0010 / +0.0038.
  * By panel, Δ hybrid is ≥ 0 on 7 of 8 panels: D7_I10_W +0.011, D12_I5_S
    +0.007, D7_I405_N +0.005, D12_I5_N +0.004, D7_I210_W +0.002, D7_I210_E
    +0.001, D7_I10_E 0. D7_I405_S is −0.005, where two sites often break down
    at once.
* **Other checks.** An exact expected-IoU decoder was tried on v6. It uses
  Poisson-binomial sums over independent cells in place of the ratio of
  expectations. It is worse: −0.0014 ± 0.0009 hybrid, −0.0020 ± 0.0008 old.
  The cells are correlated within a block. The top-m ratio decoder stays.

### More seeds (same recipe, hybrid labels)

**Setup.**
* New models: `robust cv on_v3 p1 --cond queue_onset --oprior` with seeds
  3/4/5, and `on_v2` with seeds 1/2.
* `T2_WORK=/home/user/work/t2h`, `T2_FEAT=/home/user/work/t2h/feat`,
  `T2_OOF_ALL=1`, `T2_TAGX=_new`. No relabelling is needed: the t2h tables
  already carry the hybrid labels.
* Re-running seed 0 gives an OOF file bit-identical to v6's
  (max |Δp| = 0).

| model / mean | hybrid sim | off | rec<0.05 | rec<0.2 | old sim | Δ hybrid vs v6 | Δ old |
|---|---:|---:|---:|---:|---:|---:|---:|
| v6 (0.75 on_v3 s0-2 + 0.25 on_v2 s0) | 0.8589 | 0.8841 | 0.412 | 0.704 | 0.7871 | | |
| on_v3, single seeds 0-5 | 0.8566-0.8589 | 0.884-0.897 | 0.409-0.428 | 0.699-0.705 | 0.7846-0.7872 | −0.0022 … +0.0001 | |
| on_v2, single seeds 0-2 | 0.8532-0.8546 | 0.874-0.887 | 0.377-0.386 | 0.689-0.692 | 0.7809-0.7841 | −0.0057 … −0.0043 | |
| on_v3 × 3 (s0-2), no on_v2 | 0.8596 | 0.8841 | 0.421 | 0.706 | 0.7878 | +0.0007 ± 0.0007 | +0.0007 ± 0.0007 |
| **`v3x6`**: on_v3 × 6 | 0.8597 | 0.8841 | 0.426 | 0.708 | 0.7883 | +0.0008 ± 0.0010 | +0.0013 ± 0.0010 |
| **`seeds9`**: 0.75 on_v3 × 6 + 0.25 on_v2 × 3 (v6 proportions) | 0.8593 | 0.8841 | 0.410 | 0.704 | 0.7877 | +0.0004 ± 0.0007 | +0.0006 ± 0.0007 |
| equal mean of the 9 | 0.8592 | 0.8841 | 0.406 | 0.704 | 0.7876 | +0.0003 ± 0.0006 | +0.0005 ± 0.0007 |

* **Seeds saturate at three.** On the hybrid labels the on_v2 component
  (v2 features, no physics) is about 0.004 weaker than an on_v3 seed. It
  costs about 0.0007 in the v6 mix. It was added in v5, where it helped on
  the old labels.
* Dropping it (`v3x6`) is a post-hoc choice, so it is reported as
  exploratory. More seeds alone are within noise.

### Candidates and the gate

Δ is against v6 on the 2,081 re-drawn sim windows (± paired-bootstrap SE;
p = share of bootstrap Δ ≤ 0). The slices are hybrid / old truth. The
stacked variants use stage 2 bagged over 3 seeds at w = 0.3. "Nested" picks
the weight out of fold (w = 0.3 was chosen in every fold for `seeds9_stack03`).

| candidate | hybrid sim | Δ hybrid | old sim | Δ old | Δ rec<0.05 | Δ rec<0.2 | nested Δ hybrid / old | off hybrid / old | gate |
|---|---:|---:|---:|---:|---|---|---|---|---|
| v6 | 0.8589 | | 0.7871 | | | | | 0.8841 / 0.7943 | |
| `seeds9` | 0.8593 | +0.0004 ± 0.0007 (p 0.28) | 0.7877 | +0.0006 ± 0.0007 | −0.002 ± 0.003 / +0.000 ± 0.004 | −0.000 / +0.002 | | 0.8841 / 0.7943 | nominal pass, noise |
| `v3x6` (exploratory) | 0.8597 | +0.0008 ± 0.0010 (p 0.18) | 0.7883 | +0.0013 ± 0.0010 | +0.014 / +0.015 | +0.004 / +0.005 | | 0.8841 / 0.7943 | nominal pass, noise |
| `stack03` | 0.8620 | +0.0031 ± 0.0011 (p 0.000) | 0.7887 | +0.0016 ± 0.0011 | +0.018 / +0.010 | +0.009 / +0.006 | +0.0025 / +0.0010 | 0.8897 / 0.8006 | **pass** |
| **`seeds9_stack03`** | **0.8624** | **+0.0035 ± 0.0011 (p 0.001)** | **0.7894** | **+0.0024 ± 0.0012** | **+0.019 / +0.013** | **+0.011 / +0.009** | **+0.0035 / +0.0024** | **0.8925 / 0.8037** | **pass (recommended)** |
| `v3x6_stack03` (exploratory) | 0.8627 | +0.0038 ± 0.0013 (p 0.000) | 0.7896 | +0.0025 ± 0.0013 | +0.036 / +0.026 | +0.013 / +0.011 | +0.0028 / +0.0016 | 0.8869 / 0.7974 | pass |

Notes on the table:
* The old truth with steps 1-5 kept (section 13 convention) gives the same
  Δs to ±0.0001 (`seeds9_stack03`: +0.0023 ± 0.0012).
* Windows better / worse under the hybrid truth: `stack03` 40 / 28,
  `seeds9_stack03` 49 / 28, `v3x6_stack03` 49 / 32.

**Recommended: `lgb_v8_seeds9_stack03.csv`.**
* Its recipe was fixed before it was scored: v6's model mix with more
  seeds, plus stacking at the weight the v6 OOF chose.
* It has the best nested estimate (+0.0035 ± 0.0011 hybrid, +0.0024 ± 0.0012
  old).
* Both non-recurrent slices improve under both truths.
* `v3x6_stack03` scores slightly higher, but it relies on the post-hoc drop
  of on_v2, and its nested estimate is lower.
* The seeds-only variants pass only nominally and are not worth a
  leaderboard slot.

**The files** (lgb_v6.csv with only the onset rows replaced; built by
`onset_v8.py`):
* Stage-1 models are trained on all onset windows. The four v6 models are
  reused, and the five new seeds are saved as `model_v8_*`.
* Stage 2 is trained on the OOF stage-1 mean of all onset windows. It is
  applied to the validation/private stage-1 mean. Top-m decoding at T+30.

| file | onset windows changed (of 80) | onset cells changed | added / removed | onset cells (v6 311) | mean window agreement with v6 |
|---|---:|---:|---|---:|---:|
| `lgb_v8_stack03.csv` | 5 | 13 | 11 / 2 | 320 | 0.964 |
| `lgb_v8_seeds9.csv` | 4 | 4 | 2 / 2 | 311 | 0.985 |
| `lgb_v8_v3x6.csv` | 5 | 6 | 3 / 3 | 311 | 0.979 |
| **`lgb_v8_seeds9_stack03.csv`** | **7** (4 validation, 3 private) | **15** | **14 / 1** | **324** | **0.966** |
| `lgb_v8_v3x6_stack03.csv` | 9 | 16 | 12 / 4 | 319 | 0.960 |

Checks:
* `submit.check` for every file: 174,000 rows, 0 missing, 0 extra, binary.
* Ongoing rows are identical to v6. Onset cells appear only at T+30, with
  1-9 per window and all 80 onset windows non-empty.
* Validation/private probabilities are saved as
  `/home/user/work/t2h/probs_v8_<name>_onset.parquet` (the columns and row
  order of `probs_v6_onset.parquet`).

**Self-tests.**
* `onset_v8` with the four v6 specs and no stacking reproduces
  `probs_v6_onset.parquet` (max |Δp| = 0) and `lgb_v6.csv` byte for byte.
* The stage-2 inference path (`onset_v8.stage2_probs`) matches the training
  features of `stack.build` exactly on shuffled rows.

**`seeds9_stack03` changes vs v6:**
* Every added link had a v6 probability of 0.01-0.40. Ten of the 14 added
  links are in three windows whose v6 maximum probability is at most 0.10.
* D12_I5_N validation 003: links 104, 107, 153 → 45, 103-107, 153, 154, 156,
  all at p ≈ 0.06-0.10.
* D12_I5_N validation 004: 116, 237 → 236-238.
* D12_I5_S validation 004: + link 15. D12_I5_S validation 005: + link 41.
* D7_I10_W private 003: + links 22, 28.
* D7_I405_S private 001: + link 15.
* D12_I5_N private 001: + link 238.

**Calibration bias.** The stacked probabilities are better calibrated than
v6's, so a positive logit bias hurts them sooner. For `seeds9_stack03`, Δ
hybrid / old vs plain v6 by bias:

| bias | Δ hybrid | Δ old |
|---|---:|---:|
| −0.25 | +0.0023 | +0.0015 |
| 0 | +0.0035 | +0.0024 |
| +0.25 | +0.0022 | +0.0007 |
| +0.5 | −0.0004 | −0.0018 |
| +0.75 | −0.0035 | −0.0052 |

For v6 itself, +0.5 is flat (−0.0001 / −0.0009). If a bias is applied to the
v8 probabilities, it should be at most about +0.25, not the value tuned for
v6. The 40 official windows favour a positive bias for both, by about
+0.01.

**Expected effect.**
* CV onset +0.0035 is about S_queue +0.0018.
* The gain sits in low-confidence windows, which are over-represented in
  March (section 11, and the stage-1 confidence table above). The v6 label
  fix transferred at 2.5-6× its CV gain. So the March effect may be larger,
  but only 4 validation windows change.

### Reproduce

```
export PYTHONPATH=/home/user/knee OMP_NUM_THREADS=2 T2_WORK=/home/user/work/t2h T2_FEAT=/home/user/work/t2h/feat
# extra seeds: 4-fold OOF on all onset rows, hybrid labels, re-drawn windows (~5 min each)
for s in 3 4 5; do T2_OOF_ALL=1 T2_TAGX=_new T2_SEED=$s python -m trafficflow.t2.robust cv on_v3 p1 --cond queue_onset --oprior; done
for s in 1 2; do T2_OOF_ALL=1 T2_TAGX=_new T2_SEED=$s python -m trafficflow.t2.robust cv on_v2 p1 --cond queue_onset --oprior; done
python -m trafficflow.t2.onset_eval                        # v6 and its components under both truths
# stage-2 stacking (T2_THREADS sets the stage-2 threads); ~2-4 min each
python -m trafficflow.t2.stack_v8 v6s012 --seeds 0,1,2     # stage 1 = v6's four OOF files
python -m trafficflow.t2.stack_v8 seeds9s012 $(python -c "from trafficflow.t2.stack_v8 import V3X6,V2X3; print(' '.join(V3X6+V2X3))") \
  --weights 3,3,3,3,3,3,2,2,2 --seeds 0,1,2
python -m trafficflow.t2.stack_v8 v3x6s012 $(python -c "from trafficflow.t2.stack_v8 import V3X6; print(' '.join(V3X6))") --seeds 0,1,2
python -m trafficflow.t2.stack_v8 table                    # the candidate table above
# candidate files (lgb_v6.csv with the onset rows replaced) + probs_v8_<name>_onset.parquet
S9=on_v3:0:3,on_v3:1:3,on_v3:2:3,on_v3:3:3,on_v3:4:3,on_v3:5:3,on_v2:0:2,on_v2:1:2,on_v2:2:2
python -m trafficflow.t2.onset_v8 seeds9_stack03 --specs $S9 --stack 0.3 --stack-seeds 0,1,2
python -m trafficflow.t2.onset_v8 stack03 --stack 0.3 --stack-seeds 0,1,2
python -m trafficflow.t2.onset_v8 seeds9 --specs $S9
python -m trafficflow.t2.onset_v8 v3x6 --specs on_v3:0,on_v3:1,on_v3:2,on_v3:3,on_v3:4,on_v3:5
python -m trafficflow.t2.onset_v8 v3x6_stack03 --specs on_v3:0,on_v3:1,on_v3:2,on_v3:3,on_v3:4,on_v3:5 --stack 0.3 --stack-seeds 0,1,2
```

Resources:
* Onset CV peaks at about 1.3 GB RSS, stacking at 0.7-0.9 GB and a build at
  0.7 GB.
* The candidate builds and the stacking ran with 1 thread. The new stage-1
  models were trained with 1 thread. Thread count can change LightGBM
  results in the last bits.
* Logs are in `/home/user/work/t2h/logs/`. Stage-2 OOFs are in
  `/home/user/work/t2h/stack8_oof_<tag>.parquet`.

## 16. Ongoing stage-2 stacking on the v5 probability field (`lgb_v9_ogstack08`)

**Goal.** A single-factor ongoing change on top of `lgb_v8_seeds9_stack03`.
On March, ongoing scores about 0.840 against 0.883 in CV. March also has many
more non-recurrent ongoing windows (section 11). The onset stacking of
section 15 transferred to March (about +0.004).

**Gate** (the coordinator's rule for a leaderboard candidate):
* CV sim improves: paired-bootstrap Δ > 0, ideally by more than 1 SE.
* off is not worse beyond noise.
* The non-recurrent slices are not worse. Improving them is the goal.

### Setup (`og_stack.py`)

* **Stage 1.** The v5 ongoing blend: 0.35 LWR-all, 0.35 LWR-noloc,
  0.15 v2-all and 0.15 v2-noloc (p2 config, old labels).
  * Its OOF files cover only the evaluation windows: 3,001 sim and 40 off
    ongoing windows, 1.73 M candidate cells (`robust cv` without
    `T2_OOF_ALL`).
  * So stage 2 trains on those 3,041 windows, with the same 4 week folds
    (OOF stacking). The stage-2 model of a fold never sees that fold's
    windows.
* **Stage 2.** LightGBM per candidate cell (window, step k, link): 31 leaves,
  min_data 100, lr 0.05, 300 rounds, feature and bagging fraction 0.8,
  2 threads. It is bagged over 3 seeds. The 4 folds take about 2 min per seed.
* **Final probability** = `w × p2 + (1 − w) × p1`, then top-m decoding.
* **Features** are all in traffic direction. The link axis of the W/S panels
  is reversed with `core.direction`, as in `stack.py`.
  * **Field (`s_`).**
    * p, and p at link offsets −4..+4.
    * p at steps k−1 and k+1, same link and ±1 link.
    * Step sum and max, window sum, step-sum growth vs step 1, vs step
      k−1 and vs the observed queue at T.
    * Rank in the step, p / step max, mass 0.5 / 1 / 2 km downstream and
      upstream.
    * Predicted blocks (runs of p ≥ 0.5): signed distance in links and km
      to the tail of the block containing the link or the next block
      downstream, and to the head of the containing block or the next one
      upstream.
    * Block length (links, km) and mass, number of blocks.
    * Tail and head movement vs step k−1 and vs the observed block at T.
  * **Observed queue (`o_`, data ≤ T).**
    * The queue indicator `r_now ≤ 1`: slot T where visible, else the last
      history value.
    * The observed block's tail and head distances and length, and the
      number of queued links.
    * Tail and head movement over the last 20 and 35 min. It comes from the
      filled history ratio at T−20 / T−35 min, which is `r_last − d_r15` /
      `r_last − d_r30`.
    * The observed tail extrapolated linearly to T+5k.
    * These replace feat_v3's `lw_*` columns. Those are mirrored on the W/S
      panels (section 13), so they are not used here.
  * **Window context (`x_`).**
    * Recurrence (`robust.recurrence`).
    * Queued links at the end of the history and at T, and the 15 / 60-min
      trends.
    * Weekend flag and time of day.
    * Per link: `r_now`, `r_last`, `d_r15`.
  * **Static.** Step k, link length, relative position, panel code.
* **Evaluator (`OngoingEval`).** It subclasses `OnsetEval`, so `compare()` and
  `stack_v8.nested_weight` are reused.
  * Top-m decoding per window.
  * Two truths on the original windows:
    * **old**: `ds_<p>.npz["y"]`, the training labels;
    * **hybrid**: the truthfix hybrid truth, `ds_<p>_y2.npz["y"]`.
  * Official aggregation for sim and off. The recurrence slices are plain
    means over 78 / 298 windows.
  * Paired bootstrap by window, 2,000 replicates.
  * It reproduces v5 exactly: sim 0.8833, off 0.8906, rec<0.05 0.599,
    rec<0.2 0.781 (old truth). The hybrid truth gives 0.8804 / 0.8938 /
    0.593 / 0.774.
  * The t2h re-drawn windows were not scored: only fast-config ongoing OOFs
    exist there.

### Results

All rows use w = 0.8, the weight used by the candidate. Δ is against v5, ±
the paired-bootstrap SE.
* "Nested" chooses w per fold on the other three folds (old truth, grid
  0.2-1.0) and reports the old / hybrid Δ with the weights chosen.
* The last column is the plain-mean Δ on D7_I10_W's non-recurrent windows
  (18 of the 78 with recurrence < 0.05). All five March D7_I10_W ongoing
  windows are of this kind.

| stage 2 (seeds) | old sim | Δ old | Δ hybrid | off | rec<0.05 | rec<0.2 | nested Δ old / hybrid (w per fold) | D7_I10_W rec<0.05 |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| v5 (stage 1) | 0.8833 | | | 0.8906 | 0.599 | 0.781 | | |
| all features (3) | 0.8901 | +0.0068 ± 0.0011 | +0.0069 | 0.9017 | 0.621 | 0.798 | +0.0062 / +0.0065 (0.8/0.8/1/0.7) | −0.041 |
| all features (1) | 0.8899 | +0.0066 ± 0.0010 | +0.0069 | 0.9025 | 0.619 | 0.799 | +0.0065 / +0.0067 (1/1/1/0.7) | −0.037 |
| + window weights (1) | 0.8894 | +0.0061 ± 0.0011 | +0.0058 | 0.8960 | 0.626 | 0.800 | +0.0055 / +0.0054 | −0.025 |
| + loc / noloc component p (1) | 0.8892 | +0.0058 ± 0.0011 | +0.0062 | 0.9004 | 0.617 | 0.798 | +0.0053 / +0.0055 | −0.049 |
| + time-of-day prior `pq_k` (1) | 0.8903 | +0.0070 ± 0.0011 | +0.0070 | 0.9017 | 0.626 | 0.799 | +0.0069 / +0.0068 | −0.032 |
| − panel code (1) | 0.8898 | +0.0065 ± 0.0011 | +0.0067 | 0.9023 | 0.627 | 0.802 | +0.0064 / +0.0066 | −0.041 |
| − context (`x_*`, panel, position) (1) | 0.8892 | +0.0059 ± 0.0009 | +0.0063 | 0.9042 | 0.629 | 0.801 | +0.0059 / +0.0062 | −0.028 |
| field only (− `o_*`, `x_*`) (1) | 0.8885 | +0.0052 ± 0.0009 | +0.0050 | 0.9025 | 0.613 | 0.792 | +0.0048 / +0.0046 | −0.031 |
| **`dyn`: − recurrence, time of day, weekend, panel, position (3)** | **0.8899** | **+0.0065 ± 0.0009** | **+0.0069** | **0.9031** | **0.630** | **0.801** | **+0.0065 ± 0.0009 / +0.0068 (0.8/0.7/0.8/0.8)** | **−0.020** |
| − context (3) | 0.8896 | +0.0063 ± 0.0009 | +0.0065 | 0.9040 | 0.630 | 0.802 | +0.0063 / +0.0065 (0.8 ×4) | −0.027 |
| − context, 63 leaves / 500 rounds (1) | 0.8902 | +0.0069 ± 0.0011 | +0.0067 | 0.9050 | 0.636 | 0.801 | +0.0065 / +0.0065 | −0.025 |

Blend weight (`dyn`, 3 seeds, Δ old):

| w | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | **0.8** | 1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Δ sim | +0.0023 | +0.0034 | +0.0043 | +0.0051 | +0.0059 | +0.0065 | **+0.0065** | +0.0061 |
| off | 0.8919 | 0.8940 | 0.8964 | 0.8989 | 0.9001 | 0.9012 | **0.9031** | 0.9046 |
| rec<0.05 | 0.601 | 0.604 | 0.605 | 0.605 | 0.619 | 0.628 | **0.630** | 0.610 |

* **The variants are within noise of each other.**
  * `dyn` − all features: −0.0003 ± 0.0006.
  * `dyn` − no context: +0.0002 ± 0.0003.
  * 63 leaves − 31 leaves: +0.0006 ± 0.0006.
  * Most of the gain is the field itself (+0.0052).
  * The observed-queue features add the non-recurrent gain: rec<0.05 goes
    from +0.014 (field only) to +0.030.
* **The time and location context does not help the non-recurrent slices.**
  It adds nothing on sim either. Dropping recurrence, time of day, weekend,
  panel code and position (`dyn`) keeps sim and raises rec<0.05 by +0.009.
  It also removes the features most exposed to the March shift. So `dyn` is
  the candidate.
* **The weight rule.** w = 0.8 is what the nested procedure picks when run on
  all four folds (the argmax of old-truth sim). The nested estimate of that
  procedure is +0.0065 ± 0.0009 (old) and +0.0068 (hybrid).

### Where the gain comes from (`dyn`, w = 0.8)

* **Folds:** +0.0071 / +0.0085 / +0.0041 / +0.0063 (hybrid +0.0076 / +0.0087 /
  +0.0041 / +0.0069).
* **Panels (plain mean):** 7 of 8 positive.
  * D7_I210_E +0.020, D7_I10_E +0.016, D12_I5_N +0.007, D7_I405_N +0.006.
  * D12_I5_S +0.004, D7_I210_W +0.003, D7_I405_S +0.001.
  * D7_I10_W −0.004 ± 0.003.
* **By stage-1 confidence** (the decoder's expected-IoU surrogate), sim
  windows:

  | stage-1 surrogate | windows | v5 IoU | Δ old | Δ hybrid | validation / private windows |
  |---|---:|---:|---:|---:|---|
  | ≤ 0.6 | 93 | 0.495 | +0.099 | +0.095 | 3 / 2 |
  | 0.6-0.8 | 417 | 0.732 | +0.016 | +0.019 | 7 / 10 |
  | 0.8-0.9 | 919 | 0.890 | +0.003 | +0.004 | 17 / 12 |
  | 0.9-0.95 | 982 | 0.935 | +0.001 | +0.001 | 9 / 10 |
  | > 0.95 | 590 | 0.962 | +0.001 | +0.001 | 4 / 6 |

  The surrogate averages 0.831 on validation and 0.837 on private, against
  0.872 on CV sim windows. Reweighting the per-bucket gains gives about
  +0.012 (validation) and +0.010 (private) as plain means, against +0.007 on
  the CV mix. This is indicative only: few windows.
* **By true queue size.**
  * ≤ 6 cells: +0.077 (67 windows).
  * 6-12 cells: +0.091 (54 windows).
  * 12-24: +0.007; 24-48: +0.005; 48-96: +0.000; > 96: +0.003.
  * The small, dissipating queues that section 6 named the main error are
    where it gains.
* **It is not calibration.**
  * Per-step isotonic calibration of v5, fitted out of fold: +0.0005 ± 0.0003.
  * A logit bias on v5: at best +0.0002 (hybrid truth, +0.25), and negative
    under the old truth.
  * Stage 2 leaves the mean predicted set unchanged (147.9 cells per
    window). It moves cells, it does not add them.
* **Growth on the rec<0.05 windows.** Mean predicted cells per step, T+5 →
  T+30:
  * v5: 8.7 → 11.5;
  * stack: 8.6 → 12.0;
  * truth: 9.4 → 16.1.

  Correct cells at T+30 rise from 10.33 to 10.58. The growth deficit is only
  partly corrected.
* **Watch: D7_I10_W non-recurrent windows.** 18 CV windows score −0.020 ±
  0.014 (9 better, 7 worse). The other 60 rec<0.05 windows gain +0.046 ±
  0.020. Every variant loses there, from −0.020 to −0.049, so the cause is
  the field re-scoring, not the context features.
  * The losses come from windows whose queue dissipated while the stack
    extended it. Example: true cells 4 → 0 at T+30, 12 predicted.
  * On average the stack moves D7_I10_W's growth toward the truth. New cells
    at T+30 on the low-index side of the T+5 set: truth 111, v5 60, stack 78.
  * On the five March D7_I10_W windows the stack changes cells both ways:
    27 → 23, 38 → 43, 38 → 40, 11 → 11, 23 → 19 (agreement 0.69-0.95).
  * This slice is the main risk for the March transfer.
* **Surprise: queue growth is not always upstream.** In traffic direction,
  new cells between T+5 and T+30 appear upstream of the T+5 set on
  D12_I5_N, D7_I405_N, D7_I210_W and D7_I405_S. On D7_I10_E (94% of new
  cells), D7_I210_E and D12_I5_S they appear downstream of it. D7_I10_W is
  mixed. This was checked against the truth, and `core.direction` is
  topology-based. A fixed "the tail moves upstream" rule would be wrong on
  half the panels. Stage 2 learns the pattern from the field and the
  observed geometry.

### Decoder check (task 4): logit bias b before top-m

Δ is against plain v5.

| b | −0.5 | −0.25 | 0 | +0.25 | +0.5 | +0.75 | +1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v5, Δ old / hybrid | −0.0036 / −0.0070 | −0.0006 / −0.0024 | 0 / 0 | −0.0013 / +0.0002 | −0.0045 / −0.0018 | −0.0098 / −0.0056 | −0.0175 / −0.0120 |
| `dyn` w=0.8, Δ old / hybrid | +0.0040 / +0.0014 | +0.0066 / +0.0049 | **+0.0065 / +0.0069** | +0.0052 / +0.0068 | +0.0018 / +0.0045 | −0.0029 / +0.0009 | −0.0088 / −0.0042 |
| `dyn` w=0.8, off (old) | 0.8978 | 0.8977 | 0.9031 | 0.9041 | 0.9025 | 0.9012 | 0.8951 |

b = 0 is the balanced optimum for the stacked probabilities. −0.25 suits only
the old truth and +0.25 only the hybrid truth. No decoder change is proposed.

### Gate and verdict

| criterion | result |
|---|---|
| CV sim Δ (paired bootstrap) | +0.0065 ± 0.0009 old (7 SE), +0.0069 ± 0.0009 hybrid; nested +0.0065 / +0.0068 |
| off | +0.0125 old, +0.0119 hybrid (0.9031 / 0.9057) |
| rec<0.05 | +0.031 ± 0.016 old, +0.029 hybrid |
| rec<0.2 | +0.020 ± 0.005 old, +0.019 hybrid |

**Passes.** The candidate is `lgb_v9_ogstack08.csv`. Caveat: the
D7_I10_W non-recurrent slice (−0.020 ± 0.014 on 18 windows) is the closest
CV analogue of March's ongoing shift.

### The file

`/home/user/work/t2/lgb_v9_ogstack08.csv` is `lgb_v8_seeds9_stack03.csv` with
only the ongoing rows replaced. It is built by `ongoing_v9.py`.
* **Stage 1.** The v5 validation/private probabilities: the ongoing rows of
  `probs_lgb_v5.parquet`. Decoding them reproduces the base file's 87,000
  ongoing rows exactly.
* **Stage 2.** `dyn` features, 3 seeds, trained on all 1.73 M OOF rows
  (3,041 windows), predictions averaged. w = 0.8, then top-m decoding.
* **Data rule.** Features of a validation/private window use only its stage-1
  field and its own feature-table rows (released history, masked view at T,
  full-train profile), plus static link data. That is data ≤ T only.

Checks:
* `submit.check`: 174,000 rows, 0 missing, 0 extra, binary, positive rate
  0.0608.
* Onset lines are byte-identical to the base file.

Changes vs the base file (ongoing rows only):

| split | windows changed (of 40) | cells changed | added / removed | ongoing cells, base → new | mean (min) window agreement |
|---|---:|---:|---|---|---|
| validation | 27 | 139 | 94 / 45 | 5,365 → 5,414 | 0.945 (0.667) |
| private | 27 | 78 | 36 / 42 | 4,841 → 4,835 | 0.962 (0.500) |

Largest changes:
* D7_I10_W private 010: 14 → 7 cells.
* D12_I5_N validation 009: 9 → 6.
* D7_I210_E private 009: 15 → 22, the one private window with recurrence
  < 0.05.
* D7_I10_E validation 007 / 010: 121 → 148 and 137 → 159 (stage-1
  surrogate 0.80 / 0.82).

Other outputs:
* **Probabilities:** `/home/user/work/t2/probs_v9_ogstack08_ongoing.parquet`,
  with columns window_id, panel, k, link, p, p1, p2, split. It has the rows and
  order of the `probs_lgb_v5` ongoing rows, and `p1` equals v5 exactly. The
  stage-2 surrogate rises to 0.848 / 0.859 on validation / private.
* **Models:** `/home/user/work/t2/model_v9_ogstack08_stage2_s{0,1,2}_queue_ongoing.txt`.

Self-tests (`ongoing_v9 selftest`):
* (a) The test-time feature path on shuffled rows with window_id keys equals
  the training features exactly on D7_I10_W, D7_I405_N and D12_I5_S.
* (b) The 10 official train windows per panel, run through the
  released-history tables (`feat_<p>_train`), share 24,318 of 24,330 cells
  with the training rows. Only the window aggregates touched by the 12
  missing cells differ, and the recurrence differs by design (full-train
  profile).
* Re-decoding the saved probabilities reproduces the file byte for byte.

**Expected effect.** The CV ongoing gain of +0.0065 is about S_queue +0.003.
The gain sits in low-confidence and small-queue windows, which are
over-represented in March and April. The per-bucket reweighting suggests
+0.010 to +0.012 on ongoing.

### Reproduce

```
export PYTHONPATH=/home/user/knee OMP_NUM_THREADS=2 T2_THREADS=2 T2_WORK=/home/user/work/t2 T2_FEAT=/home/user/work/t2/feat_v3
# OOF stage 2 + table vs v5 + nested weight; writes WORK/ogstack_oof_<tag>.parquet (~2 min per seed)
python -m trafficflow.t2.og_stack dyn_s012 --seeds 0,1,2 --drop 'x_rec,x_tod,x_wkend,s_pcode,s_relpos'
python -m trafficflow.t2.og_stack base_s012 --seeds 0,1,2
python -m trafficflow.t2.og_stack noctx_s012 --seeds 0,1,2 --drop 'x_*,s_pcode,s_relpos'
# single-seed variants (tags of the table)
python -m trafficflow.t2.og_stack base_s0 --seeds 0
python -m trafficflow.t2.og_stack w_s0 --seeds 0 --weighted
python -m trafficflow.t2.og_stack comp_s0 --seeds 0 --comp
python -m trafficflow.t2.og_stack loc_s0 --seeds 0 --loc
python -m trafficflow.t2.og_stack nopcode_s0 --seeds 0 --drop s_pcode
python -m trafficflow.t2.og_stack noctx_s0 --seeds 0 --drop 'x_*,s_pcode,s_relpos'
python -m trafficflow.t2.og_stack field_s0 --seeds 0 --drop 'x_*,o_*,s_pcode,s_relpos'
python -m trafficflow.t2.og_stack noctx_big_s0 --seeds 0 --leaves 63 --rounds 500 --drop 'x_*,s_pcode,s_relpos'
python -m trafficflow.t2.og_stack table16 0.8          # the results table
python -m trafficflow.t2.og_stack diag dyn_s012 0.8    # folds, panels, confidence, size, growth, calibration baselines
python -m trafficflow.t2.og_stack bias dyn_s012 0.8    # decoder check
python -m trafficflow.t2.og_stack watch 0.8 dyn_s012 base_s012 noctx_s012
# candidate + probs_v9_<name>_ongoing.parquet + stage-2 models (~4 min)
python -m trafficflow.t2.ongoing_v9 ogstack08 --stack 0.8 --seeds 0,1,2 --drop 'x_rec,x_tod,x_wkend,s_pcode,s_relpos'
python -m trafficflow.t2.ongoing_v9 selftest
```

Resources:
* Stage-2 CV peaks at about 2.1 GB RSS and the build at about 2 GB.
* All runs used 2 threads.
* Logs are in `/home/user/work/t2/logs_og9/`. Stage-2 OOFs are
  `/home/user/work/t2/ogstack_oof_<tag>.parquet` (11 files, 27 MB each).
