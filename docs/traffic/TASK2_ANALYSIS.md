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
