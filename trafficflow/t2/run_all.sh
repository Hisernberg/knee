#!/bin/bash
# End-to-end Task 2 pipeline (<= 2 threads, peak RSS ~3.6 GB). Outputs in $T2_WORK (default /home/user/work/t2).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD OMP_NUM_THREADS=2
python -W ignore -m trafficflow.t2.dataset                          # selector simulation + window datasets (ds_<panel>.npz), ~4 min
python -W ignore -m trafficflow.t2.baselines                        # rule baselines on simulated / official train windows
python -W ignore -m trafficflow.t2.kinematic                        # persistence variants + kinematic rule (ongoing)
python -W ignore -m trafficflow.t2.build_features                   # feature tables (feat_v2/), ~5 min
python -W ignore -m trafficflow.t2.cv queue_onset p1 --oprior       # final onset config, 4-fold CV (~4 min)
python -W ignore -m trafficflow.t2.cv queue_ongoing p2 --weighted   # final ongoing config, 4-fold CV (~35 min)
python -W ignore -m trafficflow.t2.write_rules                      # persistence / persistence_fill / range_prior csvs
python -W ignore -m trafficflow.t2.pipeline lgb_v3 --onset-cfg p1 --ongoing-cfg p2 --ongoing-weighted   # ~15 min
# shift-robust variant (section 11 of TASK2_ANALYSIS.md)
python -W ignore -m trafficflow.t2.robust cv noloc p2 --weighted     # CV of the no-location-prior ongoing model
python -W ignore -c "from trafficflow.t2.robust_pipeline import train_variant; from trafficflow.t2.core import WORK; train_variant('noloc','p2',True).save_model(str(WORK/'model_rob_noloc_p2w_queue_ongoing.txt'))"
python -W ignore -m trafficflow.t2.robust_pipeline lgb_v4_robust --onset-model lgb_v3 --ongoing lgb_v3:0.5 --ongoing rob_noloc_p2w:0.5
# v5: physics (onset) + LWR shockwave (ongoing) features, seed/model blends (section 12)
export T2_FEAT=/home/user/work/t2/feat_v3
T2_PHYSICS=1 python -W ignore -m trafficflow.t2.build_features
for s in 0 1 2; do T2_SEED=$s python -W ignore -m trafficflow.t2.robust cv on_v3 p1 --cond queue_onset --oprior; done
python -W ignore -m trafficflow.t2.robust cv og_v3 p2 --weighted
python -W ignore -m trafficflow.t2.robust cv og_v3_noloc p2 --weighted
python -W ignore -m trafficflow.t2.robust_pipeline lgb_v5 \
  --onset train:on_v3:p1:nw:op:0.25:0 --onset train:on_v3:p1:nw:op:0.25:1 --onset train:on_v3:p1:nw:op:0.25:2 --onset lgb_v3:0.25 \
  --ongoing train:og_v3:p2:w:noop:0.35 --ongoing train:og_v3_noloc:p2:w:noop:0.35 --ongoing lgb_v3:0.15 --ongoing rob_noloc_p2w:0.15
