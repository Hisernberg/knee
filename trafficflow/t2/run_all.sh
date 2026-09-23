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
