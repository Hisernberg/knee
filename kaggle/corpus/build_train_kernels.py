#!/usr/bin/env python3
"""Generate per-fold Kaggle GPU training kernels for train_raptor96.py.

Each kernel: script type, T4x2, internet ON (timm downloads the ImageNet-12k CoAtNet-2 weights), inputs =
the three corpus96 kernel outputs + the competition tables; labels_consensus.csv is embedded in the script
folder. Output: raptor96_best.pt, raptor96_swa.pt, oof.csv, history.json, summary.json.

  python kaggle/corpus/build_train_kernels.py --owner kragglenote2forwork --folds 0,1 --out build/train
  kaggle kernels push -p build/train/fold0        # ~5-6 h on T4x2 per fold (12 epochs)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAUNCHER = '''# Kaggle launcher for train_raptor96.py (fold {fold}); the trainer itself is unchanged.
import os, subprocess, sys, glob
parts = sorted({{os.path.dirname(p) for p in glob.glob('/kaggle/input/**/vols_part*.npy', recursive=True)}})
assert parts, 'corpus96 parts not mounted'
comp = next(p for p in ('/kaggle/input/competitions/rsna-knee-abnormality-detection', '/kaggle/input/rsna-knee-abnormality-detection') if os.path.isfile(p + '/train.csv'))
cmd = [sys.executable, 'train_raptor96.py', '--corpus-dirs', ','.join(parts), '--train-csv', comp + '/train.csv',
       '--labels', 'labels_consensus.csv', '--fold', '{fold}', '--epochs', '{epochs}', '--bs', '{bs}', '--accum', '{accum}',
       '--k', '{k}', '--k-eval', '{k_eval}', '--arch', '{arch}', '--res', '{res}', '--workers', '4', '--out', '/kaggle/working']
print(' '.join(cmd), flush=True)
sys.exit(subprocess.call(cmd))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--folds", default="0")
    ap.add_argument("--out", default="build/train")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--k-eval", type=int, default=94)
    ap.add_argument("--arch", default="coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k")
    ap.add_argument("--res", type=int, default=384)
    ap.add_argument("--corpus-kernels", default="rsna-knee-corpus96-part0,rsna-knee-corpus96-part1,rsna-knee-corpus96-part2")
    a = ap.parse_args()
    for fold in [int(f) for f in a.folds.split(",")]:
        d = Path(a.out) / f"fold{fold}"
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE / "train_raptor96.py", d / "train_raptor96.py")
        shutil.copy2(HERE / "labels_consensus.csv", d / "labels_consensus.csv")
        (d / "launch.py").write_text(LAUNCHER.format(fold=fold, epochs=a.epochs, bs=a.bs, accum=a.accum, k=a.k, k_eval=a.k_eval, arch=a.arch, res=a.res))
        meta = {
            "id": f"{a.owner}/rsna-knee-raptor96-train-f{fold}", "title": f"RSNA Knee Raptor96 Train F{fold}",
            "code_file": "launch.py", "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"],
            "kernel_sources": [f"{a.owner}/{k}" for k in a.corpus_kernels.split(",")], "model_sources": [],
        }
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        print(d)


if __name__ == "__main__":
    main()
