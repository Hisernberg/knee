"""Build the Kaggle notebook and execute its cells locally against synthetic data (offline dry-run)."""
import os
from pathlib import Path

import nbformat
import pandas as pd

from kneemri.models import ModelConfig
from kneemri.preprocess import PrepConfig, prepare_split
from kneemri.schema import ID_COL, TARGETS, validate_submission
from kneemri.synthetic import make_synthetic_competition
from kneemri.train import TrainConfig, train_fold
from kneemri.folds import stratified_group_kfold

import importlib.util

spec = importlib.util.spec_from_file_location("build_notebook", Path(__file__).resolve().parents[1] / "kaggle" / "build_notebook.py")
bn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bn)


def test_notebook_builds_and_runs(tmp_path):
    root = make_synthetic_competition(tmp_path / "comp", n_train=12, n_test=4, size=40, seed=3, slices=(4, 6),
                                      series_per_study=(1, 2))
    prep = PrepConfig(size=40, ctx=3, max_centers=3, max_series=2)
    prepare_split(root, "train", tmp_path / "cache", prep)
    folds = stratified_group_kfold(pd.read_csv(root / "train.csv"), n_folds=2, seed=0)
    folds.to_csv(tmp_path / "folds.csv", index=False)
    res = train_fold(TrainConfig(cache_dir=str(tmp_path / "cache"), train_csv=str(root / "train.csv"),
                                 folds_csv=str(tmp_path / "folds.csv"), out_dir=str(tmp_path / "runs"), fold=0,
                                 epochs=1, batch_size=2, accum=1, num_workers=0, ema_decay=0.0, max_windows=4,
                                 model=ModelConfig(tiny=True, d_model=16, gru_hidden=8, n_heads=2), prep=prep))
    variant = {"checkpoints": [str(Path(res["ckpt"]).parent.parent / "fold*" / "best_filament_unet.pth")],
               "blend": "rank", "tta_flip": True, "workers": 0}
    nb_path = bn.build("vtest", variant, tmp_path / "build", "someone", "rsna-knee-infer", ["someone/weights"])
    assert (tmp_path / "build" / "kernel-metadata.json").is_file()
    nb = nbformat.read(nb_path, as_version=4)
    os.environ["KNEE_INPUT_ROOT"] = str(root)
    os.environ["KNEE_OUT_CSV"] = str(tmp_path / "submission.csv")
    g = {"__name__": "__main__"}
    # the embedded package must not shadow the repo copy in this process: exec cells in an isolated namespace
    for c in nb.cells:
        if c.cell_type == "code":
            exec(compile(c.source, "<cell>", "exec"), g)
    sub = pd.read_csv(tmp_path / "submission.csv")
    validate_submission(sub, pd.read_csv(root / "test.csv")[ID_COL].astype(str).tolist())
    assert list(sub.columns) == [ID_COL] + TARGETS
