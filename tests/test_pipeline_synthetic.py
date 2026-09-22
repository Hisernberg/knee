"""End-to-end CPU test on synthetic DICOM data: prepare -> folds -> train -> infer -> validate."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kneemri.dicom import read_series
from kneemri.folds import fold_audit, stratified_group_kfold
from kneemri.infer import InferConfig, run_inference
from kneemri.models import ModelConfig
from kneemri.preprocess import PrepConfig, prepare_split
from kneemri.schema import ID_COL, TARGETS, validate_submission
from kneemri.synthetic import make_synthetic_competition
from kneemri.train import TrainConfig, train_fold


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("comp")
    make_synthetic_competition(root, n_train=20, n_test=6, size=48, seed=1, slices=(5, 9), series_per_study=(2, 3))
    return root


def test_read_series_orders_and_inverts(synth):
    tr = pd.read_csv(synth / "train_series.csv")
    # study 0, series 1 is written as MONOCHROME1 -> must be inverted back to bright bone
    row = tr[(tr[ID_COL] == "train_study_0000")].iloc[1]
    vol = read_series(synth / "train_series" / row[ID_COL] / row["SeriesInstanceUID"])
    assert vol is not None and vol.n_slices >= 5 and vol.pixels.dtype == np.uint8
    assert vol.order_method == "position"
    # series 2 of study 0 has no ImagePositionPatient (i % 7 == 0) -> instance number fallback
    row2 = tr[(tr[ID_COL] == "train_study_0000")].iloc[2]
    vol2 = read_series(synth / "train_series" / row2[ID_COL] / row2["SeriesInstanceUID"])
    assert vol2.order_method == "instance_number"


def test_end_to_end(synth, tmp_path):
    prep = PrepConfig(size=48, ctx=3, max_centers=4, max_series=3)
    cache = tmp_path / "cache"
    manifest = prepare_split(synth, "train", cache, prep, workers=1)
    assert (manifest["n_windows"] > 0).all()
    train_df = pd.read_csv(synth / "train.csv")
    folds = stratified_group_kfold(train_df, n_folds=3, seed=0)
    assert set(folds["fold"]) == {0, 1, 2}
    audit = fold_audit(train_df, folds)
    assert len(audit) == 3
    folds.to_csv(tmp_path / "folds.csv", index=False)
    cfg = TrainConfig(cache_dir=str(cache), train_csv=str(synth / "train.csv"), folds_csv=str(tmp_path / "folds.csv"),
                      out_dir=str(tmp_path / "runs"), fold=0, epochs=2, batch_size=2, accum=1, num_workers=0,
                      ema_decay=0.0, max_windows=8, lr_encoder=1e-3, lr_head=1e-3,
                      model=ModelConfig(tiny=True, d_model=32, gru_hidden=16, n_heads=4, transformer_layers=1),
                      prep=prep)
    res = train_fold(cfg)
    ckpt = Path(res["ckpt"])
    assert ckpt.name == "best_filament_unet.pth" and ckpt.is_file()
    assert (ckpt.parent / "oof.csv").is_file()
    hist = json.loads((ckpt.parent / "history.json").read_text())
    assert len(hist) == 2 and np.isfinite(hist[-1]["train_loss"])
    out = run_inference(InferConfig(input_root=str(synth), checkpoints=(str(ckpt), str(ckpt)), weights=(1.0, 0.5),
                                    out_csv=str(tmp_path / "submission.csv"), tta_flip=True, blend="rank",
                                    num_workers=0, prior_csv=str(synth / "train.csv")))
    sub = pd.read_csv(out)
    test_ids = pd.read_csv(synth / "test.csv")[ID_COL].astype(str).tolist()
    validate_submission(sub, test_ids)
    assert list(sub.columns) == [ID_COL] + TARGETS
    receipt = json.loads((tmp_path / "submission.receipt.json").read_text())
    assert receipt["n_test"] == len(test_ids) and len(receipt["arms"]) == 2
