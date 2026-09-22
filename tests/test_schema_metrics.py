import numpy as np
import pandas as pd
import pytest

from kneemri.metrics import macro_auc, rank_average, to_rank
from kneemri.schema import ID_COL, TARGETS, make_submission, validate_submission, write_submission


def test_submission_roundtrip(tmp_path):
    ids = [f"s{i}" for i in range(5)]
    probs = np.random.default_rng(0).random((5, len(TARGETS)))
    sub = make_submission(ids, probs)
    assert list(sub.columns) == [ID_COL] + TARGETS
    path = write_submission(sub, tmp_path / "submission.csv", test_ids=ids)
    back = pd.read_csv(path)
    assert list(back.columns) == [ID_COL] + TARGETS
    assert np.allclose(back[TARGETS].to_numpy(), probs, atol=1e-9)
    with pytest.raises(ValueError):
        validate_submission(sub, ids[:-1])
    bad = sub.copy()
    bad.loc[0, "ACL"] = np.nan
    with pytest.raises(ValueError):
        validate_submission(bad, ids)


def test_macro_auc_masks_nan():
    y = np.array([[1, np.nan], [0, 1], [1, 0], [0, np.nan]], dtype=float)
    p = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.3], [0.1, 0.2]])
    assert macro_auc(y, p) == pytest.approx(1.0)


def test_rank_average_permutation_invariant():
    a = np.array([[0.1, 0.5], [0.1, 0.4], [0.9, 0.4]])
    r = to_rank(a)
    assert r[0, 0] == r[1, 0]  # ties share the average rank
    perm = [2, 0, 1]
    rp = rank_average([a[perm]])
    assert np.allclose(rp, rank_average([a])[perm])
