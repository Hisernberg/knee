"""Stage-2 onset stacking on the hybrid-truth OOF (re-drawn windows, t2h).

``stack.build`` / ``stack.cv`` on the t2h OOF files: stage 1 = the mean of
the given onset OOF files (every onset window, trained on hybrid labels);
stage 2 = LightGBM on the stage-1 probability profile of the window (block
shape in traffic direction, see ``stack.panel_features``), trained on the
hybrid labels with the same 4 week folds (OOF stacking; every source,
cand/sim/off, is used for training). Scored with ``onset_eval`` under the
hybrid and the old truth, paired bootstrap against v6.

    T2_WORK=/home/user/work/t2h T2_FEAT=/home/user/work/t2h/feat \
    python -m trafficflow.t2.stack_v8 TAG [OOF_FILE ...] [--weights W,...] [--extra] [--seeds 0,1,2]
        [--rounds 300] [--leaves 31]

Writes WORK/stack8_oof_<TAG>.parquet (gw, link, y, fold, src, p1, p2).
``python -m trafficflow.t2.stack_v8 table`` prints the section 15 comparison.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from . import stack
from .core import K, WORK
from .onset_eval import V6, OnsetEval, table


def mean_oof(files: list[str], weights: list[float], tag: str) -> str:
    """Weighted mean of OOF files (same rows) written to WORK/oof_queue_onset_mean_<tag>.parquet
    (``stack.build`` averages its files with equal weights). Returns the file name."""
    B = None
    for f, w in zip(files, weights):
        x = pd.read_parquet(WORK / f)
        if B is None:
            B = x[["gw", "k", "link", "y"]].copy(); B["p"] = 0.0
        assert (x[["gw", "k", "link"]].to_numpy() == B[["gw", "k", "link"]].to_numpy()).all(), f
        B["p"] += w * x.p.to_numpy(np.float64)
    B["p"] = (B.p / sum(weights)).astype(np.float32)
    name = f"oof_queue_onset_mean_{tag}.parquet"
    B.to_parquet(WORK / name)
    return name


def stage2_oof(files: list[str], extra=False, seeds=(0,), rounds=300, leaves=31, min_data=100, lr=0.05,
               weights: list[float] | None = None, tag: str = "tmp"):
    stack.EXTRA = extra
    if weights is not None and len(set(weights)) > 1:
        files = [mean_oof(files, weights, tag)]
    X, cols = stack.build(files)
    p2 = np.zeros(len(X))
    for s in seeds:
        p2 += stack.cv(X, cols, {**stack.P2, "num_leaves": leaves, "min_data_in_leaf": min_data,
                                 "learning_rate": lr, "seed": s}, rounds)
    X["p1"] = X["s_p"]
    X["p2"] = p2 / len(seeds)
    return X, cols


def nested_weight(E: OnsetEval, p1: np.ndarray, p2: np.ndarray, weights=(0.0, 0.2, 0.3, 0.4, 0.5, 0.7),
                  truth: str = "hybrid") -> tuple[pd.DataFrame, dict]:
    """Honest blend-weight selection: for each week fold, pick the weight that
    maximises the sim score (official aggregation, ``truth``) on the other three
    folds' windows and use it on the held-out fold. Returns the combined
    window_iou frame and the weight chosen per fold."""
    D = {w: E.window_iou(w * p2 + (1 - w) * p1) for w in weights}
    fold = E.meta.fold.reindex(D[weights[0]].gw.to_numpy()).to_numpy()
    out = D[weights[0]].copy()
    chosen = {}
    for f in range(4):
        tr = (fold != f) & (out.src == "sim").to_numpy()
        best = max(weights, key=lambda w: E.agg(D[w][tr], f"iou_{truth}"))
        chosen[f] = best
        te = fold == f
        for t in E.truths:
            out.loc[te, f"iou_{t}"] = D[best].loc[te, f"iou_{t}"].to_numpy()
        out.loc[te, "m"] = D[best].loc[te, "m"].to_numpy()
    return out, chosen


def saved(tag: str, E: OnsetEval | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(p1, p2) of a saved WORK/stack8_oof_<tag>.parquet aligned to the evaluator rows."""
    E = E or OnsetEval()
    X = pd.read_parquet(WORK / f"stack8_oof_{tag}.parquet")
    X["k"] = K
    return E.align(X, "p1"), E.align(X, "p2")


def oof_name(variant: str, seed: int) -> str:
    return f"oof_queue_onset_rob_{variant}_p1_op{f'_s{seed}' if seed else ''}_all_new.parquet"


V3X6 = [oof_name("on_v3", s) for s in range(6)]
V2X3 = [oof_name("on_v2", s) for s in range(3)]
W9 = [3.0] * 6 + [2.0] * 3          # 0.75 on the six on_v3 seeds, 0.25 on the three on_v2 seeds (v6 proportions)


def v8_table(n_boot: int = 2000) -> pd.DataFrame:
    """Section 15 comparison from saved OOF files (run the stack_v8 CLI for the stack8_oof_* tags first)."""
    E = OnsetEval()
    p6 = E.load_mean(V6)
    V = {"v6": p6, "seeds9": E.load_mean(V3X6 + V2X3, W9), "v3x6": E.load_mean(V3X6)}
    for name, tag, stage1 in (("stack03", "v6s012", p6), ("seeds9+stack03", "seeds9s012", V["seeds9"]),
                              ("v3x6+stack03", "v3x6s012", V["v3x6"])):
        if (WORK / f"stack8_oof_{tag}.parquet").exists():
            p1, p2 = saved(tag, E)
            assert np.allclose(p1, stage1, atol=1e-6), tag
            V[name] = 0.3 * p2 + 0.7 * p1
    t = table(E, V, ref="v6", n_boot=n_boot)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    print(t.round(4).T.to_string(), flush=True)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--extra", action="store_true")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--min-data", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--weights", default=None, help="comma list of stage-1 weights, aligned with the files")
    a = ap.parse_args()
    files = a.files or V6
    if a.tag == "table":
        v8_table()
        return
    weights = [float(w) for w in a.weights.split(",")] if a.weights else None
    t = time.time()
    X, cols = stage2_oof(files, a.extra, [int(s) for s in a.seeds.split(",")], a.rounds, a.leaves, a.min_data, a.lr,
                         weights=weights, tag=a.tag)
    X[["gw", "link", "y", "fold", "src", "p1", "p2"]].to_parquet(WORK / f"stack8_oof_{a.tag}.parquet")
    E = OnsetEval()
    X["k"] = K
    p1 = E.align(X, "p1"); p2 = E.align(X, "p2")
    V = {"v6": E.load_mean(V6), "stage1": p1, "stage2": p2}
    assert np.allclose(p1, E.load_mean(files, weights), atol=1e-6), "stage-1 mean mismatch"
    for w in (0.2, 0.3, 0.4, 0.5, 0.7):
        V[f"blend{w}"] = w * p2 + (1 - w) * p1
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    print(a.tag, files, f"{len(cols)} stage-2 features, {time.time() - t:.0f}s")
    print(table(E, V, ref="v6").round(4).T.to_string(), flush=True)
    nest, chosen = nested_weight(E, p1, p2)
    c = E.compare(nest, E.window_iou(V["v6"]))
    print("nested weight per fold", chosen, {t: {k: round(v, 4) for k, v in c[t].items()} for t in E.truths}, flush=True)


if __name__ == "__main__":
    main()
