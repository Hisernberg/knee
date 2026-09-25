"""lgb_v8 onset candidates: lgb_v6.csv with only the onset rows replaced.

Stage 1: the mean of LightGBM onset models of the v6 recipe (p1 config,
location prior, hybrid labels, every train onset window of the re-drawn set),
given as ``variant:seed[:weight]`` specs. The four v6 models are loaded from
``WORK/model_v6_<variant>_s<seed>_queue_onset.txt``. Any other model is
trained once and saved as ``WORK/model_v8_<variant>_s<seed>_queue_onset.txt``.

Stage 2 (``--stack W``, optional): ``stack.panel_features`` on the stage-1
probability profile of the window. It is trained on the OOF stage-1 mean of
the same specs (``oof_queue_onset_rob_<variant>_p1_op[_s<seed>]_all_new``;
every train onset window, hybrid labels, all four folds) with the stage-2
seeds given, and the seed predictions are averaged. The final probability is
``W * p2 + (1 - W) * p1``.

Decoding: top-m expected IoU at T+30. Steps 1-5 of onset windows stay empty.
Features of the validation/private windows use only data <= T (released
history, masked view up to T, train profiles, train location prior). Stage 2
only reads the same window's stage-1 probabilities and static link data.

    T2_WORK=/home/user/work/t2h T2_FEAT=/home/user/work/t2h/feat \
    python -m trafficflow.t2.onset_v8 NAME --specs on_v3:0,on_v3:1,on_v3:2,on_v2:0 [--stack 0.3 --stack-seeds 0,1,2]

``--specs`` items are ``variant:seed[:weight]`` (weight 1 by default).

Writes /home/user/work/t2/lgb_v8_<NAME>.csv and WORK/probs_v8_<NAME>_onset.parquet
(window_id, panel, k, link, p, split; the columns of probs_v6_onset.parquet).
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..data import tindex
from . import stack
from .core import K, PANELS8, WORK, official_windows, statics
from .models import eiou_topm
from .robust_pipeline import probs, train_variant
from .submit import check

V6_CSV = Path("/home/user/work/t2/lgb_v6.csv")
OUT_DIR = Path("/home/user/work/t2")
KEY = ["window_id", "panel", "k", "link"]


def oof_file(variant: str, seed: int) -> str:
    return f"oof_queue_onset_rob_{variant}_p1_op{f'_s{seed}' if seed else ''}_all_new.parquet"


def stage1_model(variant: str, seed: int) -> lgb.Booster:
    for f in (WORK / f"model_v6_{variant}_s{seed}_queue_onset.txt", WORK / f"model_v8_{variant}_s{seed}_queue_onset.txt"):
        if f.exists():
            return lgb.Booster(model_file=str(f))
    m = train_variant(variant, "p1", False, cond="queue_onset", seed=seed, oprior=True)
    m.save_model(str(WORK / f"model_v8_{variant}_s{seed}_queue_onset.txt"))
    return m


def stage1_probs(specs, split: str) -> pd.DataFrame:
    parts = []
    for variant, seed, _ in specs:
        m = stage1_model(variant, seed)
        parts.append(probs(m, split, "queue_onset"))
        del m; gc.collect()
    B = parts[0][KEY].copy()
    for q in parts[1:]:
        assert (q[KEY].to_numpy() == B[KEY].to_numpy()).all()
    w = np.array([wt for _, _, wt in specs], np.float64)
    P = np.stack([q.p.to_numpy() for q in parts])
    B["p"] = P.mean(0) if len(set(w)) == 1 else np.tensordot(w / w.sum(), P, 1)   # mean: bit-identical to v6
    return B


def stage2_models(specs, seeds, rounds=300, leaves=31, name="tmp") -> tuple[list[lgb.Booster], list[str]]:
    """Stage-2 models trained on every train onset window (OOF stage-1 weighted mean of the specs)."""
    from .stack_v8 import mean_oof
    stack.EXTRA = False
    files = [oof_file(v, s) for v, s, _ in specs]
    weights = [w for _, _, w in specs]
    tmp = mean_oof(files, weights, f"v8_{name}") if len(set(weights)) > 1 else None
    X, cols = stack.build([tmp] if tmp else files)
    if tmp:
        (WORK / tmp).unlink()
    ds = lgb.Dataset(X[cols].to_numpy(np.float32), X.y.to_numpy(np.float32), feature_name=cols).construct()
    del X; gc.collect()
    ms = [lgb.train({**stack.P2, "num_leaves": leaves, "seed": s}, ds, rounds) for s in seeds]
    return ms, cols


def stage2_probs(P: pd.DataFrame, models: list[lgb.Booster], cols: list[str]) -> np.ndarray:
    """P: one split's stage-1 rows (window_id, panel, link, p); returns stage-2 p aligned to P."""
    out = np.full(len(P), np.nan)
    for panel, g in P.groupby("panel", sort=False):
        L = statics(panel)["L"]
        wids = g.window_id.unique()
        pos = {w: i for i, w in enumerate(wids)}
        wi = g.window_id.map(pos).to_numpy(); li = g.link.to_numpy().astype(int)
        Pm = np.zeros((len(wids), L), np.float32); Pm[wi, li] = g.p.to_numpy()
        F = stack.panel_features(Pm, panel)
        Xf = np.stack([F[c][wi, li] for c in cols], 1).astype(np.float32)
        out[g.index.to_numpy()] = np.mean([m.predict(Xf, num_threads=stack.P2["num_threads"]) for m in models], 0)
    assert np.isfinite(out).all()
    return out


def decode(B: pd.DataFrame) -> dict:
    preds = {}
    for wid, g in B.groupby("window_id"):
        L = statics(g.panel.iloc[0])["L"]
        ll = g.link.to_numpy().astype(int)
        A = np.zeros(L, bool); A[ll[eiou_topm(g.p.to_numpy())[0]]] = True
        preds[wid] = A
    return preds


def write_from_v6(preds: dict, out: Path) -> dict:
    """lgb_v6.csv with the onset rows replaced; returns change counts vs v6."""
    sub = pd.read_csv(V6_CSV, dtype=str)
    W = pd.concat([official_windows(p, s) for p in PANELS8 for s in ("validation", "private")]).set_index("window_id")
    onset = sub.window_id.map(W.condition).to_numpy() == "queue_onset"
    k = tindex(sub.timestamp) - sub.window_id.map(W["T"]).to_numpy() - 1
    q_old = sub.queue_pred.astype(int).to_numpy()
    q = q_old.copy()
    for i in np.flatnonzero(onset):
        wid = sub.window_id.iat[i]
        lid = statics(W.loc[wid, "panel"])["lid"][sub.link_id.iat[i]]
        q[i] = int(k[i] == K - 1 and preds[wid][lid])
    sub["queue_pred"] = q
    sub.to_csv(out, index=False)
    ch = q != q_old
    d = pd.DataFrame({"wid": sub.window_id[onset], "old": q_old[onset], "new": q[onset]})
    per = d.groupby("wid").apply(lambda g: pd.Series({"inter": int(((g.old == 1) & (g.new == 1)).sum()),
                                                      "union": int(((g.old == 1) | (g.new == 1)).sum()),
                                                      "n_old": int(g.old.sum()), "n_new": int(g.new.sum())}))
    agree = np.where(per.union > 0, per.inter / per.union.clip(lower=1), 1.0)
    return dict(onset_windows=int(len(per)), windows_changed=int((per.inter != per.union).sum()),
                cells_changed=int(ch.sum()), cells_added=int(((q == 1) & (q_old == 0)).sum()),
                cells_removed=int(((q == 0) & (q_old == 1)).sum()), onset_cells_v6=int(per.n_old.sum()),
                onset_cells_new=int(per.n_new.sum()), ongoing_rows_changed=int(ch[~onset].sum()),
                mean_window_agreement=round(float(agree.mean()), 4), min_window_agreement=round(float(agree.min()), 4),
                cells_per_window_min=int(per.n_new.min()), cells_per_window_median=float(per.n_new.median()),
                cells_per_window_max=int(per.n_new.max()),
                onset_cells_not_at_T30=int(((q == 1) & onset & (k != K - 1)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--specs", default="on_v3:0,on_v3:1,on_v3:2,on_v2:0")
    ap.add_argument("--stack", type=float, default=0.0, help="stage-2 blend weight W (0 = stage 1 only)")
    ap.add_argument("--stack-seeds", default="0")
    ap.add_argument("--stack-rounds", type=int, default=300)
    ap.add_argument("--stack-leaves", type=int, default=31)
    ap.add_argument("--out-dir", default=str(OUT_DIR), help="where lgb_v8_<NAME>.csv goes")
    ap.add_argument("--probs-dir", default=str(WORK), help="where probs_v8_<NAME>_onset.parquet goes")
    a = ap.parse_args()
    specs = [(f[0], int(f[1]), float(f[2]) if len(f) > 2 else 1.0) for f in (s.split(":") for s in a.specs.split(","))]
    allp = []
    for split in ("validation", "private"):
        B = stage1_probs(specs, split)
        allp.append(B.assign(split=split))
    P = pd.concat(allp, ignore_index=True)
    if a.stack > 0:
        ms, cols = stage2_models(specs, [int(s) for s in a.stack_seeds.split(",")], a.stack_rounds, a.stack_leaves,
                                 name=a.name)
        p2 = np.empty(len(P))
        for split in ("validation", "private"):
            idx = np.flatnonzero(P.split.to_numpy() == split)
            p2[idx] = stage2_probs(P.iloc[idx].reset_index(drop=True), ms, cols)
        P["p"] = a.stack * p2 + (1 - a.stack) * P.p.to_numpy()
    P[KEY + ["p", "split"]].to_parquet(Path(a.probs_dir) / f"probs_v8_{a.name}_onset.parquet")
    preds = decode(P)
    out = Path(a.out_dir) / f"lgb_v8_{a.name}.csv"
    ch = write_from_v6(preds, out)
    print(json.dumps({"file": str(out), "check": check(out), "vs_v6": ch}), flush=True)


if __name__ == "__main__":
    main()
