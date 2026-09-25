"""lgb_v9 ongoing candidates: lgb_v8_seeds9_stack03.csv with only the ongoing rows replaced.

Stage 1: the v5 ongoing blend. Its validation/private probabilities are the
ongoing rows of ``probs_lgb_v5.parquet`` (decoding them with top-m reproduces
the ongoing rows of the base file exactly). With ``--comp`` the four saved v5
component models are re-run on the validation/private tables to get the loc /
noloc component probabilities (their 0.35/0.35/0.15/0.15 mean is checked
against probs_lgb_v5).

Stage 2 (``og_stack``): trained on the OOF stage-1 rows of all 3,041 ongoing
evaluation windows (sim + off, old labels), one model per seed, predictions
averaged. The features of a validation/private window use only its own
stage-1 field, its feature-table rows (released history, masked view at slot
T, full-train time-of-day profile) and static link data: data <= T only.

Final probability ``W * p2 + (1 - W) * p1``, top-m expected-IoU decoding.

    T2_WORK=/home/user/work/t2 T2_FEAT=/home/user/work/t2/feat_v3 \
    python -m trafficflow.t2.ongoing_v9 NAME --stack 0.5 --seeds 0,1,2 [--weighted] [--comp] [--loc]
        [--drop COLS] [--rounds 300] [--leaves 31] [--min-data 100]

Writes /home/user/work/t2/lgb_v9_<NAME>.csv and WORK/probs_v9_<NAME>_ongoing.parquet
(window_id, panel, k, link, p, p1, p2, split).
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
from . import og_stack as og
from .core import FEAT, K, PANELS8, WORK, official_windows, statics
from .models import eiou_topm
from .robust import recurrence
from .submit import check

BASE_CSV = Path("/home/user/work/t2/lgb_v8_seeds9_stack03.csv")
OUT_DIR = Path("/home/user/work/t2")
PKEY = ["window_id", "panel", "k", "link"]
# v5 component models (robust_pipeline names) -> the OOF file of the same recipe
COMPONENTS = {"lgb_v5_og_v3": "oof_queue_ongoing_rob_og_v3_p2_w.parquet",
              "lgb_v5_og_v3_noloc": "oof_queue_ongoing_rob_og_v3_noloc_p2_w.parquet",
              "lgb_v3": "oof_queue_ongoing_p2w.parquet",
              "rob_noloc_p2w": "oof_queue_ongoing_rob_noloc_p2_w.parquet"}


def stage1_test(comp: bool = False) -> pd.DataFrame:
    P = pd.read_parquet(WORK / "probs_lgb_v5.parquet")
    P = P[P.condition == "queue_ongoing"].reset_index(drop=True)[PKEY + ["p", "split"]]
    if not comp:
        return P
    from .robust_pipeline import probs
    acc = {c: np.zeros(len(P)) for c in ("p", "p_loc", "p_noloc")}
    wl = sum(og.V5[f] for f in og.LOC_FILES); wn = sum(og.V5[f] for f in og.NOLOC_FILES)
    for name, f in COMPONENTS.items():
        m = lgb.Booster(model_file=str(WORK / f"model_{name}_queue_ongoing.txt"))
        parts = [probs(m, s, "queue_ongoing").assign(split=s) for s in ("validation", "private")]
        q = P[PKEY + ["split"]].merge(pd.concat(parts), on=PKEY + ["split"], how="left")
        assert q.p.notna().all(), name
        acc["p"] += og.V5[f] * q.p.to_numpy()
        if f in og.LOC_FILES:
            acc["p_loc"] += og.V5[f] / wl * q.p.to_numpy()
        if f in og.NOLOC_FILES:
            acc["p_noloc"] += og.V5[f] / wn * q.p.to_numpy()
        del m; gc.collect()
    d = np.abs(acc["p"] / sum(og.V5.values()) - P.p.to_numpy()).max()
    print("component blend vs probs_lgb_v5: max |dp| =", d, flush=True)
    assert d < 1e-5
    return P.assign(p_loc=acc["p_loc"], p_noloc=acc["p_noloc"])


def stage2_models(seeds, params: dict, rounds: int, weighted=False, comp=False, loc=False, drop=()):
    """Stage-2 models trained on every OOF row (all four folds)."""
    X, cols = og.build(og.stage1(og.V5, comp=comp), use_loc=loc, drop=drop)
    ds = lgb.Dataset(X[cols].to_numpy(np.float32), X.y.to_numpy(np.float32), feature_name=list(cols),
                     weight=og.window_weights(X.gw.to_numpy()) if weighted else None,
                     params={"verbose": -1, "max_bin": 255}).construct()
    del X; gc.collect()
    return [lgb.train({**params, "seed": s}, ds, rounds) for s in seeds], cols


def test_features(P: pd.DataFrame, split: str, cols: list[str], loc=False) -> pd.DataFrame:
    """Stage-2 features of one split's stage-1 rows P (window_id, panel, k, link, p[, comps]), aligned to P."""
    rec = recurrence(split)
    out = []
    for panel, R in P.groupby("panel", sort=False):
        M = pd.read_parquet(FEAT / f"meta_{panel}_{split}.parquet").set_index("window_id")
        ws = M.w.reindex(R.window_id.unique()).astype(int).tolist()
        Fr = og.read_ctx(panel, split, ws, loc)
        Fr["window_id"] = Fr.w.map(M.reset_index().set_index("w").window_id)
        out.append(og.panel_rows(R, Fr, panel, rec, "window_id", loc, cols))
    return pd.concat(out).loc[P.index]


def stage2_probs(P: pd.DataFrame, split: str, models, cols, loc=False) -> np.ndarray:
    X = test_features(P, split, cols, loc)
    A = X[cols].to_numpy(np.float32)
    p = np.mean([m.predict(A, num_threads=og.THREADS) for m in models], 0)
    assert np.isfinite(p).all()
    return p


def decode(B: pd.DataFrame) -> dict:
    preds = {}
    for wid, g in B.groupby("window_id"):
        L = statics(g.panel.iloc[0])["L"]
        kk = g.k.to_numpy().astype(int) - 1; ll = g.link.to_numpy().astype(int)
        idx = eiou_topm(g.p.to_numpy())[0]
        A = np.zeros((K, L), bool); A[kk[idx], ll[idx]] = True
        preds[wid] = A
    return preds


def write_from_base(preds: dict, out: Path, base: Path = BASE_CSV) -> dict:
    """base file with the ongoing rows replaced; returns change counts vs base."""
    sub = pd.read_csv(base, dtype=str)
    W = pd.concat([official_windows(p, s) for p in PANELS8 for s in ("validation", "private")]).set_index("window_id")
    og_ = sub.window_id.map(W.condition).to_numpy() == "queue_ongoing"
    k = tindex(sub.timestamp) - sub.window_id.map(W["T"]).to_numpy() - 1
    q_old = sub.queue_pred.astype(int).to_numpy()
    q = q_old.copy()
    for i in np.flatnonzero(og_):
        wid = sub.window_id.iat[i]
        lid = statics(W.loc[wid, "panel"])["lid"][sub.link_id.iat[i]]
        q[i] = int(preds[wid][k[i], lid])
    sub["queue_pred"] = q
    sub.to_csv(out, index=False)
    d = pd.DataFrame({"wid": sub.window_id[og_], "old": q_old[og_], "new": q[og_]})
    d["split"] = np.where(d.wid.str.contains("_validation_"), "validation", "private")
    res = {"onset_rows_changed": int((q != q_old)[~og_].sum())}
    for s, g0 in d.groupby("split"):
        per = g0.groupby("wid").apply(lambda g: pd.Series({"inter": int(((g.old == 1) & (g.new == 1)).sum()),
                                                           "union": int(((g.old == 1) | (g.new == 1)).sum()),
                                                           "n_old": int(g.old.sum()), "n_new": int(g.new.sum())}))
        agree = np.where(per.union > 0, per.inter / per.union.clip(lower=1), 1.0)
        res[s] = dict(windows=int(len(per)), windows_changed=int((per.inter != per.union).sum()),
                      cells_changed=int((g0.old != g0.new).sum()), cells_added=int(((g0.new == 1) & (g0.old == 0)).sum()),
                      cells_removed=int(((g0.new == 0) & (g0.old == 1)).sum()), cells_base=int(per.n_old.sum()),
                      cells_new=int(per.n_new.sum()), mean_window_agreement=round(float(agree.mean()), 4),
                      min_window_agreement=round(float(agree.min()), 4), empty_windows=int((per.n_new == 0).sum()))
    return res


def onset_lines_identical(a: Path, b: Path) -> bool:
    """Raw text of every non-ongoing row (and the header) is byte-identical."""
    W = pd.concat([official_windows(p, s) for p in PANELS8 for s in ("validation", "private")]).set_index("window_id")
    la = a.read_bytes().split(b"\n"); lb = b.read_bytes().split(b"\n")
    if len(la) != len(lb) or la[0] != lb[0]:
        return False
    for x, y in zip(la[1:], lb[1:]):
        if not x:
            if y:
                return False
            continue
        wid = x.split(b",", 1)[0].decode()
        if W.condition.get(wid) != "queue_ongoing" and x != y:
            return False
    return True


def selftest(panels=("D7_I10_W", "D7_I405_N", "D12_I5_S")) -> None:
    """(a) The test-time feature path (window_id keys, shuffled rows) equals the
    training features of ``og_stack.build`` exactly. (b) The 10 official train
    windows per panel through the test path (feat_<p>_train tables, built from
    the released window histories like validation/private) against the training
    rows of the same windows (``off``), recurrence excluded (the test path uses
    the full-train profile)."""
    S = og.stage1(og.V5)
    X, cols = og.build(S)
    rng = np.random.default_rng(0)
    rec = recurrence("train")
    for panel in panels:
        pc = PANELS8.index(panel)
        R = S[S.gw // 100000 == pc]
        R = R.iloc[rng.permutation(len(R))].assign(window_id=lambda d: "g" + d.gw.astype(str), panel=panel)
        ws = np.unique(R.gw % 100000).tolist()
        Fr = og.read_ctx(panel, "cv", ws, False)
        Fr["window_id"] = "g" + (pc * 100000 + Fr.w.astype(np.int64)).astype(str)
        Fr = Fr.iloc[rng.permutation(len(Fr))]
        rs = pd.Series(rec.to_numpy(), index="g" + rec.index.astype(str))
        A = og.panel_rows(R.drop(columns="gw"), Fr, panel, rs, "window_id", False, cols)
        B = X.loc[A.index, cols]
        same = all(np.array_equal(A[c].to_numpy(), B[c].to_numpy(), equal_nan=True) for c in cols)
        print(f"(a) {panel}: {len(A)} shuffled rows, test path == training features: {same}", flush=True)
        assert same
    # (b) official train windows
    Mi = og.meta_index("queue_ongoing")
    off = X[X.src == "off"]
    rows = []
    for panel in PANELS8:
        pc = PANELS8.index(panel)
        o = off[off.gw // 100000 == pc]
        Mt = pd.read_parquet(FEAT / f"meta_{panel}_train.parquet")
        Mt = Mt[Mt.condition == "queue_ongoing"].set_index("T")
        wid = Mi.loc[o.gw.to_numpy(), "T"].map(Mt.window_id).to_numpy()
        R = o[og.KEY + ["p"]].assign(window_id=wid, panel=panel).drop(columns="gw")
        Ft = pd.read_parquet(FEAT / f"feat_{panel}_train.parquet", columns=["w", "k", "link"])
        Ft["window_id"] = Ft.w.map(Mt.reset_index().set_index("w").window_id)
        Ft = Ft[Ft.window_id.isin(set(wid))]
        common = R.merge(Ft[["window_id", "k", "link"]], on=["window_id", "k", "link"]).shape[0]
        A = test_features(R.loc[R.set_index(["window_id", "k", "link"]).index.isin(
            Ft.set_index(["window_id", "k", "link"]).index)], "train", cols)
        B = off.loc[A.index, cols]
        for c in cols:
            a, b = A[c].to_numpy(), B[c].to_numpy()
            rows.append(dict(panel=panel, col=c, rows=len(A), rows_off=len(R), rows_common=common,
                             equal=float(np.mean(np.isclose(a, b, equal_nan=True, atol=1e-5)))))
    D = pd.DataFrame(rows)
    print("(b) official train windows via the test path: rows (off / common / compared):",
          D.groupby("panel")[["rows_off", "rows_common", "rows"]].first().sum().to_dict(), flush=True)
    t = D.groupby("col").equal.min().sort_values()
    print("share of equal values, worst panel per column (x_rec differs by design):\n" + t.head(15).round(4).to_string())


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--stack", type=float, required=True, help="stage-2 blend weight W")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--min-data", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--weighted", action="store_true")
    ap.add_argument("--comp", action="store_true")
    ap.add_argument("--loc", action="store_true")
    ap.add_argument("--drop", default="")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--probs-dir", default=str(WORK))
    a = ap.parse_args()
    drop = tuple(c for c in a.drop.split(",") if c)
    P = stage1_test(a.comp)
    params = {**og.P2, "num_leaves": a.leaves, "min_data_in_leaf": a.min_data, "learning_rate": a.lr}
    ms, cols = stage2_models([int(s) for s in a.seeds.split(",")], params, a.rounds, a.weighted, a.comp, a.loc, drop)
    for i, m in enumerate(ms):
        m.save_model(str(WORK / f"model_v9_{a.name}_stage2_s{i}_queue_ongoing.txt"))
    P["p1"] = P.p.to_numpy(); P["p2"] = np.nan
    for split in ("validation", "private"):
        idx = np.flatnonzero(P.split.to_numpy() == split)
        P.loc[P.index[idx], "p2"] = stage2_probs(P.iloc[idx], split, ms, cols, a.loc)
    P["p"] = a.stack * P.p2 + (1 - a.stack) * P.p1
    P[PKEY + ["p", "p1", "p2", "split"]].to_parquet(Path(a.probs_dir) / f"probs_v9_{a.name}_ongoing.parquet")
    out = Path(a.out_dir) / f"lgb_v9_{a.name}.csv"
    ch = write_from_base(decode(P), out)
    ch["onset_lines_byte_identical"] = onset_lines_identical(BASE_CSV, out)
    print(json.dumps({"file": str(out), "check": check(out), "vs_base": ch, "features": len(cols)}), flush=True)


if __name__ == "__main__":
    main()
