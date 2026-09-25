"""Shift-weighted cross-validation for Task 2 (TASK2_ANALYSIS.md section 17).

Plain CV scores every selector-drawn train window (``sim``) equally. The public
month (March = validation) and the private month (April) draw a different mix
of windows: more non-recurrent queues, more weekend windows (March starts on a
Saturday), smaller and dissipating queues, lower model confidence. This module
re-weights the CV windows toward one target month:

1. **Window features** (``window_features``), computed with the same code for
   the CV windows and the official validation / private windows, from data
   <= T only: the feature tables (released history, masked view at slot T,
   train time-of-day profiles; fold-excluded for CV windows, full-train for the
   official windows, as the models see them), the history coverage and the
   stage-1 probabilities of the reference model (v5 ongoing, v6 onset; OOF for
   CV windows, the full-train models for validation / private).
2. **Importance weights** (``weights``): a cross-fitted, L2-regularised
   logistic regression separates the 40 target windows of a condition from the
   CV sim windows (panel fixed effects, so only within-panel differences
   count). ``w = p / (1 - p)``, normalised to mean 1 within each panel,
   clipped at ``CLIP`` times the panel mean and renormalised. The official
   aggregation then runs unchanged on the weighted panel means (windows ->
   panel -> family -> mean), so every panel keeps its official weight.
3. **Weighted score** (``score``): weighted official aggregation of a
   per-window IoU table, and for a pair of tables the weighted Δ with a paired
   bootstrap SE (windows resampled within panel, weights fixed);
   ``boot_refit`` adds the weight-model uncertainty (target and CV windows
   resampled, classifier refitted per replicate). ``lb_predictive`` gives the
   spread of a 40-window (5 per panel) leaderboard draw from the weighted CV.
4. **Footprint check** (``footprint_check``, truth-free): does a change edit
   the target windows (share of windows edited, cells added / removed per
   window) the way it edits LB-sized weighted draws of CV windows? A target
   footprint at or beyond the 95th percentile means the CV is extrapolating.

    from trafficflow.t2 import shift_cv as sc
    w = sc.weights("queue_ongoing", "validation")        # pd.Series by gw, mean 1 within panel
    sc.score(df_new, "queue_ongoing", "validation", ref=df_old)   # {col: {est, se, p_le0}}
    sc.footprint_check("queue_ongoing", "G3", "v5", "validation")

    python -m trafficflow.t2.shift_cv features   # window features -> OUT/winfeat_<cond>.parquet
    python -m trafficflow.t2.shift_cv weights    # classifier / ESS / balance diagnostics
    python -m trafficflow.t2.shift_cv repro      # LB reproduction and level tables
    python -m trafficflow.t2.shift_cv mitigate   # ongoing stage-2 variants under the weighted CV
    python -m trafficflow.t2.shift_cv build [VARIANT [TAG]]   # lgb_v10_<TAG>.csv (ongoing rows replaced)

Run with ``T2_WORK=/home/user/work/t2 T2_FEAT=/home/user/work/t2/feat_v3``
(the defaults set here; the ongoing evaluator reads core.WORK / core.FEAT);
the onset side uses the t2h paths explicitly. Outputs go to ``T2_SHIFT_OUT``
(default /home/user/work/t2shift) except the candidate files.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("T2_WORK", "/home/user/work/t2")
os.environ.setdefault("T2_FEAT", "/home/user/work/t2/feat_v3")
from . import core  # noqa: E402
from .core import FAMILY, K, PANELS8, official_windows, statics  # noqa: E402
from .models import eiou_topm

WORK_OG = Path("/home/user/work/t2")            # ongoing: original windows, feat_v3 tables, old truth
FEAT_OG = WORK_OG / "feat_v3"
WORK_ON = Path("/home/user/work/t2h")           # onset: windows re-drawn on the hybrid truth
FEAT_ON = WORK_ON / "feat"
OUT = Path(os.environ.get("T2_SHIFT_OUT", "/home/user/work/t2shift"))
CONDS = ("queue_onset", "queue_ongoing")
TARGETS = ("validation", "private")
SPLITS = ("validation", "private")
V5_OG = {"oof_queue_ongoing_rob_og_v3_p2_w.parquet": 0.35, "oof_queue_ongoing_rob_og_v3_noloc_p2_w.parquet": 0.35,
         "oof_queue_ongoing_p2w.parquet": 0.15, "oof_queue_ongoing_rob_noloc_p2_w.parquet": 0.15}
V6_ON = ["oof_queue_onset_rob_on_v3_p1_op_all_new.parquet", "oof_queue_onset_rob_on_v3_p1_op_s1_all_new.parquet",
         "oof_queue_onset_rob_on_v3_p1_op_s2_all_new.parquet", "oof_queue_onset_rob_on_v2_p1_op_all_new.parquet"]
FAMS = sorted(set(FAMILY.values()))
PAN_FAM = np.array([FAMS.index(FAMILY[p]) for p in PANELS8])


# ----------------------------------------------------------------------------
# window features (data <= T only)
def _runs(q: np.ndarray) -> int:
    """Number of runs of True in a 1-d bool array (queue blocks / fragments)."""
    q = np.asarray(q, bool)
    return int((q & ~np.r_[False, q[:-1]]).sum())


def _stage1_window(p: np.ndarray, k: np.ndarray | None = None) -> dict:
    """Confidence summaries of one window's stage-1 probabilities (candidate cells)."""
    sel, r = eiou_topm(p)
    out = {"sur": r, "m": len(sel), "pmax": float(p.max()), "psum": float(p.sum())}
    if k is not None:                                          # ongoing: predicted growth T+5 -> T+30
        ks = k[sel]
        out["m1"] = int((ks == 1).sum()); out["m6"] = int((ks == K).sum())
    return out


def _tables(feat: Path, panel: str, split: str | None, cols: list[str], k: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows at step ``k`` of the CV table (split None) or of the official table of ``split``."""
    f = feat / (f"feat_{panel}.parquet" if split is None else f"feat_{panel}_{split}.parquet")
    m = feat / (f"meta_{panel}.parquet" if split is None else f"meta_{panel}_{split}.parquet")
    return pd.read_parquet(f, columns=["w", "link"] + cols, filters=[("k", "==", k)]), pd.read_parquet(m)


OG_COLS = ["r_last", "r_now", "d_r15", "d_r30", "d_r55", "c_q_now", "c_qT", "c_q_trend", "c_q_trend60", "c_rmin",
           "tod", "dow", "rT_ok", "pq_T", "rq7_T", "f_last"]
ON_COLS = ["r_last", "r_now", "c_rmin", "tod", "dow", "rT_ok", "pq_k", "rq7_k", "f_last", "ph_c_qobs", "ph_fcap_ext",
           "ph_qobs", "early_qfrac"]


def _og_window(g: pd.DataFrame, L: int, length: np.ndarray) -> dict:
    """Ongoing window features from its k == 1 rows (candidate cells near queue activity)."""
    link = g.link.to_numpy().astype(int)
    qT = np.zeros(L, bool); qT[link[g.r_now.to_numpy() <= 1.0]] = True
    ql = np.zeros(L, bool); ql[link[g.r_last.to_numpy() <= 1.0]] = True
    q30 = np.zeros(L, bool); q30[link[(g.r_last - g.d_r30).to_numpy() <= 1.0]] = True
    q60 = np.zeros(L, bool); q60[link[(g.r_last - g.d_r55).to_numpy() <= 1.0]] = True
    n_last = float(g.c_q_now.iat[0] * L); n_T = float(g.c_qT.iat[0] * L)
    n15 = n_last - float(g.c_q_trend.iat[0] * L); n60b = n_last - float(g.c_q_trend60.iat[0] * L)
    pq = g.pq_T.to_numpy()[g.r_last.to_numpy() <= 1.0]
    return dict(nq_T=n_T, nq_last=n_last, nq_km=float(length[qT].sum()), nblk=_runs(qT),
                g15=(n_last - n15) / max(n_last, n15, 1.0), g30=(ql.sum() - q30.sum()) / max(ql.sum(), q30.sum(), 1),
                g60=(n_last - n60b) / max(n_last, n60b, 1.0), gT=(n_T - n_last) / max(n_T, n_last, 1.0),
                rmin=float(g.c_rmin.iat[0]), tod=float(g.tod.iat[0]), dow=float(g.dow.iat[0]),
                visT=float(g.rT_ok.mean()), rec=float(pq.mean()) if len(pq) else np.nan,
                rq7=float(g.rq7_T.to_numpy()[g.r_last.to_numpy() <= 1.0].mean()) if len(pq) else np.nan,
                fq=float(np.nanmean(g.f_last.to_numpy()[g.r_last.to_numpy() <= 1.0])) if len(pq) else np.nan)


def _on_window(g: pd.DataFrame, p: np.ndarray) -> dict:
    """Onset window features from its T+30 rows (every link) and the stage-1 probabilities p (same order)."""
    ps = max(p.sum(), 1e-9)
    idx = np.flatnonzero(p >= 0.05)
    nsite = 0 if not len(idx) else int((np.diff(idx) > 2).sum() + 1)
    return dict(rmin=float(np.nanmin(g.r_now.to_numpy())), nslow=int((g.r_now.to_numpy() <= 1.3).sum()),
                tod=float(g.tod.iat[0]), dow=float(g.dow.iat[0]), visT=float(g.rT_ok.mean()),
                orec=float((p * g.pq_k.to_numpy()).sum() / ps), orq7=float((p * np.nan_to_num(g.rq7_k.to_numpy())).sum() / ps),
                fmax=float(np.nanmax(g.f_last.to_numpy())), fext=float(np.nanmax(g.ph_fcap_ext.to_numpy())),
                qobs=float(g.ph_c_qobs.iat[0]), eq=float(np.nanmean(g.early_qfrac.to_numpy())), nsite=nsite)


def _cv_stage1(cond: str) -> pd.DataFrame:
    """Stage-1 OOF rows of the reference model for the CV evaluation windows: gw, k, link, p."""
    if cond == "queue_ongoing":
        B = None
        for f, w in V5_OG.items():
            x = pd.read_parquet(WORK_OG / f, columns=["gw", "k", "link", "p"])
            if B is None:
                B = x[["gw", "k", "link"]].copy(); B["p"] = 0.0
            B["p"] += w * x.p.to_numpy(np.float64)
        B["p"] /= sum(V5_OG.values())
        return B
    B = None
    for f in V6_ON:
        x = pd.read_parquet(WORK_ON / f, columns=["gw", "k", "link", "p"])
        if B is None:
            B = x[["gw", "k", "link"]].copy(); B["p"] = 0.0
        B["p"] += x.p.to_numpy(np.float64) / len(V6_ON)
    return B


def _test_stage1(cond: str) -> pd.DataFrame:
    if cond == "queue_ongoing":
        P = pd.read_parquet(WORK_OG / "probs_lgb_v5.parquet")
        return P[P.condition == "queue_ongoing"][["window_id", "panel", "k", "link", "p", "split"]]
    return pd.read_parquet(WORK_ON / "probs_v6_onset.parquet")


def build_features(cond: str) -> pd.DataFrame:
    """One row per window: CV windows (sim + off, index gw) and the official validation / private windows
    (index window_id). Columns: panel, src (sim / off / validation / private), fold (CV windows), features."""
    feat = FEAT_OG if cond == "queue_ongoing" else FEAT_ON
    work = WORK_OG if cond == "queue_ongoing" else WORK_ON
    cols = OG_COLS if cond == "queue_ongoing" else ON_COLS
    kk = 1.0 if cond == "queue_ongoing" else float(K)
    S1 = _cv_stage1(cond)
    T1 = _test_stage1(cond)
    rows = []
    for p in PANELS8:
        pc = PANELS8.index(p)
        st = statics(p); L = st["L"]
        z = np.load(work / f"ds_{p}.npz", allow_pickle=True)
        cov = z["w_cov"]
        for split in (None,) + SPLITS:
            X, M = _tables(feat, p, split, cols, kk)
            M = M[M.condition == cond]
            X = X[X.w.isin(set(M.w))]
            if split is None:
                M = M[M.src.isin(["sim", "off"])]
                keys = {int(w): pc * 100000 + int(w) for w in M.w}
                src = M.set_index("w").src; fold = M.set_index("w").fold
                s1 = S1[S1.gw // 100000 == pc]
                s1g = {g: d for g, d in s1.groupby("gw")}
                ow = None
            else:
                keys = dict(zip(M.w.astype(int), M.window_id))
                ow = official_windows(p, split).set_index("window_id")
                s1 = T1[(T1.panel == p) & (T1.split == split)]
                s1g = {g: d for g, d in s1.groupby("window_id")}
            for w, g in X.groupby("w", sort=False):
                w = int(w)
                if w not in keys:
                    continue
                key = keys[w]
                q = s1g[key].sort_values(["k", "link"])
                if cond == "queue_ongoing":
                    r = _og_window(g, L, st["length"])
                    r.update(_stage1_window(q.p.to_numpy(), q.k.to_numpy().astype(int)))
                else:
                    g = g.sort_values("link")
                    pv = np.zeros(L); pv[q.link.to_numpy().astype(int)] = q.p.to_numpy()
                    r = _on_window(g, pv[g.link.to_numpy().astype(int)])
                    r.update(_stage1_window(q.p.to_numpy()))
                r["cov"] = float(cov[w]) if split is None else float(ow.loc[key, "history_coverage"])
                r.update(key=key, panel=p, src=(src[w] if split is None else split),
                         fold=(int(fold[w]) if split is None else -1))
                rows.append(r)
        del z
    D = pd.DataFrame(rows)
    return D


@lru_cache(None)
def window_features(cond: str) -> pd.DataFrame:
    """Cached ``build_features`` (OUT/winfeat_<cond>.parquet)."""
    f = OUT / f"winfeat_{cond}.parquet"
    if not f.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        D = build_features(cond)
        D["key"] = D.key.astype(str)
        D.to_parquet(f)
    return pd.read_parquet(f)


# ----------------------------------------------------------------------------
# importance weights
FEATS = {
    # pre-specified from the section 11/16 shift diagnosis: confidence, recurrence, recent-days activity,
    # queue size / fragments / growth, speed level, observation share, weekend, hour
    "queue_ongoing": ["sur", "rec", "rq7", "lnq", "nblk", "g30", "g60", "gT", "rmin", "cov", "visT", "wkend",
                      "tod_s", "tod_c"],
    "queue_onset": ["sur", "orec", "orq7", "lpsum", "nsite", "rmin", "fext", "cov", "visT", "wkend", "tod_s", "tod_c"],
}
CLIP = 10.0            # weights are capped at CLIP x the panel mean (then renormalised)
LAMS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
NFOLD, NREP = 5, 5


def design(D: pd.DataFrame, cond: str, feats=None) -> pd.DataFrame:
    """Window design matrix (raw units): derived columns, NaN -> median of the CV sim windows."""
    X = pd.DataFrame(index=D.index)
    X["wkend"] = (D.dow >= 5).astype(float)
    X["tod_s"] = np.sin(2 * np.pi * D.tod / 24); X["tod_c"] = np.cos(2 * np.pi * D.tod / 24)
    X["sur"] = np.log(D.sur / (1 - D.sur).clip(lower=1e-3))          # logit of the expected-IoU surrogate
    if cond == "queue_ongoing":
        X["lnq"] = np.log1p(D.nq_T); X["pgrow"] = np.log((D.m6 + 1) / (D.m1 + 1))
        for c in ("rec", "rq7", "nblk", "g15", "g30", "g60", "gT", "rmin", "cov", "visT", "fq"):
            X[c] = D[c]
    else:
        X["lpsum"] = np.log(D.psum.clip(lower=1e-3)); X["lm"] = np.log1p(D.m); X["lpmax"] = np.log(D.pmax / (1 - D.pmax).clip(lower=1e-4))
        X["nslow"] = np.log1p(D.nslow)
        for c in ("orec", "orq7", "nsite", "rmin", "fext", "fmax", "cov", "visT"):
            X[c] = D[c]
    X = X[list(feats or FEATS[cond])]
    med = X[(D.src == "sim").to_numpy()].median()
    return X.fillna(med)


def _nll(theta, Z, P, y, lam):
    """Penalised logistic NLL with unpenalised panel intercepts: theta = [alpha (npanel), beta]."""
    npn = P.shape[1]
    a, b = theta[:npn], theta[npn:]
    eta = P @ a + Z @ b
    ll = y * eta - np.logaddexp(0.0, eta)
    pr = 1.0 / (1.0 + np.exp(-eta))
    r = pr - y
    f = -ll.sum() + 0.5 * lam * (b @ b)
    g = np.r_[P.T @ r, Z.T @ r + lam * b]
    return f, g


def fit_logit(Z: np.ndarray, P: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    from scipy.optimize import minimize
    theta0 = np.zeros(P.shape[1] + Z.shape[1])
    base = np.log(np.maximum(P.T @ y, 0.5) / np.maximum(P.T @ (1 - y), 1.0))
    theta0[:P.shape[1]] = base
    r = minimize(_nll, theta0, args=(Z, P, y, lam), jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    return r.x


def _folds(src: np.ndarray, panel: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Fold ids stratified by class and panel."""
    rng = np.random.default_rng(seed)
    f = np.empty(len(y), int)
    for key in np.unique(np.char.add(panel.astype(str), y.astype(int).astype(str))):
        idx = np.flatnonzero(np.char.add(panel.astype(str), y.astype(int).astype(str)) == key)
        idx = rng.permutation(idx)
        f[idx] = np.arange(len(idx)) % NFOLD
    return f


def _prep(cond: str, target: str, feats=None, extra: pd.DataFrame | None = None):
    D = window_features(cond)
    D = D[D.src.isin(["sim", target])].reset_index(drop=True)
    X = design(D, cond, feats)
    if extra is not None:                        # e.g. the change footprint (``footprint``), indexed by key
        E_ = extra.reindex(D.key.astype(str).to_numpy())
        assert E_.notna().all().all(), "extra covariates missing for some windows"
        for c in E_.columns:
            X[c] = E_[c].to_numpy(np.float64)
    sim = (D.src == "sim").to_numpy()
    mu, sd = X[sim].mean(), X[sim].std().replace(0, 1.0)
    Z = ((X - mu) / sd).to_numpy(np.float64)
    P = (D.panel.to_numpy()[:, None] == np.array(PANELS8)[None, :]).astype(np.float64)
    y = (~sim).astype(np.float64)
    return D, X, Z, P, y


def choose_lam(Z, P, y, panel, lams=LAMS, seed=0) -> tuple[float, dict]:
    """lambda maximising the cross-validated held-out log-likelihood (5 folds, stratified)."""
    f = _folds(np.zeros(len(y)), panel, y, seed)
    res = {}
    for lam in lams:
        ll = 0.0
        for k in range(NFOLD):
            tr, te = f != k, f == k
            th = fit_logit(Z[tr], P[tr], y[tr], lam)
            eta = P[te] @ th[:P.shape[1]] + Z[te] @ th[P.shape[1]:]
            ll += float((y[te] * eta - np.logaddexp(0.0, eta)).sum())
        res[lam] = ll
    return max(res, key=res.get), res


def _auc(s: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _within_panel_auc(s, y, panel) -> float:
    """AUC of target vs sim windows compared within the same panel (panel base rates removed)."""
    num = den = 0.0
    for p in np.unique(panel):
        m = panel == p
        if y[m].sum() and (1 - y[m]).sum():
            n = y[m].sum() * (1 - y[m]).sum()
            num += _auc(s[m], y[m]) * n; den += n
    return num / den


def clip_normalise(w: np.ndarray, panel: np.ndarray, clip: float = CLIP) -> tuple[np.ndarray, float]:
    """Normalise to mean 1 within each panel, cap at ``clip`` x the panel mean, renormalise (iterated)."""
    w = w.astype(np.float64).copy()
    clipped = np.zeros(len(w), bool)
    for p in np.unique(panel):
        m = panel == p
        v = w[m] / w[m].mean()
        for _ in range(50):
            c = v > clip
            if not c.any():
                break
            clipped[np.flatnonzero(m)[c]] = True
            v = np.minimum(v, clip); v = v / v.mean()
            if v.max() <= clip * (1 + 1e-9):
                break
        w[m] = v
    return w, float(clipped.mean())


def fit_weights(cond: str, target: str, feats=None, lam: float | None = None, method: str = "logit",
                clip: float = CLIP, seed: int = 0, extra: pd.DataFrame | None = None) -> dict:
    """Cross-fitted importance weights of the CV sim windows toward ``target`` (validation / private / off).
    Returns dict(w: pd.Series by gw, lam, auc (cross-fitted, within panel), coef (full-data fit, per SD),
    ess, clipped share, eta (cross-fitted log odds for every row), D)."""
    D, X, Z, P, y = _prep(cond, target, feats, extra)
    panel = D.panel.to_numpy()
    sim = y == 0
    if method == "logit":
        if lam is None:
            lam, _ = choose_lam(Z, P, y, panel, seed=seed)
        eta = np.zeros(len(y))
        for r in range(NREP):
            f = _folds(np.zeros(len(y)), panel, y, seed + 100 + r)
            for k in range(NFOLD):
                tr, te = f != k, f == k
                th = fit_logit(Z[tr], P[tr], y[tr], lam)
                eta[te] += Z[te] @ th[P.shape[1]:] / NREP            # panel intercept dropped (cancels within panel)
        th = fit_logit(Z, P, y, lam)
        coef = pd.Series(th[P.shape[1]:], index=X.columns)
    elif method == "gbm":
        import lightgbm as lgb
        prm = dict(objective="binary", learning_rate=0.05, num_leaves=4, min_data_in_leaf=20, feature_fraction=0.8,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=2, verbose=-1)
        A = np.c_[Z, P.argmax(1)]
        eta = np.zeros(len(y))
        for r in range(NREP):
            f = _folds(np.zeros(len(y)), panel, y, seed + 100 + r)
            for k in range(NFOLD):
                tr, te = f != k, f == k
                m = lgb.train({**prm, "seed": r}, lgb.Dataset(A[tr], y[tr], categorical_feature=[A.shape[1] - 1]), 150)
                pr = np.clip(m.predict(A[te]), 1e-6, 1 - 1e-6)
                eta[te] += np.log(pr / (1 - pr)) / NREP
        coef = None; lam = None
    else:
        raise ValueError(method)
    w_raw = np.exp(eta[sim] - eta[sim].max())
    w, share = clip_normalise(w_raw, panel[sim], clip)
    ws = pd.Series(w, index=D.key[sim].astype(np.int64).to_numpy(), name="w")
    return dict(w=ws, lam=lam, auc=_within_panel_auc(eta, y, panel), coef=coef, clipped=share,
                ess=ess(ws, D.panel[sim].to_numpy()), eta=eta, D=D, X=X)


def ess(w: pd.Series, panel: np.ndarray) -> dict:
    """Kish effective sample size per panel and of the official aggregation (a_i = w_i / (8 * sum_panel w))."""
    w = np.asarray(w, np.float64)
    per = {}
    a = np.empty(len(w))
    for p in PANELS8:
        m = panel == p
        per[p] = float(w[m].sum() ** 2 / (w[m] ** 2).sum())
        a[m] = w[m] / w[m].sum() / len(PANELS8)
    n = {p: int((panel == p).sum()) for p in PANELS8}
    plain = 1.0 / sum(1.0 / (len(PANELS8) ** 2 * n[p]) for p in PANELS8)
    return {"agg": float(1.0 / (a ** 2).sum()), "agg_plain": float(plain), "total": float(sum(per.values())),
            "n": int(len(w)), "per_panel": per}


@lru_cache(None)
def _cached(cond: str, target: str) -> dict:
    return fit_weights(cond, target)


def weights(cond: str, target: str) -> pd.Series:
    """Importance weights (mean 1 within each panel) of the CV sim windows of ``cond`` toward ``target``
    ('validation', 'private', 'off'; 'plain' = all ones), indexed by gw. Main method: cross-fitted
    L2 logistic regression on FEATS[cond] with panel fixed effects, CV-chosen lambda, clip CLIP."""
    if target == "plain":
        D = window_features(cond)
        g = D.key[D.src == "sim"].astype(np.int64).to_numpy()
        return pd.Series(1.0, index=g, name="w")
    return _cached(cond, target)["w"]


# ----------------------------------------------------------------------------
# weighted scores
def _agg_panels(pm: np.ndarray, panels: list[str]) -> np.ndarray:
    """Official aggregation of panel scores [..., n_panels] -> family means -> mean."""
    fam = np.array([FAMS.index(FAMILY[p]) for p in panels])
    return np.stack([pm[..., fam == f].mean(-1) for f in np.unique(fam)], -1).mean(-1)


def _panel_arrays(df: pd.DataFrame, w: pd.Series, col: str):
    s = df[df.src == "sim"]
    ww = w.reindex(s.gw.astype(np.int64).to_numpy()).to_numpy()
    assert np.isfinite(ww).all(), "windows without a weight"
    out = []
    for p in PANELS8:
        m = (s.panel == p).to_numpy()
        if m.any():
            out.append((p, s[col].to_numpy(np.float64)[m], ww[m]))
    return out


def weighted_agg(df: pd.DataFrame, w: pd.Series, col: str) -> float:
    A = _panel_arrays(df, w, col)
    return float(_agg_panels(np.array([(d * v).sum() / v.sum() for _, d, v in A]), [p for p, _, _ in A]))


def score(df: pd.DataFrame, cond: str, target: str, ref: pd.DataFrame | None = None, cols=None,
          n_boot: int = 2000, seed: int = 0, w: pd.Series | None = None) -> dict:
    """Weighted official aggregation of the per-window columns ``cols`` (default iou_*) of a window_iou frame
    (gw, panel, src, ...; only the sim windows are used). With ``ref`` (same windows): Δ = df - ref, weighted,
    with a paired bootstrap SE (windows resampled within panel, weights fixed) and P(bootstrap Δ <= 0)."""
    w = weights(cond, target) if w is None else w
    cols = cols or [c for c in df.columns if c.startswith("iou_")]
    x = df.set_index("gw")
    if ref is not None:
        r = ref.set_index("gw").reindex(x.index)
        x = x.assign(**{c: x[c] - r[c] for c in cols})
    x = x.reset_index()
    rng = np.random.default_rng(seed)
    out = {}
    for c in cols:
        A = _panel_arrays(x, w, c)
        est = float(_agg_panels(np.array([(d * v).sum() / v.sum() for _, d, v in A]), [p for p, _, _ in A]))
        B = np.empty((n_boot, len(A)))
        for j, (_, d, v) in enumerate(A):
            ii = rng.integers(0, len(d), (n_boot, len(d)))
            B[:, j] = (d[ii] * v[ii]).sum(1) / v[ii].sum(1)
        boot = _agg_panels(B, [p for p, _, _ in A])
        out[c] = {"est": est, "se": float(boot.std()), "p_le0": float((boot <= 0).mean())}
    return out


def lb_predictive(d: pd.DataFrame, w: pd.Series, col: str, per_panel: int = 5, n_sim: int = 20000,
                  seed: int = 0) -> dict:
    """Distribution of the official-aggregation mean of ``col`` over a leaderboard-sized draw (``per_panel``
    windows per panel, drawn with probability proportional to the weights) - the LB noise of a paired Δ if the
    weighted CV population were the target population."""
    rng = np.random.default_rng(seed)
    A = _panel_arrays(d, w, col)
    S = np.empty((n_sim, len(A)))
    for j, (_, x, v) in enumerate(A):
        ii = rng.choice(len(x), (n_sim, per_panel), p=v / v.sum())
        S[:, j] = x[ii].mean(1)
    s = _agg_panels(S, [p for p, _, _ in A])
    return {"mean": float(s.mean()), "sd": float(s.std()), "q05": float(np.quantile(s, 0.05)),
            "q95": float(np.quantile(s, 0.95)), "draws": s}


# ----------------------------------------------------------------------------
# leaderboard reproduction (public = March = validation; task-level Δ = total Δ / 0.15)
ON_OLD = ["oof_queue_onset_rob_on_v3_p1_op_all_old.parquet", "oof_queue_onset_rob_on_v3_p1_op_s1_all_old.parquet",
          "oof_queue_onset_rob_on_v3_p1_op_s2_all_old.parquet", "oof_queue_onset_rob_on_v2_p1_op_all_old.parquet"]
# (name, condition, new, ref, LB Δ of the condition score on March, how the LB Δ is known)
CHANGES = [
    ("onset v4 -> v5 (physics features, 4-model mix)", "queue_onset", "v5", "v4", +0.0122,
     "P3/P4/D1 probes: v5 onset = 0.7281 - 0.0302 = 0.6979 vs A 0.6857"),
    ("onset v5 -> v6 (hybrid labels)", "queue_onset", "v6", "v5", +0.0302, "D1 - C1 = +0.00453 / 0.15"),
    ("F1: v6 onset, logit bias +0.5", "queue_onset", "F1", "v6", -0.0157, "-0.00235 / 0.15"),
    ("F2: v6 onset, site2 decoder", "queue_onset", "F2", "v6", -0.0115, "-0.00173 / 0.15"),
    ("G2: onset v8 seeds9_stack03", "queue_onset", "G2", "v6", +0.0040, "+0.00060 / 0.15"),
    ("ongoing v3 -> v4 (robust blend)", "queue_ongoing", "v4", "v3", +0.024,
     "approx.: B1 - A = +0.0053 minus the local Task-1 gating estimate +0.0016"),
    ("ongoing v4 -> v5 (physics / LWR blend)", "queue_ongoing", "v5", "v4", +0.0040,
     "C1 - B1 = +0.00243 -> dS_queue +0.0081, minus onset +0.0122"),
    ("G3: ongoing v9 stage-2 stack (w 0.8, dyn)", "queue_ongoing", "G3", "v5", -0.0233, "-0.00350 / 0.15"),
]
LB_LEVEL = {("queue_onset", "v4"): 0.6857, ("queue_onset", "v5"): 0.6979, ("queue_onset", "v6"): 0.7281,
            ("queue_ongoing", "v3"): 0.8117, ("queue_ongoing", "v5"): 0.840, ("queue_ongoing", "G3"): 0.817}
PRIMARY = {"queue_onset": "iou_hybrid", "queue_ongoing": "iou_old"}


def shift_logit(p: np.ndarray, b: float) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + b)))


def onset_window_iou_sets(E, sets: list[np.ndarray]) -> pd.DataFrame:
    """OnsetEval.window_iou for given predicted link sets (one per evaluation window)."""
    out = {t: np.empty(len(E.win)) for t in E.truths}
    for i, (w, s) in enumerate(zip(E.win, sets)):
        for t in E.truths:
            y = w[f"y_{t}"]
            inter = int(y[s].sum()); union = len(s) + int(y.sum()) - inter + w[f"e_{t}"]
            out[t][i] = 1.0 if union == 0 else inter / union
    return E.info.assign(**{f"iou_{t}": out[t] for t in E.truths}, m=[len(s) for s in sets])


def onset_decoder_iou(E, p: np.ndarray, dec) -> pd.DataFrame:
    sets = []
    for w in E.win:
        L = len(w["y_hybrid"])
        pv = np.zeros(L); pv[w["link"]] = p[w["idx"]]
        sets.append(np.asarray(dec(pv), int))
    return onset_window_iou_sets(E, sets)


@lru_cache(None)
def onset_eval():
    from .onset_eval import OnsetEval
    return OnsetEval(work=WORK_ON)


@lru_cache(None)
def ongoing_eval():
    assert Path(core.FEAT) == FEAT_OG and Path(core.WORK) == WORK_OG, \
        "run with T2_WORK=/home/user/work/t2 T2_FEAT=/home/user/work/t2/feat_v3 (the ongoing evaluator reads core.FEAT)"
    from .og_stack import OngoingEval
    return OngoingEval(work=WORK_OG)


@lru_cache(None)
def onset_probs() -> dict:
    """OOF probability vectors (aligned to OnsetEval rows) of the onset versions."""
    E = onset_eval()
    p6 = E.load_mean(V6_ON)
    X = pd.read_parquet(WORK_ON / "stack8_oof_seeds9s012.parquet"); X["k"] = K
    p1, p2 = E.align(X, "p1"), E.align(X, "p2")
    return {"v4": E.load(ON_OLD[3]), "v5": E.load_mean(ON_OLD), "v6": p6, "G2": 0.3 * p2 + 0.7 * p1,
            "seeds9": p1, "seeds9_p2": p2}


@lru_cache(None)
def ongoing_probs() -> dict:
    E = ongoing_eval()
    v3 = E.load("oof_queue_ongoing_p2w.parquet")
    v5 = sum(w * E.load(f) for f, w in V5_OG.items()) / sum(V5_OG.values())
    X = pd.read_parquet(WORK_OG / "ogstack_oof_dyn_s012.parquet")
    p1, p2 = E.align(X, "p1"), E.align(X, "p2")
    assert np.allclose(p1, v5, atol=1e-6)
    return {"v3": v3, "v4": 0.5 * v3 + 0.5 * E.load("oof_queue_ongoing_rob_noloc_p2_w.parquet"), "v5": v5,
            "G3": 0.8 * p2 + 0.2 * p1, "p2": p2}


@lru_cache(None)
def tables(cond: str) -> dict:
    """Per-window IoU frames (window_iou) of every version / change of ``cond``."""
    if cond == "queue_onset":
        from .onset_decode import DECODERS
        E = onset_eval(); P = onset_probs()
        T = {k: E.window_iou(P[k]) for k in ("v4", "v5", "v6", "G2")}
        T["F1"] = E.window_iou(shift_logit(P["v6"], 0.5))
        T["F2"] = onset_decoder_iou(E, P["v6"], DECODERS["site2_lo.05_r.5"])
        return T
    E = ongoing_eval(); P = ongoing_probs()
    return {k: E.window_iou(P[k]) for k in ("v3", "v4", "v5", "G3")}


def boot_refit(df: pd.DataFrame, ref: pd.DataFrame, col: str, cond: str, target: str, n_boot: int = 200,
               seed: int = 0, feats=None, lam: float | None = None, clip: float = CLIP) -> dict:
    """Weighted Δ = df - ref with an SE that includes the weight-model uncertainty: every replicate resamples
    the target windows and the CV sim windows within panel, refits the logistic model (lambda fixed at the
    main fit's value) and re-scores the resampled CV windows."""
    rng = np.random.default_rng(seed)
    D, X, Z, P, y = _prep(cond, target, feats)
    if lam is None:
        lam = _cached(cond, target)["lam"] if feats is None else choose_lam(Z, P, y, D.panel.to_numpy())[0]
    panel = D.panel.to_numpy()
    x = df.set_index("gw"); r = ref.set_index("gw")
    key = D.key.to_numpy()
    sim = y == 0
    dd = np.full(len(D), np.nan)
    dd[sim] = (x[col] - r[col]).reindex(key[sim].astype(np.int64)).to_numpy()
    groups = [np.flatnonzero((panel == p) & (y == c)) for p in PANELS8 for c in (0, 1)]
    est = []
    for _ in range(n_boot):
        ii = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups if len(g)])
        th = fit_logit(Z[ii], P[ii], y[ii], lam)
        s = ii[y[ii] == 0]
        w = np.exp(Z[s] @ th[P.shape[1]:])
        w, _ = clip_normalise(w, panel[s], clip)
        pm = np.array([(w[panel[s] == p] * dd[s][panel[s] == p]).sum() / w[panel[s] == p].sum() for p in PANELS8])
        est.append(float(_agg_panels(pm, PANELS8)))
    est = np.array(est)
    return {"se": float(est.std()), "p_le0": float((est <= 0).mean()), "mean": float(est.mean())}


# ----------------------------------------------------------------------------
# change footprint (data <= T: both versions' predicted sets), for change-aware weights
def _test_probs(cond: str, version: str) -> pd.DataFrame:
    """Validation / private probabilities of a version: window_id, panel, k, link, p, split."""
    if cond == "queue_onset":
        f = {"v4": WORK_OG / "probs_lgb_v4_robust.parquet", "v5": WORK_OG / "probs_lgb_v5.parquet",
             "v6": WORK_ON / "probs_v6_onset.parquet", "G2": WORK_ON / "probs_v8_seeds9_stack03_onset.parquet"}
        P = pd.read_parquet(f["v6" if version in ("F1", "F2") else version])
        if "condition" in P:
            P = P[P.condition == "queue_onset"]
        if version == "F1":
            P = P.assign(p=shift_logit(P.p.to_numpy(), 0.5))
        return P
    f = {"v3": WORK_OG / "probs_lgb_v3.parquet", "v4": WORK_OG / "probs_lgb_v4_robust.parquet",
         "v5": WORK_OG / "probs_lgb_v5.parquet", "G3": WORK_OG / "probs_v9_ogstack08_ongoing.parquet"}
    P = pd.read_parquet(f[version])
    return P[P.condition == "queue_ongoing"] if "condition" in P else P


def test_sets(cond: str, version: str) -> dict:
    """window_id -> set of predicted cells (k, link) on the validation / private windows."""
    from .onset_decode import DECODERS
    P = _test_probs(cond, version)
    out = {}
    for wid, g in P.groupby("window_id"):
        g = g.sort_values(["k", "link"])
        kk = g.k.to_numpy().astype(int); ll = g.link.to_numpy().astype(int); p = g.p.to_numpy()
        if version == "F2":
            L = statics(g.panel.iat[0])["L"]
            pv = np.zeros(L); pv[ll] = p
            sel = np.asarray(DECODERS["site2_lo.05_r.5"](pv), int)
            out[wid] = {(K, int(l)) for l in sel}
        else:
            sel = eiou_topm(p)[0]
            out[wid] = set(zip(kk[sel].tolist(), ll[sel].tolist()))
    return out


def cv_sets(cond: str, version: str) -> dict:
    """gw -> set of predicted cells (k, link) on the CV evaluation windows."""
    from .onset_decode import DECODERS
    if cond == "queue_onset":
        E = onset_eval(); P = onset_probs()
        p = shift_logit(P["v6"], 0.5) if version == "F1" else P["v6" if version == "F2" else version]
        out = {}
        for w in E.win:
            if version == "F2":
                pv = np.zeros(len(w["y_hybrid"])); pv[w["link"]] = p[w["idx"]]
                sel = np.asarray(DECODERS["site2_lo.05_r.5"](pv), int)
            else:
                sel = w["link"][eiou_topm(p[w["idx"]])[0]]
            out[w["gw"]] = {(K, int(l)) for l in sel}
        return out
    E = ongoing_eval(); p = ongoing_probs()[version]
    return {w["gw"]: set(zip((w["kk"][s] + 1).tolist(), w["ll"][s].tolist()))
            for w, s in zip(E.win, E.decode(p))}


@lru_cache(None)
def footprint(cond: str, new: str, ref: str) -> pd.DataFrame:
    """Per window (key = gw for CV windows, window_id for validation / private): log1p of the number of
    cells the change edits, and whether it edits the window at all. Both versions are functions of data <= T,
    so these are legitimate covariates for change-aware weights."""
    return edits(cond, new, ref)[["fp_lchg", "fp_any"]]


@lru_cache(None)
def edits(cond: str, new: str, ref: str) -> pd.DataFrame:
    """Per window (CV key gw as str, or window_id): cells added / removed by the change and the reference
    set size, plus the footprint covariates."""
    rows = {}
    for sets_new, sets_ref in ((cv_sets(cond, new), cv_sets(cond, ref)), (test_sets(cond, new), test_sets(cond, ref))):
        for k in sets_new:
            a, b = sets_new[k], sets_ref[k]
            n = len(a ^ b)
            rows[str(k)] = (np.log1p(n), float(n > 0), len(a - b), len(b - a), len(b))
    return pd.DataFrame.from_dict(rows, orient="index", columns=["fp_lchg", "fp_any", "add", "rem", "nref"])


FP_FLAG = 0.95          # a target footprint at or beyond this percentile of the weighted CV draws is flagged


def footprint_check(cond: str, new: str | pd.DataFrame, ref: str | None, target: str, w: pd.Series | None = None,
                    n_sim: int = 20000, seed: int = 0) -> dict:
    """Truth-free transfer check: does the change edit the target windows the way it edits LB-sized draws
    (5 windows per panel, drawn in proportion to the target weights) of the CV windows? Returns, per
    statistic (share of windows edited, cells added / removed per window), the target value, the CV
    predictive mean / sd and the target's mid-rank percentile. A change whose target footprint is atypical
    (percentile >= FP_FLAG) is being applied outside the behaviour the CV scored. ``new`` is a version name
    (with ``ref``) or an edits table (key -> fp_any, add, rem) as returned by ``edits``."""
    rng = np.random.default_rng(seed)
    w = weights(cond, target) if w is None else w
    Ed = edits(cond, new, ref) if isinstance(new, str) else new
    D = window_features(cond)
    sim = D[D.src == "sim"]; tg = D[D.src == target]
    xs = Ed.reindex(sim.key.astype(str).to_numpy()); xt = Ed.reindex(tg.key.astype(str).to_numpy())
    ww = w.reindex(sim.key.astype(np.int64).to_numpy()).to_numpy()
    pan = sim.panel.to_numpy()
    out = {}
    for stat in ("fp_any", "add", "rem"):
        v = xs[stat].to_numpy(np.float64)
        S = np.zeros(n_sim)
        for p in PANELS8:
            m = np.flatnonzero(pan == p)
            ii = rng.choice(m, (n_sim, 5), p=ww[m] / ww[m].sum())
            S += v[ii].sum(1)
        S /= 5 * len(PANELS8)
        t = float(xt[stat].mean())
        out[stat] = {"target": t, "cv_mean": float(S.mean()), "cv_sd": float(S.std()), "pct": float((S < t).mean() + 0.5 * (S == t).mean())}
    return out


# ----------------------------------------------------------------------------
# ongoing stage-2 variants (section 16 stack, mitigations); identical rules on CV OOF and validation / private
def og_gate(D: pd.DataFrame, gate: str | None) -> np.ndarray:
    """Window-level gate (True = apply stage 2) from window features (data <= T)."""
    if gate is None:
        return np.ones(len(D), bool)
    if gate == "grow":                       # observed queue grew over the last 35 min
        return (D.g30 > 0).to_numpy()
    if gate == "grow_T":                     # ... and did not shrink between the last history slot and T
        return ((D.g30 > 0) & (D.gT >= 0)).to_numpy()
    if gate == "rec":                        # recurrent queue (train time-of-day probability >= 0.2)
        return (D.rec.fillna(0) >= 0.2).to_numpy()
    if gate == "rec05":                      # not a non-recurrent queue (>= 0.05; the section 16 watch slice)
        return (D.rec.fillna(0) >= 0.05).to_numpy()
    raise ValueError(gate)


def _decode_capped(p: np.ndarray, kk: np.ndarray, p_ref: np.ndarray, cap: str | None) -> np.ndarray:
    """Top-m expected-IoU decoding of ``p``. Set rules against the reference decoding (``p_ref``, v5):
    'window' keeps its set size (stage 2 only re-ranks), 'step' keeps its size per step, 'shrink' keeps
    only reference cells that the new decoding also selects (stage 2 may only remove cells), 'grow' the
    union (stage 2 may only add)."""
    if cap is None:
        return eiou_topm(p)[0]
    ref = eiou_topm(p_ref)[0]
    if cap in ("shrink", "grow"):            # stage 2 may only veto reference cells / only add cells
        new = eiou_topm(p)[0]
        return np.intersect1d(ref, new) if cap == "shrink" else np.union1d(ref, new)
    if cap == "window":
        return np.argsort(-p, kind="stable")[:len(ref)]
    if cap == "step":
        out = []
        nk = np.bincount(kk[ref], minlength=K + 1)
        for k in np.unique(kk):
            idx = np.flatnonzero(kk == k)
            out.append(idx[np.argsort(-p[idx], kind="stable")[:nk[k]]])
        return np.concatenate(out) if out else np.array([], int)
    raise ValueError(cap)


OG_VARIANTS = {   # name -> (stage-2 weight, gate, set rule)
    "G3 (w0.8)": (0.8, None, None), "w0.5": (0.5, None, None), "w0.3": (0.3, None, None),
    "w0.8 grow": (0.8, "grow", None), "w0.8 grow_T": (0.8, "grow_T", None), "w0.8 rec": (0.8, "rec", None),
    "w0.8 size-cap": (0.8, None, "window"), "w0.8 step-cap": (0.8, None, "step"),
    "w0.8 shrink-only": (0.8, None, "shrink"), "w0.8 add-only": (0.8, None, "grow"),
    "w0.8 shrink-only rec05": (0.8, "rec05", "shrink"), "w0.3 rec05": (0.3, "rec05", None),
}
CANDIDATE = "w0.3 rec05"


def _iou_frame(E, sets: list[np.ndarray]) -> pd.DataFrame:
    out = {t: np.empty(len(E.win)) for t in E.truths}
    for j, (win, s) in enumerate(zip(E.win, sets)):
        k, l = win["kk"][s], win["ll"][s]
        for t in E.truths:
            inter = int(win[f"y_{t}"][k, l].sum()); union = len(s) + win[f"n_{t}"] - inter
            out[t][j] = 1.0 if union == 0 else inter / union
    return E.info.assign(**{f"iou_{t}": out[t] for t in E.truths}, m=[len(s) for s in sets])


@lru_cache(None)
def og_variant_cv(name: str) -> tuple[pd.DataFrame, dict]:
    """CV window_iou frame (old / hybrid truth) and gw -> predicted cell set of an ongoing variant."""
    w, gate, cap = OG_VARIANTS[name]
    E = ongoing_eval(); P = ongoing_probs()
    D = window_features("queue_ongoing"); D = D[D.src.isin(["sim", "off"])]
    g = pd.Series(og_gate(D, gate), index=D.key.astype(np.int64).to_numpy())
    p1, p2 = P["v5"], P["p2"]
    sets = []
    for win in E.win:
        i = win["idx"]
        if bool(g.get(win["gw"], True)):
            sets.append(_decode_capped(w * p2[i] + (1 - w) * p1[i], win["kk"], p1[i], cap))
        else:
            sets.append(eiou_topm(p1[i])[0])
    cells = {win["gw"]: set(zip((win["kk"][s] + 1).tolist(), win["ll"][s].tolist())) for win, s in zip(E.win, sets)}
    return _iou_frame(E, sets), cells


@lru_cache(None)
def og_variant_test(name: str) -> tuple[pd.DataFrame, dict]:
    """Validation / private stage-1/2 rows (probs_v9 layout: p1 = v5, p2 = stage 2, p = the variant's
    probability where stage 2 applies) and window_id -> predicted cell set."""
    w, gate, cap = OG_VARIANTS[name]
    V = pd.read_parquet(WORK_OG / "probs_v9_ogstack08_ongoing.parquet")
    D = window_features("queue_ongoing"); D = D[D.src.isin(SPLITS)]
    g = pd.Series(og_gate(D, gate), index=D.key.to_numpy())
    on = V.window_id.map(g).to_numpy().astype(bool)
    V["p"] = np.where(on, w * V.p2 + (1 - w) * V.p1, V.p1)
    V["stage2"] = on
    cells = {}
    for wid, x in V.groupby("window_id", sort=False):
        kk = x.k.to_numpy().astype(int); ll = x.link.to_numpy().astype(int)
        s = (_decode_capped(x.p.to_numpy(), kk, x.p1.to_numpy(), cap) if bool(g[wid])
             else eiou_topm(x.p1.to_numpy())[0])
        cells[wid] = set(zip(kk[s].tolist(), ll[s].tolist()))
    return V, cells


def og_variant_edits(name: str) -> pd.DataFrame:
    """Edits of an ongoing variant against v5 on the CV and the validation / private windows (``edits`` layout)."""
    _, cv = og_variant_cv(name); _, te = og_variant_test(name)
    rcv, rte = cv_sets("queue_ongoing", "v5"), test_sets("queue_ongoing", "v5")
    rows = {}
    for sets, ref in ((cv, rcv), (te, rte)):
        for k, a in sets.items():
            b = ref[k]; n = len(a ^ b)
            rows[str(k)] = (np.log1p(n), float(n > 0), len(a - b), len(b - a), len(b))
    return pd.DataFrame.from_dict(rows, orient="index", columns=["fp_lchg", "fp_any", "add", "rem", "nref"])


# ----------------------------------------------------------------------------
# diagnostics and tables (section 17)
def balance(cond: str, target: str) -> pd.DataFrame:
    """Within-panel standardised mean difference (target - CV sim, in SD of the sim windows) of every design
    column, unweighted and after weighting: how much of the target mix the weights reproduce."""
    r = _cached(cond, target)
    D, X = r["D"], r["X"]
    sim = (D.src == "sim").to_numpy(); tg = ~sim
    w = r["w"].reindex(D.key[sim].astype(np.int64).to_numpy()).to_numpy()
    pan = D.panel.to_numpy()
    rows = {}
    for c in X.columns:
        x = X[c].to_numpy(np.float64)
        sd = np.sqrt(np.mean([np.var(x[sim & (pan == p)]) for p in PANELS8])) or 1.0
        d0 = d1 = 0.0
        for p in PANELS8:
            ms, mt = sim & (pan == p), tg & (pan == p)
            wp = w[pan[sim] == p]
            d0 += (x[mt].mean() - x[ms].mean()) / len(PANELS8)
            d1 += (x[mt].mean() - (wp * x[ms]).sum() / wp.sum()) / len(PANELS8)
        rows[c] = {"smd_unweighted": d0 / sd, "smd_weighted": d1 / sd}
    return pd.DataFrame(rows).T


def weights_report() -> pd.DataFrame:
    """Classifier and weight diagnostics per condition and target (validation, private, and the official train
    windows as a null check)."""
    rows = []
    for cond in CONDS:
        for t in ("validation", "private", "off"):
            r = _cached(cond, t)
            e = r["ess"]
            rows.append({"cond": cond, "target": t, "lambda": r["lam"], "auc_within_panel": r["auc"],
                         "ess_agg": e["agg"], "ess_agg_plain": e["agg_plain"], "ess_ratio": e["agg"] / e["agg_plain"],
                         "ess_min_panel": min(e["per_panel"].values()), "clipped_share": r["clipped"],
                         "w_max": float(r["w"].max()), "w_q99": float(r["w"].quantile(0.99)),
                         **{f"coef_{k}": v for k, v in r["coef"].items()}})
    return pd.DataFrame(rows)


def repro_table(n_boot: int = 2000, n_refit: int = 200) -> pd.DataFrame:
    """Section 17 table: every LB fact with plain / validation-weighted / private-weighted / change-aware Δ
    (primary truth: hybrid for onset, old for ongoing; the other truth alongside), the SE including the
    weight-model refit, the LB predictive z under plain and validation weights, and the footprint check."""
    rows = []
    for name, cond, new, ref, lb, how in CHANGES:
        T = tables(cond); a, b = T[new], T[ref]
        col = PRIMARY[cond]; alt = "iou_old" if col == "iou_hybrid" else "iou_hybrid"
        r = {"change": name, "cond": cond, "LB": lb, "LB_source": how}
        for tgt in ("plain", "validation", "private"):
            s = score(a, cond, tgt, ref=b, cols=[col, alt], n_boot=n_boot)
            r[f"{tgt}"] = s[col]["est"]; r[f"{tgt}_se"] = s[col]["se"]; r[f"{tgt}_alt"] = s[alt]["est"]
        r["validation_se_refit"] = boot_refit(a, b, col, cond, "validation", n_boot=n_refit)["se"]
        ca = fit_weights(cond, "validation", extra=footprint(cond, new, ref))
        r["validation_change_aware"] = weighted_agg(a.assign(_d=a[col] - b[col]), ca["w"], "_d")
        d = a.assign(_d=a[col] - b[col])
        for tgt in ("plain", "validation"):
            pr = lb_predictive(d, weights(cond, tgt), "_d")
            r[f"z_{tgt}"] = (lb - pr["mean"]) / pr["sd"]; r[f"lb_sd_{tgt}"] = pr["sd"]
        fc = footprint_check(cond, new, ref, "validation")
        r["fp_validation"] = " ".join(f"{k}:{v['target']:.2f}/{v['cv_mean']:.2f}(p{100 * v['pct']:.0f})" for k, v in fc.items())
        r["fp_flag"] = any(v["pct"] >= FP_FLAG for v in fc.values())
        o = (a.src == "off").to_numpy()
        from .onset_eval import OnsetEval
        r["off_actual"] = OnsetEval.agg(a[o].assign(_d=(a[col] - b[col])[o]), "_d")
        rows.append(r)
    return pd.DataFrame(rows)


def levels_table() -> pd.DataFrame:
    """CV levels (plain / validation-weighted) of the versions whose March level is known."""
    rows = []
    for (cond, v), lb in LB_LEVEL.items():
        T = tables(cond)[v]
        r = {"cond": cond, "version": v, "LB_March": lb}
        for tgt in ("plain", "validation", "private"):
            for c in ("iou_hybrid", "iou_old"):
                r[f"{tgt}:{c[4:]}"] = weighted_agg(T, weights(cond, tgt), c)
        r["lb_level_sd"] = lb_predictive(T, weights(cond, "validation"), PRIMARY[cond])["sd"]
        rows.append(r)
    return pd.DataFrame(rows)


def mitigation_table(names=None, n_boot: int = 2000) -> pd.DataFrame:
    """Ongoing stage-2 variants vs v5: plain / validation- / private-weighted Δ (old and hybrid truth), the
    section 16 slices, the footprint check on both months and the number of edited windows."""
    b = tables("queue_ongoing")["v5"]
    rows = []
    for name in names or OG_VARIANTS:
        d, _ = og_variant_cv(name)
        r = {"variant": name}
        for tgt in ("plain", "validation", "private"):
            s = score(d, "queue_ongoing", tgt, ref=b, cols=["iou_old", "iou_hybrid"], n_boot=n_boot)
            for c in ("iou_old", "iou_hybrid"):
                r[f"{tgt}:{c[4:]}"] = s[c]["est"]; r[f"{tgt}:{c[4:]}_se"] = s[c]["se"]
        s_ = (d.src == "sim").to_numpy(); rr = d.rec.to_numpy(); dd = (d.iou_old - b.iou_old).to_numpy()
        o = (d.src == "off").to_numpy()
        from .onset_eval import OnsetEval
        r["off"] = OnsetEval.agg(d[o].assign(_d=dd[o]), "_d")
        r["rec<0.05"] = float(dd[s_ & (rr < 0.05)].mean())
        r["I10W rec<0.05"] = float(dd[s_ & (rr < 0.05) & (d.panel == "D7_I10_W").to_numpy()].mean())
        Ed = og_variant_edits(name)
        for tgt in SPLITS:
            fc = footprint_check("queue_ongoing", Ed, None, tgt)
            r[f"{tgt[:3]}:add"] = f"{fc['add']['target']:.2f} (p{100 * fc['add']['pct']:.0f})"
            r[f"{tgt[:3]}:rem"] = f"{fc['rem']['target']:.2f} (p{100 * fc['rem']['pct']:.0f})"
            r[f"{tgt[:3]}:flag"] = any(v["pct"] >= FP_FLAG for v in fc.values())
            te = Ed.reindex([k for k in Ed.index if f"_{tgt}_" in k])
            r[f"{tgt[:3]}:windows"] = int((te.fp_any > 0).sum()); r[f"{tgt[:3]}:cells"] = int((te["add"] + te.rem).sum())
        rows.append(r)
    return pd.DataFrame(rows).set_index("variant")


# ----------------------------------------------------------------------------
# candidate file
def build(name: str = CANDIDATE, tag: str | None = None) -> dict:
    """/home/user/work/t2/lgb_v10_<tag>.csv = lgb_v8_seeds9_stack03.csv with only the ongoing rows replaced by
    the variant's decoding; also WORK_OG/probs_v10_<tag>_ongoing.parquet (probs_v9 layout + stage2 flag)."""
    import json
    from .ongoing_v9 import BASE_CSV, onset_lines_identical, write_from_base
    from .submit import check
    tag = tag or name.replace(" ", "_").replace(".", "").replace("(", "").replace(")", "")
    V, cells = og_variant_test(name)
    preds = {}
    for wid, g in V.groupby("window_id", sort=False):
        A = np.zeros((K, statics(g.panel.iat[0])["L"]), bool)
        for k, l in cells[wid]:
            A[k - 1, l] = True
        preds[wid] = A
    out = WORK_OG / f"lgb_v10_{tag}.csv"
    assert not out.exists(), f"{out} exists"
    ch = write_from_base(preds, out)
    ch["onset_lines_byte_identical"] = onset_lines_identical(BASE_CSV, out)
    V[["window_id", "panel", "k", "link", "p", "p1", "p2", "stage2", "split"]].to_parquet(
        WORK_OG / f"probs_v10_{tag}_ongoing.parquet")
    res = {"file": str(out), "variant": name, "rule": OG_VARIANTS[name], "check": check(out), "vs_base": ch}
    print(json.dumps(res, default=str), flush=True)
    return res


def main():
    pd.set_option("display.width", 320); pd.set_option("display.max_columns", 80)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "repro"
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "features":
        for c in CONDS:
            print(c, window_features(c).groupby("src").size().to_dict())
    elif cmd == "weights":
        R = weights_report(); R.to_csv(OUT / "weights_report.csv", index=False)
        print(R.round(3).T.to_string())
        for c in CONDS:
            for t in TARGETS:
                print(c, t, "balance (within-panel SMD, unweighted -> weighted):\n" + balance(c, t).round(3).to_string())
    elif cmd == "repro":
        R = repro_table(); R.to_csv(OUT / "repro_table.csv", index=False)
        print(R.drop(columns=["LB_source"]).round(4).to_string(index=False))
        L = levels_table(); L.to_csv(OUT / "levels_table.csv", index=False)
        print(L.round(4).to_string(index=False))
    elif cmd == "mitigate":
        R = mitigation_table(); R.to_csv(OUT / "mitigation_table.csv")
        print(R.round(4).T.to_string())
    elif cmd == "build":
        build(sys.argv[2] if len(sys.argv) > 2 else CANDIDATE, sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
