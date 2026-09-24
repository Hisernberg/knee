"""Onset queue extent: diagnosis, extent predictors and structured decoders.

The onset truth is the first queued slot (T+30) of a new queue: one block (or
two) of links upstream of a bottleneck. ``diagnose`` compares, on the OOF
probabilities, the predicted (top-m) and true blocks where the site is right:
extent in links / km, head and tail offsets, false positives and negatives
inside and outside the site.

    python -m trafficflow.t2.onset_extent diagnose OOF_FILE
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ..data import SLOTS
from .core import FEAT, K, PANELS8, WORK, aggregate, iou, statics
from .models import eiou_topm


def runs(mask: np.ndarray, gap: int = 1) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    cut = np.flatnonzero(np.diff(idx) > gap + 1)
    return list(zip(np.r_[idx[0], idx[cut + 1]], np.r_[idx[cut], idx[-1]]))


def window_probs(oof_file: str):
    """Yield (gw, meta row, p [L], y6 [L]) for every OOF onset window."""
    from .robust import meta_index, truth_lookup_y
    O = pd.read_parquet(WORK / oof_file).sort_values("gw", kind="stable")
    Mi = meta_index("queue_onset"); Y = truth_lookup_y()
    g = O.gw.to_numpy(); b = np.flatnonzero(np.diff(g)) + 1
    ll = O.link.to_numpy().astype(int); pp = O.p.to_numpy()
    for idx in np.split(np.arange(len(O)), b):
        gw = int(g[idx[0]]); m = Mi.loc[gw]
        y6 = Y[m.panel][int(m.w)][K - 1]
        p = np.zeros(len(y6)); p[ll[idx]] = pp[idx]
        yield gw, m, p, y6


def iou6(A6: np.ndarray, gw: int, m) -> float:
    """IoU of a T+30-only prediction against the full 6-step (approximate) truth,
    the metric used by the earlier CV tables."""
    from .robust import truth_lookup_y
    yt = truth_lookup_y()[m.panel][int(m.w)]
    P = np.zeros_like(yt); P[K - 1] = A6
    return iou(P, yt)


def diagnose(oof_file: str) -> pd.DataFrame:
    rows = []
    for gw, m, p, y in window_probs(oof_file):
        st = statics(m.panel)
        length = st["length"]
        P = np.zeros(len(y), bool); P[eiou_topm(p)[0]] = True
        r = dict(gw=gw, panel=m.panel, src=m.src, T=int(m["T"]), iou=iou(P, y), n_true=int(y.sum()), n_pred=int(P.sum()))
        tr = runs(y); pr = runs(P)
        r["n_true_runs"] = len(tr); r["n_pred_runs"] = len(pr)
        # the predicted cluster and truth cluster with the largest overlap
        best = None
        for a, b in pr:
            for c, d in tr:
                ov = int((P[max(a, c):min(b, d) + 1] & y[max(a, c):min(b, d) + 1]).sum())
                if ov > 0 and (best is None or ov > best[0]):
                    best = (ov, a, b, c, d)
        r["site_right"] = best is not None
        if best is not None:
            _, a, b, c, d = best
            r.update(p_head=b, p_tail=a, t_head=d, t_tail=c,
                     head_off=b - d, tail_off=a - c,            # >0: predicted downstream of truth
                     n_true_site=int(y[c:d + 1].sum()), n_pred_site=int(P[a:b + 1].sum()),
                     len_true_site=float(length[c:d + 1][y[c:d + 1]].sum()),
                     len_pred_site=float(length[a:b + 1][P[a:b + 1]].sum()),
                     mean_link_len=float(length[c:d + 1].mean()),
                     fp_site=int((P[a:b + 1] & ~y[a:b + 1]).sum()), fn_site=int((y[c:d + 1] & ~P[c:d + 1]).sum()))
            lo, hi = min(a, c), max(b, d)
            outside = np.ones(len(y), bool); outside[lo:hi + 1] = False
            r["fp_out"] = int((P & outside).sum()); r["fn_out"] = int((y & outside).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def summarize_diag(d: pd.DataFrame) -> None:
    s = d[d.src == "sim"].copy()
    s["tod_h"] = (s["T"] % SLOTS) // 12
    print("windows", len(s), "site right", round(s.site_right.mean(), 3), "IoU", round(s.iou.mean(), 3),
          "IoU | right", round(s[s.site_right].iou.mean(), 3))
    r = s[s.site_right]
    print("| right: n_true_site %.2f n_pred_site %.2f | head_off mean %.2f (|.|>0: %.2f) tail_off mean %.2f (|.|>0: %.2f)"
          % (r.n_true_site.mean(), r.n_pred_site.mean(), r.head_off.mean(), (r.head_off != 0).mean(),
             r.tail_off.mean(), (r.tail_off != 0).mean()))
    print("| right: fp_site %.2f fn_site %.2f fp_out %.2f fn_out %.2f"
          % (r.fp_site.mean(), r.fn_site.mean(), r.fp_out.mean(), r.fn_out.mean()))
    r = r.assign(ext_err=r.n_pred_site - r.n_true_site)
    print("extent error (pred - true links) distribution:",
          r.ext_err.clip(-5, 5).value_counts().sort_index().to_dict())
    print(r.groupby("panel").agg(n=("iou", "size"), iou=("iou", "mean"), n_true=("n_true_site", "mean"),
                                 n_pred=("n_pred_site", "mean"), head_off=("head_off", "mean"),
                                 tail_off=("tail_off", "mean"), fp=("fp_site", "mean"), fn=("fn_site", "mean"),
                                 link_len=("mean_link_len", "mean")).round(2).to_string())
    r["len_bin"] = pd.cut(r.mean_link_len, [0, 0.2, 0.3, 0.45, 0.7, 5])
    print(r.groupby("len_bin").agg(n=("iou", "size"), iou=("iou", "mean"), n_true=("n_true_site", "mean"),
                                   n_pred=("n_pred_site", "mean"), km_true=("len_true_site", "mean"),
                                   km_pred=("len_pred_site", "mean")).round(2).to_string())
    print(r.groupby("tod_h").agg(n=("iou", "size"), iou=("iou", "mean"), n_true=("n_true_site", "mean"),
                                 n_pred=("n_pred_site", "mean")).round(2).to_string())



# ----------------------------------------------------------------------------
# library (shape-prior) decoding
class Library:
    """Truth sets (T+30) of the training onset windows of one panel: unique sets
    as a boolean matrix, with per-window membership (set index, fold, tod)."""

    def __init__(self, panel: str):
        from .baselines import fold_of
        z = np.load(WORK / f"ds_{panel}.npz", allow_pickle=True)
        src = z["w_src"]; cond = z["w_condition"]
        idx = np.flatnonzero((cond == "queue_onset") & (src == "cand"))
        Y = z["y"][idx, K - 1]
        keep = Y.any(1)
        Y = Y[keep]; T = z["w_T"][idx][keep].astype(np.int64)
        U, inv = np.unique(Y, axis=0, return_inverse=True)
        self.U = U.astype(bool)
        self.set_of = inv.ravel()
        self.fold = fold_of(T)
        self.tod = T % SLOTS
        inter = self.U.astype(np.int32) @ self.U.T.astype(np.int32)
        sz = self.U.sum(1)
        self.iou = inter / (sz[:, None] + sz[None, :] - inter)

    def prior(self, fold: int | None, tod: int, tod_scale: float | None) -> np.ndarray:
        ok = np.ones(len(self.set_of), bool) if fold is None else (self.fold != fold)
        w = np.ones(ok.sum())
        if tod_scale:
            w = np.exp(-np.abs(self.tod[ok] - tod) / tod_scale)
        return np.bincount(self.set_of[ok], weights=w, minlength=len(self.U))


def surrogate(A: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Independent-Bernoulli expected-IoU surrogate for candidate sets A [c, L]."""
    s = A.astype(np.float64) @ p
    return s / (A.sum(1) + p.sum() - s)


def decode_library(p: np.ndarray, lib: Library, fold, tod, tau=0.3, alpha=0.7, tod_scale=None,
                   extra: np.ndarray | None = None) -> np.ndarray:
    pc = np.clip(p, 1e-4, 1 - 1e-4)
    lp, lq = np.log(pc), np.log1p(-pc)
    U = lib.U
    loglik = U @ lp + (~U) @ lq
    prior = lib.prior(fold, tod, tod_scale)
    ok = prior > 0
    post = np.zeros(len(U))
    if ok.any():
        z = tau * (loglik[ok] - loglik[ok].max())
        post[ok] = prior[ok] * np.exp(z)
        post /= post.sum()
    topm = np.zeros(len(p), bool); topm[eiou_topm(p)[0]] = True
    cands = [U[ok], topm[None]] + ([extra] if extra is not None else [])   # own-fold-only sets excluded
    C = np.vstack(cands)
    inter = C.astype(np.int32) @ U.T.astype(np.int32)
    union = C.sum(1)[:, None] + U.sum(1)[None, :] - inter
    e_lib = (inter / union) @ post
    e_ind = surrogate(C, p)
    score = alpha * e_lib + (1 - alpha) * e_ind
    return C[int(np.argmax(score))]


def eval_library(oof_file: str, grid: list[dict]) -> pd.DataFrame:
    from .baselines import fold_of
    from .robust import onset_recurrence
    libs = {p: Library(p) for p in PANELS8}
    rec = onset_recurrence()
    rows = []
    for gw, m, p, y in window_probs(oof_file):
        f = int(fold_of(np.array([m["T"]]))[0]); tod = int(m["T"] % SLOTS)
        r = dict(gw=gw, panel=m.panel, condition="queue_onset", src=m.src, recur=rec.get(gw, np.nan))
        P = np.zeros(len(y), bool); P[eiou_topm(p)[0]] = True
        r["topm"] = iou6(P, gw, m)
        for g in grid:
            name = "lib_" + "_".join(f"{k}{v}" for k, v in g.items())
            A = decode_library(p, libs[m.panel], f, tod, **g)
            r[name] = iou6(A, gw, m)
        rows.append(r)
    return pd.DataFrame(rows)


def summarize_methods(df: pd.DataFrame, names=None) -> pd.DataFrame:
    names = names or [c for c in df.columns if c == "topm" or c.startswith(("lib_", "he_", "hy_"))]
    out = {}
    for n in names:
        s = df[df.src == "sim"]
        out[n] = {"sim": aggregate(s.rename(columns={n: "iou"}))["queue_onset"],
                  "off": aggregate(df[df.src == "off"].rename(columns={n: "iou"}))["queue_onset"],
                  "rec<0.05": s[s.recur < 0.05][n].mean(), "rec<0.2": s[s.recur < 0.2][n].mean()}
    return pd.DataFrame(out).T.round(4)


# ----------------------------------------------------------------------------
# head + extent decoding
def direction(panel: str) -> int:
    """+1 if the link index increases downstream (E/N panels), -1 if it
    increases upstream (W/S panels: link i+1 is the incoming link of link i)."""
    from ..data import network
    net = network(panel); topo = net["topo"]; links = net["links"]
    nxt = {l: (str(o).split(";") if isinstance(o, str) else []) for l, o in zip(topo.link_id.astype(str), topo.outgoing_link_ids)}
    prv = {l: (str(o).split(";") if isinstance(o, str) else []) for l, o in zip(topo.link_id.astype(str), topo.incoming_link_ids)}
    fwd = sum(links[i + 1] in nxt[links[i]] for i in range(len(links) - 1))
    bwd = sum(links[i + 1] in prv[links[i]] for i in range(len(links) - 1))
    return 1 if fwd >= bwd else -1


class ExtentPrior:
    """Empirical extent (links, tail..head inclusive, traffic direction) of the
    main truth run of training onset windows, by head link."""

    def __init__(self, panel: str):
        from .baselines import fold_of
        self.dir = direction(panel)
        z = np.load(WORK / f"ds_{panel}.npz", allow_pickle=True)
        src = z["w_src"]; cond = z["w_condition"]
        idx = np.flatnonzero((cond == "queue_onset") & (src == "cand"))
        rows = []
        for i in idx:
            y = z["y"][i, K - 1]
            rr = runs(y)
            if not rr:
                continue
            a, b = max(rr, key=lambda ab: y[ab[0]:ab[1] + 1].sum())
            head = b if self.dir > 0 else a
            rows.append((head, b - a + 1, int(z["w_T"][i])))
        d = pd.DataFrame(rows, columns=["head", "n", "T"])
        d["fold"] = fold_of(d["T"].to_numpy())
        self.d = d

    def best_extent(self, head: int, fold, kmax: int = 15, min_n: int = 5) -> int | None:
        d = self.d[(self.d["head"] == head) & ((self.d.fold != fold) if fold is not None else True)]
        if len(d) < min_n:
            return None
        n = d.n.to_numpy()
        ks = np.arange(1, kmax + 1)
        # block-vs-block IoU of a k-link block with the same head against each training extent
        sc = (np.minimum(ks[:, None], n[None, :]) / np.maximum(ks[:, None], n[None, :])).mean(1)
        return int(ks[np.argmax(sc)])


def block_upstream(L: int, head: int, k: int, dirn: int) -> np.ndarray:
    A = np.zeros(L, bool)
    if dirn > 0:
        A[max(0, head - k + 1):head + 1] = True
    else:
        A[head:min(L, head + k)] = True
    return A


def decode_head_extent(p: np.ndarray, ep: ExtentPrior, fold, mode: str = "block", head_from: str = "topm") -> np.ndarray:
    L = len(p)
    topm = np.zeros(L, bool); topm[eiou_topm(p)[0]] = True
    cl = runs(topm)
    if not cl:
        return topm
    a, b = max(cl, key=lambda ab: p[ab[0]:ab[1] + 1].sum())
    if head_from == "topm":
        head = b if ep.dir > 0 else a
    else:  # marginal probability of being the head (queued, next downstream not)
        nxt = np.r_[p[1:], 0.0] if ep.dir > 0 else np.r_[0.0, p[:-1]]
        head = int(np.argmax(p * (1 - nxt)))
    k = ep.best_extent(head, fold)
    if k is None:
        return topm
    blk = block_upstream(L, head, k, ep.dir)
    if mode == "block":
        # keep any other top-m clusters (second site) untouched
        other = topm.copy(); other[min(a, b):max(a, b) + 1] = False
        return blk | other
    # hybrid: top-m restricted to the predicted block widened by one link each side
    win = blk | np.r_[blk[1:], False] | np.r_[False, blk[:-1]]
    keep = topm & win
    other = topm.copy(); other[min(a, b):max(a, b) + 1] = False
    return keep | other if keep.any() else topm


def eval_head_extent(oof_file: str) -> pd.DataFrame:
    from .baselines import fold_of
    from .robust import onset_recurrence
    eps = {p: ExtentPrior(p) for p in PANELS8}
    rec = onset_recurrence()
    rows = []
    for gw, m, p, y in window_probs(oof_file):
        f = int(fold_of(np.array([m["T"]]))[0])
        r = dict(gw=gw, panel=m.panel, condition="queue_onset", src=m.src, recur=rec.get(gw, np.nan))
        P = np.zeros(len(y), bool); P[eiou_topm(p)[0]] = True
        r["topm"] = iou6(P, gw, m)
        for mode in ("block", "hybrid"):
            for hf in ("topm", "marg"):
                r[f"he_{mode}_{hf}"] = iou6(decode_head_extent(p, eps[m.panel], f, mode, hf), gw, m)
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if sys.argv[1] == "diagnose":
        d = diagnose(sys.argv[2])
        d.to_parquet(WORK / "onset_extent_diag.parquet")
        summarize_diag(d)
    elif sys.argv[1] == "library":
        grid = [dict(tau=t, alpha=a) for t in (0.1, 0.3, 1.0) for a in (0.5, 0.8, 1.0)]
        df = eval_library(sys.argv[2], grid)
        df.to_parquet(WORK / "onset_library_eval.parquet")
        print(summarize_methods(df).to_string())
    elif sys.argv[1] == "head_extent":
        df = eval_head_extent(sys.argv[2])
        df.to_parquet(WORK / "onset_head_extent_eval.parquet")
        print(summarize_methods(df).to_string())
