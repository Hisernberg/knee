"""Onset CV evaluator on the re-drawn windows under the hybrid and the old truth.

The windows are the ones the reproduced selector draws on the hybrid truth
(``/home/user/work/t2h``, TASK2_ANALYSIS.md sections 13 and 15): 2,081 ``sim``
and 40 official (``off``) onset windows. A probability vector aligned to the
rows of an onset OOF file of that set (``robust cv ... --cond queue_onset``
with ``T2_OOF_ALL=1``: one row per (window, link) at T+30 for every onset
window, including the ``cand`` training windows) is decoded per window with
top-m expected-IoU (``models.eiou_topm``) and scored against

* ``hybrid``: ``ds_<panel>.npz["y"]``, the truthfix hybrid truth (v6 labels);
* ``old``: ``ds_<panel>.npz["y_old"]``, the time-first interpolated truth
  (the labels up to v5; the conservative evaluation);
* ``old_e`` (optional, ``truths=ALL_TRUTHS``): the old truth with steps 1..5
  kept, the convention of the section 13 old-truth row (v6 0.7605 there).

For ``hybrid`` and ``old``, steps 1..5 of every onset window are empty, as in
``robust.load_truth``. The organizer confirms the onset horizon is empty
before T+30, and we predict T+30 only. Under the old truth, 261 of the 2,081
re-drawn sim windows have queue cells at steps 1..5.

Returned per truth:
* ``sim`` and ``off`` with the official aggregation (window -> panel ->
  family -> mean);
* the plain mean IoU of the sim windows whose onset recurrence is < 0.05 or
  < 0.2. The recurrence is ``robust.onset_recurrence`` on the t2h tables: the
  mean fold-excluded train time-of-day queue probability ``pq_k`` at T+30
  over the links queued at T+30 under the hybrid truth. That gives the same
  144 and 438 windows for every truth, as in section 13. It is for
  evaluation only.

    from trafficflow.t2.onset_eval import OnsetEval, V6
    E = OnsetEval()
    p6 = E.load_mean(V6)                # v6 onset = mean of its four OOF files
    E.evaluate(p6)                      # {"hybrid": {...}, "old": {...}}
    E.compare(p_new, p6)                # paired bootstrap by window, new - v6

    python -m trafficflow.t2.onset_eval [OOF_FILE ...]   # v6 and (mean of files) vs v6
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .core import FAMILY, K, PANELS8
from .models import eiou_topm

T2H = Path(os.environ.get("T2_EVAL_WORK", "/home/user/work/t2h"))
V6 = ["oof_queue_onset_rob_on_v3_p1_op_all_new.parquet", "oof_queue_onset_rob_on_v3_p1_op_s1_all_new.parquet",
      "oof_queue_onset_rob_on_v3_p1_op_s2_all_new.parquet", "oof_queue_onset_rob_on_v2_p1_op_all_new.parquet"]
KEY = ["gw", "k", "link"]
# name -> (ds key, keep steps 1..5)
ALL_TRUTHS = {"hybrid": ("y", False), "old": ("y_old", False), "old_e": ("y_old", True)}
TRUTHS = ("hybrid", "old")
THR = (0.05, 0.2)
FAMS = sorted(set(FAMILY.values()))
PAN_FAM = np.array([FAMS.index(FAMILY[p]) for p in PANELS8])


class OnsetEval:
    """Scores per-row onset probabilities (aligned to ``self.rows``) per window."""

    def __init__(self, base: str = V6[0], work: Path = T2H, feat: Path | None = None, truths=TRUTHS):
        self.work = Path(work)
        self.feat = Path(feat) if feat else self.work / "feat"
        self.truths = tuple(truths)
        self.rows = pd.read_parquet(self.work / base, columns=KEY)
        gw = self.rows.gw.to_numpy()
        Ms = []
        for p in PANELS8:
            M = pd.read_parquet(self.feat / f"meta_{p}.parquet")
            M["gw"] = PANELS8.index(p) * 100000 + M.w.astype(np.int64)
            Ms.append(M[M.condition == "queue_onset"])
        self.meta = pd.concat(Ms).drop_duplicates("gw").set_index("gw")
        link = self.rows.link.to_numpy().astype(np.int64)
        assert (self.rows.k.to_numpy() == K).all(), "onset rows are T+30 only"
        truth = {p: self._truth(p) for p in PANELS8}
        order = np.argsort(gw, kind="stable")
        W = []
        for idx in np.split(order, np.flatnonzero(np.diff(gw[order])) + 1):
            g = int(gw[idx[0]]); m = self.meta.loc[g]
            if m.src not in ("sim", "off"):
                continue
            w = dict(gw=g, panel=m.panel, src=m.src, idx=idx, link=link[idx])
            for t in self.truths:
                y6, early = truth[m.panel][t]
                w[f"y_{t}"] = y6[int(m.w)]
                w[f"e_{t}"] = int(early[int(m.w)]) if ALL_TRUTHS[t][1] else 0
            W.append(w)
        self.win = W
        info = pd.DataFrame([{k: w[k] for k in ("gw", "panel", "src")} for w in W])
        info["pi"] = info.panel.map({p: i for i, p in enumerate(PANELS8)})
        for t in self.truths:
            info[f"n_{t}"] = [int(w[f"y_{t}"].sum()) for w in W]
        self.info = info.join(self._recurrence(info), on="gw")

    # ------------------------------------------------------------------
    def _truth(self, p: str) -> dict:
        """{truth: ([n, L] bool at T+30, [n] number of queued cells at steps 1..5)}."""
        z = np.load(self.work / f"ds_{p}.npz", allow_pickle=True)
        out = {}
        for t in self.truths:
            y = z[ALL_TRUTHS[t][0]]
            out[t] = (y[:, K - 1].astype(bool), y[:, :K - 1].sum((1, 2)))
        return out

    def _recurrence(self, info: pd.DataFrame) -> pd.Series:
        """``robust.onset_recurrence`` for the evaluation windows: mean pq_k over
        the links with y == 1 (hybrid truth at T+30) in the t2h feature table."""
        out = []
        for p in PANELS8:
            pc = PANELS8.index(p)
            ids = (info.gw[info.panel == p].to_numpy() % 100000).tolist()
            X = pd.read_parquet(self.feat / f"feat_{p}.parquet", columns=["w", "pq_k", "y"],
                                filters=[("w", "in", ids)])
            r = X[X.y == 1].groupby("w").pq_k.mean().reindex(ids).fillna(0.0)
            r.index = pc * 100000 + r.index.astype(np.int64)
            out.append(r)
        return pd.concat(out).rename("rec")

    # ------------------------------------------------------------------
    def align(self, df: pd.DataFrame, col: str = "p") -> np.ndarray:
        """Probability vector aligned to ``self.rows`` from a frame with gw,k,link,<col>."""
        m = self.rows.merge(df[KEY + [col]], on=KEY, how="left")
        assert len(m) == len(self.rows) and m[col].notna().all(), "rows missing"
        return m[col].to_numpy(np.float64)

    def load(self, f: str) -> np.ndarray:
        x = pd.read_parquet(self.work / f, columns=KEY + ["p"])
        if len(x) == len(self.rows) and (x[KEY].to_numpy() == self.rows[KEY].to_numpy()).all():
            return x.p.to_numpy(np.float64)
        return self.align(x)

    def load_mean(self, files: list[str], weights: list[float] | None = None) -> np.ndarray:
        w = weights or [1.0] * len(files)
        return sum(wi * self.load(f) for wi, f in zip(w, files)) / sum(w)

    # ------------------------------------------------------------------
    def decode(self, p: np.ndarray) -> list[np.ndarray]:
        """Top-m expected-IoU set (link indices at T+30) per evaluation window."""
        p = np.asarray(p, np.float64)
        assert len(p) == len(self.rows), (len(p), len(self.rows))
        return [w["link"][eiou_topm(p[w["idx"]])[0]] for w in self.win]

    def window_iou(self, p: np.ndarray) -> pd.DataFrame:
        """Per evaluation window: IoU under each truth and the predicted set size m."""
        sets = self.decode(p)
        out = {t: np.empty(len(self.win)) for t in self.truths}
        for i, (w, s) in enumerate(zip(self.win, sets)):
            for t in self.truths:
                y = w[f"y_{t}"]
                inter = int(y[s].sum()); union = len(s) + int(y.sum()) - inter + w[f"e_{t}"]
                out[t][i] = 1.0 if union == 0 else inter / union
        return self.info.assign(**{f"iou_{t}": out[t] for t in self.truths}, m=[len(s) for s in sets])

    @staticmethod
    def agg(df: pd.DataFrame, col: str) -> float:
        """Official aggregation for one condition: window -> panel -> family -> mean."""
        pm = df.groupby("panel")[col].mean()
        return float(pm.groupby(pm.index.map(FAMILY)).mean().mean())

    def summary(self, df: pd.DataFrame) -> dict:
        s = df[df.src == "sim"]; o = df[df.src == "off"]
        out = {}
        for t in self.truths:
            r = {"sim": self.agg(s, f"iou_{t}"), "off": self.agg(o, f"iou_{t}")}
            for thr in THR:
                x = s[s.rec < thr]
                r[f"recur<{thr}"] = float(x[f"iou_{t}"].mean())
                r[f"n<{thr}"] = len(x)
            out[t] = r
        return out

    def evaluate(self, p: np.ndarray) -> dict:
        return self.summary(self.window_iou(p))

    # ------------------------------------------------------------------
    def compare(self, pa, pb, n_boot: int = 2000, seed: int = 0) -> dict:
        """a - b (e.g. candidate - v6) on the same windows, per truth: Δ sim / off
        (official aggregation), Δ of the recurrence slices, paired-bootstrap SE of
        Δ sim (sim windows resampled jointly with replacement, as in
        og_labelfix_eval) and of the slice means, share of bootstrap Δ sim <= 0,
        and the number of sim windows better / worse. ``pa``/``pb`` are
        probability vectors or window_iou() frames."""
        da = pa if isinstance(pa, pd.DataFrame) else self.window_iou(pa)
        db = pb if isinstance(pb, pd.DataFrame) else self.window_iou(pb)
        rng = np.random.default_rng(seed)
        s = (da.src == "sim").to_numpy(); o = (da.src == "off").to_numpy()
        pi = da.pi.to_numpy()[s]; rec = da.rec.to_numpy()[s]
        n = int(s.sum())
        B = rng.integers(0, n, (n_boot, n))
        flat = (np.arange(n_boot)[:, None] * 8 + pi[B]).ravel()
        cnt = np.bincount(flat, minlength=8 * n_boot).reshape(n_boot, 8)
        out = {}
        for t in self.truths:
            d = (da[f"iou_{t}"] - db[f"iou_{t}"]).to_numpy()
            ds = d[s]
            num = np.bincount(flat, weights=ds[B].ravel(), minlength=8 * n_boot).reshape(n_boot, 8)
            pm = num / np.maximum(cnt, 1)
            boot = np.stack([pm[:, PAN_FAM == f].mean(1) for f in range(len(FAMS))], 1).mean(1)
            r = {"d_sim": self.agg(da[s].assign(_d=ds), "_d"), "se_sim": float(boot.std()),
                 "p_le0": float((boot <= 0).mean()), "d_off": self.agg(da[o].assign(_d=d[o]), "_d"),
                 "better": int((ds > 1e-9).sum()), "worse": int((ds < -1e-9).sum())}
            for thr in THR:
                dm = ds[rec < thr]
                r[f"d_recur<{thr}"] = float(dm.mean())
                r[f"se_recur<{thr}"] = float(dm[rng.integers(0, len(dm), (n_boot, len(dm)))].mean(1).std())
            out[t] = r
        return out


def table(E: OnsetEval, variants: dict, ref: str | None = None, n_boot: int = 2000) -> pd.DataFrame:
    """One row per variant: sim/off/slices under each truth, and Δ ± SE vs ``ref``."""
    dfs = {k: E.window_iou(v) for k, v in variants.items()}
    rows = []
    for k, df in dfs.items():
        s = E.summary(df)
        r = {"variant": k}
        for t in E.truths:
            for c in ("sim", "off", "recur<0.05", "recur<0.2"):
                r[f"{t}:{c}"] = s[t][c]
        if ref is not None and k != ref:
            c = E.compare(df, dfs[ref], n_boot=n_boot)
            for t in E.truths:
                r[f"{t}:Δsim"] = c[t]["d_sim"]; r[f"{t}:se"] = c[t]["se_sim"]
                r[f"{t}:Δoff"] = c[t]["d_off"]
                r[f"{t}:Δ<0.05"] = c[t]["d_recur<0.05"]; r[f"{t}:Δ<0.2"] = c[t]["d_recur<0.2"]
                r[f"{t}:b/w"] = f"{c[t]['better']}/{c[t]['worse']}"
        rows.append(r)
    return pd.DataFrame(rows).set_index("variant")


def main(files: list[str]):
    E = OnsetEval(truths=tuple(ALL_TRUTHS))
    V = {"v6": E.load_mean(V6)}
    for f in V6:
        V[f.replace("oof_queue_onset_rob_", "").replace(".parquet", "")] = E.load(f)
    if files:
        V["candidate"] = E.load_mean(files)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    print(table(E, V, ref="v6").round(4).T.to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
