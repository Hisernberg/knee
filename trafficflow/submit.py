"""Assemble the single Kaggle upload and run the pre-submission checklist.

Inputs are three long frames keyed the way the per-task templates are keyed:
  state: panel, timestamp(text), station_id, link_id, mask_regime, speed_kmh, flow_vph
         (validation + private rows; split is implied by the timestamp)
  queue: window_id, timestamp(text), link_id, queue_pred
  odme : panel, departure_time, path_id, path_flow
Every check prints PASS/FAIL; any FAIL aborts unless force=True.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import REL, PANELS

KEY = REL / "submission_key.csv"
SAMPLE = REL / "sample_submission.csv"


class Checks:
    def __init__(self):
        self.rows = []

    def __call__(self, name: str, ok: bool, info: str = ""):
        self.rows.append((name, bool(ok), info))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)

    @property
    def failed(self):
        return [r for r in self.rows if not r[1]]


def load_key() -> pd.DataFrame:
    return pd.read_csv(KEY, dtype=str, keep_default_na=False)


def assemble(state: pd.DataFrame, queue: pd.DataFrame, odme: pd.DataFrame, out: Path,
             force: bool = False, note: str = "") -> Path:
    C = Checks()
    key = load_key()
    sample = pd.read_csv(SAMPLE, usecols=["submission_id", "task"], dtype=str)
    C("key/sample same ids", (key.submission_id.values == sample.submission_id.values).all())
    sub = pd.DataFrame({"submission_id": key.submission_id.astype(np.int64), "task": key.task})
    for c in ("speed_kmh", "flow_vph", "queue_pred", "path_flow"):
        sub[c] = 0.0

    # ---------------- state
    ks = key[key.task == "state"][["submission_id", "panel", "timestamp", "station_id", "link_id", "mask_regime"]]
    st = state.copy()
    for c in ("panel", "timestamp", "station_id", "link_id", "mask_regime"):
        st[c] = st[c].astype(str)
    C("state no duplicate keys", not st.duplicated(["panel", "timestamp", "station_id", "link_id", "mask_regime"]).any())
    m = ks.merge(st, on=["panel", "timestamp", "station_id", "link_id", "mask_regime"], how="left")
    C("state rows == key rows", len(m) == len(ks), f"{len(m)} vs {len(ks)}")
    miss = int(m.speed_kmh.isna().sum() + m.flow_vph.isna().sum())
    C("state no missing values", miss == 0, f"missing={miss}")
    C("state all finite", np.isfinite(m.speed_kmh).all() and np.isfinite(m.flow_vph).all())
    C("speed within [3, 140]", m.speed_kmh.between(3, 140).all(), f"min={m.speed_kmh.min():.2f} max={m.speed_kmh.max():.2f}")
    C("flow within [0, 16000]", m.flow_vph.between(0, 16000).all(), f"min={m.flow_vph.min():.1f} max={m.flow_vph.max():.1f}")
    for p in PANELS:
        mp = m[m.panel == p]
        for r in ("R1", "R2", "R3"):
            x = mp[mp.mask_regime == r]
            if len(x):
                C(f"{p} {r} FD empty-road guard (<50vph share<=0.2)", (x.flow_vph < 50).mean() <= 0.2,
                  f"share={(x.flow_vph < 50).mean():.4f}")
    C("mean speed plausible (80-120)", 80 < m.speed_kmh.mean() < 120, f"{m.speed_kmh.mean():.2f}")
    idx = m.submission_id.astype(np.int64).to_numpy() - 1
    sub.loc[idx, "speed_kmh"] = m.speed_kmh.to_numpy()
    sub.loc[idx, "flow_vph"] = m.flow_vph.to_numpy()

    # ---------------- queue
    kq = key[key.task == "queue"][["submission_id", "window_id", "timestamp", "link_id"]]
    q = queue.copy()
    for c in ("window_id", "timestamp", "link_id"):
        q[c] = q[c].astype(str)
    C("queue no duplicate keys", not q.duplicated(["window_id", "timestamp", "link_id"]).any())
    mq = kq.merge(q, on=["window_id", "timestamp", "link_id"], how="left")
    C("queue rows == key rows", len(mq) == len(kq))
    C("queue no missing", mq.queue_pred.notna().all(), f"missing={int(mq.queue_pred.isna().sum())}")
    C("queue binary", mq.queue_pred.dropna().isin([0, 1]).all())
    widx = pd.concat([pd.read_csv(f) for f in REL.glob("task2/*/*/window_index.csv")])
    widx = widx[widx.split.isin(["validation", "private"])]
    per_w = mq.groupby("window_id").queue_pred.sum()
    C("every window has >=1 predicted queue cell", (per_w.reindex(widx.window_id).fillna(0) > 0).all(),
      f"empty windows={(per_w.reindex(widx.window_id).fillna(0) == 0).sum()}")
    ons = widx[widx.condition == "queue_onset"]
    mo = mq.merge(ons[["window_id", "forecast_end"]], on="window_id")
    C("onset predictions only at T+30", (mo[mo.queue_pred == 1].timestamp == mo[mo.queue_pred == 1].forecast_end).all())
    C("queue positive rate sane (<0.5)", mq.queue_pred.mean() < 0.5, f"{mq.queue_pred.mean():.4f}")
    idx = mq.submission_id.astype(np.int64).to_numpy() - 1
    sub.loc[idx, "queue_pred"] = mq.queue_pred.fillna(0).astype(int).to_numpy()

    # ---------------- odme
    ko = key[key.task == "odme"][["submission_id", "panel", "departure_time", "path_id", "origin_zone", "destination_zone"]]
    o = odme.copy()
    for c in ("panel", "departure_time", "path_id"):
        o[c] = o[c].astype(str)
    C("odme no duplicate keys", not o.duplicated(["panel", "departure_time", "path_id"]).any())
    mo = ko.merge(o[["panel", "departure_time", "path_id", "path_flow"]], on=["panel", "departure_time", "path_id"], how="left")
    C("odme rows == key rows", len(mo) == len(ko))
    C("odme no missing", mo.path_flow.notna().all(), f"missing={int(mo.path_flow.isna().sum())}")
    C("odme finite & >=0", np.isfinite(mo.path_flow).all() and (mo.path_flow >= 0).all())
    for p in PANELS:
        x = mo[mo.panel == p]
        C(f"odme {p} total > 0", x.path_flow.sum() > 0, f"sum={x.path_flow.sum():.0f}")
    idx = mo.submission_id.astype(np.int64).to_numpy() - 1
    sub.loc[idx, "path_flow"] = mo.path_flow.to_numpy()

    # ---------------- global
    C("no NaN anywhere", not sub.isna().any().any())
    C("row count == sample", len(sub) == len(sample))
    C("ids sequential", (sub.submission_id.values == np.arange(1, len(sub) + 1)).all())
    C("non-state rows have zero state cols", (sub.loc[sub.task != "state", ["speed_kmh", "flow_vph"]] == 0).all().all())
    C("non-queue rows have zero queue", (sub.loc[sub.task != "queue", "queue_pred"] == 0).all())
    C("non-odme rows have zero path_flow", (sub.loc[sub.task != "odme", "path_flow"] == 0).all())
    fails = C.failed
    print(f"{len(C.rows)} checks, {len(fails)} failed")
    if fails and not force:
        raise SystemExit(f"FAILED checks: {[f[0] for f in fails]}")
    out.parent.mkdir(parents=True, exist_ok=True)
    sub["queue_pred"] = sub.queue_pred.astype(int)
    sub.to_csv(out, index=False, float_format="%.4f")
    json.dump({"note": note, "checks": [(n, ok, i) for n, ok, i in C.rows]},
              open(out.with_suffix(".checks.json"), "w"), indent=1)
    print("wrote", out)
    return out
