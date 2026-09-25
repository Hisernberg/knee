"""Daily submission loop: guarded Kaggle submissions, diffs, status and rank.

The runbook is docs/traffic/LOOP.md. Every submission goes through `submit`, which
refuses:
- a file whose checks failed (unless --probe);
- a duplicate;
- a sixth submission in the UTC day;
- a submission after 23:45 UTC;
- a submission while another one is still scoring.

    python -m trafficflow.loop status
    python -m trafficflow.loop pack OUT.csv          # zip a built CSV (checks.json stays next to it), delete the CSV
    python -m trafficflow.loop diff A.zip B.zip      # changed rows per task
    python -m trafficflow.loop submit X.zip -m MSG [--probe] [--dry-run]
    python -m trafficflow.loop rank
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

COMP = "2026-ieee-big-data-traffic-flow-bench"
DAILY = 5
LAST_SUBMIT = dt.time(23, 45)
DEADLINE = dt.datetime(2026, 11, 7, 6, 55)
SUBS = Path("/home/user/work/subs")
LB = Path("/home/user/research/lb")
REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "docs" / "traffic" / "lb_log.csv"
CRITICAL = ["/home/user/data/kaggle_public/submission_key.csv", "/home/user/cache",
            "/home/user/work/t1/pred/state_full3.parquet", "/home/user/work/t2/lgb_v6.csv",
            "/home/user/work/t2h/probs_v6_onset.parquet", "/home/user/work/t4/t4_l2proj.csv"]
TASK_COLS = {"state": ["speed_kmh", "flow_vph"], "queue": ["queue_pred"], "odme": ["path_flow"]}


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def creds() -> str:
    if (Path.home() / ".kaggle" / "kaggle.json").exists():
        return "~/.kaggle/kaggle.json"
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return "env KAGGLE_USERNAME/KAGGLE_KEY"
    return ""


def api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    a = KaggleApi()
    a.authenticate()
    return a


def _date(x) -> dt.datetime:
    if isinstance(x, dt.datetime):
        return x.replace(tzinfo=None)
    return pd.Timestamp(str(x)).to_pydatetime().replace(tzinfo=None)


def submissions(a) -> list[dict]:
    out = []
    for s in a.competition_submissions(COMP, page_size=200) or []:
        if s is None:
            continue
        pub = s.public_score
        out.append(dict(ref=s.ref, file=s.file_name, date=_date(s.date), status=str(s.status).split(".")[-1],
                        public=float(pub) if pub not in (None, "") else None, desc=s.description or "",
                        error=getattr(s, "error_description", "") or "", team=s.team_name))
    return sorted(out, key=lambda d: d["date"])


def today(subs: list[dict], now: dt.datetime | None = None) -> list[dict]:
    d0 = (now or utcnow()).replace(hour=0, minute=0, second=0, microsecond=0)
    return [s for s in subs if s["date"] >= d0]


def best(subs: list[dict]) -> dict | None:
    done = [s for s in subs if s["status"] == "COMPLETE" and s["public"] is not None]
    return max(done, key=lambda s: s["public"]) if done else None


# ---------------------------------------------------------------------------- commands
def status():
    now = utcnow()
    info = {"utc": now.strftime("%Y-%m-%d %H:%M"), "credentials": creds() or "MISSING",
            "disk_free_gb": round(shutil.disk_usage("/home/user").free / 1e9, 1),
            "days_to_deadline": round((DEADLINE - now).total_seconds() / 86400, 1),
            "missing_files": [p for p in CRITICAL if not Path(p).exists()]}
    if info["credentials"] != "MISSING":
        subs = submissions(api())
        t = today(subs, now)
        b = best(subs)
        info.update(used_today=len(t), left_today=DAILY - len(t),
                    pending=[s["file"] for s in subs if s["status"] not in ("COMPLETE", "ERROR")],
                    best={"file": b["file"], "public": b["public"], "date": str(b["date"])[:16]} if b else None,
                    today=[{"file": s["file"], "status": s["status"], "public": s["public"]} for s in t])
    print(json.dumps(info, indent=1))
    return info


def pack(csv_path: str) -> Path:
    src = Path(csv_path)
    ck = src.with_suffix(".checks.json")
    if not ck.exists():
        raise SystemExit(f"no checks file next to {src}")
    z = src.with_suffix(".zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
        f.write(src, src.name)
    src.unlink()
    print("packed", z, f"{z.stat().st_size / 1e6:.1f} MB")
    return z


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as f:
            name = [n for n in f.namelist() if n.endswith(".csv")][0]
            with f.open(name) as h:
                return pd.read_csv(h)
    return pd.read_csv(path)


def diff(a: str, b: str) -> dict:
    A, B = _read(Path(a)), _read(Path(b))
    assert (A.submission_id.values == B.submission_id.values).all() and (A.task.values == B.task.values).all()
    out = {}
    for task, cols in TASK_COLS.items():
        m = (A.task == task).to_numpy()
        d = np.zeros(m.sum(), bool)
        mad = {}
        for c in cols:
            x, y = A.loc[m, c].to_numpy(float), B.loc[m, c].to_numpy(float)
            d |= np.abs(x - y) > 1e-6
            mad[c] = float(np.abs(x - y).mean())
        out[task] = {"rows": int(m.sum()), "changed": int(d.sum()), "mean_abs_diff": mad}
    print(json.dumps(out, indent=1))
    return out


def _log(row: dict):
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def submit(path: str, msg: str, probe: bool = False, dry: bool = False, timeout_min: int = 45):
    z = Path(path)
    now = utcnow()
    errors = []
    if not z.exists() or z.suffix != ".zip":
        errors.append(f"{z} is not an existing .zip")
    ck = z.with_suffix(".checks.json")
    if not ck.exists():
        errors.append("no checks.json")
    else:
        failed = [c[0] for c in json.load(open(ck))["checks"] if not c[1]]
        if failed and not probe:
            errors.append(f"failed checks {failed} (a deliberate probe needs --probe)")
    if now.time() >= LAST_SUBMIT or now >= DEADLINE:
        errors.append(f"too late ({now:%H:%M} UTC)")
    a = api()
    subs = submissions(a)
    if len(today(subs, now)) >= DAILY:
        errors.append("daily quota used")
    pend = [s["file"] for s in subs if s["status"] not in ("COMPLETE", "ERROR")]
    if pend:
        errors.append(f"still scoring: {pend}")
    if any(s["file"] == z.name for s in subs):
        errors.append("already submitted")
    b = best(subs)
    print(json.dumps({"file": z.name, "kind": "probe" if probe else "candidate", "used_today": len(today(subs, now)),
                      "best_before": b and {"file": b["file"], "public": b["public"]}, "refused": errors}, indent=1))
    if errors or dry:
        raise SystemExit(1 if errors else 0)
    t0 = time.time()
    a.competition_submit(file_name=str(z), message=msg, competition=COMP)
    res = None
    while time.time() - t0 < timeout_min * 60:
        time.sleep(20)
        mine = [s for s in submissions(a) if s["file"] == z.name]
        if mine and mine[-1]["status"] in ("COMPLETE", "ERROR"):
            res = mine[-1]
            break
    if res is None:
        raise SystemExit(f"not scored after {timeout_min} min; check `status` later")
    delta = None if res["public"] is None or b is None else round(res["public"] - b["public"], 5)
    _log(dict(utc=res["date"].strftime("%Y-%m-%d %H:%M"), file=z.name, kind="probe" if probe else "candidate",
              status=res["status"], public=res["public"], best_before=b and b["public"], delta=delta, message=msg))
    print(json.dumps({"file": z.name, "status": res["status"], "public": res["public"], "delta_vs_best": delta,
                      "error": res["error"], "minutes": round((time.time() - t0) / 60, 1)}, indent=1))
    if res["status"] == "ERROR":
        raise SystemExit(2)
    return res


def rank():
    a = api()
    subs = submissions(a)
    team = subs[-1]["team"] if subs else None
    a.competition_leaderboard_download(COMP, path=str(LB))
    with zipfile.ZipFile(LB / f"{COMP}.zip") as f:
        name = f.namelist()[0]
        with f.open(name) as h:
            lb = pd.read_csv(h)
    stamp = utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    lb.to_csv(LB / f"{COMP}-publicleaderboard-{stamp}.csv", index=False)
    era = pd.read_csv(LB / "lb_with_era.csv")
    old = era.set_index("TeamId")[["Score", "era"]]
    lb["era"] = [old.era[t] if t in old.index and abs(old.Score[t] - s) < 1e-9 else "post"
                 for t, s in zip(lb.TeamId, lb.Score)]
    post = lb[lb.era == "post"].sort_values("Score", ascending=False).reset_index(drop=True)
    me = lb[lb.TeamName == team]
    out = {"team": team, "teams": len(lb), "post_rebuild_teams": len(post)}
    if len(me):
        s = float(me.Score.iloc[0])
        out.update(score=s, rank_overall=int((lb.Score > s).sum() + 1), rank_post=int((post.Score > s).sum() + 1),
                   gap_to_post_1=round(float(post.Score.iloc[0]) - s, 5))
    out["post_top"] = [f"{r.TeamName}: {r.Score:.5f}" for r in post.head(10).itertuples()]
    # carry the old dating columns only where the score has not changed
    same = [t in old.index and abs(old.Score[t] - s) < 1e-9 for t, s in zip(lb.TeamId, lb.Score)]
    upd = lb.copy()
    for c in era.columns:
        if c not in upd.columns:
            upd[c] = upd.TeamId.map(era.set_index("TeamId")[c]).where(same)
    upd.to_csv(LB / "lb_with_era.csv", index=False)
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "pack", "diff", "submit", "rank"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("-m", "--message", default="")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.cmd == "status":
        status()
    elif a.cmd == "pack":
        pack(a.files[0])
    elif a.cmd == "diff":
        diff(a.files[0], a.files[1])
    elif a.cmd == "submit":
        if not a.message:
            sys.exit("a message (-m) is required")
        submit(a.files[0], a.message, a.probe, a.dry_run)
    elif a.cmd == "rank":
        rank()
