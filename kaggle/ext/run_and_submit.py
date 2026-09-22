#!/usr/bin/env python3
"""Push the extended-Raptor notebook, wait for the run, submit the variant CSVs, and collect public scores.

  python kaggle/ext/run_and_submit.py --build-dir build/ext_v1 [--variants anchor,main,c3,c4,c5] [--max-submits 5]
  python kaggle/ext/run_and_submit.py --build-dir build/ext_v1 --submit-only --version 1      # run already finished

Code competition: each submission is (notebook version, output file). One GPU run therefore yields up to five
distinct submissions (submission_anchor.csv, submission_main.csv, submission_c3/c4/c5.csv) at no extra GPU cost.
Needs Kaggle credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY). Never prints the key.
Exit codes: 0 ok, 2 quota/blocked (retry later), 3 run error.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

COMP = "rsna-knee-abnormality-detection"
FILES = {"anchor": "submission_anchor.csv", "main": "submission_main.csv", "c3": "submission_c3.csv",
         "c4": "submission_c4.csv", "c5": "submission_c5.csv"}
LOG = Path(__file__).resolve().parent / "run_log.jsonl"


def log(**kw):
    kw["ts"] = datetime.now(timezone.utc).isoformat()
    with LOG.open("a") as f:
        f.write(json.dumps(kw, default=str) + "\n")
    print(json.dumps(kw, default=str), flush=True)


def run(cmd: list[str], retries: int = 4) -> subprocess.CompletedProcess:
    last = None
    for attempt in range(retries + 1):
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode == 0:
            return p
        last = p
        out = (p.stdout or "") + (p.stderr or "")
        if not re.search(r"(timed? ?out|connection|reset|502|503|504|429|temporar)", out, re.I) or attempt == retries:
            break
        time.sleep([2, 4, 8, 16][min(attempt, 3)])
    return last


def kernel_ref(build_dir: Path) -> str:
    return json.loads((build_dir / "kernel-metadata.json").read_text())["id"]


def push(build_dir: Path) -> int:
    p = run(["kaggle", "kernels", "push", "-p", str(build_dir)])
    out = (p.stdout or "") + (p.stderr or "")
    if "quota" in out.lower():
        log(event="push_blocked", reason=out.strip()[-300:])
        sys.exit(2)
    if p.returncode != 0:
        log(event="push_failed", output=out.strip()[-800:])
        sys.exit(3)
    m = re.search(r"[Vv]ersion\s+(\d+)", out)
    version = int(m.group(1)) if m else -1
    log(event="pushed", output=out.strip()[-300:], version=version)
    return version


def status(ref: str) -> str:
    p = run(["kaggle", "kernels", "status", ref])
    out = (p.stdout or "") + (p.stderr or "")
    m = re.search(r'status\s+"?([A-Za-z_.]+)"?', out)
    return (m.group(1) if m else out.strip()).lower()


def wait(ref: str, poll: int, timeout_s: float) -> str:
    t0 = time.time()
    while True:
        st = status(ref)
        if any(k in st for k in ("complete", "error", "cancel")):
            log(event="kernel_finished", status=st, minutes=round((time.time() - t0) / 60, 1))
            return st
        if time.time() - t0 > timeout_s:
            log(event="kernel_timeout", status=st)
            return "timeout"
        print(f"[wait] {ref}: {st} ({(time.time() - t0) / 60:.0f} min)", flush=True)
        time.sleep(poll)


def submit(ref: str, version: int, name: str, message: str) -> str:
    cmd = ["kaggle", "competitions", "submit", "-c", COMP, "-k", ref, "-f", FILES[name], "-m", message]
    if version > 0:
        cmd += ["-v", str(version)]
    p = run(cmd)
    out = ((p.stdout or "") + (p.stderr or "")).strip()
    log(event="submitted" if p.returncode == 0 else "submit_failed", variant=name, message=message, output=out[-400:])
    return out


def submissions() -> list[dict]:
    p = run(["kaggle", "competitions", "submissions", "-c", COMP, "-v"])
    return list(csv.DictReader(io.StringIO(p.stdout or "")))


def wait_scores(messages: list[str], timeout_s: float = 4 * 3600, poll: int = 120) -> dict:
    t0 = time.time()
    scores: dict[str, str] = {}
    while time.time() - t0 < timeout_s and len(scores) < len(messages):
        for row in submissions():
            desc = row.get("description", "")
            if desc in messages and desc not in scores:
                if row.get("publicScore") or "error" in str(row.get("status", "")).lower():
                    scores[desc] = row.get("publicScore") or f"no score ({row.get('status')})"
        if len(scores) < len(messages):
            time.sleep(poll)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--variants", default="main,anchor,c3,c4,c5", help="submission order (daily limit is 5)")
    ap.add_argument("--max-submits", type=int, default=5)
    ap.add_argument("--submit-only", action="store_true", help="skip push/wait; submit the given --version")
    ap.add_argument("--version", type=int, default=-1)
    ap.add_argument("--poll", type=int, default=300)
    ap.add_argument("--no-submit", action="store_true")
    a = ap.parse_args()
    build_dir = Path(a.build_dir)
    ref = kernel_ref(build_dir)
    version = a.version
    if not a.submit_only:
        version = push(build_dir)
        st = wait(ref, a.poll, 9.9 * 3600)
        if "complete" not in st:
            print(f"run did not complete ({st}); inspect with: kaggle kernels output {ref} -p out && kaggle kernels logs {ref}")
            sys.exit(3)
    if a.no_submit:
        return
    names = [n for n in a.variants.split(",") if n in FILES][: a.max_submits]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    messages = []
    for name in names:
        msg = f"ext_v1 {name} v{version} {stamp}"
        out = submit(ref, version, name, msg)
        if re.search(r"limit|maximum|quota", out, re.I):
            print("daily submission limit reached; remaining variants can be submitted tomorrow with --submit-only")
            break
        messages.append(msg)
    scores = wait_scores(messages)
    for msg in messages:
        log(event="scored", message=msg, public_score=scores.get(msg, "pending"))
    print(json.dumps({m: scores.get(m, "pending") for m in messages}, indent=1))


if __name__ == "__main__":
    main()
