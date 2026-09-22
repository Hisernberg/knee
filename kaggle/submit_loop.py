#!/usr/bin/env python3
"""Push N inference notebooks to Kaggle, wait for them, submit each finished version, collect scores.

This is a CODE competition: a submission is a completed notebook version, not an uploaded CSV. The loop:
  build notebook (kaggle/build_notebook.py) -> `kaggle kernels push` -> poll `kaggle kernels status`
  -> `kaggle competitions submit -k <owner>/<kernel> -v <version> -f submission.csv` -> poll submissions for score.

Auth: export KAGGLE_API_TOKEN=... (or ~/.kaggle/kaggle.json). The token is never printed or logged.
Needs network access to www.kaggle.com / api.kaggle.com (blocked in some sandboxes).

Examples:
  python kaggle/submit_loop.py --owner me --select v1_arm_a_5fold,v2_arm_a_b_rank --dry-run
  python kaggle/submit_loop.py --owner me --all --max-concurrent 2
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebook import build  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
COMP = "rsna-knee-abnormality-detection"
LOG = REPO / "kaggle" / "submission_log.jsonl"
BACKOFF = [2, 4, 8, 16]


def log_event(**kw):
    kw["ts"] = datetime.now(timezone.utc).isoformat()
    with LOG.open("a") as f:
        f.write(json.dumps(kw) + "\n")


def run(cmd: list[str], retries: int = 4, dry: bool = False) -> subprocess.CompletedProcess:
    """Run a kaggle CLI command with exponential backoff on network-type failures."""
    if dry:
        print("[dry-run]", " ".join(cmd))
        return subprocess.CompletedProcess(cmd, 0, "Kernel version 1 successfully pushed", "")
    last = None
    for attempt in range(retries + 1):
        p = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
        out = (p.stdout or "") + (p.stderr or "")
        if p.returncode == 0:
            return p
        last = p
        transient = re.search(r"(timed? ?out|connection|reset|502|503|504|429|temporar|proxy)", out, re.I)
        if not transient or attempt == retries:
            break
        wait = BACKOFF[min(attempt, len(BACKOFF) - 1)]
        print(f"[retry] {cmd[1:3]} failed (attempt {attempt + 1}); waiting {wait}s", flush=True)
        time.sleep(wait)
    print(last.stdout[-2000:] if last and last.stdout else "", last.stderr[-2000:] if last and last.stderr else "",
          file=sys.stderr)
    raise RuntimeError(f"command failed: {' '.join(cmd[:3])}")


def push_kernel(folder: Path, dry: bool) -> int:
    p = run(["kaggle", "kernels", "push", "-p", str(folder)], dry=dry)
    m = re.search(r"[Vv]ersion\s+(\d+)", p.stdout or "")
    if not m:
        raise RuntimeError(f"could not parse kernel version from push output:\n{p.stdout}")
    return int(m.group(1))


def kernel_status(ref: str, dry: bool) -> str:
    if dry:
        return "complete"
    p = run(["kaggle", "kernels", "status", ref], dry=dry)
    m = re.search(r'status\s+"?([A-Za-z]+)"?', p.stdout or "")
    return (m.group(1) if m else (p.stdout or "").strip()).lower()


def wait_for_kernel(ref: str, dry: bool, poll: int = 90, timeout_s: float = 9.75 * 3600) -> str:
    t0 = time.time()
    while True:
        st = kernel_status(ref, dry)
        if st in {"complete", "error", "cancelled", "cancelacknowledged"}:
            return st
        if time.time() - t0 > timeout_s:
            return "timeout"
        print(f"[wait] {ref}: {st} ({(time.time() - t0) / 60:.0f} min)", flush=True)
        time.sleep(poll)


def submit_kernel(ref: str, version: int, message: str, dry: bool) -> str:
    p = run(["kaggle", "competitions", "submit", "-c", COMP, "-k", ref, "-v", str(version), "-f", "submission.csv",
             "-m", message], dry=dry)
    return (p.stdout or "").strip()


def latest_submissions(dry: bool) -> list[dict]:
    if dry:
        return []
    p = run(["kaggle", "competitions", "submissions", "-c", COMP, "-v"], dry=dry)
    rows = list(csv.DictReader(io.StringIO(p.stdout or "")))
    return rows


def wait_for_score(message: str, dry: bool, poll: int = 60, timeout_s: float = 3600) -> dict | None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        for r in latest_submissions(dry):
            if r.get("description") == message:
                if str(r.get("status", "")).lower() == "complete" or r.get("publicScore"):
                    return r
                break
        if dry:
            return {"publicScore": "n/a (dry-run)", "status": "complete"}
        time.sleep(poll)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=str(REPO / "kaggle" / "variants.yaml"))
    ap.add_argument("--owner", default=None)
    ap.add_argument("--select", default=None, help="comma-separated variant names (default: all)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--build-dir", default=str(REPO / "build"))
    ap.add_argument("--max-concurrent", type=int, default=2, help="kernels running at once (Kaggle GPU quota)")
    ap.add_argument("--max-per-day", type=int, default=5, help="competition daily submission limit")
    ap.add_argument("--no-submit", action="store_true", help="push and wait only")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    spec = yaml.safe_load(Path(a.variants).read_text())
    owner = a.owner or spec["owner"]
    if owner == "YOUR_KAGGLE_USERNAME":
        sys.exit("set --owner or edit kaggle/variants.yaml: owner")
    names = list(spec["variants"]) if (a.all or not a.select) else a.select.split(",")
    names = names[: a.max_per_day]
    if not a.dry_run and not (os.environ.get("KAGGLE_API_TOKEN") or (Path.home() / ".kaggle" / "kaggle.json").exists()):
        sys.exit("no Kaggle credentials: export KAGGLE_API_TOKEN=... or create ~/.kaggle/kaggle.json")

    # 1) build + push (bounded concurrency)
    pending, results = [], {}
    for name in names:
        folder = Path(a.build_dir) / name
        build(name, spec["variants"][name], folder, owner, spec.get("kernel_prefix", "rsna-knee-infer"),
              spec.get("weights_datasets", []), spec.get("machine_shape", "NvidiaTeslaT4"))
        meta = json.loads((folder / "kernel-metadata.json").read_text())
        pending.append((name, folder, meta["id"]))
    running: list[tuple[str, str, int]] = []
    while pending or running:
        while pending and len(running) < a.max_concurrent:
            name, folder, ref = pending.pop(0)
            try:
                version = push_kernel(folder, a.dry_run)
            except Exception as e:  # noqa: BLE001
                results[name] = {"status": "push_failed", "error": str(e)}
                log_event(variant=name, event="push_failed", error=str(e))
                continue
            print(f"[push] {ref} version {version}", flush=True)
            log_event(variant=name, event="pushed", kernel=ref, version=version)
            running.append((name, ref, version))
        still = []
        for name, ref, version in running:
            st = kernel_status(ref, a.dry_run)
            if st in {"complete", "error", "cancelled", "timeout"}:
                results[name] = {"status": st, "kernel": ref, "version": version}
                log_event(variant=name, event="kernel_finished", kernel=ref, version=version, status=st)
                print(f"[done] {ref} v{version}: {st}", flush=True)
            else:
                still.append((name, ref, version))
        running = still
        if running:
            time.sleep(0 if a.dry_run else 90)

    # 2) submit every completed version, then collect scores
    if a.no_submit:
        print(json.dumps(results, indent=1))
        return
    n_sub = 0
    for name in names:
        r = results.get(name, {})
        if r.get("status") != "complete":
            print(f"[skip] {name}: {r}", flush=True)
            continue
        if n_sub >= a.max_per_day:
            print("[stop] daily submission limit reached", flush=True)
            break
        msg = f"kneemri {name} v{r['version']} {datetime.now(timezone.utc):%Y%m%d-%H%M}"
        try:
            out = submit_kernel(r["kernel"], r["version"], msg, a.dry_run)
        except Exception as e:  # noqa: BLE001
            r["submit_error"] = str(e)
            log_event(variant=name, event="submit_failed", error=str(e))
            if re.search(r"limit|maximum", str(e), re.I):
                break
            continue
        n_sub += 1
        log_event(variant=name, event="submitted", message=msg, response=out[:300])
        score = wait_for_score(msg, a.dry_run)
        r["public_score"] = (score or {}).get("publicScore")
        r["submission_status"] = (score or {}).get("status")
        log_event(variant=name, event="scored", public_score=r["public_score"], status=r["submission_status"])
        print(f"[score] {name}: public LB = {r['public_score']}", flush=True)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
