#!/usr/bin/env python3
"""Package trained checkpoints (work/runs/<arm>/fold*/best_filament_unet.pth) into a Kaggle dataset.

  python kaggle/upload_weights.py --owner me --slug rsna-knee-kneemri-weights --runs work/runs [--update]
Creates a staging folder with only the .pth files (+ oof.csv, config.json for provenance) and calls
`kaggle datasets create` (or `version` with --update). Requires Kaggle network access.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--slug", default="rsna-knee-kneemri-weights")
    ap.add_argument("--runs", default="work/runs")
    ap.add_argument("--staging", default="work/upload")
    ap.add_argument("--ckpt-name", default="best_filament_unet.pth")
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    staging = Path(a.staging)
    if staging.exists():
        shutil.rmtree(staging)
    n = 0
    for ck in sorted(Path(a.runs).glob(f"*/fold*/{a.ckpt_name}")):
        dst = staging / ck.parent.parent.name / ck.parent.name
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ck, dst / ck.name)
        for extra in ("oof.csv", "config.json"):
            if (ck.parent / extra).is_file():
                shutil.copy2(ck.parent / extra, dst / extra)
        n += 1
    if n == 0:
        raise SystemExit(f"no checkpoints named {a.ckpt_name} under {a.runs}")
    (staging / "dataset-metadata.json").write_text(json.dumps({
        "title": "RSNA Knee kneemri weights", "id": f"{a.owner}/{a.slug}", "licenses": [{"name": "CC0-1.0"}]}, indent=2))
    cmd = ["kaggle", "datasets", "version" if a.update else "create", "-p", str(staging), "--dir-mode", "zip"]
    if a.update:
        cmd += ["-m", f"update {n} checkpoints"]
    print(f"staged {n} checkpoints ->", " ".join(cmd))
    if not a.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
