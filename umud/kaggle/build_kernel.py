"""Flatten umud/scale.py + umud/seg.py into one Kaggle script kernel and write its metadata.

    python kaggle/build_kernel.py --user <kaggle username> --slug umud-seg-train --out build/kernel
    kaggle kernels push -p build/kernel
"""
import argparse, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--user", required=True)
ap.add_argument("--slug", default="umud-seg-train")
ap.add_argument("--out", default=str(ROOT / "build" / "kernel"))
ap.add_argument("--args", default="", help="extra CLI args appended to sys.argv of seg.py")
a = ap.parse_args()

scale = (ROOT / "umud" / "scale.py").read_text()
seg = (ROOT / "umud" / "seg.py").read_text()
seg = re.sub(r"sys\.path\.insert.*?detect_scale = None\n", "", seg, flags=re.S)
head = (
    "import subprocess, sys\n"
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch'], check=False)\n"
    f"sys.argv += {a.args.split()!r}\n"
)
body = head + "\n# ---- scale.py ----\n" + scale.replace("from __future__ import annotations\n", "") + \
    "\n# ---- seg.py ----\n" + seg.replace("from __future__ import annotations\n", "")
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
(out / "kernel.py").write_text(body)
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{a.user}/{a.slug}", "title": a.slug, "code_file": "kernel.py", "language": "python",
    "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
    "competition_sources": ["umud-challenge-muscle-architecture-in-ultrasound-data"],
    "dataset_sources": [], "kernel_sources": [],
}, indent=1))
print("wrote", out)
