"""Flatten the umud package into one Kaggle script kernel and write its metadata.

Training kernel (GPU; writes apo.pt / fasc.pt / test probability maps):
    python kaggle/build_kernel.py --user <kaggle user> --slug umud-seg-train --out build/train
End-to-end submission kernel (loads the training kernel's weights, writes submission.csv):
    python kaggle/build_kernel.py --user <kaggle user> --slug umud-pipeline-submission --mode submit \
        --weights-kernel <kaggle user>/umud-seg-train --out build/submit
    kaggle kernels push -p build/submit
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SUBMIT_TAIL = """
# ---- end-to-end submission ----
import glob as _glob
import json as _json
_W = sorted(_glob.glob('/kaggle/input/**/apo.pt', recursive=True))
print('weights', _W)
_WD = Path(_W[0]).parent
_DATA = Path(sorted(_glob.glob('/kaggle/input/**/test_images_v2', recursive=True))[0]).parent
_OUT = Path('/kaggle/working')
_dev = 'cuda' if torch.cuda.is_available() else 'cpu'
infer_test(_DATA, _OUT, _WD, _dev)
_feat = extract_features(_OUT)
_feat.to_csv(_OUT / 'features.csv', index=False)
_cfg = _json.loads(__CFG__)
_groups = video_groups(list(_feat.image_id), _DATA / 'test_images_v2' / 'test_set_v2')
_sub = make_submission(_feat, _cfg, _groups)
_sub.to_csv(_OUT / 'submission.csv', index=False)
print(_sub.describe())
"""


def strip(src: str) -> str:
    src = src.replace("from __future__ import annotations\n", "")
    return src.replace('if __name__ == "__main__":\n    main()\n', "")


ap = argparse.ArgumentParser()
ap.add_argument("--user", required=True)
ap.add_argument("--slug", default="umud-seg-train")
ap.add_argument("--out", default=str(ROOT / "build" / "kernel"))
ap.add_argument("--args", default="", help="extra CLI args appended to sys.argv of seg.py (train mode)")
ap.add_argument("--mode", choices=["train", "submit"], default="train")
ap.add_argument("--weights-kernel", default=None, help="<user>/<slug> of the training kernel (submit mode)")
ap.add_argument("--config", default=str(ROOT / "configs" / "a_raw.json"))
ap.add_argument("--cpu", action="store_true", help="no GPU (e.g. weekly quota used up)")
a = ap.parse_args()

scale = (ROOT / "umud" / "scale.py").read_text()
seg = (ROOT / "umud" / "seg.py").read_text()
seg = re.sub(r"sys\.path\.insert.*?detect_scale = None\n", "", seg, flags=re.S)
head = (
    "import subprocess, sys\n"
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch'], check=False)\n"
    f"sys.argv += {a.args.split()!r}\n"
)
body = head + "\n# ---- scale.py ----\n" + strip(scale) + "\n# ---- seg.py ----\n"
if a.mode == "train":
    body += seg.replace("from __future__ import annotations\n", "")
else:
    geo = strip((ROOT / "umud" / "geometry.py").read_text())
    pred = strip((ROOT / "umud" / "predict.py").read_text()).replace("from umud.geometry import analyse\n", "")
    cfg = json.dumps(Path(a.config).read_text())
    body += strip(seg) + "\n# ---- geometry.py ----\n" + geo + "\n# ---- predict.py ----\n" + pred
    body += SUBMIT_TAIL.replace("__CFG__", cfg)

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
(out / "kernel.py").write_text(body)
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{a.user}/{a.slug}", "title": a.slug, "code_file": "kernel.py", "language": "python",
    "kernel_type": "script", "is_private": True, "enable_gpu": a.mode == "train" and not a.cpu, "enable_internet": True,
    "competition_sources": ["umud-challenge-muscle-architecture-in-ultrasound-data"],
    "dataset_sources": [],
    "kernel_sources": [a.weights_kernel] if a.weights_kernel else [],
}, indent=1))
print("wrote", out)
