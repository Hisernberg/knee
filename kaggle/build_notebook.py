#!/usr/bin/env python3
"""Build a self-contained, offline Kaggle inference notebook from src/kneemri.

The notebook embeds the package source (no internet, no pip), loads checkpoints from attached Kaggle
datasets, runs `kneemri.infer.run_inference`, and writes /kaggle/working/submission.csv.

Usage:
  python kaggle/build_notebook.py --variant kaggle/variants.yaml:v1 --out build/v1
  -> build/v1/rsna-knee-infer-v1.ipynb + build/v1/kernel-metadata.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat as nbf
import yaml

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "kneemri"


def embed_package_cell() -> str:
    files = {p.name: p.read_text() for p in sorted(SRC.glob("*.py"))}
    lines = ["# --- embedded package: kneemri (built from src/kneemri) ---", "import os, sys, json, pathlib",
             "PKG = pathlib.Path('/kaggle/working/pkg/kneemri'); PKG.mkdir(parents=True, exist_ok=True)",
             f"_FILES = json.loads({json.dumps(json.dumps(files))})",
             "for _name, _src in _FILES.items():", "    (PKG / _name).write_text(_src)",
             "sys.path.insert(0, str(PKG.parent))", "import kneemri; print('kneemri', kneemri.__version__, 'embedded')"]
    return "\n".join(lines)


def config_cell(variant: dict) -> str:
    return "\n".join([
        "# --- variant configuration (edit here) ---",
        f"VARIANT = json.loads({json.dumps(json.dumps(variant))})",
        "INPUT_ROOT = os.environ.get('KNEE_INPUT_ROOT', '/kaggle/input')",
        "OUT_CSV = os.environ.get('KNEE_OUT_CSV', '/kaggle/working/submission.csv')",
        "print(json.dumps(VARIANT, indent=1))",
    ])


RUN_CELL = '''# --- run inference ---
import glob, time, torch
from kneemri.infer import InferConfig, run_inference
from kneemri.schema import find_competition_root

t0 = time.time()
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
ckpts = []
for pat in VARIANT['checkpoints']:
    found = sorted(glob.glob(pat))
    assert found, f'no checkpoint matches {pat}'
    ckpts.extend(found)
weights = VARIANT.get('weights') or [1.0] * len(ckpts)
assert len(weights) == len(ckpts), 'weights/checkpoints length mismatch'
root = find_competition_root(INPUT_ROOT, 'test')
cfg = InferConfig(input_root=INPUT_ROOT, checkpoints=tuple(ckpts), weights=tuple(weights), out_csv=OUT_CSV,
                  tta_flip=bool(VARIANT.get('tta_flip', True)), blend=VARIANT.get('blend', 'prob'),
                  max_seconds=float(VARIANT.get('max_seconds', 8 * 3600)), num_workers=int(VARIANT.get('workers', 3)),
                  prior_csv=str(root / 'train.csv') if (root / 'train.csv').is_file() else None)
out = run_inference(cfg)
print('done in %.0fs' % (time.time() - t0))
'''

CHECK_CELL = '''# --- final contract check ---
import pandas as pd
from kneemri.schema import ID_COL, TARGETS, validate_submission, read_split
sub = pd.read_csv(OUT_CSV)
test_df, _ = read_split(root, 'test')
validate_submission(sub, test_df[ID_COL].astype(str).tolist())
print(sub.shape, list(sub.columns) == [ID_COL] + TARGETS)
print(sub.head())
'''


def build(variant_name: str, variant: dict, out_dir: Path, owner: str, kernel_prefix: str, weights_datasets: list[str],
          machine_shape: str = "NvidiaTeslaT4") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(f"# RSNA Knee Abnormality Detection - offline inference ({variant_name})\n"
                                 "Self-contained: embeds the `kneemri` package, loads trained checkpoints from attached "
                                 "datasets, writes `/kaggle/working/submission.csv`. Internet off."),
        nbf.v4.new_code_cell(embed_package_cell()),
        nbf.v4.new_code_cell(config_cell(variant)),
        nbf.v4.new_code_cell(RUN_CELL),
        nbf.v4.new_code_cell(CHECK_CELL),
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    slug = f"{kernel_prefix}-{variant_name}".lower().replace("_", "-")
    nb_path = out_dir / f"{slug}.ipynb"
    nbf.write(nb, nb_path)
    meta = {
        "id": f"{owner}/{slug}",
        "title": f"RSNA Knee infer {variant_name}",
        "code_file": nb_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": machine_shape,
        "dataset_sources": weights_datasets,
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "kernel_sources": [],
        "model_sources": [],
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    # validate every code cell compiles (a `from __future__` inside an indented block is only caught by compile())
    for c in nb.cells:
        if c.cell_type == "code":
            compile(c.source, "<cell>", "exec")
    return nb_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=str(REPO / "kaggle" / "variants.yaml"))
    ap.add_argument("--variant", required=True, help="variant name in variants.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--owner", default=None, help="Kaggle username (defaults to variants.yaml: owner)")
    a = ap.parse_args()
    spec = yaml.safe_load(Path(a.variants).read_text())
    owner = a.owner or spec["owner"]
    v = spec["variants"][a.variant]
    p = build(a.variant, v, Path(a.out), owner, spec.get("kernel_prefix", "rsna-knee-infer"),
              spec.get("weights_datasets", []), spec.get("machine_shape", "NvidiaTeslaT4"))
    print(p)


if __name__ == "__main__":
    main()
