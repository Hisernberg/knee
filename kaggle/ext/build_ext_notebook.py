#!/usr/bin/env python3
"""Build the 'Speedy Raptors + extra Raptor views' submission notebook from Mattia Angeli's public notebook.

The public 0.943 graph is kept byte-identical except for three surgical patches:
  1. ASSET_ROOTS gains the two extra checkpoint datasets (finespacing v9, widedense v4).
  2. The Raptor worker runs two extra views with weight 0.0 (so the anchor blend is unchanged) on the
     otherwise idle time of both GPUs; a failure of an extra view is isolated and logged, never fatal.
  3. A final CPU cell (kaggle/ext/variants_cell.py) rebuilds the anchor from its saved parts (parity check),
     then writes submission_{anchor,main,c3,c4,c5}.csv and promotes MAIN to submission.csv.

Usage:
  kaggle kernels pull mattiaangeli/bend-the-knee-to-speedy-raptors-the-original -p /path/src -m
  python kaggle/ext/build_ext_notebook.py --source /path/src/bend-the-knee-to-speedy-raptors-the-original.ipynb \
         --out build/ext_v1 --owner kragglenote2forwork --slug rsna-knee-raptor-ext-v1
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

EXTRA_DATASETS = ["dreaddevelopment/raptor-knee-finespacing", "dreaddevelopment/raptor-knee-widedense"]
EXTRA_ROOTS = ["/kaggle/input/datasets/dreaddevelopment/raptor-knee-finespacing", "/kaggle/input/raptor-knee-finespacing",
               "/kaggle/input/datasets/dreaddevelopment/raptor-knee-widedense", "/kaggle/input/raptor-knee-widedense"]

ASSET_TAIL = "'/kaggle/input/models/metaresearch/dinov2/pytorch/small/1', '/kaggle/input/dinov2/pytorch/small/1']"
EXEC_LINE = "exec(compile(_KE_SRC, '<raptor>', 'exec'), _KE_NS)\n"
ARMS_EXT = EXEC_LINE + '''# --- [ext] two additional public Raptor views. Weight 0.0 keeps the anchor blend byte-identical; the
# --- variants cell re-blends them from raptor_raw.npz. Geometry per the checkpoint datasets' own descriptions.
_EXT_SLOTS80 = [("Sagittal", 1, 22), ("Sagittal", 0, 18), ("Coronal", 1, 15), ("Coronal", 0, 10), ("Axial", -1, 15)]
_EXT_SLOTS64 = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12), ("Coronal", 0, 8), ("Axial", -1, 12)]
_KE_NS['ARMS'].extend([
    {"name": "finespacing-v9", "file": "raptor_ft_coatnet_v9_full.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
     "res": 384, "img": 336, "slots": _EXT_SLOTS80, "span": (0.02, 0.98), "k_eval": 78, "reverse": False, "w": 0.0},
    {"name": "widedense-v4", "file": "raptor_ft_coatnet_v4_full.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
     "res": 384, "img": 336, "slots": _EXT_SLOTS64, "span": (0.06, 0.94), "k_eval": 62, "reverse": False, "w": 0.0},
])
'''
GPU_OLD = '''    def gpu_zero():
        with torch.cuda.device(0):
            run_shared_maxspan(torch.device('cuda:0'))

    def gpu_one():
        with torch.cuda.device(1):
            run_single(1, torch.device('cuda:1'))
            run_single(3, torch.device('cuda:1'))
'''
GPU_NEW = '''    def _run_extra(arm_index, device):
        # [ext] an extra view may fail without touching the anchor (its weight is 0.0).
        try:
            run_single(arm_index, device)
        except Exception as error:
            rsna_event('raptor_extra_arm_failed', arm=str(arms[arm_index]['name']),
                       error=f'{type(error).__name__}: {str(error)[:500]}')
            print(f"[raptor-ext] extra view {arms[arm_index]['name']} FAILED (anchor unaffected): "
                  f"{type(error).__name__}: {str(error)[:300]}", flush=True)
            _EXT_FAILED_ARMS.append(str(arms[arm_index]['name']))

    def gpu_zero():
        with torch.cuda.device(0):
            run_shared_maxspan(torch.device('cuda:0'))
            for _extra in range(4, len(arms)):
                if _extra % 2 == 0:
                    _run_extra(_extra, torch.device('cuda:0'))

    def gpu_one():
        with torch.cuda.device(1):
            run_single(1, torch.device('cuda:1'))
            run_single(3, torch.device('cuda:1'))
            for _extra in range(4, len(arms)):
                if _extra % 2 == 1:
                    _run_extra(_extra, torch.device('cuda:1'))
'''
IDS_OLD = "_ke_input_ids = set()\n"
IDS_NEW = "_ke_input_ids = set()\n_EXT_FAILED_ARMS = []\n"
PRINT_OLD = '''        f"[raptor-fast] {len(test_ids)} studies; balanced arm groups "
        f"cuda:0=[0,2], cuda:1=[1,3]",
'''
PRINT_NEW = '''        f"[raptor-fast] {len(test_ids)} studies; balanced arm groups "
        f"cuda:0=[0,2,{list(range(4, len(arms), 2))}], cuda:1=[1,3,{list(range(5, len(arms), 2))}]",
'''
PREP_OLD = "for name in ('maxspan-v5','native384dense-v10','native384-v8')"
PREP_NEW = "for name in [_a['name'] for _a in _KE_NS['ARMS'] if not _a.get('reverse') and _a['name'] not in _EXT_FAILED_ARMS]"

HEADER_MD = """# RSNA Knee | Speedy Raptors + two extra public Raptor views

Fork of **Mattia Angeli's** public notebook *Bend the Knee to Speedy Raptors - The Original* (public LB 0.943).
The whole 0.943 graph runs unchanged and its output is kept as `submission_anchor.csv`.

What is added (inference only, no training, no test-label use):

* two more public **dreaddevelopment** CoAtNet "Raptor" checkpoints are scored in the same Raptor worker, at their
  own training geometry: `raptor-knee-finespacing` (v9, 80-slice/78-window corpus, 0.932 public solo) and
  `raptor-knee-widedense` (v4, 64-slice/6-94% corpus, 0.927 public solo at 62 windows). Their weight in the
  anchor blend is 0.0, so the anchor stays byte-identical;
* a final CPU cell re-blends the saved raw probabilities into `submission_main.csv` (extra views inside the
  public-Raptor stage), `submission_c3.csv` (balanced views + quality-weighted CoAt family), `submission_c4.csv`
  (flat 0.65 outer weight) and `submission_c5.csv` (CoAt family share 0.5). `submission.csv` = MAIN.

Credits: Mattia Angeli, dreaddevelopment / Johnathan Wagner, Jiwei Liu, Pilkwang, Sofia Anjenje, Antoine G.,
prvsiyan, Marwan Mahmoud, renta.k, Anvith Pothula and everyone credited in the source notebook. Please upvote the originals.
"""


def patch_once(text: str, old: str, new: str, tag: str) -> str:
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{tag}: expected exactly one occurrence, found {n}")
    return text.replace(old, new)


def build(source: Path, out_dir: Path, owner: str, slug: str, title: str) -> Path:
    nb = json.loads(source.read_text())
    cells = nb["cells"]
    src = ["".join(c["source"]) for c in cells]
    where = {}
    for i, s in enumerate(src):
        if ASSET_TAIL in s:
            where["assets"] = i
        if EXEC_LINE in s and GPU_OLD in s:
            where["raptor"] = i
        if PREP_OLD in s:
            where["final"] = i
    for key in ("assets", "raptor", "final"):
        if key not in where:
            raise RuntimeError(f"could not locate the {key} cell in {source}")
    nb = copy.deepcopy(nb)
    cells = nb["cells"]

    a = where["assets"]
    cells[a]["source"] = patch_once(src[a], ASSET_TAIL, ASSET_TAIL[:-1] + ", " + ", ".join(repr(r) for r in EXTRA_ROOTS) + "]", "ASSET_ROOTS")
    r = where["raptor"]
    t = src[r]
    t = patch_once(t, EXEC_LINE, ARMS_EXT, "ARMS extension")
    t = patch_once(t, GPU_OLD, GPU_NEW, "gpu scheduling")
    t = patch_once(t, IDS_OLD, IDS_NEW, "failed-arm list")
    t = patch_once(t, PRINT_OLD, PRINT_NEW, "raptor print")
    cells[r]["source"] = t
    f = where["final"]
    cells[f]["source"] = patch_once(src[f], PREP_OLD, PREP_NEW, "expected preparations")

    variants = (HERE / "variants_cell.py").read_text()
    md_head = {"cell_type": "markdown", "metadata": {}, "source": HEADER_MD}
    md_tail = {"cell_type": "markdown", "metadata": {}, "source": "## Blend variants (CPU only)\n\nRebuilds the anchor from its saved parts (parity must be exact), then writes the variant CSVs. `submission.csv` = MAIN; the exact anchor is kept as `submission_anchor.csv`."}
    code_tail = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": variants}
    nb["cells"] = [md_head] + cells + [md_tail, code_tail]
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), "<cell>", "exec")
            c["outputs"] = []
            c["execution_count"] = None
    out_dir.mkdir(parents=True, exist_ok=True)
    nb_path = out_dir / f"{slug}.ipynb"
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    meta = {
        "id": f"{owner}/{slug}",
        "title": title,
        "code_file": nb_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "dreaddevelopment/raptor-knee-maxspan", "dreaddevelopment/raptor-knee-native384",
            "dreaddevelopment/raptor-knee-native384dense", "mattiaangeli/knee-mri-fold-weights",
            "mattiaangeli/opencv-python-headless-4120088-x86", "marwanmath/resnet-50-radimagenet-marwan",
            "mattiaangeli/rsna-knee-coat-resgated-ep10-top3", "mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2",
            "mattiaangeli/rsna-knee-coatnet-global96-top3", "antoinegg1/rsna-knee-e11-diverse-heads-v20",
            "antoinegg1/rsna-knee-e9-radimagenet-heads-v15", "pilkwang/rsna-knee-llm-labels",
            "prvsiyan/rsna-knee-v52-radimagenet-heads-20260812", "pilkwang/rsna-knee-weights",
        ] + EXTRA_DATASETS,
        "kernel_sources": ["sofiaanjenje/rsna-knee-e11-train", "sofiaanjenje/rsna-knee-e13-train"],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": ["metaresearch/dinov2/PyTorch/small/1"],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
        "machine_shape": "NvidiaTeslaT4",
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return nb_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="pulled bend-the-knee-to-speedy-raptors-the-original.ipynb")
    ap.add_argument("--out", required=True)
    ap.add_argument("--owner", required=True)
    ap.add_argument("--slug", default="rsna-knee-raptor-ext-v1")
    ap.add_argument("--title", default="RSNA Knee | Speedy Raptors + finespacing/widedense views")
    a = ap.parse_args()
    p = build(Path(a.source), Path(a.out), a.owner, a.slug, a.title)
    print(p)


if __name__ == "__main__":
    main()
