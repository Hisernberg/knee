"""Data access: COCO parsing, per-annotator instance masks, image cache."""
import json, os
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from pycocotools import mask as mu

ROOT = Path(os.environ.get("CH_DATA", "/home/user/data/fil/MAGFiLO_1.0_Kaggle_2026"))
WORK = Path(os.environ.get("CH_WORK", "/home/user/work"))
H = W = 2048


def load_coco():
    d = json.load(open(ROOT / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    anns = defaultdict(list)
    for a in d["annotations"]:
        anns[a["image_id"]].append(a)
    recs = defaultdict(list)  # file stem -> list of (record_id, [anns])
    for im in d["images"]:
        recs[Path(im["file_name"]).stem].append((im["id"], anns.get(im["id"], [])))
    return dict(recs)


def ann_rle(a):
    rles = mu.frPyObjects(a["segmentation"], H, W)
    return mu.merge(rles)


def record_rles(anns):
    return [ann_rle(a) for a in anns if not a.get("iscrowd", 0)]


def read_gray(path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def train_path(stem):
    return ROOT / "train" / "train_images" / f"{stem}.jpeg"


def test_paths():
    return sorted((ROOT / "test" / "test_images").glob("*.jpeg"))
