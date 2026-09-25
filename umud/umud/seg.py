"""Aponeurosis + fascicle segmentation: training and test-set inference.

Self-contained (only depends on numpy / opencv / pillow / torch /
segmentation_models_pytorch) so it can be pushed as a Kaggle script kernel
unchanged. Usage:

    python seg.py --data <competition dir> --out <dir> [--epochs-apo 40 --epochs-fasc 30]
    python seg.py --data <dir> --out <dir> --infer-only --weights <dir with *.pt>

Outputs in --out:
    apo.pt / fasc.pt                  trained weights
    probs/<image_id>_apo.png          aponeurosis probability (uint8) on the B-mode crop
    probs/<image_id>_fasc.png         fascicle probability (uint8) on the B-mode crop
    crops.csv                         image_id, scale and crop box used
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from umud.scale import detect_scale
except Exception:  # when run as a flat Kaggle script, scale.py is inlined below
    detect_scale = None

IN_H, IN_W = 512, 768
WORKERS = 4
SEED = 42


def seed_all(s: int = SEED) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def read_gray(path: str) -> np.ndarray:
    with Image.open(path) as im:
        a = np.asarray(im)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=-1)
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        a = (a - a.min()) / max(a.max() - a.min(), 1e-6) * 255
    return a.astype(np.uint8)


def letterbox(img: np.ndarray, h: int | None = None, w: int | None = None, interp=cv2.INTER_AREA):
    """Resize keeping aspect ratio and pad to (h, w). Returns image and (scale, oy, ox)."""
    h, w = h or IN_H, w or IN_W
    ih, iw = img.shape[:2]
    s = min(h / ih, w / iw)
    nh, nw = max(1, round(ih * s)), max(1, round(iw * s))
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    out = np.zeros((h, w), dtype=img.dtype)
    oy, ox = (h - nh) // 2, (w - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = r
    return out, (s, oy, ox, nh, nw)


def unletterbox(p: np.ndarray, meta, ih: int, iw: int) -> np.ndarray:
    s, oy, ox, nh, nw = meta
    return cv2.resize(p[oy:oy + nh, ox:ox + nw], (iw, ih), interpolation=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------- data
def list_pairs(data: Path, kind: str, dedupe: bool = True):
    """Image/mask pairs; exact duplicate pairs (same image and mask bytes, see host topic 740356) kept once."""
    import hashlib
    imgs = sorted(glob.glob(str(data / f"{kind}_imgs_v1" / "*" / "*.tif")))
    out, seen = [], set()
    for p in imgs:
        m = p.replace(f"{kind}_imgs_v1", f"{kind}_masks_v1").replace(f"{kind}_images", f"{kind}_masks")
        if not os.path.exists(m):
            continue
        if dedupe:
            key = hashlib.md5(open(p, "rb").read()).hexdigest() + hashlib.md5(open(m, "rb").read()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
        out.append((p, m))
    return out


def load_cache(pairs, cache_h=640, cache_w=960):
    """Pre-load images/masks resized to a fixed cache size (keeps RAM bounded)."""
    X, Y = [], []
    for ip, mp in pairs:
        g = read_gray(ip)
        m = read_gray(mp)
        m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        if (m > 127).mean() > 0.5:  # some masks are stored inverted
            m = 255 - m
        X.append(cv2.resize(g, (cache_w, cache_h), interpolation=cv2.INTER_AREA))
        Y.append((cv2.resize(m, (cache_w, cache_h), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8))
    return X, Y


class SegDS(torch.utils.data.Dataset):
    def __init__(self, X, Y, idx, train: bool):
        self.X, self.Y, self.idx, self.train = X, Y, idx, train

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        x, y = self.X[self.idx[i]], self.Y[self.idx[i]]
        if self.train:
            H, W = x.shape
            # random crop (covers test-time crops of the B-mode area)
            if random.random() < 0.7:
                sc = random.uniform(0.55, 1.0)
                ch, cw = int(H * random.uniform(sc, 1.0)), int(W * sc)
                y0, x0 = random.randint(0, H - ch), random.randint(0, W - cw)
                x, y = x[y0:y0 + ch, x0:x0 + cw], y[y0:y0 + ch, x0:x0 + cw]
            if random.random() < 0.5:
                x, y = x[:, ::-1], y[:, ::-1]
            if random.random() < 0.3:  # small rotation
                ang = random.uniform(-8, 8)
                M = cv2.getRotationMatrix2D((x.shape[1] / 2, x.shape[0] / 2), ang, 1.0)
                x = cv2.warpAffine(np.ascontiguousarray(x), M, (x.shape[1], x.shape[0]), flags=cv2.INTER_LINEAR)
                y = cv2.warpAffine(np.ascontiguousarray(y), M, (y.shape[1], y.shape[0]), flags=cv2.INTER_NEAREST)
            x = x.astype(np.float32)
            # photometric: gamma / contrast / brightness / noise
            x = 255 * (x / 255) ** random.uniform(0.7, 1.4)
            x = x * random.uniform(0.75, 1.25) + random.uniform(-20, 20)
            if random.random() < 0.3:
                x = x + np.random.randn(*x.shape) * random.uniform(2, 10)
            if random.random() < 0.2:
                x = cv2.GaussianBlur(x, (0, 0), random.uniform(0.5, 1.5))
            x = np.clip(x, 0, 255).astype(np.uint8)
        x, _ = letterbox(np.ascontiguousarray(x))
        y, _ = letterbox(np.ascontiguousarray(y), interp=cv2.INTER_NEAREST)
        xt = torch.from_numpy(x).float().div(255).sub(0.45).div(0.225)[None].repeat(3, 1, 1)
        return xt, torch.from_numpy(y).float()[None]


ENCODER = "resnet34"


def make_model(weights: str | None = "imagenet"):
    import segmentation_models_pytorch as smp
    return smp.Unet(ENCODER, encoder_weights=weights, in_channels=3, classes=1)


def dice_loss(logits, y, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * y).sum((2, 3)) + eps
    den = p.sum((2, 3)) + y.sum((2, 3)) + eps
    return 1 - (num / den).mean()


INIT_DIR = None
LR = 3e-4


def train_kind(data: Path, out: Path, kind: str, epochs: int, bs: int, dev: str) -> None:
    seed_all()
    pairs = list_pairs(data, kind)
    print(f"[{kind}] {len(pairs)} pairs", flush=True)
    t0 = time.time()
    X, Y = load_cache(pairs)
    print(f"[{kind}] cached in {time.time()-t0:.0f}s", flush=True)
    idx = np.random.RandomState(SEED).permutation(len(X))
    nval = max(20, len(X) // 12)
    va, tr = idx[:nval], idx[nval:]
    dl = torch.utils.data.DataLoader(SegDS(X, Y, tr, True), batch_size=bs, shuffle=True, num_workers=WORKERS,
                                     drop_last=True, pin_memory=True, persistent_workers=WORKERS > 0)
    dv = torch.utils.data.DataLoader(SegDS(X, Y, va, False), batch_size=bs, num_workers=min(2, WORKERS))
    model = make_model(None if INIT_DIR else "imagenet")
    if INIT_DIR:  # fine-tune from earlier weights
        model.load_state_dict(torch.load(Path(INIT_DIR) / f"{kind}.pt", map_location="cpu"))
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=epochs * len(dl), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    pos_w = torch.tensor([3.0 if kind == "fasc" else 2.0], device=dev)
    best = -1.0
    for ep in range(epochs):
        model.train()
        tl = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev, enabled=dev == "cuda"):
                lo = model(xb)
                loss = F.binary_cross_entropy_with_logits(lo, yb, pos_weight=pos_w) + dice_loss(lo.float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item()
        model.eval()
        inter = union = 0.0
        with torch.no_grad():
            for xb, yb in dv:
                p = (torch.sigmoid(model(xb.to(dev))) > 0.5).float().cpu()
                inter += (p * yb).sum().item()
                union += (p.sum() + yb.sum()).item()
        vd = 2 * inter / max(union, 1)
        print(f"[{kind}] ep {ep+1}/{epochs} loss {tl/len(dl):.4f} val_dice {vd:.4f} t {time.time()-t0:.0f}s", flush=True)
        if vd > best:
            best = vd
            torch.save(model.state_dict(), out / f"{kind}.pt")
    print(f"[{kind}] best val dice {best:.4f}", flush=True)


@torch.no_grad()
def predict(model, g: np.ndarray, dev: str) -> np.ndarray:
    x, meta = letterbox(g)
    xt = torch.from_numpy(x).float().div(255).sub(0.45).div(0.225)[None, None].repeat(1, 3, 1, 1).to(dev)
    xt = torch.cat([xt, xt.flip(-1)])
    p = torch.sigmoid(model(xt)).float()
    p = ((p[0] + p[1].flip(-1)) / 2)[0].cpu().numpy()
    return unletterbox(p, meta, *g.shape[:2])


def infer_test(data: Path, out: Path, wdir: Path, dev: str) -> None:
    models = {}
    for k in ("apo", "fasc"):
        m = make_model(None)
        m.load_state_dict(torch.load(wdir / f"{k}.pt", map_location="cpu"))
        models[k] = m.to(dev).eval()
    pdir = out / "probs"
    pdir.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(data / "test_images_v2" / "*" / "IMG_*")))
    rows = []
    for f in files:
        name = os.path.basename(f)
        with Image.open(f) as im:
            a = np.asarray(im)
        sc = detect_scale(a, name.rsplit(".", 1)[-1])
        g = read_gray(f)[sc.t:sc.b, sc.l:sc.r]
        for k, m in models.items():
            p = predict(m, g, dev)
            cv2.imwrite(str(pdir / f"{name}_{k}.png"), np.clip(p * 255, 0, 255).astype(np.uint8))
        rows.append((name, sc.px_per_mm, sc.l, sc.t, sc.r, sc.b, sc.family))
    with open(out / "crops.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "px_per_mm", "l", "t", "r", "b", "family"])
        w.writerows(rows)
    print(f"inferred {len(rows)} test images", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/kaggle/input/competitions/umud-challenge-muscle-architecture-in-ultrasound-data")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--epochs-apo", type=int, default=40)
    ap.add_argument("--epochs-fasc", type=int, default=30)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--infer-only", action="store_true")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--size", default="512x768", help="network input HxW")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--init", default=None, help="dir with apo.pt/fasc.pt to fine-tune from")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--kinds", default="apo,fasc", help="which models to train")
    a, _ = ap.parse_known_args()
    global ENCODER, IN_H, IN_W, WORKERS, INIT_DIR, LR
    ENCODER = a.encoder
    IN_H, IN_W = (int(v) for v in a.size.split("x"))
    WORKERS = a.workers
    INIT_DIR, LR = a.init, a.lr
    if INIT_DIR and not (Path(INIT_DIR) / "apo.pt").exists():  # Kaggle mounts kernel outputs at varying depths
        hits = sorted(glob.glob(str(Path(INIT_DIR).parent / "**" / "apo.pt"), recursive=True))
        if hits:
            INIT_DIR = str(Path(hits[0]).parent)
        print("init weights from", INIT_DIR, flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.infer_only:
        kinds = a.kinds.split(",")
        for k in ("apo", "fasc"):
            if k in kinds:
                train_kind(data, out, k, a.epochs_apo if k == "apo" else a.epochs_fasc, a.bs, dev)
            elif a.init:  # untouched model: carry the old weights over for inference
                import shutil
                shutil.copy(Path(INIT_DIR) / f"{k}.pt", out / f"{k}.pt")
    infer_test(data, out, Path(a.weights) if a.weights else out, dev)


if __name__ == "__main__":
    main()
