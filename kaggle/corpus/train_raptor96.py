#!/usr/bin/env python3
"""Train a 2.5D per-finding attention-MIL classifier on the corpus96 (dense 96-slice, 336 px, 140 mm).

Design (the public "Raptor" recipe, re-implemented): every 3-neighbouring-slice window of a study is
encoded by a timm backbone at RES px; a per-finding attention head pools the windows; 12 logits.
Labels: soft report labels (consensus of clean public LLM tables) for the 4,349 report-only studies,
confidence-weighted BCE; the 58 expert-labelled studies are NEVER trained on and gate every epoch.
5-fold OOF on the report-only studies (fold = stable hash of the study UID) gives the honest blend metric.

  python train_raptor96.py --corpus-dirs /kaggle/input/rsna-knee-corpus96-part0,... \
      --train-csv train.csv --labels labels_consensus.csv --fold 0 --epochs 12 --out runs/f0
  python train_raptor96.py --smoke        # CPU smoke test on a synthetic corpus (tiny backbone)
Checkpoint dict: model, arch, res, lab, epoch, gold_auc, aucs (same keys as the public Raptor files).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
       "Synovitis", "Baker's", "Contusion", "Fracture"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ----------------------------------------------------------------------------- corpus
class Corpus:
    """Concatenation of corpus parts, each (n,D,H,W) uint8 memmap + masks + ids."""

    def __init__(self, dirs):
        self.vols, self.masks, self.ids = [], [], []
        for d in dirs:
            d = Path(d)
            for vp in sorted(d.glob("vols_part*.npy")):
                k = vp.name[len("vols_part"):-4]
                self.vols.append(np.load(vp, mmap_mode="r"))
                self.masks.append(np.load(d / f"masks_part{k}.npy"))
                self.ids.append(np.load(d / f"ids_part{k}.npy").astype(str))
        if not self.vols:
            raise FileNotFoundError(f"no vols_part*.npy under {dirs}")
        self.index = {}
        for pi, ids in enumerate(self.ids):
            for ri, u in enumerate(ids):
                self.index[u] = (pi, ri)
        self.D = self.vols[0].shape[1]

    def get(self, uid):
        pi, ri = self.index[uid]
        return self.vols[pi][ri], self.masks[pi][ri]


def valid_centers(mask):
    idx = np.where(mask > 0)[0]
    return [c for c in idx if c - 1 >= 0 and c + 1 < len(mask) and mask[c - 1] and mask[c + 1]]


class StudyWindows(Dataset):
    def __init__(self, corpus, uids, labels, weights, k, res, train):
        self.c, self.uids, self.labels, self.weights, self.k, self.res, self.train = corpus, list(uids), labels, weights, k, res, train

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        uid = self.uids[i]
        vol, mask = self.c.get(uid)
        cs = valid_centers(mask) or [max(1, min(len(mask) // 2, len(mask) - 2))]
        if self.train:
            if len(cs) >= self.k:
                cs = sorted(np.random.choice(cs, self.k, replace=False).tolist())
            else:
                cs = sorted(np.random.choice(cs, self.k, replace=True).tolist())
        else:
            cs = [cs[j] for j in np.linspace(0, len(cs) - 1, min(self.k, len(cs))).round().astype(int)]
        wins = np.stack([np.stack([vol[c - 1], vol[c], vol[c + 1]], 0) for c in cs]).astype(np.float32) / 255.0
        x = torch.from_numpy(wins)
        if self.train and np.random.rand() < 0.5:
            x = x.flip(-1)  # label-preserving horizontal flip (a flipped left knee is a plausible right knee)
        if x.shape[-1] != self.res:
            x = F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False)
        x = (x - MEAN) / STD
        y = torch.from_numpy(self.labels[uid].astype(np.float32))
        w = torch.from_numpy(self.weights[uid].astype(np.float32))
        return x, y, w, uid


def collate(batch):
    ks = [b[0].shape[0] for b in batch]
    kmin = min(ks)
    x = torch.stack([b[0][:kmin] for b in batch])
    return x, torch.stack([b[1] for b in batch]), torch.stack([b[2] for b in batch]), [b[3] for b in batch]


# ----------------------------------------------------------------------------- model
class RaptorClassifier(nn.Module):
    def __init__(self, backbone, f_dim, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(f_dim)
        self.att = nn.Sequential(nn.Linear(f_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, f_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)

    def forward(self, x, micro=8):
        B, K = x.shape[:2]
        flat = x.flatten(0, 1)
        feats = torch.cat([self.backbone(flat[i:i + micro]) for i in range(0, flat.shape[0], micro)], 0).view(B, K, -1)
        h = self.norm(feats)
        a = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb


def build_model(arch, pretrained):
    import timm

    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext", "resnet", "efficientnet", "tf_efficientnet"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    kw.update(global_pool="token", dynamic_img_size=True) if is_vit else kw.update(global_pool="avg")
    bb = timm.create_model(arch, **kw)
    return RaptorClassifier(bb, bb.num_features)


# ----------------------------------------------------------------------------- labels / folds
def fold_of(uid, n_folds=5):
    return int(hashlib.sha1(uid.encode()).hexdigest(), 16) % n_folds


def load_labels(train_csv, labels_csv, unk_weight=0.3):
    tr = pd.read_csv(train_csv, dtype={"StudyInstanceUID": str})
    gold = tr[tr[LAB].notna().all(axis=1)].set_index("StudyInstanceUID")[LAB].astype(float)
    soft = pd.read_csv(labels_csv, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    wcols = [c + "__w" for c in LAB]
    labels, weights = {}, {}
    for uid, row in soft.iterrows():
        if uid in gold.index:
            continue
        y = row[LAB].to_numpy(float)
        w = row[wcols].to_numpy(float) if all(c in soft.columns for c in wcols) else np.where(np.abs(y - 0.5) < 0.26, unk_weight, 1.0)
        labels[uid], weights[uid] = y, w
    for uid, row in gold.iterrows():
        labels[uid], weights[uid] = row.to_numpy(float), np.ones(12)
    return labels, weights, list(gold.index)


def macro_auc(y, p):
    from sklearn.metrics import roc_auc_score

    aucs = {}
    for j, l in enumerate(LAB):
        if len(np.unique(y[:, j] > 0.5)) == 2:
            aucs[l] = float(roc_auc_score(y[:, j] > 0.5, p[:, j]))
    return float(np.mean(list(aucs.values()))) if aucs else float("nan"), aucs


# ----------------------------------------------------------------------------- train
@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    out, ys, ids = [], [], []
    for x, y, _, uid in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(x.to(device, non_blocking=True))
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
        ys.append(y.numpy())
        ids += uid
    return np.concatenate(out), np.concatenate(ys), ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dirs", default="")
    ap.add_argument("--train-csv", default="")
    ap.add_argument("--labels", default="", help="csv: StudyInstanceUID + 12 soft labels (+ optional __w weight columns)")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--arch", default="coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k")
    ap.add_argument("--res", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--k", type=int, default=16, help="train windows per study")
    ap.add_argument("--k-eval", type=int, default=94)
    ap.add_argument("--bb-lr", type=float, default=4e-5)
    ap.add_argument("--head-lr", type=float, default=4e-4)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--topk", type=int, default=3, help="epochs averaged into the SWA checkpoint")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--out", default="runs/f0")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(1400 + a.fold)
    np.random.seed(1400 + a.fold)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.smoke:  # synthetic corpus + labels, tiny backbone, CPU
        tmp = out / "smoke_corpus"
        tmp.mkdir(exist_ok=True)
        n, D, H = 24, 96, 64
        ids = np.array([f"1.2.{i}" for i in range(n)])
        vols = np.lib.format.open_memmap(tmp / "vols_part0.npy", mode="w+", dtype=np.uint8, shape=(n, D, H, H))
        vols[:] = np.random.randint(0, 255, (n, D, H, H), dtype=np.uint8)
        masks = np.ones((n, D), np.uint8)
        masks[:, 80:] = 0
        np.save(tmp / "masks_part0.npy", masks)
        np.save(tmp / "ids_part0.npy", ids)
        soft = pd.DataFrame(np.random.rand(n, 12), columns=LAB)
        soft.insert(0, "StudyInstanceUID", ids)
        soft.iloc[:6, 1:] = np.random.randint(0, 2, (6, 12))  # 6 "gold" rows
        tr = soft.copy()
        tr.iloc[6:, 1:] = np.nan
        tr.to_csv(tmp / "train.csv", index=False)
        soft.iloc[6:].to_csv(tmp / "labels.csv", index=False)
        a.corpus_dirs, a.train_csv, a.labels = str(tmp), str(tmp / "train.csv"), str(tmp / "labels.csv")
        a.arch, a.res, a.epochs, a.bs, a.k, a.k_eval, a.workers, a.no_pretrained = "resnet18", 64, 2, 2, 4, 6, 0, True

    corpus = Corpus(a.corpus_dirs.split(","))
    labels, weights, gold_ids = load_labels(a.train_csv, a.labels)
    weak = [u for u in labels if u not in set(gold_ids) and u in corpus.index]
    gold_ids = [u for u in gold_ids if u in corpus.index]
    train_ids = [u for u in weak if fold_of(u, a.n_folds) != a.fold]
    oof_ids = [u for u in weak if fold_of(u, a.n_folds) == a.fold]
    print(f"train {len(train_ids)} | oof fold {a.fold}: {len(oof_ids)} | gold {len(gold_ids)} | corpus D={corpus.D}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    model = build_model(a.arch, not a.no_pretrained).to(device)
    core = model
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  # T4x2: one study per GPU at bs=2
        print(f"DataParallel over {torch.cuda.device_count()} GPUs", flush=True)
    tl = DataLoader(StudyWindows(corpus, train_ids, labels, weights, a.k, a.res, True), batch_size=a.bs, shuffle=True,
                    num_workers=a.workers, drop_last=True, collate_fn=collate, persistent_workers=a.workers > 0)
    gl = DataLoader(StudyWindows(corpus, gold_ids, labels, weights, a.k_eval, a.res, False), batch_size=1, num_workers=a.workers, collate_fn=collate)
    ol = DataLoader(StudyWindows(corpus, oof_ids, labels, weights, a.k_eval, a.res, False), batch_size=1, num_workers=a.workers, collate_fn=collate)
    head_params = [p for n_, p in core.named_parameters() if not n_.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": core.backbone.parameters(), "lr": a.bb_lr}, {"params": head_params, "lr": a.head_lr}], weight_decay=a.wd)
    steps = max(1, len(tl) // a.accum) * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.bb_lr, a.head_lr], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=amp)
    hist, topk, best = [], [], -1.0
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        tot, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, (x, y, w, _) in enumerate(tl):
            x, y, w = x.to(device, non_blocking=True), y.to(device), w.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
            loss = (F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none") * w).sum() / w.sum().clamp_min(1.0)
            scaler.scale(loss / a.accum).backward()
            if (step + 1) % a.accum == 0 or step + 1 == len(tl):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if sched.last_epoch < steps - 1:
                    sched.step()
            tot += float(loss)
            nb += 1
        pg, yg, _ = predict(model, gl, device, amp)
        au, aucs = macro_auc(yg, pg)
        hist.append({"epoch": ep, "loss": tot / max(nb, 1), "gold_auc": au, "seconds": round(time.time() - t0)})
        print(f"ep{ep} loss {tot / max(nb, 1):.4f} | gold macro-AUC {au:.4f} (best {max(best, au):.4f}) | {time.time() - t0:.0f}s", flush=True)
        state = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
        ck = {"model": state, "arch": a.arch, "res": a.res, "lab": LAB, "epoch": ep, "gold_auc": au, "aucs": aucs, "src": "corpus96"}
        if au > best:
            best = au
            torch.save(ck, out / "raptor96_best.pt")
        topk.append(ck)
        topk.sort(key=lambda c: -c["gold_auc"])
        topk = topk[: a.topk]
        (out / "history.json").write_text(json.dumps(hist, indent=1))
    # SWA over the top-k epochs by gold AUC
    avg = {k: sum(c["model"][k].float() for c in topk) / len(topk) if topk[0]["model"][k].is_floating_point() else topk[0]["model"][k] for k in topk[0]["model"]}
    core.load_state_dict(avg)
    pg, yg, _ = predict(model, gl, device, amp)
    au_swa, aucs_swa = macro_auc(yg, pg)
    torch.save({"model": avg, "arch": a.arch, "res": a.res, "lab": LAB, "epoch": [c["epoch"] for c in topk], "swa_over": [c["epoch"] for c in topk],
                "gold_auc": au_swa, "aucs": aucs_swa, "src": "corpus96"}, out / "raptor96_swa.pt")
    po, yo, ids = predict(model, ol, device, amp)
    oof = pd.DataFrame(po, columns=LAB)
    oof.insert(0, "StudyInstanceUID", ids)
    oof.to_csv(out / "oof.csv", index=False)
    au_oof, _ = macro_auc(yo, po)
    print(f"DONE fold {a.fold}: gold best-epoch {best:.4f} | SWA{[c['epoch'] for c in topk]} gold {au_swa:.4f} | OOF (vs soft labels) {au_oof:.4f}", flush=True)
    (out / "summary.json").write_text(json.dumps({"fold": a.fold, "gold_best": best, "gold_swa": au_swa, "oof_auc_vs_soft": au_oof,
                                                  "swa_epochs": [c["epoch"] for c in topk], "n_train": len(train_ids), "n_oof": len(oof_ids)}, indent=1))


if __name__ == "__main__":
    main()
