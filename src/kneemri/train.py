"""Fold training with masked BCE, fp16 AMP (T4-friendly), EMA, cosine schedule, OOF export.

Best checkpoint per fold is saved under the user-requested name (default: best_filament_unet.pth) with the
model config, target order, preprocessing config and validation metrics embedded.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import StudyDataset, collate_studies
from .metrics import macro_auc, per_target_auc
from .models import MaskedBCE, ModelConfig, StudyModel, save_checkpoint
from .preprocess import PrepConfig
from .schema import ID_COL, TARGETS

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    cache_dir: str = "work/cache"
    train_csv: str = "data/train.csv"
    folds_csv: str = "work/folds.csv"
    out_dir: str = "work/runs/exp"
    ckpt_name: str = "best_filament_unet.pth"
    aux_labels_csv: str | None = None  # optional weak (LLM/report) labels, same schema as train.csv
    aux_weight: float = 0.3  # loss weight for aux-only cells
    aux_silent_value: float = 0.5  # aux value meaning "report silent" -> treated as unknown
    fold: int = 0
    seed: int = 42
    epochs: int = 8
    batch_size: int = 2
    accum: int = 2
    lr_encoder: float = 1e-4
    lr_head: float = 5e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.06
    ema_decay: float = 0.998
    grad_clip: float = 2.0
    amp_dtype: str = "float16"  # T4 has no native bf16
    max_windows: int = 96
    series_dropout: float = 0.15
    num_workers: int = 4
    smoothing: float = 0.02
    gamma_neg: float = 0.0
    eval_every: int = 1
    max_train_studies: int | None = None  # debugging
    model: ModelConfig = field(default_factory=ModelConfig)
    prep: PrepConfig = field(default_factory=PrepConfig)

    def to_dict(self):
        d = asdict(self)
        return d


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])


def merge_labels(train_df: pd.DataFrame, aux_csv: str | None, silent_value: float, aux_weight: float):
    """Expert cells override aux cells. Returns (labels_df, weights_df) with NaN for unknown."""
    lab = train_df[[ID_COL] + TARGETS].copy()
    for t in TARGETS:
        lab[t] = pd.to_numeric(lab[t], errors="coerce")
    w = lab.copy()
    for t in TARGETS:
        w[t] = lab[t].notna().astype(float)
    if aux_csv:
        aux = pd.read_csv(aux_csv)
        aux[ID_COL] = aux[ID_COL].astype(str)
        aux = aux.set_index(ID_COL).reindex(lab[ID_COL].astype(str))
        for t in TARGETS:
            if t not in aux.columns:
                continue
            a = pd.to_numeric(aux[t], errors="coerce").to_numpy()
            a = np.where(np.isclose(a, silent_value), np.nan, a)
            fill = lab[t].isna().to_numpy() & np.isfinite(a)
            lab.loc[fill, t] = a[fill]
            cell_w = np.ones_like(a)
            if f"{t}__w" in aux.columns:  # calibrated report states carry their own confidence
                cell_w = pd.to_numeric(aux[f"{t}__w"], errors="coerce").fillna(1.0).to_numpy()
            w.loc[fill, t] = aux_weight * cell_w[fill]
    return lab, w


def build_loaders(cfg: TrainConfig):
    train_df = pd.read_csv(cfg.train_csv)
    train_df[ID_COL] = train_df[ID_COL].astype(str)
    folds = pd.read_csv(cfg.folds_csv)
    folds[ID_COL] = folds[ID_COL].astype(str)
    aux_csv = cfg.aux_labels_csv.format(fold=cfg.fold) if cfg.aux_labels_csv else None
    lab, w = merge_labels(train_df, aux_csv, cfg.aux_silent_value, cfg.aux_weight)
    lab = lab.merge(folds[[ID_COL, "fold"]], on=ID_COL)
    w = w.merge(folds[[ID_COL, "fold"]], on=ID_COL)
    tr = lab[lab.fold != cfg.fold].reset_index(drop=True)
    va = lab[lab.fold == cfg.fold].reset_index(drop=True)
    wtr = w[w.fold != cfg.fold].reset_index(drop=True)
    if cfg.max_train_studies:
        tr, wtr = tr.iloc[: cfg.max_train_studies], wtr.iloc[: cfg.max_train_studies]
    # only train on studies with at least one known label
    keep = tr[TARGETS].notna().any(axis=1).to_numpy()
    tr, wtr = tr[keep].reset_index(drop=True), wtr[keep].reset_index(drop=True)
    ds_tr = StudyDataset(tr, cfg.cache_dir, cfg.prep, train=True, max_windows=cfg.max_windows,
                         series_dropout=cfg.series_dropout, seed=cfg.seed)
    ds_va = StudyDataset(va, cfg.cache_dir, cfg.prep, train=False, max_windows=None)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                       collate_fn=collate_studies, drop_last=True, pin_memory=True, persistent_workers=cfg.num_workers > 0)
    dl_va = DataLoader(ds_va, batch_size=1, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_studies)
    wtr_t = torch.tensor(wtr[TARGETS].to_numpy(dtype=np.float32))
    id_to_w = {sid: wtr_t[i] for i, sid in enumerate(tr[ID_COL].tolist())}
    return dl_tr, dl_va, id_to_w, va


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device, amp_dtype, tta_flip: bool = False):
    model.eval()
    ids, probs = [], []
    for batch in loader:
        b = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            logit = model(b).float()
            if tta_flip:
                b2 = dict(b)
                b2["tiles"] = torch.flip(b["tiles"], dims=[-1])
                logit = 0.5 * (logit + model(b2).float())
        probs.append(torch.sigmoid(logit).cpu().numpy())
        ids.extend(batch["ids"])
    return ids, (np.concatenate(probs) if probs else np.zeros((0, len(TARGETS))))


def train_fold(cfg: TrainConfig) -> dict:
    seed_everything(cfg.seed + cfg.fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.float16 if cfg.amp_dtype == "float16" else torch.bfloat16
    out_dir = Path(cfg.out_dir) / f"fold{cfg.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    dl_tr, dl_va, id_to_w, va_df = build_loaders(cfg)
    cfg.model.ctx = cfg.prep.ctx
    model = StudyModel(cfg.model).to(device)
    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": cfg.lr_encoder},
                             {"params": head_params, "lr": cfg.lr_head}], weight_decay=cfg.weight_decay)
    steps_per_epoch = max(len(dl_tr) // cfg.accum, 1)
    total = steps_per_epoch * cfg.epochs
    warm = max(int(total * cfg.warmup_frac), 1)

    def lr_lambda(step):
        if step < warm:
            return step / warm
        p = (step - warm) / max(total - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))
    crit = MaskedBCE(smoothing=cfg.smoothing, gamma_neg=cfg.gamma_neg)
    ema = EMA(model, cfg.ema_decay)
    best, history = -1.0, []
    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        t0, run_loss, n_batches = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(dl_tr):
            b = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            w = torch.stack([id_to_w[s] for s in batch["ids"]]).to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                logits = model(b)
            loss = crit(logits, b["labels"], w) / cfg.accum
            scaler.scale(loss).backward()
            if (i + 1) % cfg.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                ema.update(model)
                step += 1
            run_loss += loss.item() * cfg.accum
            n_batches += 1
        rec = {"epoch": epoch, "train_loss": run_loss / max(n_batches, 1), "time_s": time.time() - t0}
        if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            ids, probs = predict(ema.shadow, dl_va, device, amp_dtype)
            y = va_df.set_index(ID_COL).loc[ids, TARGETS].to_numpy(dtype=np.float64)
            rec["val_macro_auc"] = macro_auc(y, probs)
            rec["val_per_target"] = per_target_auc(y, probs)
            if rec["val_macro_auc"] > best or not np.isfinite(best):
                best = rec["val_macro_auc"]
                save_checkpoint(out_dir / cfg.ckpt_name, ema.shadow, {"prep": asdict(cfg.prep), "fold": cfg.fold,
                                "epoch": epoch, "val_macro_auc": best, "per_target": rec["val_per_target"],
                                "train_cfg": cfg.to_dict()})
                oof = pd.DataFrame(probs, columns=TARGETS)
                oof.insert(0, ID_COL, ids)
                oof.to_csv(out_dir / "oof.csv", index=False)
        history.append(rec)
        log.info(json.dumps({k: v for k, v in rec.items() if k != "val_per_target"}))
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()
                          if k != "val_per_target"}), flush=True)
        (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    save_checkpoint(out_dir / "last.pth", ema.shadow, {"prep": asdict(cfg.prep), "fold": cfg.fold,
                    "train_cfg": cfg.to_dict()})
    return {"best_val_macro_auc": best, "history": history, "ckpt": str(out_dir / cfg.ckpt_name)}
