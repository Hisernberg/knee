"""Study-level 2.5D attention-MIL model.

  window tiles [B, W, ctx, H, W] --(timm backbone, in_chans=ctx)--> window features [B, W, D]
      + protocol/geometry embeddings (plane, fluid-sensitive, fat-suppressed, relative depth, series slot)
      --(optional within-series BiGRU; optional cross-series transformer)--> tokens
      --(12 learned finding queries, masked attention pooling)--> per-finding vectors --> 12 logits

Series boundaries are explicit (GRU runs per series). Any number of series/windows per study is supported
via the padding mask. Backbone is any timm model (CoAtNet, ConvNeXt, EfficientNet, MaxViT, ...).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import N_TARGETS, TARGETS


@dataclass
class ModelConfig:
    backbone: str = "convnext_tiny"
    pretrained: bool = True
    ctx: int = 3
    d_model: int = 256
    gru_hidden: int = 128
    use_gru: bool = True
    transformer_layers: int = 1
    n_heads: int = 4
    dropout: float = 0.2
    drop_path: float = 0.1
    n_targets: int = N_TARGETS
    grad_checkpoint: bool = False
    tiny: bool = False  # synthetic-test encoder, no pretrained capability

    def to_dict(self):
        return asdict(self)


class TinyEncoder(nn.Module):
    """Very small CNN for CPU tests; exposes `.num_features` like timm models."""

    def __init__(self, in_chans: int, width: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, width, 3, 2, 1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, 3, 2, 1), nn.BatchNorm2d(width * 2), nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 4, 3, 2, 1), nn.BatchNorm2d(width * 4), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.num_features = width * 4

    def forward(self, x):
        return self.net(x)


def build_encoder(cfg: ModelConfig) -> nn.Module:
    if cfg.tiny:
        return TinyEncoder(cfg.ctx)
    import timm

    enc = timm.create_model(cfg.backbone, pretrained=cfg.pretrained, in_chans=cfg.ctx, num_classes=0,
                            drop_path_rate=cfg.drop_path)
    if cfg.grad_checkpoint and hasattr(enc, "set_grad_checkpointing"):
        enc.set_grad_checkpointing(True)
    return enc


class TargetAttentionPool(nn.Module):
    """Each finding has its own query attending over all valid window tokens of the study."""

    def __init__(self, d_model: int, n_targets: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_targets, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        out, _ = self.attn(q, tokens, tokens, key_padding_mask=~mask, need_weights=False)
        return self.norm(out + q)  # [B, T, D]


class StudyModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = build_encoder(cfg)
        feat_dim = int(self.encoder.num_features)
        D = cfg.d_model
        self.proj = nn.Sequential(nn.Linear(feat_dim, D), nn.LayerNorm(D), nn.GELU(), nn.Dropout(cfg.dropout))
        self.plane_emb = nn.Embedding(4, D)
        self.fluid_emb = nn.Embedding(2, D)
        self.fat_emb = nn.Embedding(2, D)
        self.series_emb = nn.Embedding(16, D)
        self.depth_mlp = nn.Sequential(nn.Linear(1, D), nn.GELU(), nn.Linear(D, D))
        self.gru = nn.GRU(D, cfg.gru_hidden, batch_first=True, bidirectional=True) if cfg.use_gru else None
        self.gru_out = nn.Linear(2 * cfg.gru_hidden, D) if cfg.use_gru else None
        if cfg.transformer_layers > 0:
            layer = nn.TransformerEncoderLayer(D, cfg.n_heads, dim_feedforward=2 * D, dropout=cfg.dropout,
                                               batch_first=True, norm_first=True, activation="gelu")
            self.transformer = nn.TransformerEncoder(layer, cfg.transformer_layers)
        else:
            self.transformer = None
        self.pool = TargetAttentionPool(D, cfg.n_targets, cfg.n_heads, cfg.dropout)
        self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(D, 1))

    # ---- helpers -------------------------------------------------------------------------------
    def encode_windows(self, tiles: torch.Tensor, mask: torch.Tensor, chunk: int = 64) -> torch.Tensor:
        """Encode only real windows, in chunks, to bound memory. Returns [B, W, feat]."""
        B, W = tiles.shape[:2]
        flat = tiles[mask]  # [N, C, H, W]
        feats = []
        for i in range(0, flat.shape[0], chunk):
            feats.append(self.encoder(flat[i : i + chunk]))
        f = torch.cat(feats, 0) if feats else tiles.new_zeros((0, self.encoder.num_features))
        out = tiles.new_zeros((B, W, f.shape[-1]), dtype=f.dtype)
        out[mask] = f
        return out

    def _run_gru(self, tok: torch.Tensor, mask: torch.Tensor, series_idx: torch.Tensor) -> torch.Tensor:
        """BiGRU within each (study, series) sequence, in depth order (windows are already depth-ordered)."""
        B, W, D = tok.shape
        seqs, index = [], []
        for b in range(B):
            for s in torch.unique(series_idx[b][mask[b]]).tolist():
                idx = torch.where(mask[b] & (series_idx[b] == s))[0]
                seqs.append(tok[b, idx])
                index.append((b, idx))
        if not seqs:
            return tok
        lengths = torch.tensor([len(s) for s in seqs])
        padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        packed = nn.utils.rnn.pack_padded_sequence(padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = self.gru_out(out)
        res = tok.clone()
        for k, (b, idx) in enumerate(index):
            res[b, idx] = tok[b, idx] + out[k, : len(idx)]
        return res

    def forward(self, batch: dict) -> torch.Tensor:
        tiles, mask = batch["tiles"], batch["mask"]
        B, W = mask.shape
        if mask.sum() == 0:  # no usable image anywhere in the batch
            return self.head(self.pool.queries).view(1, -1).expand(B, -1)
        feats = self.encode_windows(tiles, mask)
        # sequence/attention head in fp32: fp16 overflowed here in public T4 runs
        with torch.autocast(device_type=tiles.device.type, enabled=False):
            return self._head(feats.float(), mask, batch)

    def _head(self, feats: torch.Tensor, mask: torch.Tensor, batch: dict) -> torch.Tensor:
        tok = self.proj(feats)
        tok = tok + self.plane_emb(batch["plane"].clamp(0, 3)) + self.fluid_emb(batch["fluid"].clamp(0, 1)) \
            + self.fat_emb(batch["fat"].clamp(0, 1)) + self.series_emb(batch["series_idx"].clamp(0, 15)) \
            + self.depth_mlp(batch["depth"].unsqueeze(-1))
        if self.gru is not None:
            tok = self._run_gru(tok, mask, batch["series_idx"])
        if self.transformer is not None:
            tok = self.transformer(tok, src_key_padding_mask=~mask)
        pooled = self.pool(tok, mask)  # [B, T, D]
        return self.head(pooled).squeeze(-1)  # [B, T]


class MaskedBCE(nn.Module):
    """BCE on labelled cells only (NaN = unknown). Supports soft labels, per-cell weights, label smoothing
    and an asymmetric-focal variant (gamma_neg > 0) that down-weights easy negatives."""

    def __init__(self, smoothing: float = 0.0, gamma_neg: float = 0.0, gamma_pos: float = 0.0, pos_weight=None):
        super().__init__()
        self.smoothing, self.gamma_neg, self.gamma_pos = smoothing, gamma_neg, gamma_pos
        self.register_buffer("pos_weight", torch.as_tensor(pos_weight, dtype=torch.float32) if pos_weight is not None
                             else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor | None = None):
        valid = torch.isfinite(targets)
        if valid.sum() == 0:
            return logits.sum() * 0.0
        y = torch.where(valid, targets, torch.zeros_like(targets)).float()
        if self.smoothing > 0:
            y = y * (1 - self.smoothing) + 0.5 * self.smoothing
        logits = logits.float()
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            p = torch.sigmoid(logits)
            loss_pos = -y * torch.log(p.clamp_min(1e-6)) * (1 - p).pow(self.gamma_pos)
            loss_neg = -(1 - y) * torch.log((1 - p).clamp_min(1e-6)) * p.pow(self.gamma_neg)
            loss = loss_pos + loss_neg
        else:
            loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none",
                                                      pos_weight=self.pos_weight)
        w = valid.float() if weights is None else valid.float() * weights.float()
        return (loss * w).sum() / w.sum().clamp_min(1.0)


def save_checkpoint(path, model: StudyModel, extra: dict | None = None) -> None:
    torch.save({"model_cfg": model.cfg.to_dict(), "targets": TARGETS, "state_dict": model.state_dict(),
                **(extra or {})}, path)


def load_checkpoint(path, map_location="cpu") -> tuple[StudyModel, dict]:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ModelConfig(**{**ck["model_cfg"], "pretrained": False})
    model = StudyModel(cfg)
    model.load_state_dict(ck["state_dict"], strict=True)
    return model, ck
