"""
OddsPatch-iTransformer — Channel-independent feature embedding + cross-market Transformer.

Core idea (from iTransformer ICLR2024 + PatchTST ICLR2023):
  - Each odds feature (euro_h, imp_d, margin, ...) gets its own temporal embedding
  - Transformer attention runs over features (not time steps) for cross-market interaction
  - [CLS] token aggregates global feature representations

Architecture:
    Bucketed features [B, 10, 9, 2]
         ↓  reshape
    Per-feature temporal signature [B, 9, 20]
         ↓  9 × Linear(20, d_model)
    Feature embeddings [B, 9, d_model]
         ↓  prepend [CLS]
    [B, 10, d_model]
         ↓  N × CrossMarketTransformerBlock (norm_first=True)
    Extract CLS [B, d_model]
         ↓  MLP Head
    [B, 3] logits

Parameter budget (d_model=64, n_layers=2, d_ff=128):
    Feature embeds:  9 × 20 × 64     = 11,520
    CLS token:       64               =     64
    Transformer ×2:  2 × ~16,448      = 32,896
    Final norm:      64               =     64
    MLP head:        64×32 + 32×3     =  2,144
    Total                              ≈ 79,688 (< 100K ✓)
"""

import math
from collections import defaultdict
from typing import Optional

import torch
from torch import nn


# ── Constants ──────────────────────────────────────────────────────────

BUCKET_EDGES = [
    (float('inf'), 168, 'opening'),
    (168, 72, '7d'),
    (72, 24, '3d'),
    (24, 12, '1d'),
    (12, 6, '12h'),
    (6, 3, '6h'),
    (3, 1, '3h'),
    (1, 0.5, '1h'),
    (0.5, 0, '30m'),
    (0, -float('inf'), 'closing'),
]
N_BUCKETS = len(BUCKET_EDGES)

CORE_FEAT_INDICES = [0, 1, 2, 9, 10, 11, 12, 17, 24]
CORE_FEAT_NAMES = [
    'euro_h', 'euro_d', 'euro_a',
    'imp_h', 'imp_d', 'imp_a',
    'margin', 'time_log', 'has_euro',
]
N_CORE_FEATS = len(CORE_FEAT_INDICES)
N_AGG = 2


# ── Feature extraction ────────────────────────────────────────────────

def extract_core_feature(event: dict, feat_idx: int) -> float:
    try:
        if feat_idx == 0:
            return float(event.get('euro_h', 0) or 0)
        if feat_idx == 1:
            return float(event.get('euro_d', 0) or 0)
        if feat_idx == 2:
            return float(event.get('euro_a', 0) or 0)
        eh = float(event.get('euro_h', 0) or 0)
        ed = float(event.get('euro_d', 0) or 0)
        ea = float(event.get('euro_a', 0) or 0)
        if eh > 0 and ed > 0 and ea > 0:
            rh, rd, ra = 1.0 / eh, 1.0 / ed, 1.0 / ea
            total = rh + rd + ra
            if total > 0:
                if feat_idx == 9:
                    return rh / total
                if feat_idx == 10:
                    return rd / total
                if feat_idx == 11:
                    return ra / total
        if feat_idx in (9, 10, 11):
            return 1.0 / 3.0
        if feat_idx == 12:
            if eh > 0 and ed > 0 and ea > 0:
                return (1.0 / eh + 1.0 / ed + 1.0 / ea) - 1.0
            return 0.0
        if feat_idx == 17:
            return math.log1p(float(event.get('minutes_before_kickoff', 0)))
        if feat_idx == 24:
            return 1.0
        return 0.0
    except Exception:
        return 0.0


def bucketize_sample(raw_timeline: list) -> torch.Tensor:
    """Convert raw timeline events → [N_BUCKETS, N_CORE_FEATS, N_AGG]."""
    buckets = defaultdict(list)
    for e in raw_timeline:
        hours = e.get('minutes_before_kickoff', 0) / 60.0
        for idx, (hi, lo, _) in enumerate(BUCKET_EDGES):
            if hours <= hi and hours > lo:
                buckets[idx].append(e)
                break

    result = torch.zeros(N_BUCKETS, N_CORE_FEATS, N_AGG)
    for bucket_idx in range(N_BUCKETS):
        evts = sorted(
            buckets.get(bucket_idx, []),
            key=lambda e: e.get('minutes_before_kickoff', 0),
            reverse=True,
        )
        if not evts:
            continue
        for fi, feat_idx in enumerate(CORE_FEAT_INDICES):
            vals = []
            for e in evts:
                v = extract_core_feature(e, feat_idx)
                if math.isfinite(v):
                    vals.append(v)
            if not vals:
                continue
            vt = torch.tensor(vals, dtype=torch.float32)
            result[bucket_idx, fi, 0] = vt[-1]
            result[bucket_idx, fi, 1] = vt.mean()
    return result


# ── RMSNorm ───────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.float() * norm


# ── Transformer block ─────────────────────────────────────────────────

class CrossMarketBlock(nn.Module):
    """Pre-norm Transformer block with RMSNorm, bias=False throughout."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True, bias=False,
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        xn = self.attn_norm(x)
        x = residual + self.attn(xn, xn, xn)[0]
        residual = x
        x = residual + self.ffn(self.ffn_norm(x))
        return x


# ── Main model ────────────────────────────────────────────────────────

class OddsPatchITransformer(nn.Module):
    """
    OddsPatch-iTransformer for 1X2 prediction.

    Input:  [B, N_BUCKETS=10, N_CORE_FEATS=9, N_AGG=2]
    Output: [B, 3] logits
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        n_buckets: int = N_BUCKETS,
        n_features: int = N_CORE_FEATS,
        n_agg: int = N_AGG,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        temporal_dim = n_buckets * n_agg

        self.feature_embeds = nn.ModuleList([
            nn.Linear(temporal_dim, d_model, bias=False)
            for _ in range(n_features)
        ])

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.layers = nn.ModuleList([
            CrossMarketBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_ff // 2, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        x = x.permute(0, 2, 1, 3).reshape(B, self.n_features, -1)

        parts = [self.feature_embeds[i](x[:, i]) for i in range(self.n_features)]
        h = torch.stack(parts, dim=1)

        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)

        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)

        return self.head(h[:, 0])
