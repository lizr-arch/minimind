"""
P3b: OddsPatch-iTransformer V2 — Cross-market features (Euro + Asian + OU).

Extends P3a design with full cross-market feature extraction:
  - Euro odds (3) + implied (3) + margin + has_euro  = 8 features
  - Asian handicap (5) + margin + has_asian          = 7 features
  - Over/Under (5) + margin + has_ou                 = 7 features
  - Time/Quality (3)                                  = 3 features
                                                    ─────────────
  Total                                               25 features

Architecture:
    Bucketed features [B, 10, 25, 2]
         ↓  reshape
    Per-feature temporal signature [B, 25, 20]
         ↓  25 × Linear(20, d_model)  +  market_type_embed + feature_id_embed
    Feature embeddings [B, 25, d_model]
         ↓  prepend [CLS]
    [B, 26, d_model]
         ↓  N × CrossMarketTransformerBlock (norm_first=True, SiLU FFN)
    Extract CLS [B, d_model]
         ↓  MLP Head
    [B, 3] logits
"""

import math
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from torch import nn


# ── Constants ──────────────────────────────────────────────────────────

BUCKET_EDGES = [
    (float('inf'), 168, 'opening'),
    (168, 72, '7d'), (72, 24, '3d'), (24, 12, '1d'),
    (12, 6, '12h'), (6, 3, '6h'), (3, 1, '3h'),
    (1, 0.5, '1h'), (0.5, 0, '30m'), (0, -float('inf'), 'closing'),
]
N_BUCKETS = 10
N_AGG = 2  # last, mean


# ── Market types ───────────────────────────────────────────────────────

MARKET_EURO = 0
MARKET_ASIAN = 1
MARKET_OU = 2
MARKET_TIME = 3


# ── Feature list (25 features) ─────────────────────────────────────────

FEATURE_DEFS = [
    # Euro (8)
    ('euro_h', MARKET_EURO), ('euro_d', MARKET_EURO), ('euro_a', MARKET_EURO),
    ('imp_h', MARKET_EURO), ('imp_d', MARKET_EURO), ('imp_a', MARKET_EURO),
    ('euro_margin', MARKET_EURO), ('has_euro', MARKET_EURO),
    # Asian (7)
    ('asian_line', MARKET_ASIAN), ('upper_water', MARKET_ASIAN), ('lower_water', MARKET_ASIAN),
    ('asian_upper_implied', MARKET_ASIAN), ('asian_lower_implied', MARKET_ASIAN),
    ('asian_margin', MARKET_ASIAN), ('has_asian', MARKET_ASIAN),
    # OU (7)
    ('over_under_line', MARKET_OU), ('over_water', MARKET_OU), ('under_water', MARKET_OU),
    ('over_implied', MARKET_OU), ('under_implied', MARKET_OU),
    ('ou_margin', MARKET_OU), ('has_ou', MARKET_OU),
    # Time/Quality (3)
    ('time_log', MARKET_TIME), ('bucket_valid_count', MARKET_TIME), ('market_present_count', MARKET_TIME),
]
N_FEATURES = len(FEATURE_DEFS)  # 25
MARKET_TYPE_COUNT = 4


# ── Feature extraction ────────────────────────────────────────────────

def extract_cross_market_feature(event: Dict, feature_name: str) -> float:
    """Extract a single feature from a raw event dict."""
    try:
        if feature_name == 'euro_h':
            return float(event.get('euro_h', 0) or 0)
        if feature_name == 'euro_d':
            return float(event.get('euro_d', 0) or 0)
        if feature_name == 'euro_a':
            return float(event.get('euro_a', 0) or 0)

        eh = float(event.get('euro_h', 0) or 0)
        ed = float(event.get('euro_d', 0) or 0)
        ea = float(event.get('euro_a', 0) or 0)

        if feature_name == 'imp_h':
            if eh > 0:
                inv_h = 1.0 / eh
                total = inv_h + 1.0 / max(ed, 1e-6) + 1.0 / max(ea, 1e-6)
                return inv_h / total if total > 0 else 1.0 / 3.0
            return 1.0 / 3.0
        if feature_name == 'imp_d':
            if ed > 0:
                inv_d = 1.0 / ed
                total = 1.0 / max(eh, 1e-6) + inv_d + 1.0 / max(ea, 1e-6)
                return inv_d / total if total > 0 else 1.0 / 3.0
            return 1.0 / 3.0
        if feature_name == 'imp_a':
            if ea > 0:
                inv_a = 1.0 / ea
                total = 1.0 / max(eh, 1e-6) + 1.0 / max(ed, 1e-6) + inv_a
                return inv_a / total if total > 0 else 1.0 / 3.0
            return 1.0 / 3.0
        if feature_name == 'euro_margin':
            if eh > 0 and ed > 0 and ea > 0:
                return (1.0 / eh + 1.0 / ed + 1.0 / ea) - 1.0
            return 0.0
        if feature_name == 'has_euro':
            return 1.0 if eh > 0 and ed > 0 and ea > 0 else 0.0

        # Asian
        if feature_name == 'asian_line':
            return float(event.get('asian_line', 0) or 0)
        if feature_name == 'upper_water':
            return float(event.get('upper_water', 0) or 0)
        if feature_name == 'lower_water':
            return float(event.get('lower_water', 0) or 0)

        uw = float(event.get('upper_water', 0) or 0)
        lw = float(event.get('lower_water', 0) or 0)

        if feature_name == 'asian_upper_implied':
            if uw > 0:
                inv_u = 1.0 / uw
                total = inv_u + 1.0 / max(lw, 1e-6)
                return inv_u / total if total > 0 else 0.5
            return 0.5
        if feature_name == 'asian_lower_implied':
            if lw > 0:
                inv_l = 1.0 / lw
                total = 1.0 / max(uw, 1e-6) + inv_l
                return inv_l / total if total > 0 else 0.5
            return 0.5
        if feature_name == 'asian_margin':
            if uw > 0 and lw > 0:
                return (1.0 / uw + 1.0 / lw) - 1.0
            return 0.0
        if feature_name == 'has_asian':
            if 'asian_source' in event:
                src = event['asian_source']
                if src == 'missing':
                    return 0.0
                if src in ('raw_update', 'forward_fill'):
                    return 1.0
            al = float(event.get('asian_line', 0) or 0)
            if al == 0.0 and uw == 1.0 and lw == 1.0:
                return 0.0
            return 1.0 if uw > 0 or lw > 0 else 0.0

        # OU
        if feature_name == 'over_under_line':
            return float(event.get('over_under_line', 0) or 0)
        if feature_name == 'over_water':
            return float(event.get('over_water', 0) or 0)
        if feature_name == 'under_water':
            return float(event.get('under_water', 0) or 0)

        ow = float(event.get('over_water', 0) or 0)
        uw_ou = float(event.get('under_water', 0) or 0)

        if feature_name == 'over_implied':
            if ow > 0:
                inv_o = 1.0 / ow
                total = inv_o + 1.0 / max(uw_ou, 1e-6)
                return inv_o / total if total > 0 else 0.5
            return 0.5
        if feature_name == 'under_implied':
            if uw_ou > 0:
                inv_u = 1.0 / uw_ou
                total = 1.0 / max(ow, 1e-6) + inv_u
                return inv_u / total if total > 0 else 0.5
            return 0.5
        if feature_name == 'ou_margin':
            if ow > 0 and uw_ou > 0:
                return (1.0 / ow + 1.0 / uw_ou) - 1.0
            return 0.0
        if feature_name == 'has_ou':
            if 'over_under_source' in event:
                src = event['over_under_source']
                if src == 'missing':
                    return 0.0
                if src in ('raw_update', 'forward_fill'):
                    return 1.0
            ou_line = float(event.get('over_under_line', 0) or 0)
            if ou_line == 2.5 and ow == 1.0 and uw_ou == 1.0:
                return 0.0
            return 1.0 if ou_line > 0 and ow > 0 and uw_ou > 0 else 0.0

        # Time/Quality
        if feature_name == 'time_log':
            return math.log1p(float(event.get('minutes_before_kickoff', 0)))
        if feature_name == 'bucket_valid_count':
            return 1.0  # will be overwritten in bucketize
        if feature_name == 'market_present_count':
            return 1.0  # will be overwritten in bucketize

        return 0.0
    except Exception:
        return 0.0


# ── Bucketize ─────────────────────────────────────────────────────────

def bucketize_sample_v2(raw_timeline: List[Dict]) -> torch.Tensor:
    """Convert raw timeline events → [N_BUCKETS, N_FEATURES, N_AGG]."""
    buckets = defaultdict(list)
    for e in raw_timeline:
        hours = e.get('minutes_before_kickoff', 0) / 60.0
        for idx, (hi, lo, _) in enumerate(BUCKET_EDGES):
            if hours <= hi and hours > lo:
                buckets[idx].append(e)
                break

    result = torch.zeros(N_BUCKETS, N_FEATURES, N_AGG)
    feature_names = [fd[0] for fd in FEATURE_DEFS]

    for bucket_idx in range(N_BUCKETS):
        evts = sorted(
            buckets.get(bucket_idx, []),
            key=lambda e: e.get('minutes_before_kickoff', 0),
            reverse=True,
        )
        if not evts:
            continue

        for fi, feat_name in enumerate(feature_names):
            if feat_name == 'bucket_valid_count':
                result[bucket_idx, fi, 0] = float(len(evts))
                result[bucket_idx, fi, 1] = float(len(evts))
                continue
            if feat_name == 'market_present_count':
                last_evt = evts[-1]
                has_e = extract_cross_market_feature(last_evt, 'has_euro')
                has_a = extract_cross_market_feature(last_evt, 'has_asian')
                has_o = extract_cross_market_feature(last_evt, 'has_ou')
                cnt = has_e + has_a + has_o
                result[bucket_idx, fi, 0] = cnt
                result[bucket_idx, fi, 1] = cnt
                continue

            vals = []
            for e in evts:
                v = extract_cross_market_feature(e, feat_name)
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
    """Pre-norm Transformer block with RMSNorm, SiLU FFN, bias=False throughout."""

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

class OddsPatchITransformerV2(nn.Module):
    """
    P3b: Cross-market OddsPatch-iTransformer with Euro + Asian + OU features.

    Input:  [B, N_BUCKETS=10, N_FEATURES=25, N_AGG=2]
    Output: [B, 3] logits
    """

    def __init__(self, d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.1,
                 n_buckets=N_BUCKETS, n_features=N_FEATURES, n_agg=N_AGG):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        temporal_dim = n_buckets * n_agg  # 20

        # Per-feature independent embedding
        self.feature_embeds = nn.ModuleList([
            nn.Linear(temporal_dim, d_model, bias=False)
            for _ in range(n_features)
        ])

        # Market type embedding (4 types)
        self.market_type_embed = nn.Embedding(MARKET_TYPE_COUNT, d_model)
        # Feature id embedding
        self.feature_id_embed = nn.Embedding(n_features, d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer layers
        self.layers = nn.ModuleList([
            CrossMarketBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)

        # MLP Head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_ff // 2, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, 3),
        )

        # Pre-compute market type ids
        self.register_buffer('market_type_ids',
            torch.tensor([fd[1] for fd in FEATURE_DEFS], dtype=torch.long))
        self.register_buffer('feature_ids',
            torch.arange(n_features, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # [B, 10, 25, 2] -> [B, 25, 20]
        x = x.permute(0, 2, 1, 3).reshape(B, self.n_features, -1)

        # Per-feature embedding: [B, 25, 64]
        parts = [self.feature_embeds[i](x[:, i]) for i in range(self.n_features)]
        h = torch.stack(parts, dim=1)

        # Add market type and feature id embeddings
        h = h + self.market_type_embed(self.market_type_ids).unsqueeze(0)
        h = h + self.feature_id_embed(self.feature_ids).unsqueeze(0)

        # Prepend CLS: [B, 26, 64]
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)

        # Transformer
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)

        return self.head(h[:, 0])