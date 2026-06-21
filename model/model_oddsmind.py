"""
OddsMind — minimal odds time-series prediction model scaffold.

Architecture:
    OddsTimelineInput  [B, T, F]
         ↓
    OddsEventEncoder   [B, T, H]
         ↓
    Transformer Blocks (self-contained minimal encoder, P0.6 → swap to MiniMindBlock)
         ↓
    Mean Pooling       [B, H]
         ↓
    ┌─────────────────┴──────────────────┐
    ↓                                    ↓
    EuroResultHead [B, 3]   AsianResultHead [B, 3]

Key design decisions for P0.1:
- No tokenizer, no vocab, no LM head.
- Minimal self-contained Transformer block (no dependency on model_minimind.py).
- Mean pooling over time dimension (with mask support).
- Two independent classification heads.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


# ── Config ──────────────────────────────────────────────────────────────

@dataclass
class OddsMindConfig:
    """Configuration for OddsMind model.

    All fields have sensible defaults so that `OddsMindConfig()` produces a
    working 4-layer / 256-dim model suitable for smoke tests.
    """
    # Feature input
    feature_dim: int = 7            # euro_h, euro_d, euro_a, asian_line, upper_water, lower_water, minutes_before_kickoff

    # Transformer body
    hidden_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    intermediate_size: int = field(default_factory=lambda: 256 * 4)
    dropout: float = 0.1
    max_seq_len: int = 512

    # Classification heads
    euro_num_classes: int = 3       # home / draw / away
    asian_num_classes: int = 3      # 3-class or 5-class (P0.3)
    head_dropout: float = 0.1

    # Transformer backend (P0.6)
    transformer_backend: str = "odds_native"  # "odds_native" or "minimind"

    # Future extensions
    num_leagues: int = 0            # 0 = no league embedding
    num_bookmakers: int = 0         # 0 = no bookmaker embedding


# ── RMSNorm (self-contained, same formula as MiniMind) ───────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


# ── Minimal Transformer Block ────────────────────────────────────────────

class OddsTransformerBlock(nn.Module):
    """
    Minimal pre-norm Transformer encoder block.

    This is a self-contained implementation for P0.1 scaffold.
    In P0.6 it will be replaced by MiniMindBlock from model.model_minimind.
    """
    def __init__(self, config: OddsMindConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ffn_norm = RMSNorm(config.hidden_size, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with pre-norm
        residual = x
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
        x = residual + attn_out

        # FFN with pre-norm
        residual = x
        x = residual + self.ffn(self.ffn_norm(x))
        return x


# ── Pooling ──────────────────────────────────────────────────────────────

def masked_mean_pool(
    hidden_states: torch.Tensor,     # [B, T, H]
    attention_mask: torch.Tensor,    # [B, T]  bool or float, True/1 = valid
) -> torch.Tensor:                   # [B, H]
    """Mean pool over the time dimension, ignoring masked positions."""
    if attention_mask.dtype != hidden_states.dtype:
        attention_mask = attention_mask.to(hidden_states.dtype)
    mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
    masked = hidden_states * mask
    summed = masked.sum(dim=1)           # [B, H]
    counts = mask.sum(dim=1).clamp(min=1)  # [B, 1]
    return summed / counts


# ── OddsMind Model ───────────────────────────────────────────────────────

class OddsMindModel(nn.Module):
    """
    Minimal OddsMind model: encoder → transformer blocks → pooling → heads.
    """

    def __init__(self, config: Optional[OddsMindConfig] = None):
        super().__init__()
        self.config = config or OddsMindConfig()

        from model.odds_encoder import OddsEventEncoder
        from model.odds_heads import EuroResultHead, AsianResultHead

        self.encoder = OddsEventEncoder(
            feature_dim=self.config.feature_dim,
            hidden_size=self.config.hidden_size,
            dropout=self.config.dropout,
        )

        # ── Transformer backend ──
        backend = self.config.transformer_backend
        if backend == "odds_native":
            self.layers = nn.ModuleList([
                OddsTransformerBlock(self.config)
                for _ in range(self.config.num_hidden_layers)
            ])
            self._use_native_forward = True
        elif backend == "minimind":
            from model.oddsmind_minimind_adapter import MiniMindBlockAdapter
            self._minimind_adapter = MiniMindBlockAdapter(
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_hidden_layers,
                num_heads=self.config.num_attention_heads,
                num_kv_heads=self.config.num_attention_heads // 2,
                max_seq_len=self.config.max_seq_len,
                dropout=self.config.dropout,
            )
            self._use_native_forward = False
        else:
            raise ValueError(f"Unknown transformer_backend: {backend}")

        self.final_norm = RMSNorm(self.config.hidden_size)

        self.euro_head = EuroResultHead(
            hidden_size=self.config.hidden_size,
            num_classes=self.config.euro_num_classes,
            dropout=self.config.head_dropout,
        )

        self.asian_head = AsianResultHead(
            hidden_size=self.config.hidden_size + 3,  # +1 for line, +2 for line_type
            num_classes=self.config.asian_num_classes,
            dropout=self.config.head_dropout,
        )

        from model.odds_heads import ScoreHead
        self.score_head = ScoreHead(
            hidden_size=self.config.hidden_size,
            dropout=self.config.head_dropout,
        )

        # P1.5 bookmaker embedding
        from dataset.odds_dataset import BOOKMAKER_COUNT
        self.bookmaker_embed = nn.Sequential(
            nn.Embedding(BOOKMAKER_COUNT, self.config.hidden_size // 4),
            nn.Linear(self.config.hidden_size // 4, self.config.hidden_size, bias=False),
        )
        # P1.6 consensus feature projection (6 → H)
        self.consensus_proj = nn.Sequential(
            nn.Linear(6, self.config.hidden_size // 4, bias=False),
            nn.SiLU(),
            nn.Linear(self.config.hidden_size // 4, self.config.hidden_size, bias=False),
        )

    def forward(
        self,
        features: torch.Tensor,                  # [B, T, F]
        attention_mask: Optional[torch.Tensor] = None,  # [B, T]  True=valid
        euro_labels: Optional[torch.Tensor] = None,      # [B]
        asian_labels: Optional[torch.Tensor] = None,     # [B]
        score_labels: Optional[torch.Tensor] = None,     # [B, 2] P1.4
        bookmaker_ids: Optional[torch.Tensor] = None,    # [B] P1.5
        consensus_feats: Optional[torch.Tensor] = None,  # [B, 6] P1.6
        score_loss_weight: float = 0.1,
        score_loss_type: str = "mse",  # P1.8A: "mse" or "poisson"
    ) -> dict:
        # ... (encoder + transformer unchanged)
        h = self.encoder(features)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        if self._use_native_forward:
            for layer in self.layers:
                h = layer(h, key_padding_mask=key_padding_mask)
        else:
            h = self._minimind_adapter(h, key_padding_mask=key_padding_mask)
        h = self.final_norm(h)
        if attention_mask is None:
            pooled = h.mean(dim=1)
        else:
            pooled = masked_mean_pool(h, attention_mask)

        # P1.5: add bookmaker embedding to pooled representation
        if bookmaker_ids is not None:
            bk_emb = self.bookmaker_embed(bookmaker_ids)  # [B, H]
            pooled = pooled + bk_emb

        # P1.6: add consensus features to pooled representation
        if consensus_feats is not None:
            consensus_proj = self.consensus_proj(consensus_feats)  # [B, H]
            pooled = pooled + consensus_proj

        euro_logits = self.euro_head(pooled)

        # Extract closing asian_line from features [B, T, 7], index 4 = asian_line
        B = features.shape[0]
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).long() - 1
            lengths = lengths.clamp(min=0)
            closing_line = features[torch.arange(B), lengths, 4:5]  # [B, 1]
        else:
            closing_line = features[:, -1, 4:5]  # [B, 1]

        # Add line type features: half-line (push impossible), quarter-line
        line_val = closing_line.squeeze(-1)  # [B]
        # Integer: line % 1 == 0.  Half: line % 0.5 == 0 AND line % 1 != 0
        line_is_int = (line_val % 1.0).abs() < 1e-5
        line_is_half = (line_val % 0.5).abs() < 1e-5
        line_is_half = line_is_half & (~line_is_int)  # half but not integer
        line_is_quarter = ~line_is_int & ~line_is_half
        line_type_feats = torch.stack([
            line_is_half.float(), line_is_quarter.float()
        ], dim=-1)  # [B, 2]

        asian_input = torch.cat([pooled, closing_line, line_type_feats], dim=-1)  # [B, H+1+2]
        asian_logits = self.asian_head(asian_input)

        # P1.8B: legal class mask — prevent impossible predictions
        num_ac = self.config.asian_num_classes
        legal_mask = torch.ones(B, num_ac, device=features.device)
        if num_ac == 3:
            # 3-class: push (class 1) illegal on half/quarter lines
            push_illegal = line_is_half | line_is_quarter  # [B]
            legal_mask[push_illegal, 1] = -1e9  # push → impossible
        elif num_ac == 5:
            # 5-class: push (class 2) illegal on half/quarter
            push_illegal = line_is_half | line_is_quarter
            legal_mask[push_illegal, 2] = -1e9
            # half_win/half_loss (class 1,3) illegal on integer
            legal_mask[line_is_int, 1] = -1e9
            legal_mask[line_is_int, 3] = -1e9
            # half_win/half_loss illegal on half
            legal_mask[line_is_half, 1] = -1e9
            legal_mask[line_is_half, 3] = -1e9
        asian_logits = asian_logits + legal_mask  # mask out illegal classes

        score_raw = self.score_head(pooled)  # [B, 2] raw output

        # score_preds: always positive (exp for poisson, softplus for mse)
        if score_loss_type == "poisson":
            score_preds = torch.exp(score_raw)
        else:
            score_preds = F.softplus(score_raw)

        result = {"euro_logits": euro_logits, "asian_logits": asian_logits, "score_preds": score_preds}

        # 7. Loss
        if euro_labels is not None and asian_labels is not None:
            euro_loss = F.cross_entropy(euro_logits, euro_labels)
            asian_loss = F.cross_entropy(asian_logits, asian_labels)
            total = euro_loss + asian_loss
            result["euro_loss"] = euro_loss
            result["asian_loss"] = asian_loss
            if score_labels is not None:
                if score_loss_type == "poisson":
                    score_loss = F.poisson_nll_loss(
                        score_raw, score_labels.float(),
                        log_input=True, full=True, reduction="mean",
                    )
                else:
                    score_loss = F.mse_loss(score_preds, score_labels.float())
                total = total + score_loss_weight * score_loss
                result["score_loss"] = score_loss
            result["loss"] = total

        return result
