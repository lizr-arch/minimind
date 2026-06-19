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
            hidden_size=self.config.hidden_size,
            num_classes=self.config.asian_num_classes,
            dropout=self.config.head_dropout,
        )

    def forward(
        self,
        features: torch.Tensor,                  # [B, T, F]
        attention_mask: Optional[torch.Tensor] = None,  # [B, T]  True=valid
        euro_labels: Optional[torch.Tensor] = None,      # [B]
        asian_labels: Optional[torch.Tensor] = None,     # [B]
    ) -> dict:
        """
        Returns dict with keys:
            euro_logits, asian_logits, loss (if labels provided)
        """
        # 1. Encode raw odds features → hidden_size
        h = self.encoder(features)  # [B, T, H]

        # 2. Build key_padding_mask from attention_mask
        #    nn.MultiheadAttention expects key_padding_mask where True = PAD (inverted logic)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()  # True where PAD

        # 3. Pass through Transformer blocks (backend-specific)
        if self._use_native_forward:
            for layer in self.layers:
                h = layer(h, key_padding_mask=key_padding_mask)
        else:
            h = self._minimind_adapter(h, key_padding_mask=key_padding_mask)

        # 4. Final norm
        h = self.final_norm(h)

        # 5. Pool over time
        if attention_mask is None:
            pooled = h.mean(dim=1)  # [B, H]
        else:
            pooled = masked_mean_pool(h, attention_mask)

        # 6. Classification heads
        euro_logits = self.euro_head(pooled)    # [B, 3]
        asian_logits = self.asian_head(pooled)  # [B, 3]

        result = {"euro_logits": euro_logits, "asian_logits": asian_logits}

        # 7. Loss
        if euro_labels is not None and asian_labels is not None:
            euro_loss = F.cross_entropy(euro_logits, euro_labels)
            asian_loss = F.cross_entropy(asian_logits, asian_labels)
            result["loss"] = euro_loss + asian_loss
            result["euro_loss"] = euro_loss
            result["asian_loss"] = asian_loss

        return result
