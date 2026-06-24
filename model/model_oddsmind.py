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
    ┌─────────────────┴───────────────────┐
    ↓                                     ↓
    EuroResultHead [B, 3]    AsianResultHead [B, 3]
    ↓
    ScoreHeadV2  [B, 2] + total/diff aux  (P1.10)

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
    feature_dim: int = 13           # auto-set from schema: v1=10, v2=13
    feature_schema_version: str = "v2"  # "v1"=10dim no mask, "v2"=13dim with availability mask

    # Transformer body
    hidden_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    intermediate_size: int = field(default_factory=lambda: 256 * 4)
    dropout: float = 0.1
    max_seq_len: int = 512

    # P1.16: Pooling mode
    pooling_mode: str = "mean"       # "mean" | "attention" | "cls"

    # Classification heads
    euro_num_classes: int = 3       # home / draw / away
    asian_num_classes: int = 3      # 3-class or 5-class (P0.3)
    head_dropout: float = 0.1

    # Transformer backend (P0.6)
    transformer_backend: str = "odds_native"  # "odds_native" or "minimind"

    # Score head (P1.10)
    score_head_version: str = "v2"  # "v1" (old) | "v2" (deeper+bounded+aux) | "grid" (8x8 ScoreGridHead P1.13A)
    score_bounded_log_rate: bool = True  # True=tanh soft bound, False=hard clamp
    score_log_rate_bound: float = 5.0

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


# ── P1.16: Alternative pooling methods ──────────────────────────────────

class AttentionPooling(nn.Module):
    """
    Learnable attention pooling: a query vector attends over [B, T, H]
    to produce a weighted sum [B, H].
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=1,
            batch_first=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, T, H]
        attention_mask: torch.Tensor,  # [B, T] True=valid
    ) -> torch.Tensor:                 # [B, H]
        B = hidden_states.shape[0]
        query = self.query.expand(B, -1, -1)  # [B, 1, H]
        # key_padding_mask: True=PAD (inverted from attention_mask)
        key_pad = ~attention_mask.bool() if attention_mask is not None else None
        out, _ = self.attn(query, hidden_states, hidden_states, key_padding_mask=key_pad)
        return out.squeeze(1)  # [B, H]


class CLSTokenPooling(nn.Module):
    """
    CLS-token pooling: prepends a learnable [CLS] token, uses its final hidden state.

    This module is instantiated per-call with the sequence, so it must be integrated
    at the model level (prepend token before encoder, extract after).
    For simplicity, we use a per-layer approach: the cls token is added before
    the transformer blocks and its final state extracted after.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

    def prepend(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Prepend CLS token: [B, T, H] → [B, 1+T, H]"""
        B = hidden_states.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        return torch.cat([cls, hidden_states], dim=1)

    @staticmethod
    def extract(hidden_states: torch.Tensor) -> torch.Tensor:
        """Extract CLS token (first position): [B, 1+T, H] → [B, H]"""
        return hidden_states[:, 0, :]

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

        # P1.16: Pooling mode
        self._pooling_mode = self.config.pooling_mode
        if self._pooling_mode == "attention":
            self.attention_pool = AttentionPooling(self.config.hidden_size)
        elif self._pooling_mode == "cls":
            self.cls_pool = CLSTokenPooling(self.config.hidden_size)

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

        from model.odds_heads import ScoreHead, ScoreHeadV2, ScoreGridHead
        if self.config.score_head_version == "grid":
            self.score_head = ScoreGridHead(
                hidden_size=self.config.hidden_size,
                dropout=self.config.head_dropout,
            )
        elif self.config.score_head_version == "v2":
            self.score_head = ScoreHeadV2(
                hidden_size=self.config.hidden_size,
                dropout=self.config.head_dropout,
                bounded_log_rate=self.config.score_bounded_log_rate,
                log_rate_bound=self.config.score_log_rate_bound,
            )
        else:
            self.score_head = ScoreHead(
                hidden_size=self.config.hidden_size,
                dropout=self.config.head_dropout,
            )
        self._score_head_v2 = (self.config.score_head_version == "v2")
        self._score_head_grid = (self.config.score_head_version == "grid")

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
        missing_mask: Optional[torch.Tensor] = None,     # [B, T, F] P1.16
        score_loss_weight: float = 0.1,
        score_loss_type: str = "mse",  # P1.8A: "mse" or "poisson"
    ) -> dict:
        # P1.16: pass missing_mask to encoder (no-op passthrough for now)
        h = self.encoder(features, missing_mask=missing_mask)

        # P1.16: CLS token prepend (before transformer)
        if self._pooling_mode == "cls":
            h = self.cls_pool.prepend(h)
            # Extend attention_mask with a valid position for the CLS token
            if attention_mask is not None:
                cls_mask = torch.ones(attention_mask.shape[0], 1, dtype=torch.bool, device=attention_mask.device)
                attention_mask = torch.cat([cls_mask, attention_mask], dim=1)

        # Store original attention_mask for closing_line extraction (before CLS prepend)
        _orig_attention_mask = attention_mask
        if self._pooling_mode == "cls" and attention_mask is not None:
            # Remove CLS position for feature indexing
            _orig_attention_mask = attention_mask[:, 1:]

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        if self._use_native_forward:
            for layer in self.layers:
                h = layer(h, key_padding_mask=key_padding_mask)
        else:
            h = self._minimind_adapter(h, key_padding_mask=key_padding_mask)
        h = self.final_norm(h)

        # P1.16: Pooling — route by pooling_mode
        if self._pooling_mode == "cls":
            # CLS token was prepended before transformer; extract its final state
            pooled = CLSTokenPooling.extract(h)
            h = h[:, 1:, :]  # strip cls for any downstream use (not used)
        elif self._pooling_mode == "attention":
            if attention_mask is None:
                pooled = self.attention_pool(h, torch.ones(h.shape[0], h.shape[1], dtype=torch.bool, device=h.device))
            else:
                pooled = self.attention_pool(h, attention_mask)
        else:
            # "mean" (default)
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
        if _orig_attention_mask is not None:
            lengths = _orig_attention_mask.sum(dim=1).long() - 1
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
        # P1.18 fix: save raw logits for loss computation (no mask penalty).
        # Only apply mask to the output logits used for prediction.
        asian_logits_raw = asian_logits  # for loss
        num_ac = self.config.asian_num_classes
        MASK_VAL = -1e4
        legal_mask = torch.ones(B, num_ac, device=features.device)
        if num_ac == 3:
            push_illegal = line_is_half | line_is_quarter
            legal_mask[push_illegal, 1] = MASK_VAL
        elif num_ac == 5:
            push_illegal = line_is_half | line_is_quarter
            legal_mask[push_illegal, 2] = MASK_VAL
            legal_mask[line_is_int, 1] = MASK_VAL
            legal_mask[line_is_int, 3] = MASK_VAL
            legal_mask[line_is_half, 1] = MASK_VAL
            legal_mask[line_is_half, 3] = MASK_VAL
        asian_logits = asian_logits + legal_mask  # masked for prediction

        # ── P1.10/P1.13A Score Head ──
        if self._score_head_grid:
            score_out = self.score_head(pooled)  # dict with grid_logits, grid_probs
            score_grid_logits = score_out["score_grid_logits"]  # [B, 64]
            score_grid_probs = score_out["score_grid_probs"]    # [B, 8, 8]
            score_preds = score_out["score_grid_flat"]           # [B, 64]
            total_pred = None
            diff_pred = None
            score_raw = score_grid_logits  # for range monitoring
            score_log_rate = score_grid_logits
            # Compute expected goals from grid for eval compatibility
            from model.score_grid_utils import score_grid_predictions
            grid_preds = score_grid_predictions(score_grid_probs)
            score_preds_2d = torch.stack([
                grid_preds["expected_home_goals"],
                grid_preds["expected_away_goals"]
            ], dim=-1)  # [B, 2]
        elif self._score_head_v2:
            score_out = self.score_head(pooled)  # dict
            score_raw = score_out["score_raw"]           # [B, 2]
            score_log_rate = score_out["score_log_rate"]  # [B, 2] bounded
            score_preds = score_out["score_preds"]        # [B, 2]
            total_pred = score_out["total_pred"]          # [B, 1]
            diff_pred = score_out["diff_pred"]            # [B, 1]
        else:
            score_raw = self.score_head(pooled)  # [B, 2]
            # P1.10: clamp eval preds too, log raw bounds for monitoring
            score_log_rate = torch.clamp(score_raw, -5.0, 5.0)
            if score_loss_type == "poisson":
                score_preds = torch.exp(score_log_rate)
            else:
                score_preds = F.softplus(score_raw)
            total_pred = None
            diff_pred = None

        result = {
            "euro_logits": euro_logits,
            "asian_logits": asian_logits,
            "score_preds": score_preds_2d if self._score_head_grid else score_preds,
            "score_raw_min": score_raw.min().item(),
            "score_raw_max": score_raw.max().item(),
            "score_log_rate_min": score_log_rate.min().item(),
            "score_log_rate_max": score_log_rate.max().item(),
        }
        if self._score_head_grid:
            result["score_grid_probs"] = score_grid_probs
            result["score_grid_logits"] = score_grid_logits

        # 7. Loss (computed on raw logits, not masked)
        if euro_labels is not None and asian_labels is not None:
            euro_loss = F.cross_entropy(euro_logits, euro_labels)
            asian_loss = F.cross_entropy(asian_logits_raw, asian_labels)
            total = euro_loss + asian_loss
            result["euro_loss"] = euro_loss
            result["asian_loss"] = asian_loss
            if score_labels is not None:
                if self._score_head_grid:
                    # P1.13A: ScoreGridHead loss — CE + marginal losses
                    from model.score_grid_utils import compute_score_grid_loss
                    score_loss, aux_losses = compute_score_grid_loss(
                        score_grid_logits, score_labels,
                        ce_weight=1.0, result_weight=0.3,
                        total_weight=0.3, diff_weight=0.3,
                        soft_target=True, soft_self_weight=0.75,
                    )
                    aux_losses = {k: v for k, v in aux_losses.items() if k != "total"}
                    result["aux_losses"] = aux_losses
                elif score_loss_type == "poisson":
                    score_loss = F.poisson_nll_loss(
                        score_log_rate, score_labels.float(),
                        log_input=True, full=True, reduction="mean",
                    )
                    # P1.10: auxiliary losses (total_goals, goal_diff)
                    aux_losses = {}
                    if self._score_head_v2 and total_pred is not None:
                        total_true = score_labels.sum(-1, keepdim=True).float()  # [B, 1]
                        diff_true = (score_labels[:, 0:1] - score_labels[:, 1:2]).float()  # [B, 1]
                        total_aux = F.smooth_l1_loss(total_pred, total_true)
                        diff_aux = F.smooth_l1_loss(diff_pred, diff_true)
                        score_loss = score_loss + 0.2 * total_aux + 0.2 * diff_aux
                        aux_losses["total_aux"] = total_aux.item()
                        aux_losses["diff_aux"] = diff_aux.item()
                    result["aux_losses"] = aux_losses
                else:
                    score_loss = F.mse_loss(score_preds, score_labels.float())
                total = total + score_loss_weight * score_loss
                result["score_loss"] = score_loss
            result["loss"] = total

        return result
