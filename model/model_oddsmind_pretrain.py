"""
OddsMind Pretrain Model Wrapper (P0.7)

Wraps OddsEventEncoder + Transformer backend + pretrain heads
for self-supervised pretraining tasks.

Does NOT use euro/asian classification heads, labels, tokenizer, or LM head.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from model.odds_encoder import OddsEventEncoder
from model.odds_pretrain_heads import OddsReconstructionHead, OddsEventPredictionHead


@dataclass
class OddsPretrainConfig:
    task: str = "masked_reconstruction"
    feature_dim: int = 7
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_len: int = 512
    transformer_backend: str = "odds_native"

    # Aliases needed by OddsTransformerBlock (native backend)
    @property
    def num_attention_heads(self): return self.num_heads

    @property
    def intermediate_size(self):
        import math
        return int(math.ceil(self.hidden_size * math.pi / 64) * 64)

    @property
    def num_key_value_heads(self): return None


class OddsMindPretrainModel(nn.Module):
    def __init__(self, config: Optional[OddsPretrainConfig] = None):
        super().__init__()
        self.config = config or OddsPretrainConfig()

        self.encoder = OddsEventEncoder(
            feature_dim=self.config.feature_dim,
            hidden_size=self.config.hidden_size,
            dropout=self.config.dropout,
        )

        # Transformer backend
        backend = self.config.transformer_backend
        if backend == "odds_native":
            from model.model_oddsmind import OddsTransformerBlock
            self.layers = nn.ModuleList([
                OddsTransformerBlock(self.config)
                for _ in range(self.config.num_layers)
            ])
            self._forward_transformer = self._forward_native
        elif backend == "minimind":
            from model.oddsmind_minimind_adapter import MiniMindBlockAdapter
            self._minimind_adapter = MiniMindBlockAdapter(
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads,
                num_kv_heads=self.config.num_heads // 2,
                max_seq_len=self.config.max_seq_len,
                dropout=self.config.dropout,
            )
            self._forward_transformer = self._forward_minimind
        else:
            raise ValueError(f"Unknown backend: {backend}")

        from model.model_oddsmind import RMSNorm
        self.final_norm = RMSNorm(self.config.hidden_size)

        self.recon_head = OddsReconstructionHead(
            hidden_size=self.config.hidden_size,
            feature_dim=self.config.feature_dim,
        )
        self.event_head = OddsEventPredictionHead(
            hidden_size=self.config.hidden_size,
            feature_dim=self.config.feature_dim,
        )

    def _forward_native(self, h, key_padding_mask):
        for layer in self.layers:
            h = layer(h, key_padding_mask=key_padding_mask)
        return h

    def _forward_minimind(self, h, key_padding_mask):
        return self._minimind_adapter(h, key_padding_mask=key_padding_mask)

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_features: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        target_event: Optional[torch.Tensor] = None,
    ) -> dict:
        B, T, F = features.shape

        # Encode
        h = self.encoder(features)

        # Build key padding mask
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        # Transformer
        h = self._forward_transformer(h, key_padding_mask)
        h = self.final_norm(h)

        task = self.config.task
        result = {}

        if task == "masked_reconstruction":
            recon = self.recon_head(h)  # [B, T, F]
            result["reconstructed_features"] = recon
            if target_features is not None and target_mask is not None:
                diff = (recon - target_features) * target_mask.float()
                mse = (diff.pow(2).sum()) / target_mask.sum().clamp(min=1)
                result["loss"] = mse

        elif task in ("next_event", "closing_prediction"):
            # Pool: use the last valid position for each sample
            if attention_mask is not None:
                # Find last valid index per sample
                lengths = attention_mask.sum(dim=1).long() - 1  # [B]
                lengths = lengths.clamp(min=0)
                pooled = h[torch.arange(B), lengths]  # [B, H]
            else:
                pooled = h[:, -1, :]  # [B, H]

            pred_event = self.event_head(pooled)  # [B, F]
            result["predicted_event"] = pred_event
            if target_event is not None:
                mse = torch.nn.functional.mse_loss(pred_event, target_event, reduction="mean")
                result["loss"] = mse

        return result
