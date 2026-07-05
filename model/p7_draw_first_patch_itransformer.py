"""P7 draw-first factorized Patch-iTransformer objective probe model."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from model.odds_patch_itransformer_v2 import CrossMarketBlock, N_AGG, N_BUCKETS, RMSNorm


class P7DrawFirstPatchITransformer(nn.Module):
    """Patch-iTransformer encoder with a draw-first factorized 1X2 head.

    The head emits two binary logits rather than a final 3-way softmax:
    ``draw_logit`` and ``home_away_logit``. Probabilities are derived by
    ``tools.p7_train_draw_first.draw_first_probs``.
    """

    head_type = "draw_first_factorized"

    def __init__(
        self,
        n_features: int,
        n_buckets: int = N_BUCKETS,
        n_agg: int = N_AGG,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.10,
        market_type_ids: Sequence[int] | None = None,
        n_market_types: int = 4,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.n_buckets = int(n_buckets)
        self.n_agg = int(n_agg)
        self.d_model = int(d_model)

        temporal_dim = self.n_buckets * self.n_agg
        self.feature_embeds = nn.ModuleList(
            [nn.Linear(temporal_dim, self.d_model, bias=False) for _ in range(self.n_features)]
        )
        self.feature_id_embed = nn.Embedding(self.n_features, self.d_model)
        self.register_buffer("feature_ids", torch.arange(self.n_features, dtype=torch.long))

        self.market_type_embed: nn.Embedding | None = None
        if market_type_ids is not None:
            if len(market_type_ids) != self.n_features:
                raise ValueError(
                    f"market_type_ids length must match n_features={self.n_features}, got {len(market_type_ids)}"
                )
            self.market_type_embed = nn.Embedding(int(n_market_types), self.d_model)
            self.register_buffer("market_type_ids", torch.tensor(list(market_type_ids), dtype=torch.long))
        else:
            self.register_buffer("market_type_ids", torch.empty(0, dtype=torch.long))

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.layers = nn.ModuleList(
            [CrossMarketBlock(self.d_model, int(n_heads), int(d_ff), float(dropout)) for _ in range(int(n_layers))]
        )
        self.final_norm = RMSNorm(self.d_model)
        head_hidden = max(16, int(d_ff) // 2)
        self.factorized_head = nn.Sequential(
            nn.Linear(self.d_model, head_hidden, bias=False),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(head_hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | str]:
        expected = (self.n_buckets, self.n_features, self.n_agg)
        if tuple(x.shape[1:]) != expected:
            raise ValueError(f"Expected input shape [B, {expected}], got {tuple(x.shape)}")

        batch_size = x.shape[0]
        x = x.float().permute(0, 2, 1, 3).reshape(batch_size, self.n_features, -1)
        parts = [self.feature_embeds[i](x[:, i]) for i in range(self.n_features)]
        h = torch.stack(parts, dim=1)
        h = h + self.feature_id_embed(self.feature_ids).unsqueeze(0)
        if self.market_type_embed is not None:
            h = h + self.market_type_embed(self.market_type_ids).unsqueeze(0)

        cls = self.cls_token.expand(batch_size, -1, -1)
        h = torch.cat([cls, h], dim=1)
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)
        logits = self.factorized_head(h[:, 0])
        return {
            "draw_logit": logits[:, 0],
            "home_away_logit": logits[:, 1],
            "head_type": self.head_type,
        }
