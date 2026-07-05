"""P12 explicit score-prior residual model."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from model.p6_residual_patch_itransformer import GoalCountRateHead, P6ResidualPatchITransformer


class ResidualDeltaHead(nn.Module):
    """A zero-initialized 3-logit residual head."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(int(hidden_size), 3)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.linear(pooled)


class P12ScorePriorResidualITransformer(P6ResidualPatchITransformer):
    """P6 Patch-iTransformer encoder with P12 score-prior residual heads."""

    def __init__(
        self,
        n_features: int,
        n_buckets: int = 10,
        n_agg: int = 2,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.10,
        market_type_ids: Sequence[int] | None = None,
        n_market_types: int = 4,
        enable_anchor_head: bool = False,
    ) -> None:
        super().__init__(
            n_features=n_features,
            n_buckets=n_buckets,
            n_agg=n_agg,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            market_type_ids=market_type_ids,
            n_market_types=n_market_types,
            enable_goal_count_head=False,
            enable_draw_aux_head=False,
        )
        self.goal_count_head = GoalCountRateHead(self.d_model)
        self.residual_delta_head = ResidualDeltaHead(self.d_model)
        self.enable_anchor_head = bool(enable_anchor_head)
        self.anchor_head: nn.Module | None = None
        if self.enable_anchor_head:
            head_hidden = max(16, int(d_ff) // 2)
            self.anchor_head = nn.Sequential(
                nn.Linear(self.d_model, head_hidden, bias=False),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(head_hidden, 3),
            )

    def initialize_goal_count_from_means(self, mean_home_goals: float, mean_away_goals: float) -> None:
        self.goal_count_head.initialize_from_means(mean_home_goals, mean_away_goals)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = self.encode_pooled(x)
        out = {
            "residual_delta": self.residual_delta_head(pooled),
            "goal_count_lambdas": self.goal_count_head(pooled),
        }
        if self.anchor_head is not None:
            out["anchor_logits"] = self.anchor_head(pooled)
        return out
