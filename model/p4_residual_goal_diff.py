"""P4 residual goal-diff objective-probe model."""

from __future__ import annotations

import torch
from torch import nn


class P4ResidualGoalDiffModel(nn.Module):
    """A compact MLP probe over bucketized odds tensors.

    The first P4 iteration deliberately uses a flat bucket-tensor encoder to
    validate the objective before copying it into a Transformer variant.
    """

    def __init__(
        self,
        n_features: int,
        n_buckets: int = 10,
        n_agg: int = 2,
        d_model: int = 128,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.n_buckets = int(n_buckets)
        self.n_agg = int(n_agg)
        input_dim = self.n_buckets * self.n_features * self.n_agg
        hidden_dim = max(16, int(d_model))
        mid_dim = max(16, hidden_dim // 2)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.delta_1x2_head = nn.Linear(mid_dim, 3)
        self.goal_diff_head = nn.Linear(mid_dim, 7)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        expected = (self.n_buckets, self.n_features, self.n_agg)
        if tuple(x.shape[1:]) != expected:
            raise ValueError(f"Expected input shape [B, {expected}], got {tuple(x.shape)}")
        h = self.encoder(x.float())
        return {
            "delta_logits": self.delta_1x2_head(h),
            "goal_diff_logits": self.goal_diff_head(h),
        }
