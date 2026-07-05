"""P6 residual Patch-iTransformer objective-probe model."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from model.odds_patch_itransformer_v2 import CrossMarketBlock, N_AGG, N_BUCKETS, RMSNorm


class GoalCountRateHead(nn.Module):
    """Predict bounded independent Poisson rates for home and away goals."""

    def __init__(self, hidden_size: int, min_rate: float = 0.05, max_rate: float = 5.50) -> None:
        super().__init__()
        if min_rate <= 0 or max_rate <= min_rate:
            raise ValueError("GoalCountRateHead requires 0 < min_rate < max_rate")
        self.min_rate = float(min_rate)
        self.max_rate = float(max_rate)
        self.linear = nn.Linear(int(hidden_size), 2)
        self.log_lambda_min = float(torch.log(torch.tensor(self.min_rate)).item())
        self.log_lambda_max = float(torch.log(torch.tensor(self.max_rate)).item())

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        raw = self.linear(pooled)
        log_lambdas = self.log_lambda_min + (self.log_lambda_max - self.log_lambda_min) * torch.sigmoid(raw)
        return torch.exp(log_lambdas).clamp(min=self.min_rate, max=self.max_rate)

    def initialize_from_means(self, mean_home_goals: float, mean_away_goals: float) -> None:
        targets = torch.tensor([float(mean_home_goals), float(mean_away_goals)], dtype=self.linear.bias.dtype)
        targets = targets.clamp(min=self.min_rate, max=self.max_rate)
        target_log = torch.log(targets)
        position = (target_log - self.log_lambda_min) / (self.log_lambda_max - self.log_lambda_min)
        position = position.clamp(1e-6, 1.0 - 1e-6)
        raw = torch.log(position / (1.0 - position))
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.copy_(raw.to(self.linear.bias.device))


class P6ResidualPatchITransformer(nn.Module):
    """Patch-iTransformer encoder with P4-style residual 1X2 and goal-diff heads.

    Input shape is ``[B, n_buckets, n_features, n_agg]``. The model encodes each
    selected feature's bucket history independently, prepends a CLS token, and
    emits residual logits that are combined with the Euro anchor by the P4 loss.
    """

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
        enable_draw_aux_head: bool = False,
        enable_goal_count_head: bool = False,
        enable_draw_risk_head: bool = False,
        n_draw_risk_classes: int = 3,
        enable_ah_cover_head: bool = False,
        n_ah_cover_classes: int = 5,
        enable_ah_unit_head: bool = False,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.n_buckets = int(n_buckets)
        self.n_agg = int(n_agg)
        self.d_model = int(d_model)
        self.enable_draw_aux_head = bool(enable_draw_aux_head)
        self.enable_goal_count_head = bool(enable_goal_count_head)
        self.enable_draw_risk_head = bool(enable_draw_risk_head)
        self.enable_ah_cover_head = bool(enable_ah_cover_head)
        self.enable_ah_unit_head = bool(enable_ah_unit_head)

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
        self.delta_1x2_head = nn.Sequential(
            nn.Linear(self.d_model, head_hidden, bias=False),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(head_hidden, 3),
        )
        self.goal_diff_head = nn.Sequential(
            nn.Linear(self.d_model, head_hidden, bias=False),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(head_hidden, 7),
        )
        self.draw_aux_head: nn.Sequential | None = None
        if self.enable_draw_aux_head:
            self.draw_aux_head = nn.Sequential(
                nn.Linear(self.d_model, head_hidden, bias=False),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(head_hidden, 1),
            )
        self.goal_count_head: GoalCountRateHead | None = None
        if self.enable_goal_count_head:
            self.goal_count_head = GoalCountRateHead(self.d_model)
        self.draw_risk_head: nn.Sequential | None = None
        if self.enable_draw_risk_head:
            self.draw_risk_head = nn.Sequential(
                nn.Linear(self.d_model, head_hidden, bias=False),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(head_hidden, int(n_draw_risk_classes)),
            )
        self.ah_cover_head: nn.Sequential | None = None
        if self.enable_ah_cover_head:
            self.ah_cover_head = nn.Sequential(
                nn.Linear(self.d_model, head_hidden, bias=False),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(head_hidden, int(n_ah_cover_classes)),
            )
        self.ah_unit_head: nn.Sequential | None = None
        if self.enable_ah_unit_head:
            self.ah_unit_head = nn.Sequential(
                nn.Linear(self.d_model, head_hidden, bias=False),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(head_hidden, 1),
                nn.Tanh(),
            )

    def encode_pooled(self, x: torch.Tensor) -> torch.Tensor:
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
        return h[:, 0]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        cls_h = self.encode_pooled(x)
        out = {
            "delta_logits": self.delta_1x2_head(cls_h),
            "goal_diff_logits": self.goal_diff_head(cls_h),
        }
        if self.draw_aux_head is not None:
            out["draw_aux_logit"] = self.draw_aux_head(cls_h).squeeze(-1)
        if self.goal_count_head is not None:
            out["goal_count_lambdas"] = self.goal_count_head(cls_h)
        if self.draw_risk_head is not None:
            out["draw_risk_logits"] = self.draw_risk_head(cls_h)
        if self.ah_cover_head is not None:
            out["ah_cover_logits"] = self.ah_cover_head(cls_h)
        if self.ah_unit_head is not None:
            out["ah_unit_pred"] = self.ah_unit_head(cls_h).squeeze(-1)
        return out
