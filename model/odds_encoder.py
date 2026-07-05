"""
OddsMind Event Encoder — maps raw odds time series features into hidden_size vectors.

This module replaces the token embedding layer of a language model.
Instead of a discrete vocab lookup, it applies a learned continuous projection
(MLP or small conv) to each time-step's feature vector.

Input:  [batch_size, seq_len, feature_dim]
Output: [batch_size, seq_len, hidden_size]
"""

import torch
from torch import nn
from typing import Optional


class OddsEventEncoder(nn.Module):
    """
    Projects each time-step's raw odds features into hidden_size.

    The default is a simple two-layer MLP with SiLU activation.
    This can later be upgraded to a 1-D conv, residual blocks, or a
    small per-time-step Transformer encoder.

    Feature vector (feature_dim=10, P1.15 with over/under):
        0: minutes_before_kickoff
        1: euro_h
        2: euro_d
        3: euro_a
        4: asian_line
        5: upper_water
        6: lower_water
        7: over_under_line
        8: over_water
        9: under_water
    """

    def __init__(
        self,
        feature_dim: int = 10,
        hidden_size: int = 768,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size

        self.proj = nn.Linear(feature_dim, hidden_size, bias=False)

        # P1.17: Per-feature missing embedding — learned replacement for missing values.
        # Shape [1, 1, feature_dim] broadcasts over [B, T, F].
        # Initialized near zero so early training behaves like the no-op baseline.
        self.missing_embed = nn.Parameter(torch.zeros(1, 1, feature_dim))

    def forward(
        self,
        x: torch.Tensor,
        missing_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, feature_dim]
            missing_mask: [batch_size, seq_len, feature_dim] float, 1.0=present, 0.0=missing.
                          P1.17: if provided, replaces missing feature values with a learned
                          per-feature embedding before the MLP projection.
        Returns:
            [batch_size, seq_len, hidden_size]
        """
        if missing_mask is not None:
            # Replace missing positions with learned per-feature embedding.
            # missing_mask=1 → use real value; missing_mask=0 → use learned embed.
            x = x * missing_mask + self.missing_embed * (1.0 - missing_mask)
        return self.proj(x)
