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


class OddsEventEncoder(nn.Module):
    """
    Projects each time-step's raw odds features into hidden_size.

    The default is a simple two-layer MLP with SiLU activation.
    This can later be upgraded to a 1-D conv, residual blocks, or a
    small per-time-step Transformer encoder.

    Feature vector (feature_dim=7, the v0.1 default):
        0: minutes_before_kickoff
        1: euro_h
        2: euro_d
        3: euro_a
        4: asian_line
        5: upper_water
        6: lower_water
    """

    def __init__(
        self,
        feature_dim: int = 7,
        hidden_size: int = 768,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size

        self.proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_size, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, feature_dim]
        Returns:
            [batch_size, seq_len, hidden_size]
        """
        return self.proj(x)
