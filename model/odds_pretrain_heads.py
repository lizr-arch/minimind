"""
OddsMind Pretrain Heads (P0.7)

Reconstruction and event prediction heads for self-supervised pretraining.
Independent of the supervised EuroResultHead / AsianResultHead.
"""

import torch
from torch import nn


class OddsReconstructionHead(nn.Module):
    """
    Per-timestep reconstruction head.
    Input:  [B, T, H]
    Output: [B, T, F] reconstructed features
    """

    def __init__(self, hidden_size: int = 256, feature_dim: int = 7, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, feature_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, H]
        Returns:
            [B, T, feature_dim]
        """
        return self.head(x)


class OddsEventPredictionHead(nn.Module):
    """
    Single-event prediction head (for next_event / closing_prediction).
    Input:  [B, H] pooled or last-timestep hidden state
    Output: [B, F] predicted event features
    """

    def __init__(self, hidden_size: int = 256, feature_dim: int = 7, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, feature_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H]
        Returns:
            [B, feature_dim]
        """
        return self.head(x)
