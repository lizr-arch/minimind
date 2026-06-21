"""
OddsMind heads.

EuroResultHead:  3-class (home / draw / away)
AsianResultHead: 3-class (upper / push / lower)
                 5-class (upper_full_win / upper_half_win / push /
                           upper_half_loss / upper_full_loss)
ScoreHead:       regress home_goals, away_goals (P1.4)
"""

import torch
from torch import nn


class EuroResultHead(nn.Module):
    """Predicts home / draw / away from pooled hidden state."""

    def __init__(self, hidden_size: int = 768, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 4, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_classes, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class AsianResultHead(nn.Module):
    """Predicts upper / push / lower from pooled hidden state."""

    def __init__(self, hidden_size: int = 768, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 4, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_classes, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class ScoreHead(nn.Module):
    """Predicts expected goals (home, away) from pooled hidden state.  P1.4 / P1.8A"""

    def __init__(self, hidden_size: int = 768, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 4, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 2, bias=True),  # raw output [B, 2]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)  # [B, 2]: raw logits for score
