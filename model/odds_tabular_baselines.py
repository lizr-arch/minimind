"""
OddsMind Trainable Tabular Baselines (P0.5B)

Lightweight models for tabular odds features:
  - OddsLogisticRegression: linear classifier (logistic regression)
  - OddsTinyMLP: small 2-hidden-layer MLP

Both output:
  - euro_logits  [B, 3]
  - asian_logits [B, 3 or 5]

No dependency on the OddsMind Transformer stack.
"""

import torch
import torch.nn.functional as F
from torch import nn

from dataset.odds_features import feature_dim


class OddsLogisticRegression(nn.Module):
    """
    Multi-output logistic regression (one linear layer per head).
    Equivalent to a linear probe on the tabular features.
    """

    def __init__(self, asian_num_classes: int = 3):
        super().__init__()
        self.feat_dim = feature_dim()
        self.asian_num_classes = asian_num_classes

        self.euro_head = nn.Linear(self.feat_dim, 3)
        self.asian_head = nn.Linear(self.feat_dim, asian_num_classes)

    def forward(self, features: torch.Tensor) -> dict:
        """
        Args:
            features: [B, feat_dim]
        Returns:
            {"euro_logits": [B, 3], "asian_logits": [B, asian_num_classes]}
        """
        return {
            "euro_logits": self.euro_head(features),
            "asian_logits": self.asian_head(features),
        }


class OddsTinyMLP(nn.Module):
    """
    Small 2-hidden-layer MLP with SiLU activation and dropout.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        asian_num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feat_dim = feature_dim()
        self.asian_num_classes = asian_num_classes

        self.shared = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.euro_head = nn.Linear(hidden_size, 3)
        self.asian_head = nn.Linear(hidden_size, asian_num_classes)

    def forward(self, features: torch.Tensor) -> dict:
        """
        Args:
            features: [B, feat_dim]
        Returns:
            {"euro_logits": [B, 3], "asian_logits": [B, asian_num_classes]}
        """
        h = self.shared(features)
        return {
            "euro_logits": self.euro_head(h),
            "asian_logits": self.asian_head(h),
        }
