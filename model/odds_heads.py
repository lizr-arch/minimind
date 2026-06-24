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


class ScoreHeadV2(nn.Module):
    """P1.10: Deeper score head with bounded log-rate and auxiliary outputs.

    Architecture:
        hidden → hidden/2 → hidden/4 → score_repr
                                        ├→ home_lambda   (1)
                                        ├→ away_lambda   (1)
                                        ├→ total_goals   (1)  auxiliary
                                        └→ goal_diff     (1)  auxiliary

    Log-rate bounding:
        bounded=True:  5.0 * tanh(raw / 5.0)  — soft gradient at boundary
        bounded=False: clamp(raw, -5, 5)      — hard clamp (baseline)
    """

    def __init__(self, hidden_size: int = 768, dropout: float = 0.1,
                 bounded_log_rate: bool = True, log_rate_bound: float = 5.0):
        super().__init__()
        self.bounded_log_rate = bounded_log_rate
        self.log_rate_bound = log_rate_bound

        mid = hidden_size // 2
        narrow = hidden_size // 4

        self.trunk = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, mid, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, narrow, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.home_head = nn.Linear(narrow, 1, bias=True)
        self.away_head = nn.Linear(narrow, 1, bias=True)
        self.total_head = nn.Linear(narrow, 1, bias=True)
        self.diff_head = nn.Linear(narrow, 1, bias=True)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Returns dict with:
            score_raw:      [B, 2] raw log_lambda (home, away) — unbounded
            score_log_rate: [B, 2] bounded log_lambda
            score_preds:    [B, 2] exp(log_rate) — predicted goals
            total_pred:     [B, 1] predicted total goals
            diff_pred:      [B, 1] predicted goal difference
        """
        r = self.trunk(x)

        home_raw = self.home_head(r)  # [B, 1]
        away_raw = self.away_head(r)  # [B, 1]
        score_raw = torch.cat([home_raw, away_raw], dim=-1)  # [B, 2]

        if self.bounded_log_rate:
            score_log_rate = self.log_rate_bound * torch.tanh(score_raw / self.log_rate_bound)
        else:
            score_log_rate = torch.clamp(score_raw, -self.log_rate_bound, self.log_rate_bound)

        score_preds = torch.exp(score_log_rate)

        total_pred = self.total_head(r)    # [B, 1]
        diff_pred = self.diff_head(r)      # [B, 1]

        return {
            "score_raw": score_raw,
            "score_log_rate": score_log_rate,
            "score_preds": score_preds,
            "total_pred": total_pred,
            "diff_pred": diff_pred,
        }


class ScoreGridHead(nn.Module):
    """P1.13A: Direct 8x8 score grid distribution head.

    Instead of predicting two lambda values, this head directly outputs
    a full probability distribution over 64 scorelines (0-0 through 7+-7+).

    Architecture:
        hidden → hidden/2 → hidden/4 → 64 logits → softmax → [B, 8, 8] probs

    Buckets:
        home_goals: 0,1,2,3,4,5,6,7+
        away_goals: 0,1,2,3,4,5,6,7+

    No Poisson assumption. No independence assumption.
    Model learns the full bivariate distribution directly.
    """

    GRID_SIZE: int = 8          # buckets per team
    NUM_CLASSES: int = 64       # 8*8
    MAX_GOAL: int = 7           # goals >= 7 go into last bucket

    def __init__(self, hidden_size: int = 768, dropout: float = 0.1):
        super().__init__()
        mid = hidden_size // 2
        narrow = hidden_size // 4

        self.trunk = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, mid, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, narrow, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(narrow, self.NUM_CLASSES, bias=True),
        )

        # Precompute index tensors for marginal extraction
        self.register_buffer('_grid_idx', torch.arange(self.NUM_CLASSES))
        self.register_buffer('_home_idx', torch.arange(self.NUM_CLASSES) // self.GRID_SIZE)
        self.register_buffer('_away_idx', torch.arange(self.NUM_CLASSES) % self.GRID_SIZE)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: [B, hidden_size] pooled transformer output

        Returns dict with:
            score_grid_logits: [B, 64] raw logits
            score_grid_probs:  [B, 8, 8] probability grid
            score_grid_flat:   [B, 64] flat probabilities
        """
        logits = self.trunk(x)  # [B, 64]
        probs_flat = torch.softmax(logits, dim=-1)  # [B, 64]
        probs_grid = probs_flat.reshape(-1, self.GRID_SIZE, self.GRID_SIZE)  # [B, 8, 8]

        return {
            "score_grid_logits": logits,
            "score_grid_probs": probs_grid,
            "score_grid_flat": probs_flat,
        }

    @staticmethod
    def bucket_goals(goals: torch.Tensor) -> torch.Tensor:
        """Map goal counts to bucket indices 0..7."""
        return goals.clamp(0, ScoreGridHead.MAX_GOAL).long()

    @staticmethod
    def goals_to_class(home_goals: torch.Tensor, away_goals: torch.Tensor) -> torch.Tensor:
        """Convert (home_goals, away_goals) to flat class index 0..63."""
        hb = ScoreGridHead.bucket_goals(home_goals)
        ab = ScoreGridHead.bucket_goals(away_goals)
        return hb * ScoreGridHead.GRID_SIZE + ab

    @staticmethod
    def extract_marginals(probs_grid: torch.Tensor) -> dict:
        """
        Extract marginal probabilities from [B, 8, 8] score grid.

        Returns dict with:
            home_win_prob:  [B]
            draw_prob:      [B]
            away_win_prob:  [B]
            total_goals_prob: [B, 15] (0..14+)
            goal_diff_prob:   [B, 15] (-7..+7)
        """
        B = probs_grid.shape[0]

        # 1X2 from grid
        home_win = torch.zeros(B, device=probs_grid.device)
        draw = torch.zeros(B, device=probs_grid.device)
        away_win = torch.zeros(B, device=probs_grid.device)
        for i in range(8):
            for j in range(8):
                p = probs_grid[:, i, j]
                if i > j:
                    home_win += p
                elif i == j:
                    draw += p
                else:
                    away_win += p

        # Total goals distribution: 0..14+ (15 bins)
        total_probs = torch.zeros(B, 15, device=probs_grid.device)
        for i in range(8):
            for j in range(8):
                t = min(i + j, 14)
                total_probs[:, t] += probs_grid[:, i, j]

        # Goal diff distribution: -7..+7 (15 bins)
        diff_probs = torch.zeros(B, 15, device=probs_grid.device)
        for i in range(8):
            for j in range(8):
                d = i - j + 7  # shift to 0..14
                diff_probs[:, d] += probs_grid[:, i, j]

        return {
            "home_win_prob": home_win,
            "draw_prob": draw,
            "away_win_prob": away_win,
            "total_goals_prob": total_probs,
            "goal_diff_prob": diff_probs,
        }
