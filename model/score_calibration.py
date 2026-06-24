"""
P1.12: Score Distribution Calibration — post-hoc correction of independent Poisson score grid.

Does NOT train the model. Applies calibration parameters to the already-computed
lambda_home / lambda_away predictions from a frozen ScoreHeadV2.

Components:
  - Temperature scaling: flatten or sharpen the score grid distribution
  - Lambda scaling: adjust total goal mean
  - Dixon-Coles style low-score correction: adjust 0-0, 1-0, 0-1, 1-1 probabilities

All calibration params are fit on val set only, then applied once on test set.
"""
import math
import json
import torch
from typing import Dict, Optional, Tuple


class ScoreDistributionCalibrator:
    """Lightweight post-hoc calibrator for independent Poisson score grids.

    Parameters:
        total_temperature:  >1 flattens the total-goal distribution (wider)
        diff_temperature:   >1 flattens the goal-diff distribution (wider)
        lambda_scale:       multiplies lambda_home/away (shifts mean)
        dc_rho:             Dixon-Coles rho for low-score correction [-1, 1]
        tail_boost:         additive probability mass boost for extreme scorelines
    """

    def __init__(
        self,
        total_temperature: float = 1.0,
        diff_temperature: float = 1.0,
        lambda_scale: float = 1.0,
        dc_rho: float = 0.0,
        tail_boost: float = 0.0,
    ):
        self.total_temperature = total_temperature
        self.diff_temperature = diff_temperature
        self.lambda_scale = lambda_scale
        self.dc_rho = dc_rho
        self.tail_boost = tail_boost

    def calibrate(
        self,
        lambda_home: float,
        lambda_away: float,
        max_goals: int = 10,
    ) -> Dict:
        """
        Apply full calibration pipeline to a single (lambda_home, lambda_away) pair.

        Returns same dict format as independent_poisson_score_grid().
        """
        from model.score_utils import independent_poisson_score_grid

        # Step 1: Lambda scaling
        lh = lambda_home * self.lambda_scale
        la = lambda_away * self.lambda_scale

        # Step 2: Base independent Poisson grid
        base = independent_poisson_score_grid(lh, la, max_goals)
        grid = base["score_matrix"].clone()  # [G, G]

        # Step 3: Dixon-Coles style low-score correction
        if self.dc_rho != 0.0:
            grid = _apply_dixon_coles_correction(grid, lh, la, self.dc_rho)

        # Step 4: Temperature scaling on total-goal distribution
        if self.total_temperature != 1.0:
            grid = _apply_total_temperature(grid, self.total_temperature)

        # Step 5: Temperature scaling on goal-diff distribution
        if self.diff_temperature != 1.0:
            grid = _apply_diff_temperature(grid, self.diff_temperature)

        # Step 6: Tail boost for extreme scorelines
        if self.tail_boost > 0.0:
            grid = _apply_tail_boost(grid, self.tail_boost)

        # Recompute derived quantities
        G = max_goals + 1
        home_win = 0.0
        draw = 0.0
        away_win = 0.0
        for i in range(G):
            for j in range(G):
                p = grid[i, j].item()
                if i > j:
                    home_win += p
                elif i == j:
                    draw += p
                else:
                    away_win += p

        expected_total = lh + la  # scaled lambdas
        expected_diff = lh - la

        return {
            "home_win_prob": round(home_win, 6),
            "draw_prob": round(draw, 6),
            "away_win_prob": round(away_win, 6),
            "expected_total_goals": round(expected_total, 3),
            "expected_goal_diff": round(expected_diff, 3),
            "score_matrix": grid,
        }

    def to_dict(self) -> Dict:
        return {
            "total_temperature": self.total_temperature,
            "diff_temperature": self.diff_temperature,
            "lambda_scale": self.lambda_scale,
            "dc_rho": self.dc_rho,
            "tail_boost": self.tail_boost,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ScoreDistributionCalibrator":
        return cls(
            total_temperature=d.get("total_temperature", 1.0),
            diff_temperature=d.get("diff_temperature", 1.0),
            lambda_scale=d.get("lambda_scale", 1.0),
            dc_rho=d.get("dc_rho", 0.0),
            tail_boost=d.get("tail_boost", 0.0),
        )

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ScoreDistributionCalibrator":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


# ── Internal calibration transforms ──────────────────────────────────────


def _apply_dixon_coles_correction(
    grid: torch.Tensor,
    lambda_home: float,
    lambda_away: float,
    rho: float,
) -> torch.Tensor:
    """
    Dixon-Coles style low-score correction.

    This is a simplified approximation of the Dixon-Coles (1997) bivariate
    Poisson model. The full model would fit rho via MLE; here we apply a
    heuristic correction to the 0-0, 1-0, 0-1, 1-1 cells.

    Correction factor for cell (i,j):
        if i=0,j=0:  factor = 1 + rho * (lambda_home * lambda_away)
        if i=0,j=1:  factor = 1 - rho * lambda_home
        if i=1,j=0:  factor = 1 - rho * lambda_away
        if i=1,j=1:  factor = 1 + rho

    rho > 0:  more 0-0 and 1-1, fewer 1-0 and 0-1
    rho < 0:  fewer 0-0 and 1-1, more 1-0 and 0-1

    This is NOT a full fitted Dixon-Coles model — it's a style correction
    for demonstration and approximate calibration only.
    """
    G = grid.shape[0]
    corrected = grid.clone()

    # Only apply if grid is large enough
    if G < 2:
        return corrected

    # 0-0
    factor_00 = 1.0 + rho * lambda_home * lambda_away
    corrected[0, 0] *= max(factor_00, 0.0)

    # 1-0
    if G > 1:
        factor_10 = 1.0 - rho * lambda_away
        corrected[1, 0] *= max(factor_10, 0.0)

    # 0-1
    if G > 1:
        factor_01 = 1.0 - rho * lambda_home
        corrected[0, 1] *= max(factor_01, 0.0)

    # 1-1
    if G > 1:
        factor_11 = 1.0 + rho
        corrected[1, 1] *= max(factor_11, 0.0)

    # Renormalize
    total = corrected.sum()
    if total > 0:
        corrected = corrected / total
    else:
        corrected = grid / grid.sum()

    return corrected


def _apply_total_temperature(
    grid: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Apply temperature scaling to total-goal distribution.

    For each total k = i+j, compute P(k) = sum_{i+j=k} P(i,j),
    then raise to power 1/T and renormalize within each k-slice.

    T > 1: flattens total distribution → higher variance
    T < 1: sharpens total distribution → lower variance
    """
    if temperature == 1.0 or temperature <= 0:
        return grid

    G = grid.shape[0]
    max_total = 2 * (G - 1)

    # Compute total-goal distribution
    total_probs = torch.zeros(max_total + 1)
    for k in range(max_total + 1):
        for i in range(G):
            j = k - i
            if 0 <= j < G:
                total_probs[k] += grid[i, j]

    # Apply temperature to total distribution
    eps = 1e-10
    total_probs_raised = (total_probs + eps) ** (1.0 / temperature)
    total_probs_raised = total_probs_raised / total_probs_raised.sum()

    # Redistribute within each k-slice
    calibrated = torch.zeros_like(grid)
    for k in range(max_total + 1):
        slice_mask = torch.zeros(G, G, dtype=torch.bool)
        for i in range(G):
            j = k - i
            if 0 <= j < G:
                slice_mask[i, j] = True

        if slice_mask.sum() > 0 and total_probs[k] > 0:
            # Within slice, keep relative proportions from original grid
            slice_vals = grid[slice_mask]
            if total_probs[k] > eps:
                calibrated[slice_mask] = total_probs_raised[k] * (
                    slice_vals / total_probs[k]
                )

    # Final normalize
    s = calibrated.sum()
    if s > 0:
        calibrated = calibrated / s
    return calibrated


def _apply_diff_temperature(
    grid: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Apply temperature scaling to goal-difference distribution.

    For each diff d = i-j, compute P(d), apply temperature, redistribute
    within each d-slice.

    T > 1: flattens diff distribution → higher diff variance
    """
    if temperature == 1.0 or temperature <= 0:
        return grid

    G = grid.shape[0]
    min_diff = -(G - 1)
    max_diff = G - 1
    n_diffs = max_diff - min_diff + 1

    diff_probs = torch.zeros(n_diffs)
    for i in range(G):
        for j in range(G):
            d = i - j
            idx = d - min_diff
            diff_probs[idx] += grid[i, j]

    eps = 1e-10
    diff_probs_raised = (diff_probs + eps) ** (1.0 / temperature)
    diff_probs_raised = diff_probs_raised / diff_probs_raised.sum()

    calibrated = torch.zeros_like(grid)
    for d_idx in range(n_diffs):
        d = d_idx + min_diff
        slice_mask = torch.zeros(G, G, dtype=torch.bool)
        slice_sum = 0.0
        for i in range(G):
            j = i - d
            if 0 <= j < G:
                slice_mask[i, j] = True
                slice_sum += grid[i, j].item()

        if slice_mask.sum() > 0 and slice_sum > eps:
            calibrated[slice_mask] = diff_probs_raised[d_idx] * (
                grid[slice_mask] / slice_sum
            )

    s = calibrated.sum()
    if s > 0:
        calibrated = calibrated / s
    return calibrated


def _apply_tail_boost(
    grid: torch.Tensor,
    boost: float,
) -> torch.Tensor:
    """
    Boost extreme scoreline probabilities.

    Adds 'boost' probability mass evenly to cells with high total goals
    (>=5) or large goal difference (>=3), then renormalizes.
    """
    if boost <= 0:
        return grid

    G = grid.shape[0]
    boosted = grid.clone()

    for i in range(G):
        for j in range(G):
            total = i + j
            diff = abs(i - j)
            if total >= 5 or diff >= 3:
                boosted[i, j] += boost / (G * G)  # spread evenly

    s = boosted.sum()
    if s > 0:
        boosted = boosted / s
    return boosted
