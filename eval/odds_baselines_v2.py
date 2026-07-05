"""
Phase 1a Baseline v2 — 欧赔+亚盘联合最优分布（方案 C）

Replaces the pure-euro implied-prob baseline with a joint euro+asian
optimal 1X2 probability distribution.

Supported handicap types:
  - No asian data        → pure euro no-vig
  - line = 0.0 (平手)    → euro for p_draw, asian for home/away split
  - line = ±0.5 (半球)   → direct substitution, scale remaining two outcomes
  - line = ±1.0 (一球)   → TODO: Poisson (fallback to pure euro)
  - line = ±0.25, ±0.75  → TODO: Poisson (fallback to pure euro)

All outputs are torch Tensors of shape [3] = [p_home, p_draw, p_away].

Usage:
    from eval.odds_baselines_v2 import Phase1Baseline

    baseline = Phase1Baseline()
    probs = baseline.compute(
        close_euro_h=2.30, close_euro_d=3.20, close_euro_a=3.10,
        close_asian_line=-0.5,
        close_asian_upper_water=0.92, close_asian_lower_water=0.88,
    )
    # → tensor([0.xxx, 0.xxx, 0.xxx])
"""

import math
from typing import Optional

import torch

EPS = 1e-9


class Phase1Baseline:
    """Joint euro+asian optimal 1X2 probability distribution."""

    # ── Public API ──────────────────────────────────────────────────

    def compute(
        self,
        close_euro_h: Optional[float],
        close_euro_d: Optional[float],
        close_euro_a: Optional[float],
        close_asian_line: Optional[float],
        close_asian_upper_water: Optional[float],
        close_asian_lower_water: Optional[float],
    ) -> torch.Tensor:
        """
        Compute joint optimal [p_h, p_d, p_a] from closing prices.

        Falls back to pure euro no-vig when asian data is unavailable
        or the handicap type is not yet implemented.
        """
        # ── Euro prior (always required) ──
        if not self._has_euro(close_euro_h, close_euro_d, close_euro_a):
            return torch.full((3,), 1.0 / 3.0)

        q_h, q_d, q_a = self.compute_euro_prior(
            close_euro_h, close_euro_d, close_euro_a
        )

        # ── No asian → pure euro ──
        if not self._has_asian(close_asian_line, close_asian_upper_water, close_asian_lower_water):
            return torch.tensor([q_h, q_d, q_a])

        p_upper = self.compute_asian_p_upper(
            close_asian_upper_water, close_asian_lower_water
        )

        line = float(close_asian_line)

        # ── Dispatch by handicap type ──
        try:
            if line == 0.0:
                return self.combine_line_zero(q_h, q_d, q_a, p_upper)
            elif line == -0.5:
                return self.combine_half_ball(q_h, q_d, q_a, p_upper, home_gives=True)
            elif line == 0.5:
                return self.combine_half_ball(q_h, q_d, q_a, p_upper, home_gives=False)
            elif line in (-1.0, 1.0, -0.25, 0.25, -0.75, 0.75):
                return self._solve_poisson_from_constraints(
                    q_h, q_d, q_a, p_upper, line
                )
            else:
                # Unknown line → pure euro
                return torch.tensor([q_h, q_d, q_a])
        except (ValueError, ZeroDivisionError):
            return torch.tensor([q_h, q_d, q_a])

    # ── Euro prior ─────────────────────────────────────────────────

    @staticmethod
    def compute_euro_prior(
        euro_h: float, euro_d: float, euro_a: float
    ):
        """
        Convert decimal odds to no-vig implied probabilities via
        1/odds normalisation.

        Returns (q_h, q_d, q_a) as three floats summing to 1.
        """
        raw_h = 1.0 / max(euro_h, EPS)
        raw_d = 1.0 / max(euro_d, EPS)
        raw_a = 1.0 / max(euro_a, EPS)
        total = raw_h + raw_d + raw_a
        if total <= EPS:
            return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        return (raw_h / total, raw_d / total, raw_a / total)

    # ── Asian p_upper ──────────────────────────────────────────────

    @staticmethod
    def compute_asian_p_upper(
        upper_water: float, lower_water: float
    ) -> float:
        """
        Compute no-vig probability that the upper side covers the handicap.

        p_upper = (1/upper_water) / (1/upper_water + 1/lower_water)
        """
        inv_u = 1.0 / max(upper_water, EPS)
        inv_l = 1.0 / max(lower_water, EPS)
        total = inv_u + inv_l
        if total <= EPS:
            return 0.5
        return inv_u / total

    # ── line = 0.0 (平手盘) ─────────────────────────────────────────

    @staticmethod
    def combine_line_zero(
        q_h: float, q_d: float, q_a: float, p_upper: float
    ) -> torch.Tensor:
        """
        Level ball (line=0.0):
          - Draw probability comes from the euro prior: p_d = q_d
          - Asian water tells us the home/away split among non-draws:
            r_ah = p_upper = p_h / (p_h + p_a)
          - Therefore: p_h = (1 - p_d) * r_ah,  p_a = (1 - p_d) * (1 - r_ah)

        Intuition:
          Euro 1X2 encodes the draw probability.  The Asian level-ball
          market asks "who is better, home or away?" and ignores draws
          (which push).  So we take the draw from euro and the relative
          strength from asian.
        """
        r_ah = min(max(p_upper, EPS), 1.0 - EPS)
        p_d = q_d
        non_draw = 1.0 - p_d
        p_h = non_draw * r_ah
        p_a = non_draw * (1.0 - r_ah)
        return Phase1Baseline._normalize(torch.tensor([p_h, p_d, p_a]))

    # ── line = ±0.5 (半球盘) ────────────────────────────────────────

    @staticmethod
    def combine_half_ball(
        q_h: float,
        q_d: float,
        q_a: float,
        p_upper: float,
        home_gives: bool,
    ) -> torch.Tensor:
        """
        Half-ball handicap (±0.5):

          line = -0.5 (home gives half ball):
            Upper = home team.  Covers iff home wins.
            Constraint: p_h = p_upper
            Scale p_d, p_a from euro prior proportionally to fill 1-p_h.

          line = +0.5 (home receives half ball = away gives):
            Upper = away team.  Covers iff away wins.
            Constraint: p_a = p_upper
            Scale p_h, p_d from euro prior proportionally to fill 1-p_a.
        """
        p_cover = min(max(p_upper, EPS), 1.0 - EPS)

        if home_gives:
            # line = -0.5: p_h = p_upper
            p_h = p_cover
            rem = 1.0 - p_h
            denom = q_d + q_a
            if denom < EPS:
                p_d = rem / 2.0
                p_a = rem / 2.0
            else:
                p_d = (q_d / denom) * rem
                p_a = (q_a / denom) * rem
        else:
            # line = +0.5: p_a = p_upper
            p_a = p_cover
            rem = 1.0 - p_a
            denom = q_h + q_d
            if denom < EPS:
                p_h = rem / 2.0
                p_d = rem / 2.0
            else:
                p_h = (q_h / denom) * rem
                p_d = (q_d / denom) * rem

        return Phase1Baseline._normalize(torch.tensor([p_h, p_d, p_a]))

    # ── Poisson solver for quarter-ball and -1.0/+1.0 ───────────────

    @staticmethod
    def _solve_poisson_from_constraints(
        q_h: float,
        q_d: float,
        q_a: float,
        p_upper: float,
        line: float,
        max_goals: int = 8,
        lr: float = 0.05,
        steps: int = 200,
        alpha: float = 10.0,
    ) -> torch.Tensor:
        """
        Find λ_h, λ_a such that the Poisson-derived 1X2 distribution:
          - Approximates the euro prior [q_h, q_d, q_a] (KL penalty)
          - Satisfies the Asian handicap constraint (cover = p_upper)

        Uses gradient descent on λ_h, λ_a (no NN, pure numerical opt).

        Args:
            q_h, q_d, q_a: euro prior floats.
            p_upper: Asian no-vig upper probability.
            line: handicap spread (e.g. -0.25, -1.0, +0.75).
            max_goals: truncate Poisson at this many goals.
            lr: learning rate for λ updates.
            steps: number of optimisation iterations.
            alpha: weight on the Asian-constraint term.

        Returns:
            torch.Tensor [3] = [p_h, p_d, p_a].
        """
        q_tensor = torch.tensor([q_h, q_d, q_a])

        # ── Initialise λ from euro prior (rough heuristic) ──
        # Total goals ≈ 2.7 in football.  Assign proportionally.
        init_total = 2.7
        w_h = q_h + 0.5 * q_d
        w_a = q_a + 0.5 * q_d
        w_sum = w_h + w_a
        if w_sum > EPS:
            lam_h = torch.tensor(init_total * w_h / w_sum, requires_grad=True)
            lam_a = torch.tensor(init_total * w_a / w_sum, requires_grad=True)
        else:
            lam_h = torch.tensor(1.35, requires_grad=True)
            lam_a = torch.tensor(1.35, requires_grad=True)

        # ── Precompute Poisson log-prob tables ──
        # We'll recompute each iteration (cheap for 2×8).
        goals_range = torch.arange(0, max_goals + 1, dtype=torch.float32)

        # ── Define cover-prob function for each handicap type ──
        def _cover_prob(p_hw, p_draw, p_aw, l):
            """Asian cover probability given 1X2 probs and handicap line."""
            if l == -0.25:
                # Home -0.25: home win = win, draw = half loss
                return p_hw + 0.5 * p_draw
            elif l == 0.25:
                # Home +0.25: away win = win, draw = half loss
                return p_aw + 0.5 * p_draw
            elif l == -0.75:
                # Home -0.75: home win by 2+ = win, home win by 1 = half win
                return _cover_from_joint(-0.75)
            elif l == 0.75:
                return _cover_from_joint(0.75)
            elif l == -1.0:
                # Home -1.0: home win by 2+ = win, home win by 1 = push
                return _cover_from_joint(-1.0)
            elif l == 1.0:
                return _cover_from_joint(1.0)
            return p_hw  # fallback

        def _cover_from_joint(l):
            """Compute cover prob directly from joint Poisson grid."""
            # This is called inside the optimisation loop so has access
            # to the current joint_probs.  We'll handle it differently.
            pass

        # We need to compute cover from joint probs for -0.75, +0.75, -1.0, +1.0.
        # Strategy: compute the full joint prob grid and aggregate.

        best_loss = float("inf")
        best_probs = torch.tensor([q_h, q_d, q_a])

        for _step in range(steps):
            # ── Poisson probabilities ──
            # log P(k; λ) = k*log(λ) - λ - log(k!)
            # We use the PMF via torch.poisson — actually, torch doesn't have
            # a differentiable Poisson PMF.  We'll use the explicit formula.
            log_lam_h = torch.log(lam_h.clamp(min=EPS))
            log_lam_a = torch.log(lam_a.clamp(min=EPS))

            # log(k!) via lgamma
            log_fact = torch.lgamma(goals_range + 1.0)

            log_p_h = goals_range * log_lam_h - lam_h - log_fact  # [G+1]
            log_p_a = goals_range * log_lam_a - lam_a - log_fact  # [G+1]

            p_h = torch.exp(log_p_h)  # [G+1]
            p_a = torch.exp(log_p_a)  # [G+1]

            # Joint probability grid [G+1, G+1]
            joint = p_h.unsqueeze(1) * p_a.unsqueeze(0)  # outer product

            # ── Derived 1X2 ──
            # Home win: i > j
            home_win_mask = (
                goals_range.unsqueeze(1) > goals_range.unsqueeze(0)
            ).float()  # [G+1, G+1]
            p_hw = (joint * home_win_mask).sum()

            # Draw: i == j
            draw_mask = (
                goals_range.unsqueeze(1) == goals_range.unsqueeze(0)
            ).float()
            p_draw = (joint * draw_mask).sum()

            # Away win: i < j
            away_win_mask = (
                goals_range.unsqueeze(1) < goals_range.unsqueeze(0)
            ).float()
            p_aw = (joint * away_win_mask).sum()

            # ── Derived cover probability ──
            line_f = float(line)
            if line_f == -0.25:
                cover = p_hw + 0.5 * p_draw
            elif line_f == 0.25:
                cover = p_aw + 0.5 * p_draw
            elif line_f == -0.75:
                # Home win by 2+ = win , win by 1 = half
                win2 = ((goals_range.unsqueeze(1) - goals_range.unsqueeze(0)) >= 2).float()
                win1 = ((goals_range.unsqueeze(1) - goals_range.unsqueeze(0)) == 1).float()
                cover = (joint * win2).sum() + 0.5 * (joint * win1).sum()
            elif line_f == 0.75:
                lose2 = ((goals_range.unsqueeze(0) - goals_range.unsqueeze(1)) >= 2).float()
                lose1 = ((goals_range.unsqueeze(0) - goals_range.unsqueeze(1)) == 1).float()
                cover = (joint * lose2).sum() + 0.5 * (joint * lose1).sum()
            elif line_f == -1.0:
                win2 = ((goals_range.unsqueeze(1) - goals_range.unsqueeze(0)) >= 2).float()
                cover = (joint * win2).sum()
            elif line_f == 1.0:
                lose2 = ((goals_range.unsqueeze(0) - goals_range.unsqueeze(1)) >= 2).float()
                cover = (joint * lose2).sum()
            else:
                cover = p_hw  # fallback

            derived = torch.stack([p_hw, p_draw, p_aw])
            derived = derived / derived.sum().clamp(min=EPS)

            # ── Loss ──
            # KL(derived || q)  (not symmetric)
            kl = (
                derived
                * (torch.log(derived.clamp(min=EPS))
                   - torch.log(q_tensor.clamp(min=EPS)))
            ).sum()
            constraint = (cover - p_upper) ** 2
            loss = kl + alpha * constraint

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_probs = derived.detach().clone()

            # ── Gradient step ──
            grad_lam_h, grad_lam_a = torch.autograd.grad(
                loss, [lam_h, lam_a], retain_graph=False
            )

            with torch.no_grad():
                new_h = (lam_h - lr * grad_lam_h).clamp(0.1, 5.0)
                new_a = (lam_a - lr * grad_lam_a).clamp(0.1, 5.0)
            lam_h = new_h.detach().clone().requires_grad_(True)
            lam_a = new_a.detach().clone().requires_grad_(True)

        return Phase1Baseline._normalize(best_probs)

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _normalize(probs: torch.Tensor) -> torch.Tensor:
        """Clamp to [0,1] and re-normalize to sum 1."""
        probs = probs.clamp(0.0, 1.0)
        total = probs.sum()
        if total > EPS:
            return probs / total
        return torch.full_like(probs, 1.0 / 3.0)

    @staticmethod
    def _has_euro(h: Optional[float], d: Optional[float], a: Optional[float]) -> bool:
        """Check that all three euro odds are positive finite numbers."""
        try:
            vals = [float(h), float(d), float(a)]
            return all(
                v is not None
                and math.isfinite(v)
                and v > 1.0
                for v in vals
            )
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _has_asian(
        line: Optional[float],
        upper: Optional[float],
        lower: Optional[float],
    ) -> bool:
        """Check that asian line is finite (can be 0.0 for level ball)
        and both water prices are positive finite numbers."""
        try:
            # asian_line can be 0.0 (level ball) — only check that it's finite
            raw_line = float(line)
            raw_upper = float(upper)
            raw_lower = float(lower)
            return (
                raw_line is not None
                and math.isfinite(raw_line)
                and raw_upper is not None
                and math.isfinite(raw_upper)
                and raw_upper > 0.0
                and raw_lower is not None
                and math.isfinite(raw_lower)
                and raw_lower > 0.0
            )
        except (TypeError, ValueError):
            return False


# ── Track 1: KL divergence ─────────────────────────────────────────

def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    """
    KL(p || q) = sum_i p_i * log(p_i / q_i)

    Args:
        p: [..., C] first distribution (e.g. model output).
        q: [..., C] second distribution (e.g. market prior).
        eps: clamp threshold to avoid log(0).

    Returns:
        Scalar KL divergence (float).
    """
    p = p.clamp(min=eps, max=1.0 - eps)
    q = q.clamp(min=eps, max=1.0 - eps)
    return float((p * (p.log() - q.log())).sum(dim=-1).mean().item())


# ── Convenience: compute baseline for a batch of flat rows ──────────

def compute_baseline_probs(
    rows: list,
    baseline: Optional[Phase1Baseline] = None,
) -> torch.Tensor:
    """
    Compute joint baseline probs for a list of flat-format rows.

    Args:
        rows: list of dicts with keys close_euro_h/d/a, close_asian_line,
              close_asian_upper_water, close_asian_lower_water.
        baseline: Phase1Baseline instance (created if None).

    Returns:
        torch.Tensor [N, 3] of baseline probabilities.
    """
    if baseline is None:
        baseline = Phase1Baseline()

    probs_list = []
    for row in rows:
        p = baseline.compute(
            close_euro_h=row.get("close_euro_h"),
            close_euro_d=row.get("close_euro_d"),
            close_euro_a=row.get("close_euro_a"),
            close_asian_line=row.get("close_asian_line"),
            close_asian_upper_water=row.get("close_asian_upper_water"),
            close_asian_lower_water=row.get("close_asian_lower_water"),
        )
        probs_list.append(p)

    return torch.stack(probs_list)
