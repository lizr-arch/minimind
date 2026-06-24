"""
P1.11: Score utility functions — independent Poisson score grid, score-derived 1X2, top-k scorelines.

All computations assume independent Poisson(home_lambda, away_lambda):
  P(home=i, away=j) = Poisson(i|lambda_home) * Poisson(j|lambda_away)

This is NOT a true bivariate Poisson (no shared covariance term).
The under-dispersion in calibration reports is expected — goals are correlated in reality.
Future P1.12 may add Dixon-Coles correction or dispersion scaling.
"""
import math
import torch
from typing import List, Tuple, Dict


def _log_factorial_cache(max_n: int = 20) -> torch.Tensor:
    """Precompute log(n!) for n = 0..max_n."""
    lf = torch.zeros(max_n + 1)
    for i in range(1, max_n + 1):
        lf[i] = lf[i - 1] + math.log(i)
    return lf


# Module-level cache, computed once
_LOG_FACTORIAL = None


def _get_log_factorial(max_n: int = 20) -> torch.Tensor:
    global _LOG_FACTORIAL
    if _LOG_FACTORIAL is None or len(_LOG_FACTORIAL) <= max_n:
        _LOG_FACTORIAL = _log_factorial_cache(max_n)
    return _LOG_FACTORIAL


def independent_poisson_score_grid(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = 10,
) -> Dict:
    """
    Compute score-derived probabilities from INDEPENDENT Poisson model.

    P(home=i, away=j) = Poisson(i|lambda_home) * Poisson(j|lambda_away)
    This assumes zero correlation between home and away goals.
    For a true bivariate model, see future P1.12 (Dixon-Coles / covariance).

    Args:
        lambda_home: expected home goals (λ_home)
        lambda_away: expected away goals (λ_away)
        max_goals: maximum goals to consider per team

    Returns dict with:
        home_win_prob, draw_prob, away_win_prob: 1X2 probabilities (sum to 1)
        score_matrix: [max_goals+1, max_goals+1] tensor of P(home=i, away=j)
        expected_total: E[home+away]
        expected_diff: E[home-away]
    """
    lf = _get_log_factorial(max_goals)
    G = max_goals + 1

    # Home Poisson log-probs: log P(home=i) = -λ + i*log(λ) - log(i!)
    i_vals = torch.arange(G, dtype=torch.float32)
    home_log_p = -lambda_home + i_vals * math.log(max(lambda_home, 1e-10)) - lf[:G]
    away_log_p = -lambda_away + i_vals * math.log(max(lambda_away, 1e-10)) - lf[:G]

    # Joint log-prob matrix: [G, G]
    joint_log_p = home_log_p.unsqueeze(1) + away_log_p.unsqueeze(0)  # [G, G]
    joint_p = torch.exp(joint_log_p)

    # Normalize (handle truncation at max_goals)
    total_mass = joint_p.sum()
    if total_mass > 0:
        joint_p = joint_p / total_mass

    # 1X2 probabilities
    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    for i in range(G):
        for j in range(G):
            p = joint_p[i, j].item()
            if i > j:
                home_win += p
            elif i == j:
                draw += p
            else:
                away_win += p

    expected_total = lambda_home + lambda_away
    expected_diff = lambda_home - lambda_away

    return {
        "home_win_prob": round(home_win, 6),
        "draw_prob": round(draw, 6),
        "away_win_prob": round(away_win, 6),
        "expected_total_goals": round(expected_total, 3),
        "expected_goal_diff": round(expected_diff, 3),
        "score_matrix": joint_p,  # tensor, caller can convert
    }


def top_k_scorelines(
    lambda_home: float,
    lambda_away: float,
    k: int = 5,
    max_goals: int = 10,
) -> List[Dict]:
    """
    Return top-k most likely scorelines from bivariate Poisson.

    Returns list of {"score": "i-j", "home": i, "away": j, "prob": float}
    sorted by probability descending.
    """
    result = independent_poisson_score_grid(lambda_home, lambda_away, max_goals)
    joint_p = result["score_matrix"]
    G = max_goals + 1

    scorelines = []
    for i in range(G):
        for j in range(G):
            scorelines.append({
                "score": f"{i}-{j}",
                "home": i,
                "away": j,
                "prob": round(joint_p[i, j].item(), 6),
            })

    scorelines.sort(key=lambda x: x["prob"], reverse=True)
    return scorelines[:k]


def consistency_check(
    euro_probs: Dict[str, float],
    score_1x2: Dict[str, float],
) -> Dict:
    """
    Compare Euro Head probs with score-derived 1X2 probs.

    Returns:
        euro_home/draw/away: from Euro head
        score_home/draw/away: from bivariate Poisson
        js_distance: Jensen-Shannon divergence (0 = identical)
        agreement: whether both predict same winner
    """
    import numpy as np

    p = np.array([
        euro_probs.get("home", euro_probs.get("home_win_prob", 0)),
        euro_probs.get("draw", euro_probs.get("draw_prob", 0)),
        euro_probs.get("away", euro_probs.get("away_win_prob", 0)),
    ])
    q = np.array([
        score_1x2.get("home_win_prob", 0),
        score_1x2.get("draw_prob", 0),
        score_1x2.get("away_win_prob", 0),
    ])

    # Normalize
    p = p / (p.sum() + 1e-10)
    q = q / (q.sum() + 1e-10)

    # Jensen-Shannon distance
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + 1e-10) / (m + 1e-10)))
    kl_qm = np.sum(q * np.log((q + 1e-10) / (m + 1e-10)))
    js_distance = float(np.sqrt(0.5 * kl_pm + 0.5 * kl_qm))

    euro_winner = np.argmax(p)
    score_winner = np.argmax(q)
    agreement = bool(euro_winner == score_winner)

    return {
        "euro_home": round(float(p[0]), 4),
        "euro_draw": round(float(p[1]), 4),
        "euro_away": round(float(p[2]), 4),
        "score_home": round(float(q[0]), 4),
        "score_draw": round(float(q[1]), 4),
        "score_away": round(float(q[2]), 4),
        "js_distance": round(js_distance, 4),
        "winner_agreement": agreement,
    }


def disagreement_policy(
    euro_probs: Dict[str, float],
    score_derived_probs: Dict[str, float],
    js_divergence: float,
    winner_agreement: bool,
) -> Dict:
    """
    P1.12: Score-Euro disagreement policy — model consistency diagnostic.

    Categorizes the relationship between Euro Head and Score-derived 1X2 probs
    and suggests a confidence adjustment. NOT betting advice.

    Agreement levels:
        high_agreement           — winner same, JS low (<0.10)
        same_winner_high_js      — winner same, JS high (>=0.10)
        winner_disagreement      — winner differs, JS high (>=0.10)
        score_extreme_euro_conservative — score is very confident, Euro is not

    Confidence adjustments:
        keep      — no change
        soften    — reduce confidence
        flag_review — high uncertainty, consider manual review

    This is model calibration and diagnostics only.
    No betting advice. No profit claim.
    """
    score_max = max(
        score_derived_probs.get("home_win_prob", 0),
        score_derived_probs.get("draw_prob", 0),
        score_derived_probs.get("away_win_prob", 0),
    )
    euro_max = max(
        euro_probs.get("home", 0),
        euro_probs.get("draw", 0),
        euro_probs.get("away", 0),
    )

    # Determine agreement level (order matters: more specific checks first)
    if not winner_agreement and js_divergence >= 0.10:
        agreement_level = "winner_disagreement"
        confidence_adjustment = "flag_review"
        reason = "Euro and Score heads predict different winners with high divergence. High uncertainty."
    elif not winner_agreement:
        agreement_level = "winner_disagreement"
        confidence_adjustment = "flag_review"
        reason = "Winners differ (low JS — rare edge case). Flag for review."
    elif score_max > 0.75 and euro_max < 0.55:
        agreement_level = "score_extreme_euro_conservative"
        confidence_adjustment = "soften"
        reason = "Score head is very confident while Euro is conservative. Possible over-confidence in score distribution."
    elif js_divergence >= 0.10:
        agreement_level = "same_winner_high_js"
        confidence_adjustment = "soften"
        reason = "Winner agrees but score distribution differs significantly from Euro. Confidence should be tempered."
    else:
        agreement_level = "high_agreement"
        confidence_adjustment = "keep"
        reason = "Both heads agree on winner with low JS divergence."

    return {
        "agreement_level": agreement_level,
        "confidence_adjustment": confidence_adjustment,
        "reason": reason,
    }
