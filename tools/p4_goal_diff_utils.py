"""P4 residual goal-diff objective helpers.

This module is intentionally small and framework-light: it owns label bucketing,
1X2 folding, Euro anchor extraction, and metric/loss helpers used by the P4.1
objective probe.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from eval.odds_metrics import (
    accuracy_from_probs,
    brier_from_probs,
    class_counts,
    ece_from_probs,
    logloss_from_probs,
    prediction_counts_from_probs,
)


EPS = 1e-8


def goal_diff_to_bucket(goal_diff: int) -> int:
    """Map raw home-away goal difference to the 7 P4 buckets."""
    if goal_diff <= -3:
        return 0
    if goal_diff == -2:
        return 1
    if goal_diff == -1:
        return 2
    if goal_diff == 0:
        return 3
    if goal_diff == 1:
        return 4
    if goal_diff == 2:
        return 5
    return 6


def fold_goal_diff_probs(q_goal_diff: torch.Tensor) -> torch.Tensor:
    """Fold goal-diff bucket probabilities into [home, draw, away] 1X2 order."""
    if q_goal_diff.shape[-1] != 7:
        raise ValueError("goal-diff probabilities must have 7 buckets")
    p_away = q_goal_diff[..., 0] + q_goal_diff[..., 1] + q_goal_diff[..., 2]
    p_draw = q_goal_diff[..., 3]
    p_home = q_goal_diff[..., 4] + q_goal_diff[..., 5] + q_goal_diff[..., 6]
    return torch.stack([p_home, p_draw, p_away], dim=-1)


def extract_euro_anchor_probs(row: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    """Extract the nearest pre-match valid Euro no-vig probability.

    Negative-time valid odds are counted for audit purposes but never used.
    If no valid non-negative-time Euro event exists, a uniform anchor is used.
    """
    timeline = row.get("raw_timeline", []) or row.get("odds_timeline", []) or []
    valid_non_negative: list[tuple[float, torch.Tensor]] = []
    negative_time_valid_count = 0

    for event in timeline:
        odds = _valid_euro_odds(event)
        if odds is None:
            continue
        minutes = _as_float(event.get("minutes_before_kickoff", 0.0))
        inv = torch.tensor([1.0 / odds[0], 1.0 / odds[1], 1.0 / odds[2]], dtype=torch.float32)
        probs = inv / inv.sum().clamp_min(EPS)
        if minutes < 0:
            negative_time_valid_count += 1
            continue
        valid_non_negative.append((minutes, probs))

    if not valid_non_negative:
        return torch.full((3,), 1.0 / 3.0, dtype=torch.float32), {
            "used_fallback": True,
            "anchor_minutes_before_kickoff": None,
            "negative_time_valid_euro_count": negative_time_valid_count,
        }

    minutes, probs = min(valid_non_negative, key=lambda item: item[0])
    return probs, {
        "used_fallback": False,
        "anchor_minutes_before_kickoff": minutes,
        "negative_time_valid_euro_count": negative_time_valid_count,
    }


def compute_p4_losses(
    delta_logits: torch.Tensor,
    goal_diff_logits: torch.Tensor,
    p_euro_anchor: torch.Tensor,
    y_1x2: torch.Tensor,
    y_goal_diff: torch.Tensor,
    diff_loss_weight: float = 0.30,
    consistency_loss_weight: float = 0.10,
    delta_l2_weight: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Compute P4 residual 1X2 + goal-diff auxiliary objective."""
    anchor_logits = torch.log(p_euro_anchor.clamp_min(EPS))
    final_logits = anchor_logits + delta_logits
    p_final = F.softmax(final_logits, dim=-1)

    q_diff = F.softmax(goal_diff_logits, dim=-1)
    p_from_diff = fold_goal_diff_probs(q_diff).clamp_min(EPS)
    p_from_diff = p_from_diff / p_from_diff.sum(dim=-1, keepdim=True).clamp_min(EPS)

    l_1x2 = F.cross_entropy(final_logits, y_1x2)
    l_diff = F.cross_entropy(goal_diff_logits, y_goal_diff)
    l_consistency = F.kl_div(
        F.log_softmax(final_logits, dim=-1),
        p_from_diff.detach(),
        reduction="batchmean",
    )
    l_delta = delta_logits.pow(2).mean()
    loss = (
        l_1x2
        + float(diff_loss_weight) * l_diff
        + float(consistency_loss_weight) * l_consistency
        + float(delta_l2_weight) * l_delta
    )
    return {
        "loss": loss,
        "L_1x2": l_1x2,
        "L_diff": l_diff,
        "L_consistency": l_consistency,
        "L_delta": l_delta,
        "p_final": p_final,
        "p_from_diff": p_from_diff,
        "q_diff": q_diff,
        "final_logits": final_logits,
    }


def compute_1x2_metrics(probs: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    """Compute standard 1X2 metrics plus draw diagnostics."""
    probs = probs.detach().cpu().float().clamp(EPS, 1.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(EPS)
    labels = labels.detach().cpu().long()
    preds = probs.argmax(dim=-1)
    true_draw = labels == 1
    pred_draw = preds == 1
    if int(true_draw.sum().item()) == 0:
        draw_recall = 0.0
        mean_p_draw_on_true_draw = 0.0
    else:
        draw_recall = float((pred_draw & true_draw).sum().item() / true_draw.sum().item())
        mean_p_draw_on_true_draw = float(probs[true_draw, 1].mean().item())
    return {
        "accuracy": accuracy_from_probs(probs, labels),
        "logloss": logloss_from_probs(probs, labels),
        "brier": brier_from_probs(probs, labels, 3),
        "ece": ece_from_probs(probs, labels, 3)["ece"],
        "label_counts": class_counts(labels, 3),
        "prediction_counts": prediction_counts_from_probs(probs, 3),
        "argmax_draw_count": int(pred_draw.sum().item()),
        "draw_recall": draw_recall,
        "mean_p_draw": float(probs[:, 1].mean().item()),
        "mean_p_draw_on_true_draw": mean_p_draw_on_true_draw,
    }


def compute_anchor_only_metrics(p_euro_anchor: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    """Evaluate the raw Euro anchor directly on validation labels."""
    metrics = compute_1x2_metrics(p_euro_anchor, labels)
    return {
        "anchor_only_val_logloss": metrics["logloss"],
        "anchor_only_brier": metrics["brier"],
        "anchor_only_ECE": metrics["ece"],
        "anchor_only_acc": metrics["accuracy"],
        "argmax_draw_count": metrics["argmax_draw_count"],
        "draw_recall": metrics["draw_recall"],
        "mean_p_draw": metrics["mean_p_draw"],
        "mean_p_draw_on_true_draw": metrics["mean_p_draw_on_true_draw"],
    }


def _valid_euro_odds(event: dict[str, Any]) -> tuple[float, float, float] | None:
    h = _as_float(event.get("euro_h", 0.0))
    d = _as_float(event.get("euro_d", 0.0))
    a = _as_float(event.get("euro_a", 0.0))
    if h > 1.0 and d > 1.0 and a > 1.0 and all(math.isfinite(v) for v in (h, d, a)):
        return h, d, a
    return None


def _as_float(value: Any) -> float:
    try:
        return float(value if value is not None else 0.0)
    except (TypeError, ValueError):
        return 0.0
