"""
OddsMind Simple Deterministic Baselines (P0.5A)

Pure Python + PyTorch, zero external dependencies (no numpy/sklearn).

Baselines:
  - lowest_odds_euro:     pick outcome with lowest decimal odds
  - implied_prob_euro:    1/odds normalized to probabilities
  - low_water_asian_3class: pick side with lower water level
  - low_water_asian_5class: coarse projection of low_water into 5-class
  - uniform:              equal probability for all classes

All baselines operate on the *last visible event* in the filtered timeline
(closest to the cutoff time).  They do NOT use future information.
"""

from typing import Dict

import torch

EPS = 1e-9


# ── Helpers ────────────────────────────────────────────────────────────

def _get_last_event(events: list) -> dict:
    """Return the event closest to kickoff (smallest minutes_before_kickoff)."""
    if not events:
        raise ValueError("Empty timeline — cannot compute baseline")
    return min(events, key=lambda e: e["minutes_before_kickoff"])


def _check_odds(odds: list, label: str):
    """Validate that odds are positive finite numbers."""
    for v in odds:
        if not isinstance(v, (int, float)) or v <= 0 or not torch.isfinite(torch.tensor(v)):
            raise ValueError(f"Invalid {label} odds: {v}")


# ── Euro Baselines ─────────────────────────────────────────────────────

def lowest_odds_euro(events: list) -> Dict[str, float]:
    """
    Predict the outcome with the lowest decimal odds.

    Returns a one-hot probability dict: {"home": 1.0, "draw": 0.0, "away": 0.0}
    """
    e = _get_last_event(events)
    h, d, a = float(e["euro_h"]), float(e["euro_d"]), float(e["euro_a"])
    _check_odds([h, d, a], "euro")

    min_odds = min(h, d, a)
    return {
        "home": 1.0 if h == min_odds else 0.0,
        "draw": 1.0 if d == min_odds else 0.0,
        "away": 1.0 if a == min_odds else 0.0,
    }


def implied_prob_euro(events: list) -> Dict[str, float]:
    """
    Convert decimal odds to implied probabilities via 1/odds normalisation.

    Returns {"home": p_h, "draw": p_d, "away": p_a} where sum ≈ 1.
    """
    e = _get_last_event(events)
    h, d, a = float(e["euro_h"]), float(e["euro_d"]), float(e["euro_a"])
    _check_odds([h, d, a], "euro")

    raw_h = 1.0 / h
    raw_d = 1.0 / d
    raw_a = 1.0 / a
    total = raw_h + raw_d + raw_a
    if total <= EPS:
        raise ValueError(f"Implied prob total too small: {total}")
    return {
        "home": raw_h / total,
        "draw": raw_d / total,
        "away": raw_a / total,
    }


# ── Asian Baselines ────────────────────────────────────────────────────

def low_water_asian_3class(events: list, eps: float = 0.005) -> Dict[str, float]:
    """
    Predict the side with the lower water level (more favourable odds).

    If upper_water < lower_water → upper.
    If lower_water < upper_water → lower.
    If |diff| <= eps → push.

    Returns {"upper": p, "push": p, "lower": p} one-hot.
    """
    e = _get_last_event(events)
    uw = float(e["upper_water"])
    lw = float(e["lower_water"])
    diff = uw - lw

    if abs(diff) <= eps:
        return {"upper": 0.0, "push": 1.0, "lower": 0.0}
    elif diff < 0:
        return {"upper": 1.0, "push": 0.0, "lower": 0.0}
    else:
        return {"upper": 0.0, "push": 0.0, "lower": 1.0}


def low_water_asian_5class(events: list, eps: float = 0.005) -> Dict[str, float]:
    """
    Coarse 5-class projection of the low-water baseline.

    Since we don't have score information, we can only predict:
      - upper_full_win when upper_water is lower
      - push when waters are equal
      - upper_full_loss when lower_water is lower

    Half-win/loss classes (1, 3) are never predicted by this baseline.
    This is an acknowledged limitation documented in ODDSMIND_P0_5A_BASELINES.md.
    """
    e = _get_last_event(events)
    uw = float(e["upper_water"])
    lw = float(e["lower_water"])
    diff = uw - lw

    if abs(diff) <= eps:
        return {"upper_full_win": 0.0, "upper_half_win": 0.0,
                "push": 1.0, "upper_half_loss": 0.0, "upper_full_loss": 0.0}
    elif diff < 0:
        return {"upper_full_win": 1.0, "upper_half_win": 0.0,
                "push": 0.0, "upper_half_loss": 0.0, "upper_full_loss": 0.0}
    else:
        return {"upper_full_win": 0.0, "upper_half_win": 0.0,
                "push": 0.0, "upper_half_loss": 0.0, "upper_full_loss": 1.0}


# ── Uniform Baseline ───────────────────────────────────────────────────

def uniform_baseline(num_classes: int) -> Dict[str, float]:
    """
    Return equal probability for all classes.

    3-class keys: upper/push/lower
    5-class keys: upper_full_win/upper_half_win/push/upper_half_loss/upper_full_loss
    """
    if num_classes == 3:
        return {"upper": 1/3, "push": 1/3, "lower": 1/3}
    elif num_classes == 5:
        return {"upper_full_win": 0.2, "upper_half_win": 0.2,
                "push": 0.2, "upper_half_loss": 0.2, "upper_full_loss": 0.2}
    else:
        raise ValueError(f"Unsupported num_classes: {num_classes}")


# ── Prob dict → tensor ─────────────────────────────────────────────────

# Key order must match the label maps in dataset/odds_dataset.py
EURO_KEY_ORDER = ["home", "draw", "away"]
ASIAN_3_KEY_ORDER = ["upper", "push", "lower"]
ASIAN_5_KEY_ORDER = ["upper_full_win", "upper_half_win", "push",
                     "upper_half_loss", "upper_full_loss"]


def euro_probs_to_tensor(probs: Dict[str, float]) -> torch.Tensor:
    """Convert euro prob dict to [3] tensor."""
    return torch.tensor([probs[k] for k in EURO_KEY_ORDER], dtype=torch.float32)


def asian_probs_to_tensor(probs: Dict[str, float], num_classes: int) -> torch.Tensor:
    """Convert asian prob dict to [num_classes] tensor."""
    keys = ASIAN_3_KEY_ORDER if num_classes == 3 else ASIAN_5_KEY_ORDER
    return torch.tensor([probs[k] for k in keys], dtype=torch.float32)
