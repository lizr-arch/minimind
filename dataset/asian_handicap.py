"""
Asian Handicap Settlement — canonical label calculation (P0.3).

All calculations are from the *upper side* perspective.

Terminology:
  - asian_line: handicap applied to the upper side (e.g. -0.5, +0.25)
  - raw_margin = upper_goals - lower_goals
  - adjusted   = raw_margin + asian_line

Quarter lines (-0.25, -0.75, +0.25, +0.75) are split into two adjacent
half lines.  Each half line settles independently as win/push/loss.
The combination of two half-line outcomes yields the 5-class result.

5-Class labels (from upper side perspective):
  0: upper_full_win
  1: upper_half_win
  2: push
  3: upper_half_loss
  4: upper_full_loss

3-Class mapping:
  upper_full_win, upper_half_win  -> upper (0)
  push                             -> push  (1)
  upper_half_loss, upper_full_loss -> lower (2)
"""

import math
from typing import Tuple

EPS = 1e-9


# ── Line splitting ─────────────────────────────────────────────────────

def split_asian_line(line: float) -> Tuple[float, float]:
    """
    Split an Asian handicap line into two half lines.

    For integer and half-integer lines (e.g. 0, -0.5, +1.0):
        returns (line, line) — both halves identical.

    For quarter lines (-0.25, +0.75, etc.):
        returns the two adjacent half lines, ordered from *more favourable
        to upper* to *less favourable to upper*.
        e.g. -0.25 -> [0.0, -0.5]
             +0.75 -> [+1.0, +0.5]

    Only lines that are multiples of 0.25 are valid.
    """
    line = float(line)

    # Check validity: must be a multiple of 0.25
    remainder = abs(line * 4 - round(line * 4))
    if remainder > EPS:
        raise ValueError(
            f"asian_line must be a multiple of 0.25, got {line}"
        )

    quarters = round(line * 4)  # number of quarter units

    if quarters % 2 == 0:
        # Integer or half line → both halves identical
        return (line, line)
    else:
        # Quarter line → split into adjacent half lines
        # The more-favourable half goes first
        lo = (quarters - 1) / 4.0
        hi = (quarters + 1) / 4.0
        # More favourable = higher value for upper side
        return (max(lo, hi), min(lo, hi))


# ── Single half-line settlement ────────────────────────────────────────

def settle_half_line(raw_margin: float, line: float) -> str:
    """
    Settle a single half line from the upper side perspective.

    Args:
        raw_margin: upper_goals - lower_goals
        line:       a single half line (integer or .5, NOT quarter)

    Returns:
        "win"  if adjusted > 0
        "push" if adjusted == 0
        "loss" if adjusted < 0
    """
    adjusted = raw_margin + line
    if adjusted > EPS:
        return "win"
    elif abs(adjusted) <= EPS:
        return "push"
    else:
        return "loss"


# ── 5-class settlement ─────────────────────────────────────────────────

# Ordered lookup table for combining two half-line outcomes.
# Key: (first_half, second_half) -> 5-class label string
_HALF_COMBOS = {
    ("win",  "win"):  "upper_full_win",
    ("win",  "push"): "upper_half_win",
    ("push", "win"):  "upper_half_win",
    ("push", "push"): "push",
    ("loss", "push"): "upper_half_loss",
    ("push", "loss"): "upper_half_loss",
    ("loss", "loss"): "upper_full_loss",
}


def settle_asian_5class(
    upper_goals: int,
    lower_goals: int,
    asian_line: float,
) -> str:
    """
    Compute the 5-class Asian handicap result from the upper side perspective.

    Args:
        upper_goals: goals scored by the upper side team.
        lower_goals: goals scored by the lower side team.
        asian_line:  handicap line applied to the upper side.

    Returns:
        One of: "upper_full_win", "upper_half_win", "push",
                "upper_half_loss", "upper_full_loss".

    Raises:
        ValueError: if the line is invalid or the half-line combination
                    is theoretically impossible (e.g. win+loss).
    """
    raw_margin = upper_goals - lower_goals
    half1, half2 = split_asian_line(asian_line)

    r1 = settle_half_line(raw_margin, half1)
    r2 = settle_half_line(raw_margin, half2)

    combo = (r1, r2)
    if combo not in _HALF_COMBOS:
        raise ValueError(
            f"Impossible half-line combination: {combo} "
            f"(margin={raw_margin}, line={asian_line}, halves=({half1}, {half2}))"
        )
    return _HALF_COMBOS[combo]


# ── 3-class settlement (coarse mapping) ────────────────────────────────

_5TO3_MAP = {
    "upper_full_win":  "upper",
    "upper_half_win":  "upper",
    "push":            "push",
    "upper_half_loss": "lower",
    "upper_full_loss": "lower",
}


def settle_asian_3class(
    upper_goals: int,
    lower_goals: int,
    asian_line: float,
) -> str:
    """
    Compute the 3-class Asian handicap result, mapping from the 5-class.
    """
    result_5 = settle_asian_5class(upper_goals, lower_goals, asian_line)
    return _5TO3_MAP[result_5]


# ── Upper/lower goal extraction ────────────────────────────────────────

def get_upper_lower_goals(
    home_goals: int,
    away_goals: int,
    upper_side: str,
) -> Tuple[int, int]:
    """
    Extract (upper_goals, lower_goals) from home/away goals and upper_side.

    Args:
        home_goals: goals scored by home team.
        away_goals: goals scored by away team.
        upper_side: "home" or "away".

    Returns:
        (upper_goals, lower_goals)
    """
    upper_side = upper_side.lower()
    if upper_side == "home":
        return (home_goals, away_goals)
    elif upper_side == "away":
        return (away_goals, home_goals)
    else:
        raise ValueError(f"upper_side must be 'home' or 'away', got '{upper_side}'")
