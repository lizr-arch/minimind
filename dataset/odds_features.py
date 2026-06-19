"""
OddsMind Tabular Feature Extractor (P0.5B)

Extracts fixed-length feature vectors from variable-length odds timelines.
Designed for lightweight trainable baselines (logistic regression, tiny MLP).

Feature groups (per sample):
  - open event:      7 raw features from the earliest event
  - last event:      7 raw features from the event closest to cutoff
  - implied prob:    3 features (1/euro_h, 1/euro_d, 1/euro_a) normalized
  - open→last delta: 7 delta features
  - timeline stats:  min/max of euro_h, euro_d, euro_a, asian_line across timeline (8)

Total: 7 + 7 + 3 + 7 + 8 = 32 features
"""

from typing import List

import torch

# ── Feature names (for documentation / debugging) ──────────────────────

FEATURE_NAMES = (
    # Open event (7)
    ["open_minutes_before_kickoff", "open_euro_h", "open_euro_d", "open_euro_a",
     "open_asian_line", "open_upper_water", "open_lower_water"]
    # Last event (7)
    + ["last_minutes_before_kickoff", "last_euro_h", "last_euro_d", "last_euro_a",
       "last_asian_line", "last_upper_water", "last_lower_water"]
    # Implied probabilities (3)
    + ["implied_home", "implied_draw", "implied_away"]
    # Open → last deltas (7)
    + ["delta_minutes", "delta_euro_h", "delta_euro_d", "delta_euro_a",
       "delta_asian_line", "delta_upper_water", "delta_lower_water"]
    # Timeline stats: mins of euro_h, euro_d, euro_a, asian_line (8)
    + ["min_euro_h", "max_euro_h", "min_euro_d", "max_euro_d",
       "min_euro_a", "max_euro_a", "min_asian_line", "max_asian_line"]
)


def feature_dim() -> int:
    return len(FEATURE_NAMES)


# ── Extraction ─────────────────────────────────────────────────────────

def extract_features(events: List[dict]) -> torch.Tensor:
    """
    Extract a fixed-size feature vector from a filtered odds timeline.

    Args:
        events: list of event dicts (already filtered by cutoff),
                sorted by minutes_before_kickoff DESC (earliest first).

    Returns:
        Float tensor of shape [feature_dim].
    """
    if not events:
        raise ValueError("Cannot extract features from empty timeline")

    n = len(events)

    # Open = earliest event (largest minutes_before_kickoff, first in sorted list)
    open_ev = events[0]
    # Last = event closest to kickoff (smallest minutes_before_kickoff, last in list)
    last_ev = events[-1]

    # ── Raw features ──
    open_feats = [
        open_ev["minutes_before_kickoff"],
        open_ev["euro_h"], open_ev["euro_d"], open_ev["euro_a"],
        open_ev["asian_line"], open_ev["upper_water"], open_ev["lower_water"],
    ]
    last_feats = [
        last_ev["minutes_before_kickoff"],
        last_ev["euro_h"], last_ev["euro_d"], last_ev["euro_a"],
        last_ev["asian_line"], last_ev["upper_water"], last_ev["lower_water"],
    ]

    # ── Implied probabilities from last event ──
    raw_h = 1.0 / max(float(last_ev["euro_h"]), 1e-9)
    raw_d = 1.0 / max(float(last_ev["euro_d"]), 1e-9)
    raw_a = 1.0 / max(float(last_ev["euro_a"]), 1e-9)
    total = raw_h + raw_d + raw_a
    implied = [raw_h / total, raw_d / total, raw_a / total] if total > 0 else [1/3, 1/3, 1/3]

    # ── Open → last deltas ──
    deltas = [open_feats[i] - last_feats[i] for i in range(7)]

    # ── Timeline stats ──
    all_euro_h = [e["euro_h"] for e in events]
    all_euro_d = [e["euro_d"] for e in events]
    all_euro_a = [e["euro_a"] for e in events]
    all_asian_line = [e["asian_line"] for e in events]

    stats = [
        min(all_euro_h), max(all_euro_h),
        min(all_euro_d), max(all_euro_d),
        min(all_euro_a), max(all_euro_a),
        min(all_asian_line), max(all_asian_line),
    ]

    # ── Combine ──
    all_feats = open_feats + last_feats + implied + deltas + stats
    return torch.tensor(all_feats, dtype=torch.float32)


def extract_features_batch(
    timelines: List[List[dict]],
) -> torch.Tensor:
    """
    Extract features for a batch of timelines.

    Args:
        timelines: list of lists of event dicts.

    Returns:
        Float tensor [B, feature_dim].
    """
    return torch.stack([extract_features(tl) for tl in timelines])
