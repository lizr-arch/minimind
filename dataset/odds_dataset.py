"""
OddsMind Dataset — loads JSONL odds timeline data.

Each JSONL line contains a match with an odds timeline and label.
The dataset returns raw features and labels; padding/batching is handled
by OddsCollator.

Label mapping:
    euro_result:  home=0, draw=1, away=2
    asian_result: upper=0, push=1, lower=2
"""

import json
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


EURO_MAP = {"home": 0, "draw": 1, "away": 2}
ASIAN_MAP = {"upper": 0, "push": 1, "lower": 2}

# Fixed feature order (must match OddsEventEncoder.feature_dim)
FEATURE_KEYS = [
    "minutes_before_kickoff",
    "euro_h",
    "euro_d",
    "euro_a",
    "asian_line",
    "upper_water",
    "lower_water",
]


def _event_to_features(event: dict, cutoff_minutes: Optional[float] = None) -> List[float]:
    """
    Extract a fixed-order feature vector from an odds event dict.

    If cutoff_minutes is provided, only events with
    minutes_before_kickoff >= cutoff_minutes are included (the caller
    should filter before calling this function).
    """
    return [float(event[k]) for k in FEATURE_KEYS]


class OddsDataset(Dataset):
    """
    Dataset for odds time-series matches.

    Args:
        jsonl_path: Path to a JSONL file of match fixtures.
        max_seq_len: Maximum number of time-steps to keep (truncate oldest).
        cutoff_minutes: If set, drop events with minutes_before_kickoff < cutoff.
                        If None (default), use the match's own cutoff_minutes field.
    """

    def __init__(
        self,
        jsonl_path: str,
        max_seq_len: int = 64,
        cutoff_minutes: Optional[float] = None,
    ):
        self.max_seq_len = max_seq_len
        self.cutoff_minutes = cutoff_minutes
        self.samples: List[dict] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]

        # Determine effective cutoff
        cutoff = (
            self.cutoff_minutes
            if self.cutoff_minutes is not None
            else sample.get("cutoff_minutes", 0)
        )

        # Filter events by cutoff
        timeline = sample["odds_timeline"]
        filtered = [
            e for e in timeline
            if e["minutes_before_kickoff"] >= cutoff
        ]

        # Truncate to max_seq_len (keep most recent events)
        if len(filtered) > self.max_seq_len:
            filtered = filtered[-self.max_seq_len:]

        # Build feature matrix [seq_len, feature_dim]
        features = torch.tensor(
            [_event_to_features(e) for e in filtered],
            dtype=torch.float32,
        )

        # Labels
        euro_label = EURO_MAP[sample["label"]["euro_result"]]
        asian_label = ASIAN_MAP[sample["label"]["asian_result"]]

        return {
            "features": features,          # [seq_len, feature_dim]
            "euro_label": euro_label,       # int
            "asian_label": asian_label,     # int
            "match_id": sample["match_id"],
            "seq_len": len(filtered),       # for debugging
        }
