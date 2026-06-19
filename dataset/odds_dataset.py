"""
OddsMind Dataset — loads JSONL odds timeline data.

P0.2 adds cutoff modes:
  - none (default):       one sample per JSONL line (P0.1 behaviour)
  - exhaustive:           expand each match into one sample per cutoff
  - random:               at each __getitem__ randomly pick a cutoff

Label mapping:
    euro_result:  home=0, draw=1, away=2
    asian_result: upper=0, push=1, lower=2
"""

import json
import random
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset

from dataset.odds_cutoff import filter_timeline_by_cutoff, build_exhaustive_cutoff_dataset

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


def _event_to_features(event: dict) -> List[float]:
    """Extract a fixed-order feature vector from an odds event dict."""
    return [float(event[k]) for k in FEATURE_KEYS]


def _build_features_and_labels(sample: dict, max_seq_len: int) -> dict:
    """
    Core item builder: given a sample dict (which may be an original match
    or a pre-built cutoff sample), extract features, labels, and metadata.
    """
    timeline = sample["odds_timeline"]

    # Sort descending by minutes_before_kickoff (earliest first)
    sorted_timeline = sorted(
        timeline,
        key=lambda e: e["minutes_before_kickoff"],
        reverse=True,
    )

    # Truncate to max_seq_len (keep most recent = end of list)
    if len(sorted_timeline) > max_seq_len:
        sorted_timeline = sorted_timeline[-max_seq_len:]

    features = torch.tensor(
        [_event_to_features(e) for e in sorted_timeline],
        dtype=torch.float32,
    )

    euro_label = EURO_MAP[sample["label"]["euro_result"]]
    asian_label = ASIAN_MAP[sample["label"]["asian_result"]]

    match_id = sample.get("sample_id", sample.get("match_id", ""))

    return {
        "features": features,
        "euro_label": euro_label,
        "asian_label": asian_label,
        "match_id": match_id,
        "seq_len": len(sorted_timeline),
    }


class OddsDataset(Dataset):
    """
    Dataset for odds time-series matches.

    Modes:
        cutoff_mode="none" (default): one sample per JSONL line.
            Uses each sample's own 'cutoff_minutes' field or the global
            cutoff_minutes arg to filter the timeline.

        cutoff_mode="exhaustive": each match is expanded into one sample
            per cutoff value.  Filtering happens at init time.

        cutoff_mode="random": each __getitem__ randomly selects a cutoff
            from the cutoffs list and filters on the fly.

    Args:
        jsonl_path:      Path to JSONL file.
        max_seq_len:     Max time-steps per sample.
        cutoff_minutes:  Global cutoff override (used in "none" mode).
        cutoffs:         List of cutoff values for "exhaustive" / "random".
        cutoff_mode:     "none" | "exhaustive" | "random".
        min_events:      Minimum events required after filtering.
        seed:            Random seed for "random" mode.
    """

    def __init__(
        self,
        jsonl_path: str,
        max_seq_len: int = 64,
        cutoff_minutes: Optional[float] = None,
        cutoffs: Optional[List[float]] = None,
        cutoff_mode: str = "none",
        min_events: int = 1,
        seed: int = 42,
    ):
        if cutoff_mode not in ("none", "exhaustive", "random"):
            raise ValueError(f"Unknown cutoff_mode: {cutoff_mode}")

        self.max_seq_len = max_seq_len
        self.cutoff_minutes = cutoff_minutes
        self.cutoffs = cutoffs or []
        self.cutoff_mode = cutoff_mode
        self.min_events = min_events

        # Load raw matches
        raw_matches: List[dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw_matches.append(json.loads(line))

        self.num_raw_matches = len(raw_matches)

        if cutoff_mode == "exhaustive":
            # Pre-expand all matches
            self.samples = build_exhaustive_cutoff_dataset(
                raw_matches, self.cutoffs, min_events=min_events
            )
            # Count per cutoff
            self.cutoff_counts: Dict[str, int] = {}
            for s in self.samples:
                c = str(int(s["cutoff_minutes"]))
                self.cutoff_counts[c] = self.cutoff_counts.get(c, 0) + 1
        else:
            # Keep raw matches (filtering deferred to __getitem__)
            self.samples = raw_matches
            self.cutoff_counts = {}

        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]

        if self.cutoff_mode == "exhaustive":
            # Sample is already a filtered cutoff sample
            return _build_features_and_labels(sample, self.max_seq_len)

        elif self.cutoff_mode == "random":
            # Pick a random cutoff and filter
            match = sample
            valid_cutoffs = []
            for c in self.cutoffs:
                filtered = filter_timeline_by_cutoff(match["odds_timeline"], c)
                if len(filtered) >= self.min_events:
                    valid_cutoffs.append((c, filtered))

            if not valid_cutoffs:
                # Fallback: use the match's own cutoff or 0
                cutoff = self.cutoff_minutes or match.get("cutoff_minutes", 0)
                filtered = filter_timeline_by_cutoff(match["odds_timeline"], cutoff)
            else:
                cutoff, filtered = self._rng.choice(valid_cutoffs)

            # Build a temporary cutoff sample
            cutoff_sample = {
                "sample_id": f"{match['match_id']}__cutoff_{int(cutoff)}",
                "match_id": match["match_id"],
                "cutoff_minutes": cutoff,
                "odds_timeline": filtered,
                "label": match["label"],
            }
            return _build_features_and_labels(cutoff_sample, self.max_seq_len)

        else:
            # "none" mode — P0.1 behaviour
            cutoff = (
                self.cutoff_minutes
                if self.cutoff_minutes is not None
                else sample.get("cutoff_minutes", 0)
            )

            filtered = filter_timeline_by_cutoff(sample["odds_timeline"], cutoff)

            cutoff_sample = {
                "match_id": sample["match_id"],
                "odds_timeline": filtered,
                "label": sample["label"],
            }
            return _build_features_and_labels(cutoff_sample, self.max_seq_len)
