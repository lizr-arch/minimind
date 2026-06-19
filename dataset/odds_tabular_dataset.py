"""
OddsMind Tabular Dataset (P0.5B)

Wraps OddsDataset to output fixed-size feature vectors instead of
variable-length timeline tensors.  Supports all OddsDataset modes
(cutoffs, allowed_match_ids, asian_label_mode).
"""

from typing import Dict, List, Optional, Set

import torch
from torch.utils.data import Dataset

from dataset.odds_dataset import OddsDataset
from dataset.odds_features import extract_features


class OddsTabularDataset(Dataset):
    """
    Dataset that yields (features, euro_label, asian_label, metadata)
    tuples where features is a fixed-size [feature_dim] float tensor.

    Internally uses OddsDataset for cutoff filtering, split filtering,
    and label mapping.  Feature extraction happens in __getitem__.
    """

    def __init__(
        self,
        jsonl_path: str,
        max_seq_len: int = 64,
        cutoff_minutes: Optional[float] = None,
        cutoffs: Optional[List[float]] = None,
        cutoff_mode: str = "none",
        min_events: int = 1,
        asian_label_mode: str = "3class",
        allowed_match_ids: Optional[Set[str]] = None,
        seed: int = 42,
    ):
        self._odds_ds = OddsDataset(
            jsonl_path=jsonl_path,
            max_seq_len=max_seq_len,
            cutoff_minutes=cutoff_minutes,
            cutoffs=cutoffs,
            cutoff_mode=cutoff_mode,
            min_events=min_events,
            asian_label_mode=asian_label_mode,
            allowed_match_ids=allowed_match_ids,
            seed=seed,
        )
        self.asian_label_mode = asian_label_mode
        self.num_raw_matches = self._odds_ds.num_raw_matches
        self.cutoff_counts = self._odds_ds.cutoff_counts
        self.cutoff_mode = cutoff_mode
        # Expose internal samples for direct baseline eval
        self.samples = self._odds_ds.samples

    def __len__(self) -> int:
        return len(self._odds_ds)

    def __getitem__(self, index: int) -> dict:
        # Get the enriched sample from OddsDataset
        item = self._odds_ds[index]

        # We need the original timeline events (not the tensor).
        # OddsDataset stores samples in self.samples; for cutoff modes,
        # the sample dict has an 'odds_timeline' key with the filtered events.
        sample = self._odds_ds.samples[index]
        timeline = sample.get("odds_timeline", [])

        # Determine the cutoff for this specific sample
        cutoff = sample.get("cutoff_minutes", 0)

        # Filter timeline again (belt-and-suspenders: OddsDataset already did this)
        filtered = [e for e in timeline if e["minutes_before_kickoff"] >= cutoff]

        features = extract_features(filtered)

        return {
            "features": features,
            "euro_label": item["euro_label"],
            "asian_label": item["asian_label"],
            "match_id": item["match_id"],
            "cutoff_minutes": cutoff,
        }
