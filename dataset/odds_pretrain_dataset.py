"""
OddsMind Pretrain Dataset (P0.7)

Wraps OddsDataset to produce self-supervised pretraining samples.
Supports masked_reconstruction, next_event, and closing_prediction tasks.
"""

from typing import Dict, List, Optional, Set

import torch
from torch.utils.data import Dataset

from dataset.odds_dataset import OddsDataset, _event_to_features
from dataset.odds_pretrain_tasks import (
    build_masked_reconstruction_sample,
    build_next_event_sample,
    build_closing_prediction_sample,
)


class OddsPretrainDataset(Dataset):
    """
    Dataset that yields self-supervised pretraining samples.

    Internally uses OddsDataset for cutoff filtering and split filtering.
    Task-specific sample construction happens in __getitem__.
    """

    def __init__(
        self,
        jsonl_path: str,
        max_seq_len: int = 64,
        cutoffs: Optional[List[float]] = None,
        cutoff_mode: str = "none",
        min_events: int = 2,   # pretrain needs at least 2 events typically
        allowed_match_ids: Optional[Set[str]] = None,
        task: str = "masked_reconstruction",
        mask_ratio: float = 0.15,
        seed: int = 42,
        **kwargs,
    ):
        if task not in ("masked_reconstruction", "next_event", "closing_prediction"):
            raise ValueError(f"Unknown task: {task}")

        self._odds_ds = OddsDataset(
            jsonl_path=jsonl_path,
            max_seq_len=max_seq_len,
            cutoffs=cutoffs,
            cutoff_mode=cutoff_mode,
            min_events=min_events,
            asian_label_mode="3class",  # irrelevant for pretrain
            allowed_match_ids=allowed_match_ids,
            seed=seed,
            **kwargs,
        )
        self.task = task
        self.mask_ratio = mask_ratio
        self.num_raw_matches = self._odds_ds.num_raw_matches
        self.cutoff_counts = self._odds_ds.cutoff_counts
        self.samples = self._odds_ds.samples

    def __len__(self) -> int:
        return len(self._odds_ds)

    def __getitem__(self, index: int) -> dict:
        # Get the underlying sample from OddsDataset
        sample = self._odds_ds.samples[index]
        timeline = sample.get("odds_timeline", [])
        cutoff = sample.get("cutoff_minutes", 0)

        # Filter and sort (belt-and-suspenders)
        filtered = sorted(
            [e for e in timeline if e["minutes_before_kickoff"] >= cutoff],
            key=lambda e: e["minutes_before_kickoff"],
            reverse=True,
        )
        # Build feature tensor [T, F]
        features = torch.tensor(
            [_event_to_features(e) for e in filtered],
            dtype=torch.float32,
        )

        match_id = sample.get("sample_id", sample.get("match_id", ""))

        base = {
            "match_id": match_id,
            "cutoff_minutes": cutoff,
            "task": self.task,
            "seq_len": features.shape[0],
        }

        if self.task == "masked_reconstruction":
            task_data = build_masked_reconstruction_sample(
                features, mask_ratio=self.mask_ratio
            )
            return {
                **base,
                "features": task_data["corrupted_features"],
                "target_features": task_data["target_features"],
                "target_mask": task_data["target_mask"],
            }
        elif self.task == "next_event":
            task_data = build_next_event_sample(features)
            return {
                **base,
                "features": task_data["input_features"],
                "target_event": task_data["target_event"],
            }
        else:  # closing_prediction
            task_data = build_closing_prediction_sample(features)
            return {
                **base,
                "features": task_data["input_features"],
                "target_event": task_data["target_event"],
            }
