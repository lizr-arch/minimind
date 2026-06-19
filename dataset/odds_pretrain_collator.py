"""
OddsMind Pretrain Collator (P0.7)

Pads variable-length sequences and builds batch dictionaries for
each pretraining task.
"""

from typing import List

import torch


class OddsPretrainCollator:
    """Callable collator for OddsPretrainDataset."""

    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value

    def __call__(self, batch: List[dict]) -> dict:
        if not batch:
            return {}

        task = batch[0]["task"]
        feature_dim = batch[0]["features"].shape[-1]

        if task == "masked_reconstruction":
            return self._collate_masked(batch, feature_dim)
        else:
            return self._collate_event_pred(batch, feature_dim)

    def _collate_masked(self, batch, feature_dim):
        B = len(batch)
        max_len = max(item["features"].shape[0] for item in batch)

        features = torch.full((B, max_len, feature_dim), self.pad_value, dtype=torch.float32)
        target_features = torch.full((B, max_len, feature_dim), self.pad_value, dtype=torch.float32)
        target_mask = torch.zeros(B, max_len, feature_dim, dtype=torch.bool)
        attention_mask = torch.zeros(B, max_len, dtype=torch.bool)

        match_ids = []
        cutoffs = []
        for i, item in enumerate(batch):
            sl = item["features"].shape[0]
            features[i, :sl] = item["features"]
            target_features[i, :sl] = item["target_features"]
            target_mask[i, :sl] = item["target_mask"]
            attention_mask[i, :sl] = True
            match_ids.append(item["match_id"])
            cutoffs.append(item.get("cutoff_minutes", 0))

        return {
            "task": "masked_reconstruction",
            "features": features,
            "target_features": target_features,
            "target_mask": target_mask,
            "attention_mask": attention_mask,
            "match_ids": match_ids,
            "cutoffs": cutoffs,
        }

    def _collate_event_pred(self, batch, feature_dim):
        B = len(batch)
        max_len = max(item["features"].shape[0] for item in batch)

        features = torch.full((B, max_len, feature_dim), self.pad_value, dtype=torch.float32)
        target_event = torch.zeros(B, feature_dim, dtype=torch.float32)
        attention_mask = torch.zeros(B, max_len, dtype=torch.bool)

        match_ids = []
        cutoffs = []
        for i, item in enumerate(batch):
            sl = item["features"].shape[0]
            features[i, :sl] = item["features"]
            target_event[i] = item["target_event"]
            attention_mask[i, :sl] = True
            match_ids.append(item["match_id"])
            cutoffs.append(item.get("cutoff_minutes", 0))

        return {
            "task": batch[0]["task"],
            "features": features,
            "target_event": target_event,
            "attention_mask": attention_mask,
            "match_ids": match_ids,
            "cutoffs": cutoffs,
        }
