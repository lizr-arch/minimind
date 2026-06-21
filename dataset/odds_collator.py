"""
OddsMind Collator — pads variable-length odds timelines into a batch.

Output batch dict:
    features:        [batch_size, max_seq_len, feature_dim]  float32
    attention_mask:  [batch_size, max_seq_len]                bool (True=valid)
    euro_labels:     [batch_size]                              int64
    asian_labels:    [batch_size]                              int64
    match_ids:       list[str]                                 length = batch_size
"""

from typing import List

import torch


class OddsCollator:
    """
    Collate function (callable) for DataLoader.

    Pads sequences to the maximum length in the batch using zero-padding.
    Attention mask is True for real time-steps, False for padding.
    """

    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value

    def __call__(self, batch: List[dict]) -> dict:
        batch_size = len(batch)
        feature_dim = batch[0]["features"].shape[-1]

        # Find max sequence length in this batch
        max_len = max(item["features"].shape[0] for item in batch)

        # Allocate padded tensors
        features = torch.full(
            (batch_size, max_len, feature_dim),
            self.pad_value,
            dtype=torch.float32,
        )
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        euro_labels = torch.zeros(batch_size, dtype=torch.long)
        asian_labels = torch.zeros(batch_size, dtype=torch.long)
        score_labels = torch.zeros(batch_size, 2, dtype=torch.float32)
        bookmaker_ids = torch.zeros(batch_size, dtype=torch.long)
        match_ids = []

        for i, item in enumerate(batch):
            seq_len = item["features"].shape[0]
            features[i, :seq_len] = item["features"]
            attention_mask[i, :seq_len] = True
            euro_labels[i] = item["euro_label"]
            asian_labels[i] = item["asian_label"]
            score_labels[i] = item["score_label"]
            bookmaker_ids[i] = item["bookmaker_id"]
            match_ids.append(item["match_id"])

        return {
            "features": features,
            "attention_mask": attention_mask,
            "euro_labels": euro_labels,
            "asian_labels": asian_labels,
            "score_labels": score_labels,
            "bookmaker_ids": bookmaker_ids,
            "match_ids": match_ids,
        }
