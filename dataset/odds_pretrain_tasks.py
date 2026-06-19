"""
OddsMind Self-Supervised Pretraining Task Builders (P0.7)

Constructs training samples for three self-supervised tasks:

  1. masked_reconstruction — randomly mask event-feature positions
  2. next_event — predict the last event from preceding events
  3. closing_prediction — predict the final visible event from early events

All tasks operate on feature tensors [T, F] and return dicts suitable
for OddsPretrainDataset consumption.  No match labels are used.
"""

import random
from typing import Optional

import torch


def build_masked_reconstruction_sample(
    features: torch.Tensor,          # [T, F]
    mask_ratio: float = 0.15,
    mask_value: float = 0.0,
    seed: Optional[int] = None,
) -> dict:
    """
    Randomly mask a fraction of event-feature positions.

    Returns:
        corrupted_features: [T, F] with masked positions replaced by mask_value
        target_features:    [T, F] original values (for loss)
        target_mask:        [T, F] bool, True where masked (loss computed here)
    """
    T, F = features.shape
    rng = random.Random(seed)
    target_mask = torch.zeros(T, F, dtype=torch.bool)

    n_mask = max(1, int(T * F * mask_ratio))
    # Flatten indices, select without replacement
    all_indices = [(t, f) for t in range(T) for f in range(F)]
    chosen = rng.sample(all_indices, min(n_mask, len(all_indices)))
    for t, f in chosen:
        target_mask[t, f] = True

    corrupted = features.clone()
    corrupted[target_mask] = mask_value

    return {
        "corrupted_features": corrupted,
        "target_features": features.clone(),
        "target_mask": target_mask,
    }


def build_next_event_sample(
    features: torch.Tensor,          # [T, F]
    min_prefix_events: int = 1,
) -> dict:
    """
    Predict the last event from all preceding events.

    Returns:
        input_features: [T-1, F]  — all events except the last
        target_event:   [F]       — the last event
    """
    if features.shape[0] < min_prefix_events + 1:
        raise ValueError(
            f"Need at least {min_prefix_events + 1} events for next_event, "
            f"got {features.shape[0]}"
        )
    return {
        "input_features": features[:-1].clone(),
        "target_event": features[-1].clone(),
    }


def build_closing_prediction_sample(
    features: torch.Tensor,          # [T, F]
    min_prefix_events: int = 1,
) -> dict:
    """
    Predict the final (closest-to-cutoff) event from earlier events.

    Currently identical to next_event (scaffold).  Future versions may
    use more sophisticated prefix strategies (e.g. T-7d → closing).

    Returns:
        input_features: [T-1, F]
        target_event:   [F]
    """
    return build_next_event_sample(features, min_prefix_events=min_prefix_events)
