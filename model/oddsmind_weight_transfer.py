"""
OddsMind Pretrain-to-Supervised Weight Transfer (P0.7B)

Core functions to extract encoder + transformer weights from a pretrain
checkpoint and load them into a supervised OddsMindModel.

Does NOT transfer:
  - Pretrain heads (reconstruction, event prediction)
  - Supervised heads (euro, asian classification)
"""

import os
from typing import Dict, List, Optional

import torch


def extract_transferable_pretrain_state(
    pretrain_state_dict: Dict[str, torch.Tensor],
    backend: str = "odds_native",
) -> Dict[str, torch.Tensor]:
    """
    Extract encoder and transformer weights from a pretrain state_dict.

    Filters out:
      - recon_head.*
      - event_head.*
      - _minimind_adapter.* (handled separately per backend)

    Returns a dict of {key: tensor} that can be loaded into a supervised
    OddsMindModel via load_state_dict(..., strict=False).
    """
    transferable = {}
    skipped = []

    for key, tensor in pretrain_state_dict.items():
        # Skip pretrain-specific heads
        if key.startswith("recon_head.") or key.startswith("event_head."):
            skipped.append(key)
            continue

        transferable[key] = tensor

    return transferable, skipped


def load_pretrained_encoder_transformer(
    supervised_model: torch.nn.Module,
    pretrain_checkpoint_path: str,
    backend: str = "odds_native",
    strict_shapes: bool = True,
) -> dict:
    """
    Load encoder + transformer weights from a pretrain checkpoint
    into a supervised OddsMindModel.

    Args:
        supervised_model: an OddsMindModel instance (already initialized).
        pretrain_checkpoint_path: path to pretrain .pth file.
        backend: "odds_native" or "minimind".
        strict_shapes: if True, raise on shape mismatch.

    Returns:
        dict with keys: transferred, skipped, missing, unexpected, shape_mismatch.
    """
    pretrain_state = torch.load(pretrain_checkpoint_path, map_location="cpu")
    transferable, skipped = extract_transferable_pretrain_state(
        pretrain_state, backend=backend
    )

    # Get supervised model state keys
    sup_keys = set(supervised_model.state_dict().keys())

    matched = {}
    missing = set()
    shape_mismatch = []

    for key, tensor in transferable.items():
        if key not in sup_keys:
            missing.add(key)
            continue

        sup_tensor = supervised_model.state_dict()[key]
        if strict_shapes and tensor.shape != sup_tensor.shape:
            shape_mismatch.append((key, tensor.shape, sup_tensor.shape))
            continue

        matched[key] = tensor

    # Load matched weights
    if matched:
        supervised_model.load_state_dict(matched, strict=False)

    unexpected = sup_keys - set(transferable.keys())

    return {
        "transferred": len(matched),
        "transferred_keys": sorted(matched.keys()),
        "skipped": len(skipped),
        "skipped_keys": skipped,
        "missing": len(missing),
        "missing_keys": sorted(missing),
        "unexpected": len(unexpected),
        "unexpected_keys": sorted(unexpected),
        "shape_mismatch": len(shape_mismatch),
        "shape_mismatch_details": shape_mismatch,
    }
