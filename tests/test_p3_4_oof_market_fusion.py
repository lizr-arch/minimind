"""Narrow tests for P3.4 out-of-fold market fusion diagnostics."""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p3_4_oof_market_fusion import (
    assign_folds,
    build_oof_prediction_frame,
    build_report_payload,
)


def test_assign_folds_is_deterministic_balanced_and_covers_every_row_once():
    rows = [{"match_id": f"m{i}"} for i in range(11)]

    folds_a = assign_folds(rows, n_folds=5, seed=42)
    folds_b = assign_folds(rows, n_folds=5, seed=42)

    assert folds_a == folds_b
    assert sorted(folds_a) == [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert max(folds_a.count(fold) for fold in set(folds_a)) - min(folds_a.count(fold) for fold in set(folds_a)) <= 1


def test_build_oof_prediction_frame_reorders_fold_predictions_to_original_rows():
    labels = torch.tensor([0, 1, 2, 0])
    fold_predictions = {
        "euro": {
            0: (torch.tensor([1, 3]), torch.tensor([[0.2, 0.2, 0.6], [0.8, 0.1, 0.1]])),
            1: (torch.tensor([0, 2]), torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])),
        },
        "asian": {
            0: (torch.tensor([1, 3]), torch.tensor([[0.3, 0.4, 0.3], [0.5, 0.2, 0.3]])),
            1: (torch.tensor([0, 2]), torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])),
        },
    }

    frame = build_oof_prediction_frame(fold_predictions, ["euro", "asian"], labels)

    assert frame["features"].shape == (4, 6)
    assert torch.allclose(frame["features"][0], torch.tensor([0.7, 0.2, 0.1, 0.6, 0.3, 0.1]))
    assert torch.allclose(frame["features"][3], torch.tensor([0.8, 0.1, 0.1, 0.5, 0.2, 0.3]))
    assert torch.equal(frame["labels"], labels)
    assert frame["missing_count"] == 0


def test_p3_4_report_payload_schema():
    payload = build_report_payload(
        runs=[],
        summary={"best_variant": "oof_stacking"},
        inputs={"test_ids_used": False},
        verdict=["OOF_MARKET_FUSION_DIAGNOSTIC"],
    )

    assert set(payload) >= {"inputs", "runs", "summary", "verdict", "next_actions"}
    assert payload["inputs"]["test_ids_used"] is False
