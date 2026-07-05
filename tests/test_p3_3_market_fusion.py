"""Narrow tests for P3.3 market-specific fusion diagnostics."""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p3_3_market_fusion_diagnostic import (
    build_market_features,
    build_report_payload,
    make_fusion_inputs,
    split_train_calibration_rows,
)


def test_train_calibration_split_is_deterministic_and_disjoint():
    rows = [{"match_id": f"m{i}"} for i in range(20)]

    base_a, calib_a = split_train_calibration_rows(rows, seed=42, calibration_fraction=0.25)
    base_b, calib_b = split_train_calibration_rows(rows, seed=42, calibration_fraction=0.25)

    assert [row["match_id"] for row in base_a] == [row["match_id"] for row in base_b]
    assert [row["match_id"] for row in calib_a] == [row["match_id"] for row in calib_b]
    assert set(row["match_id"] for row in base_a).isdisjoint(row["match_id"] for row in calib_a)
    assert len(calib_a) == 5


def test_market_features_clean_missing_asian_and_ou_defaults():
    row = {
        "raw_timeline": [
            {
                "minutes_before_kickoff": 60,
                "asian_source": "missing",
                "asian_line": 0.0,
                "upper_water": 1.0,
                "lower_water": 1.0,
                "over_under_source": "missing",
                "over_under_line": 2.5,
                "over_water": 1.0,
                "under_water": 1.0,
            }
        ],
        "label": {"euro_result": "home"},
    }

    asian = build_market_features(row, "asian")
    ou = build_market_features(row, "ou")

    assert asian[-1].item() == 0.0
    assert ou[-1].item() == 0.0
    assert torch.count_nonzero(asian[:-1]) == 0
    assert torch.count_nonzero(ou[:-1]) == 0


def test_make_fusion_inputs_concatenates_probabilities_in_market_order():
    probs = {
        "euro": torch.tensor([[0.6, 0.2, 0.2], [0.3, 0.4, 0.3]]),
        "asian": torch.tensor([[0.5, 0.3, 0.2], [0.2, 0.2, 0.6]]),
        "ou": torch.tensor([[0.4, 0.3, 0.3], [0.1, 0.5, 0.4]]),
    }

    fused = make_fusion_inputs(probs, ["euro", "asian", "ou"])

    assert fused.shape == (2, 9)
    assert torch.allclose(fused[0], torch.tensor([0.6, 0.2, 0.2, 0.5, 0.3, 0.2, 0.4, 0.3, 0.3]))


def test_market_fusion_report_payload_schema():
    payload = build_report_payload(
        runs=[],
        summary={"best_variant": "fusion"},
        inputs={"test_ids_used": False},
        verdict=["MARKET_FUSION_DIAGNOSTIC"],
    )

    assert set(payload) >= {"runs", "summary", "inputs", "verdict", "next_actions"}
    assert payload["inputs"]["test_ids_used"] is False
