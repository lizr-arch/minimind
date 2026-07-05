"""Narrow tests for P4 residual goal-diff objective helpers."""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.odds_patch_itransformer_v2 import FEATURE_DEFS
from tools.p4_goal_diff_utils import (
    compute_anchor_only_metrics,
    compute_p4_losses,
    extract_euro_anchor_probs,
    fold_goal_diff_probs,
    goal_diff_to_bucket,
)
from tools.p4_train_residual_goal_diff import build_p4_dataset, selected_feature_names


FEATURE_INDEX = {name: idx for idx, (name, _) in enumerate(FEATURE_DEFS)}


def test_goal_diff_bucket_boundaries_and_fold_order_are_home_draw_away():
    diffs = [-5, -3, -2, -1, 0, 1, 2, 3, 7]
    assert [goal_diff_to_bucket(diff) for diff in diffs] == [0, 0, 1, 2, 3, 4, 5, 6, 6]

    q = torch.tensor([[0.10, 0.20, 0.05, 0.15, 0.25, 0.20, 0.05]])
    folded = fold_goal_diff_probs(q)

    assert torch.allclose(folded, torch.tensor([[0.50, 0.15, 0.35]]))


def test_extract_euro_anchor_probs_uses_nearest_non_negative_event_and_ignores_negative_time():
    row = {
        "raw_timeline": [
            {"minutes_before_kickoff": 45, "euro_h": 4.0, "euro_d": 4.0, "euro_a": 2.0},
            {"minutes_before_kickoff": 5, "euro_h": 2.0, "euro_d": 4.0, "euro_a": 4.0},
            {"minutes_before_kickoff": -3, "euro_h": 1.2, "euro_d": 8.0, "euro_a": 8.0},
        ]
    }

    probs, diag = extract_euro_anchor_probs(row)

    assert torch.allclose(probs, torch.tensor([0.5, 0.25, 0.25]))
    assert diag["used_fallback"] is False
    assert diag["anchor_minutes_before_kickoff"] == 5
    assert diag["negative_time_valid_euro_count"] == 1


def test_extract_euro_anchor_probs_falls_back_when_only_negative_time_odds_are_valid():
    row = {
        "raw_timeline": [
            {"minutes_before_kickoff": -1, "euro_h": 1.4, "euro_d": 7.0, "euro_a": 7.0},
            {"minutes_before_kickoff": 10, "euro_h": 0.0, "euro_d": 0.0, "euro_a": 0.0},
        ]
    }

    probs, diag = extract_euro_anchor_probs(row)

    assert torch.allclose(probs, torch.full((3,), 1.0 / 3.0))
    assert diag["used_fallback"] is True
    assert diag["negative_time_valid_euro_count"] == 1


def test_compute_p4_losses_combines_anchor_residual_diff_consistency_and_delta_penalty():
    delta_logits = torch.zeros(2, 3)
    goal_diff_logits = torch.tensor([
        [0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
    ])
    p_euro_anchor = torch.tensor([[0.5, 0.25, 0.25], [0.2, 0.3, 0.5]])
    y_1x2 = torch.tensor([0, 2])
    y_goal_diff = torch.tensor([3, 4])

    result = compute_p4_losses(
        delta_logits,
        goal_diff_logits,
        p_euro_anchor,
        y_1x2,
        y_goal_diff,
        diff_loss_weight=0.30,
        consistency_loss_weight=0.10,
        delta_l2_weight=0.01,
    )

    assert set(result) >= {"loss", "L_1x2", "L_diff", "L_consistency", "L_delta", "p_final", "p_from_diff"}
    assert result["p_final"].shape == (2, 3)
    assert result["p_from_diff"].shape == (2, 3)
    assert result["loss"].item() > result["L_1x2"].item()


def test_anchor_only_metrics_include_draw_diagnostics():
    probs = torch.tensor([
        [0.6, 0.2, 0.2],
        [0.4, 0.5, 0.1],
        [0.2, 0.3, 0.5],
    ])
    labels = torch.tensor([0, 1, 1])

    metrics = compute_anchor_only_metrics(probs, labels)

    assert set(metrics) >= {
        "anchor_only_val_logloss",
        "anchor_only_brier",
        "anchor_only_ECE",
        "anchor_only_acc",
        "argmax_draw_count",
        "draw_recall",
        "mean_p_draw",
        "mean_p_draw_on_true_draw",
    }
    assert metrics["argmax_draw_count"] == 1
    assert math.isclose(metrics["draw_recall"], 0.5)


def test_p4_feature_selection_keeps_euro_only_free_of_market_present_count_and_asian_presence():
    euro_names = selected_feature_names("euro")
    euro_asian_names = selected_feature_names("euro,asian")

    assert "market_present_count" not in euro_names
    assert "has_asian" not in euro_names
    assert euro_names[-2:] == ["time_log", "bucket_valid_count"]
    assert "has_asian" in euro_asian_names
    assert "market_present_count" not in euro_asian_names


def test_build_p4_dataset_returns_selected_features_anchors_and_label_counts():
    rows = [
        {
            "match_id": "m1",
            "raw_timeline": [
                {
                    "minutes_before_kickoff": 90,
                    "euro_h": 2.0,
                    "euro_d": 3.0,
                    "euro_a": 4.0,
                    "asian_source": "raw_update",
                    "asian_line": -1.0,
                    "upper_water": 0.9,
                    "lower_water": 1.0,
                }
            ],
            "label": {"euro_result": "home", "home_goals": 2, "away_goals": 0},
        }
    ]

    data = build_p4_dataset(rows, feature_groups="euro,asian")

    assert data["X"].shape == (1, 10, 17, 2)
    assert data["y_1x2"].tolist() == [0]
    assert data["y_goal_diff"].tolist() == [5]
    assert data["p_euro_anchor"].shape == (1, 3)
    assert data["anchor_fallback_count"] == 0
    assert data["negative_time_valid_euro_count"] == 0
