"""Narrow tests for P4 training and report payloads."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p4_generate_report import build_final_report_payload
from tools.p4_train_residual_goal_diff import build_report_payload


def test_p4_training_report_payload_contains_required_sections_and_no_test_ids():
    payload = build_report_payload(
        config={"feature_groups": "euro", "consistency_loss_weight": 0.10},
        data={"train_samples": 10, "val_samples": 4, "test_ids_used": False},
        best_epoch=2,
        best_val_logloss=0.95,
        anchor_only_baseline={
            "anchor_only_val_logloss": 0.96,
            "anchor_only_brier": 0.56,
            "anchor_only_ECE": 0.02,
            "anchor_only_acc": 0.57,
            "argmax_draw_count": 0,
            "draw_recall": 0.0,
            "mean_p_draw": 0.25,
            "mean_p_draw_on_true_draw": 0.27,
        },
        train_metrics={"loss": 0.9},
        val_metrics={
            "logloss": 0.95,
            "brier": 0.55,
            "ece": 0.02,
            "accuracy": 0.58,
            "argmax_draw_count": 0,
            "draw_recall": 0.0,
            "mean_p_draw": 0.25,
            "mean_p_draw_on_true_draw": 0.27,
        },
        goal_diff_metrics={
            "bucket_acc": 0.3,
            "zero_recall": 0.2,
            "p_from_diff_logloss": 0.98,
            "final_vs_diff_kl": 0.01,
        },
        history=[],
        warnings=[],
    )

    assert set(payload) >= {
        "phase",
        "model",
        "config",
        "data",
        "best_epoch",
        "best_val_logloss",
        "anchor_only_baseline",
        "train_metrics",
        "val_metrics",
        "goal_diff_metrics",
        "history",
        "warnings",
    }
    assert payload["phase"] == "P4 residual goal-diff objective probe"
    assert payload["data"]["test_ids_used"] is False
    assert set(payload["anchor_only_baseline"]) >= {
        "anchor_only_val_logloss",
        "anchor_only_brier",
        "anchor_only_ECE",
        "anchor_only_acc",
    }


def test_p4_final_report_payload_requires_anchor_and_four_consistency_variants():
    variants = [
        {"name": "euro_default", "feature_groups": "euro", "loss_config": "default", "mean_val_logloss": 0.951},
        {
            "name": "euro_no_consistency",
            "feature_groups": "euro",
            "loss_config": "no_consistency",
            "mean_val_logloss": 0.949,
        },
        {
            "name": "euro_asian_default",
            "feature_groups": "euro,asian",
            "loss_config": "default",
            "mean_val_logloss": 0.948,
        },
        {
            "name": "euro_asian_no_consistency",
            "feature_groups": "euro,asian",
            "loss_config": "no_consistency",
            "mean_val_logloss": 0.950,
        },
    ]

    payload = build_final_report_payload(
        inputs={"test_ids_used": False},
        audit={"anchor_fallback_count": 1, "negative_time_valid_euro_count": 2},
        anchor_only_baseline={"anchor_only_val_logloss": 0.952},
        variants=variants,
        raw_mlp_label_smoothing_logloss=0.940862,
        p3_4_oof_fusion_logloss=0.941617,
    )

    names = {variant["name"] for variant in payload["p4_1"]["variants"]}
    assert {
        "euro_default",
        "euro_no_consistency",
        "euro_asian_default",
        "euro_asian_no_consistency",
    } <= names
    assert payload["inputs"]["test_ids_used"] is False
    assert "anchor_only_baseline" in payload
    assert "consistency_ablation" in payload["p4_1"]["comparison"]
    assert "P4_PLAN_APPROVED_WITH_FIXES" in payload["verdict"]
