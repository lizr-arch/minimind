"""Narrow tests for train-only P5 draw-prone segmentation."""

import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p5_draw_segmentation import (  # noqa: E402
    SEGMENT_NAMES,
    build_segmentation_report,
    load_prediction_rows,
    validate_segmentation_request,
)


def _sample_rows():
    return [
        {
            "match_id": "m1",
            "y_true": 1,
            "p_home": 0.25,
            "p_draw": 0.30,
            "p_away": 0.45,
            "p_anchor_home": 0.30,
            "p_anchor_draw": 0.38,
            "p_anchor_away": 0.32,
            "p_home_from_diff": 0.20,
            "p_draw_from_diff": 0.35,
            "p_away_from_diff": 0.45,
            "pred_class": 2,
        },
        {
            "match_id": "m2",
            "y_true": 0,
            "p_home": 0.60,
            "p_draw": 0.20,
            "p_away": 0.20,
            "p_anchor_home": 0.52,
            "p_anchor_draw": 0.30,
            "p_anchor_away": 0.18,
            "p_home_from_diff": 0.55,
            "p_draw_from_diff": 0.25,
            "p_away_from_diff": 0.20,
            "pred_class": 0,
        },
        {
            "match_id": "m3",
            "y_true": 2,
            "p_home": 0.15,
            "p_draw": 0.25,
            "p_away": 0.60,
            "p_anchor_home": 0.20,
            "p_anchor_draw": 0.28,
            "p_anchor_away": 0.52,
            "p_home_from_diff": 0.18,
            "p_draw_from_diff": 0.22,
            "p_away_from_diff": 0.60,
            "pred_class": 2,
        },
        {
            "match_id": "m4",
            "y_true": 1,
            "p_home": 0.40,
            "p_draw": 0.34,
            "p_away": 0.26,
            "p_anchor_home": 0.32,
            "p_anchor_draw": 0.42,
            "p_anchor_away": 0.26,
            "p_home_from_diff": 0.36,
            "p_draw_from_diff": 0.30,
            "p_away_from_diff": 0.34,
            "pred_class": 0,
        },
    ]


def test_segmentation_refuses_test_split():
    with pytest.raises(ValueError, match="test split"):
        validate_segmentation_request(split="test", allow_val_diagnostic=False)


def test_segmentation_requires_train_for_param_selection():
    with pytest.raises(ValueError, match="validation diagnostic"):
        validate_segmentation_request(split="val", allow_val_diagnostic=False)

    validate_segmentation_request(split="val", allow_val_diagnostic=True)

    with pytest.raises(ValueError, match="validation diagnostic"):
        build_segmentation_report(_sample_rows(), split="val", n_bins=2)


def test_segmentation_bins_cover_all_rows():
    report = build_segmentation_report(_sample_rows(), split="train", n_bins=2)

    assert set(report["segments"]) == set(SEGMENT_NAMES)
    for segment in report["segments"].values():
        assert sum(item["n"] for item in segment["bins"]) == 4


def test_segmentation_outputs_draw_gap_metrics():
    report = build_segmentation_report(_sample_rows(), split="train", n_bins=2)
    gap_segment = report["segments"]["final_vs_diff_draw_gap_decile"]

    assert gap_segment["metric"] == "final_vs_diff_draw_gap"
    assert gap_segment["threshold_source"] == "input_split"
    assert set(gap_segment["bins"][0]) >= {
        "n",
        "true_draw_rate",
        "mean_final_p_draw",
        "mean_anchor_p_draw",
        "mean_p_diff_draw",
        "draw_calibration_gap",
        "final_logloss",
        "draw_class_nll",
        "home_away_logloss_on_non_draw",
        "argmax_draw_count",
    }


def test_load_prediction_rows_requires_exported_anchor_and_diff_columns(tmp_path):
    path = tmp_path / "predictions.csv"
    rows = _sample_rows()
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    loaded = load_prediction_rows(path)

    assert loaded[0]["match_id"] == "m1"
    assert loaded[0]["p_anchor_draw"] == pytest.approx(0.38)
    assert loaded[0]["p_draw_from_diff"] == pytest.approx(0.35)
