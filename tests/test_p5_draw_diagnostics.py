"""Narrow tests for P5 draw / goal-diff diagnostics."""

import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p5_draw_diagnostics import (
    aggregate_variant_diagnostics,
    build_report_payload,
    compute_argmax_draw_stats,
    compute_draw_probability_summary,
    compute_draw_threshold_curve,
    compute_final_vs_diff_draw_gap,
    compute_p_draw_calibration_bins,
    load_and_merge_prediction_rows,
)


def _write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _merged_rows():
    return [
        {
            "match_id": "m1",
            "y_true": 1,
            "p_home": 0.30,
            "p_draw": 0.40,
            "p_away": 0.30,
            "pred_class": 1,
            "p_draw_from_diff": 0.25,
        },
        {
            "match_id": "m2",
            "y_true": 1,
            "p_home": 0.50,
            "p_draw": 0.20,
            "p_away": 0.30,
            "pred_class": 0,
            "p_draw_from_diff": 0.35,
        },
        {
            "match_id": "m3",
            "y_true": 0,
            "p_home": 0.80,
            "p_draw": 0.10,
            "p_away": 0.10,
            "pred_class": 0,
            "p_draw_from_diff": 0.20,
        },
        {
            "match_id": "m4",
            "y_true": 2,
            "p_home": 0.10,
            "p_draw": 0.70,
            "p_away": 0.20,
            "pred_class": 1,
            "p_draw_from_diff": 0.50,
        },
    ]


def test_load_and_merge_prediction_rows_by_match_id(tmp_path):
    one_x_two = tmp_path / "val_predictions.csv"
    goal_diff = tmp_path / "val_goal_diff_predictions.csv"
    _write_csv(
        one_x_two,
        [
            {
                "match_id": "m1",
                "y_true": "1",
                "p_home": "0.3",
                "p_draw": "0.4",
                "p_away": "0.3",
                "pred_class": "1",
                "correct": "1",
            },
            {
                "match_id": "m2",
                "y_true": "0",
                "p_home": "0.6",
                "p_draw": "0.2",
                "p_away": "0.2",
                "pred_class": "0",
                "correct": "1",
            },
        ],
    )
    _write_csv(
        goal_diff,
        [
            {
                "match_id": "m2",
                "y_goal_diff": "-1",
                "p_bucket_0": "0.1",
                "p_bucket_1": "0.1",
                "p_bucket_2": "0.1",
                "p_bucket_3": "0.2",
                "p_bucket_4": "0.2",
                "p_bucket_5": "0.2",
                "p_bucket_6": "0.1",
                "p_home_from_diff": "0.4",
                "p_draw_from_diff": "0.2",
                "p_away_from_diff": "0.4",
            },
            {
                "match_id": "m1",
                "y_goal_diff": "0",
                "p_bucket_0": "0.0",
                "p_bucket_1": "0.0",
                "p_bucket_2": "0.0",
                "p_bucket_3": "0.8",
                "p_bucket_4": "0.1",
                "p_bucket_5": "0.1",
                "p_bucket_6": "0.0",
                "p_home_from_diff": "0.1",
                "p_draw_from_diff": "0.8",
                "p_away_from_diff": "0.1",
            },
        ],
    )

    rows = load_and_merge_prediction_rows(one_x_two, goal_diff)

    assert [row["match_id"] for row in rows] == ["m1", "m2"]
    assert rows[0]["y_true"] == 1
    assert rows[0]["y_goal_diff"] == 0
    assert rows[0]["p_draw"] == 0.4
    assert rows[0]["p_draw_from_diff"] == 0.8


def test_load_and_merge_prediction_rows_preserves_duplicate_match_id_occurrence_order(tmp_path):
    one_x_two = tmp_path / "val_predictions.csv"
    goal_diff = tmp_path / "val_goal_diff_predictions.csv"
    _write_csv(
        one_x_two,
        [
            {
                "match_id": "dup",
                "y_true": "1",
                "p_home": "0.2",
                "p_draw": "0.6",
                "p_away": "0.2",
                "pred_class": "1",
                "correct": "1",
            },
            {
                "match_id": "dup",
                "y_true": "0",
                "p_home": "0.6",
                "p_draw": "0.3",
                "p_away": "0.1",
                "pred_class": "0",
                "correct": "1",
            },
        ],
    )
    _write_csv(
        goal_diff,
        [
            {
                "match_id": "dup",
                "y_goal_diff": "0",
                "p_draw_from_diff": "0.7",
                "p_home_from_diff": "0.2",
                "p_away_from_diff": "0.1",
            },
            {
                "match_id": "dup",
                "y_goal_diff": "2",
                "p_draw_from_diff": "0.2",
                "p_home_from_diff": "0.7",
                "p_away_from_diff": "0.1",
            },
        ],
    )

    rows = load_and_merge_prediction_rows(one_x_two, goal_diff)

    assert [row["y_goal_diff"] for row in rows] == [0, 2]
    assert [row["p_draw_from_diff"] for row in rows] == [0.7, 0.2]


def test_load_and_merge_prediction_rows_rejects_unconsumed_goal_diff_occurrences(tmp_path):
    one_x_two = tmp_path / "val_predictions.csv"
    goal_diff = tmp_path / "val_goal_diff_predictions.csv"
    _write_csv(
        one_x_two,
        [
            {
                "match_id": "dup",
                "y_true": "1",
                "p_home": "0.2",
                "p_draw": "0.6",
                "p_away": "0.2",
                "pred_class": "1",
                "correct": "1",
            },
        ],
    )
    _write_csv(
        goal_diff,
        [
            {
                "match_id": "dup",
                "y_goal_diff": "3",
                "p_draw_from_diff": "0.7",
                "p_home_from_diff": "0.2",
                "p_away_from_diff": "0.1",
            },
            {
                "match_id": "dup",
                "y_goal_diff": "4",
                "p_draw_from_diff": "0.2",
                "p_home_from_diff": "0.7",
                "p_away_from_diff": "0.1",
            },
        ],
    )

    try:
        load_and_merge_prediction_rows(one_x_two, goal_diff)
    except ValueError as exc:
        assert "unconsumed goal-diff" in str(exc)
    else:
        raise AssertionError("expected unconsumed duplicate goal-diff occurrence to fail")


def test_draw_threshold_curve_reports_counts_recall_precision_without_changing_logloss():
    curve = compute_draw_threshold_curve(_merged_rows(), thresholds=[0.25, 0.50])

    assert curve == [
        {
            "threshold": 0.25,
            "predicted_draw_count": 2,
            "draw_recall": 0.5,
            "precision": 0.5,
            "draw_precision": 0.5,
            "decision_accuracy": 0.5,
            "logloss_unchanged": True,
        },
        {
            "threshold": 0.5,
            "predicted_draw_count": 1,
            "draw_recall": 0.0,
            "precision": 0.0,
            "draw_precision": 0.0,
            "decision_accuracy": 0.5,
            "logloss_unchanged": True,
        },
    ]


def test_p_draw_calibration_bins_report_count_draw_rate_and_mean_probability():
    bins = compute_p_draw_calibration_bins(_merged_rows(), n_bins=2)

    assert bins == [
        {"bin": 0, "low": 0.0, "high": 0.5, "count": 3, "draw_rate": 2 / 3, "mean_p_draw": 0.233333},
        {"bin": 1, "low": 0.5, "high": 1.0, "count": 1, "draw_rate": 0.0, "mean_p_draw": 0.7},
    ]


def test_final_vs_diff_draw_gap_summarizes_all_rows_and_true_draw_rows():
    gap = compute_final_vs_diff_draw_gap(_merged_rows())

    assert gap == {
        "mean_gap": 0.025,
        "mean_abs_gap": 0.15,
        "true_draw_gap": 0.0,
        "n_rows": 4,
        "n_true_draw": 2,
    }


def test_draw_probability_summary_reports_overall_true_and_non_draw_means():
    summary = compute_draw_probability_summary(_merged_rows())

    assert summary == {
        "mean_p_draw": 0.35,
        "mean_p_draw_on_true_draw": 0.3,
        "mean_p_draw_on_non_draw": 0.4,
    }


def test_argmax_draw_stats_reports_final_and_diff_draw_counts():
    stats = compute_argmax_draw_stats(_merged_rows())

    assert stats == {
        "n_rows": 4,
        "true_draw_count": 2,
        "final_argmax_draw_count": 2,
        "final_draw_recall": 0.5,
        "diff_argmax_draw_count": 2,
        "diff_draw_recall": 0.5,
    }


def test_aggregate_variant_diagnostics_summarizes_cross_seed_runs():
    runs = [
        {
            "run": "p4_euro_default_seed42",
            "variant": "euro_default",
            "diagnostics": {
                "argmax_draw_stats": {
                    "final_argmax_draw_count": 1,
                    "final_draw_recall": 0.1,
                    "diff_argmax_draw_count": 0,
                    "diff_draw_recall": 0.0,
                },
                "draw_probability_summary": {
                    "mean_p_draw": 0.20,
                    "mean_p_draw_on_true_draw": 0.22,
                    "mean_p_draw_on_non_draw": 0.19,
                },
                "final_vs_diff_draw_gap": {
                    "mean_gap": 0.01,
                    "mean_abs_gap": 0.02,
                    "true_draw_gap": 0.03,
                },
            },
        },
        {
            "run": "p4_euro_default_seed123",
            "variant": "euro_default",
            "diagnostics": {
                "argmax_draw_stats": {
                    "final_argmax_draw_count": 3,
                    "final_draw_recall": 0.3,
                    "diff_argmax_draw_count": 2,
                    "diff_draw_recall": 0.2,
                },
                "draw_probability_summary": {
                    "mean_p_draw": 0.24,
                    "mean_p_draw_on_true_draw": 0.26,
                    "mean_p_draw_on_non_draw": 0.23,
                },
                "final_vs_diff_draw_gap": {
                    "mean_gap": -0.01,
                    "mean_abs_gap": 0.04,
                    "true_draw_gap": 0.01,
                },
            },
        },
    ]

    summary = aggregate_variant_diagnostics(runs)

    assert summary["euro_default"] == {
        "run_count": 2,
        "mean_final_argmax_draw_count": 2.0,
        "mean_final_draw_recall": 0.2,
        "mean_diff_argmax_draw_count": 1.0,
        "mean_diff_draw_recall": 0.1,
        "mean_p_draw": 0.22,
        "mean_p_draw_on_true_draw": 0.24,
        "mean_p_draw_on_non_draw": 0.21,
        "mean_gap": 0.0,
        "mean_abs_gap": 0.03,
        "mean_true_draw_gap": 0.02,
    }


def test_build_report_payload_has_required_schema_and_no_test_ids():
    payload = build_report_payload(
        p4_root="runs/p4_residual_goal_diff",
        run_diagnostics=[
            {
                "run": "p4_euro_default_seed42",
                "variant": "euro_default",
                "diagnostics": {
                    "argmax_draw_stats": {
                        "final_argmax_draw_count": 0,
                        "final_draw_recall": 0.0,
                    },
                    "final_vs_diff_draw_gap": {"mean_abs_gap": 0.2},
                    "draw_probability_summary": {
                        "mean_p_draw": 0.2,
                        "mean_p_draw_on_true_draw": 0.21,
                        "mean_p_draw_on_non_draw": 0.19,
                    },
                },
            }
        ],
        p4_report={"verdict": ["DRAW_STILL_COLLAPSED"]},
    )

    assert set(payload) >= {"inputs", "models", "diagnostics", "verdict", "next_actions", "test_ids_used"}
    assert payload["inputs"]["p4_root"] == "runs/p4_residual_goal_diff"
    assert payload["test_ids_used"] is False
    assert payload["models"]["p4_euro_default_seed42"]["variant"] == "euro_default"
    assert payload["diagnostics"]["run_count"] == 1
    assert payload["diagnostics"]["variant_summaries"]["euro_default"]["run_count"] == 1
    assert "DRAW_STILL_COLLAPSED" in payload["verdict"]
