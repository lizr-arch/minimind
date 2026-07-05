import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p18_ah_cover_diagnostics import (
    P18ProtocolError,
    assert_no_test_paths,
    build_source_context_rows,
    build_report_payload,
    expected_upper_units_from_goal_diff_probs,
    latest_pre_kickoff_asian_snapshot,
    settle_upper_units,
    summarize_ah_rows,
)


def test_p18_settles_quarter_line_upper_units_from_home_perspective():
    assert settle_upper_units(home_goals=3, away_goals=0, asian_line=-2.25) == pytest.approx(1.0)
    assert settle_upper_units(home_goals=2, away_goals=0, asian_line=-2.25) == pytest.approx(-0.5)
    assert settle_upper_units(home_goals=1, away_goals=0, asian_line=-2.25) == pytest.approx(-1.0)
    assert settle_upper_units(home_goals=1, away_goals=1, asian_line=0.0) == pytest.approx(0.0)
    assert settle_upper_units(home_goals=1, away_goals=0, asian_line=-0.25) == pytest.approx(1.0)


def test_p18_expected_units_from_goal_diff_distribution_uses_canonical_settlement():
    q = {
        "away_by_3plus": 0.0,
        "away_by_2": 0.0,
        "away_by_1": 0.0,
        "draw": 0.10,
        "home_by_1": 0.20,
        "home_by_2": 0.30,
        "home_by_3plus": 0.40,
    }

    expected = expected_upper_units_from_goal_diff_probs(q, asian_line=-2.25)

    assert expected["upper_win_prob"] == pytest.approx(0.40)
    assert expected["upper_loss_prob"] == pytest.approx(0.60)
    assert expected["expected_upper_units"] == pytest.approx(0.40 - 0.30 * 0.5 - 0.30)
    assert expected["coarse_bucket_warning"] is False


def test_p18_latest_pre_kickoff_asian_snapshot_ignores_negative_time():
    row = {
        "odds_timeline": [
            {"minutes_before_kickoff": 20, "asian_line": -0.5, "upper_water": 0.90, "lower_water": 1.00, "asian_source": "raw_update"},
            {"minutes_before_kickoff": -5, "asian_line": -1.5, "upper_water": 0.80, "lower_water": 1.10, "asian_source": "raw_update"},
            {"minutes_before_kickoff": 5, "asian_line": -0.75, "upper_water": 0.95, "lower_water": 0.95, "asian_source": "forward_fill"},
        ]
    }

    snap = latest_pre_kickoff_asian_snapshot(row)

    assert snap["asian_line"] == pytest.approx(-0.75)
    assert snap["minutes_before_kickoff"] == pytest.approx(5)
    assert snap["negative_time_valid_asian_count"] == 1


def test_p18_source_context_preserves_label_index_when_ah_rows_are_missing():
    rows = [
        {
            "match_id": "missing_ah",
            "label": {"euro_result": "home", "home_goals": 1, "away_goals": 0},
            "odds_timeline": [
                {"minutes_before_kickoff": 10, "asian_line": 0.0, "upper_water": 0.0, "lower_water": 0.0, "asian_source": "missing"}
            ],
        },
        {
            "match_id": "valid_ah",
            "label": {"euro_result": "home", "home_goals": 2, "away_goals": 0},
            "odds_timeline": [
                {"minutes_before_kickoff": 10, "asian_line": -1.5, "upper_water": 0.90, "lower_water": 1.00, "asian_source": "raw_update"}
            ],
        },
    ]

    context = build_source_context_rows(rows)

    assert len(context) == 1
    assert context[0]["match_id"] == "valid_ah"
    assert context[0]["label_idx"] == 1


def test_p18_summary_reports_directional_units_and_slice_metrics():
    rows = [
        {
            "seed": 42,
            "slice": "all",
            "actual_upper_units": 1.0,
            "expected_upper_units": 0.20,
            "predicted_side": "upper",
            "asian_line": -0.5,
        },
        {
            "seed": 42,
            "slice": "all",
            "actual_upper_units": -0.5,
            "expected_upper_units": -0.10,
            "predicted_side": "lower",
            "asian_line": -2.25,
        },
        {
            "seed": 42,
            "slice": "all",
            "actual_upper_units": -1.0,
            "expected_upper_units": 0.15,
            "predicted_side": "upper",
            "asian_line": 0.0,
        },
    ]

    summary = summarize_ah_rows(rows, seed=42, slice_name="all")

    assert summary["n"] == 3
    assert summary["model_side_avg_units"] == pytest.approx((1.0 + 0.5 - 1.0) / 3.0)
    assert summary["direction_accuracy_ex_push"] == pytest.approx(2 / 3)
    assert summary["upper_pick_rate"] == pytest.approx(2 / 3)
    assert summary["mean_line"] == pytest.approx((-0.5 - 2.25 + 0.0) / 3.0)


def test_p18_report_payload_schema_and_no_test_guard():
    with pytest.raises(P18ProtocolError, match="test split"):
        assert_no_test_paths(["data/odds_real/splits_v6/test_match_ids.txt"])

    payload = build_report_payload(
        by_seed=[{"seed": 42, "slice": "all", "n": 3, "model_side_avg_units": 0.1}],
        aggregate=[{"slice": "all", "seed_count": 1, "mean_model_side_avg_units": 0.1}],
        metadata={"phase": "P18.0", "run_count": 1},
    )

    assert payload["phase"] == "P18.0"
    assert payload["input_policy"]["no_test_split_loaded"] is True
    assert payload["verdict"] in {"P18_AH_DIAGNOSTIC_READY", "P18_AH_DIAGNOSTIC_INSUFFICIENT"}
    assert set(payload) >= {"summary_by_seed", "summary_aggregate", "verdict", "next_actions"}
    json.dumps(payload)
