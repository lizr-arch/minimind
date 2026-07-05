import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p18_ah_fixture_strategy import (
    ah_side_from_expected_units,
    build_ah_strategy_payload,
    expected_upper_units_from_probs,
    summarize_ah_strategy_rows,
)


def test_p18_fixture_strategy_converts_cover_probs_to_expected_units_and_side():
    expected = expected_upper_units_from_probs([0.60, 0.10, 0.10, 0.10, 0.10])

    assert expected == pytest.approx(0.50)
    assert ah_side_from_expected_units(expected, threshold=0.20) == "upper"
    assert ah_side_from_expected_units(-0.30, threshold=0.20) == "lower"
    assert ah_side_from_expected_units(0.10, threshold=0.20) == "hold"


def test_p18_fixture_strategy_summarizes_seed_and_bookmaker_rows_by_threshold():
    rows = [
        {"variant": "p18_ahw_010", "seed": 42, "bookmaker": "A", "expected_upper_units": 0.30, "asian_line": -2.25},
        {"variant": "p18_ahw_010", "seed": 123, "bookmaker": "A", "expected_upper_units": 0.20, "asian_line": -2.25},
        {"variant": "p18_ahw_010", "seed": 42, "bookmaker": "B", "expected_upper_units": 0.10, "asian_line": -2.25},
        {"variant": "p18_ahw_010", "seed": 123, "bookmaker": "B", "expected_upper_units": 0.20, "asian_line": -2.25},
    ]

    summary = summarize_ah_strategy_rows(rows, thresholds=[0.20, 0.25])

    assert summary["mean_expected_upper_units"] == pytest.approx(0.20)
    assert summary["mean_abs_expected_units"] == pytest.approx(0.20)
    assert summary["threshold_decisions"][0]["threshold"] == pytest.approx(0.20)
    assert summary["threshold_decisions"][0]["decision"] == "upper"
    assert summary["threshold_decisions"][1]["threshold"] == pytest.approx(0.25)
    assert summary["threshold_decisions"][1]["decision"] == "hold"


def test_p18_fixture_strategy_payload_keeps_pre_match_guardrails():
    input_rows = [
        {
            "match_id": "world_cup_2907390",
            "home_team": "Argentina",
            "away_team": "Cape Verde",
            "kickoff_time_utc": "2026-07-03T22:00:00+00:00",
            "label_status": "not_available_pre_match",
            "label": None,
        }
    ]
    strategy_rows = [
        {"variant": "p18_ahw_010", "seed": 42, "bookmaker": "A", "expected_upper_units": -0.30, "asian_line": -2.25},
        {"variant": "p18_ahw_010", "seed": 123, "bookmaker": "A", "expected_upper_units": -0.20, "asian_line": -2.25},
    ]

    payload = build_ah_strategy_payload(input_rows, strategy_rows, thresholds=[0.20, 0.25], variant="p18_ahw_010")

    assert payload["fixture"]["match_id"] == "world_cup_2907390"
    assert payload["input_policy"]["no_label_fabricated"] is True
    assert payload["strategy"]["threshold_decisions"][0]["decision"] == "lower"
    assert payload["strategy"]["threshold_decisions"][1]["decision"] == "lower"
    json.dumps(payload)
