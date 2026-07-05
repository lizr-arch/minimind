import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p18_ah_robustness_slices import (
    DEFAULT_SLICE_KEYS,
    assign_time_quartiles,
    build_agreement_detail_rows,
    summarize_by_slice,
)


def test_p18_robustness_default_slices_include_seed_for_stability():
    assert "seed" in DEFAULT_SLICE_KEYS


def test_p18_robustness_builds_agreement_detail_rows_with_metadata():
    cover_by_seed = {42: [(0.30, 1.0, 0.90, 0.80), (-0.30, -1.0, 0.90, 0.80)]}
    direct_by_seed = {42: [(0.25, 1.0, 0.90, 0.80), (0.10, -1.0, 0.90, 0.80)]}
    contexts = [
        {"label_idx": 0, "match_id": "m1", "bookmaker_id": "Bet365", "league_id": "A", "line_slice": "flat", "kickoff_time": "2024-01-01"},
        {"label_idx": 1, "match_id": "m2", "bookmaker_id": "Macau", "league_id": "B", "line_slice": "favorite_ge_1_75", "kickoff_time": "2024-02-01"},
    ]

    rows = build_agreement_detail_rows(cover_by_seed, direct_by_seed, contexts, threshold=0.20, mode="both")

    assert rows[0]["selected"] is True
    assert rows[0]["decision"] == "upper"
    assert rows[0]["model_side_payoff"] == pytest.approx(0.90)
    assert rows[1]["selected"] is False
    assert rows[1]["bookmaker_id"] == "Macau"


def test_p18_robustness_summarizes_slices_and_flags_weak_selected_slice():
    rows = [
        {"selected": True, "decision": "upper", "actual_upper_units": 1.0, "model_side_units": 1.0, "model_side_payoff": 0.90, "direction_hit": 1.0, "bookmaker_id": "A"},
        {"selected": True, "decision": "upper", "actual_upper_units": -1.0, "model_side_units": -1.0, "model_side_payoff": -1.0, "direction_hit": 0.0, "bookmaker_id": "B"},
        {"selected": False, "decision": "hold", "actual_upper_units": 1.0, "model_side_units": 0.0, "model_side_payoff": 0.0, "direction_hit": None, "bookmaker_id": "B"},
    ]

    summaries = summarize_by_slice(rows, "bookmaker_id", min_selected=1)
    by_name = {row["slice_value"]: row for row in summaries}

    assert by_name["A"]["selected_n"] == 1
    assert by_name["A"]["status"] == "PASS"
    assert by_name["B"]["selected_n"] == 1
    assert by_name["B"]["status"] == "WEAK_SLICE"
    json.dumps(summaries)


def test_p18_robustness_assigns_time_quartiles_from_kickoff_strings():
    rows = [
        {"kickoff_time": "2024-01-01"},
        {"kickoff_time": "2024-02-01"},
        {"kickoff_time": "2024-03-01"},
        {"kickoff_time": "2024-04-01"},
    ]

    assign_time_quartiles(rows, n_buckets=4)

    assert [row["time_quartile"] for row in rows] == ["Q1", "Q2", "Q3", "Q4"]
