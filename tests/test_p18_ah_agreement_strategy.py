import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p18_ah_agreement_strategy import (
    agreement_decision,
    agreement_stats,
    build_fixture_agreement_payload,
    combine_fixture_strategy_rows,
)


def test_p18_agreement_decision_requires_same_direction_and_threshold():
    assert agreement_decision(0.30, 0.10, threshold=0.20, mode="any") == "upper"
    assert agreement_decision(0.30, 0.10, threshold=0.20, mode="both") == "hold"
    assert agreement_decision(0.30, -0.40, threshold=0.20, mode="any") == "hold"
    assert agreement_decision(-0.30, -0.25, threshold=0.20, mode="both") == "lower"


def test_p18_agreement_stats_scores_only_agreed_selected_rows():
    cover_pairs = [(0.30, 1.0, 0.90, 0.80), (-0.30, -1.0, 0.90, 0.80), (0.30, -1.0, 0.90, 0.80)]
    direct_pairs = [(0.10, 1.0, 0.90, 0.80), (-0.40, -1.0, 0.90, 0.80), (-0.20, -1.0, 0.90, 0.80)]

    summary = agreement_stats(cover_pairs, direct_pairs, threshold=0.20, mode="any")

    assert summary["n"] == 2
    assert summary["coverage"] == pytest.approx(2 / 3)
    assert summary["model_side_avg_units"] == pytest.approx(1.0)
    assert summary["model_side_avg_payoff"] == pytest.approx(0.85)
    assert summary["direction_accuracy_ex_push"] == pytest.approx(1.0)


def test_p18_fixture_agreement_payload_combines_cover_and_direct_rows():
    cover_rows = [
        {"seed": 42, "bookmaker": "A", "expected_upper_units": 0.30},
        {"seed": 123, "bookmaker": "A", "expected_upper_units": 0.10},
    ]
    direct_rows = [
        {"seed": 42, "bookmaker": "A", "expected_upper_units": 0.22},
        {"seed": 123, "bookmaker": "A", "expected_upper_units": 0.25},
    ]
    combined = combine_fixture_strategy_rows(cover_rows, direct_rows, threshold=0.20, mode="any")
    payload = build_fixture_agreement_payload(
        [{"match_id": "m1", "home_team": "H", "away_team": "A", "label_status": "not_available_pre_match", "label": None}],
        combined,
        threshold=0.20,
        mode="any",
        cover_variant="p18_ahw_010",
        direct_variant="p18d_ce003_direct030",
    )

    assert combined[0]["agreement_decision"] == "upper"
    assert combined[1]["agreement_decision"] == "upper"
    assert payload["strategy"]["decision"] == "upper"
    assert payload["strategy"]["mean_cover_expected_upper_units"] == pytest.approx(0.20)
    assert payload["strategy"]["mean_direct_expected_upper_units"] == pytest.approx(0.235)
    json.dumps(payload)
