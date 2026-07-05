import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p18_ah_forward_selection import (
    assign_forward_periods,
    build_forward_selection_payload,
    score_candidate_rows,
    select_best_candidate,
)


def test_p18_forward_selection_assigns_chronological_periods():
    rows = [
        {"label_idx": 0, "kickoff_time": "2024-03-01"},
        {"label_idx": 1, "kickoff_time": "2024-01-01"},
        {"label_idx": 2, "kickoff_time": "2024-04-01"},
        {"label_idx": 3, "kickoff_time": "2024-02-01"},
    ]

    period_by_idx = assign_forward_periods(rows, train_fraction=0.5)

    assert period_by_idx == {1: "selection", 3: "selection", 0: "forward", 2: "forward"}


def test_p18_forward_selection_scores_selected_rows_only():
    rows = [
        {"candidate_id": "a", "selected": True, "model_side_payoff": 0.90, "model_side_units": 1.0, "direction_hit": 1.0},
        {"candidate_id": "a", "selected": True, "model_side_payoff": -1.0, "model_side_units": -1.0, "direction_hit": 0.0},
        {"candidate_id": "a", "selected": False, "model_side_payoff": 0.0, "model_side_units": 0.0, "direction_hit": None},
    ]

    score = score_candidate_rows(rows, min_selected=1)

    assert score["total_n"] == 3
    assert score["selected_n"] == 2
    assert score["coverage"] == pytest.approx(2 / 3)
    assert score["model_side_avg_payoff"] == pytest.approx(-0.05)
    assert score["direction_accuracy_ex_push"] == pytest.approx(0.5)
    assert score["eligible"] is True


def test_p18_forward_selection_picks_by_selection_and_reports_forward():
    candidates = [
        {"candidate_id": "a", "period": "selection", "selected": True, "model_side_payoff": 0.20, "model_side_units": 0.2, "direction_hit": 1.0},
        {"candidate_id": "a", "period": "forward", "selected": True, "model_side_payoff": -1.0, "model_side_units": -1.0, "direction_hit": 0.0},
        {"candidate_id": "b", "period": "selection", "selected": True, "model_side_payoff": 0.40, "model_side_units": 0.4, "direction_hit": 1.0},
        {"candidate_id": "b", "period": "forward", "selected": True, "model_side_payoff": 0.70, "model_side_units": 0.7, "direction_hit": 1.0},
    ]

    best = select_best_candidate(candidates, min_selected=1)
    payload = build_forward_selection_payload(candidates, best, min_selected=1)

    assert best["candidate_id"] == "b"
    assert payload["selected_candidate"]["candidate_id"] == "b"
    assert payload["forward_score"]["model_side_avg_payoff"] == pytest.approx(0.70)
    assert payload["verdict"] == "P18_FORWARD_SELECTION_PASS"
    json.dumps(payload)
