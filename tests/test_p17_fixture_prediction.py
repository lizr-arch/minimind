import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.odds_patch_itransformer_v2 import N_AGG, N_BUCKETS
from tools.p14_train_market_replication import p14_selected_feature_names
from tools.p17_predict_fixture import (
    P17FixtureProtocolError,
    assert_no_test_paths,
    assert_prediction_rows_are_pre_match,
    build_ensemble_report_payload,
    build_fixture_dataset,
)


def _sample_row(bookmaker_id: str = "Bet365") -> dict:
    return {
        "schema_version": "prediction_export_v1",
        "match_id": "world_cup_2907390",
        "bookmaker_id": bookmaker_id,
        "home_team": "Argentina",
        "away_team": "Cape Verde",
        "kickoff_time_utc": "2026-07-03T22:00:00+00:00",
        "label_status": "not_available_pre_match",
        "label": None,
        "odds_timeline": [
            {
                "minutes_before_kickoff": 90.0,
                "euro_h": 1.50,
                "euro_d": 4.0,
                "euro_a": 7.0,
                "has_euro": True,
                "asian_line": -1.5,
                "upper_water": 0.90,
                "lower_water": 0.95,
                "asian_source": "raw_update",
                "over_under_line": 3.0,
                "over_water": 0.95,
                "under_water": 0.90,
                "over_under_source": "raw_update",
            },
            {
                "minutes_before_kickoff": 30.0,
                "euro_h": 2.0,
                "euro_d": 4.0,
                "euro_a": 4.0,
                "has_euro": True,
                "asian_line": -0.25,
                "upper_water": 1.00,
                "lower_water": 0.85,
                "asian_source": "raw_update",
                "over_under_line": 2.5,
                "over_water": 0.90,
                "under_water": 0.95,
                "over_under_source": "raw_update",
            },
        ],
    }


def test_p17_fixture_rejects_test_paths_and_labelled_rows():
    with pytest.raises(P17FixtureProtocolError, match="test split"):
        assert_no_test_paths(["data/splits/test_match_ids.txt"])

    labelled = _sample_row()
    labelled["label"] = {"euro_result": "home"}
    with pytest.raises(P17FixtureProtocolError, match="pre-match"):
        assert_prediction_rows_are_pre_match([labelled])


def test_p17_fixture_dataset_uses_unlabelled_rows_and_latest_nonnegative_anchor():
    rows = [_sample_row()]
    feature_names = p14_selected_feature_names("euro,asian,ou")

    dataset = build_fixture_dataset(rows, feature_names)

    assert tuple(dataset["X"].shape[1:]) == (N_BUCKETS, len(feature_names), N_AGG)
    assert dataset["match_ids"] == ["world_cup_2907390"]
    assert dataset["bookmakers"] == ["Bet365"]
    assert torch.allclose(dataset["p_market"][0], torch.tensor([0.50, 0.25, 0.25]), atol=1e-6)
    assert dataset["anchor_diagnostics"][0]["anchor_minutes_before_kickoff"] == pytest.approx(30.0)


def test_p17_fixture_report_payload_ensembles_seed_predictions_and_edges():
    rows = [_sample_row("Bet365"), _sample_row("Pinnacle")]
    seed_outputs = [
        {
            "seed": 42,
            "rows": [
                {"bookmaker": "Bet365", "p_home": 0.70, "p_draw": 0.20, "p_away": 0.10, "market_home": 0.60, "market_draw": 0.25, "market_away": 0.15},
                {"bookmaker": "Pinnacle", "p_home": 0.72, "p_draw": 0.18, "p_away": 0.10, "market_home": 0.62, "market_draw": 0.23, "market_away": 0.15},
            ],
        },
        {
            "seed": 123,
            "rows": [
                {"bookmaker": "Bet365", "p_home": 0.66, "p_draw": 0.22, "p_away": 0.12, "market_home": 0.60, "market_draw": 0.25, "market_away": 0.15},
                {"bookmaker": "Pinnacle", "p_home": 0.68, "p_draw": 0.20, "p_away": 0.12, "market_home": 0.62, "market_draw": 0.23, "market_away": 0.15},
            ],
        },
    ]

    payload = build_ensemble_report_payload(rows, seed_outputs, run_source="unit", checkpoint_note="balanced")

    assert payload["fixture"]["match_id"] == "world_cup_2907390"
    assert payload["checkpoint_note"] == "balanced"
    assert payload["ensemble"]["p_home"] == pytest.approx(0.69)
    assert payload["ensemble"]["p_draw"] == pytest.approx(0.20)
    assert payload["ensemble"]["p_away"] == pytest.approx(0.11)
    assert payload["ensemble"]["top_model_outcome"] == "home"
    assert payload["ensemble"]["top_edge_outcome"] == "home"
    assert payload["ensemble"]["edge_home"] == pytest.approx(0.08)
    json.dumps(payload)
