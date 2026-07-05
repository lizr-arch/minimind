import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p17_value_detection import (
    P17ProtocolError,
    aggregate_seed_summaries,
    assert_no_test_paths,
    build_p17_report_payload,
    build_threshold_summaries,
    build_value_bets,
    summarize_value_bets,
)


def test_p17_value_bets_use_top_positive_edge_and_no_vig_fair_odds():
    rows = [
        {
            "match_id": "m1",
            "y_true": 0,
            "p_home": 0.55,
            "p_draw": 0.25,
            "p_away": 0.20,
            "market_home": 0.50,
            "market_draw": 0.30,
            "market_away": 0.20,
        },
        {
            "match_id": "m2",
            "y_true": 2,
            "p_home": 0.45,
            "p_draw": 0.25,
            "p_away": 0.30,
            "market_home": 0.40,
            "market_draw": 0.30,
            "market_away": 0.30,
        },
    ]

    bets = build_value_bets(rows, edge_threshold=0.04)

    assert [bet["selected_outcome"] for bet in bets] == ["home", "home"]
    assert bets[0]["edge_abs"] == pytest.approx(0.05)
    assert bets[0]["fair_odds_no_vig"] == pytest.approx(2.0)
    assert bets[0]["net_profit_no_vig"] == pytest.approx(1.0)
    assert bets[1]["net_profit_no_vig"] == pytest.approx(-1.0)

    summary = summarize_value_bets(bets, threshold=0.04, slice_name="all", outcome="all")

    assert summary["n_bets"] == 2
    assert summary["hit_rate"] == pytest.approx(0.5)
    assert summary["roi_no_vig"] == pytest.approx(0.0)
    assert summary["mean_edge_abs"] == pytest.approx(0.05)
    assert summary["mean_model_edge_rel"] == pytest.approx(((0.55 / 0.50 - 1.0) + (0.45 / 0.40 - 1.0)) / 2.0)


def test_p17_threshold_summaries_include_flat_ah_slice_without_counting_test_data():
    rows = [
        {
            "match_id": "m1",
            "y_true": 1,
            "p_home": 0.30,
            "p_draw": 0.36,
            "p_away": 0.34,
            "market_home": 0.34,
            "market_draw": 0.31,
            "market_away": 0.35,
            "asian_line": 0.0,
            "over_under_line": 2.25,
        },
        {
            "match_id": "m2",
            "y_true": 0,
            "p_home": 0.60,
            "p_draw": 0.20,
            "p_away": 0.20,
            "market_home": 0.50,
            "market_draw": 0.25,
            "market_away": 0.25,
            "asian_line": 1.25,
            "over_under_line": 3.0,
        },
    ]

    summaries = build_threshold_summaries(rows, thresholds=[0.04], slice_names=["all", "ah_zero"])
    by_slice = {row["slice"]: row for row in summaries}

    assert by_slice["all"]["n_bets"] == 2
    assert by_slice["ah_zero"]["n_bets"] == 1
    assert by_slice["ah_zero"]["selected_draw_share"] == pytest.approx(1.0)
    with pytest.raises(P17ProtocolError, match="test split"):
        assert_no_test_paths(["data/odds_real/splits_v6/test_match_ids.txt"])


def test_p17_aggregate_seed_summaries_group_by_threshold_slice_and_outcome():
    rows = [
        {"threshold": 0.04, "slice": "all", "outcome": "all", "n_bets": 10, "roi_no_vig": 0.10, "hit_rate": 0.4},
        {"threshold": 0.04, "slice": "all", "outcome": "all", "n_bets": 12, "roi_no_vig": -0.02, "hit_rate": 0.5},
        {"threshold": 0.04, "slice": "ah_zero", "outcome": "all", "n_bets": 2, "roi_no_vig": 0.50, "hit_rate": 1.0},
    ]

    aggregate = aggregate_seed_summaries(rows)
    by_key = {(row["threshold"], row["slice"], row["outcome"]): row for row in aggregate}

    all_row = by_key[(0.04, "all", "all")]
    assert all_row["seed_count"] == 2
    assert all_row["mean_n_bets"] == pytest.approx(11.0)
    assert all_row["mean_roi_no_vig"] == pytest.approx(0.04)
    assert all_row["stdev_roi_no_vig"] == pytest.approx(math.sqrt(0.0072))

    ah_row = by_key[(0.04, "ah_zero", "all")]
    assert ah_row["seed_count"] == 1
    assert ah_row["stdev_roi_no_vig"] == pytest.approx(0.0)


def test_p17_report_payload_schema_is_diagnostic_only():
    config = {
        "mainline_id": "p16_b_mktkl_100x_old_balanced",
        "source_run_root": "runs/p16_market_replication_selection",
        "calibration_policy": {"default": "identity_no_fit"},
    }
    threshold_rows = [
        {"threshold": 0.04, "slice": "all", "outcome": "all", "seed_count": 3, "mean_n_bets": 20, "mean_roi_no_vig": 0.03}
    ]
    outcome_rows = [
        {"threshold": 0.04, "slice": "all", "outcome": "draw", "seed_count": 3, "mean_n_bets": 5, "mean_roi_no_vig": -0.10}
    ]

    payload = build_p17_report_payload(config, threshold_rows, outcome_rows, source_note="prediction_csv_proxy")

    assert payload["phase"] == "P17"
    assert payload["verdict"] in {"P17_VALUE_EDGE_CANDIDATE_DIAGNOSTIC_ONLY", "P17_NO_ROBUST_VALUE_EDGE"}
    assert payload["input_policy"]["no_test_split_loaded"] is True
    assert payload["input_policy"]["no_real_money_claim"] is True
    assert payload["mainline"]["mainline_id"] == "p16_b_mktkl_100x_old_balanced"
    assert set(payload) >= {
        "phase",
        "mainline",
        "input_policy",
        "threshold_summary",
        "outcome_summary",
        "verdict",
        "next_actions",
    }
    json.dumps(payload)
