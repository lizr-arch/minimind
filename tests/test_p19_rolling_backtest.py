import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p19_rolling_backtest import (
    P19ProtocolError,
    audit_ah_settlement,
    build_p19_verdict_payload,
    build_rolling_folds,
    fixture_collapsed_score,
    scan_feature_blacklist,
    validate_fold_no_fixture_overlap,
    write_fold_split_files,
)


def _row(match_id, kickoff, bookmaker="A", home_goals=2, away_goals=0, line=-1.25, asian_result="half_win"):
    return {
        "match_id": match_id,
        "bookmaker_id": bookmaker,
        "kickoff_time": kickoff,
        "odds_timeline": [
            {
                "minutes_before_kickoff": 60.0,
                "asian_source": "raw_update",
                "asian_line": line,
                "upper_water": 0.90,
                "lower_water": 0.95,
            }
        ],
        "label": {
            "home_goals": home_goals,
            "away_goals": away_goals,
            "asian_result": asian_result,
            "upper_side": "home",
            "euro_result": "home",
        },
    }


def test_p19_feature_blacklist_flags_target_like_inputs():
    report = scan_feature_blacklist(["euro_h", "goal_diff_after_match", "asian_settlement_code", "bucket_valid_count"])

    assert report["status"] == "FAIL"
    assert {hit["feature"] for hit in report["hits"]} == {"goal_diff_after_match", "asian_settlement_code"}


def test_p19_feature_blacklist_allows_current_market_features():
    report = scan_feature_blacklist(["euro_h", "asian_line", "upper_water", "has_asian", "bucket_valid_count"])

    assert report["status"] == "PASS"
    assert report["hits"] == []


def test_p19_ah_settlement_audit_recomputes_quarter_lines_and_detects_mismatch():
    rows = [
        _row("m1", "2024-01-01T12:00:00", home_goals=2, away_goals=0, line=-1.75, asian_result="half_win"),
        _row("m2", "2024-01-02T12:00:00", home_goals=2, away_goals=0, line=-1.75, asian_result="full_loss"),
    ]

    report = audit_ah_settlement(rows)

    assert report["checked_rows"] == 2
    assert report["mismatch_count"] == 1
    assert report["status"] == "FAIL"
    assert report["mismatches"][0]["match_id"] == "m2"
    json.dumps(report)


def test_p19_build_rolling_folds_keeps_bookmaker_rows_grouped_by_fixture():
    rows = []
    for idx in range(12):
        rows.append(_row(f"m{idx}", f"2024-01-{idx + 1:02d}T12:00:00", bookmaker="A"))
        rows.append(_row(f"m{idx}", f"2024-01-{idx + 1:02d}T12:00:00", bookmaker="B"))

    folds = build_rolling_folds(
        rows,
        n_folds=2,
        min_train_matches=4,
        selection_matches=2,
        forward_matches=2,
        embargo_days=0,
    )

    assert len(folds) == 2
    assert folds[0]["train_match_ids"] == ["m0", "m1", "m2", "m3"]
    assert folds[0]["selection_match_ids"] == ["m4", "m5"]
    assert folds[0]["forward_match_ids"] == ["m6", "m7"]
    assert folds[0]["train_rows"] == 8
    assert folds[0]["selection_rows"] == 4
    assert folds[0]["forward_rows"] == 4
    validate_fold_no_fixture_overlap(folds[0])


def test_p19_write_fold_split_files(tmp_path):
    folds = [
        {
            "fold_id": "fold_01",
            "train_match_ids": ["m1", "m2"],
            "selection_match_ids": ["m3"],
            "forward_match_ids": ["m4"],
        }
    ]

    write_fold_split_files(tmp_path, folds)

    assert (tmp_path / "fold_01" / "train_match_ids.txt").read_text(encoding="utf-8") == "m1\nm2\n"
    assert (tmp_path / "fold_01" / "selection_match_ids.txt").read_text(encoding="utf-8") == "m3\n"
    assert (tmp_path / "fold_01" / "forward_match_ids.txt").read_text(encoding="utf-8") == "m4\n"


def test_p19_fold_validation_rejects_overlap():
    fold = {
        "fold_id": "fold_bad",
        "train_match_ids": ["m1", "m2"],
        "selection_match_ids": ["m3"],
        "forward_match_ids": ["m2"],
    }

    with pytest.raises(P19ProtocolError):
        validate_fold_no_fixture_overlap(fold)


def test_p19_fixture_collapsed_score_uses_unique_fixtures_not_rows():
    rows = [
        {"selected": True, "match_id": "m1", "model_side_payoff": 1.0, "direction_hit": 1.0},
        {"selected": True, "match_id": "m1", "model_side_payoff": -1.0, "direction_hit": 0.0},
        {"selected": True, "match_id": "m2", "model_side_payoff": 0.5, "direction_hit": 1.0},
        {"selected": False, "match_id": "m3", "model_side_payoff": 0.0, "direction_hit": None},
    ]

    score = fixture_collapsed_score(rows)

    assert score["selected_rows"] == 3
    assert score["selected_unique_fixtures"] == 2
    assert score["row_avg_payoff"] == pytest.approx(1 / 6)
    assert score["fixture_avg_payoff"] == pytest.approx(0.25)
    assert score["fixture_direction_accuracy_ex_push"] == pytest.approx(0.75)


def test_p19_verdict_requires_clean_audit_and_positive_latest_fold():
    payload = build_p19_verdict_payload(
        audit_status="PASS",
        fold_scores=[
            {"fold_id": "fold_01", "fixture_avg_payoff": 0.20, "selected_unique_fixtures": 100},
            {"fold_id": "fold_02", "fixture_avg_payoff": -0.01, "selected_unique_fixtures": 100},
        ],
        min_fixture_payoff=0.05,
        min_positive_fold_rate=0.70,
    )

    assert payload["verdict"] == "P19_ROLLING_BACKTEST_FAIL_LATEST_DECAY"
