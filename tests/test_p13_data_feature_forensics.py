import csv
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p13_data_feature_forensics import (
    P13ProtocolError,
    ah_abs_line_bucket,
    assert_no_test_paths,
    build_context_rows,
    calibration_bins,
    choose_p13_verdict,
    draw_margin_to_top,
    draw_shrinkage_ratio,
    draw_top2,
    fair_probs_from_odds,
    load_prediction_csv,
    mark_slice_support,
    ou_line_bucket,
    reproduce_prediction_metrics,
    validate_alignment,
    write_csv,
)


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["match_id", "y_true", "p_home", "p_draw", "p_away", "pred_class", "correct"])
        writer.writeheader()
        writer.writerows(rows)


def test_fair_probs_from_odds_normalizes_and_rejects_invalid_odds():
    probs = fair_probs_from_odds(2.0, 4.0, 4.0)

    assert probs == pytest.approx((0.5, 0.25, 0.25))
    with pytest.raises(ValueError, match="invalid odds"):
        fair_probs_from_odds(1.0, 4.0, 4.0)


def test_draw_top2_and_margin_math():
    assert draw_top2((0.50, 0.30, 0.20)) is True
    assert draw_top2((0.50, 0.19, 0.31)) is False
    assert draw_margin_to_top((0.50, 0.30, 0.20)) == pytest.approx(0.20)
    assert draw_margin_to_top((0.20, 0.40, 0.40)) == pytest.approx(0.0)


def test_draw_shrinkage_ratio_uses_true_draw_rows_only():
    labels = [1, 0, 1]
    model = [0.20, 0.30, 0.10]
    market = [0.25, 0.30, 0.20]

    assert draw_shrinkage_ratio(model, market, labels) == pytest.approx(0.15 / 0.225)


def test_calibration_bins_reports_low_n_and_gap():
    rows = [
        {"confidence": 0.10, "target": 0},
        {"confidence": 0.20, "target": 0},
        {"confidence": 0.80, "target": 1},
        {"confidence": 0.90, "target": 1},
    ]

    bins = calibration_bins(rows, n_bins=2)

    assert len(bins) == 2
    assert bins[0]["n"] == 2
    assert bins[0]["pred_mean"] == pytest.approx(0.15)
    assert bins[0]["true_rate"] == pytest.approx(0.0)
    assert bins[0]["support"] == "report_only"
    assert bins[1]["calibration_gap"] == pytest.approx(0.15)


def test_ah_and_ou_bucket_boundaries():
    assert ah_abs_line_bucket(0.0) == "ah_abs_0"
    assert ah_abs_line_bucket(-0.25) == "ah_abs_le_0_25"
    assert ah_abs_line_bucket(0.5) == "ah_abs_le_0_50"
    assert ah_abs_line_bucket(0.75) == "ah_abs_0_75_1_00"
    assert ah_abs_line_bucket(1.5) == "ah_abs_ge_1_25"
    assert ou_line_bucket(2.0) == "ou_le_2_00"
    assert ou_line_bucket(2.25) == "ou_2_25"
    assert ou_line_bucket(2.5) == "ou_2_50"
    assert ou_line_bucket(3.25) == "ou_ge_3_00"


def test_mark_slice_support_uses_fixed_thresholds():
    assert mark_slice_support(99) == "report_only"
    assert mark_slice_support(100) == "diagnostic_grade"
    assert mark_slice_support(300) == "decision_grade"


def test_write_csv_keeps_header_for_empty_rows(tmp_path):
    out = tmp_path / "empty.csv"

    write_csv(out, [], fieldnames=["match_id", "issue_type"])

    assert out.read_text(encoding="utf-8").strip() == "match_id,issue_type"


def test_alignment_detects_match_id_and_label_mismatch(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_predictions(
        a,
        [
            {"match_id": "m1", "y_true": 0, "p_home": 0.5, "p_draw": 0.2, "p_away": 0.3, "pred_class": 0, "correct": 1},
            {"match_id": "m2", "y_true": 1, "p_home": 0.2, "p_draw": 0.4, "p_away": 0.4, "pred_class": 1, "correct": 1},
        ],
    )
    _write_predictions(
        b,
        [
            {"match_id": "m1", "y_true": 0, "p_home": 0.4, "p_draw": 0.3, "p_away": 0.3, "pred_class": 0, "correct": 1},
            {"match_id": "m3", "y_true": 1, "p_home": 0.2, "p_draw": 0.4, "p_away": 0.4, "pred_class": 1, "correct": 1},
        ],
    )

    with pytest.raises(ValueError, match="match_id alignment"):
        validate_alignment({"a": load_prediction_csv(a), "b": load_prediction_csv(b)})


def test_alignment_reports_duplicate_prediction_rows(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    rows = [
        {"match_id": "m1", "y_true": 0, "p_home": 0.5, "p_draw": 0.2, "p_away": 0.3, "pred_class": 0, "correct": 1},
        {"match_id": "m1", "y_true": 0, "p_home": 0.4, "p_draw": 0.3, "p_away": 0.3, "pred_class": 0, "correct": 1},
        {"match_id": "m2", "y_true": 1, "p_home": 0.2, "p_draw": 0.4, "p_away": 0.4, "pred_class": 1, "correct": 1},
    ]
    _write_predictions(a, rows)
    _write_predictions(b, rows)

    alignment = validate_alignment({"a": load_prediction_csv(a), "b": load_prediction_csv(b)})

    assert alignment["row_count"] == 3
    assert alignment["unique_match_ids"] == 2
    assert alignment["duplicate_prediction_rows"] == 1


def test_reproduce_prediction_metrics_matches_expected_values(tmp_path):
    p = tmp_path / "pred.csv"
    _write_predictions(
        p,
        [
            {"match_id": "m1", "y_true": 0, "p_home": 0.8, "p_draw": 0.1, "p_away": 0.1, "pred_class": 0, "correct": 1},
            {"match_id": "m2", "y_true": 1, "p_home": 0.2, "p_draw": 0.3, "p_away": 0.5, "pred_class": 2, "correct": 0},
        ],
    )

    metrics = reproduce_prediction_metrics(load_prediction_csv(p))

    assert metrics["draw_top2"] == pytest.approx(1.0)
    assert metrics["mean_p_draw_true_draw"] == pytest.approx(0.3)
    assert metrics["logloss"] == pytest.approx(-(__import__("math").log(0.8) + __import__("math").log(0.3)) / 2)


def test_test_split_paths_are_rejected():
    with pytest.raises(P13ProtocolError, match="test split"):
        assert_no_test_paths(["data/odds_real/splits_v6/test_match_ids.txt"])
    with pytest.raises(P13ProtocolError, match="test split"):
        assert_no_test_paths(["data/odds_real/test/foo.jsonl"])


def test_decision_matrix_prefers_data_repair_then_market_replication():
    assert choose_p13_verdict({"data_repair_required": True}) == "P13_DATA_REPAIR_REQUIRED"
    assert (
        choose_p13_verdict(
            {
                "data_repair_required": False,
                "market_replication_recommended": True,
                "feature_engineering_recommended": True,
            }
        )
        == "P13_TARGET_PIVOT_MARKET_REPLICATION_RECOMMENDED"
    )
    assert choose_p13_verdict({"feature_engineering_recommended": True}) == "P13_FEATURE_ENGINEERING_RECOMMENDED"


def test_context_rows_preserve_bookmaker_grain_for_duplicate_match_ids():
    preds = [
        {"match_id": "m1", "y_true": 0, "p": (0.5, 0.2, 0.3)},
        {"match_id": "m1", "y_true": 0, "p": (0.4, 0.3, 0.3)},
    ]
    val_rows = [
        {
            "match_id": "m1",
            "bookmaker_id": "Bet365",
            "label": {"euro_result": "home", "home_goals": 2, "away_goals": 1},
            "odds_timeline": [{"minutes_before_kickoff": 10, "euro_h": 2.0, "euro_d": 4.0, "euro_a": 4.0}],
        },
        {
            "match_id": "m1",
            "bookmaker_id": "Pinnacle",
            "label": {"euro_result": "home", "home_goals": 2, "away_goals": 1},
            "odds_timeline": [{"minutes_before_kickoff": 10, "euro_h": 3.0, "euro_d": 3.0, "euro_a": 3.0}],
        },
    ]
    p11_score = [
        {"match_id": "m1", "score": (0.6, 0.2, 0.2)},
        {"match_id": "m1", "score": (0.2, 0.4, 0.4)},
    ]
    p12_prior = [
        {"match_id": "m1", "prior": (0.6, 0.2, 0.2), "final": (0.5, 0.3, 0.2)},
        {"match_id": "m1", "prior": (1 / 3, 1 / 3, 1 / 3), "final": (0.2, 0.4, 0.4)},
    ]

    context, suspects = build_context_rows(preds, val_rows, p11_score, p12_prior)

    assert suspects == []
    assert [row["bookmaker_id"] for row in context] == ["Bet365", "Pinnacle"]
    assert context[0]["market_p_draw"] == pytest.approx(0.25)
    assert context[1]["market_p_draw"] == pytest.approx(1 / 3)
    assert context[0]["p11_score_p_draw"] == pytest.approx(0.2)
    assert context[1]["p11_score_p_draw"] == pytest.approx(0.4)


def test_context_rows_flags_source_prediction_label_mismatch():
    preds = [{"match_id": "m1", "y_true": 1, "p": (0.3, 0.4, 0.3)}]
    val_rows = [
        {
            "match_id": "m1",
            "bookmaker_id": "Bet365",
            "label": {"euro_result": "home", "home_goals": 2, "away_goals": 1},
            "odds_timeline": [{"minutes_before_kickoff": 10, "euro_h": 2.0, "euro_d": 4.0, "euro_a": 4.0}],
        }
    ]

    _, suspects = build_context_rows(preds, val_rows, [], [])

    assert [row["issue_type"] for row in suspects] == ["label_prediction_mismatch"]
