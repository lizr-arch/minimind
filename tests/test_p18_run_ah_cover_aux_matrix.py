import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p18_run_ah_cover_aux_matrix import (
    FIXED_P18_SEEDS,
    FIXED_P18_VARIANTS,
    build_candidate_decision,
    build_report_row,
    compute_ah_head_direction_summary,
    summarize_candidate_rows,
)


def _report(logloss=0.9372, market_kl=0.006, market_draw_mae=0.036, ah_acc=0.44, ah_nll=1.45):
    return {
        "val_metrics": {"logloss": logloss, "ece": 0.027, "draw_top2_recall": 0.39},
        "market_replication_metrics": {"market_kl": market_kl, "market_draw_mae": market_draw_mae},
        "ah_cover_metrics": {"ah_cover_acc": ah_acc, "ah_cover_nll": ah_nll, "ah_cover_n": 4718},
        "selected_checkpoint_metrics": {"epoch": 4},
        "config": {"ah_cover_loss_weight": 0.1},
    }


def test_p18_matrix_registry_is_fixed_and_small():
    assert FIXED_P18_SEEDS == [42, 123, 2025]
    assert list(FIXED_P18_VARIANTS) == [
        "p18_ahw_003",
        "p18_ahw_010",
        "p18_ahw_030",
        "p18u_unit030",
        "p18u_side010_unit020",
        "p18u_ce003_side010_unit020",
        "p18d_direct030",
        "p18d_ce003_direct030",
        "p18d_ce003_direct015",
        "p18d_ce003_direct060",
        "p18d_ce010_direct030",
    ]
    assert [FIXED_P18_VARIANTS[name]["ah_cover_loss_weight"] for name in list(FIXED_P18_VARIANTS)[:3]] == [0.03, 0.10, 0.30]
    assert FIXED_P18_VARIANTS["p18u_side010_unit020"]["ah_side_loss_weight"] == pytest.approx(0.10)


def test_p18_report_row_extracts_1x2_and_ah_guardrail_metrics():
    row = build_report_row("p18_ahw_010", 42, _report(), {"model_side_avg_units": 0.12, "direction_accuracy_ex_push": 0.57})

    assert row["variant"] == "p18_ahw_010"
    assert row["seed"] == 42
    assert row["val_logloss"] == pytest.approx(0.9372)
    assert row["market_kl"] == pytest.approx(0.006)
    assert row["ah_cover_acc"] == pytest.approx(0.44)
    assert row["ah_head_model_side_avg_units"] == pytest.approx(0.12)


def test_p18_candidate_summary_and_decision_requires_guardrails():
    baseline = {
        "mean_val_logloss": 0.93704,
        "mean_market_draw_mae": 0.0362,
        "mean_ah_cover_acc": 0.0,
        "mean_ah_head_model_side_avg_units": 0.10,
        "mean_direction_accuracy_ex_push": 0.55,
    }
    rows = [
        build_report_row("p18_ahw_010", 42, _report(logloss=0.9371, ah_acc=0.45), {"model_side_avg_units": 0.13, "direction_accuracy_ex_push": 0.58}),
        build_report_row("p18_ahw_010", 123, _report(logloss=0.9372, ah_acc=0.46), {"model_side_avg_units": 0.12, "direction_accuracy_ex_push": 0.57}),
        build_report_row("p18_ahw_010", 2025, _report(logloss=0.9370, ah_acc=0.44), {"model_side_avg_units": 0.11, "direction_accuracy_ex_push": 0.56}),
    ]

    summary = summarize_candidate_rows(rows)[0]
    decision = build_candidate_decision(summary, baseline)

    assert summary["seed_count"] == 3
    assert summary["mean_ah_cover_acc"] == pytest.approx(0.45)
    assert decision["verdict"] == "P18_DIRECTIONAL_GAIN_CANDIDATE"

    bad = dict(summary)
    bad["mean_val_logloss"] = 0.9400
    blocked = build_candidate_decision(bad, baseline)
    assert blocked["verdict"] == "P18_REJECT_GUARDRAIL"
    assert "P18_FAIL_1X2_LOGLOSS_DRIFT" in blocked["failed_gates"]
    json.dumps(decision)

    no_gain = dict(summary)
    no_gain["mean_ah_head_model_side_avg_units"] = 0.08
    learned_but_not_better = build_candidate_decision(no_gain, baseline)
    assert learned_but_not_better["verdict"] == "P18_LABEL_SIGNAL_NO_DIRECTIONAL_GAIN"


def test_p18_ah_head_direction_summary_converts_cover_probs_to_model_side_units():
    pred_rows = [
        {"match_id": "m1", "p_upper_full_win": "0.70", "p_upper_half_win": "0.10", "p_push": "0.10", "p_upper_half_loss": "0.05", "p_upper_full_loss": "0.05"},
        {"match_id": "m2", "p_upper_full_win": "0.05", "p_upper_half_win": "0.05", "p_push": "0.10", "p_upper_half_loss": "0.10", "p_upper_full_loss": "0.70"},
        {"match_id": "m3", "p_upper_full_win": "0.40", "p_upper_half_win": "0.20", "p_push": "0.20", "p_upper_half_loss": "0.10", "p_upper_full_loss": "0.10"},
    ]
    context_rows = [
        {"match_id": "m1", "actual_upper_units": 1.0, "asian_line": -0.25, "slice": "favorite_le_0_50"},
        {"match_id": "m2", "actual_upper_units": -1.0, "asian_line": 0.75, "slice": "favorite_0_75_to_1_50"},
        {"match_id": "m3", "actual_upper_units": 0.0, "asian_line": 0.0, "slice": "flat"},
    ]

    summary = compute_ah_head_direction_summary(pred_rows, context_rows)

    assert summary["n"] == 3
    assert summary["model_side_avg_units"] == pytest.approx(2.0 / 3.0)
    assert summary["direction_accuracy_ex_push"] == pytest.approx(1.0)
    assert summary["upper_pick_rate"] == pytest.approx(2.0 / 3.0)
    assert summary["lower_pick_rate"] == pytest.approx(1.0 / 3.0)


def test_p18_ah_head_direction_summary_prefers_label_idx_over_duplicate_match_id():
    pred_rows = [
        {"match_id": "dup", "p_upper_full_win": "0.90", "p_upper_half_win": "0.00", "p_push": "0.00", "p_upper_half_loss": "0.00", "p_upper_full_loss": "0.10"},
        {"match_id": "dup", "p_upper_full_win": "0.10", "p_upper_half_win": "0.00", "p_push": "0.00", "p_upper_half_loss": "0.00", "p_upper_full_loss": "0.90"},
    ]
    context_rows = [
        {"label_idx": 0, "match_id": "dup", "actual_upper_units": 1.0, "asian_line": -0.25, "slice": "favorite_le_0_50"},
        {"label_idx": 1, "match_id": "dup", "actual_upper_units": -1.0, "asian_line": 0.25, "slice": "favorite_le_0_50"},
    ]

    summary = compute_ah_head_direction_summary(pred_rows, context_rows)

    assert summary["n"] == 2
    assert summary["model_side_avg_units"] == pytest.approx(1.0)
    assert summary["direction_accuracy_ex_push"] == pytest.approx(1.0)


def test_p18_ah_head_direction_summary_prefers_direct_expected_units_when_present():
    pred_rows = [
        {
            "match_id": "m1",
            "direct_expected_upper_units": "-0.50",
            "p_upper_full_win": "0.90",
            "p_upper_half_win": "0.00",
            "p_push": "0.00",
            "p_upper_half_loss": "0.00",
            "p_upper_full_loss": "0.10",
        }
    ]
    context_rows = [
        {"label_idx": 0, "match_id": "m1", "actual_upper_units": -1.0, "asian_line": 0.25, "slice": "favorite_le_0_50"},
    ]

    summary = compute_ah_head_direction_summary(pred_rows, context_rows)

    assert summary["model_side_avg_units"] == pytest.approx(1.0)
    assert summary["lower_pick_rate"] == pytest.approx(1.0)
