import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p16_run_market_replication_selection import (
    FIXED_P16_CONTROL,
    FIXED_P16_SEEDS,
    FIXED_P16_VARIANTS,
    P16_ARTIFACTS,
    P16ProtocolError,
    build_calibration_diagnostic,
    build_calibration_diagnostics_for_predictions,
    build_promotion_decision_matrix,
    build_protocol_audit,
    compute_prediction_metrics,
    compute_slice_stability_rows,
    evaluate_slice_gate,
    parse_registered_seeds,
    parse_registered_variants,
    select_checkpoint_for_rule,
    write_reviewer_report,
)


def test_p16_registry_is_fixed_to_market_kl_grid_and_control():
    assert FIXED_P16_CONTROL == "p16_control_no_market_kl"
    assert FIXED_P16_SEEDS == [42, 123, 2025]
    assert list(FIXED_P16_VARIANTS) == [
        "p16_a_mktkl_050x",
        "p16_b_mktkl_100x",
        "p16_c_mktkl_150x",
        "p16_d_mktkl_200x",
    ]
    assert [FIXED_P16_VARIANTS[name]["market_loss_weight"] for name in FIXED_P16_VARIANTS] == [
        0.15,
        0.30,
        0.45,
        0.60,
    ]


def test_p16_runner_rejects_unregistered_variant_seed_and_test_split():
    with pytest.raises(ValueError, match="Unregistered P16 variant"):
        parse_registered_variants("p16_a_mktkl_050x,p16_x")
    with pytest.raises(ValueError, match="Unregistered P16 seed"):
        parse_registered_seeds("42,7")
    with pytest.raises(P16ProtocolError, match="test split"):
        build_protocol_audit("train_match_ids.txt", "test_match_ids.txt", list(FIXED_P16_VARIANTS), FIXED_P16_SEEDS)

    audit = build_protocol_audit("train_match_ids.txt", "val_match_ids.txt", list(FIXED_P16_VARIANTS), FIXED_P16_SEEDS)
    assert audit["expected_total_runs"] == 15
    assert audit["hard_fail_reasons"] == []
    assert audit["no_new_model_family"] is True
    assert audit["control_name"] == FIXED_P16_CONTROL


def test_p16_checkpoint_selection_rules_are_deterministic():
    history = [
        {
            "epoch": 1,
            "val_logloss": 0.9365,
            "val_ece": 0.030,
            "val_draw_top2": 0.350,
            "market_kl": 0.006,
            "market_draw_mae": 0.050,
        },
        {
            "epoch": 2,
            "val_logloss": 0.9372,
            "val_ece": 0.031,
            "val_draw_top2": 0.410,
            "market_kl": 0.004,
            "market_draw_mae": 0.030,
        },
        {
            "epoch": 3,
            "val_logloss": 0.9377,
            "val_ece": 0.031,
            "val_draw_top2": 0.440,
            "market_kl": 0.003,
            "market_draw_mae": 0.020,
        },
    ]
    control = {"val_logloss": 0.9370, "val_ece": 0.028}

    assert select_checkpoint_for_rule(history, "best_logloss", control)["epoch"] == 1
    assert select_checkpoint_for_rule(history, "old_balanced", control)["epoch"] == 2
    assert select_checkpoint_for_rule(history, "balanced_v2_relative", control)["epoch"] == 2
    assert select_checkpoint_for_rule(history, "draw_safeguard", control)["epoch"] == 3
    assert select_checkpoint_for_rule(history, "draw_safeguard", control)["diagnostic_only"] is True


def test_p16_promotion_matrix_blocks_draw_safeguard_and_promotes_balanced_v2():
    control = {
        "mean_val_logloss": 0.9370,
        "mean_ece": 0.028,
        "draw_top2": 0.345,
        "mean_p_draw_true_draw": 0.198,
        "market_draw_mae": 0.053,
    }
    candidate = {
        "variant": "p16_b_mktkl_100x",
        "selection_rule": "balanced_v2_relative",
        "mean_val_logloss": 0.9371,
        "mean_ece": 0.031,
        "draw_top2": 0.382,
        "mean_p_draw_true_draw": 0.214,
        "market_draw_mae": 0.040,
        "model_minus_market_p_draw": -0.036,
        "max_seed_logloss_delta": 0.0004,
        "max_seed_ece": 0.034,
        "draw_top2_stdev": 0.020,
        "mean_p_draw_true_draw_stdev": 0.003,
        "slice_gate_pass": True,
    }

    promoted = build_promotion_decision_matrix(candidate, control)

    assert promoted["verdict"] == "P16_PROMOTE_P14_MARKET_REPLICATION_MAINLINE"
    blocked = build_promotion_decision_matrix({**candidate, "selection_rule": "draw_safeguard"}, control)
    assert blocked["verdict"] == "P16_RETAIN_P6_MAINLINE"
    assert "P16_FAIL_DIAGNOSTIC_RULE_NOT_PROMOTABLE" in blocked["failed_gates"]


def test_p16_calibration_diagnostic_tags_val_fitted_as_not_promotable():
    val_fit = build_calibration_diagnostic("scalar_temperature", fit_split="val", logloss=0.93, ece=0.02)
    train_fit = build_calibration_diagnostic("scalar_temperature", fit_split="train_cal", logloss=0.94, ece=0.025)

    assert val_fit["tag"] == "DIAGNOSTIC_ONLY_NOT_PROMOTABLE"
    assert val_fit["promotable"] is False
    assert train_fit["promotable"] is True


def test_p16_prediction_metrics_compute_draw_and_calibration_values():
    rows = [
        {"y_true": 1, "p_home": 0.20, "p_draw": 0.50, "p_away": 0.30},
        {"y_true": 0, "p_home": 0.60, "p_draw": 0.20, "p_away": 0.20},
    ]

    metrics = compute_prediction_metrics(rows)

    assert metrics["n"] == 2
    assert metrics["logloss"] == pytest.approx(-0.5 * (math.log(0.50) + math.log(0.60)))
    assert metrics["draw_top2"] == pytest.approx(1.0)
    assert metrics["mean_p_draw_true_draw"] == pytest.approx(0.50)
    assert 0.0 <= metrics["ece"] <= 1.0


def test_p16_slice_stability_uses_real_ah_ou_market_slices_and_gate():
    context_rows = [
        {"asian_line": 0.25, "over_under_line": 2.25},
        {"asian_line": 0.50, "over_under_line": 2.00},
        {"asian_line": 1.50, "over_under_line": 3.25},
    ]
    control_rows = [
        {"y_true": 1, "p_home": 0.45, "p_draw": 0.20, "p_away": 0.35, "market_home": 0.30, "market_draw": 0.40, "market_away": 0.30},
        {"y_true": 1, "p_home": 0.40, "p_draw": 0.25, "p_away": 0.35, "market_home": 0.25, "market_draw": 0.45, "market_away": 0.30},
        {"y_true": 0, "p_home": 0.70, "p_draw": 0.15, "p_away": 0.15, "market_home": 0.65, "market_draw": 0.20, "market_away": 0.15},
    ]
    candidate_rows = [
        {"y_true": 1, "p_home": 0.35, "p_draw": 0.34, "p_away": 0.31, "market_home": 0.30, "market_draw": 0.40, "market_away": 0.30},
        {"y_true": 1, "p_home": 0.34, "p_draw": 0.33, "p_away": 0.33, "market_home": 0.25, "market_draw": 0.45, "market_away": 0.30},
        {"y_true": 0, "p_home": 0.68, "p_draw": 0.16, "p_away": 0.16, "market_home": 0.65, "market_draw": 0.20, "market_away": 0.15},
    ]

    rows = compute_slice_stability_rows(
        context_rows,
        control_rows,
        candidate_rows,
        variant="p16_b_mktkl_100x",
        selection_rule="balanced_v2_relative",
        min_decision_n=2,
    )
    by_slice = {row["slice"]: row for row in rows}

    assert by_slice["ah_abs_le_0_50"]["n"] == 2
    assert by_slice["ah_abs_le_0_50"]["support"] == "decision_grade"
    assert by_slice["ah_abs_le_0_50"]["candidate_draw_top2"] > by_slice["ah_abs_le_0_50"]["control_draw_top2"]
    assert by_slice["ah_abs_le_0_50"]["market_mean_p_draw"] == pytest.approx(0.425)
    assert evaluate_slice_gate(rows)["slice_gate_pass"] is True


def test_p16_slice_gate_fails_when_decision_slice_only_worsens_logloss():
    rows = [
        {
            "slice": "ah_abs_le_0_50",
            "support": "decision_grade",
            "candidate_logloss": 0.95,
            "control_logloss": 0.90,
            "candidate_draw_top2": 0.40,
            "control_draw_top2": 0.40,
            "candidate_mean_p_draw_true_draw": 0.20,
            "control_mean_p_draw_true_draw": 0.20,
        }
    ]

    gate = evaluate_slice_gate(rows)

    assert gate["slice_gate_pass"] is False
    assert gate["failed_slices"] == ["ah_abs_le_0_50"]


def test_p16_calibration_diagnostics_are_computed_and_val_fit_is_not_promotable():
    rows = [
        {"y_true": 0, "p_home": 0.90, "p_draw": 0.05, "p_away": 0.05},
        {"y_true": 1, "p_home": 0.90, "p_draw": 0.05, "p_away": 0.05},
        {"y_true": 2, "p_home": 0.05, "p_draw": 0.05, "p_away": 0.90},
    ]

    diagnostics = build_calibration_diagnostics_for_predictions(rows)
    by_method = {row["method"]: row for row in diagnostics}

    assert by_method["identity_no_fit"]["fit_split"] == "none"
    assert by_method["scalar_temperature_val_grid"]["tag"] == "DIAGNOSTIC_ONLY_NOT_PROMOTABLE"
    assert by_method["scalar_temperature_val_grid"]["promotable"] is False
    assert by_method["scalar_temperature_val_grid"]["temperature"] != pytest.approx(1.0)


def test_p16_artifact_schema_lists_required_report_outputs(tmp_path):
    expected = {
        "p16_inputs_manifest.json",
        "p16_control_summary.json",
        "p16_weight_grid_summary.csv",
        "p16_checkpoint_selection_ablation.csv",
        "p16_selected_checkpoint_summary.json",
        "p16_calibration_diagnostic.json",
        "p16_slice_stability.csv",
        "p16_market_replication_metrics.json",
        "p16_draw_metrics.json",
        "p16_promotion_decision_matrix.json",
        "p16_report.md",
        "reviewer_report.json",
        "reviewer_report.md",
    }
    assert set(P16_ARTIFACTS) == expected
    path = tmp_path / "artifacts.json"
    path.write_text(json.dumps({"artifacts": P16_ARTIFACTS}), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["artifacts"] == P16_ARTIFACTS


def test_p16_reviewer_does_not_treat_retain_p6_as_draw_safeguard_promotion(tmp_path):
    aggregate = {
        "promotion_decisions": [
            {
                "selection_rule": "draw_safeguard",
                "verdict": "P16_RETAIN_P6_MAINLINE",
            }
        ]
    }
    audit = {"hard_fail_reasons": []}

    write_reviewer_report(tmp_path, audit, aggregate)

    payload = json.loads((tmp_path / "reviewer_report.json").read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["findings"] == []
