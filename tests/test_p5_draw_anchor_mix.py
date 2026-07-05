"""Narrow tests for P5 segmented draw-anchor mix calibration."""

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p5_draw_anchor_mix import (  # noqa: E402
    apply_anchor_mix,
    assign_train_inner_splits,
    build_candidate_grid,
    evaluate_locked_params,
    select_anchor_mix,
    stable_bucket,
    validate_eval_request,
    validate_fit_request,
)


def _row(match_id, y_true=1, p_draw=0.20, p_anchor_draw=0.30, p_home=0.50, p_away=0.30):
    anchor_remaining = 1.0 - p_anchor_draw
    return {
        "match_id": match_id,
        "y_true": y_true,
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_anchor_home": anchor_remaining * 0.60,
        "p_anchor_draw": p_anchor_draw,
        "p_anchor_away": anchor_remaining * 0.40,
        "p_home_from_diff": p_home,
        "p_draw_from_diff": p_draw,
        "p_away_from_diff": p_away,
    }


def test_train_inner_split_is_deterministic():
    rows = [_row(f"m{i}") for i in range(20)]

    first = assign_train_inner_splits(rows)
    second = assign_train_inner_splits(rows)

    assert [row["inner_split"] for row in first] == [row["inner_split"] for row in second]
    assert stable_bucket("same-id") == stable_bucket("same-id")


def test_cli_help_runs_as_script():
    result = subprocess.run(
        [sys.executable, "tools/p5_draw_anchor_mix.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--predictions-csv" in result.stdout


def test_train_inner_split_has_no_overlap():
    rows = assign_train_inner_splits([_row(f"m{i}") for i in range(200)])
    groups = {name: {row["match_id"] for row in rows if row["inner_split"] == name} for name in {
        "train_base_fit",
        "train_cal_fit",
        "train_cal_eval",
    }}

    assert groups["train_base_fit"].isdisjoint(groups["train_cal_fit"])
    assert groups["train_base_fit"].isdisjoint(groups["train_cal_eval"])
    assert groups["train_cal_fit"].isdisjoint(groups["train_cal_eval"])
    assert set().union(*groups.values()) == {row["match_id"] for row in rows}


def test_anchor_mix_alpha_zero_identity():
    rows = [_row("m1", p_draw=0.20, p_anchor_draw=0.40, p_home=0.55, p_away=0.25)]

    mixed = apply_anchor_mix(rows, {"family": "global_anchor_mix", "alpha": 0.0})

    assert mixed[0]["p_home"] == pytest.approx(0.55)
    assert mixed[0]["p_draw"] == pytest.approx(0.20)
    assert mixed[0]["p_away"] == pytest.approx(0.25)


def test_anchor_mix_probability_sum_one():
    rows = [_row("m1", p_draw=0.20, p_anchor_draw=0.40, p_home=0.55, p_away=0.25)]

    mixed = apply_anchor_mix(rows, {"family": "global_anchor_mix", "alpha": 0.10})

    assert mixed[0]["p_home"] + mixed[0]["p_draw"] + mixed[0]["p_away"] == pytest.approx(1.0)


def test_anchor_mix_preserves_home_away_ratio():
    rows = [_row("m1", p_draw=0.20, p_anchor_draw=0.40, p_home=0.60, p_away=0.20)]

    mixed = apply_anchor_mix(rows, {"family": "global_anchor_mix", "alpha": 0.10})

    assert mixed[0]["p_home"] / mixed[0]["p_away"] == pytest.approx(3.0)


def test_anchor_mix_never_uses_val_for_fit():
    with pytest.raises(ValueError, match="validation fit"):
        validate_fit_request(fit_split="val", forbid_val_fit=True, no_test=True)

    with pytest.raises(ValueError, match="validation fit"):
        select_anchor_mix([_row("m1")], fit_split="val", forbid_val_fit=True, candidate_names=["global_anchor_mix"])


def test_anchor_mix_refuses_test_rows():
    with pytest.raises(ValueError, match="test"):
        validate_fit_request(fit_split="test", forbid_val_fit=True, no_test=True)
    with pytest.raises(ValueError, match="test"):
        validate_eval_request(eval_split="test", no_refit_on_val=True, no_test=True)


def test_cli_validates_split_before_reading_predictions(tmp_path):
    missing = tmp_path / "missing_test_predictions.csv"
    result = subprocess.run(
        [
            sys.executable,
            "tools/p5_draw_anchor_mix.py",
            "--predictions-csv",
            str(missing),
            "--base-variant",
            "euro_default",
            "--fit-split",
            "test",
            "--out-dir",
            str(tmp_path / "out"),
            "--no-test",
            "--forbid-val-fit",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "test split fitting is forbidden" in result.stderr
    assert "No such file" not in result.stderr


def test_locked_eval_cli_requires_no_test_before_reading_predictions(tmp_path):
    params = tmp_path / "params.json"
    params.write_text('{"family":"global_anchor_mix","alpha":0.05}', encoding="utf-8")
    missing = tmp_path / "missing_predictions.csv"
    result = subprocess.run(
        [
            sys.executable,
            "tools/p5_draw_anchor_mix.py",
            "--predictions-csv",
            str(missing),
            "--base-variant",
            "euro_default",
            "--locked-params",
            str(params),
            "--eval-split",
            "val",
            "--out-dir",
            str(tmp_path / "out"),
            "--no-refit-on-val",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--no-test is required" in result.stderr
    assert "No such file" not in result.stderr


def test_margin_gate_uses_train_quantiles_only():
    rows = [
        {**_row("fit1", p_anchor_draw=0.30), "inner_split": "train_cal_fit", "p_anchor_home": 0.55, "p_anchor_away": 0.15},
        {**_row("fit2", p_anchor_draw=0.30), "inner_split": "train_cal_fit", "p_anchor_home": 0.50, "p_anchor_away": 0.20},
        {**_row("eval1", p_anchor_draw=0.30), "inner_split": "train_cal_eval", "p_anchor_home": 0.95, "p_anchor_away": 0.01},
    ]

    candidates = build_candidate_grid(rows, ["margin_gated_anchor_mix"])
    thresholds = {candidate["margin_threshold"] for candidate in candidates}

    assert thresholds
    assert max(thresholds) <= 0.40


def test_gated_candidates_require_train_cal_fit_rows():
    rows = [{**_row("eval1"), "inner_split": "train_cal_eval"}]

    with pytest.raises(ValueError, match="train_cal_fit"):
        build_candidate_grid(rows, ["margin_gated_anchor_mix"])


def test_suppression_gate_uses_train_quantiles_only():
    rows = [
        {**_row("fit1", p_draw=0.20, p_anchor_draw=0.30), "inner_split": "train_cal_fit"},
        {**_row("fit2", p_draw=0.21, p_anchor_draw=0.31), "inner_split": "train_cal_fit"},
        {**_row("eval1", p_draw=0.01, p_anchor_draw=0.99), "inner_split": "train_cal_eval"},
    ]

    candidates = build_candidate_grid(rows, ["suppression_gated_anchor_mix"])
    thresholds = {candidate["suppression_threshold"] for candidate in candidates}

    assert thresholds
    assert max(thresholds) < 1.0


def test_selection_requires_train_cal_eval_rows():
    rows = [{**_row("fit1"), "inner_split": "train_cal_fit"}]

    with pytest.raises(ValueError, match="train_cal_eval"):
        select_anchor_mix(rows, fit_split="train", forbid_val_fit=True, candidate_names=["global_anchor_mix"])


def test_locked_val_eval_cannot_change_params():
    rows = [_row("v1", y_true=1), _row("v2", y_true=0, p_home=0.60, p_draw=0.20, p_away=0.20)]
    locked = {"family": "global_anchor_mix", "alpha": 0.05, "source": "train_only_selection"}

    result = evaluate_locked_params(rows, locked_params=locked, eval_split="val", no_refit_on_val=True, no_test=True)

    assert result["mode"] == "locked_eval"
    assert result["params"] == locked
    assert result["metrics"]["overall_logloss"] > 0.0

    with pytest.raises(ValueError, match="no-refit"):
        evaluate_locked_params(rows, locked_params=locked, eval_split="val", no_refit_on_val=False, no_test=True)
