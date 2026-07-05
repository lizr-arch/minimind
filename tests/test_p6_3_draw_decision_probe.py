import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p6_3_draw_decision_probe import (
    apply_draw_decision_policy,
    evaluate_policy_on_predictions,
    run_policy_probe,
)


def _write_predictions(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["match_id", "y_true", "p_home", "p_draw", "p_away", "pred_class", "correct"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_top2_margin_policy_selects_draw_when_close_rank2():
    pred = apply_draw_decision_policy([0.36, 0.34, 0.30], "draw_top2_margin_le_0_03")

    assert pred == 1


def test_top2_margin_policy_does_not_select_rank3_draw():
    pred = apply_draw_decision_policy([0.45, 0.20, 0.35], "draw_top2_margin_le_0_30")

    assert pred == 0


def test_policy_metrics_report_draw_precision_recall_and_accuracy_delta(tmp_path):
    path = tmp_path / "val_predictions.csv"
    _write_predictions(
        path,
        [
            {"match_id": "m1", "y_true": 1, "p_home": 0.36, "p_draw": 0.34, "p_away": 0.30, "pred_class": 0, "correct": 0},
            {"match_id": "m2", "y_true": 0, "p_home": 0.36, "p_draw": 0.34, "p_away": 0.30, "pred_class": 0, "correct": 1},
            {"match_id": "m3", "y_true": 2, "p_home": 0.20, "p_draw": 0.30, "p_away": 0.50, "pred_class": 2, "correct": 1},
        ],
    )

    metrics = evaluate_policy_on_predictions(path, "draw_top2_margin_le_0_03")

    assert metrics["draw_recall"] == pytest.approx(1.0)
    assert metrics["draw_precision"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["accuracy_delta_vs_argmax"] == pytest.approx(0.0)
    assert metrics["selected_policy"] is None


def test_probe_runs_fixed_policies_without_selecting_winner(tmp_path):
    root = tmp_path / "runs"
    run_dir = root / "p6_euro_default_seed42"
    _write_predictions(
        run_dir / "val_predictions.csv",
        [
            {"match_id": "m1", "y_true": 1, "p_home": 0.36, "p_draw": 0.34, "p_away": 0.30, "pred_class": 0, "correct": 0},
            {"match_id": "m2", "y_true": 0, "p_home": 0.60, "p_draw": 0.20, "p_away": 0.20, "pred_class": 0, "correct": 1},
        ],
    )

    payload = run_policy_probe(root, "p6_euro_default_seed", [42], ["argmax", "draw_top2_any"])

    assert payload["selected_policy"] is None
    assert [item["policy"] for item in payload["policies"]] == ["argmax", "draw_top2_any"]
    assert payload["policies"][1]["draw_recall"] == pytest.approx(1.0)
