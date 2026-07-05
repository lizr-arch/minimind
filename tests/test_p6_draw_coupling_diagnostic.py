import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p6_draw_coupling_diagnostic import diagnose_prediction_file


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


def test_draw_margin_histogram_counts_true_draws(tmp_path):
    path = tmp_path / "val_predictions.csv"
    _write_predictions(
        path,
        [
            {"match_id": "m1", "y_true": 1, "p_home": 0.40, "p_draw": 0.35, "p_away": 0.25, "pred_class": 0, "correct": 0},
            {"match_id": "m2", "y_true": 1, "p_home": 0.34, "p_draw": 0.33, "p_away": 0.33, "pred_class": 0, "correct": 0},
            {"match_id": "m3", "y_true": 0, "p_home": 0.50, "p_draw": 0.25, "p_away": 0.25, "pred_class": 0, "correct": 1},
        ],
    )

    payload = diagnose_prediction_file(path)

    assert payload["true_draw_count"] == 2
    assert sum(payload["draw_margin_histogram"].values()) == 2


def test_draw_rank_histogram_sums_to_true_draw_count(tmp_path):
    path = tmp_path / "val_predictions.csv"
    _write_predictions(
        path,
        [
            {"match_id": "m1", "y_true": 1, "p_home": 0.20, "p_draw": 0.60, "p_away": 0.20, "pred_class": 1, "correct": 1},
            {"match_id": "m2", "y_true": 1, "p_home": 0.34, "p_draw": 0.33, "p_away": 0.33, "pred_class": 0, "correct": 0},
            {"match_id": "m3", "y_true": 1, "p_home": 0.50, "p_draw": 0.20, "p_away": 0.30, "pred_class": 0, "correct": 0},
        ],
    )

    payload = diagnose_prediction_file(path)

    assert sum(payload["draw_rank_histogram"].values()) == payload["true_draw_count"]
    assert payload["draw_rank_histogram"]["rank1"] == 1
    assert payload["draw_rank_histogram"]["rank2"] == 1
    assert payload["draw_rank_histogram"]["rank3"] == 1


def test_aux_prob_missing_is_handled_without_crash(tmp_path):
    path = tmp_path / "val_predictions.csv"
    _write_predictions(
        path,
        [
            {"match_id": "m1", "y_true": 1, "p_home": 0.40, "p_draw": 0.35, "p_away": 0.25, "pred_class": 0, "correct": 0},
        ],
    )

    payload = diagnose_prediction_file(path)

    assert payload["aux_probability_available"] is False


def test_diagnostic_does_not_select_lambda(tmp_path):
    path = tmp_path / "val_predictions.csv"
    _write_predictions(
        path,
        [
            {"match_id": "m1", "y_true": 1, "p_home": 0.40, "p_draw": 0.35, "p_away": 0.25, "pred_class": 0, "correct": 0},
        ],
    )

    payload = diagnose_prediction_file(path)

    assert payload["selected_lambda"] is None
