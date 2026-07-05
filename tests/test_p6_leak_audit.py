import csv
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p6_leak_audit import audit_p6_runs, write_markdown_report


def _write_ids(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


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


def _write_run(
    root,
    seed=42,
    *,
    data_overrides=None,
    feature_names=None,
    pred_rows=None,
):
    run_dir = root / f"p6_euro_default_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "train_samples": 2,
        "val_samples": 2,
        "test_ids_used": False,
        "validation_fit_used": False,
        "feature_names": feature_names
        or ["euro_h", "euro_d", "euro_a", "imp_h", "imp_d", "imp_a", "euro_margin", "has_euro", "time_log", "bucket_valid_count"],
        "anchor_fallback_count": 0,
        "negative_time_valid_euro_count": 0,
    }
    if data_overrides:
        data.update(data_overrides)
    (run_dir / "report.json").write_text(
        json.dumps({"config": {"seed": seed}, "data": data, "best_epoch": 1, "best_val_logloss": 0.93}),
        encoding="utf-8",
    )
    rows = pred_rows or [
        {"match_id": "v1", "y_true": 0, "p_home": 0.6, "p_draw": 0.2, "p_away": 0.2, "pred_class": 0, "correct": 1},
        {"match_id": "v1", "y_true": 2, "p_home": 0.2, "p_draw": 0.2, "p_away": 0.6, "pred_class": 2, "correct": 1},
    ]
    _write_predictions(run_dir / "val_predictions.csv", rows)
    return run_dir


def test_p6_leak_audit_passes_and_reports_duplicate_match_ids(tmp_path):
    runs_root = tmp_path / "runs"
    _write_run(runs_root)
    train_ids = tmp_path / "train.txt"
    val_ids = tmp_path / "val.txt"
    _write_ids(train_ids, ["t1", "t2"])
    _write_ids(val_ids, ["v1", "v2"])

    payload = audit_p6_runs(
        runs_root=runs_root,
        expected_seeds=[42],
        train_ids_path=train_ids,
        val_ids_path=val_ids,
        expected_train_rows=2,
        expected_val_rows=2,
    )

    assert payload["status"] == "pass"
    assert payload["duplicate_match_id_count"] == 1
    assert "P6_LEAK_AUDIT_PASS" in payload["verdict"]


def test_p6_leak_audit_missing_audit_flag_fails_closed(tmp_path):
    runs_root = tmp_path / "runs"
    _write_run(runs_root, data_overrides={"validation_fit_used": None})
    train_ids = tmp_path / "train.txt"
    val_ids = tmp_path / "val.txt"
    _write_ids(train_ids, ["t1", "t2"])
    _write_ids(val_ids, ["v1", "v2"])

    payload = audit_p6_runs(runs_root, [42], train_ids, val_ids, 2, 2)

    assert payload["status"] == "fail"
    assert "P6_LEAK_FAIL_VALIDATION_FIT_FLAG" in payload["verdict"]


def test_p6_leak_audit_train_val_overlap_fails_closed(tmp_path):
    runs_root = tmp_path / "runs"
    _write_run(runs_root)
    train_ids = tmp_path / "train.txt"
    val_ids = tmp_path / "val.txt"
    _write_ids(train_ids, ["m1", "m2"])
    _write_ids(val_ids, ["m2", "m3"])

    payload = audit_p6_runs(runs_root, [42], train_ids, val_ids, 2, 2)

    assert payload["status"] == "fail"
    assert "P6_LEAK_FAIL_TRAIN_VAL_OVERLAP" in payload["verdict"]


def test_p6_leak_audit_label_feature_fails_closed(tmp_path):
    runs_root = tmp_path / "runs"
    _write_run(runs_root, feature_names=["euro_h", "final_score_home"])
    train_ids = tmp_path / "train.txt"
    val_ids = tmp_path / "val.txt"
    _write_ids(train_ids, ["t1", "t2"])
    _write_ids(val_ids, ["v1", "v2"])

    payload = audit_p6_runs(runs_root, [42], train_ids, val_ids, 2, 2)

    assert payload["status"] == "fail"
    assert "P6_LEAK_FAIL_LABEL_FEATURE" in payload["verdict"]


def test_p6_leak_audit_probability_sum_fails_closed(tmp_path):
    runs_root = tmp_path / "runs"
    _write_run(
        runs_root,
        pred_rows=[
            {"match_id": "v1", "y_true": 0, "p_home": 0.9, "p_draw": 0.9, "p_away": 0.9, "pred_class": 0, "correct": 1},
            {"match_id": "v2", "y_true": 2, "p_home": 0.2, "p_draw": 0.2, "p_away": 0.6, "pred_class": 2, "correct": 1},
        ],
    )
    train_ids = tmp_path / "train.txt"
    val_ids = tmp_path / "val.txt"
    _write_ids(train_ids, ["t1", "t2"])
    _write_ids(val_ids, ["v1", "v2"])

    payload = audit_p6_runs(runs_root, [42], train_ids, val_ids, 2, 2)

    assert payload["status"] == "fail"
    assert "P6_LEAK_FAIL_PROBABILITY_SUM" in payload["verdict"]


def test_p6_leak_audit_test_and_val_fit_flags_fail_closed(tmp_path):
    runs_root = tmp_path / "runs"
    _write_run(runs_root, data_overrides={"test_ids_used": True, "validation_fit_used": True})
    train_ids = tmp_path / "train.txt"
    val_ids = tmp_path / "val.txt"
    _write_ids(train_ids, ["t1", "t2"])
    _write_ids(val_ids, ["v1", "v2"])

    payload = audit_p6_runs(runs_root, [42], train_ids, val_ids, 2, 2)

    assert payload["status"] == "fail"
    assert "P6_LEAK_FAIL_TEST_USED" in payload["verdict"]
    assert "P6_LEAK_FAIL_VALIDATION_FIT_FLAG" in payload["verdict"]


def test_p6_leak_markdown_lists_verdict(tmp_path):
    payload = {"status": "pass", "verdict": ["P6_LEAK_AUDIT_PASS"], "runs": []}
    out = tmp_path / "leak.md"

    write_markdown_report(payload, out)

    text = out.read_text(encoding="utf-8")
    assert "P6 Leak Audit" in text
    assert "P6_LEAK_AUDIT_PASS" in text
