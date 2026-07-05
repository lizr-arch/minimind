"""Narrow tests for exporting P4 train/val split predictions for P5.2a."""

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p5_export_p4_split_predictions import (  # noqa: E402
    EXPORT_FIELDS,
    build_export_plan,
    build_prediction_rows,
)


def _write_report(run_dir, **config_overrides):
    config = {
        "data": "data.jsonl",
        "train_ids": "train_ids.txt",
        "val_ids": "val_ids.txt",
        "feature_groups": "euro",
        "d_model": 128,
        "dropout": 0.15,
        "batch_size": 64,
    }
    config.update(config_overrides)
    (run_dir / "report.json").write_text(json.dumps({"config": config}), encoding="utf-8")
    (run_dir / "scaler.json").write_text(json.dumps({"scaling": "none"}), encoding="utf-8")


def test_export_refuses_test_split(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_report(run_dir)
    (run_dir / "best_model.pth").write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="test split"):
        build_export_plan(run_dir=run_dir, split="test", out_csv=tmp_path / "out.csv", no_test=True)


def test_export_requires_existing_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_report(run_dir)

    with pytest.raises(FileNotFoundError, match="best_model.pth"):
        build_export_plan(run_dir=run_dir, split="train", out_csv=tmp_path / "out.csv", no_test=True)


def test_export_rows_include_final_anchor_and_diff_columns():
    rows = build_prediction_rows(
        match_ids=["m1"],
        y_true=torch.tensor([1]),
        y_goal_diff=torch.tensor([3]),
        p_anchor=torch.tensor([[0.30, 0.40, 0.30]]),
        preds={
            "p_final": torch.tensor([[0.25, 0.50, 0.25]]),
            "p_from_diff": torch.tensor([[0.20, 0.60, 0.20]]),
            "q_diff": torch.tensor([[0.01, 0.02, 0.03, 0.90, 0.02, 0.01, 0.01]]),
        },
    )

    assert list(rows[0]) == EXPORT_FIELDS
    assert rows[0]["match_id"] == "m1"
    assert rows[0]["y_true"] == 1
    assert rows[0]["y_goal_diff"] == 3
    assert rows[0]["p_draw"] == pytest.approx(0.50)
    assert rows[0]["p_anchor_draw"] == pytest.approx(0.40)
    assert rows[0]["p_draw_from_diff"] == pytest.approx(0.60)
    assert rows[0]["pred_class"] == 1
    assert rows[0]["correct"] == 1
    assert rows[0]["pred_diff_bucket"] == 3
    assert rows[0]["correct_diff_bucket"] == 1


def test_export_uses_report_config_for_feature_groups_and_scaler(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_report(
        run_dir,
        data="matches.jsonl",
        train_ids="splits/train.txt",
        val_ids="splits/val.txt",
        feature_groups="euro,asian",
        d_model=96,
        dropout=0.25,
        batch_size=32,
    )
    (run_dir / "best_model.pth").write_bytes(b"checkpoint")

    plan = build_export_plan(run_dir=run_dir, split="val", out_csv=tmp_path / "out.csv", no_test=True)

    assert plan.data_path == run_dir / "matches.jsonl"
    assert plan.ids_path == run_dir / "splits" / "val.txt"
    assert plan.feature_groups == "euro,asian"
    assert plan.d_model == 96
    assert plan.dropout == 0.25
    assert plan.batch_size == 32
    assert plan.scaler_path == run_dir / "scaler.json"
    assert plan.test_ids_used is False
