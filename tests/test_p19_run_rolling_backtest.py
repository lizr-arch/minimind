import csv
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p19_run_rolling_backtest import (
    P19RunProtocolError,
    build_export_command,
    build_train_command,
    load_export_pairs,
    replay_agreement_candidates,
    select_nested_candidate,
)


def _write_pred_csv(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "match_id",
                "y_ah_cover",
                "pred_ah_cover",
                "p_upper_full_win",
                "p_upper_half_win",
                "p_push",
                "p_upper_half_loss",
                "p_upper_full_loss",
            ],
        )
        writer.writeheader()
        for idx, expected in enumerate(values):
            # Keep probabilities normalized while encoding the requested unit.
            row = [(1.0 + expected) / 2.0, 0.0, 0.0, 0.0, (1.0 - expected) / 2.0]
            writer.writerow(
                {
                    "match_id": f"m{idx}",
                    "y_ah_cover": 0,
                    "pred_ah_cover": 0,
                    "p_upper_full_win": row[0],
                    "p_upper_half_win": row[1],
                    "p_push": row[2],
                    "p_upper_half_loss": row[3],
                    "p_upper_full_loss": row[4],
                }
            )


def test_p19_build_train_command_freezes_registered_variant(tmp_path):
    cmd = build_train_command(
        data_path="data.jsonl",
        train_ids="fold/train_match_ids.txt",
        val_ids="fold/selection_match_ids.txt",
        variant="p18d_direct030",
        seed=42,
        out_dir="runs/p19_tmp/run",
        epochs=3,
        batch_size=64,
        device="cpu",
        max_samples=128,
    )

    assert "tools/p18_train_ah_cover_aux.py" in cmd
    assert "--ah-cover-loss-weight" in cmd
    assert cmd[cmd.index("--ah-cover-loss-weight") + 1] == "0.0"
    assert "--ah-direct-unit-loss-weight" in cmd
    assert cmd[cmd.index("--ah-direct-unit-loss-weight") + 1] == "0.3"
    assert "--max-samples" in cmd


def test_p19_build_train_command_rejects_unregistered_variant(tmp_path):
    with pytest.raises(P19RunProtocolError):
        build_train_command(
            data_path="data.jsonl",
            train_ids="fold/train_match_ids.txt",
            val_ids="fold/selection_match_ids.txt",
            variant="new_model_family",
            seed=42,
            out_dir="runs/p19_tmp/run",
            epochs=1,
            batch_size=8,
            device="cpu",
            max_samples=0,
        )


def test_p19_build_export_command_targets_frozen_checkpoint(tmp_path):
    cmd = build_export_command(
        data_path="data.jsonl",
        ids_path="fold/forward_match_ids.txt",
        checkpoint_dir="runs/p19_tmp/checkpoint",
        out_dir="runs/p19_tmp/export",
        split_name="forward",
        device="cpu",
    )

    assert "tools/p19_export_ah_predictions.py" in cmd
    assert "--checkpoint-dir" in cmd
    assert "runs/p19_tmp/checkpoint" in cmd
    assert "--split-name" in cmd
    assert "forward" in cmd


def test_p19_replay_agreement_candidates_selects_on_selection_not_forward(tmp_path):
    cover_csv = tmp_path / "cover.csv"
    direct_csv = tmp_path / "direct.csv"
    _write_pred_csv(cover_csv, [0.26, 0.26, 0.12, -0.30])
    _write_pred_csv(direct_csv, [0.27, 0.27, 0.13, -0.31])
    contexts = [
        {"label_idx": 0, "match_id": "m0", "actual_upper_units": 1.0, "upper_water": 0.9, "lower_water": 0.9},
        {"label_idx": 1, "match_id": "m1", "actual_upper_units": 1.0, "upper_water": 0.9, "lower_water": 0.9},
        {"label_idx": 2, "match_id": "m2", "actual_upper_units": -1.0, "upper_water": 0.9, "lower_water": 0.9},
        {"label_idx": 3, "match_id": "m3", "actual_upper_units": -1.0, "upper_water": 0.9, "lower_water": 0.9},
    ]

    cover_pairs = load_export_pairs(cover_csv, contexts)
    direct_pairs = load_export_pairs(direct_csv, contexts)
    candidates = replay_agreement_candidates(
        cover_pairs,
        direct_pairs,
        contexts,
        thresholds=[0.10, 0.25],
        mode="both",
        seed=42,
        split_name="selection",
    )
    selected = select_nested_candidate(candidates, min_unique_fixtures=1)

    assert selected["threshold"] == pytest.approx(0.25)
    assert selected["score"]["selected_unique_fixtures"] == 3
    assert selected["score"]["fixture_avg_payoff"] == pytest.approx(0.9)
    json.dumps(candidates)
