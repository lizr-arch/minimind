"""Narrow tests for P4 audit and formal report collection."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p4_audit_goal_diff_labels import build_split_audit_payload
from tools.p4_generate_report import collect_variants


def test_p4_audit_payload_has_train_val_sections_and_no_test_ids():
    train_rows = [
        {
            "raw_timeline": [
                {
                    "minutes_before_kickoff": 10,
                    "euro_h": 2.0,
                    "euro_d": 3.0,
                    "euro_a": 4.0,
                    "asian_source": "raw_update",
                    "asian_line": -0.5,
                    "upper_water": 0.9,
                    "lower_water": 1.0,
                }
            ],
            "label": {"euro_result": "home", "home_goals": 2, "away_goals": 0},
        },
        {
            "raw_timeline": [
                {"minutes_before_kickoff": -2, "euro_h": 1.5, "euro_d": 5.0, "euro_a": 8.0}
            ],
            "label": {"euro_result": "draw"},
        },
    ]
    val_rows = [
        {
            "raw_timeline": [
                {
                    "minutes_before_kickoff": 20,
                    "euro_h": 2.5,
                    "euro_d": 3.0,
                    "euro_a": 3.0,
                    "asian_source": "missing",
                    "asian_line": 0.0,
                    "upper_water": 1.0,
                    "lower_water": 1.0,
                }
            ],
            "label": {"euro_result": "draw", "home_goals": 1, "away_goals": 1},
        }
    ]

    payload = build_split_audit_payload(train_rows, val_rows)

    assert set(payload) >= {"train", "val", "test_ids_used"}
    assert payload["test_ids_used"] is False
    assert payload["train"]["rows"] == 2
    assert payload["train"]["labelled_rows"] == 1
    assert payload["train"]["missing_score_rows"] == 1
    assert payload["train"]["asian_present_rate"] == 0.5
    assert payload["train"]["negative_time_valid_euro_count"] == 1
    assert payload["val"]["anchor_fallback_count"] == 0


def test_collect_variants_uses_audit_file_without_multiplying_counts(tmp_path):
    audit_dir = tmp_path / "audit"
    run_dir = tmp_path / "p4_1_runs" / "p4_euro_default_seed42"
    audit_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (audit_dir / "goal_diff_audit.json").write_text(
        json.dumps(
            {
                "train": {"anchor_fallback_count": 3, "negative_time_valid_euro_count": 5},
                "val": {"anchor_fallback_count": 7, "negative_time_valid_euro_count": 11},
                "test_ids_used": False,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "config": {"feature_groups": "euro", "consistency_loss_weight": 0.10, "seed": 42},
                "best_val_logloss": 0.95,
                "anchor_only_baseline": {"anchor_only_val_logloss": 0.96},
                "val_metrics": {"logloss": 0.95},
                "goal_diff_metrics": {},
                "history": [],
                "run_path": str(run_dir),
            }
        ),
        encoding="utf-8",
    )

    _, _, audit = collect_variants(tmp_path)

    assert audit["train"]["anchor_fallback_count"] == 3
    assert audit["val"]["anchor_fallback_count"] == 7
    assert audit["val"]["negative_time_valid_euro_count"] == 11
