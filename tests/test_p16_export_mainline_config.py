import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p16_export_mainline_config import build_mainline_config, write_mainline_artifacts


def _summary(verdict: str = "P16_PROMOTE_P14_MARKET_REPLICATION_MAINLINE") -> dict:
    promoted = {
        "variant": "p16_b_mktkl_100x",
        "selection_rule": "old_balanced",
        "mean_val_logloss": 0.9370384613672892,
        "mean_ece": 0.022996366964771604,
        "draw_top2": 0.3851449290911357,
        "mean_p_draw_true_draw": 0.2160190294186274,
        "market_draw_mae": 0.03618671620885531,
        "model_minus_market_p_draw": -0.03618650138378143,
        "slice_gate_pass": True,
        "decision": {"verdict": "P16_PROMOTE_P14_MARKET_REPLICATION_MAINLINE", "failed_gates": []},
    }
    return {
        "verdict": verdict,
        "control_summary": {
            "variant": "p16_control_no_market_kl",
            "mean_val_logloss": 0.9368612766265869,
            "mean_ece": 0.02723284174256006,
            "draw_top2": 0.3072463870048523,
            "mean_p_draw_true_draw": 0.19887619217236838,
            "market_draw_mae": 0.05234424521525701,
        },
        "candidate_summaries": [promoted],
    }


def test_build_mainline_config_freezes_p16_promoted_p14_training_args():
    config = build_mainline_config(_summary(), source_run_root="runs/p16_market_replication_selection")

    assert config["mainline_id"] == "p16_b_mktkl_100x_old_balanced"
    assert config["status"] == "promoted"
    assert config["train_script"] == "tools/p14_train_market_replication.py"
    assert config["train_args"]["feature_groups"] == "euro,asian,ou"
    assert config["train_args"]["market_loss_weight"] == 0.30
    assert config["train_args"]["checkpoint_selection"] == "balanced"
    assert config["train_args"]["balanced_logloss_ceiling"] == 0.9375
    assert config["train_args"]["draw_risk_loss_weight"] == 0.0
    assert config["selection_rule"] == "old_balanced"
    assert config["calibration_policy"]["val_fitted_calibration_promotable"] is False
    assert config["promotion_evidence"]["metrics"]["draw_top2"] == pytest.approx(0.3851449290911357)


def test_build_mainline_config_rejects_non_promoted_p16_summary():
    with pytest.raises(ValueError, match="P16 summary is not promoted"):
        build_mainline_config(_summary("P16_RETAIN_P6_MAINLINE"), source_run_root="runs/p16_market_replication_selection")


def test_write_mainline_artifacts_writes_json_and_markdown(tmp_path):
    config = build_mainline_config(_summary(), source_run_root="runs/p16_market_replication_selection")
    json_path = tmp_path / "configs" / "oddsmind_mainline.json"
    md_path = tmp_path / "docs" / "oddsmind_mainline_status.md"

    write_mainline_artifacts(config, json_path, md_path)

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    md = md_path.read_text(encoding="utf-8")
    assert loaded["mainline_id"] == "p16_b_mktkl_100x_old_balanced"
    assert "P16_PROMOTE_P14_MARKET_REPLICATION_MAINLINE" in md
    assert "market_loss_weight: `0.3`" in md
    assert "val-fitted calibration is diagnostic only" in md
