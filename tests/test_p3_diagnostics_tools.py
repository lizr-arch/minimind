"""Narrow tests for P3.1 diagnostic helpers."""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.odds_patch_itransformer_v2 import FEATURE_DEFS
from tools.p3b_train_patch_itransformer import (
    apply_clean_missing_markets,
    apply_feature_group_mask,
)
from tools.p3_generate_diagnostics_report import build_final_report_payload
from tools.p3_prediction_diagnostics import compute_prediction_diagnostics
from tools.p3_export_val_predictions import _p3b_preprocessing_config


FEATURE_INDEX = {name: idx for idx, (name, _) in enumerate(FEATURE_DEFS)}


def test_feature_group_mask_zeroes_unselected_market_groups_but_keeps_time_quality():
    x = torch.ones(2, 10, len(FEATURE_DEFS), 2)
    x[:, :, FEATURE_INDEX["market_present_count"], :] = 3.0

    masked = apply_feature_group_mask(x.clone(), "euro")

    asian_indices = [FEATURE_INDEX[name] for name in [
        "asian_line",
        "upper_water",
        "lower_water",
        "asian_upper_implied",
        "asian_lower_implied",
        "asian_margin",
        "has_asian",
    ]]
    ou_indices = [FEATURE_INDEX[name] for name in [
        "over_under_line",
        "over_water",
        "under_water",
        "over_implied",
        "under_implied",
        "ou_margin",
        "has_ou",
    ]]
    time_passthrough_indices = [FEATURE_INDEX[name] for name in [
        "time_log",
        "bucket_valid_count",
    ]]

    assert torch.count_nonzero(masked[:, :, asian_indices, :]) == 0
    assert torch.count_nonzero(masked[:, :, ou_indices, :]) == 0
    assert torch.all(masked[:, :, time_passthrough_indices, :] == 1)
    assert torch.all(masked[:, :, FEATURE_INDEX["market_present_count"], :] == 1)


def test_clean_missing_markets_zeroes_default_ou_implied_values_when_has_ou_is_zero():
    x = torch.zeros(1, 1, len(FEATURE_DEFS), 2)
    x[:, :, FEATURE_INDEX["over_under_line"], :] = 2.5
    x[:, :, FEATURE_INDEX["over_water"], :] = 1.0
    x[:, :, FEATURE_INDEX["under_water"], :] = 1.0
    x[:, :, FEATURE_INDEX["over_implied"], :] = 0.5
    x[:, :, FEATURE_INDEX["under_implied"], :] = 0.5
    x[:, :, FEATURE_INDEX["ou_margin"], :] = 1.0
    x[:, :, FEATURE_INDEX["has_ou"], :] = 0.0

    cleaned = apply_clean_missing_markets(x.clone())

    for name in [
        "over_under_line",
        "over_water",
        "under_water",
        "over_implied",
        "under_implied",
        "ou_margin",
    ]:
        assert torch.count_nonzero(cleaned[:, :, FEATURE_INDEX[name], :]) == 0
    assert torch.count_nonzero(cleaned[:, :, FEATURE_INDEX["has_ou"], :]) == 0


def test_clean_missing_markets_removes_missing_ou_fillers_before_bucket_means():
    class DummyDataset:
        def __init__(self):
            self.samples = [
                {
                    "raw_timeline": [
                        {
                            "minutes_before_kickoff": 60,
                            "euro_h": 2.0,
                            "euro_d": 3.0,
                            "euro_a": 4.0,
                            "over_under_source": "missing",
                            "over_under_line": 2.5,
                            "over_water": 1.0,
                            "under_water": 1.0,
                        },
                        {
                            "minutes_before_kickoff": 50,
                            "euro_h": 2.0,
                            "euro_d": 3.0,
                            "euro_a": 4.0,
                            "over_under_source": "raw_update",
                            "over_under_line": 2.75,
                            "over_water": 2.0,
                            "under_water": 2.0,
                        },
                    ]
                }
            ]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return {"euro_label": 0}

    from tools.p3b_train_patch_itransformer import build_bucketed_dataset_v2

    x, _ = build_bucketed_dataset_v2(
        DummyDataset().samples,
        DummyDataset(),
        feature_groups="euro,asian,ou",
        clean_missing_markets=True,
    )

    over_implied = x[0, :, FEATURE_INDEX["over_implied"], 1]
    nonzero_values = over_implied[over_implied > 0]
    assert torch.allclose(nonzero_values, torch.tensor([0.25]))


def test_prediction_diagnostics_detects_draw_argmax_collapse_and_draw_recall():
    rows = [
        {"model": "toy", "y_true": 1, "p_home": 0.60, "p_draw": 0.20, "p_away": 0.20, "pred_class": 0},
        {"model": "toy", "y_true": 1, "p_home": 0.55, "p_draw": 0.25, "p_away": 0.20, "pred_class": 0},
        {"model": "toy", "y_true": 0, "p_home": 0.70, "p_draw": 0.20, "p_away": 0.10, "pred_class": 0},
        {"model": "toy", "y_true": 2, "p_home": 0.20, "p_draw": 0.20, "p_away": 0.60, "pred_class": 2},
    ]

    diag = compute_prediction_diagnostics("toy", rows)

    assert diag["predicted_class_distribution"]["draw"] == 0
    assert diag["draw_argmax_collapse"] is True
    assert diag["per_class"]["draw"]["recall"] == 0.0
    assert diag["mean_p_draw_on_true_draw"] == 0.225


def test_final_report_payload_contains_required_top_level_sections(tmp_path):
    prediction_path = tmp_path / "prediction.json"
    seed_path = tmp_path / "seed.json"
    ablation_path = tmp_path / "ablation.json"
    clean_path = tmp_path / "clean.json"
    for path, payload in [
        (prediction_path, {"models": {}, "verdict": []}),
        (seed_path, {"models": {}, "verdict": []}),
        (ablation_path, {"variants": [], "verdict": []}),
        (clean_path, {"runs": [], "verdict": []}),
    ]:
        path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_final_report_payload(
        prediction_json=prediction_path,
        seed_json=seed_path,
        ablation_json=ablation_path,
        clean_json=clean_path,
    )

    assert set(report) >= {
        "prediction_diagnostics",
        "seed_summary",
        "ablation_summary",
        "p3b_clean",
        "verdict",
        "next_actions",
    }


def test_p3b_export_requires_explicit_clean_flag_for_legacy_checkpoint_metadata():
    class Args:
        feature_groups = "euro,asian,ou"
        clean_missing_markets = False
        no_clean_missing_markets = False

    try:
        _p3b_preprocessing_config({"config": {}}, {}, Args())
    except ValueError as exc:
        assert "clean_missing_markets" in str(exc)
    else:
        raise AssertionError("legacy P3b export should require explicit clean flag")


def test_final_report_payload_accepts_baseline_run_paths(tmp_path):
    prediction_path = tmp_path / "prediction.json"
    seed_path = tmp_path / "seed.json"
    ablation_path = tmp_path / "ablation.json"
    clean_path = tmp_path / "clean.json"
    for path, payload in [
        (prediction_path, {"models": {}, "verdict": []}),
        (seed_path, {"models": {}, "verdict": []}),
        (ablation_path, {"variants": [], "verdict": []}),
        (clean_path, {"runs": [], "verdict": []}),
    ]:
        path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_final_report_payload(
        prediction_json=prediction_path,
        seed_json=seed_path,
        ablation_json=ablation_path,
        clean_json=clean_path,
        baseline_run_paths=["custom/raw", "custom/p3a", "custom/p3b"],
    )

    assert report["inputs"]["baseline_run_paths"] == ["custom/raw", "custom/p3a", "custom/p3b"]
