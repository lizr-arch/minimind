"""Narrow tests for P3.2 scaling and calibration helpers."""

import os
import subprocess
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.odds_patch_itransformer_v2 import FEATURE_DEFS
import tools.p3b_train_patch_itransformer as p3b
import tools.p3_save_raw_pooled_mlp_baseline as raw_mlp


FEATURE_INDEX = {name: idx for idx, (name, _) in enumerate(FEATURE_DEFS)}


def _empty_feature_tensor(n_samples: int) -> torch.Tensor:
    return torch.zeros(n_samples, 1, len(FEATURE_DEFS), 1)


def test_standard_scaler_uses_train_stats_and_keeps_mask_and_count_features():
    assert hasattr(p3b, "fit_feature_scaler")
    assert hasattr(p3b, "apply_feature_scaler")

    x_train = _empty_feature_tensor(2)
    x_val = _empty_feature_tensor(1)
    x_train[:, :, FEATURE_INDEX["euro_h"], :] = torch.tensor([[[2.0]], [[4.0]]])
    x_val[:, :, FEATURE_INDEX["euro_h"], :] = 5.0

    for name, value in [
        ("has_euro", 1.0),
        ("has_asian", 0.0),
        ("has_ou", 1.0),
        ("bucket_valid_count", 7.0),
        ("market_present_count", 2.0),
    ]:
        x_train[:, :, FEATURE_INDEX[name], :] = value
        x_val[:, :, FEATURE_INDEX[name], :] = value

    scaler = p3b.fit_feature_scaler(x_train, "standard")
    transformed = p3b.apply_feature_scaler(x_val.clone(), scaler)

    assert torch.allclose(transformed[:, :, FEATURE_INDEX["euro_h"], :], torch.tensor([[[2.0]]]))
    for name, value in [
        ("has_euro", 1.0),
        ("has_asian", 0.0),
        ("has_ou", 1.0),
        ("bucket_valid_count", 7.0),
        ("market_present_count", 2.0),
    ]:
        assert torch.all(transformed[:, :, FEATURE_INDEX[name], :] == value)
        assert name in scaler["excluded_features"]


def test_robust_scaler_falls_back_to_unit_scale_when_iqr_and_std_are_zero():
    x_train = _empty_feature_tensor(4)
    x_val = _empty_feature_tensor(1)
    x_train[:, :, FEATURE_INDEX["euro_h"], :] = 10.0
    x_val[:, :, FEATURE_INDEX["euro_h"], :] = 12.0

    scaler = p3b.fit_feature_scaler(x_train, "robust")
    transformed = p3b.apply_feature_scaler(x_val.clone(), scaler)

    euro_h = scaler["features"]["euro_h"]
    assert euro_h["center"] == 10.0
    assert euro_h["scale"] == 1.0
    assert torch.allclose(transformed[:, :, FEATURE_INDEX["euro_h"], :], torch.tensor([[[2.0]]]))


def test_weighted_ce_mild_draw_matches_torch_cross_entropy():
    assert hasattr(p3b, "build_class_weights")
    assert hasattr(p3b, "compute_training_loss")
    assert hasattr(raw_mlp, "compute_training_loss")

    logits = torch.tensor([[2.0, 0.5, 0.1], [0.2, 1.5, 0.3], [0.1, 0.2, 1.7]])
    labels = torch.tensor([0, 1, 2])
    weights = p3b.build_class_weights(
        labels,
        class_weights="mild_draw",
        draw_weight=1.5,
        device=torch.device("cpu"),
    )

    loss = p3b.compute_training_loss(
        logits,
        labels,
        loss_name="weighted_ce",
        class_weights=weights,
        label_smoothing=0.0,
        focal_gamma=2.0,
    )

    assert torch.allclose(weights, torch.tensor([1.0, 1.5, 1.0]))
    assert torch.allclose(loss, F.cross_entropy(logits, labels, weight=weights))


def test_label_smoothing_loss_matches_torch_cross_entropy():
    logits = torch.tensor([[2.0, 0.5, 0.1], [0.2, 1.5, 0.3]])
    labels = torch.tensor([0, 1])

    loss = p3b.compute_training_loss(
        logits,
        labels,
        loss_name="label_smoothing",
        class_weights=None,
        label_smoothing=0.02,
        focal_gamma=2.0,
    )

    assert torch.allclose(loss, F.cross_entropy(logits, labels, label_smoothing=0.02))


def test_p3_2_final_report_payload_contains_required_sections():
    from tools.p3_2_generate_scaling_calibration_report import build_final_report_payload

    payload = build_final_report_payload(
        scaling_summary={"best_scaling": "robust", "runs": []},
        scaled_ablation_summary={"best_variant": "euro", "runs": []},
        calibration_summary={"best_variant": "raw_ce", "runs": []},
        raw_mlp_baseline_logloss=0.9414,
    )

    assert set(payload) >= {
        "scaling_summary",
        "scaled_ablation_summary",
        "calibration_summary",
        "best_model",
        "verdict",
        "next_actions",
    }


def test_p3_2_report_script_runs_as_path_cli():
    result = subprocess.run(
        [sys.executable, "tools/p3_2_generate_scaling_calibration_report.py", "--help"],
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Generate P3.2 scaling and calibration reports" in result.stdout
