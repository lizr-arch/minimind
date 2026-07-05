import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.p6_residual_patch_itransformer import P6ResidualPatchITransformer
from tools.p6_train_residual_patch_itransformer import (
    compute_draw_auxiliary_metrics,
    compute_p6_losses,
)


def test_p6_draw_aux_head_is_optional_and_disabled_by_default():
    model = P6ResidualPatchITransformer(
        n_features=10,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
    )

    out = model(torch.randn(2, 10, 10, 2))

    assert set(out) == {"delta_logits", "goal_diff_logits"}


def test_p6_draw_aux_head_returns_single_logit_when_enabled():
    model = P6ResidualPatchITransformer(
        n_features=10,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        enable_draw_aux_head=True,
    )

    out = model(torch.randn(2, 10, 10, 2))

    assert out["draw_aux_logit"].shape == (2,)


def test_p6_lambda_draw_zero_keeps_baseline_loss_path():
    delta_logits = torch.tensor([[0.1, -0.2, 0.0], [-0.1, 0.2, 0.0]], dtype=torch.float32)
    goal_diff_logits = torch.randn(2, 7)
    anchor = torch.tensor([[0.5, 0.25, 0.25], [0.3, 0.3, 0.4]], dtype=torch.float32)
    y_1x2 = torch.tensor([0, 1])
    y_goal = torch.tensor([4, 3])

    no_aux = compute_p6_losses(
        {"delta_logits": delta_logits, "goal_diff_logits": goal_diff_logits},
        anchor,
        y_1x2,
        y_goal,
        diff_loss_weight=0.3,
        consistency_loss_weight=0.1,
        delta_l2_weight=0.01,
        lambda_draw_bce=0.0,
    )
    aux_zero = compute_p6_losses(
        {"delta_logits": delta_logits, "goal_diff_logits": goal_diff_logits, "draw_aux_logit": torch.zeros(2)},
        anchor,
        y_1x2,
        y_goal,
        diff_loss_weight=0.3,
        consistency_loss_weight=0.1,
        delta_l2_weight=0.01,
        lambda_draw_bce=0.0,
    )

    assert aux_zero["L_draw_aux"].item() == pytest.approx(0.0)
    assert aux_zero["loss"].item() == pytest.approx(no_aux["loss"].item())


def test_p6_draw_aux_loss_is_finite_when_enabled():
    losses = compute_p6_losses(
        {
            "delta_logits": torch.randn(4, 3),
            "goal_diff_logits": torch.randn(4, 7),
            "draw_aux_logit": torch.randn(4),
        },
        torch.full((4, 3), 1.0 / 3.0),
        torch.tensor([0, 1, 2, 1]),
        torch.tensor([2, 3, 4, 3]),
        diff_loss_weight=0.3,
        consistency_loss_weight=0.1,
        delta_l2_weight=0.01,
        lambda_draw_bce=0.01,
    )

    assert torch.isfinite(losses["loss"])
    assert losses["L_draw_aux"].item() > 0


def test_compute_draw_auxiliary_metrics_reports_required_fields():
    probs = torch.tensor(
        [
            [0.60, 0.25, 0.15],
            [0.30, 0.40, 0.30],
            [0.20, 0.20, 0.60],
            [0.35, 0.34, 0.31],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 2, 1])

    metrics = compute_draw_auxiliary_metrics(probs, labels)

    assert set(metrics) >= {
        "draw_class_nll",
        "draw_precision",
        "draw_top2_recall",
        "mean_p_draw_on_non_draw",
        "classwise_ece_draw",
    }
    assert metrics["draw_recall"] == pytest.approx(0.5)
    assert metrics["draw_precision"] == pytest.approx(1.0)
    assert metrics["draw_top2_recall"] == pytest.approx(1.0)
