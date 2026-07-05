import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p6_train_residual_patch_itransformer import (
    apply_draw_logit_coupling,
    compute_p6_losses,
)


def _base_inputs():
    return {
        "delta_logits": torch.tensor([[0.1, -0.2, 0.0], [-0.1, 0.2, 0.0]], dtype=torch.float32),
        "goal_diff_logits": torch.randn(2, 7),
        "anchor": torch.tensor([[0.5, 0.25, 0.25], [0.3, 0.3, 0.4]], dtype=torch.float32),
        "y_1x2": torch.tensor([0, 1]),
        "y_goal": torch.tensor([4, 3]),
    }


def test_lambda_final_draw_bce_zero_identity():
    data = _base_inputs()
    base = compute_p6_losses(
        {"delta_logits": data["delta_logits"], "goal_diff_logits": data["goal_diff_logits"]},
        data["anchor"],
        data["y_1x2"],
        data["y_goal"],
        diff_loss_weight=0.3,
        consistency_loss_weight=0.1,
        delta_l2_weight=0.01,
        lambda_draw_bce=0.0,
        lambda_final_draw_bce=0.0,
    )
    final_zero = compute_p6_losses(
        {"delta_logits": data["delta_logits"], "goal_diff_logits": data["goal_diff_logits"]},
        data["anchor"],
        data["y_1x2"],
        data["y_goal"],
        diff_loss_weight=0.3,
        consistency_loss_weight=0.1,
        delta_l2_weight=0.01,
        lambda_draw_bce=0.0,
        lambda_final_draw_bce=0.0,
    )

    assert final_zero["L_final_draw"].item() == pytest.approx(0.0)
    assert final_zero["loss"].item() == pytest.approx(base["loss"].item())


def test_final_draw_bce_loss_is_finite_when_enabled():
    data = _base_inputs()
    losses = compute_p6_losses(
        {"delta_logits": data["delta_logits"], "goal_diff_logits": data["goal_diff_logits"]},
        data["anchor"],
        data["y_1x2"],
        data["y_goal"],
        diff_loss_weight=0.3,
        consistency_loss_weight=0.1,
        delta_l2_weight=0.01,
        lambda_draw_bce=0.0,
        lambda_final_draw_bce=0.005,
    )

    assert torch.isfinite(losses["loss"])
    assert losses["L_final_draw"].item() > 0


def test_final_draw_bce_uses_p_final_draw_not_aux_prob():
    data = _base_inputs()
    without_aux = compute_p6_losses(
        {"delta_logits": data["delta_logits"], "goal_diff_logits": data["goal_diff_logits"]},
        data["anchor"],
        data["y_1x2"],
        data["y_goal"],
        diff_loss_weight=0.3,
        consistency_loss_weight=0.1,
        delta_l2_weight=0.01,
        lambda_draw_bce=0.0,
        lambda_final_draw_bce=0.005,
    )
    with_extreme_aux = compute_p6_losses(
        {
            "delta_logits": data["delta_logits"],
            "goal_diff_logits": data["goal_diff_logits"],
            "draw_aux_logit": torch.tensor([100.0, -100.0]),
        },
        data["anchor"],
        data["y_1x2"],
        data["y_goal"],
        diff_loss_weight=0.3,
        consistency_loss_weight=0.1,
        delta_l2_weight=0.01,
        lambda_draw_bce=0.0,
        lambda_final_draw_bce=0.005,
        enable_draw_logit_coupling=False,
    )

    assert with_extreme_aux["L_final_draw"].item() == pytest.approx(without_aux["L_final_draw"].item())
    assert with_extreme_aux["loss"].item() == pytest.approx(without_aux["loss"].item())


def test_draw_logit_coupling_disabled_by_default():
    delta = torch.zeros(2, 3)
    adjusted = apply_draw_logit_coupling(delta, torch.tensor([10.0, -10.0]))

    assert torch.equal(adjusted, delta)


def test_draw_logit_coupling_changes_final_logits_when_enabled():
    delta = torch.zeros(2, 3)
    adjusted = apply_draw_logit_coupling(
        delta,
        torch.tensor([10.0, -10.0]),
        enable_draw_logit_coupling=True,
        draw_coupling_scale=0.25,
    )

    assert adjusted[0, 1] > 0
    assert adjusted[1, 1] < 0
    assert adjusted[:, [0, 2]].abs().sum().item() == pytest.approx(0.0)


def test_draw_logit_coupling_is_bounded():
    delta = torch.zeros(2, 3)
    adjusted = apply_draw_logit_coupling(
        delta,
        torch.tensor([1000.0, -1000.0]),
        enable_draw_logit_coupling=True,
        draw_coupling_scale=0.25,
    )

    assert adjusted[:, 1].abs().max().item() <= 0.25 + 1e-6


def test_draw_logit_coupling_preserves_probability_sum():
    data = _base_inputs()
    losses = compute_p6_losses(
        {
            "delta_logits": data["delta_logits"],
            "goal_diff_logits": data["goal_diff_logits"],
            "draw_aux_logit": torch.tensor([3.0, -3.0]),
        },
        data["anchor"],
        data["y_1x2"],
        data["y_goal"],
        diff_loss_weight=0.3,
        consistency_loss_weight=0.1,
        delta_l2_weight=0.01,
        lambda_draw_bce=0.0,
        lambda_final_draw_bce=0.005,
        enable_draw_logit_coupling=True,
        draw_coupling_scale=0.25,
    )

    assert torch.allclose(losses["p_final"].sum(dim=-1), torch.ones(2), atol=1e-6)
