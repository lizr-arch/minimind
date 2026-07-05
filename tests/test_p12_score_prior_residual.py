import json
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.p6_residual_patch_itransformer import GoalCountRateHead, P6ResidualPatchITransformer
from model.p12_score_prior_residual import P12ScorePriorResidualITransformer, ResidualDeltaHead
from tools.p12_run_matrix import (
    FIXED_P12_CONTROL,
    FIXED_P12_SEEDS,
    FIXED_P12_VARIANTS,
    P12ProtocolError,
    build_protocol_audit,
    parse_registered_seeds,
    parse_registered_variants,
)
from tools.p12_train_score_prior_residual import (
    compute_p12_losses,
    geometric_blend_prior,
    initialize_goal_count_head_from_means,
    residual_centered_l2,
)


def test_p6_default_forward_keys_remain_unchanged_after_p12():
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


def test_residual_delta_head_is_zero_initialized():
    head = ResidualDeltaHead(hidden_size=5)

    out = head(torch.randn(4, 5))

    assert torch.allclose(out, torch.zeros(4, 3), atol=1e-7)


def test_goal_count_head_bias_initializes_to_train_means():
    head = GoalCountRateHead(hidden_size=4)

    initialize_goal_count_head_from_means(head, mean_home_goals=1.8, mean_away_goals=1.1)
    lambdas = head(torch.zeros(3, 4))

    assert lambdas[:, 0].tolist() == pytest.approx([1.8, 1.8, 1.8], rel=1e-5)
    assert lambdas[:, 1].tolist() == pytest.approx([1.1, 1.1, 1.1], rel=1e-5)


def test_p12_model_outputs_required_heads_and_zero_residual():
    model = P12ScorePriorResidualITransformer(
        n_features=10,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        enable_anchor_head=True,
    )

    out = model(torch.randn(2, 10, 10, 2))

    assert set(out) == {"residual_delta", "goal_count_lambdas", "anchor_logits"}
    assert out["residual_delta"].shape == (2, 3)
    assert out["goal_count_lambdas"].shape == (2, 2)
    assert out["anchor_logits"].shape == (2, 3)
    assert torch.allclose(out["residual_delta"], torch.zeros(2, 3), atol=1e-7)


def test_geometric_blend_prior_normalizes_and_respects_alpha():
    anchor = torch.tensor([[0.60, 0.20, 0.20]], dtype=torch.float32)
    score = torch.tensor([[0.20, 0.60, 0.20]], dtype=torch.float32)

    blended = geometric_blend_prior(anchor, score, alpha=0.75)

    expected = (anchor**0.75) * (score**0.25)
    expected = expected / expected.sum(dim=-1, keepdim=True)
    assert torch.allclose(blended, expected, atol=1e-7)
    assert torch.allclose(blended.sum(dim=-1), torch.ones(1), atol=1e-7)


def test_geometric_blend_prior_clamps_zero_probs():
    anchor = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    score = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)

    blended = geometric_blend_prior(anchor, score, alpha=0.5)

    assert torch.isfinite(blended).all()
    assert torch.all(blended > 0)
    assert torch.allclose(blended.sum(dim=-1), torch.ones(1), atol=1e-7)


def test_residual_centered_l2_is_class_shift_invariant():
    residual = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 0.0, 2.0]], dtype=torch.float32)
    shifted = residual + torch.tensor([[5.0], [-3.0]])

    assert residual_centered_l2(residual).item() == pytest.approx(residual_centered_l2(shifted).item())


def _base_inputs():
    return {
        "out": {
            "residual_delta": torch.tensor([[0.1, -0.2, 0.0], [-0.1, 0.2, 0.0]], dtype=torch.float32),
            "goal_count_lambdas": torch.tensor([[1.5, 1.0], [0.8, 0.8]], dtype=torch.float32),
            "anchor_logits": torch.tensor([[0.3, -0.2, 0.0], [0.0, 0.2, -0.1]], dtype=torch.float32),
        },
        "y_1x2": torch.tensor([0, 1]),
        "home_goals": torch.tensor([2.0, 1.0]),
        "away_goals": torch.tensor([1.0, 1.0]),
    }


def test_final_ce_does_not_backprop_through_score_or_anchor_prior():
    data = _base_inputs()
    out = {key: value.clone().requires_grad_(True) for key, value in data["out"].items()}

    losses = compute_p12_losses(
        out,
        data["y_1x2"],
        data["home_goals"],
        data["away_goals"],
        prior_mode="blend",
        blend_alpha=0.75,
        count_loss_weight=0.0,
        residual_l2_weight=0.0,
        anchor_ce_weight=0.0,
    )
    losses["loss"].backward()

    assert out["residual_delta"].grad is not None
    assert out["goal_count_lambdas"].grad is None or torch.all(out["goal_count_lambdas"].grad == 0)
    assert out["anchor_logits"].grad is None or torch.all(out["anchor_logits"].grad == 0)


def test_count_and_anchor_losses_backprop_to_their_heads():
    data = _base_inputs()
    out = {key: value.clone().requires_grad_(True) for key, value in data["out"].items()}

    losses = compute_p12_losses(
        out,
        data["y_1x2"],
        data["home_goals"],
        data["away_goals"],
        prior_mode="blend",
        blend_alpha=0.90,
        count_loss_weight=0.10,
        residual_l2_weight=0.0,
        anchor_ce_weight=0.50,
    )
    losses["loss"].backward()

    assert out["goal_count_lambdas"].grad is not None
    assert out["anchor_logits"].grad is not None
    assert out["residual_delta"].grad is not None


def test_p12_score_prior_loss_uses_log_prior_plus_residual():
    data = _base_inputs()
    out = data["out"]
    losses = compute_p12_losses(
        out,
        data["y_1x2"],
        data["home_goals"],
        data["away_goals"],
        prior_mode="score",
        count_loss_weight=0.0,
        residual_l2_weight=0.0,
        anchor_ce_weight=0.0,
    )
    expected = F.cross_entropy(losses["final_logits"], data["y_1x2"])

    assert losses["L_final"].item() == pytest.approx(expected.item())
    assert torch.allclose(losses["p_prior"].sum(dim=-1), torch.ones(2), atol=1e-6)


def test_p12_matrix_has_exact_control_variants_and_seeds():
    assert FIXED_P12_CONTROL == "p12_control_p6_euro_default"
    assert FIXED_P12_SEEDS == [42, 123, 2025]
    assert list(FIXED_P12_VARIANTS) == [
        "p12_a_score_prior_residual_free",
        "p12_b_score_prior_residual_l2_005",
        "p12_c_anchor_blend_075",
        "p12_d_anchor_blend_090",
    ]


def test_p12_runner_rejects_unregistered_variant_seed_and_test_split(tmp_path):
    with pytest.raises(ValueError, match="Unregistered P12 variant"):
        parse_registered_variants("p12_a_score_prior_residual_free,p12_x")
    with pytest.raises(ValueError, match="Unregistered P12 seed"):
        parse_registered_seeds("42,7")
    with pytest.raises(P12ProtocolError, match="test split"):
        build_protocol_audit("train_match_ids.txt", "test_match_ids.txt", list(FIXED_P12_VARIANTS), FIXED_P12_SEEDS)

    audit = build_protocol_audit("train_match_ids.txt", "val_match_ids.txt", list(FIXED_P12_VARIANTS), FIXED_P12_SEEDS)
    out = tmp_path / "protocol_audit.json"
    out.write_text(json.dumps(audit), encoding="utf-8")
    loaded = json.loads(out.read_text(encoding="utf-8"))

    assert loaded["control_name"] == FIXED_P12_CONTROL
    assert loaded["expected_total_runs"] == 15
    assert loaded["explicit_prior_residual_only"] is True
    assert loaded["hard_fail_reasons"] == []
