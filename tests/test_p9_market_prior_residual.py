import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.p6_residual_patch_itransformer import P6ResidualPatchITransformer
from tools.p6_train_residual_patch_itransformer import compute_p6_losses
from tools.p9_run_matrix import (
    FIXED_P9_SEEDS,
    FIXED_P9_VARIANTS,
    P9ProtocolError,
    build_protocol_audit,
    parse_registered_seeds,
    parse_registered_variants,
)
from tools.p9_train_market_prior_residual import compute_p9_losses


def _base_inputs():
    return {
        "out": {
            "delta_logits": torch.tensor([[0.1, -0.2, 0.0], [-0.1, 0.2, 0.0]], dtype=torch.float32),
            "goal_diff_logits": torch.randn(2, 7),
        },
        "anchor": torch.tensor([[0.50, 0.25, 0.25], [0.30, 0.30, 0.40]], dtype=torch.float32),
        "y_1x2": torch.tensor([0, 1]),
        "y_goal": torch.tensor([4, 3]),
    }


def test_p9_zero_weights_matches_p6_loss():
    data = _base_inputs()

    p6 = compute_p6_losses(
        data["out"],
        data["anchor"],
        data["y_1x2"],
        data["y_goal"],
        diff_loss_weight=0.30,
        consistency_loss_weight=0.10,
        delta_l2_weight=0.01,
    )
    p9 = compute_p9_losses(
        data["out"],
        data["anchor"],
        data["y_1x2"],
        data["y_goal"],
        diff_loss_weight=0.30,
        consistency_loss_weight=0.10,
        delta_l2_weight=0.01,
        draw_delta_l2_weight=0.0,
        anchor_kl_weight=0.0,
        true_draw_floor_weight=0.0,
    )

    assert p9["loss"].item() == pytest.approx(p6["loss"].item())
    assert p9["L_draw_delta"].item() == pytest.approx(0.0)
    assert p9["L_anchor_kl"].item() == pytest.approx(0.0)
    assert p9["L_true_draw_floor"].item() == pytest.approx(0.0)


def test_draw_delta_l2_only_penalizes_draw_residual():
    data = _base_inputs()
    base = compute_p9_losses(data["out"], data["anchor"], data["y_1x2"], data["y_goal"], 0.3, 0.1, 0.01)
    draw_changed = {
        "delta_logits": data["out"]["delta_logits"].clone(),
        "goal_diff_logits": data["out"]["goal_diff_logits"],
    }
    draw_changed["delta_logits"][:, 1] = torch.tensor([2.0, -2.0])

    losses = compute_p9_losses(
        draw_changed,
        data["anchor"],
        data["y_1x2"],
        data["y_goal"],
        0.3,
        0.1,
        0.01,
        draw_delta_l2_weight=0.5,
    )

    assert losses["L_draw_delta"].item() == pytest.approx(4.0)
    assert losses["loss"].item() > base["loss"].item()


def test_anchor_kl_penalty_increases_when_final_moves_from_anchor():
    data = _base_inputs()
    near_anchor = compute_p9_losses(
        {"delta_logits": torch.zeros_like(data["out"]["delta_logits"]), "goal_diff_logits": data["out"]["goal_diff_logits"]},
        data["anchor"],
        data["y_1x2"],
        data["y_goal"],
        0.3,
        0.1,
        0.01,
        anchor_kl_weight=1.0,
    )
    far = compute_p9_losses(
        {"delta_logits": torch.tensor([[4.0, -4.0, 0.0], [-4.0, 4.0, 0.0]]), "goal_diff_logits": data["out"]["goal_diff_logits"]},
        data["anchor"],
        data["y_1x2"],
        data["y_goal"],
        0.3,
        0.1,
        0.01,
        anchor_kl_weight=1.0,
    )

    assert near_anchor["L_anchor_kl"].item() < 1e-6
    assert far["L_anchor_kl"].item() > near_anchor["L_anchor_kl"].item()


def test_true_draw_floor_penalizes_draw_suppression_only_on_true_draws():
    data = _base_inputs()
    suppress_draw = {
        "delta_logits": torch.tensor([[0.0, -8.0, 0.0], [0.0, -8.0, 0.0]], dtype=torch.float32),
        "goal_diff_logits": data["out"]["goal_diff_logits"],
    }

    losses = compute_p9_losses(
        suppress_draw,
        data["anchor"],
        data["y_1x2"],
        data["y_goal"],
        0.3,
        0.1,
        0.01,
        true_draw_floor_weight=1.0,
    )

    assert losses["L_true_draw_floor"].item() > 0.0


def test_p9_matrix_has_exact_registered_variants():
    assert list(FIXED_P9_VARIANTS) == [
        "p9_a_p6_explicit_market_anchor",
        "p9_b_draw_delta_l2_001",
        "p9_c_anchor_kl_001",
        "p9_d_true_draw_floor_010",
    ]


def test_p9_matrix_has_exact_registered_seeds():
    assert FIXED_P9_SEEDS == [42, 123, 2025]


def test_p9_runner_rejects_unregistered_variant():
    with pytest.raises(ValueError, match="Unregistered P9 variant"):
        parse_registered_variants("p9_a_p6_explicit_market_anchor,p9_x")


def test_p9_runner_rejects_unregistered_seed():
    with pytest.raises(ValueError, match="Unregistered P9 seed"):
        parse_registered_seeds("42,7")


def test_p9_runner_rejects_test_split():
    with pytest.raises(P9ProtocolError, match="test split"):
        build_protocol_audit("train_match_ids.txt", "test_match_ids.txt", list(FIXED_P9_VARIANTS), FIXED_P9_SEEDS)


def test_p9_runner_writes_protocol_fields(tmp_path):
    audit = build_protocol_audit("train_match_ids.txt", "val_match_ids.txt", list(FIXED_P9_VARIANTS), FIXED_P9_SEEDS)
    out = tmp_path / "protocol_audit.json"
    out.write_text(json.dumps(audit), encoding="utf-8")

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["no_test_split_loaded"] is True
    assert loaded["candidate_count"] == 4
    assert loaded["candidate_count_mutation"] is False


def test_p6_default_unchanged_by_p9():
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
