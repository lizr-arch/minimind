"""Narrow tests for the P6 residual Patch-iTransformer model."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.p6_residual_patch_itransformer import P6ResidualPatchITransformer


def test_p6_model_forward_returns_residual_and_goal_diff_heads():
    model = P6ResidualPatchITransformer(
        n_features=10,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
    )
    x = torch.randn(3, 10, 10, 2)

    out = model(x)

    assert set(out) == {"delta_logits", "goal_diff_logits"}
    assert out["delta_logits"].shape == (3, 3)
    assert out["goal_diff_logits"].shape == (3, 7)


def test_p6_model_rejects_wrong_feature_count():
    model = P6ResidualPatchITransformer(
        n_features=10,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
    )

    with pytest.raises(ValueError, match="Expected input shape"):
        model(torch.randn(2, 10, 11, 2))


def test_p6_model_accepts_market_type_ids_when_supplied():
    model = P6ResidualPatchITransformer(
        n_features=10,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        market_type_ids=[0] * 8 + [3, 3],
    )

    out = model(torch.randn(2, 10, 10, 2))

    assert out["delta_logits"].shape == (2, 3)
    assert out["goal_diff_logits"].shape == (2, 7)


def test_p6_model_can_enable_optional_ah_cover_head_without_changing_default_outputs():
    base = P6ResidualPatchITransformer(
        n_features=10,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
    )
    with_ah = P6ResidualPatchITransformer(
        n_features=10,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        enable_ah_cover_head=True,
    )
    x = torch.randn(2, 10, 10, 2)

    assert "ah_cover_logits" not in base(x)
    out = with_ah(x)

    assert out["ah_cover_logits"].shape == (2, 5)


def test_p6_model_can_enable_optional_ah_unit_head():
    model = P6ResidualPatchITransformer(
        n_features=10,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        enable_ah_unit_head=True,
    )

    out = model(torch.randn(2, 10, 10, 2))

    assert out["ah_unit_pred"].shape == (2,)
    assert torch.all(out["ah_unit_pred"] <= 1.0)
    assert torch.all(out["ah_unit_pred"] >= -1.0)
