"""Narrow tests for the P4 residual goal-diff objective-probe model."""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.p4_residual_goal_diff import P4ResidualGoalDiffModel


def test_p4_model_forward_returns_delta_and_goal_diff_logits():
    model = P4ResidualGoalDiffModel(n_features=17, d_model=32, dropout=0.0)
    x = torch.randn(4, 10, 17, 2)

    output = model(x)

    assert set(output) == {"delta_logits", "goal_diff_logits"}
    assert output["delta_logits"].shape == (4, 3)
    assert output["goal_diff_logits"].shape == (4, 7)


def test_p4_model_accepts_euro_only_feature_count():
    model = P4ResidualGoalDiffModel(n_features=10, d_model=16, dropout=0.0)
    x = torch.randn(2, 10, 10, 2)

    output = model(x)

    assert output["delta_logits"].shape == (2, 3)
    assert output["goal_diff_logits"].shape == (2, 7)
