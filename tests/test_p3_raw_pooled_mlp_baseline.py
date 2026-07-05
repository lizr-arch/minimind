"""Tests for the formal RawPooledMLP baseline script."""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p3_save_raw_pooled_mlp_baseline import (
    FEATURE_NAMES,
    RawPooledMLP,
    build_raw_pooled_features,
)


def test_build_raw_pooled_features_returns_nine_euro_features():
    row = {
        "odds_timeline": [
            {
                "euro_h": 2.0,
                "euro_d": 4.0,
                "euro_a": 4.0,
                "minutes_before_kickoff": 60,
                "has_euro": True,
            },
            {
                "euro_h": 4.0,
                "euro_d": 2.0,
                "euro_a": 4.0,
                "minutes_before_kickoff": 0,
                "has_euro": True,
            },
        ],
    }

    feats = build_raw_pooled_features(row)

    assert FEATURE_NAMES == [
        "euro_h",
        "euro_d",
        "euro_a",
        "imp_h",
        "imp_d",
        "imp_a",
        "margin",
        "time_log",
        "has_euro",
    ]
    assert feats.shape == (9,)
    assert torch.isclose(feats[0], torch.tensor(3.0))
    assert torch.isclose(feats[1], torch.tensor(3.0))
    assert torch.isclose(feats[2], torch.tensor(4.0))
    assert torch.isclose(feats[3], torch.tensor(0.375))
    assert torch.isclose(feats[4], torch.tensor(0.375))
    assert torch.isclose(feats[5], torch.tensor(0.25))
    assert torch.isclose(feats[6], torch.tensor(0.0))
    assert torch.isclose(feats[7], torch.tensor(math.log1p(60) / 2))
    assert torch.isclose(feats[8], torch.tensor(1.0))


def test_raw_pooled_mlp_outputs_three_logits():
    model = RawPooledMLP(input_dim=9)
    logits = model(torch.randn(4, 9))

    assert logits.shape == (4, 3)
