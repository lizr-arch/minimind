"""P6 train/val-only guard tests."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p6_train_residual_patch_itransformer import (
    build_arg_parser,
    p6_selected_feature_market_type_ids,
    p6_selected_feature_names,
)


def test_p6_cli_does_not_accept_test_ids_argument():
    parser = build_arg_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--data",
                "data.jsonl",
                "--train-ids",
                "train.txt",
                "--val-ids",
                "val.txt",
                "--out-dir",
                "runs/p6_tmp",
                "--test-ids",
                "data/odds_real/splits_v6/test_match_ids.txt",
            ]
        )


def test_p6_feature_groups_are_limited_to_euro_or_euro_asian():
    assert p6_selected_feature_names("euro") == [
        "euro_h",
        "euro_d",
        "euro_a",
        "imp_h",
        "imp_d",
        "imp_a",
        "euro_margin",
        "has_euro",
        "time_log",
        "bucket_valid_count",
    ]

    assert "has_asian" in p6_selected_feature_names("euro,asian")
    with pytest.raises(ValueError, match="feature_groups"):
        p6_selected_feature_names("euro,ou")


def test_p6_market_type_ids_align_with_selected_features():
    assert p6_selected_feature_market_type_ids("euro") == [0] * 8 + [3, 3]
    assert len(p6_selected_feature_market_type_ids("euro,asian")) == len(
        p6_selected_feature_names("euro,asian")
    )
