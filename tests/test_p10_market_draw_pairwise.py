import json
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p6_train_residual_patch_itransformer import compute_p6_losses
from tools.p10_run_matrix import (
    FIXED_P10_CONTROL,
    FIXED_P10_SEEDS,
    FIXED_P10_VARIANTS,
    P10ProtocolError,
    build_protocol_audit,
    parse_registered_seeds,
    parse_registered_variants,
)
from tools.p10_train_market_draw_pairwise import (
    build_teacher_coverage,
    compute_market_pairwise_metrics,
    compute_p10_losses,
    market_draw_rank_tie_aware,
    market_pairwise_targets,
    true_draw_top2_margin_loss,
)


def _base_inputs():
    return {
        "out": {
            "delta_logits": torch.tensor([[0.1, -0.2, 0.0], [-0.1, 0.2, 0.0]], dtype=torch.float32),
            "goal_diff_logits": torch.randn(2, 7),
        },
        "anchor": torch.tensor([[0.50, 0.25, 0.25], [0.30, 0.30, 0.40]], dtype=torch.float32),
        "market": torch.tensor([[0.45, 0.30, 0.25], [0.35, 0.33, 0.32]], dtype=torch.float32),
        "valid_teacher": torch.tensor([True, True]),
        "y_1x2": torch.tensor([0, 1]),
        "y_goal": torch.tensor([4, 3]),
    }


def test_market_pairwise_targets_use_draw_vs_home_and_away_ratios():
    market = torch.tensor([[0.50, 0.25, 0.25], [0.20, 0.30, 0.50]], dtype=torch.float32)

    targets = market_pairwise_targets(market)

    assert targets["draw_vs_home"].tolist() == pytest.approx([1 / 3, 0.6])
    assert targets["draw_vs_away"].tolist() == pytest.approx([0.5, 0.375])


def test_tie_aware_market_draw_rank_treats_near_tie_as_top2():
    market = torch.tensor(
        [
            [0.40, 0.30, 0.30],  # only home clearly above draw, so draw is top2
            [0.40, 0.30, 0.31],  # home and away above draw, so rank 3
            [0.30, 0.30, 0.40],  # draw tied second, so top2
            [0.30, 0.40, 0.40],  # draw tied first
        ],
        dtype=torch.float32,
    )

    ranks = market_draw_rank_tie_aware(market)

    assert ranks.tolist() == [2, 3, 2, 1]
    assert (ranks <= 2).tolist() == [True, False, True, True]


def test_p10_zero_weights_matches_p6_loss():
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
    p10 = compute_p10_losses(
        data["out"],
        data["anchor"],
        data["y_1x2"],
        data["y_goal"],
        data["market"],
        data["valid_teacher"],
        diff_loss_weight=0.30,
        consistency_loss_weight=0.10,
        delta_l2_weight=0.01,
        market_draw_pairwise_weight=0.0,
        true_draw_top2_margin_weight=0.0,
    )

    assert p10["loss"].item() == pytest.approx(p6["loss"].item())
    assert p10["L_market_draw_pairwise"].item() == pytest.approx(0.0)
    assert p10["L_true_draw_top2_margin"].item() == pytest.approx(0.0)


def test_pairwise_loss_uses_valid_teacher_mask_only():
    data = _base_inputs()
    data["valid_teacher"] = torch.tensor([True, False])
    losses = compute_p10_losses(
        data["out"],
        data["anchor"],
        data["y_1x2"],
        data["y_goal"],
        data["market"],
        data["valid_teacher"],
        0.30,
        0.10,
        0.01,
        market_draw_pairwise_weight=1.0,
        market_draw_pairwise_mode="all",
    )
    final_logits = losses["final_logits"]
    targets = market_pairwise_targets(data["market"])
    expected = 0.5 * F.binary_cross_entropy_with_logits(
        final_logits[:1, 1] - final_logits[:1, 0],
        targets["draw_vs_home"][:1],
    ) + 0.5 * F.binary_cross_entropy_with_logits(
        final_logits[:1, 1] - final_logits[:1, 2],
        targets["draw_vs_away"][:1],
    )

    assert losses["market_pairwise_n"].item() == pytest.approx(1.0)
    assert losses["L_market_draw_pairwise"].item() == pytest.approx(expected.item())


def test_true_draw_top2_margin_uses_min_opponent_and_focus_mask():
    logits = torch.tensor(
        [
            [2.0, 1.9, 0.0],  # draw already beats min opponent, small loss
            [2.0, 0.0, 1.0],  # draw below min opponent, larger loss
            [0.0, 5.0, 4.0],  # not true draw, masked out
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([1, 1, 0])
    mask = torch.tensor([True, True, True])

    loss = true_draw_top2_margin_loss(logits, labels, mask, margin=0.05)

    expected = torch.stack(
        [
            F.softplus(torch.tensor(0.0 - 1.9 + 0.05)),
            F.softplus(torch.tensor(1.0 - 0.0 + 0.05)),
        ]
    ).mean()
    assert loss.item() == pytest.approx(expected.item())


def test_teacher_coverage_counts_train_and_val_separately():
    train_teacher = torch.tensor([[0.5, 0.3, 0.2], [float("nan"), 0.3, 0.2]])
    val_teacher = torch.tensor([[0.4, 0.3, 0.3], [0.0, 0.0, 0.0]])
    coverage = build_teacher_coverage(
        train_teacher,
        val_teacher,
        train_invalid_reasons=["valid", "nan_or_inf"],
        val_invalid_reasons=["valid", "non_positive_sum"],
    )

    assert coverage["train_total_rows"] == 2
    assert coverage["train_valid_teacher_rows"] == 1
    assert coverage["train_teacher_coverage"] == pytest.approx(0.5)
    assert coverage["val_total_rows"] == 2
    assert coverage["val_valid_teacher_rows"] == 1
    assert coverage["invalid_reason_counts_train"] == {"nan_or_inf": 1}
    assert coverage["invalid_reason_counts_val"] == {"non_positive_sum": 1}


def test_market_pairwise_metrics_reports_valid_and_skipped_counts():
    probs = torch.tensor([[0.5, 0.3, 0.2], [0.4, 0.2, 0.4], [0.3, 0.4, 0.3]], dtype=torch.float32)
    market = torch.tensor([[0.45, 0.30, 0.25], [0.3, 0.4, 0.3], [0.3, 0.4, 0.3]], dtype=torch.float32)
    valid = torch.tensor([True, False, True])

    metrics = compute_market_pairwise_metrics(probs, market, valid)

    assert metrics["valid_n"] == 2
    assert metrics["skipped_n"] == 1
    assert "market_pairwise_agreement" in metrics
    assert "model_vs_market_draw_corr" in metrics


def test_p10_matrix_has_exact_variants_control_and_seeds():
    assert FIXED_P10_CONTROL == "p10_control_p6_euro_default"
    assert FIXED_P10_SEEDS == [42, 123, 2025]
    assert list(FIXED_P10_VARIANTS) == [
        "p10_a_pairwise_all_005",
        "p10_b_pairwise_all_020",
        "p10_c_pairwise_market_top2_010",
        "p10_d_pairwise_top2_true_margin",
    ]


def test_p10_runner_rejects_unregistered_variant_and_seed():
    with pytest.raises(ValueError, match="Unregistered P10 variant"):
        parse_registered_variants("p10_a_pairwise_all_005,p10_x")
    with pytest.raises(ValueError, match="Unregistered P10 seed"):
        parse_registered_seeds("42,7")


def test_p10_runner_rejects_test_split_and_reports_control_requirement(tmp_path):
    with pytest.raises(P10ProtocolError, match="test split"):
        build_protocol_audit("train_match_ids.txt", "test_match_ids.txt", list(FIXED_P10_VARIANTS), FIXED_P10_SEEDS)

    audit = build_protocol_audit("train_match_ids.txt", "val_match_ids.txt", list(FIXED_P10_VARIANTS), FIXED_P10_SEEDS)
    out = tmp_path / "protocol_audit.json"
    out.write_text(json.dumps(audit), encoding="utf-8")
    loaded = json.loads(out.read_text(encoding="utf-8"))

    assert loaded["control_required"] is True
    assert loaded["control_name"] == FIXED_P10_CONTROL
    assert loaded["candidate_count"] == 4
    assert loaded["expected_total_runs"] == 15
    assert loaded["hard_fail_reasons"] == []
