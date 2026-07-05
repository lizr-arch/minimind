import json
import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.p6_residual_patch_itransformer import GoalCountRateHead, P6ResidualPatchITransformer
from tools.p11_run_matrix import (
    FIXED_P11_CONTROL,
    FIXED_P11_SEEDS,
    FIXED_P11_VARIANTS,
    P11ProtocolError,
    build_protocol_audit,
    parse_registered_seeds,
    parse_registered_variants,
)
from tools.p11_train_score_count_structure import (
    compute_p11_losses,
    constant_mean_rate_count_nll,
    poisson_count_nll,
    score_label_audit,
    score_rates_to_1x2,
)


def _base_out() -> dict[str, torch.Tensor]:
    return {
        "delta_logits": torch.tensor([[0.1, -0.2, 0.0], [-0.1, 0.2, 0.0]], dtype=torch.float32),
        "goal_diff_logits": torch.randn(2, 7),
        "goal_count_lambdas": torch.tensor([[1.5, 1.0], [0.8, 0.8]], dtype=torch.float32),
    }


def _base_inputs() -> dict[str, torch.Tensor]:
    return {
        "anchor": torch.tensor([[0.50, 0.25, 0.25], [0.30, 0.30, 0.40]], dtype=torch.float32),
        "y_1x2": torch.tensor([0, 1]),
        "y_goal": torch.tensor([4, 3]),
        "home_goals": torch.tensor([2.0, 1.0]),
        "away_goals": torch.tensor([1.0, 1.0]),
    }


def test_p6_default_forward_keys_are_unchanged_and_goal_count_is_opt_in():
    model = P6ResidualPatchITransformer(
        n_features=10,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
    )
    x = torch.randn(3, 10, 10, 2)

    assert set(model(x)) == {"delta_logits", "goal_diff_logits"}

    with_head = P6ResidualPatchITransformer(
        n_features=10,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        enable_goal_count_head=True,
    )
    out = with_head(x)

    assert set(out) == {"delta_logits", "goal_diff_logits", "goal_count_lambdas"}
    assert out["goal_count_lambdas"].shape == (3, 2)
    assert torch.all(out["goal_count_lambdas"] >= 0.05)
    assert torch.all(out["goal_count_lambdas"] <= 5.50)


def test_goal_count_rate_head_bounds_rates():
    head = GoalCountRateHead(hidden_size=4)
    pooled = torch.tensor([[1000.0, -1000.0, 500.0, -500.0]], dtype=torch.float32)

    lambdas = head(pooled)

    assert lambdas.shape == (1, 2)
    assert float(lambdas.min().item()) >= 0.05
    assert float(lambdas.max().item()) <= 5.50


def test_poisson_count_nll_matches_manual_formula():
    lambdas = torch.tensor([[2.0, 1.0]], dtype=torch.float32)
    home = torch.tensor([3.0])
    away = torch.tensor([0.0])

    loss = poisson_count_nll(lambdas, home, away)

    manual_home = 2.0 - 3.0 * math.log(2.0) + math.lgamma(4.0)
    manual_away = 1.0 - 0.0 * math.log(1.0) + math.lgamma(1.0)
    assert loss.item() == pytest.approx(0.5 * (manual_home + manual_away))


def test_score_rates_fold_to_1x2_probs_and_reports_tail_mass():
    lambdas = torch.tensor([[1.2, 0.8], [1.0, 1.0]], dtype=torch.float32)

    folded = score_rates_to_1x2(lambdas, max_goals=10)

    assert folded["p_score_1x2"].shape == (2, 3)
    assert torch.allclose(folded["p_score_1x2"].sum(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.all(folded["tail_mass"] >= 0.0)
    assert float(folded["tail_mass"].max().item()) < 0.01
    assert folded["score_probs"].shape == (2, 11, 11)


def test_p11_zero_weights_matches_p6_loss():
    from tools.p6_train_residual_patch_itransformer import compute_p6_losses

    data = _base_inputs()
    out = _base_out()

    p6 = compute_p6_losses(
        {"delta_logits": out["delta_logits"], "goal_diff_logits": out["goal_diff_logits"]},
        data["anchor"],
        data["y_1x2"],
        data["y_goal"],
        diff_loss_weight=0.30,
        consistency_loss_weight=0.10,
        delta_l2_weight=0.01,
    )
    p11 = compute_p11_losses(
        out,
        data["anchor"],
        data["y_1x2"],
        data["y_goal"],
        data["home_goals"],
        data["away_goals"],
        diff_loss_weight=0.30,
        consistency_loss_weight=0.10,
        delta_l2_weight=0.01,
        count_loss_weight=0.0,
        score_kl_weight=0.0,
    )

    assert p11["loss"].item() == pytest.approx(p6["loss"].item())
    assert p11["L_count"].item() == pytest.approx(0.0)
    assert p11["L_score_1x2_kl"].item() == pytest.approx(0.0)


def test_score_kl_is_detached_from_goal_count_head():
    data = _base_inputs()
    out = _base_out()
    out["delta_logits"] = out["delta_logits"].clone().requires_grad_(True)
    out["goal_count_lambdas"] = out["goal_count_lambdas"].clone().requires_grad_(True)

    losses = compute_p11_losses(
        out,
        data["anchor"],
        data["y_1x2"],
        data["y_goal"],
        data["home_goals"],
        data["away_goals"],
        diff_loss_weight=0.0,
        consistency_loss_weight=0.0,
        delta_l2_weight=0.0,
        count_loss_weight=0.0,
        score_kl_weight=1.0,
    )
    losses["loss"].backward()

    assert out["delta_logits"].grad is not None
    assert out["goal_count_lambdas"].grad is None or torch.all(out["goal_count_lambdas"].grad == 0)


def test_close_game_kl_mask_uses_goal_diff_abs_le_one():
    data = _base_inputs()
    out = _base_out()
    final_logits = torch.log(data["anchor"].clamp_min(1e-8)) + out["delta_logits"]
    score_probs = score_rates_to_1x2(out["goal_count_lambdas"])["p_score_1x2"].detach()
    expected = F.kl_div(F.log_softmax(final_logits[:1], dim=-1), score_probs[:1], reduction="batchmean")

    losses = compute_p11_losses(
        out,
        data["anchor"],
        data["y_1x2"],
        data["y_goal"],
        torch.tensor([2.0, 4.0]),
        torch.tensor([1.0, 1.0]),
        diff_loss_weight=0.0,
        consistency_loss_weight=0.0,
        delta_l2_weight=0.0,
        count_loss_weight=0.0,
        score_kl_weight=1.0,
        close_game_kl_only=True,
    )

    assert losses["score_kl_n"].item() == pytest.approx(1.0)
    assert losses["L_score_1x2_kl"].item() == pytest.approx(expected.item())


def test_score_label_audit_requires_nearly_full_coverage():
    rows = [
        {"label": {"euro_result": "home", "home_goals": 2, "away_goals": 1}},
        {"label": {"euro_result": "draw", "home_goals": 1, "away_goals": 1}},
        {"label": {"euro_result": "away"}},
        {"label": {"euro_result": "void", "home_goals": 0, "away_goals": 0}},
    ]

    audit = score_label_audit(rows)

    assert audit["eligible_1x2_rows"] == 3
    assert audit["valid_score_label_rows"] == 2
    assert audit["score_label_coverage"] == pytest.approx(2 / 3)
    assert audit["passes_min_coverage"] is False


def test_constant_mean_rate_baseline_uses_train_means_on_val_labels():
    train_home = torch.tensor([2.0, 0.0])
    train_away = torch.tensor([0.0, 2.0])
    val_home = torch.tensor([1.0])
    val_away = torch.tensor([1.0])

    nll = constant_mean_rate_count_nll(train_home, train_away, val_home, val_away)

    assert nll == pytest.approx(poisson_count_nll(torch.tensor([[1.0, 1.0]]), val_home, val_away).item())


def test_p11_matrix_has_exact_control_variants_and_seeds():
    assert FIXED_P11_CONTROL == "p11_control_p6_euro_default"
    assert FIXED_P11_SEEDS == [42, 123, 2025]
    assert list(FIXED_P11_VARIANTS) == [
        "p11_a_goal_count_aux_010",
        "p11_b_goal_count_kl_005",
        "p11_c_goal_count_kl_020",
        "p11_d_goal_count_close_kl_010",
    ]


def test_p11_runner_rejects_unregistered_variant_seed_and_test_split(tmp_path):
    with pytest.raises(ValueError, match="Unregistered P11 variant"):
        parse_registered_variants("p11_a_goal_count_aux_010,p11_x")
    with pytest.raises(ValueError, match="Unregistered P11 seed"):
        parse_registered_seeds("42,7")
    with pytest.raises(P11ProtocolError, match="test split"):
        build_protocol_audit("train_match_ids.txt", "test_match_ids.txt", list(FIXED_P11_VARIANTS), FIXED_P11_SEEDS)

    audit = build_protocol_audit("train_match_ids.txt", "val_match_ids.txt", list(FIXED_P11_VARIANTS), FIXED_P11_SEEDS)
    out = tmp_path / "protocol_audit.json"
    out.write_text(json.dumps(audit), encoding="utf-8")
    loaded = json.loads(out.read_text(encoding="utf-8"))

    assert loaded["control_name"] == FIXED_P11_CONTROL
    assert loaded["expected_total_runs"] == 15
    assert loaded["score_count_head_is_optional"] is True
    assert loaded["hard_fail_reasons"] == []
