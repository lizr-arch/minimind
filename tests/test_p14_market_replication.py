import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p14_train_market_replication import (
    P14ProtocolError,
    assert_no_test_paths,
    build_epoch_history_row,
    compute_market_replication_metrics,
    compute_p14_losses,
    compute_draw_risk_metrics,
    draw_risk_bucket,
    load_best_model_state,
    p14_selected_feature_names,
    prepare_p14_output_dir,
    select_checkpoint_row,
)


def _toy_out(delta_logits: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "delta_logits": delta_logits,
        "goal_diff_logits": torch.zeros((delta_logits.shape[0], 7), dtype=torch.float32),
    }


def test_p14_feature_groups_include_ou_without_market_present_count():
    names = p14_selected_feature_names("euro,asian,ou")

    assert "over_under_line" in names
    assert "has_ou" in names
    assert "time_log" in names
    assert "bucket_valid_count" in names
    assert "market_present_count" not in names


def test_p14_market_kl_is_zero_when_final_matches_market_anchor():
    anchor = torch.tensor([[0.50, 0.25, 0.25], [0.25, 0.30, 0.45]], dtype=torch.float32)
    labels = torch.tensor([0, 2], dtype=torch.long)
    goal_diff = torch.tensor([4, 2], dtype=torch.long)

    losses = compute_p14_losses(
        _toy_out(torch.zeros_like(anchor)),
        anchor,
        labels,
        goal_diff,
        market_loss_weight=1.0,
        label_loss_weight=0.0,
        diff_loss_weight=0.0,
        consistency_loss_weight=0.0,
        delta_l2_weight=0.0,
    )

    assert losses["L_market_kl"].item() == pytest.approx(0.0, abs=1e-7)
    assert losses["loss"].item() == pytest.approx(0.0, abs=1e-7)


def test_p14_market_kl_penalizes_residual_drift_from_market_anchor():
    anchor = torch.tensor([[0.50, 0.25, 0.25]], dtype=torch.float32)
    labels = torch.tensor([0], dtype=torch.long)
    goal_diff = torch.tensor([4], dtype=torch.long)

    losses = compute_p14_losses(
        _toy_out(torch.tensor([[0.0, -1.0, 1.0]], dtype=torch.float32)),
        anchor,
        labels,
        goal_diff,
        market_loss_weight=1.0,
        label_loss_weight=0.0,
        diff_loss_weight=0.0,
        consistency_loss_weight=0.0,
        delta_l2_weight=0.0,
    )

    assert losses["L_market_kl"].item() > 0.01


def test_draw_risk_bucket_prefers_small_nonzero_ah_and_low_ou():
    assert draw_risk_bucket(0.25, 2.5) == 2
    assert draw_risk_bucket(0.50, 2.25) == 2
    assert draw_risk_bucket(0.0, 2.5) == 1
    assert draw_risk_bucket(1.5, 3.25) == 0


def test_p14_draw_risk_aux_loss_requires_logits_when_enabled():
    anchor = torch.tensor([[0.50, 0.25, 0.25]], dtype=torch.float32)
    labels = torch.tensor([0], dtype=torch.long)
    goal_diff = torch.tensor([4], dtype=torch.long)
    draw_risk = torch.tensor([2], dtype=torch.long)

    with pytest.raises(ValueError, match="draw_risk_logits"):
        compute_p14_losses(
            _toy_out(torch.zeros_like(anchor)),
            anchor,
            labels,
            goal_diff,
            y_draw_risk=draw_risk,
            draw_risk_loss_weight=0.2,
            market_loss_weight=0.0,
            label_loss_weight=0.0,
            diff_loss_weight=0.0,
            consistency_loss_weight=0.0,
            delta_l2_weight=0.0,
        )


def test_p14_draw_risk_aux_loss_is_added_when_enabled():
    anchor = torch.tensor([[0.50, 0.25, 0.25], [0.25, 0.30, 0.45]], dtype=torch.float32)
    labels = torch.tensor([0, 2], dtype=torch.long)
    goal_diff = torch.tensor([4, 2], dtype=torch.long)
    draw_risk = torch.tensor([2, 0], dtype=torch.long)
    out = _toy_out(torch.zeros_like(anchor))
    out["draw_risk_logits"] = torch.tensor([[0.0, 0.0, 3.0], [3.0, 0.0, 0.0]], dtype=torch.float32)

    losses = compute_p14_losses(
        out,
        anchor,
        labels,
        goal_diff,
        y_draw_risk=draw_risk,
        draw_risk_loss_weight=0.2,
        market_loss_weight=0.0,
        label_loss_weight=0.0,
        diff_loss_weight=0.0,
        consistency_loss_weight=0.0,
        delta_l2_weight=0.0,
    )

    assert losses["L_draw_risk"].item() < 0.1
    assert losses["loss"].item() == pytest.approx(0.2 * losses["L_draw_risk"].item())


def test_draw_risk_metrics_report_accuracy_and_counts():
    logits = torch.tensor([[0.0, 0.0, 3.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=torch.float32)
    labels = torch.tensor([2, 0, 0], dtype=torch.long)

    metrics = compute_draw_risk_metrics(logits, labels)

    assert metrics["draw_risk_acc"] == pytest.approx(2 / 3)
    assert metrics["draw_risk_counts"] == {"low": 2, "mid": 0, "high": 1}


def test_market_replication_metrics_capture_draw_gap():
    market = torch.tensor([[0.40, 0.35, 0.25], [0.30, 0.30, 0.40]], dtype=torch.float32)
    final = torch.tensor([[0.50, 0.20, 0.30], [0.35, 0.20, 0.45]], dtype=torch.float32)
    labels = torch.tensor([1, 2], dtype=torch.long)

    metrics = compute_market_replication_metrics(final, market, labels)

    assert metrics["market_kl"] > 0
    assert metrics["market_draw_mae"] == pytest.approx(0.125)
    assert metrics["model_minus_market_p_draw"] == pytest.approx(-0.125)
    assert metrics["market_draw_top2"] == pytest.approx(1.0)
    assert metrics["final_draw_top2"] == pytest.approx(0.0)


def test_p14_epoch_history_row_contains_p16_selection_metrics():
    row = build_epoch_history_row(
        3,
        train_metrics={"loss": 1.2, "L_market_kl": 0.04, "L_draw_risk": 0.0},
        val_metrics={
            "logloss": 0.937,
            "ece": 0.029,
            "draw_class_nll": 1.55,
            "draw_top2_recall": 0.41,
            "mean_p_draw": 0.20,
            "mean_p_draw_on_true_draw": 0.216,
        },
        market_metrics={
            "market_kl": 0.005,
            "market_draw_mae": 0.036,
            "model_minus_market_p_draw": -0.036,
            "market_draw_top2": 0.63,
            "final_mean_p_draw": 0.20,
        },
        goal_diff_metrics={"draw_risk_acc": 0.0},
    )

    assert row["epoch"] == 3
    assert row["val_logloss"] == pytest.approx(0.937)
    assert row["val_ece"] == pytest.approx(0.029)
    assert row["val_draw_class_nll"] == pytest.approx(1.55)
    assert row["val_mean_p_draw"] == pytest.approx(0.20)
    assert row["market_draw_top2"] == pytest.approx(0.63)


def test_prepare_p14_output_dir_refuses_existing_outputs(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "report.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError):
        prepare_p14_output_dir(str(out))


def test_p14_rejects_test_split_paths():
    with pytest.raises(P14ProtocolError, match="test split"):
        assert_no_test_paths(["data/odds_real/splits_v6/test_match_ids.txt"])
    with pytest.raises(P14ProtocolError, match="test split"):
        assert_no_test_paths(["data/odds_real/test/foo.jsonl"])


def test_load_best_model_state_restores_checkpoint_weights(tmp_path):
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.fill_(1.5)
        model.bias.fill_(0.25)
    checkpoint = tmp_path / "best_model.pth"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    with torch.no_grad():
        model.weight.fill_(9.0)
        model.bias.fill_(9.0)

    load_best_model_state(model, checkpoint, torch.device("cpu"))

    assert model.weight.detach().flatten().tolist() == pytest.approx([1.5, 1.5])
    assert model.bias.detach().item() == pytest.approx(0.25)


def test_balanced_checkpoint_selection_prefers_market_kl_under_logloss_ceiling():
    history = [
        {"epoch": 1, "val_logloss": 0.9359, "val_draw_top2": 0.3326, "market_kl": 0.0075},
        {"epoch": 5, "val_logloss": 0.9370, "val_draw_top2": 0.4174, "market_kl": 0.0056},
        {"epoch": 9, "val_logloss": 0.9394, "val_draw_top2": 0.3174, "market_kl": 0.0092},
    ]

    assert select_checkpoint_row(history, "logloss", 0.9375)["epoch"] == 1
    assert select_checkpoint_row(history, "balanced", 0.9375)["epoch"] == 5


def test_balanced_checkpoint_selection_falls_back_to_logloss_when_no_candidate_meets_ceiling():
    history = [
        {"epoch": 1, "val_logloss": 0.9400, "val_draw_top2": 0.50, "market_kl": 0.0010},
        {"epoch": 2, "val_logloss": 0.9380, "val_draw_top2": 0.20, "market_kl": 0.0200},
    ]

    assert select_checkpoint_row(history, "balanced", 0.9375)["epoch"] == 2
