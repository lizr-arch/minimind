import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p18_train_ah_cover_aux import (
    AH_COVER_LABELS,
    ah_cover_label_from_row,
    compute_ah_cover_metrics,
    compute_p18_losses,
)


def _row(home_goals: int, away_goals: int, asian_line: float | None) -> dict:
    event = (
        {"minutes_before_kickoff": 10, "asian_line": asian_line, "upper_water": 0.90, "lower_water": 1.00, "asian_source": "raw_update"}
        if asian_line is not None
        else {"minutes_before_kickoff": 10, "asian_line": 0.0, "upper_water": 0.0, "lower_water": 0.0, "asian_source": "missing"}
    )
    return {
        "label": {"euro_result": "home", "home_goals": home_goals, "away_goals": away_goals},
        "odds_timeline": [event],
    }


def _toy_out(batch: int) -> dict[str, torch.Tensor]:
    return {
        "delta_logits": torch.zeros((batch, 3), dtype=torch.float32),
        "goal_diff_logits": torch.zeros((batch, 7), dtype=torch.float32),
        "ah_cover_logits": torch.zeros((batch, 5), dtype=torch.float32),
        "ah_unit_pred": torch.zeros((batch,), dtype=torch.float32),
    }


def test_p18_ah_cover_label_from_row_uses_five_class_settlement_and_missing_ignore():
    assert AH_COVER_LABELS == ["upper_full_win", "upper_half_win", "push", "upper_half_loss", "upper_full_loss"]
    assert ah_cover_label_from_row(_row(3, 0, -2.25)) == 0
    assert ah_cover_label_from_row(_row(2, 0, -2.25)) == 3
    assert ah_cover_label_from_row(_row(1, 1, 0.0)) == 2
    assert ah_cover_label_from_row(_row(1, 0, None)) == -100


def test_p18_loss_requires_ah_logits_when_weight_enabled():
    anchor = torch.tensor([[0.50, 0.25, 0.25]], dtype=torch.float32)
    labels = torch.tensor([0], dtype=torch.long)
    goal_diff = torch.tensor([4], dtype=torch.long)
    ah = torch.tensor([0], dtype=torch.long)
    out = _toy_out(1)
    out.pop("ah_cover_logits")

    with pytest.raises(ValueError, match="ah_cover_logits"):
        compute_p18_losses(
            out,
            anchor,
            labels,
            goal_diff,
            ah,
            ah_cover_loss_weight=0.2,
            market_loss_weight=0.0,
            label_loss_weight=0.0,
            diff_loss_weight=0.0,
            consistency_loss_weight=0.0,
            delta_l2_weight=0.0,
        )


def test_p18_loss_adds_ah_aux_loss_and_ignores_missing_labels():
    anchor = torch.tensor([[0.50, 0.25, 0.25], [0.40, 0.30, 0.30]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    goal_diff = torch.tensor([4, 3], dtype=torch.long)
    ah = torch.tensor([0, -100], dtype=torch.long)
    out = _toy_out(2)
    out["ah_cover_logits"] = torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 5.0, 0.0]], dtype=torch.float32)

    losses = compute_p18_losses(
        out,
        anchor,
        labels,
        goal_diff,
        ah,
        ah_cover_loss_weight=0.2,
        market_loss_weight=0.0,
        label_loss_weight=0.0,
        diff_loss_weight=0.0,
        consistency_loss_weight=0.0,
        delta_l2_weight=0.0,
    )

    assert losses["L_ah_cover"].item() < 0.05
    assert losses["loss"].item() == pytest.approx(0.2 * losses["L_ah_cover"].item())


def test_p18_loss_can_use_unit_and_side_aligned_objectives():
    anchor = torch.tensor([[0.50, 0.25, 0.25], [0.40, 0.30, 0.30]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    goal_diff = torch.tensor([4, 3], dtype=torch.long)
    ah = torch.tensor([0, -100], dtype=torch.long)
    out = _toy_out(2)
    out["ah_cover_logits"] = torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 5.0, 0.0]], dtype=torch.float32)

    losses = compute_p18_losses(
        out,
        anchor,
        labels,
        goal_diff,
        ah,
        ah_cover_loss_weight=0.0,
        ah_unit_loss_weight=0.5,
        ah_side_loss_weight=0.25,
        market_loss_weight=0.0,
        label_loss_weight=0.0,
        diff_loss_weight=0.0,
        consistency_loss_weight=0.0,
        delta_l2_weight=0.0,
    )

    assert losses["L_ah_unit"].item() < 0.01
    assert losses["L_ah_side"].item() < 0.02
    assert losses["loss"].item() == pytest.approx(0.5 * losses["L_ah_unit"].item() + 0.25 * losses["L_ah_side"].item())


def test_p18_loss_can_use_direct_unit_head():
    anchor = torch.tensor([[0.50, 0.25, 0.25], [0.40, 0.30, 0.30]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    goal_diff = torch.tensor([4, 3], dtype=torch.long)
    ah = torch.tensor([0, 4], dtype=torch.long)
    out = _toy_out(2)
    out["ah_unit_pred"] = torch.tensor([1.0, -1.0], dtype=torch.float32)

    losses = compute_p18_losses(
        out,
        anchor,
        labels,
        goal_diff,
        ah,
        ah_cover_loss_weight=0.0,
        ah_direct_unit_loss_weight=0.7,
        market_loss_weight=0.0,
        label_loss_weight=0.0,
        diff_loss_weight=0.0,
        consistency_loss_weight=0.0,
        delta_l2_weight=0.0,
    )

    assert losses["L_ah_direct_unit"].item() == pytest.approx(0.0)
    assert losses["loss"].item() == pytest.approx(0.0)


def test_p18_ah_cover_metrics_respect_ignore_index():
    logits = torch.tensor(
        [
            [4.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 4.0],
            [0.0, 0.0, 4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 4, 2, -100], dtype=torch.long)

    metrics = compute_ah_cover_metrics(logits, labels)

    assert metrics["ah_cover_n"] == 3
    assert metrics["ah_cover_acc"] == pytest.approx(1.0)
    assert metrics["ah_side_acc"] == pytest.approx(1.0)
    assert metrics["ah_unit_mae"] < 0.1
    assert metrics["ah_cover_counts"] == {"upper_full_win": 1, "upper_half_win": 0, "push": 1, "upper_half_loss": 0, "upper_full_loss": 1}
