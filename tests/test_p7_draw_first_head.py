import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.p6_residual_patch_itransformer import P6ResidualPatchITransformer
from model.p7_draw_first_patch_itransformer import P7DrawFirstPatchITransformer
from tools.p7_train_draw_first import (
    conditional_home_away_aux_loss,
    compute_p7_losses,
    draw_first_probs,
    factorized_nll,
)


def test_draw_first_probs_sum_to_one():
    probs = draw_first_probs(
        torch.tensor([-2.0, 0.0, 2.0]),
        torch.tensor([-1.0, 0.0, 1.0]),
    )

    assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-6)


def test_draw_first_probs_non_negative():
    probs = draw_first_probs(torch.linspace(-8, 8, 9), torch.linspace(8, -8, 9))

    assert torch.all(probs >= 0.0)
    assert torch.all(probs <= 1.0)


def test_draw_first_no_final_softmax():
    draw_logit = torch.tensor([0.0])
    home_away_logit = torch.tensor([0.0])

    probs = draw_first_probs(draw_logit, home_away_logit)

    assert probs[0, 1].item() == pytest.approx(0.5)
    assert probs[0, 0].item() == pytest.approx(0.25)
    assert probs[0, 2].item() == pytest.approx(0.25)
    assert probs[0, 1].item() != pytest.approx(torch.softmax(torch.zeros(3), dim=0)[1].item())


def test_draw_first_home_away_conditional_split():
    probs = draw_first_probs(torch.tensor([0.0]), torch.tensor([10.0]))

    assert probs[0, 1].item() == pytest.approx(0.5)
    assert probs[0, 0].item() > 0.49
    assert probs[0, 2].item() < 0.01


def test_factorized_nll_handles_home_draw_away():
    probs = torch.tensor(
        [
            [0.80, 0.10, 0.10],
            [0.20, 0.70, 0.10],
            [0.10, 0.20, 0.70],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 2])

    nll = factorized_nll(probs, labels)

    expected = -torch.log(torch.tensor([0.80, 0.70, 0.70])).mean()
    assert nll.item() == pytest.approx(expected.item())


def test_draw_bce_lambda_zero_matches_ce_only_path():
    out = {"draw_logit": torch.tensor([-1.0, 0.5]), "home_away_logit": torch.tensor([0.2, -0.4])}
    labels = torch.tensor([0, 1])

    ce_only = compute_p7_losses(out, labels, lambda_draw_bce=0.0, lambda_cond_ha_aux=0.0)
    draw_zero = compute_p7_losses(out, labels, lambda_draw_bce=0.0, lambda_cond_ha_aux=0.0)

    assert draw_zero["L_draw_bce"].item() == pytest.approx(0.0)
    assert draw_zero["loss"].item() == pytest.approx(ce_only["loss"].item())


def test_conditional_ha_aux_masks_draw_rows():
    logits = torch.tensor([10.0, -10.0, -10.0])
    labels = torch.tensor([0, 1, 2])

    loss = conditional_home_away_aux_loss(logits, labels)
    without_draw = conditional_home_away_aux_loss(logits[[0, 2]], torch.tensor([0, 2]))

    assert loss.item() == pytest.approx(without_draw.item())


def test_p7_model_forward_returns_factorized_logits():
    model = P7DrawFirstPatchITransformer(
        n_features=10,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
    )

    out = model(torch.randn(2, 10, 10, 2))

    assert out["draw_logit"].shape == (2,)
    assert out["home_away_logit"].shape == (2,)
    assert out["head_type"] == "draw_first_factorized"


def test_p6_default_unchanged_without_p7_flag():
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
