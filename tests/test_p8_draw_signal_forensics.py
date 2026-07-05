import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p8_draw_signal_forensics import (
    FIXED_P8_PROBES,
    FIXED_P8_SEEDS,
    P8ProtocolError,
    build_label_split_audit,
    build_market_prior_row,
    build_p6_vs_market_audit,
    compute_p8_decision_gates,
    devig_1x2,
    draw_rank_metrics,
    parse_registered_probes,
    parse_registered_seeds,
    validate_probe_protocol,
)


def _row(match_id="m1", result="draw", minutes=(100.0, 5.0), odds=((2.0, 3.0, 4.0), (1.8, 3.2, 4.4))):
    return {
        "match_id": match_id,
        "league_id": "league_a",
        "season": "2025",
        "odds_timeline": [
            {
                "minutes_before_kickoff": minute,
                "has_euro": True,
                "euro_h": triplet[0],
                "euro_d": triplet[1],
                "euro_a": triplet[2],
            }
            for minute, triplet in zip(minutes, odds)
        ],
        "label": {"euro_result": result, "home_goals": 1, "away_goals": 1},
    }


def test_devig_probabilities_sum_to_one():
    probs, overround = devig_1x2(2.0, 3.0, 4.0)

    assert sum(probs) == pytest.approx(1.0)
    assert all(0.0 < p < 1.0 for p in probs)
    assert overround == pytest.approx((1 / 2.0) + (1 / 3.0) + (1 / 4.0))


def test_devig_rejects_invalid_odds():
    with pytest.raises(ValueError, match="invalid odds"):
        devig_1x2(2.0, 0.0, 4.0)


def test_market_prior_snapshots_do_not_use_results():
    draw = build_market_prior_row(_row(match_id="same", result="draw"))
    home = build_market_prior_row(_row(match_id="same", result="home"))

    assert draw["q_draw_close"] == pytest.approx(home["q_draw_close"])
    assert draw["y_true"] != home["y_true"]


def test_market_prior_missing_reason_required():
    row = _row()
    row["odds_timeline"] = []

    prior = build_market_prior_row(row)

    assert prior["close_missing_reason"] == "no_valid_pre_kickoff_euro"
    assert prior["q_draw_close"] == ""


def test_market_prior_schema_stable():
    prior = build_market_prior_row(_row())

    assert set(prior) >= {
        "match_id",
        "y_true",
        "q_home_open",
        "q_draw_open",
        "q_away_open",
        "q_home_close",
        "q_draw_close",
        "q_away_close",
        "open_missing_reason",
        "close_missing_reason",
        "favorite_strength",
        "home_away_edge",
        "odds_entropy",
    }


def test_draw_rank_metrics_match_manual_fixture():
    probs = torch.tensor([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.6, 0.1, 0.3]])
    labels = torch.tensor([1, 1, 0])

    metrics = draw_rank_metrics(probs, labels)

    assert metrics["draw_rank1_count"] == 1
    assert metrics["draw_rank2_count"] == 1
    assert metrics["draw_rank3_count"] == 1
    assert metrics["draw_top2"] == pytest.approx(1.0)


def test_draw_margin_metrics_match_manual_fixture():
    probs = torch.tensor([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2]])
    labels = torch.tensor([1, 1])

    metrics = draw_rank_metrics(probs, labels)

    assert metrics["mean_draw_margin_to_top"] == pytest.approx(0.1)


def test_p6_vs_market_audit_required_columns():
    market_rows = [
        build_market_prior_row(_row("m1", "draw")),
        build_market_prior_row(_row("m2", "home")),
    ]
    p6_predictions = [
        {"match_id": "m1", "y_true": 1, "p_home": 0.5, "p_draw": 0.2, "p_away": 0.3},
        {"match_id": "m2", "y_true": 0, "p_home": 0.6, "p_draw": 0.2, "p_away": 0.2},
    ]

    audit = build_p6_vs_market_audit(market_rows, p6_predictions, snapshot="close")

    assert set(audit["overall"]) >= {
        "market_logloss",
        "market_draw_nll",
        "market_draw_recall_argmax",
        "market_draw_top2",
        "p6_vs_market_draw_corr",
        "p6_vs_market_draw_mae",
        "p6_draw_shrinkage_ratio",
    }


def test_label_split_audit_detects_class_mapping_error():
    train = [_row("t1", "draw")]
    val = [_row("v1", "not_a_class")]

    audit = build_label_split_audit(train, val)

    assert audit["class_mapping_integrity"] is False
    assert "invalid_label:v1" in audit["hard_fail_reasons"]


def test_label_split_audit_detects_duplicate_match_ids():
    audit = build_label_split_audit([_row("dup", "home")], [_row("dup", "away")])

    assert audit["duplicate_match_ids"] == ["dup"]
    assert "duplicate_match_ids" in audit["hard_fail_reasons"]


def test_label_split_audit_detects_after_kickoff_events():
    audit = build_label_split_audit([_row("t1", "home", minutes=(-1.0,))], [_row("v1", "away")])

    assert audit["odds_after_kickoff_count"] == 1
    assert "after_kickoff_events" in audit["hard_fail_reasons"]


def test_probe_matrix_exact_probe_ids():
    assert list(FIXED_P8_PROBES) == [
        "probe_0_train_class_prior",
        "probe_1_market_close_de_vig",
        "probe_2_market_open_de_vig",
        "probe_3_train_only_multinomial_logistic_market_features",
        "probe_4_train_only_binary_draw_then_conditional_home_away_logistic",
        "probe_5_train_only_tiny_tabular_mlp_market_features",
    ]


def test_probe_matrix_exact_seeds():
    assert FIXED_P8_SEEDS == [42, 123, 2025]


def test_probe_runner_rejects_test_split():
    with pytest.raises(P8ProtocolError, match="test split"):
        validate_probe_protocol("train.txt", "test_match_ids.txt", list(FIXED_P8_PROBES), FIXED_P8_SEEDS)


def test_probe_runner_rejects_unregistered_probe():
    with pytest.raises(ValueError, match="Unregistered P8 probe"):
        parse_registered_probes("probe_0_train_class_prior,probe_x")


def test_probe_training_uses_train_only():
    protocol = validate_probe_protocol(
        "train_match_ids.txt",
        "val_match_ids.txt",
        list(FIXED_P8_PROBES),
        FIXED_P8_SEEDS,
    )

    assert protocol["official_val_used_for_fitting"] is False
    assert protocol["probe_promoted_to_mainline"] is False


def test_p8_cannot_write_accepted_mainline(tmp_path):
    protocol = validate_probe_protocol(
        "train_match_ids.txt",
        "val_match_ids.txt",
        list(FIXED_P8_PROBES),
        FIXED_P8_SEEDS,
        promote_mainline=True,
    )
    out = tmp_path / "protocol_audit.json"
    out.write_text(json.dumps(protocol), encoding="utf-8")

    assert protocol["probe_promoted_to_mainline"] is True
    assert "probe_promoted_to_mainline" in protocol["hard_fail_reasons"]


def test_p8_decision_gates_recommend_market_prior_residual_when_p6_shrinks_draw():
    gates = compute_p8_decision_gates(
        p6_vs_market={
            "overall": {
                "market_draw_recall_argmax": 0.0,
                "market_draw_rank1_count": 0,
                "market_draw_top2": 0.63,
                "p6_draw_top2": 0.33,
                "mean_p6_minus_market_p_draw_true_draw": -0.05,
                "p6_draw_shrinkage_ratio": 0.78,
            }
        },
        label_audit={"hard_fail_reasons": []},
        probe_summary={"probes": []},
    )

    assert "P8_STOP_DRAW_ARGMAX_PRIMARY_GATE" in gates["verdict"]
    assert gates["recommended_next_phase"] == "P9 Market-Prior Anchored Residual Model"


def test_p8_decision_gates_hard_stop_on_label_split_failure():
    gates = compute_p8_decision_gates(
        p6_vs_market={"overall": {}},
        label_audit={"hard_fail_reasons": ["duplicate_match_ids"]},
        probe_summary={"probes": []},
    )

    assert gates["recommended_next_phase"] == "P9 Data/Split Repair"
    assert "P8_HARD_STOP_DATA_OR_SPLIT_REPAIR" in gates["verdict"]
