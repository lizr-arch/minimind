"""Narrow tests for P6 training reports and aggregation."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p6_generate_report import aggregate_p6_reports, p6_verdict, write_markdown_report
from tools.p6_train_residual_patch_itransformer import build_p6_report_payload


def _write_p6_report(
    root,
    name,
    *,
    seed,
    loss,
    ece=0.03,
    draw_recall=0.0,
    argmax_draw_count=0,
    test_ids_used=False,
    validation_fit_used=False,
    posthoc_val_fit_used=False,
    train_samples=26441,
    val_samples=4747,
    run_mode="formal",
):
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "phase": "P6 residual Patch-iTransformer objective probe",
                "run_mode": run_mode,
                "config": {"seed": seed, "feature_groups": "euro"},
                "best_val_logloss": loss,
                "data": {
                    "train_samples": train_samples,
                    "val_samples": val_samples,
                    "test_ids_used": test_ids_used,
                    "validation_fit_used": validation_fit_used,
                    "posthoc_val_fit_used": posthoc_val_fit_used,
                    "anchor_fallback_count": 0,
                    "negative_time_valid_euro_count": 0,
                },
                "val_metrics": {
                    "logloss": loss,
                    "ece": ece,
                    "draw_recall": draw_recall,
                    "argmax_draw_count": argmax_draw_count,
                },
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_p6_report_contains_required_sections_and_no_test_or_val_fit():
    payload = build_p6_report_payload(
        config={"feature_groups": "euro", "seed": 42},
        data={
            "train_samples": 10,
            "val_samples": 4,
            "test_ids_used": False,
            "validation_fit_used": False,
            "posthoc_val_fit_used": False,
        },
        best_epoch=1,
        best_val_logloss=0.94,
        val_metrics={"logloss": 0.94, "ece": 0.03, "draw_recall": 0.0, "argmax_draw_count": 0},
        goal_diff_metrics={"diff_nll": 1.8, "zero_recall": 0.0, "p_from_diff_logloss": 0.95},
        anchor_only_baseline={"anchor_only_val_logloss": 0.943},
        warnings=[],
    )

    assert set(payload) >= {
        "phase",
        "model",
        "config",
        "data",
        "best_epoch",
        "best_val_logloss",
        "anchor_only_baseline",
        "val_metrics",
        "goal_diff_metrics",
        "history",
        "warnings",
    }
    assert payload["data"]["test_ids_used"] is False
    assert payload["data"]["validation_fit_used"] is False
    assert payload["phase"] == "P6 residual Patch-iTransformer objective probe"


def test_p6_aggregate_requires_three_seed_mean_not_best_seed_only(tmp_path):
    root = tmp_path / "p6"
    for seed, loss in [(42, 0.9400), (123, 0.9390), (2025, 0.9410)]:
        run_dir = root / f"p6_euro_default_seed{seed}"
        run_dir.mkdir(parents=True)
        (run_dir / "report.json").write_text(
            json.dumps(
                {
                    "config": {"seed": seed, "feature_groups": "euro"},
                    "best_val_logloss": loss,
                    "data": {"test_ids_used": False, "validation_fit_used": False, "posthoc_val_fit_used": False},
                    "val_metrics": {"logloss": loss, "ece": 0.03, "draw_recall": 0.0},
                }
            ),
            encoding="utf-8",
        )

    payload = aggregate_p6_reports(root, p4_mean_val_logloss=0.9390417536)

    assert payload["variants"][0]["seed_count"] == 3
    assert payload["variants"][0]["mean_val_logloss"] == pytest.approx(0.94)
    assert payload["best_variant"]["mean_val_logloss"] == pytest.approx(0.94)


def test_p6_aggregate_marks_incomplete_when_fewer_than_three_seeds(tmp_path):
    root = tmp_path / "p6"
    run_dir = root / "p6_euro_default_seed42"
    run_dir.mkdir(parents=True)
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "config": {"seed": 42, "feature_groups": "euro"},
                "best_val_logloss": 0.9380,
                "data": {"test_ids_used": False, "validation_fit_used": False, "posthoc_val_fit_used": False},
                "val_metrics": {"logloss": 0.9380, "ece": 0.03, "draw_recall": 0.0},
            }
        ),
        encoding="utf-8",
    )

    payload = aggregate_p6_reports(root, p4_mean_val_logloss=0.9390417536)

    assert payload["variants"][0]["status"] == "incomplete"
    assert "P6_NEEDS_THREE_SEEDS" in payload["verdict"]
    assert "P6_STRONG_PASS" not in payload["verdict"]
    assert "P6_MINIMUM_PASS" not in payload["verdict"]


def test_p6_verdict_red_lights_block_pass_labels():
    verdict = p6_verdict(
        mean_val_logloss=0.9386,
        std_val_logloss=0.0002,
        mean_ece=0.030,
        p4_mean_val_logloss=0.9390417536,
        p4_mean_ece=0.0309487,
        test_ids_used=True,
        validation_fit_used=True,
        seed_count=1,
    )

    assert "P6_FAIL_TEST_READ" in verdict
    assert "P6_FAIL_VAL_FIT" in verdict
    assert "P6_NEEDS_THREE_SEEDS" in verdict
    assert "P6_STRONG_PASS" not in verdict
    assert "P6_MINIMUM_PASS" not in verdict


def test_p6_aggregate_counts_unique_seeds(tmp_path):
    root = tmp_path / "p6"
    for idx, loss in enumerate([0.9389, 0.9390, 0.9391]):
        run_dir = root / f"duplicate_seed42_run{idx}"
        run_dir.mkdir(parents=True)
        (run_dir / "report.json").write_text(
            json.dumps(
                {
                    "config": {"seed": 42, "feature_groups": "euro"},
                    "best_val_logloss": loss,
                    "data": {"test_ids_used": False, "validation_fit_used": False, "posthoc_val_fit_used": False},
                    "val_metrics": {"logloss": loss, "ece": 0.03, "draw_recall": 0.0},
                }
            ),
            encoding="utf-8",
        )

    payload = aggregate_p6_reports(root, p4_mean_val_logloss=0.9390417536)

    assert payload["variants"][0]["seed_count"] == 1
    assert payload["variants"][0]["status"] == "invalid"
    assert "P6_FAIL_DUPLICATE_FORMAL_SEED" in payload["verdict"]
    assert "P6_FAIL_REGRESSION" in payload["verdict"]


def test_p6_report_ignores_smoke_by_default_and_uses_formal_metrics(tmp_path):
    root = tmp_path / "p6"
    formal = [
        (123, 0.9371197224, 0.024781, 0.001087),
        (2025, 0.9374096990, 0.024784, 0.001087),
        (42, 0.9374788404, 0.032911, 0.001087),
    ]
    for seed, loss, ece, draw_recall in formal:
        _write_p6_report(
            root,
            f"p6_euro_default_seed{seed}",
            seed=seed,
            loss=loss,
            ece=ece,
            draw_recall=draw_recall,
            argmax_draw_count=1,
        )
    _write_p6_report(
        root,
        "smoke_cpu_128_seed42",
        seed=42,
        loss=0.9600,
        ece=0.2000,
        draw_recall=0.0,
        run_mode="smoke",
        train_samples=128,
        val_samples=128,
    )

    payload = aggregate_p6_reports(
        root,
        p4_mean_val_logloss=0.9390417536,
        variant="euro_default",
        expected_seeds=[42, 123, 2025],
    )
    variant = payload["best_variant"]

    assert payload["raw_report_count"] == 4
    assert payload["ignored_report_count"] == 1
    assert payload["formal_report_count"] == 3
    assert payload["formal_seed_count"] == 3
    assert payload["ignored_reports"][0]["reason"] == "smoke_run"
    assert variant["formal_report_count"] == 3
    assert variant["seed_count"] == 3
    assert variant["mean_val_logloss"] == pytest.approx(0.9373360872, abs=1e-10)
    assert variant["std_val_logloss"] == pytest.approx(0.0001555752, abs=1e-10)
    assert variant["mean_ece"] == pytest.approx(0.027492, abs=1e-6)
    assert variant["mean_draw_recall"] == pytest.approx(0.001087, abs=1e-6)
    assert "P6_BEATS_P4" in payload["verdict"]
    assert "P6_SEQUENCE_OBJECTIVE_VALIDATED" in payload["verdict"]
    assert "DRAW_STILL_COLLAPSED" in payload["verdict"]


def test_p6_report_duplicate_formal_seed_fails_closed(tmp_path):
    root = tmp_path / "p6"
    for seed, loss in [(42, 0.9374), (123, 0.9371), (2025, 0.9375)]:
        _write_p6_report(root, f"p6_euro_default_seed{seed}", seed=seed, loss=loss)
    _write_p6_report(root, "p6_euro_default_seed42_retry", seed=42, loss=0.9360)

    payload = aggregate_p6_reports(
        root,
        p4_mean_val_logloss=0.9390417536,
        variant="euro_default",
        expected_seeds=[42, 123, 2025],
    )

    assert payload["best_variant"]["duplicate_formal_seeds"] == [42]
    assert payload["best_variant"]["status"] == "invalid"
    assert "P6_FAIL_DUPLICATE_FORMAL_SEED" in payload["verdict"]
    assert "P6_FAIL_REGRESSION" in payload["verdict"]
    assert "P6_STRONG_PASS" not in payload["verdict"]


def test_p6_report_expected_seeds_required(tmp_path):
    root = tmp_path / "p6"
    _write_p6_report(root, "p6_euro_default_seed42", seed=42, loss=0.9374)
    _write_p6_report(root, "p6_euro_default_seed123", seed=123, loss=0.9371)

    payload = aggregate_p6_reports(
        root,
        p4_mean_val_logloss=0.9390417536,
        variant="euro_default",
        expected_seeds=[42, 123, 2025],
    )

    assert payload["best_variant"]["missing_expected_seeds"] == [2025]
    assert "P6_MISSING_EXPECTED_SEEDS" in payload["verdict"]
    assert "P6_NEEDS_THREE_SEEDS" in payload["verdict"]


def test_p6_draw_aux_report_groups_by_lambda_and_aggregates_draw_metrics(tmp_path):
    root = tmp_path / "p6_1"
    for seed, loss, draw_nll, precision in [
        (42, 0.93730, 1.20, 0.40),
        (123, 0.93720, 1.18, 0.50),
        (2025, 0.93725, 1.16, 0.45),
    ]:
        run_dir = root / f"p6_euro_default_draw_aux_001_seed{seed}"
        run_dir.mkdir(parents=True)
        (run_dir / "report.json").write_text(
            json.dumps(
                {
                    "config": {
                        "seed": seed,
                        "feature_groups": "euro",
                        "lambda_draw_bce": 0.01,
                    },
                    "best_val_logloss": loss,
                    "data": {
                        "test_ids_used": False,
                        "validation_fit_used": False,
                        "posthoc_val_fit_used": False,
                    },
                    "val_metrics": {
                        "logloss": loss,
                        "ece": 0.027,
                        "draw_class_nll": draw_nll,
                        "draw_recall": 0.01,
                        "draw_precision": precision,
                        "draw_top2_recall": 0.62,
                        "argmax_draw_count": 9,
                        "mean_p_draw_on_true_draw": 0.29,
                        "mean_p_draw_on_non_draw": 0.25,
                        "classwise_ece_draw": 0.04,
                    },
                }
            ),
            encoding="utf-8",
        )

    payload = aggregate_p6_reports(
        root,
        variant="euro_default_draw_aux_001",
        expected_seeds=[42, 123, 2025],
    )
    variant = payload["best_variant"]

    assert variant["name"] == "euro_default_draw_aux_001"
    assert variant["lambda_draw_bce"] == pytest.approx(0.01)
    assert variant["mean_draw_class_nll"] == pytest.approx(1.18)
    assert variant["mean_draw_precision"] == pytest.approx(0.45)
    assert variant["mean_draw_top2_recall"] == pytest.approx(0.62)
    assert variant["mean_classwise_ece_draw"] == pytest.approx(0.04)


def test_p6_report_ignored_reports_are_listed_in_markdown(tmp_path):
    root = tmp_path / "p6"
    for seed, loss in [(42, 0.9374), (123, 0.9371), (2025, 0.9375)]:
        _write_p6_report(root, f"p6_euro_default_seed{seed}", seed=seed, loss=loss)
    _write_p6_report(root, "smoke_cpu_128_seed42", seed=42, loss=0.9600, run_mode="smoke")
    payload = aggregate_p6_reports(root, variant="euro_default", expected_seeds=[42, 123, 2025])
    out_md = tmp_path / "report.md"

    write_markdown_report(payload, out_md)

    text = out_md.read_text(encoding="utf-8")
    assert "## Ignored Reports" in text
    assert "smoke_run" in text
    assert "formal_report_count" in text


def test_p6_aggregate_missing_audit_flags_fail_closed(tmp_path):
    root = tmp_path / "p6"
    for seed, loss in [(42, 0.9388), (123, 0.9387), (2025, 0.9389)]:
        run_dir = root / f"p6_euro_default_seed{seed}"
        run_dir.mkdir(parents=True)
        (run_dir / "report.json").write_text(
            json.dumps(
                {
                    "config": {"seed": seed, "feature_groups": "euro"},
                    "best_val_logloss": loss,
                    "data": {},
                    "val_metrics": {"logloss": loss, "ece": 0.03, "draw_recall": 0.0},
                }
            ),
            encoding="utf-8",
        )

    payload = aggregate_p6_reports(root, p4_mean_val_logloss=0.9390417536)

    assert payload["test_ids_used"] is True
    assert payload["validation_fit_used"] is True
    assert payload["posthoc_val_fit_used"] is True
    assert "P6_FAIL_TEST_READ" in payload["verdict"]
    assert "P6_FAIL_VAL_FIT" in payload["verdict"]
    assert "P6_FAIL_POSTHOC_VAL_FIT" in payload["verdict"]


def test_p6_verdict_compares_against_p4_baseline():
    verdict = p6_verdict(
        mean_val_logloss=0.9386,
        std_val_logloss=0.0002,
        mean_ece=0.030,
        p4_mean_val_logloss=0.9390417536,
        p4_mean_ece=0.0309487,
        test_ids_used=False,
        validation_fit_used=False,
    )

    assert "P6_STRONG_PASS" in verdict
