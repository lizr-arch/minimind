import csv
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p6_sanity_audit import (
    aggregate_seed_pairs,
    paired_bootstrap,
    pearson_corr,
    topk_contribution_share,
    write_markdown_report,
)


def _write_predictions(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["match_id", "y_true", "p_home", "p_draw", "p_away", "pred_class", "correct"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_report(run_dir, seed, *, test_ids_used=False, validation_fit_used=False):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "config": {"seed": seed},
                "data": {
                    "test_ids_used": test_ids_used,
                    "validation_fit_used": validation_fit_used,
                    "anchor_fallback_count": 0,
                    "negative_time_valid_euro_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )


def test_paired_bootstrap_detects_consistent_improvement():
    result = paired_bootstrap([-0.1, -0.2, -0.05, -0.15], samples=200, seed=7)

    assert result["mean_delta"] == pytest.approx(-0.125)
    assert result["p_candidate_better"] == pytest.approx(1.0)
    assert result["ci95_high"] < 0
    assert result["ci99_high"] < 0


def test_topk_contribution_share_reports_concentration():
    result = topk_contribution_share([-10.0, -1.0, -1.0, 2.0], k=1)

    assert result["topk"] == 1
    assert result["improvement_share"] == pytest.approx(10.0 / 12.0)
    assert result["regression_share"] == pytest.approx(1.0)


def test_sanity_verdict_reports_top20_not_concentrated_for_broad_gain(tmp_path):
    baseline_root = tmp_path / "p4"
    candidate_root = tmp_path / "p6"
    rows_base = [
        {"match_id": f"m{i}", "y_true": 0, "p_home": 0.50, "p_draw": 0.25, "p_away": 0.25, "pred_class": 0, "correct": 1}
        for i in range(100)
    ]
    rows_cand = [
        {"match_id": f"m{i}", "y_true": 0, "p_home": 0.55, "p_draw": 0.25, "p_away": 0.20, "pred_class": 0, "correct": 1}
        for i in range(100)
    ]
    bdir = baseline_root / "p4_euro_default_seed42"
    cdir = candidate_root / "p6_euro_default_seed42"
    _write_report(bdir, 42)
    _write_report(cdir, 42)
    _write_predictions(bdir / "val_predictions.csv", rows_base)
    _write_predictions(cdir / "val_predictions.csv", rows_cand)

    payload = aggregate_seed_pairs(
        baseline_root=baseline_root,
        candidate_root=candidate_root,
        expected_seeds=[42],
        bootstrap_samples=100,
    )

    assert payload["top20_contribution"]["improvement_share"] == pytest.approx(0.2)
    assert "P6_SANITY_TOP20_NOT_CONCENTRATED" in payload["verdict"]


def test_pearson_corr_handles_identical_vectors():
    assert pearson_corr([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]) == pytest.approx(1.0)


def test_aggregate_seed_pairs_uses_match_level_delta_without_tripling_matches(tmp_path):
    baseline_root = tmp_path / "p4"
    candidate_root = tmp_path / "p6"
    rows_base = [
        {"match_id": "m1", "y_true": 0, "p_home": 0.50, "p_draw": 0.25, "p_away": 0.25, "pred_class": 0, "correct": 1},
        {"match_id": "m2", "y_true": 2, "p_home": 0.30, "p_draw": 0.20, "p_away": 0.50, "pred_class": 2, "correct": 1},
    ]
    rows_cand = [
        {"match_id": "m1", "y_true": 0, "p_home": 0.60, "p_draw": 0.20, "p_away": 0.20, "pred_class": 0, "correct": 1},
        {"match_id": "m2", "y_true": 2, "p_home": 0.20, "p_draw": 0.20, "p_away": 0.60, "pred_class": 2, "correct": 1},
    ]
    for seed in [42, 123]:
        bdir = baseline_root / f"p4_euro_default_seed{seed}"
        cdir = candidate_root / f"p6_euro_default_seed{seed}"
        _write_report(bdir, seed)
        _write_report(cdir, seed)
        _write_predictions(bdir / "val_predictions.csv", rows_base)
        _write_predictions(cdir / "val_predictions.csv", rows_cand)

    payload = aggregate_seed_pairs(
        baseline_root=baseline_root,
        candidate_root=candidate_root,
        expected_seeds=[42, 123],
        bootstrap_samples=200,
        bootstrap_seed=11,
    )

    assert payload["match_count"] == 2
    assert payload["seed_count"] == 2
    assert payload["mean_match_delta_nll"] < 0
    assert payload["bootstrap"]["p_candidate_better"] == pytest.approx(1.0)
    assert payload["top20_contribution"]["improvement_share"] is not None
    assert payload["audit"]["test_ids_used"] is False
    assert payload["audit"]["validation_fit_used"] is False


def test_aggregate_seed_pairs_preserves_duplicate_match_ids_by_row_order(tmp_path):
    baseline_root = tmp_path / "p4"
    candidate_root = tmp_path / "p6"
    rows_base = [
        {"match_id": "m1", "y_true": 0, "p_home": 0.50, "p_draw": 0.25, "p_away": 0.25, "pred_class": 0, "correct": 1},
        {"match_id": "m1", "y_true": 2, "p_home": 0.30, "p_draw": 0.20, "p_away": 0.50, "pred_class": 2, "correct": 1},
    ]
    rows_cand = [
        {"match_id": "m1", "y_true": 0, "p_home": 0.60, "p_draw": 0.20, "p_away": 0.20, "pred_class": 0, "correct": 1},
        {"match_id": "m1", "y_true": 2, "p_home": 0.20, "p_draw": 0.20, "p_away": 0.60, "pred_class": 2, "correct": 1},
    ]
    bdir = baseline_root / "p4_euro_default_seed42"
    cdir = candidate_root / "p6_euro_default_seed42"
    _write_report(bdir, 42)
    _write_report(cdir, 42)
    _write_predictions(bdir / "val_predictions.csv", rows_base)
    _write_predictions(cdir / "val_predictions.csv", rows_cand)

    payload = aggregate_seed_pairs(
        baseline_root=baseline_root,
        candidate_root=candidate_root,
        expected_seeds=[42],
        bootstrap_samples=50,
    )

    assert payload["match_count"] == 2
    assert payload["row_pairing"] == "row_order"


def test_aggregate_seed_pairs_fails_closed_on_audit_flags(tmp_path):
    baseline_root = tmp_path / "p4"
    candidate_root = tmp_path / "p6"
    rows = [
        {"match_id": "m1", "y_true": 0, "p_home": 0.50, "p_draw": 0.25, "p_away": 0.25, "pred_class": 0, "correct": 1},
    ]
    bdir = baseline_root / "p4_euro_default_seed42"
    cdir = candidate_root / "p6_euro_default_seed42"
    _write_report(bdir, 42)
    _write_report(cdir, 42, validation_fit_used=True)
    _write_predictions(bdir / "val_predictions.csv", rows)
    _write_predictions(cdir / "val_predictions.csv", rows)

    payload = aggregate_seed_pairs(
        baseline_root=baseline_root,
        candidate_root=candidate_root,
        expected_seeds=[42],
        bootstrap_samples=50,
    )

    assert payload["audit"]["validation_fit_used"] is True
    assert "P6_SANITY_FAIL_VAL_FIT" in payload["verdict"]


def test_markdown_report_includes_bootstrap_and_verdict(tmp_path):
    payload = {
        "match_count": 2,
        "seed_count": 1,
        "mean_match_delta_nll": -0.1,
        "bootstrap": {"ci95_low": -0.2, "ci95_high": -0.05, "p_candidate_better": 1.0},
        "verdict": ["P6_SANITY_BOOTSTRAP_PASS"],
        "seed_pairs": [],
        "audit": {"test_ids_used": False, "validation_fit_used": False},
    }
    out = tmp_path / "audit.md"

    write_markdown_report(payload, out)

    text = out.read_text(encoding="utf-8")
    assert "P6 Sanity Audit" in text
    assert "P6_SANITY_BOOTSTRAP_PASS" in text
    assert "p_candidate_better" in text
