"""Run P16 market-replication selection and calibration diagnostics.

P16 does not introduce a new model family. It reuses the P14 training script,
fixes a narrow market-KL weight grid, and evaluates pre-registered checkpoint
selection rules against same-protocol no-market-KL controls.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class P16ProtocolError(RuntimeError):
    pass


FIXED_P16_SEEDS = [42, 123, 2025]
FIXED_P16_CONTROL = "p16_control_no_market_kl"
P16_LAMBDA14 = 0.30
FIXED_P16_VARIANTS: dict[str, dict[str, Any]] = {
    "p16_a_mktkl_050x": {"market_loss_weight": 0.15, "lambda_multiplier": 0.50},
    "p16_b_mktkl_100x": {"market_loss_weight": 0.30, "lambda_multiplier": 1.00},
    "p16_c_mktkl_150x": {"market_loss_weight": 0.45, "lambda_multiplier": 1.50},
    "p16_d_mktkl_200x": {"market_loss_weight": 0.60, "lambda_multiplier": 2.00},
}

PROMOTABLE_SELECTION_RULES = {"best_logloss", "old_balanced", "balanced_v2_relative"}
P16_SELECTION_RULES = ("best_logloss", "old_balanced", "balanced_v2_relative", "draw_safeguard")
P16_DRAW_RISK_SLICES = {
    "ah_abs_le_0_25",
    "ah_abs_le_0_50",
    "ou_le_2_00",
    "ah_abs_le_0_50_and_ou_le_2_50",
    "market_draw_top2",
    "market_draw_top2_and_ou_le_2_50",
    "market_draw_top2_and_ah_abs_le_0_50",
}
P16_ARTIFACTS = [
    "p16_inputs_manifest.json",
    "p16_control_summary.json",
    "p16_weight_grid_summary.csv",
    "p16_checkpoint_selection_ablation.csv",
    "p16_selected_checkpoint_summary.json",
    "p16_calibration_diagnostic.json",
    "p16_slice_stability.csv",
    "p16_market_replication_metrics.json",
    "p16_draw_metrics.json",
    "p16_promotion_decision_matrix.json",
    "p16_report.md",
    "reviewer_report.json",
    "reviewer_report.md",
]


def parse_registered_variants(raw: str) -> list[str]:
    variants = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in variants if item not in FIXED_P16_VARIANTS]
    if unknown:
        raise ValueError(f"Unregistered P16 variant(s): {unknown}")
    if len(set(variants)) != len(variants):
        raise ValueError("Duplicate P16 variants are not allowed")
    return variants


def parse_registered_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    unknown = [item for item in seeds if item not in FIXED_P16_SEEDS]
    if unknown:
        raise ValueError(f"Unregistered P16 seed(s): {unknown}")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate P16 seeds are not allowed")
    return seeds


def _contains_test_path(path: str | Path) -> bool:
    parts = [part.lower() for part in Path(path).parts]
    return any(part == "test" or part.startswith("test_") or part.startswith("test-") for part in parts)


def build_protocol_audit(train_ids: str, val_ids: str, variants: list[str], seeds: list[int]) -> dict[str, Any]:
    if _contains_test_path(train_ids) or _contains_test_path(val_ids):
        raise P16ProtocolError("P16 refuses test split paths")
    hard_fail_reasons = []
    if variants != list(FIXED_P16_VARIANTS):
        hard_fail_reasons.append("P16_FAIL_MATRIX_VARIANT_MUTATION")
    if seeds != FIXED_P16_SEEDS:
        hard_fail_reasons.append("P16_FAIL_SEED_LIST_MUTATION")
    return {
        "phase": "P16",
        "no_test_split_loaded": True,
        "no_test_artifact_written": True,
        "no_new_model_family": True,
        "no_val_fitted_calibration_promotion": True,
        "control_required": True,
        "control_name": FIXED_P16_CONTROL,
        "control_seed_list": seeds,
        "variant_count": len(variants),
        "variant_count_mutation": variants != list(FIXED_P16_VARIANTS),
        "seed_list": seeds,
        "seed_list_mutation": seeds != FIXED_P16_SEEDS,
        "expected_control_runs": len(seeds),
        "expected_candidate_runs": len(variants) * len(seeds),
        "expected_total_runs": len(seeds) + len(variants) * len(seeds),
        "selection_rules": list(P16_SELECTION_RULES),
        "hard_fail_reasons": hard_fail_reasons,
    }


def _row_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    return float(default if value is None else value)


def _copy_selected(row: dict[str, Any], rule: str, diagnostic_only: bool = False, fallback_reason: str | None = None) -> dict[str, Any]:
    selected = dict(row)
    selected["selection_rule"] = rule
    selected["diagnostic_only"] = bool(diagnostic_only)
    if fallback_reason:
        selected["fallback_reason"] = fallback_reason
    return selected


def select_checkpoint_for_rule(history: list[dict[str, Any]], rule: str, control_row: dict[str, Any]) -> dict[str, Any]:
    if not history:
        raise ValueError("Cannot select checkpoint from empty P16 history")
    if rule not in P16_SELECTION_RULES:
        raise ValueError(f"Unknown P16 checkpoint selection rule: {rule}")
    if rule == "best_logloss":
        row = min(
            history,
            key=lambda item: (
                _row_float(item, "val_logloss"),
                _row_float(item, "market_draw_mae"),
                -_row_float(item, "val_draw_top2"),
                int(item.get("epoch", 0)),
            ),
        )
        return _copy_selected(row, rule)
    if rule == "old_balanced":
        eligible = [row for row in history if _row_float(row, "val_logloss") <= 0.9375]
        pool = eligible or history
        row = min(
            pool,
            key=lambda item: (
                _row_float(item, "market_kl") if eligible else _row_float(item, "val_logloss"),
                _row_float(item, "val_logloss"),
                -_row_float(item, "val_draw_top2"),
                int(item.get("epoch", 0)),
            ),
        )
        return _copy_selected(row, rule, fallback_reason=None if eligible else "no_epoch_under_old_ceiling")
    if rule == "balanced_v2_relative":
        logloss_ceiling = _row_float(control_row, "val_logloss") + 0.0003
        ece_ceiling = max(_row_float(control_row, "val_ece") + 0.005, 0.033)
        eligible = [
            row
            for row in history
            if _row_float(row, "val_logloss") <= logloss_ceiling and _row_float(row, "val_ece") <= ece_ceiling
        ]
        if eligible:
            row = min(
                eligible,
                key=lambda item: (
                    _row_float(item, "market_draw_mae"),
                    _row_float(item, "val_logloss"),
                    -_row_float(item, "val_draw_top2"),
                    int(item.get("epoch", 0)),
                ),
            )
            return _copy_selected(row, rule)
        return _copy_selected(select_checkpoint_for_rule(history, "best_logloss", control_row), rule, fallback_reason="no_balanced_v2_eligible")
    logloss_ceiling = _row_float(control_row, "val_logloss") + 0.0008
    eligible = [
        row for row in history if _row_float(row, "val_logloss") <= logloss_ceiling and _row_float(row, "val_ece") <= 0.036
    ]
    if eligible:
        row = min(
            eligible,
            key=lambda item: (
                -_row_float(item, "val_draw_top2"),
                _row_float(item, "val_logloss"),
                _row_float(item, "market_draw_mae"),
                int(item.get("epoch", 0)),
            ),
        )
        return _copy_selected(row, rule, diagnostic_only=True)
    return _copy_selected(
        select_checkpoint_for_rule(history, "best_logloss", control_row),
        rule,
        diagnostic_only=True,
        fallback_reason="no_draw_safeguard_eligible",
    )


def build_calibration_diagnostic(method: str, fit_split: str, logloss: float, ece: float) -> dict[str, Any]:
    promotable = fit_split == "train_cal"
    return {
        "method": method,
        "fit_split": fit_split,
        "logloss": float(logloss),
        "ece": float(ece),
        "promotable": promotable,
        "tag": "PROMOTABLE_TRAIN_CAL_ONLY" if promotable else "DIAGNOSTIC_ONLY_NOT_PROMOTABLE",
    }


def compute_prediction_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "logloss": 0.0,
            "ece": 0.0,
            "draw_class_nll": 0.0,
            "draw_top2": 0.0,
            "mean_p_draw": 0.0,
            "mean_p_draw_true_draw": 0.0,
        }
    losses = []
    draw_losses = []
    draw_top2_hits = []
    draw_probs = []
    true_draw_probs = []
    conf_pred_target = []
    for row in rows:
        y = int(row["y_true"])
        probs = _normalise_triplet([_row_float(row, "p_home"), _row_float(row, "p_draw"), _row_float(row, "p_away")])
        losses.append(-math.log(max(probs[y], 1e-12)))
        draw_probs.append(probs[1])
        pred = max(range(3), key=lambda idx: probs[idx])
        conf_pred_target.append((probs[pred], 1.0 if pred == y else 0.0))
        if y == 1:
            draw_losses.append(-math.log(max(probs[1], 1e-12)))
            top2 = sorted(range(3), key=lambda idx: probs[idx], reverse=True)[:2]
            draw_top2_hits.append(1.0 if 1 in top2 else 0.0)
            true_draw_probs.append(probs[1])
    return {
        "n": len(rows),
        "logloss": statistics.fmean(losses),
        "ece": _ece_from_confidences(conf_pred_target),
        "draw_class_nll": statistics.fmean(draw_losses) if draw_losses else 0.0,
        "draw_top2": statistics.fmean(draw_top2_hits) if draw_top2_hits else 0.0,
        "mean_p_draw": statistics.fmean(draw_probs) if draw_probs else 0.0,
        "mean_p_draw_true_draw": statistics.fmean(true_draw_probs) if true_draw_probs else 0.0,
    }


def build_calibration_diagnostics_for_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    identity = compute_prediction_metrics(rows)
    diagnostics = [
        {
            **build_calibration_diagnostic("identity_no_fit", fit_split="none", logloss=identity["logloss"], ece=identity["ece"]),
            "temperature": 1.0,
            "n": identity["n"],
        }
    ]
    best_t = 1.0
    best_metrics = identity
    for idx in range(10, 61):
        temperature = idx / 20.0
        calibrated = [_temperature_scale_row(row, temperature) for row in rows]
        metrics = compute_prediction_metrics(calibrated)
        if metrics["logloss"] < best_metrics["logloss"]:
            best_t = temperature
            best_metrics = metrics
    diagnostics.append(
        {
            **build_calibration_diagnostic(
                "scalar_temperature_val_grid",
                fit_split="val",
                logloss=best_metrics["logloss"],
                ece=best_metrics["ece"],
            ),
            "temperature": best_t,
            "n": best_metrics["n"],
        }
    )
    return diagnostics


def compute_slice_stability_rows(
    context_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    variant: str,
    selection_rule: str,
    min_decision_n: int = 100,
) -> list[dict[str, Any]]:
    aligned = []
    for context, control, candidate in zip(context_rows, control_rows, candidate_rows):
        item = {"context": context, "control": control, "candidate": candidate}
        aligned.append(item)
    specs = _slice_specs()
    rows = []
    for slice_name, predicate in specs:
        items = [item for item in aligned if predicate(item)]
        control_slice = [item["control"] for item in items]
        candidate_slice = [item["candidate"] for item in items]
        control_metrics = compute_prediction_metrics(control_slice)
        candidate_metrics = compute_prediction_metrics(candidate_slice)
        market_draws = [_market_prob(item["candidate"], 1) for item in items]
        row = {
            "variant": variant,
            "selection_rule": selection_rule,
            "slice": slice_name,
            "n": len(items),
            "support": "decision_grade" if len(items) >= int(min_decision_n) else "low_support",
            "draw_rate": _draw_rate(candidate_slice),
            "control_logloss": control_metrics["logloss"],
            "candidate_logloss": candidate_metrics["logloss"],
            "slice_logloss_delta": candidate_metrics["logloss"] - control_metrics["logloss"],
            "control_draw_top2": control_metrics["draw_top2"],
            "candidate_draw_top2": candidate_metrics["draw_top2"],
            "control_mean_p_draw_true_draw": control_metrics["mean_p_draw_true_draw"],
            "candidate_mean_p_draw_true_draw": candidate_metrics["mean_p_draw_true_draw"],
            "market_mean_p_draw": statistics.fmean(market_draws) if market_draws else 0.0,
            "model_minus_market_p_draw": _model_minus_market(candidate_slice),
            "market_draw_mae": _market_draw_mae(candidate_slice),
        }
        rows.append(row)
    return rows


def evaluate_slice_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failed = []
    checked = 0
    for row in rows:
        if row.get("support") != "decision_grade" or row.get("slice") not in P16_DRAW_RISK_SLICES:
            continue
        checked += 1
        logloss_harm = _row_float(row, "candidate_logloss") - _row_float(row, "control_logloss") > 0.0015
        draw_top2_improved = _row_float(row, "candidate_draw_top2") > _row_float(row, "control_draw_top2")
        true_draw_p_improved = _row_float(row, "candidate_mean_p_draw_true_draw") > _row_float(
            row, "control_mean_p_draw_true_draw"
        )
        if logloss_harm and not (draw_top2_improved or true_draw_p_improved):
            failed.append(str(row.get("slice")))
    return {"slice_gate_pass": not failed, "checked_slices": checked, "failed_slices": failed}


def build_promotion_decision_matrix(candidate: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    failed: list[str] = []
    if candidate.get("selection_rule") not in PROMOTABLE_SELECTION_RULES:
        failed.append("P16_FAIL_DIAGNOSTIC_RULE_NOT_PROMOTABLE")
    if _row_float(candidate, "mean_val_logloss", 999.0) > _row_float(control, "mean_val_logloss", 999.0) + 0.0002:
        failed.append("P16_FAIL_LOGLOSS")
    if _row_float(candidate, "mean_ece", 999.0) > max(_row_float(control, "mean_ece", 999.0) + 0.004, 0.033):
        failed.append("P16_FAIL_ECE")
    if _row_float(candidate, "draw_top2") < _row_float(control, "draw_top2") + 0.035:
        failed.append("P16_FAIL_DRAW_TOP2_DELTA")
    if _row_float(candidate, "draw_top2") < 0.375:
        failed.append("P16_FAIL_DRAW_TOP2_ABS")
    if _row_float(candidate, "mean_p_draw_true_draw") < _row_float(control, "mean_p_draw_true_draw") + 0.014:
        failed.append("P16_FAIL_TRUE_DRAW_P_DRAW_DELTA")
    if _row_float(candidate, "mean_p_draw_true_draw") < 0.212:
        failed.append("P16_FAIL_TRUE_DRAW_P_DRAW_ABS")
    if _row_float(candidate, "market_draw_mae", 999.0) > _row_float(control, "market_draw_mae", 999.0) - 0.012:
        failed.append("P16_FAIL_MARKET_DRAW_MAE")
    model_minus_market = _row_float(candidate, "model_minus_market_p_draw", -999.0)
    if not (-0.045 <= model_minus_market <= -0.015):
        failed.append("P16_FAIL_MARKET_DRAW_GAP_RANGE")
    if _row_float(candidate, "max_seed_logloss_delta", 999.0) > 0.0008:
        failed.append("P16_FAIL_SEED_LOGLOSS_STABILITY")
    if _row_float(candidate, "max_seed_ece", 999.0) > 0.036:
        failed.append("P16_FAIL_SEED_ECE_STABILITY")
    if _row_float(candidate, "draw_top2_stdev", 999.0) > 0.035:
        failed.append("P16_FAIL_DRAW_TOP2_STABILITY")
    if _row_float(candidate, "mean_p_draw_true_draw_stdev", 999.0) > 0.006:
        failed.append("P16_FAIL_TRUE_DRAW_P_DRAW_STABILITY")
    if not bool(candidate.get("slice_gate_pass", False)):
        failed.append("P16_FAIL_SLICE_GATE")
    verdict = "P16_PROMOTE_P14_MARKET_REPLICATION_MAINLINE" if not failed else "P16_RETAIN_P6_MAINLINE"
    return {
        "variant": candidate.get("variant"),
        "selection_rule": candidate.get("selection_rule"),
        "verdict": verdict,
        "failed_gates": failed,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(_row_float(row, key) for row in rows) if rows else 0.0


def _stdev(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.pstdev([_row_float(row, key) for row in rows]) if len(rows) > 1 else 0.0


def _normalise_triplet(values: list[float]) -> list[float]:
    cleaned = [max(float(value), 1e-12) for value in values]
    total = sum(cleaned)
    return [value / total for value in cleaned]


def _ece_from_confidences(conf_pred_target: list[tuple[float, float]], n_bins: int = 10) -> float:
    if not conf_pred_target:
        return 0.0
    ece = 0.0
    total = len(conf_pred_target)
    for bin_idx in range(n_bins):
        lo = bin_idx / n_bins
        hi = (bin_idx + 1) / n_bins
        bucket = [
            (conf, target)
            for conf, target in conf_pred_target
            if (lo <= conf < hi) or (bin_idx == n_bins - 1 and lo <= conf <= hi)
        ]
        if not bucket:
            continue
        avg_conf = statistics.fmean(conf for conf, _ in bucket)
        avg_acc = statistics.fmean(target for _, target in bucket)
        ece += len(bucket) / total * abs(avg_conf - avg_acc)
    return ece


def _temperature_scale_row(row: dict[str, Any], temperature: float) -> dict[str, Any]:
    probs = _normalise_triplet([_row_float(row, "p_home"), _row_float(row, "p_draw"), _row_float(row, "p_away")])
    inv_t = 1.0 / max(float(temperature), 1e-6)
    scaled = [prob**inv_t for prob in probs]
    scaled = _normalise_triplet(scaled)
    return {**row, "p_home": scaled[0], "p_draw": scaled[1], "p_away": scaled[2]}


def _market_prob(row: dict[str, Any], idx: int) -> float:
    keys = ("market_home", "market_draw", "market_away")
    return _normalise_triplet([_row_float(row, key) for key in keys])[idx]


def _model_minus_market(rows: list[dict[str, Any]]) -> float:
    deltas = [_row_float(row, "p_draw") - _market_prob(row, 1) for row in rows]
    return statistics.fmean(deltas) if deltas else 0.0


def _market_draw_mae(rows: list[dict[str, Any]]) -> float:
    deltas = [abs(_row_float(row, "p_draw") - _market_prob(row, 1)) for row in rows]
    return statistics.fmean(deltas) if deltas else 0.0


def _draw_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return statistics.fmean(1.0 if int(row["y_true"]) == 1 else 0.0 for row in rows)


def _market_draw_top2(item: dict[str, Any]) -> bool:
    row = item["candidate"]
    market_probs = [_market_prob(row, 0), _market_prob(row, 1), _market_prob(row, 2)]
    top2 = sorted(range(3), key=lambda idx: market_probs[idx], reverse=True)[:2]
    return 1 in top2


def _context_float(item: dict[str, Any], key: str) -> float | None:
    value = item["context"].get(key)
    if value in ("", None):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _slice_specs() -> list[tuple[str, Any]]:
    return [
        ("overall", lambda item: True),
        ("ah_abs_le_0_25", lambda item: _context_float(item, "asian_line") is not None and abs(_context_float(item, "asian_line")) <= 0.25),
        ("ah_abs_le_0_50", lambda item: _context_float(item, "asian_line") is not None and abs(_context_float(item, "asian_line")) <= 0.50),
        ("ah_abs_ge_1_25", lambda item: _context_float(item, "asian_line") is not None and abs(_context_float(item, "asian_line")) >= 1.25),
        ("ou_le_2_00", lambda item: _context_float(item, "over_under_line") is not None and _context_float(item, "over_under_line") <= 2.00),
        ("ou_ge_3_00", lambda item: _context_float(item, "over_under_line") is not None and _context_float(item, "over_under_line") >= 3.00),
        (
            "ah_abs_le_0_50_and_ou_le_2_50",
            lambda item: _context_float(item, "asian_line") is not None
            and _context_float(item, "over_under_line") is not None
            and abs(_context_float(item, "asian_line")) <= 0.50
            and _context_float(item, "over_under_line") <= 2.50,
        ),
        ("market_draw_top2", _market_draw_top2),
        (
            "market_draw_top2_and_ou_le_2_50",
            lambda item: _market_draw_top2(item)
            and _context_float(item, "over_under_line") is not None
            and _context_float(item, "over_under_line") <= 2.50,
        ),
        (
            "market_draw_top2_and_ah_abs_le_0_50",
            lambda item: _market_draw_top2(item)
            and _context_float(item, "asian_line") is not None
            and abs(_context_float(item, "asian_line")) <= 0.50,
        ),
    ]


def _selected_to_summary_row(run_name: str, seed: int, selected: dict[str, Any], control_selected: dict[str, Any]) -> dict[str, Any]:
    return {
        "run": run_name,
        "seed": seed,
        "selection_rule": selected["selection_rule"],
        "epoch": int(selected.get("epoch", 0)),
        "diagnostic_only": bool(selected.get("diagnostic_only", False)),
        "val_logloss": _row_float(selected, "val_logloss"),
        "val_ece": _row_float(selected, "val_ece"),
        "val_draw_class_nll": _row_float(selected, "val_draw_class_nll"),
        "draw_top2": _row_float(selected, "val_draw_top2"),
        "mean_p_draw_true_draw": _row_float(selected, "val_mean_p_draw_on_true_draw"),
        "market_kl": _row_float(selected, "market_kl"),
        "market_draw_mae": _row_float(selected, "market_draw_mae"),
        "model_minus_market_p_draw": _row_float(selected, "model_minus_market_p_draw"),
        "seed_logloss_delta": _row_float(selected, "val_logloss") - _row_float(control_selected, "val_logloss"),
    }


def aggregate_results(
    out_root: Path,
    variants: list[str],
    seeds: list[int],
    slice_gate_by_variant: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    slice_gate_by_variant = slice_gate_by_variant or {}
    control_by_seed: dict[int, dict[str, Any]] = {}
    control_rows = []
    for seed in seeds:
        report_path = out_root / f"{FIXED_P16_CONTROL}_seed{seed}" / "report.json"
        if not report_path.exists():
            continue
        history = _read_json(report_path).get("history", [])
        selected = select_checkpoint_for_rule(history, "best_logloss", {"val_logloss": 999.0, "val_ece": 999.0})
        control_by_seed[seed] = selected
        control_rows.append(_selected_to_summary_row(FIXED_P16_CONTROL, seed, selected, selected))
    control_summary = summarize_selected_rows(control_rows, FIXED_P16_CONTROL, "best_logloss", seeds)

    ablation_rows: list[dict[str, Any]] = []
    candidate_summaries: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for variant in variants:
        for rule in P16_SELECTION_RULES:
            rows = []
            for seed in seeds:
                report_path = out_root / f"{variant}_seed{seed}" / "report.json"
                control_selected = control_by_seed.get(seed)
                if not report_path.exists() or control_selected is None:
                    continue
                history = _read_json(report_path).get("history", [])
                selected = select_checkpoint_for_rule(history, rule, control_selected)
                row = _selected_to_summary_row(variant, seed, selected, control_selected)
                rows.append(row)
                ablation_rows.append(row)
            summary = summarize_selected_rows(rows, variant, rule, seeds)
            slice_gate = slice_gate_by_variant.get(variant, {"slice_gate_pass": True, "checked_slices": 0, "failed_slices": []})
            summary["slice_gate_pass"] = bool(slice_gate.get("slice_gate_pass", False))
            summary["slice_gate_source"] = "best_logloss_predictions"
            summary["slice_gate_checked_slices"] = int(slice_gate.get("checked_slices", 0))
            summary["slice_gate_failed_slices"] = slice_gate.get("failed_slices", [])
            summary["decision"] = build_promotion_decision_matrix(summary, control_summary)
            candidate_summaries.append(summary)
            decision_rows.append(summary["decision"])
    promotable = [row for row in decision_rows if row["verdict"] == "P16_PROMOTE_P14_MARKET_REPLICATION_MAINLINE"]
    if promotable:
        final_verdict = "P16_PROMOTE_P14_MARKET_REPLICATION_MAINLINE"
    elif control_summary.get("status") != "complete":
        final_verdict = "P16_INCONCLUSIVE"
    else:
        final_verdict = "P16_RETAIN_P6_MAINLINE"
    return {
        "control_summary": control_summary,
        "checkpoint_selection_rows": ablation_rows,
        "candidate_summaries": candidate_summaries,
        "promotion_decisions": decision_rows,
        "verdict": final_verdict,
    }


def summarize_selected_rows(rows: list[dict[str, Any]], variant: str, selection_rule: str, seeds: list[int]) -> dict[str, Any]:
    if not rows:
        return {
            "variant": variant,
            "selection_rule": selection_rule,
            "seed_count": 0,
            "status": "missing",
            "slice_gate_pass": False,
        }
    return {
        "variant": variant,
        "selection_rule": selection_rule,
        "seed_count": len(rows),
        "status": "complete" if len(rows) == len(seeds) else "incomplete",
        "mean_val_logloss": _mean(rows, "val_logloss"),
        "mean_ece": _mean(rows, "val_ece"),
        "draw_class_nll": _mean(rows, "val_draw_class_nll"),
        "draw_top2": _mean(rows, "draw_top2"),
        "draw_top2_stdev": _stdev(rows, "draw_top2"),
        "mean_p_draw_true_draw": _mean(rows, "mean_p_draw_true_draw"),
        "mean_p_draw_true_draw_stdev": _stdev(rows, "mean_p_draw_true_draw"),
        "market_kl": _mean(rows, "market_kl"),
        "market_draw_mae": _mean(rows, "market_draw_mae"),
        "model_minus_market_p_draw": _mean(rows, "model_minus_market_p_draw"),
        "max_seed_logloss_delta": max(_row_float(row, "seed_logloss_delta") for row in rows),
        "max_seed_ece": max(_row_float(row, "val_ece") for row in rows),
        "slice_gate_pass": True,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_val_predictions_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "match_id": row.get("match_id", ""),
                    "y_true": int(row["y_true"]),
                    "p_home": float(row["p_home"]),
                    "p_draw": float(row["p_draw"]),
                    "p_away": float(row["p_away"]),
                }
            )
    return rows


def read_market_replication_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "match_id": row.get("match_id", ""),
                    "y_true": int(row["y_true"]),
                    "p_home": float(row["final_home"]),
                    "p_draw": float(row["final_draw"]),
                    "p_away": float(row["final_away"]),
                    "market_home": float(row["market_home"]),
                    "market_draw": float(row["market_draw"]),
                    "market_away": float(row["market_away"]),
                }
            )
    return rows


def load_val_slice_context(data_path: str, val_ids_path: str, max_samples: int = 0) -> list[dict[str, Any]]:
    from tools.p14_train_market_replication import _latest_pre_kickoff_market_value
    from tools.p4_train_residual_goal_diff import EURO_MAP, load_rows_for_ids, load_split_ids

    val_ids = load_split_ids(val_ids_path)
    rows = load_rows_for_ids(data_path, val_ids)
    if max_samples > 0:
        rows = rows[: min(int(max_samples), len(rows))]
    context = []
    for row_idx, row in enumerate(rows):
        label = row.get("label", {}) or {}
        result = label.get("euro_result")
        if result not in EURO_MAP or "home_goals" not in label or "away_goals" not in label:
            continue
        context.append(
            {
                "row_idx": row_idx,
                "match_id": str(row.get("match_id", row_idx)),
                "asian_line": _latest_pre_kickoff_market_value(row, "asian_line", "has_asian"),
                "over_under_line": _latest_pre_kickoff_market_value(row, "over_under_line", "has_over_under"),
                "league": row.get("league_id") or row.get("league") or row.get("competition_type") or "unknown",
                "season": row.get("season") or "unknown",
            }
        )
    return context


def build_slice_stability_report(out_root: Path, variants: list[str], seeds: list[int], context_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = []
    gate_by_variant: dict[str, dict[str, Any]] = {}
    for variant in variants:
        variant_rows = []
        for seed in seeds:
            control = read_market_replication_csv(out_root / f"{FIXED_P16_CONTROL}_seed{seed}" / "val_market_replication.csv")
            candidate = read_market_replication_csv(out_root / f"{variant}_seed{seed}" / "val_market_replication.csv")
            if not control or not candidate:
                continue
            slice_rows = compute_slice_stability_rows(
                context_rows,
                control,
                candidate,
                variant=variant,
                selection_rule="best_logloss",
            )
            for row in slice_rows:
                row["seed"] = seed
            rows.extend(slice_rows)
            variant_rows.extend(slice_rows)
        gate_by_variant[variant] = evaluate_slice_gate(variant_rows)
    return rows, gate_by_variant


def build_calibration_report(out_root: Path, variants: list[str], seeds: list[int]) -> list[dict[str, Any]]:
    rows = []
    run_names = [FIXED_P16_CONTROL, *variants]
    for run_name in run_names:
        for seed in seeds:
            pred_rows = read_val_predictions_csv(out_root / f"{run_name}_seed{seed}" / "val_predictions.csv")
            for diag in build_calibration_diagnostics_for_predictions(pred_rows):
                rows.append(
                    {
                        "variant": run_name,
                        "seed": seed,
                        "selection_rule": "best_logloss",
                        **diag,
                    }
                )
    return rows


def write_report_md(path: Path, aggregate: dict[str, Any]) -> None:
    lines = ["# P16 Market-Replication Selection Diagnostics", ""]
    lines.append(f"Verdict: `{aggregate.get('verdict')}`")
    lines.append("")
    control = aggregate.get("control_summary", {})
    lines.append("## Same-Protocol Control")
    lines.append(
        f"- {FIXED_P16_CONTROL}: seeds={control.get('seed_count', 0)}, "
        f"logloss={_row_float(control, 'mean_val_logloss'):.6f}, "
        f"ece={_row_float(control, 'mean_ece'):.6f}, "
        f"draw_top2={_row_float(control, 'draw_top2'):.6f}"
    )
    lines.append("")
    lines.append("## Candidate Summaries")
    for row in aggregate.get("candidate_summaries", []):
        lines.append(
            f"- {row.get('variant')} / {row.get('selection_rule')}: seeds={row.get('seed_count', 0)}, "
            f"logloss={_row_float(row, 'mean_val_logloss'):.6f}, "
            f"draw_top2={_row_float(row, 'draw_top2'):.6f}, "
            f"true_draw_p={_row_float(row, 'mean_p_draw_true_draw'):.6f}, "
            f"slice_gate={row.get('slice_gate_pass')}, "
            f"decision={row.get('decision', {}).get('verdict')}"
        )
    lines.append("")
    lines.append("## Diagnostic Notes")
    lines.append("- slice stability is computed from best_logloss prediction CSVs; non-logloss selection rules use this as a diagnostic proxy.")
    lines.append("- val-fitted calibration diagnostics are tagged DIAGNOSTIC_ONLY_NOT_PROMOTABLE.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reviewer_report(out_root: Path, audit: dict[str, Any], aggregate: dict[str, Any]) -> None:
    findings = []
    if audit.get("hard_fail_reasons"):
        findings.append("protocol audit has hard failures")
    if any(
        row.get("selection_rule") == "draw_safeguard"
        and row.get("verdict") == "P16_PROMOTE_P14_MARKET_REPLICATION_MAINLINE"
        for row in aggregate.get("promotion_decisions", [])
    ):
        findings.append("draw_safeguard used for promotion")
    payload = {"findings": findings, "passed": not findings}
    (out_root / "reviewer_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# P16 Reviewer Report", "", f"- passed: `{payload['passed']}`"]
    for finding in findings:
        lines.append(f"- finding: {finding}")
    (out_root / "reviewer_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run P16 market-replication selection diagnostics")
    parser.add_argument("--variants", default=",".join(FIXED_P16_VARIANTS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in FIXED_P16_SEEDS))
    parser.add_argument("--data", default="data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl")
    parser.add_argument("--train-ids", default="data/odds_real/splits_v6/train_match_ids.txt")
    parser.add_argument("--val-ids", default="data/odds_real/splits_v6/val_match_ids.txt")
    parser.add_argument("--out-root", default="runs/p16_market_replication_selection")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-test", action="store_true")
    return parser


def _base_train_args(args: argparse.Namespace) -> list[str]:
    return [
        "--data",
        args.data,
        "--train-ids",
        args.train_ids,
        "--val-ids",
        args.val_ids,
        "--feature-groups",
        "euro,asian,ou",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--d-model",
        "64",
        "--n-heads",
        "4",
        "--n-layers",
        "2",
        "--d-ff",
        "128",
        "--dropout",
        "0.10",
        "--diff-loss-weight",
        "0.30",
        "--consistency-loss-weight",
        "0.10",
        "--delta-l2-weight",
        "0.01",
        "--checkpoint-selection",
        "logloss",
        "--scaling",
        "robust",
        "--device",
        args.device,
    ]


def _run_command(cmd: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _run_p14(args: argparse.Namespace, run_name: str, seed: int, market_loss_weight: float, log_dir: Path, out_root: Path) -> None:
    run_dir = out_root / f"{run_name}_seed{seed}"
    if args.skip_existing and (run_dir / "report.json").exists():
        print(f"SKIP existing {run_dir}", flush=True)
        return
    if (run_dir / "report.json").exists() or (run_dir / "best_model.pth").exists():
        raise SystemExit(f"Refusing to overwrite existing run: {run_dir}")
    cmd = [
        sys.executable,
        "tools/p14_train_market_replication.py",
        *_base_train_args(args),
        "--market-loss-weight",
        str(market_loss_weight),
        "--seed",
        str(seed),
        "--out-dir",
        str(run_dir),
    ]
    if args.max_samples > 0:
        cmd.extend(["--max-samples", str(args.max_samples)])
    print(f"RUN {run_name} seed={seed} market_loss={market_loss_weight} -> {run_dir}", flush=True)
    _run_command(cmd, log_dir / f"{run_name}_seed{seed}.log")
    print(f"DONE {run_name} seed={seed}", flush=True)


def write_aggregate_artifacts(
    out_root: Path,
    audit: dict[str, Any],
    aggregate: dict[str, Any],
    slice_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
) -> None:
    (out_root / "p16_control_summary.json").write_text(json.dumps(aggregate["control_summary"], indent=2), encoding="utf-8")
    write_csv(out_root / "p16_checkpoint_selection_ablation.csv", aggregate["checkpoint_selection_rows"])
    write_csv(out_root / "p16_weight_grid_summary.csv", aggregate["candidate_summaries"])
    (out_root / "p16_selected_checkpoint_summary.json").write_text(json.dumps(aggregate["candidate_summaries"], indent=2), encoding="utf-8")
    calibration = {
        "diagnostics": calibration_rows,
        "note": "P16 does not promote val-fitted calibration.",
    }
    (out_root / "p16_calibration_diagnostic.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    write_csv(out_root / "p16_slice_stability.csv", slice_rows)
    market_rows = [
        {
            "variant": row.get("variant"),
            "selection_rule": row.get("selection_rule"),
            "market_kl": row.get("market_kl"),
            "market_draw_mae": row.get("market_draw_mae"),
            "model_minus_market_p_draw": row.get("model_minus_market_p_draw"),
        }
        for row in aggregate["candidate_summaries"]
    ]
    draw_rows = [
        {
            "variant": row.get("variant"),
            "selection_rule": row.get("selection_rule"),
            "draw_top2": row.get("draw_top2"),
            "draw_class_nll": row.get("draw_class_nll"),
            "mean_p_draw_true_draw": row.get("mean_p_draw_true_draw"),
        }
        for row in aggregate["candidate_summaries"]
    ]
    (out_root / "p16_market_replication_metrics.json").write_text(json.dumps(market_rows, indent=2), encoding="utf-8")
    (out_root / "p16_draw_metrics.json").write_text(json.dumps(draw_rows, indent=2), encoding="utf-8")
    (out_root / "p16_promotion_decision_matrix.json").write_text(json.dumps(aggregate["promotion_decisions"], indent=2), encoding="utf-8")
    write_report_md(out_root / "p16_report.md", aggregate)
    write_reviewer_report(out_root, audit, aggregate)


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.no_test:
        raise SystemExit("Refusing to run P16 without --no-test")
    variants = parse_registered_variants(args.variants)
    seeds = parse_registered_seeds(args.seeds)
    audit = build_protocol_audit(args.train_ids, args.val_ids, variants, seeds)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "p16_inputs_manifest.json").write_text(
        json.dumps(
            {
                "phase": "P16",
                "control": FIXED_P16_CONTROL,
                "variants": variants,
                "seeds": seeds,
                "registry": FIXED_P16_VARIANTS,
                "lambda14": P16_LAMBDA14,
                "artifacts": P16_ARTIFACTS,
                "protocol_audit": audit,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if audit["hard_fail_reasons"]:
        raise SystemExit(f"P16 protocol audit failed: {audit['hard_fail_reasons']}")
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        _run_p14(args, FIXED_P16_CONTROL, seed, 0.0, log_dir, out_root)
    for variant in variants:
        cfg = FIXED_P16_VARIANTS[variant]
        for seed in seeds:
            _run_p14(args, variant, seed, float(cfg["market_loss_weight"]), log_dir, out_root)

    context_rows = load_val_slice_context(args.data, args.val_ids, args.max_samples)
    slice_rows, slice_gate_by_variant = build_slice_stability_report(out_root, variants, seeds, context_rows)
    aggregate = aggregate_results(out_root, variants, seeds, slice_gate_by_variant=slice_gate_by_variant)
    calibration_rows = build_calibration_report(out_root, variants, seeds)
    (out_root / "p16_results_summary.json").write_text(json.dumps(aggregate, indent=2, default=str), encoding="utf-8")
    write_aggregate_artifacts(out_root, audit, aggregate, slice_rows, calibration_rows)
    print(f"P16 verdict: {aggregate['verdict']}")
    print(f"Report saved to {out_root / 'p16_report.md'}")


if __name__ == "__main__":
    main()
