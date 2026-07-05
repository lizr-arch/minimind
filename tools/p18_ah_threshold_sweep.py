"""Threshold sweep for P18 AH directional signals.

The matrix-level all-row directional score is intentionally strict, but betting
usage usually needs a confidence gate. This report compares the frozen P18.0
goal-diff AH signal with P18.1/P18.2 AH-head expected-unit signals under
absolute expected-unit thresholds.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p18_ah_cover_diagnostics import build_source_context_rows
from tools.p18_run_ah_cover_aux_matrix import _model_side, expected_upper_units_from_ah_cover_row
from tools.p4_train_residual_goal_diff import load_rows_for_ids, load_split_ids


DEFAULT_THRESHOLDS = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25]
DEFAULT_P18_1_VARIANTS = ["p18_ahw_003", "p18_ahw_010", "p18_ahw_030"]
DEFAULT_P18_2_VARIANTS = ["p18u_unit030", "p18u_side010_unit020", "p18u_ce003_side010_unit020"]
DEFAULT_P18_4_VARIANTS = ["p18d_direct030", "p18d_ce003_direct030"]
AhPair = tuple[float, float, float, float]


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def payoff_for_side(actual_upper_units: float, side: str, upper_water: float, lower_water: float) -> float:
    if side == "upper":
        if actual_upper_units > 0:
            return actual_upper_units * upper_water
        return actual_upper_units
    if side == "lower":
        lower_units = -actual_upper_units
        if lower_units > 0:
            return lower_units * lower_water
        return lower_units
    return 0.0


def _normalize_pair(pair: tuple[float, ...]) -> AhPair:
    expected = float(pair[0])
    actual = float(pair[1])
    upper_water = float(pair[2]) if len(pair) >= 3 else 1.0
    lower_water = float(pair[3]) if len(pair) >= 4 else 1.0
    return expected, actual, upper_water, lower_water


def direction_stats(pairs: list[tuple[float, ...]], threshold: float, total_n: int) -> dict[str, Any]:
    selected = [_normalize_pair(pair) for pair in pairs if abs(float(pair[0])) >= threshold]
    model_units = []
    model_payoffs = []
    hits = []
    upper = 0
    lower = 0
    neutral = 0
    for expected, actual, upper_water, lower_water in selected:
        side = _model_side(expected)
        if side == "upper":
            model_units.append(actual)
            upper += 1
        elif side == "lower":
            model_units.append(-actual)
            lower += 1
        else:
            model_units.append(0.0)
            neutral += 1
        model_payoffs.append(payoff_for_side(actual, side, upper_water, lower_water))
        if actual > 0 and side == "upper":
            hits.append(1.0)
        elif actual < 0 and side == "lower":
            hits.append(1.0)
        elif actual != 0 and side in {"upper", "lower"}:
            hits.append(0.0)
    n = len(selected)
    return {
        "threshold": float(threshold),
        "n": n,
        "coverage": n / total_n if total_n else 0.0,
        "model_side_avg_units": _mean(model_units),
        "model_side_avg_payoff": _mean(model_payoffs),
        "direction_accuracy_ex_push": _mean(hits),
        "upper_pick_rate": upper / n if n else 0.0,
        "lower_pick_rate": lower / n if n else 0.0,
        "neutral_pick_rate": neutral / n if n else 0.0,
    }


def summarize_seed_pairs(model_name: str, rows_by_seed: dict[int, list[tuple[float, ...]]], thresholds: list[float]) -> list[dict[str, Any]]:
    total_n = max((len(rows) for rows in rows_by_seed.values()), default=0)
    out = []
    for threshold in thresholds:
        seed_stats = [direction_stats(rows, threshold, total_n) for rows in rows_by_seed.values()]
        out.append(
            {
                "model": model_name,
                "seed_count": len(seed_stats),
                "threshold": float(threshold),
                "mean_n": _mean([float(row["n"]) for row in seed_stats]),
                "mean_coverage": _mean([float(row["coverage"]) for row in seed_stats]),
                "mean_model_side_avg_units": _mean([float(row["model_side_avg_units"]) for row in seed_stats]),
                "mean_model_side_avg_payoff": _mean([float(row["model_side_avg_payoff"]) for row in seed_stats]),
                "mean_direction_accuracy_ex_push": _mean([float(row["direction_accuracy_ex_push"]) for row in seed_stats]),
                "mean_upper_pick_rate": _mean([float(row["upper_pick_rate"]) for row in seed_stats]),
                "mean_lower_pick_rate": _mean([float(row["lower_pick_rate"]) for row in seed_stats]),
            }
        )
    return out


def load_goal_diff_detail(path: Path) -> dict[int, list[tuple[float, ...]]]:
    rows_by_seed: dict[int, list[tuple[float, ...]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows_by_seed.setdefault(int(row["seed"]), []).append(
                (
                    float(row["expected_upper_units"]),
                    float(row["actual_upper_units"]),
                    float(row["upper_water"]),
                    float(row["lower_water"]),
                )
            )
    return rows_by_seed


def build_context_by_label_idx(data_path: str, val_ids_path: str) -> dict[int, dict[str, Any]]:
    val_rows = load_rows_for_ids(data_path, load_split_ids(val_ids_path))
    return {int(row["label_idx"]): row for row in build_source_context_rows(val_rows)}


def load_ah_head_predictions(root: Path, variant: str, seeds: list[int], context_by_label_idx: dict[int, dict[str, Any]]) -> dict[int, list[tuple[float, ...]]]:
    rows_by_seed: dict[int, list[tuple[float, ...]]] = {}
    for seed in seeds:
        path = root / f"{variant}_seed{seed}" / "val_ah_cover_predictions.csv"
        rows = []
        with path.open("r", encoding="utf-8", newline="") as f:
            for idx, pred in enumerate(csv.DictReader(f)):
                ctx = context_by_label_idx.get(idx)
                if ctx is None:
                    continue
                rows.append(
                    (
                        expected_upper_units_from_ah_cover_row(pred),
                        float(ctx["actual_upper_units"]),
                        float(ctx["upper_water"]),
                        float(ctx["lower_water"]),
                    )
                )
        rows_by_seed[seed] = rows
    return rows_by_seed


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# P18 AH Threshold Sweep", ""]
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        best = max(model_rows, key=lambda row: float(row["mean_model_side_avg_payoff"]))
        lines.append(
            f"- {model}: best_t={best['threshold']:.3f}, "
            f"coverage={best['mean_coverage']:.3f}, "
            f"units={best['mean_model_side_avg_units']:.6f}, "
            f"payoff={best['mean_model_side_avg_payoff']:.6f}, "
            f"dir_acc={best['mean_direction_accuracy_ex_push']:.6f}"
        )
    lines.append("")
    lines.append("Interpretation: higher thresholds are confidence filters, not full-coverage model promotions.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run P18 AH expected-unit threshold sweep")
    parser.add_argument("--data", default="data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl")
    parser.add_argument("--val-ids", default="data/odds_real/splits_v6/val_match_ids.txt")
    parser.add_argument("--goal-diff-detail", default="runs/p18_ah_cover_diagnostics/p18_ah_cover_detail.csv")
    parser.add_argument("--p18-cover-root", default="runs/p18_ah_cover_aux_matrix")
    parser.add_argument("--p18-unit-root", default="runs/p18_ah_unit_aux_matrix")
    parser.add_argument("--p18-direct-root", default="runs/p18_ah_direct_aux_matrix")
    parser.add_argument("--p18-cover-variants", default=",".join(DEFAULT_P18_1_VARIANTS))
    parser.add_argument("--p18-unit-variants", default=",".join(DEFAULT_P18_2_VARIANTS))
    parser.add_argument("--p18-direct-variants", default=",".join(DEFAULT_P18_4_VARIANTS))
    parser.add_argument("--seeds", default="42,123,2025")
    parser.add_argument("--thresholds", default=",".join(str(value) for value in DEFAULT_THRESHOLDS))
    parser.add_argument("--out-dir", default="runs/p18_ah_threshold_sweep")
    parser.add_argument("--no-test", action="store_true")
    return parser


def _parse_csv_strings(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.no_test:
        raise SystemExit("Refusing to run P18 threshold sweep without --no-test")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_csv_ints(args.seeds)
    thresholds = _parse_csv_floats(args.thresholds)
    rows: list[dict[str, Any]] = []
    rows.extend(summarize_seed_pairs("P18.0_goal_diff_baseline", load_goal_diff_detail(Path(args.goal_diff_detail)), thresholds))
    context = build_context_by_label_idx(args.data, args.val_ids)
    for variant in _parse_csv_strings(args.p18_cover_variants):
        rows.extend(summarize_seed_pairs(f"P18.1_{variant}", load_ah_head_predictions(Path(args.p18_cover_root), variant, seeds, context), thresholds))
    for variant in _parse_csv_strings(args.p18_unit_variants):
        rows.extend(summarize_seed_pairs(f"P18.2_{variant}", load_ah_head_predictions(Path(args.p18_unit_root), variant, seeds, context), thresholds))
    for variant in _parse_csv_strings(args.p18_direct_variants):
        rows.extend(summarize_seed_pairs(f"P18.4_{variant}", load_ah_head_predictions(Path(args.p18_direct_root), variant, seeds, context), thresholds))
    write_csv(out_dir / "p18_ah_threshold_sweep.csv", rows)
    (out_dir / "p18_ah_threshold_sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_report_md(out_dir / "p18_ah_threshold_sweep.md", rows)
    print(f"Threshold sweep saved to {out_dir / 'p18_ah_threshold_sweep.md'}")


if __name__ == "__main__":
    main()
