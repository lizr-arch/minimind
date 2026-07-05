"""P18.5 agreement-gated Asian-handicap strategy.

This strategy combines the high-confidence P18.1 cover-probability signal with
the broader P18.4 direct-unit signal. It emits an AH side only when both signals
point in the same direction and a fixed expected-unit threshold is passed.
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

from tools.p17_predict_fixture import assert_no_test_paths, load_prediction_rows
from tools.p18_ah_threshold_sweep import build_context_by_label_idx, load_ah_head_predictions, payoff_for_side
from tools.p18_run_ah_cover_aux_matrix import _model_side


DEFAULT_COVER_VARIANTS = ["p18_ahw_003", "p18_ahw_010", "p18_ahw_030"]
DEFAULT_DIRECT_VARIANTS = ["p18d_direct030", "p18d_ce003_direct030"]
DEFAULT_THRESHOLDS = [0.10, 0.15, 0.20, 0.25]
DEFAULT_MODES = ["any", "both"]


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _side(value: float) -> str:
    return _model_side(float(value))


def agreement_decision(cover_units: float, direct_units: float, threshold: float, mode: str = "any") -> str:
    cover_side = _side(cover_units)
    direct_side = _side(direct_units)
    if cover_side not in {"upper", "lower"} or cover_side != direct_side:
        return "hold"
    if mode == "any":
        passed = max(abs(float(cover_units)), abs(float(direct_units))) >= float(threshold)
    elif mode == "both":
        passed = min(abs(float(cover_units)), abs(float(direct_units))) >= float(threshold)
    else:
        raise ValueError(f"Unsupported agreement mode: {mode}")
    return cover_side if passed else "hold"


def agreement_stats(
    cover_pairs: list[tuple[float, ...]],
    direct_pairs: list[tuple[float, ...]],
    threshold: float,
    mode: str = "any",
) -> dict[str, Any]:
    total_n = min(len(cover_pairs), len(direct_pairs))
    model_units = []
    model_payoffs = []
    hits = []
    upper = 0
    lower = 0
    for cover, direct in zip(cover_pairs[:total_n], direct_pairs[:total_n]):
        cover_expected = float(cover[0])
        direct_expected = float(direct[0])
        actual = float(cover[1])
        upper_water = float(cover[2]) if len(cover) >= 3 else 1.0
        lower_water = float(cover[3]) if len(cover) >= 4 else 1.0
        decision = agreement_decision(cover_expected, direct_expected, threshold, mode)
        if decision == "hold":
            continue
        if decision == "upper":
            model_units.append(actual)
            upper += 1
        else:
            model_units.append(-actual)
            lower += 1
        model_payoffs.append(payoff_for_side(actual, decision, upper_water, lower_water))
        if actual > 0 and decision == "upper":
            hits.append(1.0)
        elif actual < 0 and decision == "lower":
            hits.append(1.0)
        elif actual != 0:
            hits.append(0.0)
    n = len(model_units)
    return {
        "threshold": float(threshold),
        "mode": mode,
        "n": n,
        "coverage": n / total_n if total_n else 0.0,
        "model_side_avg_units": _mean(model_units),
        "model_side_avg_payoff": _mean(model_payoffs),
        "direction_accuracy_ex_push": _mean(hits),
        "upper_pick_rate": upper / n if n else 0.0,
        "lower_pick_rate": lower / n if n else 0.0,
    }


def summarize_agreement_by_seed(
    cover_variant: str,
    direct_variant: str,
    cover_by_seed: dict[int, list[tuple[float, ...]]],
    direct_by_seed: dict[int, list[tuple[float, ...]]],
    thresholds: list[float],
    modes: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for mode in modes:
        for threshold in thresholds:
            seed_stats = []
            for seed in sorted(set(cover_by_seed) & set(direct_by_seed)):
                seed_stats.append(agreement_stats(cover_by_seed[seed], direct_by_seed[seed], threshold, mode))
            rows.append(
                {
                    "cover_variant": cover_variant,
                    "direct_variant": direct_variant,
                    "mode": mode,
                    "threshold": float(threshold),
                    "seed_count": len(seed_stats),
                    "mean_n": _mean([float(row["n"]) for row in seed_stats]),
                    "mean_coverage": _mean([float(row["coverage"]) for row in seed_stats]),
                    "mean_model_side_avg_units": _mean([float(row["model_side_avg_units"]) for row in seed_stats]),
                    "mean_model_side_avg_payoff": _mean([float(row["model_side_avg_payoff"]) for row in seed_stats]),
                    "mean_direction_accuracy_ex_push": _mean([float(row["direction_accuracy_ex_push"]) for row in seed_stats]),
                    "mean_upper_pick_rate": _mean([float(row["upper_pick_rate"]) for row in seed_stats]),
                    "mean_lower_pick_rate": _mean([float(row["lower_pick_rate"]) for row in seed_stats]),
                }
            )
    return rows


def combine_fixture_strategy_rows(
    cover_rows: list[dict[str, Any]],
    direct_rows: list[dict[str, Any]],
    threshold: float,
    mode: str = "any",
) -> list[dict[str, Any]]:
    direct_by_key = {(str(row.get("seed")), str(row.get("bookmaker"))): row for row in direct_rows}
    combined = []
    for cover in cover_rows:
        key = (str(cover.get("seed")), str(cover.get("bookmaker")))
        direct = direct_by_key.get(key)
        if direct is None:
            continue
        cover_units = _safe_float(cover.get("expected_upper_units"))
        direct_units = _safe_float(direct.get("expected_upper_units"))
        decision = agreement_decision(cover_units, direct_units, threshold, mode)
        combined_units = (cover_units + direct_units) / 2.0
        row = {
            "seed": int(float(cover.get("seed", 0))),
            "bookmaker": str(cover.get("bookmaker")),
            "cover_expected_upper_units": cover_units,
            "direct_expected_upper_units": direct_units,
            "combined_expected_upper_units": combined_units,
            "agreement_decision": decision,
            "threshold": float(threshold),
            "mode": mode,
        }
        for key_name in ("match_id", "asian_line", "upper_water", "lower_water"):
            if key_name in cover:
                row[key_name] = cover[key_name]
        combined.append(row)
    return combined


def build_fixture_agreement_payload(
    input_rows: list[dict[str, Any]],
    combined_rows: list[dict[str, Any]],
    threshold: float,
    mode: str,
    cover_variant: str,
    direct_variant: str,
) -> dict[str, Any]:
    active = [row for row in combined_rows if row.get("agreement_decision") in {"upper", "lower"}]
    upper_votes = sum(1 for row in active if row.get("agreement_decision") == "upper")
    lower_votes = sum(1 for row in active if row.get("agreement_decision") == "lower")
    if upper_votes and not lower_votes:
        decision = "upper"
    elif lower_votes and not upper_votes:
        decision = "lower"
    else:
        decision = "hold"
    first = input_rows[0]
    return {
        "phase": "P18.5",
        "task": "agreement_gated_ah_strategy",
        "fixture": {
            "match_id": first.get("match_id"),
            "source_match_id": first.get("source_match_id"),
            "home_team": first.get("home_team"),
            "away_team": first.get("away_team"),
            "kickoff_time_utc": first.get("kickoff_time_utc"),
            "label_status": first.get("label_status"),
            "bookmaker_count": len(input_rows),
        },
        "input_policy": {
            "no_test_split_loaded": True,
            "no_label_fabricated": True,
            "not_betting_advice": True,
            "strategy_layer_only_no_new_training": True,
        },
        "cover_variant": cover_variant,
        "direct_variant": direct_variant,
        "strategy": {
            "threshold": float(threshold),
            "mode": mode,
            "decision": decision,
            "active_count": len(active),
            "n_rows": len(combined_rows),
            "mean_cover_expected_upper_units": _mean([_safe_float(row.get("cover_expected_upper_units")) for row in combined_rows]),
            "mean_direct_expected_upper_units": _mean([_safe_float(row.get("direct_expected_upper_units")) for row in combined_rows]),
            "mean_combined_expected_upper_units": _mean([_safe_float(row.get("combined_expected_upper_units")) for row in combined_rows]),
            "upper_votes": upper_votes,
            "lower_votes": lower_votes,
        },
        "rows": combined_rows,
    }


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_validation_report_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# P18.5 Agreement-Gated AH Validation", ""]
    if rows:
        best = max(rows, key=lambda row: float(row["mean_model_side_avg_payoff"]))
        lines.append(
            f"Best: `{best['cover_variant']} + {best['direct_variant']}` "
            f"mode={best['mode']} t={best['threshold']:.3f}, "
            f"coverage={best['mean_coverage']:.3f}, "
            f"payoff={best['mean_model_side_avg_payoff']:.6f}, "
            f"dir_acc={best['mean_direction_accuracy_ex_push']:.6f}"
        )
        lines.append("")
    for row in sorted(rows, key=lambda item: float(item["mean_model_side_avg_payoff"]), reverse=True)[:12]:
        lines.append(
            f"- {row['cover_variant']} + {row['direct_variant']} {row['mode']} t={row['threshold']:.3f}: "
            f"coverage={row['mean_coverage']:.3f}, units={row['mean_model_side_avg_units']:.6f}, "
            f"payoff={row['mean_model_side_avg_payoff']:.6f}, dir_acc={row['mean_direction_accuracy_ex_push']:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_fixture_report_md(path: Path, payload: dict[str, Any]) -> None:
    fixture = payload["fixture"]
    strategy = payload["strategy"]
    lines = [
        "# P18.5 AH Agreement Fixture Strategy",
        "",
        f"- fixture: {fixture.get('home_team')} vs {fixture.get('away_team')}",
        f"- match_id: `{fixture.get('match_id')}`",
        f"- cover_variant: `{payload.get('cover_variant')}`",
        f"- direct_variant: `{payload.get('direct_variant')}`",
        f"- mode: `{strategy['mode']}`",
        f"- threshold: `{strategy['threshold']}`",
        f"- decision: `{strategy['decision']}`",
        f"- active_count: `{strategy['active_count']}` / `{strategy['n_rows']}`",
        f"- mean_cover_units: `{strategy['mean_cover_expected_upper_units']:.6f}`",
        f"- mean_direct_units: `{strategy['mean_direct_expected_upper_units']:.6f}`",
        f"- mean_combined_units: `{strategy['mean_combined_expected_upper_units']:.6f}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_csv_strings(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run P18.5 agreement-gated AH strategy")
    sub = parser.add_subparsers(dest="command", required=True)
    validation = sub.add_parser("validation", help="Evaluate agreement gates on validation artifacts")
    validation.add_argument("--data", default="data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl")
    validation.add_argument("--val-ids", default="data/odds_real/splits_v6/val_match_ids.txt")
    validation.add_argument("--cover-root", default="runs/p18_ah_cover_aux_matrix")
    validation.add_argument("--direct-root", default="runs/p18_ah_direct_aux_matrix")
    validation.add_argument("--cover-variants", default=",".join(DEFAULT_COVER_VARIANTS))
    validation.add_argument("--direct-variants", default=",".join(DEFAULT_DIRECT_VARIANTS))
    validation.add_argument("--seeds", default="42,123,2025")
    validation.add_argument("--thresholds", default=",".join(str(value) for value in DEFAULT_THRESHOLDS))
    validation.add_argument("--modes", default=",".join(DEFAULT_MODES))
    validation.add_argument("--out-dir", default="runs/p18_ah_agreement_strategy")
    validation.add_argument("--no-test", action="store_true")

    fixture = sub.add_parser("fixture", help="Combine existing fixture strategy CSVs")
    fixture.add_argument("--input", required=True)
    fixture.add_argument("--cover-csv", required=True)
    fixture.add_argument("--direct-csv", required=True)
    fixture.add_argument("--out-dir", required=True)
    fixture.add_argument("--cover-variant", default="p18_ahw_010")
    fixture.add_argument("--direct-variant", default="p18d_ce003_direct030")
    fixture.add_argument("--threshold", type=float, default=0.20)
    fixture.add_argument("--mode", default="any", choices=("any", "both"))
    fixture.add_argument("--allow-overwrite", action="store_true")
    fixture.add_argument("--no-test", action="store_true")
    return parser


def run_validation(args: argparse.Namespace) -> None:
    if not args.no_test:
        raise SystemExit("Refusing to run P18.5 validation without --no-test")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_csv_ints(args.seeds)
    thresholds = _parse_csv_floats(args.thresholds)
    modes = _parse_csv_strings(args.modes)
    context = build_context_by_label_idx(args.data, args.val_ids)
    rows = []
    for cover_variant in _parse_csv_strings(args.cover_variants):
        cover_by_seed = load_ah_head_predictions(Path(args.cover_root), cover_variant, seeds, context)
        for direct_variant in _parse_csv_strings(args.direct_variants):
            direct_by_seed = load_ah_head_predictions(Path(args.direct_root), direct_variant, seeds, context)
            rows.extend(summarize_agreement_by_seed(cover_variant, direct_variant, cover_by_seed, direct_by_seed, thresholds, modes))
    write_csv(out_dir / "p18_ah_agreement_summary.csv", rows)
    (out_dir / "p18_ah_agreement_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_validation_report_md(out_dir / "p18_ah_agreement_report.md", rows)
    best = max(rows, key=lambda row: float(row["mean_model_side_avg_payoff"])) if rows else {}
    print(
        "P18.5 agreement best: "
        f"{best.get('cover_variant')} + {best.get('direct_variant')} "
        f"mode={best.get('mode')} t={best.get('threshold')} "
        f"payoff={best.get('mean_model_side_avg_payoff')}"
    )
    print(f"Report saved to {out_dir / 'p18_ah_agreement_report.md'}")


def run_fixture(args: argparse.Namespace) -> None:
    if not args.no_test:
        raise SystemExit("Refusing to run P18.5 fixture strategy without --no-test")
    assert_no_test_paths([args.input, args.out_dir])
    out_dir = Path(args.out_dir)
    if not args.allow_overwrite and ((out_dir / "ah_agreement_report.json").exists() or (out_dir / "ah_agreement_report.md").exists()):
        raise SystemExit(f"Refusing to overwrite existing P18.5 AH agreement outputs in {out_dir}")
    input_rows = load_prediction_rows(Path(args.input))
    combined = combine_fixture_strategy_rows(load_csv_rows(Path(args.cover_csv)), load_csv_rows(Path(args.direct_csv)), args.threshold, args.mode)
    payload = build_fixture_agreement_payload(input_rows, combined, args.threshold, args.mode, args.cover_variant, args.direct_variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "ah_agreement_by_seed.csv", combined)
    (out_dir / "ah_agreement_report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_fixture_report_md(out_dir / "ah_agreement_report.md", payload)
    print(f"P18.5 fixture agreement decision: {payload['strategy']['decision']}")
    print(f"Report saved to {out_dir / 'ah_agreement_report.md'}")


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "validation":
        run_validation(args)
    elif args.command == "fixture":
        run_fixture(args)


if __name__ == "__main__":
    main()
