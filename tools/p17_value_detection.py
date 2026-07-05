"""P17 value-detection diagnostics from frozen mainline validation artifacts.

P17 is not a new training phase. It reads validation prediction CSVs and
market-replication CSVs, then asks whether model probability minus no-vig
market probability creates any diagnostic value edge. The reported ROI uses
fair no-vig market odds only; it is not a real-money betting claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


OUTCOMES = ("home", "draw", "away")
MODEL_KEYS = ("p_home", "p_draw", "p_away")
MARKET_KEYS = ("market_home", "market_draw", "market_away")
DEFAULT_THRESHOLDS = (0.00, 0.02, 0.04, 0.06, 0.08, 0.10)
DEFAULT_SLICES = ("all", "ah_zero", "ah_abs_le_0_25", "ah_abs_le_0_50", "ou_le_2_50", "market_draw_top2")
EPS = 1e-12


class P17ProtocolError(RuntimeError):
    pass


def _contains_test_path(path: str | Path) -> bool:
    parts = [part.lower() for part in Path(path).parts]
    return any(part == "test" or part.startswith("test_") or part.startswith("test-") for part in parts)


def assert_no_test_paths(paths: Iterable[str | Path]) -> None:
    for path in paths:
        if path and _contains_test_path(path):
            raise P17ProtocolError(f"P17 refuses test split or test artifact path: {path}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _normalise_triplet(values: Iterable[Any]) -> list[float]:
    probs = [max(_safe_float(value), 0.0) for value in values]
    total = sum(probs)
    if total <= EPS:
        return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    return [value / total for value in probs]


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _stdev(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) >= 2 else 0.0


def _market_draw_rank(row: dict[str, Any]) -> int:
    probs = _normalise_triplet([row.get(key) for key in MARKET_KEYS])
    draw = probs[1]
    return 1 + sum(1 for prob in (probs[0], probs[2]) if prob > draw + 1e-12)


def _matches_slice(row: dict[str, Any], slice_name: str) -> bool:
    if slice_name == "all":
        return True
    if slice_name == "ah_zero":
        return abs(_safe_float(row.get("asian_line"), default=999.0)) <= 1e-9
    if slice_name == "ah_abs_le_0_25":
        return abs(_safe_float(row.get("asian_line"), default=999.0)) <= 0.25 + 1e-9
    if slice_name == "ah_abs_le_0_50":
        return abs(_safe_float(row.get("asian_line"), default=999.0)) <= 0.50 + 1e-9
    if slice_name == "ou_le_2_50":
        value = _safe_float(row.get("over_under_line"), default=999.0)
        return value <= 2.50 + 1e-9
    if slice_name == "market_draw_top2":
        return _market_draw_rank(row) <= 2
    raise ValueError(f"Unknown P17 slice: {slice_name}")


def build_value_bets(rows: list[dict[str, Any]], edge_threshold: float, mode: str = "top_edge") -> list[dict[str, Any]]:
    if mode not in {"top_edge", "all_positive"}:
        raise ValueError(f"Unknown P17 value bet mode: {mode}")
    bets: list[dict[str, Any]] = []
    for row in rows:
        model_probs = _normalise_triplet(row.get(key) for key in MODEL_KEYS)
        market_probs = _normalise_triplet(row.get(key) for key in MARKET_KEYS)
        candidates = []
        for idx, outcome in enumerate(OUTCOMES):
            market_p = market_probs[idx]
            if market_p <= EPS:
                continue
            model_p = model_probs[idx]
            edge_abs = model_p - market_p
            edge_rel = (model_p / market_p) - 1.0
            candidates.append((edge_abs, edge_rel, model_p, idx, outcome, market_p))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        selected = candidates[:1] if mode == "top_edge" else [item for item in candidates if item[0] >= edge_threshold]
        for edge_abs, edge_rel, model_p, idx, outcome, market_p in selected:
            if edge_abs < edge_threshold:
                continue
            y_true = int(row["y_true"])
            hit = 1 if y_true == idx else 0
            fair_odds = 1.0 / market_p
            profit = fair_odds - 1.0 if hit else -1.0
            bet = {
                "match_id": row.get("match_id", ""),
                "y_true": y_true,
                "selected_class": idx,
                "selected_outcome": outcome,
                "model_p": model_p,
                "market_p": market_p,
                "edge_abs": edge_abs,
                "model_edge_rel": edge_rel,
                "fair_odds_no_vig": fair_odds,
                "hit": hit,
                "stake": 1.0,
                "net_profit_no_vig": profit,
            }
            for key in ("asian_line", "over_under_line"):
                if key in row:
                    bet[key] = row[key]
            bets.append(bet)
    return bets


def summarize_value_bets(
    bets: list[dict[str, Any]],
    threshold: float,
    slice_name: str,
    outcome: str = "all",
    eligible_matches: int | None = None,
) -> dict[str, Any]:
    selected = [bet for bet in bets if outcome == "all" or bet.get("selected_outcome") == outcome]
    n_bets = len(selected)
    profits = [_safe_float(bet.get("net_profit_no_vig")) for bet in selected]
    hits = [_safe_float(bet.get("hit")) for bet in selected]
    outcomes = [str(bet.get("selected_outcome", "")) for bet in selected]
    return {
        "threshold": float(threshold),
        "slice": slice_name,
        "outcome": outcome,
        "eligible_matches": int(eligible_matches if eligible_matches is not None else n_bets),
        "n_bets": n_bets,
        "hit_rate": _mean(hits),
        "total_profit_no_vig": float(sum(profits)),
        "roi_no_vig": float(sum(profits) / n_bets) if n_bets else 0.0,
        "mean_edge_abs": _mean([_safe_float(bet.get("edge_abs")) for bet in selected]),
        "mean_model_edge_rel": _mean([_safe_float(bet.get("model_edge_rel")) for bet in selected]),
        "mean_model_p": _mean([_safe_float(bet.get("model_p")) for bet in selected]),
        "mean_market_p": _mean([_safe_float(bet.get("market_p")) for bet in selected]),
        "mean_fair_odds_no_vig": _mean([_safe_float(bet.get("fair_odds_no_vig")) for bet in selected]),
        "selected_home_share": outcomes.count("home") / n_bets if n_bets else 0.0,
        "selected_draw_share": outcomes.count("draw") / n_bets if n_bets else 0.0,
        "selected_away_share": outcomes.count("away") / n_bets if n_bets else 0.0,
    }


def build_threshold_summaries(
    rows: list[dict[str, Any]],
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    slice_names: Iterable[str] = DEFAULT_SLICES,
    mode: str = "top_edge",
) -> list[dict[str, Any]]:
    summaries = []
    for threshold in thresholds:
        for slice_name in slice_names:
            slice_rows = [row for row in rows if _matches_slice(row, slice_name)]
            bets = build_value_bets(slice_rows, edge_threshold=float(threshold), mode=mode)
            summaries.append(
                summarize_value_bets(
                    bets,
                    threshold=float(threshold),
                    slice_name=slice_name,
                    outcome="all",
                    eligible_matches=len(slice_rows),
                )
            )
    return summaries


def build_outcome_summaries(
    rows: list[dict[str, Any]],
    thresholds: Iterable[float],
    slice_names: Iterable[str] = ("all",),
    mode: str = "top_edge",
) -> list[dict[str, Any]]:
    summaries = []
    for threshold in thresholds:
        for slice_name in slice_names:
            slice_rows = [row for row in rows if _matches_slice(row, slice_name)]
            bets = build_value_bets(slice_rows, edge_threshold=float(threshold), mode=mode)
            for outcome in OUTCOMES:
                summaries.append(
                    summarize_value_bets(
                        bets,
                        threshold=float(threshold),
                        slice_name=slice_name,
                        outcome=outcome,
                        eligible_matches=len(slice_rows),
                    )
                )
    return summaries


def aggregate_seed_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (float(row["threshold"]), str(row["slice"]), str(row.get("outcome", "all")))
        grouped.setdefault(key, []).append(row)
    aggregates = []
    for (threshold, slice_name, outcome), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        roi = [_safe_float(row.get("roi_no_vig")) for row in group]
        hit = [_safe_float(row.get("hit_rate")) for row in group]
        n_bets = [_safe_float(row.get("n_bets")) for row in group]
        aggregate = {
            "threshold": threshold,
            "slice": slice_name,
            "outcome": outcome,
            "seed_count": len(group),
            "total_bets": int(sum(n_bets)),
            "mean_n_bets": _mean(n_bets),
            "mean_hit_rate": _mean(hit),
            "mean_roi_no_vig": _mean(roi),
            "stdev_roi_no_vig": _stdev(roi),
        }
        for metric in (
            "eligible_matches",
            "mean_edge_abs",
            "mean_model_edge_rel",
            "mean_model_p",
            "mean_market_p",
            "mean_fair_odds_no_vig",
            "selected_home_share",
            "selected_draw_share",
            "selected_away_share",
        ):
            if any(metric in row for row in group):
                aggregate[f"mean_{metric}"] = _mean([_safe_float(row.get(metric)) for row in group])
        aggregates.append(aggregate)
    return aggregates


def build_p17_report_payload(
    mainline_config: dict[str, Any],
    threshold_summary: list[dict[str, Any]],
    outcome_summary: list[dict[str, Any]],
    source_note: str,
    decision_threshold: float = 0.04,
) -> dict[str, Any]:
    decision_rows = [
        row
        for row in threshold_summary
        if abs(float(row.get("threshold", -1.0)) - float(decision_threshold)) <= 1e-12
        and row.get("slice") == "all"
        and row.get("outcome", "all") == "all"
    ]
    decision = decision_rows[0] if decision_rows else {}
    has_candidate = (
        int(decision.get("seed_count", 0) or 0) >= 2
        and float(decision.get("mean_n_bets", 0.0) or 0.0) >= 20.0
        and float(decision.get("mean_roi_no_vig", 0.0) or 0.0) > 0.0
    )
    verdict = "P17_VALUE_EDGE_CANDIDATE_DIAGNOSTIC_ONLY" if has_candidate else "P17_NO_ROBUST_VALUE_EDGE"
    return {
        "phase": "P17",
        "mainline": {
            "mainline_id": mainline_config.get("mainline_id"),
            "source_run_root": mainline_config.get("source_run_root"),
            "calibration_policy": mainline_config.get("calibration_policy", {}),
        },
        "input_policy": {
            "no_test_split_loaded": True,
            "no_real_money_claim": True,
            "odds_basis": "no-vig market probabilities converted to fair odds",
            "source_note": source_note,
        },
        "threshold_summary": threshold_summary,
        "outcome_summary": outcome_summary,
        "decision_threshold": float(decision_threshold),
        "decision_row": decision,
        "verdict": verdict,
        "next_actions": [
            "Treat positive ROI rows as validation diagnostics only.",
            "Require train-cal or future holdout confirmation before any promotable calibration or staking policy.",
            "If P17 remains interesting, export exact balanced-checkpoint predictions instead of relying on existing prediction CSV proxy artifacts.",
        ],
    }


def read_predictions_csv(path: Path) -> list[dict[str, Any]]:
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


def read_market_csv(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "match_id": row.get("match_id", ""),
                    "y_true": int(row["y_true"]),
                    "market_home": float(row["market_home"]),
                    "market_draw": float(row["market_draw"]),
                    "market_away": float(row["market_away"]),
                }
            )
    return rows


def join_prediction_market_rows(
    predictions: list[dict[str, Any]],
    markets: list[dict[str, Any]],
    context_by_match_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    market_by_id = {str(row["match_id"]): row for row in markets}
    context_by_match_id = context_by_match_id or {}
    joined = []
    for pred in predictions:
        match_id = str(pred["match_id"])
        market = market_by_id.get(match_id)
        if market is None:
            continue
        if int(pred["y_true"]) != int(market["y_true"]):
            raise ValueError(f"Label mismatch while joining P17 rows: {match_id}")
        row = {**market, **pred}
        row.update(context_by_match_id.get(match_id, {}))
        joined.append(row)
    return joined


def load_context_by_match_id(data_path: str, val_ids_path: str) -> dict[str, dict[str, Any]]:
    if not data_path or not val_ids_path or not Path(data_path).exists() or not Path(val_ids_path).exists():
        return {}
    from tools.p16_run_market_replication_selection import load_val_slice_context

    context = load_val_slice_context(data_path, val_ids_path)
    return {str(row["match_id"]): row for row in context}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report_md(path: Path, payload: dict[str, Any]) -> None:
    decision = payload.get("decision_row", {})
    lines = [
        "# P17 Value Detection Diagnostics",
        "",
        f"Verdict: `{payload.get('verdict')}`",
        "",
        "## Scope",
        "- Uses validation artifacts only.",
        "- Converts no-vig market probabilities to fair odds for diagnostics.",
        "- Does not make a real-money betting or staking-policy claim.",
        f"- Source note: `{payload.get('input_policy', {}).get('source_note')}`",
        "",
        "## Decision Row",
        f"- threshold: `{decision.get('threshold')}`",
        f"- mean_n_bets: `{decision.get('mean_n_bets')}`",
        f"- mean_roi_no_vig: `{decision.get('mean_roi_no_vig')}`",
        f"- stdev_roi_no_vig: `{decision.get('stdev_roi_no_vig')}`",
        f"- mean_hit_rate: `{decision.get('mean_hit_rate')}`",
        "",
        "## Top Threshold Rows",
    ]
    ranked = sorted(
        payload.get("threshold_summary", []),
        key=lambda row: (float(row.get("mean_roi_no_vig", 0.0)), float(row.get("mean_n_bets", 0.0))),
        reverse=True,
    )
    for row in ranked[:10]:
        lines.append(
            f"- t={row.get('threshold')} slice={row.get('slice')} bets={row.get('mean_n_bets')} "
            f"roi={row.get('mean_roi_no_vig')} hit={row.get('mean_hit_rate')}"
        )
    lines.extend(["", "## Next Actions"])
    for action in payload.get("next_actions", []):
        lines.append(f"- {action}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run P17 value-detection diagnostics")
    parser.add_argument("--mainline-config", default="configs/oddsmind_mainline.json")
    parser.add_argument("--source-run-root", default=None)
    parser.add_argument("--variant", default="p16_b_mktkl_100x")
    parser.add_argument("--seeds", default="42,123,2025")
    parser.add_argument("--thresholds", default=",".join(f"{value:.2f}" for value in DEFAULT_THRESHOLDS))
    parser.add_argument("--decision-threshold", type=float, default=0.04)
    parser.add_argument("--out-dir", default="runs/p17_value_detection")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--no-test", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.no_test:
        raise SystemExit("Refusing to run P17 without --no-test")
    config = json.loads(Path(args.mainline_config).read_text(encoding="utf-8"))
    train_args = config.get("train_args", {})
    assert_no_test_paths(
        [
            args.mainline_config,
            train_args.get("data", ""),
            train_args.get("train_ids", ""),
            train_args.get("val_ids", ""),
        ]
    )
    source_root = Path(args.source_run_root or config.get("source_run_root", "runs/p16_market_replication_selection"))
    out_dir = Path(args.out_dir)
    if not args.allow_overwrite and ((out_dir / "p17_report.json").exists() or (out_dir / "p17_report.md").exists()):
        raise SystemExit(f"Refusing to overwrite existing P17 report in {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = _parse_float_list(args.thresholds)
    seeds = _parse_int_list(args.seeds)
    context = load_context_by_match_id(str(train_args.get("data", "")), str(train_args.get("val_ids", "")))

    per_seed_thresholds: list[dict[str, Any]] = []
    per_seed_outcomes: list[dict[str, Any]] = []
    manifest = {
        "phase": "P17",
        "variant": args.variant,
        "seeds": seeds,
        "thresholds": thresholds,
        "source_run_root": str(source_root),
        "source_note": "prediction_csv_proxy_for_p16_mainline_value_diagnostics",
        "no_test_split_loaded": True,
    }
    for seed in seeds:
        run_dir = source_root / f"{args.variant}_seed{seed}"
        pred_path = run_dir / "val_predictions.csv"
        market_path = run_dir / "val_market_replication.csv"
        if not pred_path.exists() or not market_path.exists():
            raise SystemExit(f"Missing P17 source artifacts for seed {seed}: {pred_path} / {market_path}")
        joined = join_prediction_market_rows(read_predictions_csv(pred_path), read_market_csv(market_path), context)
        seed_thresholds = build_threshold_summaries(joined, thresholds=thresholds)
        seed_outcomes = build_outcome_summaries(joined, thresholds=[float(args.decision_threshold)])
        decision_bets = build_value_bets(joined, edge_threshold=float(args.decision_threshold))
        for row in seed_thresholds:
            row.update({"seed": seed, "variant": args.variant})
        for row in seed_outcomes:
            row.update({"seed": seed, "variant": args.variant})
        for row in decision_bets:
            row.update({"seed": seed, "variant": args.variant, "threshold": float(args.decision_threshold)})
        write_csv(out_dir / f"p17_value_bets_seed{seed}.csv", decision_bets)
        per_seed_thresholds.extend(seed_thresholds)
        per_seed_outcomes.extend(seed_outcomes)

    threshold_summary = aggregate_seed_summaries(per_seed_thresholds)
    outcome_summary = aggregate_seed_summaries(per_seed_outcomes)
    payload = build_p17_report_payload(
        config,
        threshold_summary,
        outcome_summary,
        source_note=manifest["source_note"],
        decision_threshold=float(args.decision_threshold),
    )
    (out_dir / "p17_inputs_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_csv(out_dir / "p17_threshold_summary_by_seed.csv", per_seed_thresholds)
    write_csv(out_dir / "p17_outcome_summary_by_seed.csv", per_seed_outcomes)
    write_csv(out_dir / "p17_threshold_summary.csv", threshold_summary)
    write_csv(out_dir / "p17_outcome_summary.csv", outcome_summary)
    (out_dir / "p17_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report_md(out_dir / "p17_report.md", payload)
    print(f"P17 verdict: {payload['verdict']}")
    print(f"Report saved to {out_dir / 'p17_report.md'}")


if __name__ == "__main__":
    main()
