"""Run P19 rolling AH backtest folds.

The runner is intentionally narrow. It only uses the frozen P19 registry, trains
the already-existing P18 auxiliary objectives inside each fold, exports frozen
checkpoint predictions for selection/forward windows, and replays the agreement
strategy without touching test artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p18_ah_agreement_strategy import agreement_decision
from tools.p18_ah_threshold_sweep import payoff_for_side
from tools.p18_run_ah_cover_aux_matrix import FIXED_P18_VARIANTS, expected_upper_units_from_ah_cover_row
from tools.p19_rolling_backtest import (
    P19ProtocolError,
    assert_no_test_paths,
    build_p19_verdict_payload,
    fixture_collapsed_score,
    write_csv,
    write_json,
)
from tools.p4_train_residual_goal_diff import load_rows_for_ids, load_split_ids
from tools.p18_ah_robustness_slices import load_contexts_with_metadata


class P19RunProtocolError(P19ProtocolError):
    pass


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _csv_strings(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def load_registry(path: str | Path) -> dict[str, Any]:
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    if not registry.get("policy", {}).get("no_test_set", False):
        raise P19RunProtocolError("P19 registry must set policy.no_test_set=true")
    return registry


def build_train_command(
    data_path: str,
    train_ids: str | Path,
    val_ids: str | Path,
    variant: str,
    seed: int,
    out_dir: str | Path,
    epochs: int,
    batch_size: int,
    device: str,
    max_samples: int = 0,
    allow_overwrite: bool = False,
) -> list[str]:
    assert_no_test_paths([data_path, train_ids, val_ids, out_dir])
    if variant not in FIXED_P18_VARIANTS:
        raise P19RunProtocolError(f"Unregistered P18 variant for P19: {variant}")
    cfg = FIXED_P18_VARIANTS[variant]
    cmd = [
        sys.executable,
        "tools/p18_train_ah_cover_aux.py",
        "--data",
        str(data_path),
        "--train-ids",
        str(train_ids),
        "--val-ids",
        str(val_ids),
        "--epochs",
        str(int(epochs)),
        "--batch-size",
        str(int(batch_size)),
        "--device",
        str(device),
        "--seed",
        str(int(seed)),
        "--ah-cover-loss-weight",
        str(float(cfg.get("ah_cover_loss_weight", 0.0))),
        "--out-dir",
        str(out_dir),
    ]
    if "ah_unit_loss_weight" in cfg:
        cmd.extend(["--ah-unit-loss-weight", str(float(cfg["ah_unit_loss_weight"]))])
    if "ah_side_loss_weight" in cfg:
        cmd.extend(["--ah-side-loss-weight", str(float(cfg["ah_side_loss_weight"]))])
    if "ah_direct_unit_loss_weight" in cfg:
        cmd.extend(["--ah-direct-unit-loss-weight", str(float(cfg["ah_direct_unit_loss_weight"]))])
    if int(max_samples) > 0:
        cmd.extend(["--max-samples", str(int(max_samples))])
    if allow_overwrite:
        cmd.append("--allow-overwrite")
    return cmd


def build_export_command(
    data_path: str,
    ids_path: str | Path,
    checkpoint_dir: str | Path,
    out_dir: str | Path,
    split_name: str,
    device: str,
    batch_size: int = 256,
    allow_overwrite: bool = False,
) -> list[str]:
    assert_no_test_paths([data_path, ids_path, checkpoint_dir, out_dir])
    cmd = [
        sys.executable,
        "tools/p19_export_ah_predictions.py",
        "--data",
        str(data_path),
        "--ids",
        str(ids_path),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--out-dir",
        str(out_dir),
        "--split-name",
        str(split_name),
        "--device",
        str(device),
        "--batch-size",
        str(int(batch_size)),
    ]
    if allow_overwrite:
        cmd.append("--allow-overwrite")
    return cmd


def run_command(cmd: list[str], log_path: str | Path, skip_existing_path: str | Path | None = None) -> None:
    if skip_existing_path and Path(skip_existing_path).exists():
        return
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(log_path).open("w", encoding="utf-8") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def load_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_export_pairs(prediction_csv: str | Path, contexts: list[dict[str, Any]]) -> list[tuple[float, float, float, float]]:
    context_by_idx = {int(row["label_idx"]): row for row in contexts if "label_idx" in row}
    rows = []
    for idx, pred in enumerate(load_csv_rows(prediction_csv)):
        ctx = context_by_idx.get(idx)
        if ctx is None:
            continue
        rows.append(
            (
                expected_upper_units_from_ah_cover_row(pred),
                _safe_float(ctx.get("actual_upper_units")),
                _safe_float(ctx.get("upper_water"), 1.0),
                _safe_float(ctx.get("lower_water"), 1.0),
            )
        )
    return rows


def replay_agreement_candidates(
    cover_pairs: list[tuple[float, float, float, float]],
    direct_pairs: list[tuple[float, float, float, float]],
    contexts: list[dict[str, Any]],
    thresholds: list[float],
    mode: str,
    seed: int,
    split_name: str,
) -> list[dict[str, Any]]:
    out = []
    total_n = min(len(cover_pairs), len(direct_pairs), len(contexts))
    for threshold in thresholds:
        detail_rows = []
        for idx in range(total_n):
            cover = cover_pairs[idx]
            direct = direct_pairs[idx]
            ctx = contexts[idx]
            cover_expected = float(cover[0])
            direct_expected = float(direct[0])
            actual = float(cover[1])
            upper_water = float(cover[2])
            lower_water = float(cover[3])
            decision = agreement_decision(cover_expected, direct_expected, threshold, mode)
            selected = decision != "hold"
            units = actual if decision == "upper" else (-actual if decision == "lower" else 0.0)
            payoff = payoff_for_side(actual, decision, upper_water, lower_water) if selected else 0.0
            if not selected or actual == 0:
                hit = None
            elif actual > 0 and decision == "upper":
                hit = 1.0
            elif actual < 0 and decision == "lower":
                hit = 1.0
            else:
                hit = 0.0
            detail_rows.append(
                {
                    "seed": int(seed),
                    "split_name": split_name,
                    "threshold": float(threshold),
                    "mode": mode,
                    "label_idx": idx,
                    "match_id": ctx.get("match_id", ""),
                    "bookmaker_id": ctx.get("bookmaker_id", "unknown"),
                    "kickoff_time": ctx.get("kickoff_time", ""),
                    "asian_line": ctx.get("asian_line", ""),
                    "actual_upper_units": actual,
                    "cover_expected_upper_units": cover_expected,
                    "direct_expected_upper_units": direct_expected,
                    "decision": decision,
                    "selected": selected,
                    "model_side_units": units,
                    "model_side_payoff": payoff,
                    "direction_hit": hit,
                }
            )
        score = fixture_collapsed_score(detail_rows)
        out.append({"threshold": float(threshold), "mode": mode, "seed": int(seed), "split_name": split_name, "score": score, "detail_rows": detail_rows})
    return out


def select_nested_candidate(candidates: list[dict[str, Any]], min_unique_fixtures: int = 80) -> dict[str, Any]:
    eligible = [
        item
        for item in candidates
        if int(item["score"].get("selected_unique_fixtures", 0)) >= int(min_unique_fixtures)
    ]
    if not eligible:
        return {}
    return max(
        eligible,
        key=lambda item: (
            float(item["score"].get("fixture_avg_payoff", 0.0)),
            float(item["score"].get("fixture_direction_accuracy_ex_push", 0.0)),
            int(item["score"].get("selected_unique_fixtures", 0)),
        ),
    )


def _candidate_summary_row(fold_id: str, strategy_name: str, candidate: dict[str, Any]) -> dict[str, Any]:
    score = candidate.get("score", {})
    return {
        "fold_id": fold_id,
        "strategy_name": strategy_name,
        "split_name": candidate.get("split_name"),
        "seed": candidate.get("seed"),
        "threshold": candidate.get("threshold"),
        "mode": candidate.get("mode"),
        **score,
    }


def replay_fold_seed(
    data_path: str,
    fold_dir: str | Path,
    fold_id: str,
    seed: int,
    cover_export_root: str | Path,
    direct_export_root: str | Path,
    thresholds: list[float],
    locked_threshold: float,
    mode: str,
    out_dir: str | Path,
    min_unique_fixtures: int,
) -> dict[str, Any]:
    fold_dir = Path(fold_dir)
    out_dir = Path(out_dir)
    summary_rows = []
    detail_rows = []
    by_split: dict[str, list[dict[str, Any]]] = {}
    for split_name in ("selection", "forward"):
        ids_path = fold_dir / f"{split_name}_match_ids.txt"
        contexts = load_contexts_with_metadata(data_path, str(ids_path))
        cover_pairs = load_export_pairs(Path(cover_export_root) / split_name / "val_ah_cover_predictions.csv", contexts)
        direct_pairs = load_export_pairs(Path(direct_export_root) / split_name / "val_ah_cover_predictions.csv", contexts)
        candidates = replay_agreement_candidates(cover_pairs, direct_pairs, contexts, thresholds, mode, seed, split_name)
        by_split[split_name] = candidates
        for candidate in candidates:
            summary_rows.append(_candidate_summary_row(fold_id, "candidate_grid", candidate))
    selected = select_nested_candidate(by_split["selection"], min_unique_fixtures=min_unique_fixtures)
    selected_threshold = float(selected.get("threshold", locked_threshold))
    forward_candidates = {float(item["threshold"]): item for item in by_split["forward"]}
    selection_candidates = {float(item["threshold"]): item for item in by_split["selection"]}
    locked_selection = selection_candidates.get(float(locked_threshold), {})
    locked_forward = forward_candidates.get(float(locked_threshold), {})
    nested_forward = forward_candidates.get(selected_threshold, {})
    outputs = {
        "phase": "P19",
        "task": "replay_fold_seed_agreement_strategy",
        "fold_id": fold_id,
        "seed": int(seed),
        "locked_threshold": float(locked_threshold),
        "nested_selected_threshold": selected_threshold,
        "locked_selection": _candidate_summary_row(fold_id, "locked_selection", locked_selection) if locked_selection else {},
        "locked_forward": _candidate_summary_row(fold_id, "locked_forward", locked_forward) if locked_forward else {},
        "nested_selection": _candidate_summary_row(fold_id, "nested_selection", selected) if selected else {},
        "nested_forward": _candidate_summary_row(fold_id, "nested_forward", nested_forward) if nested_forward else {},
        "input_policy": {"no_test_split_loaded": True, "forward_not_used_for_selection": True},
    }
    for name, candidate in (("locked_forward", locked_forward), ("nested_forward", nested_forward)):
        for row in candidate.get("detail_rows", []):
            detail_rows.append({"strategy_name": name, **row})
    write_json(out_dir / "fold_seed_strategy_report.json", outputs)
    write_csv(out_dir / "fold_seed_candidate_summary.csv", summary_rows)
    write_csv(out_dir / "fold_seed_forward_detail.csv", detail_rows)
    return outputs


def run_fold_seed(args: argparse.Namespace, registry: dict[str, Any], fold: dict[str, Any], seed: int) -> dict[str, Any]:
    fold_id = str(fold["fold_id"])
    fold_dir = Path(args.fold_root) / fold_id
    run_root = Path(args.out_root) / "fold_runs" / fold_id / f"seed_{seed}"
    log_dir = run_root / "logs"
    cover_variant = registry["locked_strategy"]["cover_variant"]
    direct_variant = registry["locked_strategy"]["direct_variant"]
    variants = [cover_variant, direct_variant]
    for variant in variants:
        train_out = run_root / "train" / f"{variant}_seed{seed}"
        train_cmd = build_train_command(
            args.data,
            fold_dir / "train_match_ids.txt",
            fold_dir / "selection_match_ids.txt",
            variant,
            seed,
            train_out,
            args.epochs,
            args.batch_size,
            args.device,
            args.max_samples,
            args.allow_overwrite,
        )
        run_command(train_cmd, log_dir / f"train_{variant}.log", train_out / "report.json" if args.skip_existing else None)
        for split_name in ("selection", "forward"):
            export_out = run_root / "exports" / variant / split_name
            export_cmd = build_export_command(
                args.data,
                fold_dir / f"{split_name}_match_ids.txt",
                train_out,
                export_out,
                f"{fold_id}_{split_name}",
                args.device,
                args.export_batch_size,
                args.allow_overwrite,
            )
            run_command(export_cmd, log_dir / f"export_{variant}_{split_name}.log", export_out / "export_report.json" if args.skip_existing else None)
    return replay_fold_seed(
        args.data,
        fold_dir,
        fold_id,
        seed,
        run_root / "exports" / cover_variant,
        run_root / "exports" / direct_variant,
        [float(value) for value in registry.get("thresholds", [0.25])],
        float(registry["locked_strategy"]["threshold"]),
        str(registry["locked_strategy"].get("mode", "both")),
        run_root / "strategy",
        args.min_unique_fixtures,
    )


def aggregate_reports(reports: list[dict[str, Any]], min_unique_fixtures: int) -> dict[str, Any]:
    locked_scores = []
    nested_scores = []
    for report in reports:
        if report.get("locked_forward"):
            locked_scores.append(report["locked_forward"])
        if report.get("nested_forward"):
            nested_scores.append(report["nested_forward"])
    verdict = build_p19_verdict_payload(
        "PASS",
        [
            {
                "fold_id": f"{row.get('fold_id')}_seed{row.get('seed')}",
                "fixture_avg_payoff": row.get("fixture_avg_payoff", 0.0),
                "selected_unique_fixtures": row.get("selected_unique_fixtures", 0),
            }
            for row in locked_scores
        ],
        min_unique_fixtures_per_fold=min_unique_fixtures,
    )
    return {
        "phase": "P19",
        "task": "rolling_backtest_aggregate",
        "locked_forward_scores": locked_scores,
        "nested_forward_scores": nested_scores,
        "verdict": verdict,
    }


def load_fold_manifest(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload.get("folds", []))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run P19 rolling retrain/backtest")
    parser.add_argument("--data", default="data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl")
    parser.add_argument("--fold-manifest", default="runs/p19_rolling_backtest/folds/fold_manifest.json")
    parser.add_argument("--fold-root", default="runs/p19_rolling_backtest/folds")
    parser.add_argument("--registry", default="configs/p19_candidate_registry.json")
    parser.add_argument("--out-root", default="runs/p19_rolling_backtest")
    parser.add_argument("--folds", default="fold_01")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--export-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--min-unique-fixtures", type=int, default=80)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--no-test", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.no_test:
        raise SystemExit("Refusing to run P19 rolling backtest without --no-test")
    assert_no_test_paths([args.data, args.fold_manifest, args.fold_root, args.registry, args.out_root])
    registry = load_registry(args.registry)
    requested_folds = set(_csv_strings(args.folds))
    seeds = _csv_ints(args.seeds)
    reports = []
    for fold in load_fold_manifest(args.fold_manifest):
        if requested_folds and str(fold["fold_id"]) not in requested_folds:
            continue
        for seed in seeds:
            reports.append(run_fold_seed(args, registry, fold, seed))
    aggregate = aggregate_reports(reports, args.min_unique_fixtures)
    out_root = Path(args.out_root)
    write_json(out_root / "p19_rolling_backtest_summary.json", aggregate)
    write_csv(out_root / "p19_locked_forward_scores.csv", aggregate["locked_forward_scores"])
    write_csv(out_root / "p19_nested_forward_scores.csv", aggregate["nested_forward_scores"])
    print(
        "P19 rolling backtest: "
        f"runs={len(reports)} verdict={aggregate['verdict']['verdict']} "
        f"locked_payoff={aggregate['verdict']['aggregate_fixture_avg_payoff']}"
    )


if __name__ == "__main__":
    main()
