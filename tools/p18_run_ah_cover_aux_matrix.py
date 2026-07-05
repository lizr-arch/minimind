"""Run and review P18.1 AH-cover auxiliary training matrix.

The runner keeps the P14/P16 mainline as the guardrail and varies only the
AH-cover auxiliary loss weight. It writes a compact post-training review so
each round can decide whether to continue, narrow, or stop.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p18_ah_cover_diagnostics import build_source_context_rows, summarize_ah_rows
from tools.p4_train_residual_goal_diff import load_rows_for_ids, load_split_ids


FIXED_P18_SEEDS = [42, 123, 2025]
FIXED_P18_VARIANTS: dict[str, dict[str, Any]] = {
    "p18_ahw_003": {"ah_cover_loss_weight": 0.03},
    "p18_ahw_010": {"ah_cover_loss_weight": 0.10},
    "p18_ahw_030": {"ah_cover_loss_weight": 0.30},
    "p18u_unit030": {"ah_cover_loss_weight": 0.0, "ah_unit_loss_weight": 0.30, "ah_side_loss_weight": 0.0},
    "p18u_side010_unit020": {"ah_cover_loss_weight": 0.0, "ah_unit_loss_weight": 0.20, "ah_side_loss_weight": 0.10},
    "p18u_ce003_side010_unit020": {"ah_cover_loss_weight": 0.03, "ah_unit_loss_weight": 0.20, "ah_side_loss_weight": 0.10},
    "p18d_direct030": {"ah_cover_loss_weight": 0.0, "ah_direct_unit_loss_weight": 0.30},
    "p18d_ce003_direct030": {"ah_cover_loss_weight": 0.03, "ah_direct_unit_loss_weight": 0.30},
    "p18d_ce003_direct015": {"ah_cover_loss_weight": 0.03, "ah_direct_unit_loss_weight": 0.15},
    "p18d_ce003_direct060": {"ah_cover_loss_weight": 0.03, "ah_direct_unit_loss_weight": 0.60},
    "p18d_ce010_direct030": {"ah_cover_loss_weight": 0.10, "ah_direct_unit_loss_weight": 0.30},
}
DEFAULT_P18_VARIANTS = ["p18_ahw_003", "p18_ahw_010", "p18_ahw_030"]
P18_BASELINE = {
    "variant": "p14_market_kl_030_balanced",
    "mean_val_logloss": 0.9370384613672892,
    "mean_market_draw_mae": 0.03618671620885531,
    "mean_ah_head_model_side_avg_units": 0.11703405397767415,
    "mean_direction_accuracy_ex_push": 0.5703948403334906,
}
AH_PROB_UNIT_WEIGHTS = {
    "p_upper_full_win": 1.0,
    "p_upper_half_win": 0.5,
    "p_push": 0.0,
    "p_upper_half_loss": -0.5,
    "p_upper_full_loss": -1.0,
}


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _stdev(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) >= 2 else 0.0


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _model_side(expected_units: float) -> str:
    if expected_units > 1e-9:
        return "upper"
    if expected_units < -1e-9:
        return "lower"
    return "neutral"


def parse_registered_variants(raw: str) -> list[str]:
    variants = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in variants if item not in FIXED_P18_VARIANTS]
    if unknown:
        raise ValueError(f"Unregistered P18 variant(s): {unknown}")
    if len(set(variants)) != len(variants):
        raise ValueError("Duplicate P18 variants are not allowed")
    return variants


def parse_registered_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    unknown = [seed for seed in seeds if seed not in FIXED_P18_SEEDS]
    if unknown:
        raise ValueError(f"Unregistered P18 seed(s): {unknown}")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate P18 seeds are not allowed")
    return seeds


def build_report_row(
    variant: str,
    seed: int,
    report: dict[str, Any],
    ah_direction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    val = report.get("val_metrics", {})
    market = report.get("market_replication_metrics", {})
    ah = report.get("ah_cover_metrics", {})
    selected = report.get("selected_checkpoint_metrics", {})
    ah_direction = ah_direction or {}
    return {
        "variant": variant,
        "seed": int(seed),
        "status": "complete",
        "selected_epoch": int(selected.get("epoch", report.get("best_epoch", 0)) or 0),
        "val_logloss": float(val.get("logloss", 0.0) or 0.0),
        "val_ece": float(val.get("ece", 0.0) or 0.0),
        "draw_top2": float(val.get("draw_top2_recall", 0.0) or 0.0),
        "market_kl": float(market.get("market_kl", 0.0) or 0.0),
        "market_draw_mae": float(market.get("market_draw_mae", 0.0) or 0.0),
        "ah_cover_acc": float(ah.get("ah_cover_acc", 0.0) or 0.0),
        "ah_cover_nll": float(ah.get("ah_cover_nll", 0.0) or 0.0),
        "ah_unit_mae": float(ah.get("ah_unit_mae", 0.0) or 0.0),
        "ah_side_acc": float(ah.get("ah_side_acc", 0.0) or 0.0),
        "ah_direct_unit_mae": float(ah.get("ah_direct_unit_mae", 0.0) or 0.0),
        "ah_direct_side_acc": float(ah.get("ah_direct_side_acc", 0.0) or 0.0),
        "ah_cover_n": int(ah.get("ah_cover_n", 0) or 0),
        "ah_cover_loss_weight": float((report.get("config", {}) or {}).get("ah_cover_loss_weight", 0.0) or 0.0),
        "ah_unit_loss_weight": float((report.get("config", {}) or {}).get("ah_unit_loss_weight", 0.0) or 0.0),
        "ah_side_loss_weight": float((report.get("config", {}) or {}).get("ah_side_loss_weight", 0.0) or 0.0),
        "ah_direct_unit_loss_weight": float((report.get("config", {}) or {}).get("ah_direct_unit_loss_weight", 0.0) or 0.0),
        "ah_head_model_side_avg_units": float(ah_direction.get("model_side_avg_units", 0.0) or 0.0),
        "ah_head_direction_accuracy_ex_push": float(ah_direction.get("direction_accuracy_ex_push", 0.0) or 0.0),
    }


def expected_upper_units_from_ah_cover_row(row: dict[str, Any]) -> float:
    if "direct_expected_upper_units" in row and str(row.get("direct_expected_upper_units", "")).strip() != "":
        return max(-1.0, min(1.0, _f(row, "direct_expected_upper_units")))
    return float(sum(_f(row, key) * weight for key, weight in AH_PROB_UNIT_WEIGHTS.items()))


def compute_ah_head_direction_summary(
    prediction_rows: list[dict[str, Any]],
    context_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    context_by_label_idx = {
        int(row["label_idx"]): row
        for row in context_rows
        if "label_idx" in row and str(row.get("label_idx")).strip() != ""
    }
    context_by_match_id = {str(row.get("match_id")): row for row in context_rows}
    direction_rows = []
    for pred_idx, pred in enumerate(prediction_rows):
        match_id = str(pred.get("match_id"))
        ctx = context_by_label_idx.get(pred_idx) if context_by_label_idx else None
        if ctx is None:
            ctx = context_by_match_id.get(match_id)
        if ctx is None:
            continue
        expected_units = expected_upper_units_from_ah_cover_row(pred)
        side = _model_side(expected_units)
        actual_units = _f(ctx, "actual_upper_units")
        direction_rows.append(
            {
                **ctx,
                "expected_upper_units": expected_units,
                "predicted_side": side,
                "model_side_actual_units": actual_units if side == "upper" else (-actual_units if side == "lower" else 0.0),
                "coarse_bucket_warning": False,
            }
        )
    summary = summarize_ah_rows(direction_rows, "ah_head", "all")
    summary["matched_prediction_rows"] = len(direction_rows)
    return summary


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def compute_ah_head_direction_summary_from_file(prediction_path: Path, context_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not prediction_path.exists():
        return {}
    return compute_ah_head_direction_summary(load_csv_rows(prediction_path), context_rows)


def summarize_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)
    summaries = []
    for variant, group in sorted(grouped.items()):
        summary = {
            "variant": variant,
            "seed_count": len(group),
            "ah_cover_loss_weight": _mean([_f(row, "ah_cover_loss_weight") for row in group]),
            "ah_unit_loss_weight": _mean([_f(row, "ah_unit_loss_weight") for row in group]),
            "ah_side_loss_weight": _mean([_f(row, "ah_side_loss_weight") for row in group]),
            "ah_direct_unit_loss_weight": _mean([_f(row, "ah_direct_unit_loss_weight") for row in group]),
            "mean_val_logloss": _mean([_f(row, "val_logloss") for row in group]),
            "stdev_val_logloss": _stdev([_f(row, "val_logloss") for row in group]),
            "mean_val_ece": _mean([_f(row, "val_ece") for row in group]),
            "mean_draw_top2": _mean([_f(row, "draw_top2") for row in group]),
            "mean_market_kl": _mean([_f(row, "market_kl") for row in group]),
            "mean_market_draw_mae": _mean([_f(row, "market_draw_mae") for row in group]),
            "mean_ah_cover_acc": _mean([_f(row, "ah_cover_acc") for row in group]),
            "mean_ah_cover_nll": _mean([_f(row, "ah_cover_nll") for row in group]),
            "mean_ah_unit_mae": _mean([_f(row, "ah_unit_mae") for row in group]),
            "mean_ah_side_acc": _mean([_f(row, "ah_side_acc") for row in group]),
            "mean_ah_direct_unit_mae": _mean([_f(row, "ah_direct_unit_mae") for row in group]),
            "mean_ah_direct_side_acc": _mean([_f(row, "ah_direct_side_acc") for row in group]),
            "mean_ah_head_model_side_avg_units": _mean([_f(row, "ah_head_model_side_avg_units") for row in group]),
            "mean_ah_head_direction_accuracy_ex_push": _mean([_f(row, "ah_head_direction_accuracy_ex_push") for row in group]),
        }
        summaries.append(summary)
    return summaries


def build_candidate_decision(candidate: dict[str, Any], baseline: dict[str, Any] = P18_BASELINE) -> dict[str, Any]:
    failed = []
    if _f(candidate, "mean_val_logloss") > _f(baseline, "mean_val_logloss") + 0.0010:
        failed.append("P18_FAIL_1X2_LOGLOSS_DRIFT")
    if _f(candidate, "mean_market_draw_mae") > _f(baseline, "mean_market_draw_mae") + 0.015:
        failed.append("P18_FAIL_MARKET_REPLICATION_DRIFT")
    if _f(candidate, "mean_ah_cover_acc") <= 0.0:
        failed.append("P18_FAIL_AH_HEAD_MISSING")
    directional_delta = _f(candidate, "mean_ah_head_model_side_avg_units") - _f(baseline, "mean_ah_head_model_side_avg_units")
    direction_acc_delta = _f(candidate, "mean_ah_head_direction_accuracy_ex_push") - _f(baseline, "mean_direction_accuracy_ex_push")
    if failed:
        verdict = "P18_REJECT_GUARDRAIL"
    elif directional_delta >= 0.005 and direction_acc_delta >= -0.005:
        verdict = "P18_DIRECTIONAL_GAIN_CANDIDATE"
    elif _f(candidate, "mean_ah_cover_acc") >= 0.40:
        verdict = "P18_LABEL_SIGNAL_NO_DIRECTIONAL_GAIN"
    else:
        verdict = "P18_NEEDS_MORE_SIGNAL"
    return {
        "variant": candidate.get("variant"),
        "verdict": verdict,
        "failed_gates": failed,
        "directional_delta_vs_baseline": directional_delta,
        "direction_acc_delta_vs_baseline": direction_acc_delta,
        "baseline": baseline,
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


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run P18.1 AH-cover auxiliary matrix")
    parser.add_argument("--variants", default=",".join(DEFAULT_P18_VARIANTS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in FIXED_P18_SEEDS))
    parser.add_argument("--data", default="data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl")
    parser.add_argument("--train-ids", default="data/odds_real/splits_v6/train_match_ids.txt")
    parser.add_argument("--val-ids", default="data/odds_real/splits_v6/val_match_ids.txt")
    parser.add_argument("--out-root", default="runs/p18_ah_cover_aux_matrix")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--no-test", action="store_true")
    return parser


def _run_train(args: argparse.Namespace, variant: str, seed: int, out_root: Path, log_dir: Path) -> None:
    cfg = FIXED_P18_VARIANTS[variant]
    run_dir = out_root / f"{variant}_seed{seed}"
    if args.skip_existing and (run_dir / "report.json").exists():
        print(f"SKIP existing {run_dir}", flush=True)
        return
    if not args.allow_overwrite and ((run_dir / "report.json").exists() or (run_dir / "best_model.pth").exists()):
        raise SystemExit(f"Refusing to overwrite existing run: {run_dir}")
    cmd = [
        sys.executable,
        "tools/p18_train_ah_cover_aux.py",
        "--data",
        args.data,
        "--train-ids",
        args.train_ids,
        "--val-ids",
        args.val_ids,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--ah-cover-loss-weight",
        str(cfg.get("ah_cover_loss_weight", 0.0)),
        "--out-dir",
        str(run_dir),
    ]
    if "ah_unit_loss_weight" in cfg:
        cmd.extend(["--ah-unit-loss-weight", str(cfg["ah_unit_loss_weight"])])
    if "ah_side_loss_weight" in cfg:
        cmd.extend(["--ah-side-loss-weight", str(cfg["ah_side_loss_weight"])])
    if "ah_direct_unit_loss_weight" in cfg:
        cmd.extend(["--ah-direct-unit-loss-weight", str(cfg["ah_direct_unit_loss_weight"])])
    if args.max_samples > 0:
        cmd.extend(["--max-samples", str(args.max_samples)])
    if args.allow_overwrite:
        cmd.append("--allow-overwrite")
    print(
        f"RUN {variant} seed={seed} "
        f"ah_weight={cfg.get('ah_cover_loss_weight', 0.0)} "
        f"unit_weight={cfg.get('ah_unit_loss_weight', 0.0)} "
        f"side_weight={cfg.get('ah_side_loss_weight', 0.0)} "
        f"direct_weight={cfg.get('ah_direct_unit_loss_weight', 0.0)} -> {run_dir}",
        flush=True,
    )
    with (log_dir / f"{variant}_seed{seed}.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    print(f"DONE {variant} seed={seed}", flush=True)


def aggregate_existing_runs(
    out_root: Path,
    variants: list[str],
    seeds: list[int],
    context_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    context_rows = context_rows or []
    rows = []
    for variant in variants:
        for seed in seeds:
            run_dir = out_root / f"{variant}_seed{seed}"
            report_path = run_dir / "report.json"
            if not report_path.exists():
                continue
            ah_direction = {}
            if context_rows:
                ah_direction = compute_ah_head_direction_summary_from_file(run_dir / "val_ah_cover_predictions.csv", context_rows)
            rows.append(build_report_row(variant, seed, load_report(report_path), ah_direction))
    summaries = summarize_candidate_rows(rows)
    decisions = [build_candidate_decision(row, P18_BASELINE) for row in summaries]
    if any(decision["verdict"] == "P18_DIRECTIONAL_GAIN_CANDIDATE" for decision in decisions):
        verdict = "P18_HAS_DIRECTIONAL_GAIN_CANDIDATE"
    elif any(decision["verdict"] == "P18_LABEL_SIGNAL_NO_DIRECTIONAL_GAIN" for decision in decisions):
        verdict = "P18_LABEL_SIGNAL_NO_DIRECTIONAL_GAIN"
    elif rows:
        verdict = "P18_NO_PROMISING_AH_AUX_YET"
    else:
        verdict = "P18_NO_COMPLETED_RUNS"
    return {
        "baseline": P18_BASELINE,
        "run_rows": rows,
        "candidate_summaries": summaries,
        "decisions": decisions,
        "verdict": verdict,
    }


def write_report_md(path: Path, aggregate: dict[str, Any]) -> None:
    lines = ["# P18.1 AH Cover Auxiliary Matrix", "", f"Verdict: `{aggregate.get('verdict')}`", ""]
    lines.append("## Baseline")
    baseline = aggregate.get("baseline", {})
    lines.append(
        f"- {baseline.get('variant')}: logloss={baseline.get('mean_val_logloss')}, "
        f"market_draw_mae={baseline.get('mean_market_draw_mae')}, "
        f"ah_units={baseline.get('mean_ah_head_model_side_avg_units')}"
    )
    lines.append("")
    lines.append("## Candidates")
    for row in aggregate.get("candidate_summaries", []):
        decision = next((item for item in aggregate.get("decisions", []) if item.get("variant") == row.get("variant")), {})
        lines.append(
            f"- {row.get('variant')}: seeds={row.get('seed_count')}, "
            f"ah_w={row.get('ah_cover_loss_weight'):.3f}, "
            f"unit_w={row.get('ah_unit_loss_weight'):.3f}, "
            f"side_w={row.get('ah_side_loss_weight'):.3f}, "
            f"direct_w={row.get('ah_direct_unit_loss_weight'):.3f}, "
            f"logloss={row.get('mean_val_logloss'):.6f}, "
            f"ah_acc={row.get('mean_ah_cover_acc'):.6f}, "
            f"ah_nll={row.get('mean_ah_cover_nll'):.6f}, "
            f"ah_unit_mae={row.get('mean_ah_unit_mae'):.6f}, "
            f"ah_side_acc={row.get('mean_ah_side_acc'):.6f}, "
            f"ah_direct_mae={row.get('mean_ah_direct_unit_mae'):.6f}, "
            f"ah_units={row.get('mean_ah_head_model_side_avg_units'):.6f}, "
            f"ah_dir_acc={row.get('mean_ah_head_direction_accuracy_ex_push'):.6f}, "
            f"market_draw_mae={row.get('mean_market_draw_mae'):.6f}, "
            f"decision={decision.get('verdict')}"
        )
    lines.append("")
    lines.append("## Review")
    lines.append("- Promote nothing automatically; the AH head learns settlement labels, but promotion requires directional-unit gain over the frozen P14/P16 goal-diff baseline.")
    lines.append("- Next round should replace plain 5-class CE with a side/unit-aligned AH objective, then rerun the same guardrails.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.no_test:
        raise SystemExit("Refusing to run P18 matrix without --no-test")
    variants = parse_registered_variants(args.variants)
    seeds = parse_registered_seeds(args.seeds)
    val_rows = load_rows_for_ids(args.data, load_split_ids(args.val_ids))
    context_rows = build_source_context_rows(val_rows)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "phase": "P18.1",
        "variants": variants,
        "seeds": seeds,
        "registry": FIXED_P18_VARIANTS,
        "baseline": P18_BASELINE,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": args.device,
        "max_samples": args.max_samples,
        "no_test_split_loaded": True,
        "val_source_rows": len(val_rows),
        "val_ah_context_rows": len(context_rows),
    }
    (out_root / "p18_inputs_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for variant in variants:
        for seed in seeds:
            _run_train(args, variant, seed, out_root, log_dir)
    aggregate = aggregate_existing_runs(out_root, variants, seeds, context_rows)
    (out_root / "p18_matrix_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    write_csv(out_root / "p18_run_rows.csv", aggregate["run_rows"])
    write_csv(out_root / "p18_candidate_summaries.csv", aggregate["candidate_summaries"])
    (out_root / "p18_decisions.json").write_text(json.dumps(aggregate["decisions"], indent=2), encoding="utf-8")
    write_report_md(out_root / "p18_matrix_report.md", aggregate)
    print(f"P18 matrix verdict: {aggregate['verdict']}")
    print(f"Report saved to {out_root / 'p18_matrix_report.md'}")


if __name__ == "__main__":
    main()
