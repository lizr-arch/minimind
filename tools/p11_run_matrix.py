"""Run the fixed P11 score-count structure probe matrix."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


class P11ProtocolError(RuntimeError):
    pass


FIXED_P11_SEEDS = [42, 123, 2025]
FIXED_P11_CONTROL = "p11_control_p6_euro_default"
FIXED_P11_VARIANTS: dict[str, dict[str, Any]] = {
    "p11_a_goal_count_aux_010": {
        "count_loss_weight": 0.10,
        "score_kl_weight": 0.0,
        "close_game_kl_only": False,
    },
    "p11_b_goal_count_kl_005": {
        "count_loss_weight": 0.10,
        "score_kl_weight": 0.005,
        "close_game_kl_only": False,
    },
    "p11_c_goal_count_kl_020": {
        "count_loss_weight": 0.10,
        "score_kl_weight": 0.020,
        "close_game_kl_only": False,
    },
    "p11_d_goal_count_close_kl_010": {
        "count_loss_weight": 0.10,
        "score_kl_weight": 0.010,
        "close_game_kl_only": True,
    },
}
ALLOWED_P11_VERDICTS = {
    "P11_MAINLINE_CANDIDATE",
    "P11_CONTINUE_SCORE_STRUCTURE",
    "P11_SCORE_HEAD_LEARNS_BUT_NO_BRIDGE",
    "P11_REJECT_SCORE_STRUCTURE_MOVE_TO_DATA_FEATURES",
    "P11_BLOCKED_BY_LABEL_AUDIT",
    "P11_NO_ACCEPTED_VARIANT",
    "P11_MATRIX_INCOMPLETE",
    "P11_PROTOCOL_HARD_FAIL",
}


def parse_registered_variants(raw: str) -> list[str]:
    variants = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in variants if item not in FIXED_P11_VARIANTS]
    if unknown:
        raise ValueError(f"Unregistered P11 variant(s): {unknown}")
    if len(set(variants)) != len(variants):
        raise ValueError("Duplicate P11 variants are not allowed")
    return variants


def parse_registered_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    unknown = [item for item in seeds if item not in FIXED_P11_SEEDS]
    if unknown:
        raise ValueError(f"Unregistered P11 seed(s): {unknown}")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate P11 seeds are not allowed")
    return seeds


def build_protocol_audit(train_ids: str, val_ids: str, variants: list[str], seeds: list[int]) -> dict[str, Any]:
    if "test" in Path(train_ids).name.lower() or "test" in Path(val_ids).name.lower():
        raise P11ProtocolError("P11 refuses test split paths")
    hard_fail_reasons = []
    if variants != list(FIXED_P11_VARIANTS):
        hard_fail_reasons.append("P11_FAIL_MATRIX_VARIANT_MUTATION")
    if seeds != FIXED_P11_SEEDS:
        hard_fail_reasons.append("P11_FAIL_SEED_LIST_MUTATION")
    return {
        "phase": "P11",
        "no_test_split_loaded": True,
        "no_test_artifact_written": True,
        "no_val_posthoc_fit": True,
        "official_val_fixed_matrix_only": variants == list(FIXED_P11_VARIANTS),
        "control_required": True,
        "control_name": FIXED_P11_CONTROL,
        "control_seed_list": seeds,
        "candidate_count": len(variants),
        "candidate_count_mutation": variants != list(FIXED_P11_VARIANTS),
        "seed_list": seeds,
        "seed_list_mutation": seeds != FIXED_P11_SEEDS,
        "expected_control_runs": len(seeds),
        "expected_candidate_runs": len(variants) * len(seeds),
        "expected_total_runs": len(seeds) + len(variants) * len(seeds),
        "score_count_head_is_optional": True,
        "p6_default_behavior_changed": False,
        "large_score_grid_head_added": False,
        "draw_specific_loss_added": False,
        "btts_or_total_goal_aux_head_added": False,
        "hard_fail_reasons": hard_fail_reasons,
    }


def _read_report(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "report.json").read_text(encoding="utf-8"))


def _label_audit_coverage(report: dict[str, Any], split: str) -> float:
    audit = report.get("data", {}).get("score_label_audit", {}).get(split, {})
    return float(audit.get("score_label_coverage", 0.0) or 0.0)


def _row_from_report(run_dir: Path, variant: str, seed: int, is_control: bool = False) -> dict[str, Any]:
    report = _read_report(run_dir)
    val = report.get("val_metrics", {})
    score = report.get("score_count_metrics", {})
    score_derived = report.get("score_derived_1x2_metrics", {})
    score_baselines = report.get("score_baselines", {})
    manifest = report.get("training_manifest", {})
    return {
        "variant": variant,
        "seed": seed,
        "is_control": is_control,
        "score_head_enabled": bool(manifest.get("score_head_enabled", False)),
        "val_logloss": float(val.get("logloss", report.get("best_val_logloss", 0.0))),
        "ece": float(val.get("ece", 0.0)),
        "draw_class_nll": float(val.get("draw_class_nll", val.get("draw_nll", 0.0))),
        "draw_recall": float(val.get("draw_recall", 0.0)),
        "draw_precision": float(val.get("draw_precision", 0.0)),
        "draw_top2": float(val.get("draw_top2_recall", val.get("draw_top2", 0.0))),
        "mean_p_draw_true_draw": float(val.get("mean_p_draw_on_true_draw", 0.0)),
        "mean_draw_margin_to_top": float(val.get("mean_draw_margin_to_top", 0.0)),
        "argmax_draw_count": int(val.get("argmax_draw_count", 0)),
        "val_count_nll": float(score.get("val_count_nll", 0.0) or 0.0),
        "constant_train_mean_rate_count_nll": float(
            score.get("constant_train_mean_rate_count_nll", score_baselines.get("constant_train_mean_rate_count_nll", 0.0)) or 0.0
        ),
        "home_goal_mae": float(score.get("home_goal_mae", 0.0) or 0.0),
        "away_goal_mae": float(score.get("away_goal_mae", 0.0) or 0.0),
        "total_goal_mae": float(score.get("total_goal_mae", 0.0) or 0.0),
        "goal_diff_mae": float(score.get("goal_diff_mae", 0.0) or 0.0),
        "exact_score_top1": float(score.get("exact_score_top1", 0.0) or 0.0),
        "exact_score_top3": float(score.get("exact_score_top3", 0.0) or 0.0),
        "btts_accuracy": float(score.get("btts_accuracy", 0.0) or 0.0),
        "tail_mass_mean": float(score.get("tail_mass_mean", 0.0) or 0.0),
        "tail_mass_p95": float(score.get("tail_mass_p95", 0.0) or 0.0),
        "lambda_min_saturation_rate": float(score.get("lambda_min_saturation_rate", 0.0) or 0.0),
        "lambda_max_saturation_rate": float(score.get("lambda_max_saturation_rate", 0.0) or 0.0),
        "score_derived_1x2_logloss": float(score_derived.get("score_derived_1x2_logloss", 0.0) or 0.0),
        "constant_class_prior_1x2_logloss": float(
            score_derived.get("constant_class_prior_1x2_logloss", score_baselines.get("constant_class_prior_1x2_logloss", 0.0)) or 0.0
        ),
        "score_derived_draw_nll": float(score_derived.get("score_derived_draw_nll", 0.0) or 0.0),
        "score_derived_draw_top2": float(score_derived.get("score_derived_draw_top2", 0.0) or 0.0),
        "score_derived_mean_p_draw_true_draw": float(score_derived.get("score_derived_mean_p_draw_true_draw", 0.0) or 0.0),
        "score_derived_draw_margin_to_top": float(score_derived.get("score_derived_draw_margin_to_top", 0.0) or 0.0),
        "score_vs_final_draw_corr": score_derived.get("score_vs_final_draw_corr"),
        "score_vs_final_draw_mae": float(score_derived.get("score_vs_final_draw_mae", 0.0) or 0.0),
        "train_score_label_coverage": _label_audit_coverage(report, "train"),
        "val_score_label_coverage": _label_audit_coverage(report, "val"),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows) if rows else 0.0


def summarize_rows(rows: list[dict[str, Any]], variant: str, seeds: list[int]) -> dict[str, Any]:
    vr = [row for row in rows if row["variant"] == variant]
    if not vr:
        return {"variant": variant, "seed_count": 0, "status": "missing"}
    losses = [row["val_logloss"] for row in vr]
    return {
        "variant": variant,
        "seed_count": len(vr),
        "status": "complete" if len(vr) == len(seeds) else "incomplete",
        "mean_val_logloss": statistics.fmean(losses),
        "std_val_logloss": statistics.pstdev(losses) if len(losses) > 1 else 0.0,
        "mean_ece": _mean(vr, "ece"),
        "draw_class_nll": _mean(vr, "draw_class_nll"),
        "draw_recall": _mean(vr, "draw_recall"),
        "draw_precision": _mean(vr, "draw_precision"),
        "draw_top2": _mean(vr, "draw_top2"),
        "mean_p_draw_true_draw": _mean(vr, "mean_p_draw_true_draw"),
        "mean_draw_margin_to_top": _mean(vr, "mean_draw_margin_to_top"),
        "val_count_nll": _mean(vr, "val_count_nll"),
        "constant_train_mean_rate_count_nll": _mean(vr, "constant_train_mean_rate_count_nll"),
        "home_goal_mae": _mean(vr, "home_goal_mae"),
        "away_goal_mae": _mean(vr, "away_goal_mae"),
        "total_goal_mae": _mean(vr, "total_goal_mae"),
        "goal_diff_mae": _mean(vr, "goal_diff_mae"),
        "exact_score_top1": _mean(vr, "exact_score_top1"),
        "exact_score_top3": _mean(vr, "exact_score_top3"),
        "btts_accuracy": _mean(vr, "btts_accuracy"),
        "tail_mass_mean": _mean(vr, "tail_mass_mean"),
        "tail_mass_p95": _mean(vr, "tail_mass_p95"),
        "lambda_min_saturation_rate": _mean(vr, "lambda_min_saturation_rate"),
        "lambda_max_saturation_rate": _mean(vr, "lambda_max_saturation_rate"),
        "score_derived_1x2_logloss": _mean(vr, "score_derived_1x2_logloss"),
        "constant_class_prior_1x2_logloss": _mean(vr, "constant_class_prior_1x2_logloss"),
        "score_derived_draw_nll": _mean(vr, "score_derived_draw_nll"),
        "score_derived_draw_top2": _mean(vr, "score_derived_draw_top2"),
        "score_derived_mean_p_draw_true_draw": _mean(vr, "score_derived_mean_p_draw_true_draw"),
        "score_derived_draw_margin_to_top": _mean(vr, "score_derived_draw_margin_to_top"),
        "score_vs_final_draw_mae": _mean(vr, "score_vs_final_draw_mae"),
        "train_score_label_coverage": _mean(vr, "train_score_label_coverage"),
        "val_score_label_coverage": _mean(vr, "val_score_label_coverage"),
    }


def _score_protocol_failures(item: dict[str, Any]) -> list[str]:
    failures = []
    if item.get("train_score_label_coverage", 0.0) < 0.999 or item.get("val_score_label_coverage", 0.0) < 0.999:
        failures.append("P11_BLOCKED_BY_LABEL_AUDIT")
    if item.get("tail_mass_mean", 0.0) > 0.01:
        failures.append("P11_FAIL_TAIL_MASS_MEAN")
    if item.get("tail_mass_p95", 0.0) > 0.03:
        failures.append("P11_FAIL_TAIL_MASS_P95")
    if item.get("lambda_min_saturation_rate", 0.0) > 0.05 or item.get("lambda_max_saturation_rate", 0.0) > 0.05:
        failures.append("P11_FAIL_LAMBDA_SATURATION")
    return failures


def p11_variant_verdict(item: dict[str, Any], control_summary: dict[str, Any]) -> list[str]:
    if item.get("status") != "complete":
        return ["P11_INCOMPLETE"]
    protocol_failures = _score_protocol_failures(item)
    if "P11_BLOCKED_BY_LABEL_AUDIT" in protocol_failures:
        return ["P11_BLOCKED_BY_LABEL_AUDIT"]
    if protocol_failures:
        return protocol_failures

    control_logloss = float(control_summary.get("mean_val_logloss", 999.0))
    control_ece = float(control_summary.get("mean_ece", 999.0))
    count_beats_baseline = item["val_count_nll"] <= item["constant_train_mean_rate_count_nll"] - 0.005
    score_beats_prior = item["score_derived_1x2_logloss"] < item["constant_class_prior_1x2_logloss"]
    mainline = (
        item["mean_val_logloss"] <= control_logloss + 0.0003
        and item["mean_ece"] <= max(control_ece + 0.005, 0.033)
        and item["draw_top2"] >= 0.40
        and item["draw_class_nll"] <= 1.609
        and item["mean_p_draw_true_draw"] >= 0.210
        and item["mean_draw_margin_to_top"] <= 0.386
        and count_beats_baseline
        and score_beats_prior
    )
    if mainline:
        return ["P11_MAINLINE_CANDIDATE"]

    directional = (
        item["draw_top2"] >= 0.40
        and item["mean_p_draw_true_draw"] >= 0.208
        and item["mean_val_logloss"] <= control_logloss + 0.0008
        and item["score_derived_draw_top2"] >= 0.45
        and score_beats_prior
    )
    if directional:
        return ["P11_CONTINUE_SCORE_STRUCTURE"]
    if count_beats_baseline and score_beats_prior and item["score_derived_draw_top2"] >= 0.40:
        return ["P11_SCORE_HEAD_LEARNS_BUT_NO_BRIDGE"]
    return ["P11_FAIL_GATES"]


def p11_matrix_verdict(summaries: list[dict[str, Any]], control_summary: dict[str, Any], seeds: list[int]) -> list[str]:
    if control_summary.get("status") != "complete" or control_summary.get("seed_count") != len(seeds):
        return ["P11_MATRIX_INCOMPLETE"]
    complete = [item for item in summaries if item.get("status") == "complete"]
    if len(complete) != len(FIXED_P11_VARIANTS):
        return ["P11_MATRIX_INCOMPLETE"]
    if any("P11_BLOCKED_BY_LABEL_AUDIT" in item.get("verdict", []) for item in complete):
        return ["P11_BLOCKED_BY_LABEL_AUDIT"]
    if any(any(reason.startswith("P11_FAIL_TAIL") or reason == "P11_FAIL_LAMBDA_SATURATION" for reason in item.get("verdict", [])) for item in complete):
        return ["P11_NO_ACCEPTED_VARIANT"]
    if any("P11_MAINLINE_CANDIDATE" in item.get("verdict", []) for item in complete):
        return ["P11_MAINLINE_CANDIDATE"]
    if any("P11_CONTINUE_SCORE_STRUCTURE" in item.get("verdict", []) for item in complete):
        return ["P11_CONTINUE_SCORE_STRUCTURE"]
    if any("P11_SCORE_HEAD_LEARNS_BUT_NO_BRIDGE" in item.get("verdict", []) for item in complete):
        return ["P11_SCORE_HEAD_LEARNS_BUT_NO_BRIDGE"]
    if all(float(item.get("draw_top2", 0.0)) < 0.36 for item in complete):
        return ["P11_REJECT_SCORE_STRUCTURE_MOVE_TO_DATA_FEATURES"]
    if all(float(item.get("mean_p_draw_true_draw", 0.0)) < 0.205 for item in complete):
        return ["P11_REJECT_SCORE_STRUCTURE_MOVE_TO_DATA_FEATURES"]
    if all(float(item.get("score_derived_draw_top2", 0.0)) < 0.40 for item in complete):
        return ["P11_REJECT_SCORE_STRUCTURE_MOVE_TO_DATA_FEATURES"]
    return ["P11_NO_ACCEPTED_VARIANT"]


def aggregate_results(out_root: Path, variants: list[str], seeds: list[int]) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        run_dir = out_root / f"{FIXED_P11_CONTROL}_seed{seed}"
        if (run_dir / "report.json").exists():
            rows.append(_row_from_report(run_dir, FIXED_P11_CONTROL, seed, is_control=True))
    control_summary = summarize_rows(rows, FIXED_P11_CONTROL, seeds)
    summaries = []
    for variant in variants:
        for seed in seeds:
            run_dir = out_root / f"{variant}_seed{seed}"
            if (run_dir / "report.json").exists():
                rows.append(_row_from_report(run_dir, variant, seed, is_control=False))
        item = summarize_rows(rows, variant, seeds)
        item["verdict"] = p11_variant_verdict(item, control_summary)
        summaries.append(item)
    best = min([item for item in summaries if item.get("seed_count", 0) > 0], key=lambda x: x.get("mean_val_logloss", 999), default=None)
    verdict = p11_matrix_verdict(summaries, control_summary, seeds)
    if any(item not in ALLOWED_P11_VERDICTS for item in verdict):
        verdict = ["P11_NO_ACCEPTED_VARIANT"]
    return {
        "rows": rows,
        "control_summary": control_summary,
        "summaries": summaries,
        "best_variant": best,
        "verdict": verdict,
        "historical_references": {
            "p10_control_mean_val_logloss": 0.9373360475,
            "p10_best_variant_mean_val_logloss": 0.9372037848,
            "p10_best_draw_top2": 0.4840579728,
            "p10_best_mean_p_draw_true_draw": 0.2127997031,
        },
    }


def write_results_by_seed(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report_md(aggregate: dict[str, Any], path: Path) -> None:
    lines = ["# P11 Score-Count Structure Matrix", ""]
    control = aggregate.get("control_summary", {})
    lines.append("## Same-Protocol Control")
    lines.append(
        f"- {FIXED_P11_CONTROL}: seeds={control.get('seed_count', 0)}, "
        f"logloss={float(control.get('mean_val_logloss', 0.0)):.6f}, "
        f"ece={float(control.get('mean_ece', 0.0)):.6f}"
    )
    lines.append("")
    lines.append("## Candidates")
    for item in aggregate.get("summaries", []):
        lines.append(
            f"- {item.get('variant')}: seeds={item.get('seed_count', 0)}, "
            f"logloss={float(item.get('mean_val_logloss', 0.0)):.6f}, "
            f"draw_top2={float(item.get('draw_top2', 0.0)):.6f}, "
            f"p_draw_true={float(item.get('mean_p_draw_true_draw', 0.0)):.6f}, "
            f"score_draw_top2={float(item.get('score_derived_draw_top2', 0.0)):.6f}, "
            f"count_nll={float(item.get('val_count_nll', 0.0)):.6f}, "
            f"verdict={','.join(item.get('verdict', []))}"
        )
    lines.append("")
    lines.append(f"Final verdict: `{','.join(aggregate.get('verdict', []))}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed P11 score-count structure probe matrix")
    parser.add_argument("--variants", default=",".join(FIXED_P11_VARIANTS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in FIXED_P11_SEEDS))
    parser.add_argument("--data", default="data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl")
    parser.add_argument("--train-ids", default="data/odds_real/splits_v6/train_match_ids.txt")
    parser.add_argument("--val-ids", default="data/odds_real/splits_v6/val_match_ids.txt")
    parser.add_argument("--out-root", default="runs/p11_score_count_structure")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-test", action="store_true")
    return parser


def _base_model_args(args: argparse.Namespace) -> list[str]:
    return [
        "--data",
        args.data,
        "--train-ids",
        args.train_ids,
        "--val-ids",
        args.val_ids,
        "--feature-groups",
        "euro",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--d-model",
        "96",
        "--n-heads",
        "4",
        "--n-layers",
        "2",
        "--d-ff",
        "192",
        "--dropout",
        "0.10",
        "--diff-loss-weight",
        "0.30",
        "--consistency-loss-weight",
        "0.10",
        "--delta-l2-weight",
        "0.01",
        "--score-grid-max-goals",
        "10",
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


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.no_test:
        raise SystemExit("Refusing to run without --no-test")
    variants = parse_registered_variants(args.variants)
    seeds = parse_registered_seeds(args.seeds)
    audit = build_protocol_audit(args.train_ids, args.val_ids, variants, seeds)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "protocol_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if audit["hard_fail_reasons"]:
        raise SystemExit(f"P11 protocol audit failed: {audit['hard_fail_reasons']}")
    (out_root / "manifest.json").write_text(
        json.dumps(
            {
                "phase": "P11",
                "control": FIXED_P11_CONTROL,
                "variants": variants,
                "seeds": seeds,
                "registry": FIXED_P11_VARIANTS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        run_dir = out_root / f"{FIXED_P11_CONTROL}_seed{seed}"
        if args.skip_existing and (run_dir / "report.json").exists():
            print(f"SKIP existing {run_dir}", flush=True)
            continue
        if (run_dir / "report.json").exists() or (run_dir / "best_model.pth").exists():
            raise SystemExit(f"Refusing to overwrite existing run: {run_dir}")
        cmd = [
            sys.executable,
            "tools/p11_train_score_count_structure.py",
            *_base_model_args(args),
            "--seed",
            str(seed),
            "--variant",
            FIXED_P11_CONTROL,
            "--out-dir",
            str(run_dir),
        ]
        if args.max_samples > 0:
            cmd.extend(["--max-samples", str(args.max_samples)])
        print(f"RUN {FIXED_P11_CONTROL} seed={seed} -> {run_dir}", flush=True)
        _run_command(cmd, log_dir / f"{FIXED_P11_CONTROL}_seed{seed}.log")
        print(f"DONE {FIXED_P11_CONTROL} seed={seed}", flush=True)

    for variant in variants:
        cfg = FIXED_P11_VARIANTS[variant]
        for seed in seeds:
            run_dir = out_root / f"{variant}_seed{seed}"
            if args.skip_existing and (run_dir / "report.json").exists():
                print(f"SKIP existing {run_dir}", flush=True)
                continue
            if (run_dir / "report.json").exists() or (run_dir / "best_model.pth").exists():
                raise SystemExit(f"Refusing to overwrite existing run: {run_dir}")
            cmd = [
                sys.executable,
                "tools/p11_train_score_count_structure.py",
                *_base_model_args(args),
                "--enable-goal-count-head",
                "--count-loss-weight",
                str(cfg["count_loss_weight"]),
                "--score-kl-weight",
                str(cfg["score_kl_weight"]),
                "--seed",
                str(seed),
                "--variant",
                variant,
                "--out-dir",
                str(run_dir),
            ]
            if cfg["close_game_kl_only"]:
                cmd.append("--close-game-kl-only")
            if args.max_samples > 0:
                cmd.extend(["--max-samples", str(args.max_samples)])
            print(f"RUN {variant} seed={seed} -> {run_dir}", flush=True)
            _run_command(cmd, log_dir / f"{variant}_seed{seed}.log")
            print(f"DONE {variant} seed={seed}", flush=True)

    aggregate = aggregate_results(out_root, variants, seeds)
    write_results_by_seed(aggregate["rows"], out_root / "results_by_seed.csv")
    (out_root / "summary.csv").write_text((out_root / "results_by_seed.csv").read_text(encoding="utf-8"), encoding="utf-8")
    (out_root / "summary.json").write_text(json.dumps(aggregate, indent=2, default=str), encoding="utf-8")
    (out_root / "results_summary.json").write_text(json.dumps(aggregate, indent=2, default=str), encoding="utf-8")
    write_report_md(aggregate, out_root / "report.md")


if __name__ == "__main__":
    main()
