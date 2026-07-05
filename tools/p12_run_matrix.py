"""Run the fixed P12 explicit score-prior residual matrix."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


class P12ProtocolError(RuntimeError):
    pass


FIXED_P12_SEEDS = [42, 123, 2025]
FIXED_P12_CONTROL = "p12_control_p6_euro_default"
FIXED_P12_VARIANTS: dict[str, dict[str, Any]] = {
    "p12_a_score_prior_residual_free": {
        "prior_mode": "score",
        "blend_alpha": 1.0,
        "residual_l2_weight": 0.0,
        "anchor_ce_weight": 0.0,
    },
    "p12_b_score_prior_residual_l2_005": {
        "prior_mode": "score",
        "blend_alpha": 1.0,
        "residual_l2_weight": 0.005,
        "anchor_ce_weight": 0.0,
    },
    "p12_c_anchor_blend_075": {
        "prior_mode": "blend",
        "blend_alpha": 0.75,
        "residual_l2_weight": 0.0,
        "anchor_ce_weight": 0.50,
    },
    "p12_d_anchor_blend_090": {
        "prior_mode": "blend",
        "blend_alpha": 0.90,
        "residual_l2_weight": 0.0,
        "anchor_ce_weight": 0.50,
    },
}
ALLOWED_P12_VERDICTS = {
    "P12_MAINLINE_CANDIDATE",
    "P12_CONTINUE_EXPLICIT_PRIOR",
    "P12_PRIOR_LEARNS_BUT_RESIDUAL_OVERWRITES",
    "P12_REJECT_EXPLICIT_PRIOR_MOVE_TO_DATA_FEATURES",
    "P12_BLOCKED_BY_LABEL_AUDIT",
    "P12_BLOCKED_BY_PROTOCOL",
    "P12_NO_ACCEPTED_VARIANT",
    "P12_MATRIX_INCOMPLETE",
}


def parse_registered_variants(raw: str) -> list[str]:
    variants = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in variants if item not in FIXED_P12_VARIANTS]
    if unknown:
        raise ValueError(f"Unregistered P12 variant(s): {unknown}")
    if len(set(variants)) != len(variants):
        raise ValueError("Duplicate P12 variants are not allowed")
    return variants


def parse_registered_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    unknown = [item for item in seeds if item not in FIXED_P12_SEEDS]
    if unknown:
        raise ValueError(f"Unregistered P12 seed(s): {unknown}")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate P12 seeds are not allowed")
    return seeds


def build_protocol_audit(train_ids: str, val_ids: str, variants: list[str], seeds: list[int]) -> dict[str, Any]:
    if "test" in Path(train_ids).name.lower() or "test" in Path(val_ids).name.lower():
        raise P12ProtocolError("P12 refuses test split paths")
    hard_fail_reasons = []
    if variants != list(FIXED_P12_VARIANTS):
        hard_fail_reasons.append("P12_FAIL_MATRIX_VARIANT_MUTATION")
    if seeds != FIXED_P12_SEEDS:
        hard_fail_reasons.append("P12_FAIL_SEED_LIST_MUTATION")
    return {
        "phase": "P12",
        "no_test_split_loaded": True,
        "no_test_artifact_written": True,
        "no_val_posthoc_fit": True,
        "official_val_fixed_matrix_only": variants == list(FIXED_P12_VARIANTS),
        "control_required": True,
        "control_name": FIXED_P12_CONTROL,
        "control_seed_list": seeds,
        "candidate_count": len(variants),
        "seed_list": seeds,
        "expected_control_runs": len(seeds),
        "expected_candidate_runs": len(variants) * len(seeds),
        "expected_total_runs": len(seeds) + len(variants) * len(seeds),
        "explicit_prior_residual_only": True,
        "p6_default_behavior_changed": False,
        "large_score_grid_head_added": False,
        "draw_specific_loss_added": False,
        "ensemble_used": False,
        "hard_fail_reasons": hard_fail_reasons,
    }


def _read_report(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "report.json").read_text(encoding="utf-8"))


def _coverage(report: dict[str, Any], split: str) -> float:
    return float(report.get("data", {}).get("score_label_audit", {}).get(split, {}).get("score_label_coverage", 1.0) or 0.0)


def _row_from_report(run_dir: Path, variant: str, seed: int, is_control: bool = False) -> dict[str, Any]:
    report = _read_report(run_dir)
    val = report.get("val_metrics", {})
    goal = report.get("goal_count_metrics", report.get("score_count_metrics", {}))
    score = report.get("score_derived_1x2_metrics", {})
    prior = report.get("prior_residual_metrics", {})
    baselines = report.get("score_baselines", {})
    return {
        "variant": variant,
        "seed": seed,
        "is_control": is_control,
        "val_logloss": float(val.get("logloss", report.get("best_val_logloss", 0.0))),
        "ece": float(val.get("ece", 0.0)),
        "draw_class_nll": float(val.get("draw_class_nll", val.get("draw_nll", 0.0))),
        "draw_top2": float(val.get("draw_top2", val.get("draw_top2_recall", 0.0))),
        "mean_p_draw_true_draw": float(val.get("mean_p_draw_on_true_draw", 0.0)),
        "mean_draw_margin_to_top": float(val.get("mean_draw_margin_to_top", 0.0)),
        "draw_recall": float(val.get("draw_recall", 0.0)),
        "draw_precision": float(val.get("draw_precision", 0.0)),
        "val_count_nll": float(goal.get("val_count_nll", 0.0) or 0.0),
        "constant_train_mean_rate_count_nll": float(
            goal.get("constant_train_mean_rate_count_nll", baselines.get("constant_train_mean_rate_count_nll", 0.0)) or 0.0
        ),
        "tail_mass_mean": float(goal.get("tail_mass_mean", 0.0) or 0.0),
        "tail_mass_p95": float(goal.get("tail_mass_p95", 0.0) or 0.0),
        "lambda_min_saturation_rate": float(goal.get("lambda_min_saturation_rate", 0.0) or 0.0),
        "lambda_max_saturation_rate": float(goal.get("lambda_max_saturation_rate", 0.0) or 0.0),
        "p_score_derived_1x2_logloss": float(score.get("p_score_derived_1x2_logloss", 0.0) or 0.0),
        "constant_class_prior_1x2_logloss": float(
            score.get("constant_class_prior_1x2_logloss", baselines.get("constant_class_prior_1x2_logloss", 0.0)) or 0.0
        ),
        "p_score_derived_draw_top2": float(score.get("p_score_derived_draw_top2", 0.0) or 0.0),
        "p_prior_logloss": float(prior.get("p_prior_logloss", 0.0) or 0.0),
        "p_prior_draw_nll": float(prior.get("p_prior_draw_nll", 0.0) or 0.0),
        "p_prior_draw_top2": float(prior.get("p_prior_draw_top2", 0.0) or 0.0),
        "p_prior_mean_p_draw_true_draw": float(prior.get("p_prior_mean_p_draw_true_draw", 0.0) or 0.0),
        "p_prior_draw_margin_to_top": float(prior.get("p_prior_draw_margin_to_top", 0.0) or 0.0),
        "final_minus_prior_logloss": float(prior.get("final_minus_prior_logloss", 0.0) or 0.0),
        "final_vs_prior_draw_corr": prior.get("final_vs_prior_draw_corr"),
        "final_vs_prior_draw_mae": float(prior.get("final_vs_prior_draw_mae", 0.0) or 0.0),
        "residual_centered_l2": float(prior.get("residual_centered_l2", 0.0) or 0.0),
        "residual_abs_mean": float(prior.get("residual_abs_mean", 0.0) or 0.0),
        "residual_abs_p95": float(prior.get("residual_abs_p95", 0.0) or 0.0),
        "train_score_label_coverage": _coverage(report, "train"),
        "val_score_label_coverage": _coverage(report, "val"),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows) if rows else 0.0


def summarize_rows(rows: list[dict[str, Any]], variant: str, seeds: list[int]) -> dict[str, Any]:
    vr = [row for row in rows if row["variant"] == variant]
    if not vr:
        return {"variant": variant, "seed_count": 0, "status": "missing"}
    losses = [row["val_logloss"] for row in vr]
    keys = [
        "ece",
        "draw_class_nll",
        "draw_top2",
        "mean_p_draw_true_draw",
        "mean_draw_margin_to_top",
        "draw_recall",
        "draw_precision",
        "val_count_nll",
        "constant_train_mean_rate_count_nll",
        "tail_mass_mean",
        "tail_mass_p95",
        "lambda_min_saturation_rate",
        "lambda_max_saturation_rate",
        "p_score_derived_1x2_logloss",
        "constant_class_prior_1x2_logloss",
        "p_score_derived_draw_top2",
        "p_prior_logloss",
        "p_prior_draw_nll",
        "p_prior_draw_top2",
        "p_prior_mean_p_draw_true_draw",
        "p_prior_draw_margin_to_top",
        "final_minus_prior_logloss",
        "final_vs_prior_draw_mae",
        "residual_centered_l2",
        "residual_abs_mean",
        "residual_abs_p95",
        "train_score_label_coverage",
        "val_score_label_coverage",
    ]
    out = {
        "variant": variant,
        "seed_count": len(vr),
        "status": "complete" if len(vr) == len(seeds) else "incomplete",
        "mean_val_logloss": statistics.fmean(losses),
        "std_val_logloss": statistics.pstdev(losses) if len(losses) > 1 else 0.0,
    }
    out.update({key: _mean(vr, key) for key in keys})
    return out


def _protocol_failures(item: dict[str, Any]) -> list[str]:
    failures = []
    if item.get("train_score_label_coverage", 1.0) < 0.999 or item.get("val_score_label_coverage", 1.0) < 0.999:
        failures.append("P12_BLOCKED_BY_LABEL_AUDIT")
    if item.get("tail_mass_mean", 0.0) > 0.01 or item.get("tail_mass_p95", 0.0) > 0.03:
        failures.append("P12_BLOCKED_BY_PROTOCOL")
    if item.get("lambda_min_saturation_rate", 0.0) > 0.05 or item.get("lambda_max_saturation_rate", 0.0) > 0.05:
        failures.append("P12_BLOCKED_BY_PROTOCOL")
    return failures


def p12_variant_verdict(item: dict[str, Any], control_summary: dict[str, Any]) -> list[str]:
    if item.get("status") != "complete":
        return ["P12_MATRIX_INCOMPLETE"]
    failures = _protocol_failures(item)
    if failures:
        return [failures[0]]
    control_logloss = float(control_summary.get("mean_val_logloss", 999.0))
    control_ece = float(control_summary.get("ece", 999.0))
    count_ok = item["val_count_nll"] <= item["constant_train_mean_rate_count_nll"] - 0.005
    score_ok = item["p_score_derived_1x2_logloss"] < item["constant_class_prior_1x2_logloss"]
    residual_ok = item["final_minus_prior_logloss"] <= -0.003
    mainline = (
        item["mean_val_logloss"] <= control_logloss + 0.0003
        and item["ece"] <= max(control_ece + 0.005, 0.033)
        and item["draw_top2"] >= 0.40
        and item["draw_class_nll"] <= 1.610
        and item["mean_p_draw_true_draw"] >= 0.208
        and item["mean_draw_margin_to_top"] <= 0.390
        and count_ok
        and score_ok
        and residual_ok
    )
    if mainline:
        return ["P12_MAINLINE_CANDIDATE"]
    directional = (
        item["draw_top2"] >= 0.40
        and item["mean_p_draw_true_draw"] >= 0.206
        and item["mean_val_logloss"] <= control_logloss + 0.0008
        and item["ece"] <= 0.036
        and item["final_minus_prior_logloss"] <= -0.005
        and item["p_score_derived_draw_top2"] >= 0.44
    )
    if directional:
        return ["P12_CONTINUE_EXPLICIT_PRIOR"]
    if item["p_prior_draw_top2"] >= 0.44 and item["draw_top2"] <= 0.34:
        return ["P12_PRIOR_LEARNS_BUT_RESIDUAL_OVERWRITES"]
    if item["residual_abs_p95"] > 2.50 and (item.get("final_vs_prior_draw_corr") is None or item.get("final_vs_prior_draw_corr", 0.0) < 0.75):
        return ["P12_PRIOR_LEARNS_BUT_RESIDUAL_OVERWRITES"]
    return ["P12_NO_ACCEPTED_VARIANT"]


def p12_matrix_verdict(summaries: list[dict[str, Any]], control_summary: dict[str, Any], seeds: list[int]) -> list[str]:
    if control_summary.get("status") != "complete" or control_summary.get("seed_count") != len(seeds):
        return ["P12_MATRIX_INCOMPLETE"]
    complete = [item for item in summaries if item.get("status") == "complete"]
    if len(complete) != len(FIXED_P12_VARIANTS):
        return ["P12_MATRIX_INCOMPLETE"]
    if any("P12_BLOCKED_BY_LABEL_AUDIT" in item.get("verdict", []) for item in complete):
        return ["P12_BLOCKED_BY_LABEL_AUDIT"]
    if any("P12_BLOCKED_BY_PROTOCOL" in item.get("verdict", []) for item in complete):
        return ["P12_BLOCKED_BY_PROTOCOL"]
    if any("P12_MAINLINE_CANDIDATE" in item.get("verdict", []) for item in complete):
        return ["P12_MAINLINE_CANDIDATE"]
    if any("P12_CONTINUE_EXPLICIT_PRIOR" in item.get("verdict", []) for item in complete):
        return ["P12_CONTINUE_EXPLICIT_PRIOR"]
    if any("P12_PRIOR_LEARNS_BUT_RESIDUAL_OVERWRITES" in item.get("verdict", []) for item in complete):
        return ["P12_PRIOR_LEARNS_BUT_RESIDUAL_OVERWRITES"]
    control_logloss = float(control_summary.get("mean_val_logloss", 999.0))
    if all(float(item.get("draw_top2", 0.0)) < 0.36 for item in complete):
        return ["P12_REJECT_EXPLICIT_PRIOR_MOVE_TO_DATA_FEATURES"]
    if all(float(item.get("mean_p_draw_true_draw", 0.0)) < 0.203 for item in complete):
        return ["P12_REJECT_EXPLICIT_PRIOR_MOVE_TO_DATA_FEATURES"]
    if all(float(item.get("mean_val_logloss", 0.0)) > control_logloss + 0.0008 for item in complete):
        return ["P12_REJECT_EXPLICIT_PRIOR_MOVE_TO_DATA_FEATURES"]
    return ["P12_NO_ACCEPTED_VARIANT"]


def aggregate_results(out_root: Path, variants: list[str], seeds: list[int]) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        run_dir = out_root / f"{FIXED_P12_CONTROL}_seed{seed}"
        if (run_dir / "report.json").exists():
            rows.append(_row_from_report(run_dir, FIXED_P12_CONTROL, seed, is_control=True))
    control_summary = summarize_rows(rows, FIXED_P12_CONTROL, seeds)
    summaries = []
    for variant in variants:
        for seed in seeds:
            run_dir = out_root / f"{variant}_seed{seed}"
            if (run_dir / "report.json").exists():
                rows.append(_row_from_report(run_dir, variant, seed, is_control=False))
        item = summarize_rows(rows, variant, seeds)
        item["verdict"] = p12_variant_verdict(item, control_summary)
        summaries.append(item)
    best = min([item for item in summaries if item.get("seed_count", 0) > 0], key=lambda x: x.get("mean_val_logloss", 999), default=None)
    verdict = p12_matrix_verdict(summaries, control_summary, seeds)
    if any(item not in ALLOWED_P12_VERDICTS for item in verdict):
        verdict = ["P12_NO_ACCEPTED_VARIANT"]
    return {
        "rows": rows,
        "control_summary": control_summary,
        "summaries": summaries,
        "best_variant": best,
        "verdict": verdict,
        "historical_references": {
            "p11_verdict": "P11_SCORE_HEAD_LEARNS_BUT_NO_BRIDGE",
            "p11_best_score_derived_draw_top2": 0.4561594228,
            "p11_control_mean_val_logloss": 0.9373360475,
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
    lines = ["# P12 Explicit Score-Prior Residual Matrix", ""]
    control = aggregate.get("control_summary", {})
    lines.append("## Same-Protocol Control")
    lines.append(
        f"- {FIXED_P12_CONTROL}: seeds={control.get('seed_count', 0)}, "
        f"logloss={float(control.get('mean_val_logloss', 0.0)):.6f}, "
        f"ece={float(control.get('ece', 0.0)):.6f}"
    )
    lines.append("")
    lines.append("## Candidates")
    for item in aggregate.get("summaries", []):
        lines.append(
            f"- {item.get('variant')}: seeds={item.get('seed_count', 0)}, "
            f"logloss={float(item.get('mean_val_logloss', 0.0)):.6f}, "
            f"draw_top2={float(item.get('draw_top2', 0.0)):.6f}, "
            f"p_prior_ll={float(item.get('p_prior_logloss', 0.0)):.6f}, "
            f"final_minus_prior={float(item.get('final_minus_prior_logloss', 0.0)):.6f}, "
            f"verdict={','.join(item.get('verdict', []))}"
        )
    lines.append("")
    lines.append(f"Final verdict: `{','.join(aggregate.get('verdict', []))}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed P12 explicit score-prior residual matrix")
    parser.add_argument("--variants", default=",".join(FIXED_P12_VARIANTS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in FIXED_P12_SEEDS))
    parser.add_argument("--data", default="data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl")
    parser.add_argument("--train-ids", default="data/odds_real/splits_v6/train_match_ids.txt")
    parser.add_argument("--val-ids", default="data/odds_real/splits_v6/val_match_ids.txt")
    parser.add_argument("--out-root", default="runs/p12_score_prior_residual")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-test", action="store_true")
    return parser


def _base_args(args: argparse.Namespace) -> list[str]:
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
        raise SystemExit(f"P12 protocol audit failed: {audit['hard_fail_reasons']}")
    (out_root / "manifest.json").write_text(
        json.dumps({"phase": "P12", "control": FIXED_P12_CONTROL, "variants": variants, "seeds": seeds, "registry": FIXED_P12_VARIANTS}, indent=2),
        encoding="utf-8",
    )
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        run_dir = out_root / f"{FIXED_P12_CONTROL}_seed{seed}"
        if args.skip_existing and (run_dir / "report.json").exists():
            print(f"SKIP existing {run_dir}", flush=True)
            continue
        if (run_dir / "report.json").exists() or (run_dir / "best_model.pth").exists():
            raise SystemExit(f"Refusing to overwrite existing run: {run_dir}")
        cmd = [
            sys.executable,
            "tools/p11_train_score_count_structure.py",
            *_base_args(args),
            "--seed",
            str(seed),
            "--variant",
            FIXED_P12_CONTROL,
            "--out-dir",
            str(run_dir),
        ]
        if args.max_samples > 0:
            cmd.extend(["--max-samples", str(args.max_samples)])
        print(f"RUN {FIXED_P12_CONTROL} seed={seed} -> {run_dir}", flush=True)
        _run_command(cmd, log_dir / f"{FIXED_P12_CONTROL}_seed{seed}.log")
        print(f"DONE {FIXED_P12_CONTROL} seed={seed}", flush=True)

    for variant in variants:
        cfg = FIXED_P12_VARIANTS[variant]
        for seed in seeds:
            run_dir = out_root / f"{variant}_seed{seed}"
            if args.skip_existing and (run_dir / "report.json").exists():
                print(f"SKIP existing {run_dir}", flush=True)
                continue
            if (run_dir / "report.json").exists() or (run_dir / "best_model.pth").exists():
                raise SystemExit(f"Refusing to overwrite existing run: {run_dir}")
            cmd = [
                sys.executable,
                "tools/p12_train_score_prior_residual.py",
                *_base_args(args),
                "--prior-mode",
                str(cfg["prior_mode"]),
                "--blend-alpha",
                str(cfg["blend_alpha"]),
                "--count-loss-weight",
                "0.10",
                "--residual-l2-weight",
                str(cfg["residual_l2_weight"]),
                "--anchor-ce-weight",
                str(cfg["anchor_ce_weight"]),
                "--seed",
                str(seed),
                "--variant",
                variant,
                "--out-dir",
                str(run_dir),
            ]
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
