"""Run the fixed P10 market draw pairwise ranking matrix."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


class P10ProtocolError(RuntimeError):
    pass


FIXED_P10_SEEDS = [42, 123, 2025]
FIXED_P10_CONTROL = "p10_control_p6_euro_default"
FIXED_P10_VARIANTS: dict[str, dict[str, Any]] = {
    "p10_a_pairwise_all_005": {
        "market_draw_pairwise_weight": 0.05,
        "market_draw_pairwise_mode": "all",
        "true_draw_top2_margin_weight": 0.0,
    },
    "p10_b_pairwise_all_020": {
        "market_draw_pairwise_weight": 0.20,
        "market_draw_pairwise_mode": "all",
        "true_draw_top2_margin_weight": 0.0,
    },
    "p10_c_pairwise_market_top2_010": {
        "market_draw_pairwise_weight": 0.10,
        "market_draw_pairwise_mode": "market_draw_top2",
        "true_draw_top2_margin_weight": 0.0,
    },
    "p10_d_pairwise_top2_true_margin": {
        "market_draw_pairwise_weight": 0.10,
        "market_draw_pairwise_mode": "market_draw_top2",
        "true_draw_top2_margin_weight": 0.05,
    },
}


def parse_registered_variants(raw: str) -> list[str]:
    variants = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in variants if item not in FIXED_P10_VARIANTS]
    if unknown:
        raise ValueError(f"Unregistered P10 variant(s): {unknown}")
    if len(set(variants)) != len(variants):
        raise ValueError("Duplicate P10 variants are not allowed")
    return variants


def parse_registered_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    unknown = [item for item in seeds if item not in FIXED_P10_SEEDS]
    if unknown:
        raise ValueError(f"Unregistered P10 seed(s): {unknown}")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate P10 seeds are not allowed")
    return seeds


def build_protocol_audit(train_ids: str, val_ids: str, variants: list[str], seeds: list[int]) -> dict[str, Any]:
    if "test" in Path(train_ids).name.lower() or "test" in Path(val_ids).name.lower():
        raise P10ProtocolError("P10 refuses test split paths")
    hard_fail_reasons = []
    if variants != list(FIXED_P10_VARIANTS):
        hard_fail_reasons.append("P10_FAIL_MATRIX_VARIANT_MUTATION")
    if seeds != FIXED_P10_SEEDS:
        hard_fail_reasons.append("P10_FAIL_SEED_LIST_MUTATION")
    return {
        "phase": "P10",
        "no_test_split_loaded": True,
        "no_test_artifact_written": True,
        "official_val_fixed_matrix_only": variants == list(FIXED_P10_VARIANTS),
        "control_required": True,
        "control_name": FIXED_P10_CONTROL,
        "control_seed_list": seeds,
        "candidate_count": len(variants),
        "candidate_count_mutation": variants != list(FIXED_P10_VARIANTS),
        "seed_list": seeds,
        "seed_list_mutation": seeds != FIXED_P10_SEEDS,
        "expected_control_runs": len(seeds),
        "expected_candidate_runs": len(variants) * len(seeds),
        "expected_total_runs": len(seeds) + len(variants) * len(seeds),
        "p6_default_behavior_changed": False,
        "hard_fail_reasons": hard_fail_reasons,
    }


def _read_report(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "report.json").read_text(encoding="utf-8"))


def _row_from_report(run_dir: Path, variant: str, seed: int, is_control: bool = False) -> dict[str, Any]:
    report = _read_report(run_dir)
    val = report.get("val_metrics", {})
    market_pairwise = report.get("market_pairwise_metrics", {}).get("val", {})
    coverage = report.get("teacher_coverage", {})
    return {
        "variant": variant,
        "seed": seed,
        "is_control": is_control,
        "val_logloss": float(val.get("logloss", report.get("best_val_logloss", 0.0))),
        "ece": float(val.get("ece", 0.0)),
        "draw_class_nll": float(val.get("draw_class_nll", 0.0)),
        "draw_recall": float(val.get("draw_recall", 0.0)),
        "draw_precision": float(val.get("draw_precision", 0.0)),
        "draw_top2": float(val.get("draw_top2_recall", val.get("draw_top2", 0.0))),
        "mean_p_draw_true_draw": float(val.get("mean_p_draw_on_true_draw", 0.0)),
        "mean_draw_margin_to_top": float(val.get("mean_draw_margin_to_top", 0.0)),
        "argmax_draw_count": int(val.get("argmax_draw_count", 0)),
        "market_pairwise_agreement": float(market_pairwise.get("market_pairwise_agreement", 0.0) or 0.0),
        "model_vs_market_draw_corr": market_pairwise.get("model_vs_market_draw_corr"),
        "model_vs_market_draw_mae": float(market_pairwise.get("model_vs_market_draw_mae", 0.0) or 0.0),
        "val_teacher_coverage": float(coverage.get("val_teacher_coverage", 1.0) or 0.0),
        "train_teacher_coverage": float(coverage.get("train_teacher_coverage", 1.0) or 0.0),
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
        "market_pairwise_agreement": _mean(vr, "market_pairwise_agreement"),
        "model_vs_market_draw_mae": _mean(vr, "model_vs_market_draw_mae"),
        "train_teacher_coverage": _mean(vr, "train_teacher_coverage"),
        "val_teacher_coverage": _mean(vr, "val_teacher_coverage"),
    }


def p10_variant_verdict(item: dict[str, Any], control_summary: dict[str, Any]) -> list[str]:
    if item.get("status") != "complete":
        return ["P10_INCOMPLETE"]
    if item.get("train_teacher_coverage", 1.0) < 0.99 or item.get("val_teacher_coverage", 1.0) < 0.99:
        return ["P10_BLOCKED_BY_DATA_AUDIT"]
    control_logloss = float(control_summary.get("mean_val_logloss", 999.0))
    control_ece = float(control_summary.get("mean_ece", 999.0))
    mainline = (
        item["mean_val_logloss"] <= control_logloss + 0.0003
        and item["mean_ece"] <= max(control_ece + 0.005, 0.033)
        and item["draw_top2"] >= 0.42
        and item["draw_class_nll"] <= 1.5990
        and item["mean_p_draw_true_draw"] >= 0.215
        and item["mean_draw_margin_to_top"] <= 0.375
    )
    if mainline:
        return ["P10_MAINLINE_CANDIDATE"]
    directional = (
        item["draw_top2"] >= 0.42
        and item["mean_p_draw_true_draw"] >= 0.215
        and item["mean_val_logloss"] <= control_logloss + 0.0008
    )
    if directional:
        return ["P10_CONTINUE_A_DIRECTION"]
    return ["P10_FAIL_GATES"]


def p10_matrix_verdict(summaries: list[dict[str, Any]], control_summary: dict[str, Any], seeds: list[int]) -> list[str]:
    if control_summary.get("status") != "complete" or control_summary.get("seed_count") != len(seeds):
        return ["P10_MATRIX_INCOMPLETE"]
    complete = [item for item in summaries if item.get("status") == "complete"]
    if len(complete) != len(FIXED_P10_VARIANTS):
        return ["P10_MATRIX_INCOMPLETE"]
    if any("P10_BLOCKED_BY_DATA_AUDIT" in item.get("verdict", []) for item in complete):
        return ["P10_BLOCKED_BY_DATA_AUDIT"]
    if any("P10_MAINLINE_CANDIDATE" in item.get("verdict", []) for item in complete):
        return ["P10_MAINLINE_CANDIDATE"]
    if any("P10_CONTINUE_A_DIRECTION" in item.get("verdict", []) for item in complete):
        return ["P10_CONTINUE_A_DIRECTION"]
    if all(float(item.get("draw_top2", 0.0)) < 0.38 for item in complete):
        return ["P10_REJECT_A_MOVE_TO_SCORE_STRUCTURE"]
    if all(float(item.get("mean_p_draw_true_draw", 0.0)) < 0.210 for item in complete):
        return ["P10_REJECT_A_MOVE_TO_SCORE_STRUCTURE"]
    return ["P10_NO_ACCEPTED_VARIANT"]


def aggregate_results(out_root: Path, variants: list[str], seeds: list[int]) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        run_dir = out_root / f"{FIXED_P10_CONTROL}_seed{seed}"
        if (run_dir / "report.json").exists():
            rows.append(_row_from_report(run_dir, FIXED_P10_CONTROL, seed, is_control=True))
    control_summary = summarize_rows(rows, FIXED_P10_CONTROL, seeds)
    summaries = []
    for variant in variants:
        for seed in seeds:
            run_dir = out_root / f"{variant}_seed{seed}"
            if (run_dir / "report.json").exists():
                rows.append(_row_from_report(run_dir, variant, seed, is_control=False))
        item = summarize_rows(rows, variant, seeds)
        item["verdict"] = p10_variant_verdict(item, control_summary)
        summaries.append(item)
    best = min([item for item in summaries if item.get("seed_count", 0) > 0], key=lambda x: x.get("mean_val_logloss", 999), default=None)
    return {
        "rows": rows,
        "control_summary": control_summary,
        "summaries": summaries,
        "best_variant": best,
        "verdict": p10_matrix_verdict(summaries, control_summary, seeds),
        "historical_references": {
            "p9_c_anchor_kl_001_mean_val_logloss": 0.9372987747,
            "p9_d_true_draw_floor_010_draw_class_nll": 1.5990122159,
            "p9_d_true_draw_floor_010_draw_top2": 0.3818840683,
            "market_close_draw_top2": 0.6315217614,
            "market_close_mean_p_draw_true_draw": 0.2532442212,
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
    lines = ["# P10 Market Draw Pairwise Matrix", ""]
    control = aggregate.get("control_summary", {})
    lines.append("## Same-Protocol Control")
    lines.append(
        f"- {FIXED_P10_CONTROL}: seeds={control.get('seed_count', 0)}, "
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
            f"verdict={','.join(item.get('verdict', []))}"
        )
    lines.append("")
    lines.append(f"Final verdict: `{','.join(aggregate.get('verdict', []))}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed P10 market draw pairwise ranking matrix")
    parser.add_argument("--variants", default=",".join(FIXED_P10_VARIANTS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in FIXED_P10_SEEDS))
    parser.add_argument("--data", default="data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl")
    parser.add_argument("--train-ids", default="data/odds_real/splits_v6/train_match_ids.txt")
    parser.add_argument("--val-ids", default="data/odds_real/splits_v6/val_match_ids.txt")
    parser.add_argument("--out-root", default="runs/p10_market_draw_pairwise")
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
        raise SystemExit(f"P10 protocol audit failed: {audit['hard_fail_reasons']}")
    (out_root / "manifest.json").write_text(
        json.dumps(
            {
                "phase": "P10",
                "control": FIXED_P10_CONTROL,
                "variants": variants,
                "seeds": seeds,
                "registry": FIXED_P10_VARIANTS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        run_dir = out_root / f"{FIXED_P10_CONTROL}_seed{seed}"
        if args.skip_existing and (run_dir / "report.json").exists():
            print(f"SKIP existing {run_dir}", flush=True)
            continue
        if (run_dir / "report.json").exists() or (run_dir / "best_model.pth").exists():
            raise SystemExit(f"Refusing to overwrite existing run: {run_dir}")
        cmd = [
            sys.executable,
            "tools/p10_train_market_draw_pairwise.py",
            *_base_model_args(args),
            "--seed",
            str(seed),
            "--variant",
            FIXED_P10_CONTROL,
            "--out-dir",
            str(run_dir),
        ]
        if args.max_samples > 0:
            cmd.extend(["--max-samples", str(args.max_samples)])
        print(f"RUN {FIXED_P10_CONTROL} seed={seed} -> {run_dir}", flush=True)
        _run_command(cmd, log_dir / f"{FIXED_P10_CONTROL}_seed{seed}.log")
        print(f"DONE {FIXED_P10_CONTROL} seed={seed}", flush=True)

    for variant in variants:
        cfg = FIXED_P10_VARIANTS[variant]
        for seed in seeds:
            run_dir = out_root / f"{variant}_seed{seed}"
            if args.skip_existing and (run_dir / "report.json").exists():
                print(f"SKIP existing {run_dir}", flush=True)
                continue
            if (run_dir / "report.json").exists() or (run_dir / "best_model.pth").exists():
                raise SystemExit(f"Refusing to overwrite existing run: {run_dir}")
            cmd = [
                sys.executable,
                "tools/p10_train_market_draw_pairwise.py",
                *_base_model_args(args),
                "--enable-market-draw-pairwise",
                "--market-draw-pairwise-weight",
                str(cfg["market_draw_pairwise_weight"]),
                "--market-draw-pairwise-mode",
                str(cfg["market_draw_pairwise_mode"]),
                "--seed",
                str(seed),
                "--variant",
                variant,
                "--out-dir",
                str(run_dir),
            ]
            if float(cfg.get("true_draw_top2_margin_weight", 0.0)) > 0:
                cmd.extend(
                    [
                        "--enable-true-draw-top2-margin",
                        "--true-draw-top2-margin-weight",
                        str(cfg["true_draw_top2_margin_weight"]),
                    ]
                )
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
