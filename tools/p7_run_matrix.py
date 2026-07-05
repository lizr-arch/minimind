"""Run the fixed P7 draw-first factorized training matrix.

The runner intentionally refuses dynamic candidates and any test split path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


FIXED_P7_SEEDS = [42, 123, 2025]
FIXED_P7_VARIANTS: dict[str, dict[str, float]] = {
    "p7_a_factorized_ce_only": {"lambda_draw_bce": 0.0, "lambda_cond_ha_aux": 0.0},
    "p7_b_factorized_ce_draw_bce_005": {"lambda_draw_bce": 0.005, "lambda_cond_ha_aux": 0.0},
    "p7_c_factorized_ce_draw_bce_010": {"lambda_draw_bce": 0.010, "lambda_cond_ha_aux": 0.0},
    "p7_d_factorized_ce_cond_ha_aux_005": {"lambda_draw_bce": 0.0, "lambda_cond_ha_aux": 0.005},
}

P6_MAINLINE_MEAN_VAL_LOGLOSS = 0.9373360872
P7_MAINLINE_GATES = {
    "mean_val_logloss": 0.9373360872,
    "std_val_logloss": 0.0003500000,
    "mean_ece": 0.0300000000,
    "draw_class_nll": 1.6200000000,
    "draw_top2": 0.3400000000,
    "mean_p_draw_true_draw": 0.2150000000,
    "mean_draw_margin_to_top": 0.3300000000,
    "draw_recall_argmax": 0.0200000000,
    "draw_precision_argmax": 0.1800000000,
}
P7_PARTIAL_GATES = {
    "mean_val_logloss": 0.9376000000,
    "draw_class_nll": 1.6250000000,
    "draw_top2": 0.3350000000,
    "mean_draw_margin_to_top": 0.3350000000,
    "std_val_logloss": 0.0005000000,
}


def parse_registered_variants(raw: str) -> list[str]:
    variants = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in variants if item not in FIXED_P7_VARIANTS]
    if unknown:
        raise ValueError(f"Unregistered P7 variant(s): {unknown}")
    if len(set(variants)) != len(variants):
        raise ValueError("Duplicate P7 variants are not allowed")
    return variants


def parse_registered_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    unknown = [item for item in seeds if item not in FIXED_P7_SEEDS]
    if unknown:
        raise ValueError(f"Unregistered P7 seed(s): {unknown}")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate P7 seeds are not allowed")
    return seeds


def _has_test_path(*paths: str) -> bool:
    return any("test" in Path(path).name.lower() for path in paths)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def _config_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def build_protocol_audit(
    baseline_path: str,
    data_path: str,
    train_ids: str,
    val_ids: str,
    variants: list[str],
    seeds: list[int],
) -> dict[str, Any]:
    hard_fail_reasons: list[str] = []
    no_test_split_loaded = not _has_test_path(train_ids, val_ids)
    if not no_test_split_loaded:
        hard_fail_reasons.append("P7_FAIL_TEST_SPLIT_REFERENCE")
    if variants != list(FIXED_P7_VARIANTS):
        hard_fail_reasons.append("P7_FAIL_MATRIX_VARIANT_MUTATION")
    if seeds != FIXED_P7_SEEDS:
        hard_fail_reasons.append("P7_FAIL_SEED_LIST_MUTATION")
    payload = {
        "phase": "P7",
        "no_test_split_loaded": no_test_split_loaded,
        "no_test_artifact_written": True,
        "official_val_fixed_matrix_only": variants == list(FIXED_P7_VARIANTS),
        "candidate_count": len(variants),
        "candidate_count_mutation": variants != list(FIXED_P7_VARIANTS),
        "seed_list": seeds,
        "seed_list_mutation": seeds != FIXED_P7_SEEDS,
        "baseline_path": baseline_path,
        "data_path": data_path,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "git_commit": _git_commit(),
        "hard_fail_reasons": hard_fail_reasons,
    }
    payload["config_hash"] = _config_hash(payload)
    return payload


def write_protocol_audit(audit: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")


def _read_report(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "report.json").read_text(encoding="utf-8"))


def aggregate_results(out_root: Path, variants: list[str], seeds: list[int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        for seed in seeds:
            run_dir = out_root / f"{variant}_seed{seed}"
            if not (run_dir / "report.json").exists():
                continue
            report = _read_report(run_dir)
            val = report.get("val_metrics", {})
            row = {
                "variant": variant,
                "seed": seed,
                "val_logloss": float(val.get("logloss", report.get("best_val_logloss", 0.0))),
                "ece": float(val.get("ece", 0.0)),
                "draw_class_nll": float(val.get("draw_class_nll", 0.0)),
                "draw_recall_argmax": float(val.get("draw_recall", 0.0)),
                "draw_precision_argmax": float(val.get("draw_precision", 0.0)),
                "draw_top2": float(val.get("draw_top2_recall", 0.0)),
                "mean_p_draw_true_draw": float(val.get("mean_p_draw_on_true_draw", 0.0)),
                "mean_draw_margin_to_top": float(val.get("mean_draw_margin_to_top", 0.0)),
            }
            rows.append(row)

    summaries: list[dict[str, Any]] = []
    for variant in variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        if not variant_rows:
            summaries.append({"variant": variant, "seed_count": 0, "status": "missing"})
            continue
        losses = [row["val_logloss"] for row in variant_rows]
        summary = {
                "variant": variant,
                "seed_count": len(variant_rows),
                "status": "complete" if len(variant_rows) == len(seeds) else "incomplete",
                "mean_val_logloss": statistics.fmean(losses),
                "std_val_logloss": statistics.pstdev(losses) if len(losses) > 1 else 0.0,
                "mean_ece": statistics.fmean([row["ece"] for row in variant_rows]),
                "draw_class_nll": statistics.fmean([row["draw_class_nll"] for row in variant_rows]),
                "draw_recall_argmax": statistics.fmean([row["draw_recall_argmax"] for row in variant_rows]),
                "draw_precision_argmax": statistics.fmean([row["draw_precision_argmax"] for row in variant_rows]),
                "draw_top2": statistics.fmean([row["draw_top2"] for row in variant_rows]),
                "mean_p_draw_true_draw": statistics.fmean([row["mean_p_draw_true_draw"] for row in variant_rows]),
                "mean_draw_margin_to_top": statistics.fmean(
                    [row["mean_draw_margin_to_top"] for row in variant_rows]
                ),
            }
        summary["p_candidate_better_vs_p6"] = 1.0 if summary["mean_val_logloss"] < P6_MAINLINE_MEAN_VAL_LOGLOSS else 0.0
        summary["sanity_audit_vs_p6"] = "PASS" if summary["mean_val_logloss"] <= P6_MAINLINE_MEAN_VAL_LOGLOSS else "FAIL"
        summary["verdict"] = p7_variant_verdict(summary)
        summaries.append(summary)
    best = min(
        [item for item in summaries if item.get("seed_count", 0) > 0],
        key=lambda item: item.get("mean_val_logloss", float("inf")),
        default=None,
    )
    return {
        "rows": rows,
        "summaries": summaries,
        "best_variant": best,
        "p6_mainline_mean_val_logloss": P6_MAINLINE_MEAN_VAL_LOGLOSS,
        "verdict": p7_matrix_verdict(summaries),
    }


def p7_variant_verdict(summary: dict[str, Any]) -> list[str]:
    verdict: list[str] = []
    if summary.get("status") != "complete":
        verdict.append("P7_INCOMPLETE")
        return verdict
    mainline_pass = (
        summary["mean_val_logloss"] <= P7_MAINLINE_GATES["mean_val_logloss"]
        and summary["std_val_logloss"] <= P7_MAINLINE_GATES["std_val_logloss"]
        and summary["mean_ece"] <= P7_MAINLINE_GATES["mean_ece"]
        and summary["draw_class_nll"] <= P7_MAINLINE_GATES["draw_class_nll"]
        and summary["draw_top2"] >= P7_MAINLINE_GATES["draw_top2"]
        and summary["mean_p_draw_true_draw"] >= P7_MAINLINE_GATES["mean_p_draw_true_draw"]
        and summary["mean_draw_margin_to_top"] <= P7_MAINLINE_GATES["mean_draw_margin_to_top"]
        and summary["draw_recall_argmax"] >= P7_MAINLINE_GATES["draw_recall_argmax"]
        and summary["draw_precision_argmax"] >= P7_MAINLINE_GATES["draw_precision_argmax"]
        and summary["sanity_audit_vs_p6"] == "PASS"
    )
    partial_pass = (
        summary["mean_val_logloss"] <= P7_PARTIAL_GATES["mean_val_logloss"]
        and summary["draw_class_nll"] <= P7_PARTIAL_GATES["draw_class_nll"]
        and summary["draw_top2"] >= P7_PARTIAL_GATES["draw_top2"]
        and summary["mean_draw_margin_to_top"] <= P7_PARTIAL_GATES["mean_draw_margin_to_top"]
        and summary["std_val_logloss"] <= P7_PARTIAL_GATES["std_val_logloss"]
    )
    reject_direction = (
        summary["draw_recall_argmax"] < 0.0100000000
        and summary["draw_top2"] < 0.3300000000
        and summary["draw_class_nll"] > 1.6350000000
    )
    if mainline_pass:
        verdict.append("P7_MAINLINE_PASS")
    elif partial_pass:
        verdict.append("P7_PARTIAL_SUCCESS")
    else:
        verdict.append("P7_FAIL_GATES")
    if reject_direction:
        verdict.append("P7_REJECT_DIRECTION")
    if summary["sanity_audit_vs_p6"] != "PASS":
        verdict.append("P7_FAILS_VS_P6")
    return verdict


def p7_matrix_verdict(summaries: list[dict[str, Any]]) -> list[str]:
    complete = [item for item in summaries if item.get("status") == "complete"]
    verdict: list[str] = []
    if len(complete) != len(FIXED_P7_VARIANTS):
        verdict.append("P7_MATRIX_INCOMPLETE")
    if any("P7_MAINLINE_PASS" in item.get("verdict", []) for item in complete):
        verdict.append("P7_HAS_MAINLINE_PASS")
    elif any("P7_PARTIAL_SUCCESS" in item.get("verdict", []) for item in complete):
        verdict.append("P7_HAS_PARTIAL_SUCCESS")
    else:
        verdict.append("P7_NO_ACCEPTED_VARIANT")
    if all("P7_REJECT_DIRECTION" in item.get("verdict", []) for item in complete) and complete:
        verdict.append("P7_REJECT_DIRECTION")
    if all("P7_FAILS_VS_P6" in item.get("verdict", []) for item in complete) and complete:
        verdict.append("P7_ALL_VARIANTS_FAIL_VS_P6")
    return verdict


def write_results_by_seed(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed P7 draw-first matrix")
    parser.add_argument("--mode", default="official_3seed", choices=["official_3seed"])
    parser.add_argument("--variants", default=",".join(FIXED_P7_VARIANTS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in FIXED_P7_SEEDS))
    parser.add_argument("--data", default="data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl")
    parser.add_argument("--train-ids", default="data/odds_real/splits_v6/train_match_ids.txt")
    parser.add_argument("--val-ids", default="data/odds_real/splits_v6/val_match_ids.txt")
    parser.add_argument("--baseline-path", default="runs/p6_matrix/p6_euro_default")
    parser.add_argument("--out-root", default="runs/p7_draw_first_matrix")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-test", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.no_test:
        raise SystemExit("Refusing to run without --no-test")
    variants = parse_registered_variants(args.variants)
    seeds = parse_registered_seeds(args.seeds)
    audit = build_protocol_audit(
        baseline_path=args.baseline_path,
        data_path=args.data,
        train_ids=args.train_ids,
        val_ids=args.val_ids,
        variants=variants,
        seeds=seeds,
    )
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    write_protocol_audit(audit, out_root / "protocol_audit.json")
    if audit["hard_fail_reasons"]:
        raise SystemExit(f"P7 protocol audit failed: {audit['hard_fail_reasons']}")

    manifest = {
        "phase": "P7",
        "variants": variants,
        "seeds": seeds,
        "fixed_variant_registry": FIXED_P7_VARIANTS,
        "protocol_audit": str(out_root / "protocol_audit.json"),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        variant_config = FIXED_P7_VARIANTS[variant]
        for seed in seeds:
            run_dir = out_root / f"{variant}_seed{seed}"
            existing_outputs = [run_dir / "report.json", run_dir / "best_model.pth"]
            if args.skip_existing and any(path.exists() for path in existing_outputs):
                print(f"SKIP existing {run_dir}", flush=True)
                continue
            if any(path.exists() for path in existing_outputs):
                raise SystemExit(f"Refusing to overwrite existing run: {run_dir}")
            cmd = [
                sys.executable,
                "tools/p7_train_draw_first.py",
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
                "--lambda-draw-bce",
                str(variant_config["lambda_draw_bce"]),
                "--lambda-cond-ha-aux",
                str(variant_config["lambda_cond_ha_aux"]),
                "--scaling",
                "robust",
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--variant",
                variant,
                "--out-dir",
                str(run_dir),
            ]
            log_path = log_dir / f"{variant}_seed{seed}.log"
            print(f"RUN {variant} seed={seed} config={variant_config} -> {run_dir}", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
            if result.returncode != 0:
                print(f"FAILED {variant} seed={seed}; see {log_path}", flush=True)
                raise SystemExit(result.returncode)
            print(f"DONE {variant} seed={seed}; log={log_path}", flush=True)

    aggregate = aggregate_results(out_root, variants, seeds)
    write_results_by_seed(aggregate["rows"], out_root / "results_by_seed.csv")
    (out_root / "results_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    lines = ["# P7 Draw Diagnostics Summary", ""]
    for item in aggregate["summaries"]:
        lines.append(
            f"- {item['variant']}: n={item.get('seed_count', 0)}, "
            f"mean_logloss={item.get('mean_val_logloss', 0.0):.6f}, "
            f"draw_recall={item.get('draw_recall_argmax', 0.0):.6f}, "
            f"draw_top2={item.get('draw_top2', 0.0):.6f}"
        )
    (out_root / "draw_diagnostics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
