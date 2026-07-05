"""Run the fixed P9 market-prior anchored residual matrix."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


class P9ProtocolError(RuntimeError):
    pass


FIXED_P9_SEEDS = [42, 123, 2025]
FIXED_P9_VARIANTS: dict[str, dict[str, float]] = {
    "p9_a_p6_explicit_market_anchor": {
        "draw_delta_l2_weight": 0.0,
        "anchor_kl_weight": 0.0,
        "true_draw_floor_weight": 0.0,
    },
    "p9_b_draw_delta_l2_001": {
        "draw_delta_l2_weight": 0.01,
        "anchor_kl_weight": 0.0,
        "true_draw_floor_weight": 0.0,
    },
    "p9_c_anchor_kl_001": {
        "draw_delta_l2_weight": 0.0,
        "anchor_kl_weight": 0.01,
        "true_draw_floor_weight": 0.0,
    },
    "p9_d_true_draw_floor_010": {
        "draw_delta_l2_weight": 0.0,
        "anchor_kl_weight": 0.0,
        "true_draw_floor_weight": 0.10,
    },
}


def parse_registered_variants(raw: str) -> list[str]:
    variants = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in variants if item not in FIXED_P9_VARIANTS]
    if unknown:
        raise ValueError(f"Unregistered P9 variant(s): {unknown}")
    if len(set(variants)) != len(variants):
        raise ValueError("Duplicate P9 variants are not allowed")
    return variants


def parse_registered_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    unknown = [item for item in seeds if item not in FIXED_P9_SEEDS]
    if unknown:
        raise ValueError(f"Unregistered P9 seed(s): {unknown}")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate P9 seeds are not allowed")
    return seeds


def build_protocol_audit(train_ids: str, val_ids: str, variants: list[str], seeds: list[int]) -> dict[str, Any]:
    if "test" in Path(train_ids).name.lower() or "test" in Path(val_ids).name.lower():
        raise P9ProtocolError("P9 refuses test split paths")
    hard_fail_reasons = []
    if variants != list(FIXED_P9_VARIANTS):
        hard_fail_reasons.append("P9_FAIL_MATRIX_VARIANT_MUTATION")
    if seeds != FIXED_P9_SEEDS:
        hard_fail_reasons.append("P9_FAIL_SEED_LIST_MUTATION")
    return {
        "phase": "P9",
        "no_test_split_loaded": True,
        "no_test_artifact_written": True,
        "official_val_fixed_matrix_only": variants == list(FIXED_P9_VARIANTS),
        "candidate_count": len(variants),
        "candidate_count_mutation": variants != list(FIXED_P9_VARIANTS),
        "seed_list": seeds,
        "seed_list_mutation": seeds != FIXED_P9_SEEDS,
        "p6_default_behavior_changed": False,
        "hard_fail_reasons": hard_fail_reasons,
    }


def _read_report(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "report.json").read_text(encoding="utf-8"))


def aggregate_results(out_root: Path, variants: list[str], seeds: list[int]) -> dict[str, Any]:
    rows = []
    for variant in variants:
        for seed in seeds:
            run_dir = out_root / f"{variant}_seed{seed}"
            if not (run_dir / "report.json").exists():
                continue
            report = _read_report(run_dir)
            val = report.get("val_metrics", {})
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "val_logloss": float(val.get("logloss", report.get("best_val_logloss", 0.0))),
                    "ece": float(val.get("ece", 0.0)),
                    "draw_class_nll": float(val.get("draw_class_nll", 0.0)),
                    "draw_recall": float(val.get("draw_recall", 0.0)),
                    "draw_precision": float(val.get("draw_precision", 0.0)),
                    "draw_top2": float(val.get("draw_top2_recall", 0.0)),
                    "mean_p_draw_true_draw": float(val.get("mean_p_draw_on_true_draw", 0.0)),
                    "mean_draw_margin_to_top": float(val.get("mean_draw_margin_to_top", 0.0)),
                    "argmax_draw_count": int(val.get("argmax_draw_count", 0)),
                }
            )
    summaries = []
    for variant in variants:
        vr = [row for row in rows if row["variant"] == variant]
        if not vr:
            summaries.append({"variant": variant, "seed_count": 0, "status": "missing"})
            continue
        losses = [row["val_logloss"] for row in vr]
        item = {
            "variant": variant,
            "seed_count": len(vr),
            "status": "complete" if len(vr) == len(seeds) else "incomplete",
            "mean_val_logloss": statistics.fmean(losses),
            "std_val_logloss": statistics.pstdev(losses) if len(losses) > 1 else 0.0,
            "mean_ece": statistics.fmean([row["ece"] for row in vr]),
            "draw_class_nll": statistics.fmean([row["draw_class_nll"] for row in vr]),
            "draw_recall": statistics.fmean([row["draw_recall"] for row in vr]),
            "draw_precision": statistics.fmean([row["draw_precision"] for row in vr]),
            "draw_top2": statistics.fmean([row["draw_top2"] for row in vr]),
            "mean_p_draw_true_draw": statistics.fmean([row["mean_p_draw_true_draw"] for row in vr]),
            "mean_draw_margin_to_top": statistics.fmean([row["mean_draw_margin_to_top"] for row in vr]),
        }
        item["verdict"] = p9_variant_verdict(item)
        summaries.append(item)
    best = min([item for item in summaries if item.get("seed_count", 0) > 0], key=lambda x: x.get("mean_val_logloss", 999), default=None)
    return {"rows": rows, "summaries": summaries, "best_variant": best, "verdict": p9_matrix_verdict(summaries)}


def p9_variant_verdict(item: dict[str, Any]) -> list[str]:
    if item.get("status") != "complete":
        return ["P9_INCOMPLETE"]
    passed = (
        item["mean_val_logloss"] <= 0.93760
        and item["draw_class_nll"] <= 1.620
        and item["draw_top2"] >= 0.38
        and item["mean_p_draw_true_draw"] >= 0.215
        and item["mean_draw_margin_to_top"] <= 0.36
        and item["std_val_logloss"] <= 0.00060
    )
    return ["P9_MAINLINE_CANDIDATE"] if passed else ["P9_FAIL_GATES"]


def p9_matrix_verdict(summaries: list[dict[str, Any]]) -> list[str]:
    verdict = []
    complete = [item for item in summaries if item.get("status") == "complete"]
    if len(complete) != len(FIXED_P9_VARIANTS):
        verdict.append("P9_MATRIX_INCOMPLETE")
    if any("P9_MAINLINE_CANDIDATE" in item.get("verdict", []) for item in complete):
        verdict.append("P9_HAS_MAINLINE_CANDIDATE")
    else:
        verdict.append("P9_NO_MAINLINE_CANDIDATE")
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
    parser = argparse.ArgumentParser(description="Run fixed P9 market-prior anchored residual matrix")
    parser.add_argument("--variants", default=",".join(FIXED_P9_VARIANTS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in FIXED_P9_SEEDS))
    parser.add_argument("--data", default="data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl")
    parser.add_argument("--train-ids", default="data/odds_real/splits_v6/train_match_ids.txt")
    parser.add_argument("--val-ids", default="data/odds_real/splits_v6/val_match_ids.txt")
    parser.add_argument("--out-root", default="runs/p9_market_prior_residual")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-test", action="store_true")
    return parser


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
        raise SystemExit(f"P9 protocol audit failed: {audit['hard_fail_reasons']}")
    (out_root / "manifest.json").write_text(
        json.dumps({"phase": "P9", "variants": variants, "seeds": seeds, "registry": FIXED_P9_VARIANTS}, indent=2),
        encoding="utf-8",
    )
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        cfg = FIXED_P9_VARIANTS[variant]
        for seed in seeds:
            run_dir = out_root / f"{variant}_seed{seed}"
            if args.skip_existing and (run_dir / "report.json").exists():
                print(f"SKIP existing {run_dir}", flush=True)
                continue
            if (run_dir / "report.json").exists() or (run_dir / "best_model.pth").exists():
                raise SystemExit(f"Refusing to overwrite existing run: {run_dir}")
            cmd = [
                sys.executable,
                "tools/p9_train_market_prior_residual.py",
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
                "--draw-delta-l2-weight",
                str(cfg["draw_delta_l2_weight"]),
                "--anchor-kl-weight",
                str(cfg["anchor_kl_weight"]),
                "--true-draw-floor-weight",
                str(cfg["true_draw_floor_weight"]),
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
            if args.max_samples > 0:
                cmd.extend(["--max-samples", str(args.max_samples)])
            log_path = log_dir / f"{variant}_seed{seed}.log"
            print(f"RUN {variant} seed={seed} -> {run_dir}", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
            if result.returncode != 0:
                raise SystemExit(result.returncode)
            print(f"DONE {variant} seed={seed}", flush=True)
    aggregate = aggregate_results(out_root, variants, seeds)
    write_results_by_seed(aggregate["rows"], out_root / "results_by_seed.csv")
    (out_root / "results_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
