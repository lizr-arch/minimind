"""Generate P3.1 diagnostics summaries and final report."""

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch


def _read_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict) -> None:
    os.makedirs(Path(path).parent, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _metrics_from_report(report: dict) -> dict:
    val = report.get("val_metrics") or report.get("final_val") or {}
    train = report.get("train_metrics") or {}
    train_acc = train.get("final_train_acc", train.get("accuracy"))
    config = report.get("config") or {}
    return {
        "train_acc": train_acc,
        "val_acc": val.get("accuracy"),
        "val_logloss": val.get("logloss"),
        "brier": val.get("brier"),
        "ECE": val.get("ece"),
        "params": report.get("params") or report.get("model", {}).get("n_params"),
        "best_epoch": report.get("best_epoch"),
        "seed": config.get("seed"),
        "feature_groups": config.get("feature_groups"),
        "clean_missing_markets": bool(config.get("clean_missing_markets", False)),
        "run_path": config.get("out_dir") or report.get("run_path"),
        "status": report.get("status") or report.get("verdict") or "SAVED",
    }


def _model_from_run_name(name: str) -> str:
    if name.startswith("raw_mlp"):
        return "RawPooledMLP"
    if name.startswith("p3a"):
        return "P3a"
    if name.startswith("p3b"):
        return "P3b"
    return name


def summarize_seed_runs(root: str | Path) -> dict:
    root = Path(root)
    runs = []
    grouped = {}
    for report_path in sorted((root / "seed_runs").glob("*/report.json")):
        run_name = report_path.parent.name
        report = _read_json(report_path)
        row = _metrics_from_report(report)
        row["model"] = _model_from_run_name(run_name)
        row["run_name"] = run_name
        row["run_path"] = str(report_path.parent).replace("\\", "/")
        runs.append(row)
        grouped.setdefault(row["model"], []).append(row)

    models = {}
    for model, rows in grouped.items():
        losses = [row["val_logloss"] for row in rows if row["val_logloss"] is not None]
        briers = [row["brier"] for row in rows if row["brier"] is not None]
        eces = [row["ECE"] for row in rows if row["ECE"] is not None]
        accs = [row["val_acc"] for row in rows if row["val_acc"] is not None]
        models[model] = {
            "n": len(rows),
            "mean_val_logloss": statistics.mean(losses) if losses else None,
            "std_val_logloss": statistics.pstdev(losses) if len(losses) > 1 else 0.0,
            "best_val_logloss": min(losses) if losses else None,
            "worst_val_logloss": max(losses) if losses else None,
            "mean_brier": statistics.mean(briers) if briers else None,
            "mean_ECE": statistics.mean(eces) if eces else None,
            "mean_val_acc": statistics.mean(accs) if accs else None,
        }

    verdict = []
    raw = models.get("RawPooledMLP", {})
    p3b = models.get("P3b", {})
    if raw and p3b and raw.get("mean_val_logloss") is not None and p3b.get("mean_val_logloss") is not None:
        diff = p3b["mean_val_logloss"] - raw["mean_val_logloss"]
        noise = max(raw.get("std_val_logloss") or 0.0, p3b.get("std_val_logloss") or 0.0)
        if abs(diff) < noise:
            verdict.append("SEED_VARIANCE_HIGH")
        if diff > 0:
            verdict.append("TRANSFORMER_STILL_NOT_BEATING_MLP_CONFIRMED")
        if p3b.get("best_val_logloss") is not None and p3b["best_val_logloss"] <= raw["mean_val_logloss"]:
            verdict.append("P3B_UNSTABLE_POTENTIAL")
    return {"runs": runs, "models": models, "verdict": verdict}


def summarize_ablation_runs(root: str | Path) -> dict:
    root = Path(root)
    variants = []
    for report_path in sorted((root / "ablations").glob("*/report.json")):
        run_name = report_path.parent.name
        if run_name.startswith("p3b_clean"):
            continue
        report = _read_json(report_path)
        row = _metrics_from_report(report)
        row["variant"] = run_name
        row["run_path"] = str(report_path.parent).replace("\\", "/")
        variants.append(row)

    by_groups = {row.get("feature_groups"): row for row in variants}
    verdict = []
    euro = by_groups.get("euro")
    euro_asian = by_groups.get("euro,asian")
    euro_ou = by_groups.get("euro,ou")
    full = by_groups.get("euro,asian,ou")

    def loss(row):
        return row.get("val_logloss") if row else None

    if loss(euro_asian) is not None and loss(euro) is not None and loss(euro_asian) < loss(euro):
        verdict.append("ASIAN_SIGNAL_HELPFUL")
    if loss(euro_ou) is not None and loss(euro) is not None and loss(euro_ou) < loss(euro):
        verdict.append("OU_SIGNAL_HELPFUL")
    if loss(full) is not None and loss(euro_asian) is not None and loss(full) > loss(euro_asian):
        verdict.append("OU_MAY_ADD_NOISE")
    if loss(full) is not None and loss(euro) is not None and abs(loss(full) - loss(euro)) <= 0.002:
        verdict.append("CROSS_MARKET_NO_CLEAR_GAIN_CONFIRMED")

    best = min(variants, key=lambda row: row.get("val_logloss", float("inf")), default=None)
    return {"variants": variants, "best_variant": best, "verdict": verdict}


def summarize_clean_runs(root: str | Path) -> dict:
    root = Path(root)
    runs = []
    verdict = []
    for report_path in sorted((root / "ablations").glob("p3b_clean_seed*/report.json")):
        run_name = report_path.parent.name
        report = _read_json(report_path)
        row = _metrics_from_report(report)
        row["run_name"] = run_name
        row["run_path"] = str(report_path.parent).replace("\\", "/")
        runs.append(row)

    matched = []
    for clean_row in runs:
        seed = clean_row.get("seed")
        base_report = _read_json(root / "seed_runs" / f"p3b_seed{seed}" / "report.json")
        base_row = _metrics_from_report(base_report) if base_report else {}
        if clean_row.get("val_logloss") is not None and base_row.get("val_logloss") is not None:
            matched.append({"seed": seed, "clean": clean_row["val_logloss"], "base": base_row["val_logloss"]})

    if matched:
        mean_clean = statistics.mean(row["clean"] for row in matched)
        mean_base = statistics.mean(row["base"] for row in matched)
        if mean_clean < mean_base:
            verdict.extend(["MISSING_MARKET_FILLING_HURTS", "P3B_CLEAN_IMPROVES"])
        else:
            verdict.append("P3B_CLEAN_NO_GAIN")
    return {"runs": runs, "matched_seed_comparison": matched, "verdict": verdict}


def write_seed_summary(root: str | Path, payload: dict) -> None:
    root = Path(root)
    _write_json(root / "seed_summary.json", payload)
    lines = [
        "# P3.1 Seed Summary",
        "",
        "| Model | mean_val_logloss | std_val_logloss | best | worst | mean_Brier | mean_ECE | mean_val_acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, row in payload.get("models", {}).items():
        lines.append(
            f"| {model} | {_fmt(row.get('mean_val_logloss'))} | {_fmt(row.get('std_val_logloss'))} | "
            f"{_fmt(row.get('best_val_logloss'))} | {_fmt(row.get('worst_val_logloss'))} | "
            f"{_fmt(row.get('mean_brier'))} | {_fmt(row.get('mean_ECE'))} | {_fmt(row.get('mean_val_acc'))} |"
        )
    (root / "seed_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ablation_summary(root: str | Path, payload: dict) -> None:
    root = Path(root)
    _write_json(root / "ablation_summary.json", payload)
    lines = [
        "# P3.1 P3b Ablation Summary",
        "",
        "| Variant | feature_groups | val_acc | val_logloss | brier | ECE | train_acc | best_epoch | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("variants", []):
        lines.append(
            f"| {row.get('variant')} | {row.get('feature_groups')} | {_fmt(row.get('val_acc'))} | "
            f"{_fmt(row.get('val_logloss'))} | {_fmt(row.get('brier'))} | {_fmt(row.get('ECE'))} | "
            f"{_fmt(row.get('train_acc'))} | {_fmt(row.get('best_epoch'))} | {row.get('status')} |"
        )
    (root / "ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_clean_summary(root: str | Path, payload: dict) -> None:
    root = Path(root)
    _write_json(root / "p3b_clean_summary.json", payload)
    lines = [
        "# P3.1 P3b-clean Summary",
        "",
        "| Model | clean_missing_markets | val_logloss | brier | ECE | status |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in payload.get("runs", []):
        lines.append(
            f"| {row.get('run_name')} | {row.get('clean_missing_markets')} | {_fmt(row.get('val_logloss'))} | "
            f"{_fmt(row.get('brier'))} | {_fmt(row.get('ECE'))} | {row.get('status')} |"
        )
    (root / "p3b_clean_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _next_actions(verdict: list[str]) -> list[str]:
    actions = []
    if "DRAW_ARGMAX_COLLAPSE" in verdict:
        actions.append("Run draw calibration experiments: class weight, focal loss, and label smoothing as separate 3-seed comparisons.")
    if "ASIAN_SIGNAL_HELPFUL" in verdict and "OU_MAY_ADD_NOISE" in verdict:
        actions.append("Keep Euro+Asian as the next P3b candidate and demote OU to mask-only or low-weight ablation.")
    if "MISSING_MARKET_FILLING_HURTS" in verdict:
        actions.append("Promote --clean-missing-markets into the default P3b diagnostic path after a 3-seed confirmation.")
    if "SEED_VARIANCE_HIGH" in verdict:
        actions.append("Require 3-seed reporting for every follow-up P3 experiment before comparing against RawPooledMLP.")
    if "NEED_FEATURE_SCALING_FIRST" in verdict:
        actions.append("Run P3b feature-scaling diagnostics for Asian/OU line, water, implied, and margin features before changing the Transformer.")
    if "P3B_CLEAN_NO_GAIN" in verdict:
        actions.append("Do not promote --clean-missing-markets yet; keep it as an explicit diagnostic flag until a mean 3-seed gain appears.")
    if not actions:
        actions.append("Prioritize feature scaling and draw calibration before adding PLE, SAM, or masked pretraining.")
    actions.append("Do not move to PLE, SAM, or masked pretraining until draw and feature-scaling diagnostics are resolved.")
    return actions[:5]


def build_final_report_payload(
    prediction_json: str | Path,
    seed_json: str | Path,
    ablation_json: str | Path,
    clean_json: str | Path,
    diagnostic_root: str = "runs/p3_diagnostics",
    data_file: str = "data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl",
    train_ids: str = "data/odds_real/splits_v6/train_match_ids.txt",
    val_ids: str = "data/odds_real/splits_v6/val_match_ids.txt",
    test_ids: str = "data/odds_real/splits_v6/test_match_ids.txt",
    baseline_run_paths: list[str] | None = None,
) -> dict:
    prediction = _read_json(prediction_json)
    seed = _read_json(seed_json)
    ablation = _read_json(ablation_json)
    clean = _read_json(clean_json)
    verdict = []
    for section in [prediction, seed, ablation, clean]:
        for item in section.get("verdict", []):
            if item not in verdict:
                verdict.append(item)
    if "DRAW_ARGMAX_COLLAPSE" in verdict:
        verdict.append("NEED_DRAW_CALIBRATION_FIRST")
    if (
        "CROSS_MARKET_NO_CLEAR_GAIN_CONFIRMED" in verdict
        or "OU_MAY_ADD_NOISE" in verdict
        or "P3B_CLEAN_NO_GAIN" in verdict
    ):
        verdict.append("NEED_FEATURE_SCALING_FIRST")
    return {
        "environment": {
            "python_path": sys.executable,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "inputs": {
            "data_file": data_file,
            "train_ids": train_ids,
            "val_ids": val_ids,
            "test_ids": test_ids,
            "test_ids_used": False,
            "baseline_run_paths": baseline_run_paths or [
                "runs/p3_training_verification/raw_pooled_mlp_baseline",
                "runs/p3_training_verification/p3a_full",
                "runs/p3_training_verification/p3b_full",
            ],
            "diagnostic_root": diagnostic_root,
        },
        "prediction_diagnostics": prediction,
        "seed_summary": seed,
        "ablation_summary": ablation,
        "p3b_clean": clean,
        "verdict": verdict,
        "next_actions": _next_actions(verdict),
    }


def write_final_report(root: str | Path, payload: dict) -> None:
    root = Path(root)
    _write_json(root / "p3_diagnostics_report.json", payload)
    pred_models = payload.get("prediction_diagnostics", {}).get("models", {})
    seed_models = payload.get("seed_summary", {}).get("models", {})
    ablation = payload.get("ablation_summary", {})
    clean = payload.get("p3b_clean", {})
    inputs = payload.get("inputs", {})
    verdict = payload.get("verdict", [])
    next_actions = payload.get("next_actions", [])

    lines = [
        "# P3.1 Diagnostics Report",
        "",
        "## 1. Executive Summary",
        f"- draw collapse: {'DRAW_ARGMAX_COLLAPSE' in verdict}",
        f"- seed variance significant: {'SEED_VARIANCE_HIGH' in verdict}",
        f"- P3b stable vs RawMlp: {'TRANSFORMER_STILL_NOT_BEATING_MLP_CONFIRMED' in verdict}",
        f"- Asian signal helpful: {'ASIAN_SIGNAL_HELPFUL' in verdict}",
        f"- OU signal helpful: {'OU_SIGNAL_HELPFUL' in verdict}",
        f"- missing market filling hurts: {'MISSING_MARKET_FILLING_HURTS' in verdict}",
        f"- P3b-clean improves: {'P3B_CLEAN_IMPROVES' in verdict}",
        "- continue Transformer: yes, but resolve draw calibration / feature scaling first",
        "- PLE/SAM/pretraining: not next; keep as later options only",
        "- largest issue: draw calibration and cross-market feature utility need isolation",
        "",
        "## 2. Inputs",
        f"- data file: `{inputs.get('data_file')}`",
        f"- train ids: `{inputs.get('train_ids')}`",
        f"- val ids: `{inputs.get('val_ids')}`",
        f"- test ids: `{inputs.get('test_ids')}` (not used)",
        f"- diagnostic root: `{inputs.get('diagnostic_root')}`",
        "",
        "## 3. Prediction Diagnostics",
        "| Model | argmax_home_count | argmax_draw_count | argmax_away_count | draw_recall | mean_p_draw | mean_p_draw_on_true_draw | per_class_logloss_home | per_class_logloss_draw | per_class_logloss_away | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, row in pred_models.items():
        dist = row.get("predicted_class_distribution", {})
        losses = row.get("per_class_logloss", {})
        lines.append(
            f"| {model} | {dist.get('home')} | {dist.get('draw')} | {dist.get('away')} | "
            f"{_fmt(row.get('draw_recall'))} | {_fmt(row.get('mean_p_draw'))} | "
            f"{_fmt(row.get('mean_p_draw_on_true_draw'))} | {_fmt(losses.get('home'))} | "
            f"{_fmt(losses.get('draw'))} | {_fmt(losses.get('away'))} | {_fmt(row.get('ece'))} |"
        )

    lines.extend([
        "",
        "## 4. Seed Summary",
        "| Model | mean_val_logloss | std_val_logloss | best | worst | mean_Brier | mean_ECE | mean_val_acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for model, row in seed_models.items():
        lines.append(
            f"| {model} | {_fmt(row.get('mean_val_logloss'))} | {_fmt(row.get('std_val_logloss'))} | "
            f"{_fmt(row.get('best_val_logloss'))} | {_fmt(row.get('worst_val_logloss'))} | "
            f"{_fmt(row.get('mean_brier'))} | {_fmt(row.get('mean_ECE'))} | {_fmt(row.get('mean_val_acc'))} |"
        )

    lines.extend([
        "",
        "## 5. P3b Ablation Summary",
        "| Variant | feature_groups | val_acc | val_logloss | brier | ECE | train_acc | best_epoch | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in ablation.get("variants", []):
        lines.append(
            f"| {row.get('variant')} | {row.get('feature_groups')} | {_fmt(row.get('val_acc'))} | "
            f"{_fmt(row.get('val_logloss'))} | {_fmt(row.get('brier'))} | {_fmt(row.get('ECE'))} | "
            f"{_fmt(row.get('train_acc'))} | {_fmt(row.get('best_epoch'))} | {row.get('status')} |"
        )

    lines.extend([
        "",
        "## 6. P3b-clean Missing Market Experiment",
        "| Model | clean_missing_markets | val_logloss | brier | ECE | status |",
        "|---|---|---:|---:|---:|---|",
    ])
    for row in clean.get("runs", []):
        lines.append(
            f"| {row.get('run_name')} | {row.get('clean_missing_markets')} | {_fmt(row.get('val_logloss'))} | "
            f"{_fmt(row.get('brier'))} | {_fmt(row.get('ECE'))} | {row.get('status')} |"
        )

    lines.extend([
        "",
        "## 7. Interpretation",
        "1. P3b not beating baseline is attributed to the combination of draw calibration behavior and unproven cross-market feature utility unless seed summary proves otherwise.",
        "2. Draw calibration is a first-class issue when DRAW_ARGMAX_COLLAPSE or DRAW_PROB_UNDERCONFIDENT appears.",
        "3. Asian / OU usefulness is determined by seed42 ablations only; no architecture conclusion should be drawn from a single variant.",
        "4. Seed noise is material only when the RawMlp-P3b mean gap is smaller than observed run-to-run standard deviation.",
        "5. Missing market filling is a cause only if P3b-clean beats the matching P3b seed42 run.",
        "6. Continue Transformer diagnostics, but do not move to PLE/SAM/pretraining before calibration and feature scaling checks.",
        "",
        "## 8. Verdict",
    ])
    for item in verdict:
        lines.append(f"- `{item}`")
    lines.extend(["", "## 9. Next Actions"])
    for idx, action in enumerate(next_actions, 1):
        lines.append(f"{idx}. {action}")
    (root / "p3_diagnostics_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate P3.1 diagnostics summaries")
    parser.add_argument("--root", default="runs/p3_diagnostics")
    parser.add_argument("--data", default="data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl")
    parser.add_argument("--train-ids", default="data/odds_real/splits_v6/train_match_ids.txt")
    parser.add_argument("--val-ids", default="data/odds_real/splits_v6/val_match_ids.txt")
    parser.add_argument("--test-ids", default="data/odds_real/splits_v6/test_match_ids.txt")
    parser.add_argument("--baseline-run-path", action="append", dest="baseline_run_paths")
    args = parser.parse_args()
    root = Path(args.root)

    seed = summarize_seed_runs(root)
    ablation = summarize_ablation_runs(root)
    clean = summarize_clean_runs(root)
    write_seed_summary(root, seed)
    write_ablation_summary(root, ablation)
    write_clean_summary(root, clean)
    payload = build_final_report_payload(
        root / "prediction_diagnostics.json",
        root / "seed_summary.json",
        root / "ablation_summary.json",
        root / "p3b_clean_summary.json",
        diagnostic_root=args.root,
        data_file=args.data,
        train_ids=args.train_ids,
        val_ids=args.val_ids,
        test_ids=args.test_ids,
        baseline_run_paths=args.baseline_run_paths,
    )
    write_final_report(root, payload)
    print(f"Report markdown path: {root / 'p3_diagnostics_report.md'}")
    print(f"Report json path: {root / 'p3_diagnostics_report.json'}")
    print(f"Verdict: {', '.join(payload.get('verdict', []))}")


if __name__ == "__main__":
    main()
