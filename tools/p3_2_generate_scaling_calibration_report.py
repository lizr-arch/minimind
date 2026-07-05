"""Generate P3.2 scaling and calibration summaries from run artifacts."""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p3_prediction_diagnostics import compute_prediction_diagnostics, read_prediction_csv


RAW_MLP_BASELINE_LOGLOSS = 0.9414
IMPROVEMENT_EPS = 0.0001
CALIBRATION_TOLERANCE = 0.0005


def _read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _write_text(path: str | Path, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _model_key(report: dict) -> str:
    model_name = str(report.get("model", "")).lower()
    if "rawpooledmlp" in model_name:
        return "raw_mlp"
    if "itransformer" in model_name or report.get("phase", "").startswith("P3b"):
        return "p3b"
    return model_name or "unknown"


def _metric_row(report_path: Path) -> dict:
    report = _read_json(report_path)
    val = report.get("val_metrics") or {}
    train = report.get("train_metrics") or {}
    config = report.get("config") or {}
    loss_config = report.get("loss_config") or {}
    row = {
        "run_name": report_path.parent.name,
        "run_path": str(report_path.parent).replace("\\", "/"),
        "model": report.get("model") or report.get("phase"),
        "model_key": _model_key(report),
        "seed": config.get("seed"),
        "scaling": config.get("scaling", "none"),
        "feature_groups": config.get("feature_groups"),
        "loss": loss_config.get("loss", config.get("loss", "ce")),
        "class_weights": loss_config.get("class_weights", config.get("class_weights")),
        "draw_weight": loss_config.get("draw_weight", config.get("draw_weight")),
        "label_smoothing": loss_config.get("label_smoothing", config.get("label_smoothing")),
        "focal_gamma": loss_config.get("focal_gamma", config.get("focal_gamma")),
        "best_epoch": report.get("best_epoch"),
        "val_logloss": val.get("logloss", report.get("best_val_logloss")),
        "brier": val.get("brier"),
        "ECE": val.get("ece"),
        "val_acc": val.get("accuracy"),
        "train_acc": train.get("accuracy", train.get("final_train_acc")),
        "status": report.get("status", "SAVED"),
        "val_predictions": report.get("val_predictions"),
    }
    return row


def _summarize_group(rows: list[dict], extra: dict | None = None) -> dict:
    losses = [row["val_logloss"] for row in rows if row.get("val_logloss") is not None]
    briers = [row["brier"] for row in rows if row.get("brier") is not None]
    eces = [row["ECE"] for row in rows if row.get("ECE") is not None]
    val_accs = [row["val_acc"] for row in rows if row.get("val_acc") is not None]
    train_accs = [row["train_acc"] for row in rows if row.get("train_acc") is not None]
    epochs = [row["best_epoch"] for row in rows if row.get("best_epoch") is not None]
    summary = {
        "runs": rows,
        "n_runs": len(rows),
        "mean_val_logloss": _mean(losses),
        "std_val_logloss": _std(losses),
        "best_val_logloss": min(losses) if losses else None,
        "worst_val_logloss": max(losses) if losses else None,
        "mean_brier": _mean(briers),
        "mean_ECE": _mean(eces),
        "mean_val_acc": _mean(val_accs),
        "mean_train_acc": _mean(train_accs),
        "best_epoch_mean": _mean(epochs),
    }
    if extra:
        summary.update(extra)
    return summary


def _variant_from_feature_groups(feature_groups: str | None) -> str:
    value = (feature_groups or "").replace(" ", "")
    return {
        "euro": "euro",
        "euro,asian": "euro_asian",
        "euro,ou": "euro_ou",
        "euro,asian,ou": "euro_asian_ou",
    }.get(value, value.replace(",", "_") or "unknown")


def build_scaling_summary(root: str | Path, raw_mlp_baseline_logloss: float = RAW_MLP_BASELINE_LOGLOSS) -> dict:
    root = Path(root)
    rows = [_metric_row(path) for path in sorted((root / "scaling_runs").glob("p3b_full_*_seed*/report.json"))]
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["scaling"], []).append(row)

    scalings = []
    for scaling, group_rows in sorted(grouped.items()):
        scalings.append(_summarize_group(group_rows, {"scaling": scaling, "verdict": []}))

    by_scaling = {row["scaling"]: row for row in scalings}
    none_mean = by_scaling.get("none", {}).get("mean_val_logloss")
    none_ece = by_scaling.get("none", {}).get("mean_ECE")
    verdict = set()
    best_scaling = None
    best_mean = None

    for row in scalings:
        mean_loss = row.get("mean_val_logloss")
        if mean_loss is None:
            continue
        if best_mean is None or mean_loss < best_mean:
            best_mean = mean_loss
            best_scaling = row["scaling"]
        if row["scaling"] == "none" or none_mean is None:
            continue
        improves_loss = mean_loss < none_mean - IMPROVEMENT_EPS
        improves_ece = (
            row.get("mean_ECE") is not None
            and none_ece is not None
            and row["mean_ECE"] < none_ece - IMPROVEMENT_EPS
        )
        if improves_loss and mean_loss < raw_mlp_baseline_logloss:
            row["verdict"].append("P3B_SCALING_BEATS_RAWMLP")
            verdict.add("P3B_SCALING_BEATS_RAWMLP")
        elif improves_loss:
            row["verdict"].append("FEATURE_SCALING_HELPS_BUT_NOT_ENOUGH")
            verdict.add("FEATURE_SCALING_HELPS_BUT_NOT_ENOUGH")
        elif improves_ece:
            row["verdict"].append("SCALING_IMPROVES_CALIBRATION_ONLY")
            verdict.add("SCALING_IMPROVES_CALIBRATION_ONLY")
        else:
            row["verdict"].append("FEATURE_SCALING_NO_GAIN")

    if not verdict and scalings:
        verdict.add("FEATURE_SCALING_NO_GAIN")

    best_for_ablation = best_scaling
    if none_mean is not None and (best_mean is None or best_mean >= none_mean - IMPROVEMENT_EPS):
        best_for_ablation = "robust"

    return {
        "raw_mlp_baseline_logloss": raw_mlp_baseline_logloss,
        "scalings": scalings,
        "runs": rows,
        "best_scaling": best_scaling,
        "best_scaling_for_ablation": best_for_ablation,
        "verdict": sorted(verdict),
    }


def build_scaled_ablation_summary(root: str | Path) -> dict:
    root = Path(root)
    rows = [
        _metric_row(path)
        for path in sorted((root / "scaling_runs").glob("p3b_*_seed*/report.json"))
        if not path.parent.name.startswith("p3b_full_")
    ]
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        variant = _variant_from_feature_groups(row.get("feature_groups"))
        row["variant"] = variant
        grouped.setdefault((row.get("scaling", "none"), variant), []).append(row)

    variants = []
    for (scaling, variant), group_rows in sorted(grouped.items()):
        variants.append(_summarize_group(group_rows, {"scaling": scaling, "variant": variant, "verdict": []}))

    by_variant = {row["variant"]: row for row in variants}
    euro = by_variant.get("euro")
    verdict = set()
    if euro and euro.get("mean_val_logloss") is not None:
        euro_loss = euro["mean_val_logloss"]
        for variant, verdict_name in [
            ("euro_asian", "ASIAN_SIGNAL_HELPFUL_AFTER_SCALING"),
            ("euro_ou", "OU_SIGNAL_HELPFUL_AFTER_SCALING"),
            ("euro_asian_ou", "CROSS_MARKET_SIGNAL_HELPFUL_AFTER_SCALING"),
        ]:
            row = by_variant.get(variant)
            if row and row.get("mean_val_logloss") is not None and row["mean_val_logloss"] < euro_loss - IMPROVEMENT_EPS:
                row["verdict"].append(verdict_name)
                verdict.add(verdict_name)
        full = by_variant.get("euro_asian_ou")
        if full and full.get("mean_val_logloss") is not None and full["mean_val_logloss"] >= euro_loss - IMPROVEMENT_EPS:
            full["verdict"].append("CROSS_MARKET_NO_GAIN_AFTER_SCALING")
            verdict.add("CROSS_MARKET_NO_GAIN_AFTER_SCALING")

    best_variant = None
    complete = [row for row in variants if row.get("mean_val_logloss") is not None]
    if complete:
        best_variant = min(complete, key=lambda row: row["mean_val_logloss"])["variant"]

    return {
        "variants": variants,
        "runs": rows,
        "best_variant": best_variant,
        "verdict": sorted(verdict),
    }


def _calibration_key(row: dict) -> str:
    loss = row.get("loss") or "ce"
    if loss == "weighted_ce":
        return f"weighted_draw{float(row.get('draw_weight') or 0):g}"
    if loss == "label_smoothing":
        return f"label_smoothing{float(row.get('label_smoothing') or 0):g}"
    return loss


def _attach_prediction_diagnostics(row: dict) -> None:
    pred_path = row.get("val_predictions")
    if not pred_path:
        pred_path = str(Path(row["run_path"]) / "val_predictions.csv")
    if not os.path.exists(pred_path):
        row["prediction_diagnostics_status"] = "missing"
        return
    diag = compute_prediction_diagnostics(row["run_name"], read_prediction_csv(pred_path))
    losses = diag.get("per_class_logloss", {})
    row.update(
        {
            "prediction_diagnostics_status": "ok",
            "argmax_draw_count": diag.get("draw_argmax_count"),
            "draw_recall": diag.get("draw_recall"),
            "mean_p_draw": diag.get("mean_p_draw"),
            "mean_p_draw_on_true_draw": diag.get("mean_p_draw_on_true_draw"),
            "per_class_logloss_home": losses.get("home"),
            "per_class_logloss_draw": losses.get("draw"),
            "per_class_logloss_away": losses.get("away"),
        }
    )


def build_calibration_summary(root: str | Path) -> dict:
    root = Path(root)
    rows = [_metric_row(path) for path in sorted((root / "calibration_runs").glob("*/report.json"))]
    for row in rows:
        _attach_prediction_diagnostics(row)
        row["calibration_key"] = _calibration_key(row)

    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        key = (row["model_key"], row.get("scaling", "none"), row["calibration_key"])
        grouped.setdefault(key, []).append(row)

    variants = []
    for (model_key, scaling, calibration_key), group_rows in sorted(grouped.items()):
        summary = _summarize_group(
            group_rows,
            {
                "model_key": model_key,
                "scaling": scaling,
                "calibration_key": calibration_key,
                "loss": group_rows[0].get("loss"),
                "draw_weight": group_rows[0].get("draw_weight"),
                "label_smoothing": group_rows[0].get("label_smoothing"),
                "verdict": [],
            },
        )
        for metric in [
            "argmax_draw_count",
            "draw_recall",
            "mean_p_draw",
            "mean_p_draw_on_true_draw",
            "per_class_logloss_home",
            "per_class_logloss_draw",
            "per_class_logloss_away",
        ]:
            values = [row[metric] for row in group_rows if row.get(metric) is not None]
            summary[f"mean_{metric}"] = _mean(values)
        variants.append(summary)

    baseline_by_model_scaling = {
        (row["model_key"], row["scaling"]): row
        for row in variants
        if row.get("calibration_key") == "ce"
    }
    verdict = set()
    for row in variants:
        if row.get("calibration_key") == "ce":
            continue
        baseline = baseline_by_model_scaling.get((row["model_key"], row["scaling"]))
        if not baseline:
            continue
        loss_delta = None
        if row.get("mean_val_logloss") is not None and baseline.get("mean_val_logloss") is not None:
            loss_delta = row["mean_val_logloss"] - baseline["mean_val_logloss"]
            row["mean_val_logloss_delta_vs_ce"] = loss_delta
        if row.get("mean_brier") is not None and baseline.get("mean_brier") is not None:
            row["mean_brier_delta_vs_ce"] = row["mean_brier"] - baseline["mean_brier"]
        if row.get("mean_ECE") is not None and baseline.get("mean_ECE") is not None:
            row["mean_ECE_delta_vs_ce"] = row["mean_ECE"] - baseline["mean_ECE"]
        if row.get("mean_draw_recall") is not None and baseline.get("mean_draw_recall") is not None:
            row["mean_draw_recall_delta_vs_ce"] = row["mean_draw_recall"] - baseline["mean_draw_recall"]

        recall_gain = row.get("mean_draw_recall_delta_vs_ce", 0.0) > 0
        if row.get("loss") == "weighted_ce":
            if loss_delta is not None and loss_delta < -IMPROVEMENT_EPS:
                row["verdict"].append("DRAW_WEIGHT_HELPFUL")
                verdict.add("DRAW_WEIGHT_HELPFUL")
            elif recall_gain and loss_delta is not None and loss_delta > CALIBRATION_TOLERANCE:
                row["verdict"].append("DRAW_RECALL_GAIN_BUT_LOGLOSS_HURT")
                verdict.add("DRAW_RECALL_GAIN_BUT_LOGLOSS_HURT")
        if row.get("loss") == "label_smoothing":
            ece_delta = row.get("mean_ECE_delta_vs_ce")
            if ece_delta is not None and ece_delta < -IMPROVEMENT_EPS and (loss_delta is None or loss_delta <= CALIBRATION_TOLERANCE):
                row["verdict"].append("LABEL_SMOOTHING_CALIBRATION_ONLY")
                verdict.add("LABEL_SMOOTHING_CALIBRATION_ONLY")

    if not verdict and variants:
        verdict.add("DRAW_CALIBRATION_NO_GAIN")

    best_variant = None
    complete = [row for row in variants if row.get("mean_val_logloss") is not None]
    if complete:
        best = min(complete, key=lambda row: row["mean_val_logloss"])
        best_variant = {
            "model_key": best["model_key"],
            "scaling": best["scaling"],
            "calibration_key": best["calibration_key"],
            "mean_val_logloss": best["mean_val_logloss"],
        }

    return {
        "variants": variants,
        "runs": rows,
        "best_variant": best_variant,
        "verdict": sorted(verdict),
    }


def _best_p3b_logloss(scaling_summary: dict, scaled_ablation_summary: dict, calibration_summary: dict) -> float | None:
    losses = []
    for row in scaling_summary.get("scalings", []):
        if row.get("mean_val_logloss") is not None:
            losses.append(row["mean_val_logloss"])
    for row in scaled_ablation_summary.get("variants", []):
        if row.get("mean_val_logloss") is not None:
            losses.append(row["mean_val_logloss"])
    for row in calibration_summary.get("variants", []):
        if row.get("model_key") == "p3b" and row.get("mean_val_logloss") is not None:
            losses.append(row["mean_val_logloss"])
    return min(losses) if losses else None


def build_final_report_payload(
    scaling_summary: dict,
    scaled_ablation_summary: dict,
    calibration_summary: dict,
    raw_mlp_baseline_logloss: float = RAW_MLP_BASELINE_LOGLOSS,
) -> dict:
    verdict = set()
    for section in [scaling_summary, scaled_ablation_summary, calibration_summary]:
        verdict.update(section.get("verdict", []))

    best_p3b = _best_p3b_logloss(scaling_summary, scaled_ablation_summary, calibration_summary)
    if best_p3b is not None and best_p3b < raw_mlp_baseline_logloss:
        verdict.add("P3B_NOW_BEATS_RAWMLP")
    elif best_p3b is not None:
        verdict.add("RAW_MLP_STILL_BEST")

    if "CROSS_MARKET_NO_GAIN_AFTER_SCALING" in verdict:
        verdict.add("NEED_BETTER_MARKET_FEATURE_ENGINEERING")

    best_model = {
        "raw_mlp_baseline_logloss": raw_mlp_baseline_logloss,
        "best_p3b_mean_val_logloss": best_p3b,
        "best_scaling": scaling_summary.get("best_scaling"),
        "best_scaling_for_ablation": scaling_summary.get("best_scaling_for_ablation"),
        "best_scaled_ablation_variant": scaled_ablation_summary.get("best_variant"),
        "best_calibration_variant": calibration_summary.get("best_variant"),
    }
    next_actions = _next_actions(verdict, best_model)
    return {
        "scaling_summary": scaling_summary,
        "scaled_ablation_summary": scaled_ablation_summary,
        "calibration_summary": calibration_summary,
        "best_model": best_model,
        "verdict": sorted(verdict),
        "next_actions": next_actions,
    }


def _next_actions(verdict: set[str], best_model: dict) -> list[str]:
    actions = []
    if "P3B_SCALING_BEATS_RAWMLP" in verdict:
        actions.append(f"Consider promoting --scaling {best_model.get('best_scaling')} for P3b after code review.")
    elif "FEATURE_SCALING_HELPS_BUT_NOT_ENOUGH" in verdict:
        actions.append("Keep scaling as a diagnostic improvement, but do not replace RawPooledMLP on logloss yet.")
    elif "FEATURE_SCALING_NO_GAIN" in verdict:
        actions.append("Do not merge scaling as default; investigate feature engineering before deeper Transformer work.")
    if "DRAW_RECALL_GAIN_BUT_LOGLOSS_HURT" in verdict:
        actions.append("Do not merge draw weighting; keep it only as a diagnostic because logloss is harmed.")
    elif "DRAW_WEIGHT_HELPFUL" in verdict:
        actions.append("Review the best draw-weighted run for acceptance because it improves primary logloss.")
    elif "LABEL_SMOOTHING_CALIBRATION_ONLY" in verdict:
        actions.append("Treat label smoothing as calibration-only unless primary logloss also improves.")
    if "NEED_BETTER_MARKET_FEATURE_ENGINEERING" in verdict:
        actions.append("Redesign Asian/OU features before adding PLE, SAM, or masked pretraining.")
    if "RAW_MLP_STILL_BEST" in verdict:
        actions.append("Keep RawPooledMLP as the production baseline and leave Transformer as a research branch.")
    return actions[:5] or ["No merge action; preserve artifacts and review logs for failed or incomplete phases."]


def write_scaling_summary(root: str | Path, payload: dict) -> None:
    root = Path(root)
    _write_json(root / "scaling_summary.json", payload)
    lines = [
        "# P3.2 Scaling Summary",
        "",
        "| Scaling | mean_val_logloss | std | best | worst | mean_Brier | mean_ECE | mean_val_acc | mean_train_acc | best_epoch_mean | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("scalings", []):
        lines.append(
            f"| {row.get('scaling')} | {_fmt(row.get('mean_val_logloss'))} | {_fmt(row.get('std_val_logloss'))} | "
            f"{_fmt(row.get('best_val_logloss'))} | {_fmt(row.get('worst_val_logloss'))} | "
            f"{_fmt(row.get('mean_brier'))} | {_fmt(row.get('mean_ECE'))} | {_fmt(row.get('mean_val_acc'))} | "
            f"{_fmt(row.get('mean_train_acc'))} | {_fmt(row.get('best_epoch_mean'))} | {', '.join(row.get('verdict', []))} |"
        )
    lines.extend(["", "## Verdict"])
    lines.extend(f"- `{item}`" for item in payload.get("verdict", []))
    _write_text(root / "scaling_summary.md", "\n".join(lines) + "\n")


def write_scaled_ablation_summary(root: str | Path, payload: dict) -> None:
    root = Path(root)
    _write_json(root / "scaled_ablation_summary.json", payload)
    lines = [
        "# P3.2 Scaled Ablation Summary",
        "",
        "| Variant | feature_groups | scaling | mean_val_logloss | std | mean_Brier | mean_ECE | mean_val_acc | verdict |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("variants", []):
        feature_groups = row["runs"][0].get("feature_groups") if row.get("runs") else ""
        lines.append(
            f"| {row.get('variant')} | {feature_groups} | {row.get('scaling')} | {_fmt(row.get('mean_val_logloss'))} | "
            f"{_fmt(row.get('std_val_logloss'))} | {_fmt(row.get('mean_brier'))} | {_fmt(row.get('mean_ECE'))} | "
            f"{_fmt(row.get('mean_val_acc'))} | {', '.join(row.get('verdict', []))} |"
        )
    lines.extend(["", "## Verdict"])
    lines.extend(f"- `{item}`" for item in payload.get("verdict", []))
    _write_text(root / "scaled_ablation_summary.md", "\n".join(lines) + "\n")


def write_calibration_summary(root: str | Path, payload: dict) -> None:
    root = Path(root)
    _write_json(root / "calibration_summary.json", payload)
    lines = [
        "# P3.2 Draw Calibration Summary",
        "",
        "| Model | loss | draw_weight | label_smoothing | scaling | mean_val_logloss | mean_Brier | mean_ECE | argmax_draw_count | draw_recall | verdict |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("variants", []):
        lines.append(
            f"| {row.get('model_key')} | {row.get('loss')} | {_fmt(row.get('draw_weight'))} | {_fmt(row.get('label_smoothing'))} | "
            f"{row.get('scaling')} | {_fmt(row.get('mean_val_logloss'))} | {_fmt(row.get('mean_brier'))} | "
            f"{_fmt(row.get('mean_ECE'))} | {_fmt(row.get('mean_argmax_draw_count'))} | "
            f"{_fmt(row.get('mean_draw_recall'))} | {', '.join(row.get('verdict', []))} |"
        )
    lines.extend(["", "## Verdict"])
    lines.extend(f"- `{item}`" for item in payload.get("verdict", []))
    _write_text(root / "calibration_summary.md", "\n".join(lines) + "\n")


def write_final_report(root: str | Path, payload: dict) -> None:
    root = Path(root)
    _write_json(root / "p3_2_report.json", payload)
    best = payload.get("best_model", {})
    lines = [
        "# P3.2 Scaling & Calibration Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- best scaling: `{best.get('best_scaling')}`",
        f"- scaling ablation uses: `{best.get('best_scaling_for_ablation')}`",
        f"- best P3b mean val_logloss: {_fmt(best.get('best_p3b_mean_val_logloss'))}",
        f"- RawPooledMLP baseline val_logloss: {_fmt(best.get('raw_mlp_baseline_logloss'))}",
        f"- best scaled ablation variant: `{best.get('best_scaled_ablation_variant')}`",
        f"- best calibration variant: `{best.get('best_calibration_variant')}`",
        f"- verdict: {', '.join(payload.get('verdict', []))}",
        "",
        "## 2. Inputs",
        "",
        "- data file: `data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl`",
        "- train ids: `data/odds_real/splits_v6/train_match_ids.txt`",
        "- val ids: `data/odds_real/splits_v6/val_match_ids.txt`",
        "- test ids: not used",
        f"- diagnostic root: `{root.as_posix()}`",
        "",
        "## 3. Scaling Summary",
        "",
        "| Scaling | mean_val_logloss | std | best | mean_Brier | mean_ECE | mean_val_acc | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("scaling_summary", {}).get("scalings", []):
        lines.append(
            f"| {row.get('scaling')} | {_fmt(row.get('mean_val_logloss'))} | {_fmt(row.get('std_val_logloss'))} | "
            f"{_fmt(row.get('best_val_logloss'))} | {_fmt(row.get('mean_brier'))} | {_fmt(row.get('mean_ECE'))} | "
            f"{_fmt(row.get('mean_val_acc'))} | {', '.join(row.get('verdict', []))} |"
        )
    lines.extend([
        "",
        "## 4. Scaled Ablation Summary",
        "",
        "| Variant | feature_groups | scaling | mean_val_logloss | std | mean_Brier | mean_ECE | verdict |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ])
    for row in payload.get("scaled_ablation_summary", {}).get("variants", []):
        feature_groups = row["runs"][0].get("feature_groups") if row.get("runs") else ""
        lines.append(
            f"| {row.get('variant')} | {feature_groups} | {row.get('scaling')} | {_fmt(row.get('mean_val_logloss'))} | "
            f"{_fmt(row.get('std_val_logloss'))} | {_fmt(row.get('mean_brier'))} | {_fmt(row.get('mean_ECE'))} | "
            f"{', '.join(row.get('verdict', []))} |"
        )
    lines.extend([
        "",
        "## 5. Draw Calibration Summary",
        "",
        "| Model | loss | draw_weight | label_smoothing | scaling | mean_val_logloss | mean_Brier | mean_ECE | argmax_draw_count | draw_recall | verdict |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in payload.get("calibration_summary", {}).get("variants", []):
        lines.append(
            f"| {row.get('model_key')} | {row.get('loss')} | {_fmt(row.get('draw_weight'))} | {_fmt(row.get('label_smoothing'))} | "
            f"{row.get('scaling')} | {_fmt(row.get('mean_val_logloss'))} | {_fmt(row.get('mean_brier'))} | "
            f"{_fmt(row.get('mean_ECE'))} | {_fmt(row.get('mean_argmax_draw_count'))} | "
            f"{_fmt(row.get('mean_draw_recall'))} | {', '.join(row.get('verdict', []))} |"
        )
    lines.extend([
        "",
        "## 6. Interpretation",
        "",
        f"1. Scaling answer: {', '.join(payload.get('scaling_summary', {}).get('verdict', [])) or 'incomplete'}",
        f"2. Asian/OU answer: {', '.join(payload.get('scaled_ablation_summary', {}).get('verdict', [])) or 'incomplete'}",
        f"3. Draw calibration answer: {', '.join(payload.get('calibration_summary', {}).get('verdict', [])) or 'incomplete'}",
        "4. Merge scaling only if primary val_logloss improves against the appropriate baseline.",
        "5. Merge draw loss only if val_logloss improves or calibration improves within the allowed tolerance.",
        "6. PLE/SAM/masked pretraining should wait until this report supports moving beyond preprocessing/calibration.",
        "",
        "## 7. Verdict",
        "",
    ])
    lines.extend(f"- `{item}`" for item in payload.get("verdict", []))
    lines.extend(["", "## 8. Next Actions", ""])
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(payload.get("next_actions", []), start=1))
    _write_text(root / "p3_2_report.md", "\n".join(lines) + "\n")


def generate_all(root: str | Path, raw_mlp_baseline_logloss: float = RAW_MLP_BASELINE_LOGLOSS) -> dict:
    scaling = build_scaling_summary(root, raw_mlp_baseline_logloss=raw_mlp_baseline_logloss)
    write_scaling_summary(root, scaling)
    ablation = build_scaled_ablation_summary(root)
    write_scaled_ablation_summary(root, ablation)
    calibration = build_calibration_summary(root)
    write_calibration_summary(root, calibration)
    final = build_final_report_payload(
        scaling_summary=scaling,
        scaled_ablation_summary=ablation,
        calibration_summary=calibration,
        raw_mlp_baseline_logloss=raw_mlp_baseline_logloss,
    )
    write_final_report(root, final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate P3.2 scaling and calibration reports")
    parser.add_argument("--root", required=True)
    parser.add_argument("--raw-mlp-baseline-logloss", type=float, default=RAW_MLP_BASELINE_LOGLOSS)
    args = parser.parse_args()

    final = generate_all(args.root, raw_mlp_baseline_logloss=args.raw_mlp_baseline_logloss)
    best = final.get("best_model", {})
    print(f"Report markdown path: {Path(args.root) / 'p3_2_report.md'}")
    print(f"Report json path: {Path(args.root) / 'p3_2_report.json'}")
    print(f"Best scaling: {best.get('best_scaling')}")
    print(f"Best P3b val_logloss: {_fmt(best.get('best_p3b_mean_val_logloss'))}")
    print(f"RawMLP baseline val_logloss: {_fmt(best.get('raw_mlp_baseline_logloss'))}")
    print(f"Best calibration variant: {best.get('best_calibration_variant')}")
    print(f"Verdict: {', '.join(final.get('verdict', []))}")


if __name__ == "__main__":
    main()
