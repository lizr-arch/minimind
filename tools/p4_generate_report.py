"""Generate the P4 residual goal-diff objective-probe summary report."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any


REQUIRED_VARIANTS = (
    "euro_default",
    "euro_no_consistency",
    "euro_asian_default",
    "euro_asian_no_consistency",
)


def build_final_report_payload(
    inputs: dict[str, Any],
    audit: dict[str, Any],
    anchor_only_baseline: dict[str, Any],
    variants: list[dict[str, Any]],
    raw_mlp_label_smoothing_logloss: float,
    p3_4_oof_fusion_logloss: float,
) -> dict[str, Any]:
    comparison = build_comparison(anchor_only_baseline, variants)
    verdict = build_verdict(
        anchor_only_baseline,
        variants,
        comparison,
        raw_mlp_label_smoothing_logloss,
    )
    return {
        "inputs": inputs,
        "audit": audit,
        "anchor_only_baseline": anchor_only_baseline,
        "baselines": {
            "raw_mlp_label_smoothing_logloss": raw_mlp_label_smoothing_logloss,
            "p3_4_oof_fusion_logloss": p3_4_oof_fusion_logloss,
        },
        "p4_1": {
            "variants": variants,
            "comparison": comparison,
        },
        "best_model": comparison.get("best_model", {}),
        "verdict": verdict,
        "next_actions": build_next_actions(verdict),
    }


def build_comparison(anchor_only_baseline: dict[str, Any], variants: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {variant.get("name"): variant for variant in variants}
    best_euro = _best_existing([by_name.get("euro_default"), by_name.get("euro_no_consistency")])
    best_euro_asian = _best_existing([by_name.get("euro_asian_default"), by_name.get("euro_asian_no_consistency")])
    best_p4 = _best_existing([best_euro, best_euro_asian])
    anchor_logloss = _number(anchor_only_baseline.get("anchor_only_val_logloss"))
    consistency = {
        "euro": _delta_pair(by_name.get("euro_default"), by_name.get("euro_no_consistency")),
        "euro_asian": _delta_pair(by_name.get("euro_asian_default"), by_name.get("euro_asian_no_consistency")),
    }
    comparison = {
        "best_euro_variant": best_euro.get("name") if best_euro else None,
        "best_euro_asian_variant": best_euro_asian.get("name") if best_euro_asian else None,
        "asian_delta_vs_best_euro": _delta(best_euro_asian, best_euro),
        "best_p4_delta_vs_anchor_only": (
            _number(best_p4.get("mean_val_logloss")) - anchor_logloss
            if best_p4 and anchor_logloss is not None
            else None
        ),
        "consistency_ablation": consistency,
        "best_model": best_p4 or {},
    }
    return comparison


def build_verdict(
    anchor_only_baseline: dict[str, Any],
    variants: list[dict[str, Any]],
    comparison: dict[str, Any],
    raw_mlp_label_smoothing_logloss: float,
) -> list[str]:
    verdict = ["P4_PLAN_APPROVED_WITH_FIXES"]
    by_name = {variant.get("name"): variant for variant in variants}
    best_euro = by_name.get(comparison.get("best_euro_variant"))
    best_euro_asian = by_name.get(comparison.get("best_euro_asian_variant"))
    best_p4 = comparison.get("best_model") or None
    anchor_logloss = _number(anchor_only_baseline.get("anchor_only_val_logloss"))

    euro_ll = _number(best_euro.get("mean_val_logloss")) if best_euro else None
    asian_ll = _number(best_euro_asian.get("mean_val_logloss")) if best_euro_asian else None
    best_ll = _number(best_p4.get("mean_val_logloss")) if best_p4 else None

    if euro_ll is not None:
        if euro_ll < raw_mlp_label_smoothing_logloss:
            verdict.append("P4_EURO_RESIDUAL_BEATS_RAWMLP")
        elif euro_ll <= raw_mlp_label_smoothing_logloss + 0.001:
            verdict.append("P4_EURO_RESIDUAL_CLOSE_TO_RAWMLP")
        else:
            verdict.append("P4_EURO_RESIDUAL_FAILS_BASELINE")

    if euro_ll is not None and asian_ll is not None:
        if asian_ll < euro_ll - 0.0003:
            verdict.append("P4_ASIAN_GOAL_DIFF_HELPS")
        elif asian_ll > euro_ll + 0.0003:
            verdict.append("P4_ASIAN_GOAL_DIFF_HURTS")
        else:
            verdict.append("P4_ASIAN_GOAL_DIFF_NO_GAIN")

    if best_ll is not None and anchor_logloss is not None:
        if best_ll < anchor_logloss - 0.0003:
            verdict.append("P4_RESIDUAL_IMPROVES_ANCHOR")
        elif best_ll > anchor_logloss + 0.0003:
            verdict.append("P4_RESIDUAL_HURTS_ANCHOR")
        else:
            verdict.append("P4_RESIDUAL_NO_ANCHOR_GAIN")

    default_best = _best_existing([by_name.get("euro_default"), by_name.get("euro_asian_default")])
    no_consistency_best = _best_existing([by_name.get("euro_no_consistency"), by_name.get("euro_asian_no_consistency")])
    if default_best and no_consistency_best:
        default_ll = _number(default_best.get("mean_val_logloss"))
        no_cons_ll = _number(no_consistency_best.get("mean_val_logloss"))
        if default_ll < no_cons_ll - 0.0003:
            verdict.append("P4_CONSISTENCY_HELPS")
        elif default_ll > no_cons_ll + 0.0003:
            verdict.append("P4_CONSISTENCY_HURTS")
        else:
            verdict.append("P4_CONSISTENCY_NO_GAIN")

    if "P4_ASIAN_GOAL_DIFF_HELPS" in verdict and "P4_RESIDUAL_HURTS_ANCHOR" not in verdict:
        verdict.append("P4_READY_FOR_AH_SETTLEMENT_LOSS")
    else:
        verdict.append("P4_STOP_BEFORE_AH_SETTLEMENT")

    if best_ll is not None and best_ll < raw_mlp_label_smoothing_logloss:
        verdict.append("P4_NEW_BEST")
    else:
        verdict.append("RAW_MLP_STILL_BEST")

    if best_p4 and anchor_only_baseline:
        best_draw = _number(best_p4.get("mean_draw_recall"))
        anchor_draw = _number(anchor_only_baseline.get("draw_recall"))
        best_argmax_draw = _number(best_p4.get("mean_argmax_draw_count"))
        if (
            best_draw is not None
            and anchor_draw is not None
            and best_ll is not None
            and anchor_logloss is not None
            and best_draw > anchor_draw + 0.02
            and best_ll <= anchor_logloss + 0.0003
        ):
            verdict.append("DRAW_RECALL_IMPROVED_WITHOUT_LOGLOSS_HURT")
        elif best_argmax_draw == 0 or (
            best_draw is not None and anchor_draw is not None and best_draw <= anchor_draw + 0.005
        ):
            verdict.append("DRAW_STILL_COLLAPSED")
    return verdict


def build_next_actions(verdict: list[str]) -> list[str]:
    if "P4_READY_FOR_AH_SETTLEMENT_LOSS" in verdict:
        return ["Run P4.2 AH settlement loss after reviewing P4.1 diagnostics."]
    if "P4_RESIDUAL_HURTS_ANCHOR" in verdict:
        return ["Diagnose delta regularization and anchor residual behavior before expanding model capacity."]
    if "P4_CONSISTENCY_HURTS" in verdict:
        return ["Use no-consistency as the next P4.1 default and keep goal-diff supervision auxiliary."]
    return ["Review P4.1 objective-probe diagnostics before adding Transformer capacity."]


def collect_variants(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    runs_root = root / "p4_1_runs"
    groups: dict[str, list[dict[str, Any]]] = {name: [] for name in REQUIRED_VARIANTS}
    anchor = {}
    audit = _load_audit(root)
    if not runs_root.exists():
        return [], anchor, audit
    for report_path in sorted(runs_root.glob("p4_*_seed*/report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        name = _variant_name_from_report(report_path.parent.name, report)
        if name not in groups:
            continue
        groups[name].append(report)
        if not anchor:
            anchor = report.get("anchor_only_baseline", {})

    variants = []
    for name in REQUIRED_VARIANTS:
        reports = groups[name]
        if reports:
            variants.append(_aggregate_variant(name, reports))
        else:
            variants.append({"name": name, "status": "missing", "mean_val_logloss": None, "seeds": []})
    return variants, anchor, audit


def _load_audit(root: Path) -> dict[str, Any]:
    for path in [
        root / "audit" / "goal_diff_audit.json",
        root / "audit" / "p4_audit_report.json",
    ]:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {
        "train": {"anchor_fallback_count": 0, "negative_time_valid_euro_count": 0},
        "val": {"anchor_fallback_count": 0, "negative_time_valid_euro_count": 0},
        "test_ids_used": False,
        "warning": "P4 audit file not found; counts are unavailable.",
    }


def _variant_name_from_report(run_name: str, report: dict[str, Any]) -> str:
    feature_groups = report.get("config", {}).get("feature_groups", "")
    consistency = float(report.get("config", {}).get("consistency_loss_weight", 0.0))
    prefix = "euro_asian" if feature_groups == "euro,asian" else "euro"
    suffix = "default" if consistency > 0 else "no_consistency"
    if "euro_asian" in run_name:
        prefix = "euro_asian"
    elif "p4_euro_" in run_name:
        prefix = "euro"
    if "no_consistency" in run_name:
        suffix = "no_consistency"
    elif "default" in run_name:
        suffix = "default"
    return f"{prefix}_{suffix}"


def _aggregate_variant(name: str, reports: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [_number(report.get("best_val_logloss")) for report in reports]
    vals = [value for value in vals if value is not None]
    first = reports[0]
    val_metrics = [report.get("val_metrics", {}) for report in reports]
    goal_metrics = [report.get("goal_diff_metrics", {}) for report in reports]
    return {
        "name": name,
        "status": "complete",
        "feature_groups": first.get("config", {}).get("feature_groups"),
        "loss_config": "no_consistency" if "no_consistency" in name else "default",
        "seeds": [report.get("config", {}).get("seed") for report in reports],
        "run_paths": [report.get("run_path") for report in reports],
        "mean_val_logloss": _mean(vals),
        "std_val_logloss": _std(vals),
        "best_val_logloss": min(vals) if vals else None,
        "mean_brier": _mean_metric(val_metrics, "brier"),
        "mean_ECE": _mean_metric(val_metrics, "ece"),
        "mean_val_acc": _mean_metric(val_metrics, "accuracy"),
        "mean_diff_nll": _mean_metric(goal_metrics, "diff_nll"),
        "mean_diff_acc": _mean_metric(goal_metrics, "diff_acc"),
        "mean_argmax_draw_count": _mean_metric(val_metrics, "argmax_draw_count"),
        "mean_draw_recall": _mean_metric(val_metrics, "draw_recall"),
        "mean_p_draw": _mean_metric(val_metrics, "mean_p_draw"),
        "mean_p_draw_on_true_draw": _mean_metric(val_metrics, "mean_p_draw_on_true_draw"),
        "mean_goal_diff_bucket_acc": _mean_metric(goal_metrics, "bucket_acc"),
        "mean_goal_diff_zero_recall": _mean_metric(goal_metrics, "zero_recall"),
        "mean_p_from_diff_logloss": _mean_metric(goal_metrics, "p_from_diff_logloss"),
        "mean_final_vs_diff_kl": _mean_metric(goal_metrics, "final_vs_diff_kl"),
        "mean_delta_l2": _mean_metric([_last_history(report) for report in reports], "train_L_delta"),
        "mean_delta_vs_anchor_only_logloss": _mean_delta_vs_anchor(reports),
    }


def _last_history(report: dict[str, Any]) -> dict[str, Any]:
    history = report.get("history", [])
    return history[-1] if history else {}


def _mean_metric(items: list[dict[str, Any]], key: str) -> float | None:
    values = [_number(item.get(key)) for item in items]
    return _mean([value for value in values if value is not None])


def _mean_delta_vs_anchor(reports: list[dict[str, Any]]) -> float | None:
    values = []
    for report in reports:
        model_ll = _number(report.get("best_val_logloss"))
        anchor_ll = _number(report.get("anchor_only_baseline", {}).get("anchor_only_val_logloss"))
        if model_ll is not None and anchor_ll is not None:
            values.append(model_ll - anchor_ll)
    return _mean(values)


def _delta_pair(default: dict[str, Any] | None, no_consistency: dict[str, Any] | None) -> dict[str, Any]:
    default_ll = _number(default.get("mean_val_logloss")) if default else None
    no_cons_ll = _number(no_consistency.get("mean_val_logloss")) if no_consistency else None
    return {
        "default": default_ll,
        "no_consistency": no_cons_ll,
        "delta_default_minus_no_consistency": (
            default_ll - no_cons_ll if default_ll is not None and no_cons_ll is not None else None
        ),
        "winner": (
            "default"
            if default_ll is not None and no_cons_ll is not None and default_ll < no_cons_ll
            else "no_consistency"
            if default_ll is not None and no_cons_ll is not None and no_cons_ll < default_ll
            else None
        ),
    }


def _best_existing(variants: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    available = [variant for variant in variants if variant and _number(variant.get("mean_val_logloss")) is not None]
    if not available:
        return None
    return min(available, key=lambda variant: float(variant["mean_val_logloss"]))


def _delta(left: dict[str, Any] | None, right: dict[str, Any] | None) -> float | None:
    if not left or not right:
        return None
    left_ll = _number(left.get("mean_val_logloss"))
    right_ll = _number(right.get("mean_val_logloss"))
    if left_ll is None or right_ll is None:
        return None
    return left_ll - right_ll


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return float(statistics.pstdev(values))


def _number(value: Any) -> float | None:
    try:
        if value is None:
            return None
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def write_markdown_report(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# P4 Residual Goal-Diff Objective Probe",
        "",
        "## Main Comparison",
        "",
        "| Model / config | mean val_logloss |",
        "|---|---:|",
        f"| Euro anchor only | {_fmt(payload.get('anchor_only_baseline', {}).get('anchor_only_val_logloss'))} |",
        f"| RawMLP label smoothing | {_fmt(payload.get('baselines', {}).get('raw_mlp_label_smoothing_logloss'))} |",
        f"| P3.4 OOF fusion | {_fmt(payload.get('baselines', {}).get('p3_4_oof_fusion_logloss'))} |",
    ]
    for variant in payload.get("p4_1", {}).get("variants", []):
        lines.append(f"| P4 {variant.get('name')} | {_fmt(variant.get('mean_val_logloss'))} |")
    lines.extend(["", "## Draw / Goal-Diff Diagnostics", ""])
    lines.append(
        "| Variant | draw_recall | mean_p_draw | goal_diff_acc | zero_recall | p_from_diff_logloss | final_vs_diff_kl |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for variant in payload.get("p4_1", {}).get("variants", []):
        lines.append(
            f"| {variant.get('name')} | {_fmt(variant.get('mean_draw_recall'))} | "
            f"{_fmt(variant.get('mean_p_draw'))} | {_fmt(variant.get('mean_goal_diff_bucket_acc'))} | "
            f"{_fmt(variant.get('mean_goal_diff_zero_recall'))} | "
            f"{_fmt(variant.get('mean_p_from_diff_logloss'))} | {_fmt(variant.get('mean_final_vs_diff_kl'))} |"
        )
    lines.extend(["", "## Verdict", ""])
    for item in payload.get("verdict", []):
        lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    number = _number(value)
    return "N/A" if number is None else f"{number:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate P4 residual goal-diff report")
    parser.add_argument("--root", default="runs/p4_residual_goal_diff")
    parser.add_argument("--raw-mlp-label-smoothing-logloss", type=float, default=0.940862)
    parser.add_argument("--p3-4-oof-fusion-logloss", type=float, default=0.941617)
    args = parser.parse_args()

    root = Path(args.root)
    variants, anchor, audit = collect_variants(root)
    payload = build_final_report_payload(
        inputs={"root": str(root), "test_ids_used": False},
        audit=audit,
        anchor_only_baseline=anchor,
        variants=variants,
        raw_mlp_label_smoothing_logloss=args.raw_mlp_label_smoothing_logloss,
        p3_4_oof_fusion_logloss=args.p3_4_oof_fusion_logloss,
    )
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "p4_report.json"
    md_path = root / "p4_report.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    write_markdown_report(payload, md_path)
    print(f"P4 report markdown: {md_path}")
    print(f"P4 report json: {json_path}")
    print(f"Best P4 variant: {payload.get('best_model', {}).get('name', 'N/A')}")
    print(f"Verdict: {', '.join(payload.get('verdict', []))}")


if __name__ == "__main__":
    main()
