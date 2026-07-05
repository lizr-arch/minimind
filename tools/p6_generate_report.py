"""Generate the P6 residual Patch-iTransformer summary report."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


P4_MEAN_VAL_LOGLOSS = 0.9390417536
P4_MEAN_ECE = 0.0309487
P6_BASELINE_MEAN_VAL_LOGLOSS = 0.9373360872
P6_BASELINE_MEAN_ECE = 0.027492


def aggregate_p6_reports(
    root: str | Path,
    p4_mean_val_logloss: float = P4_MEAN_VAL_LOGLOSS,
    p4_mean_ece: float = P4_MEAN_ECE,
    inputs: dict[str, Any] | None = None,
    variant: str | None = None,
    expected_seeds: list[int] | None = None,
    exclude_smoke: bool = True,
) -> dict[str, Any]:
    root = Path(root)
    groups: dict[str, list[dict[str, Any]]] = {}
    ignored_reports: list[dict[str, Any]] = []
    raw_report_count = 0
    for report_path in sorted(root.glob("*/report.json")):
        raw_report_count += 1
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report_variant = _variant_name(report_path.parent.name, report)
        report["_report_path"] = str(report_path)
        report["_run_path"] = str(report_path.parent)
        ignore_reason = _ignore_reason(
            report_path=report_path,
            report=report,
            report_variant=report_variant,
            requested_variant=variant,
            exclude_smoke=exclude_smoke,
        )
        if ignore_reason:
            ignored_reports.append(
                {"path": str(report_path), "run_path": str(report_path.parent), "variant": report_variant, "reason": ignore_reason}
            )
            continue
        groups.setdefault(report_variant, []).append(report)

    variants = [
        _aggregate_variant(name, reports, expected_seeds=expected_seeds)
        for name, reports in sorted(groups.items())
    ]
    complete = [variant for variant in variants if variant.get("seed_count", 0) >= 3]
    candidates = complete or [variant for variant in variants if _number(variant.get("mean_val_logloss")) is not None]
    best_variant = min(candidates, key=lambda item: float(item["mean_val_logloss"])) if candidates else {}

    test_ids_used = any(variant.get("test_ids_used") is not False for variant in variants)
    validation_fit_used = any(variant.get("validation_fit_used") is not False for variant in variants)
    posthoc_val_fit_used = any(variant.get("posthoc_val_fit_used") is not False for variant in variants)
    duplicate_formal_seed_count = sum(len(variant.get("duplicate_formal_seeds", [])) for variant in variants)
    missing_expected_seed_count = sum(len(variant.get("missing_expected_seeds", [])) for variant in variants)
    verdict = p6_verdict(
        mean_val_logloss=best_variant.get("mean_val_logloss"),
        std_val_logloss=best_variant.get("std_val_logloss"),
        mean_ece=best_variant.get("mean_ece"),
        p4_mean_val_logloss=p4_mean_val_logloss,
        p4_mean_ece=p4_mean_ece,
        test_ids_used=test_ids_used,
        validation_fit_used=validation_fit_used,
        posthoc_val_fit_used=posthoc_val_fit_used,
        seed_count=best_variant.get("seed_count", 0),
        mean_draw_recall=best_variant.get("mean_draw_recall"),
        mean_argmax_draw_count=best_variant.get("mean_argmax_draw_count"),
        duplicate_formal_seed_count=duplicate_formal_seed_count,
        missing_expected_seed_count=missing_expected_seed_count,
    )

    return {
        "phase": "P6 residual Patch-iTransformer objective probe",
        "inputs": {
            "root": str(root),
            "test_ids_used": test_ids_used,
            "variant": variant,
            "expected_seeds": expected_seeds,
            "exclude_smoke": exclude_smoke,
            **(inputs or {}),
        },
        "test_ids_used": test_ids_used,
        "validation_fit_used": validation_fit_used,
        "posthoc_val_fit_used": posthoc_val_fit_used,
        "raw_report_count": raw_report_count,
        "ignored_report_count": len(ignored_reports),
        "formal_report_count": sum(variant.get("formal_report_count", 0) for variant in variants),
        "formal_seed_count": best_variant.get("seed_count", 0),
        "ignored_reports": ignored_reports,
        "baselines": {
            "p4_mean_val_logloss": p4_mean_val_logloss,
            "p4_mean_ece": p4_mean_ece,
            "p6_baseline_mean_val_logloss": P6_BASELINE_MEAN_VAL_LOGLOSS,
            "p6_baseline_mean_ece": P6_BASELINE_MEAN_ECE,
        },
        "variants": variants,
        "best_variant": best_variant,
        "comparison": {
            "best_variant_by": "mean_val_logloss",
            "best_delta_vs_p4_mean": (
                best_variant.get("mean_val_logloss") - p4_mean_val_logloss
                if _number(best_variant.get("mean_val_logloss")) is not None
                else None
            ),
            "best_delta_vs_p6_baseline_mean": (
                best_variant.get("mean_val_logloss") - P6_BASELINE_MEAN_VAL_LOGLOSS
                if _number(best_variant.get("mean_val_logloss")) is not None
                else None
            ),
        },
        "verdict": verdict,
    }


def p6_verdict(
    mean_val_logloss: float | None,
    std_val_logloss: float | None,
    mean_ece: float | None,
    p4_mean_val_logloss: float,
    p4_mean_ece: float | None = P4_MEAN_ECE,
    test_ids_used: bool = False,
    validation_fit_used: bool = False,
    posthoc_val_fit_used: bool = False,
    seed_count: int = 3,
    mean_draw_recall: float | None = None,
    mean_argmax_draw_count: float | None = None,
    duplicate_formal_seed_count: int = 0,
    missing_expected_seed_count: int = 0,
) -> list[str]:
    verdict: list[str] = []
    if test_ids_used:
        verdict.append("P6_FAIL_TEST_READ")
    if validation_fit_used:
        verdict.append("P6_FAIL_VAL_FIT")
    if posthoc_val_fit_used:
        verdict.append("P6_FAIL_POSTHOC_VAL_FIT")
    if duplicate_formal_seed_count:
        verdict.append("P6_FAIL_DUPLICATE_FORMAL_SEED")
    if missing_expected_seed_count:
        verdict.append("P6_MISSING_EXPECTED_SEEDS")
    if seed_count < 3:
        verdict.append("P6_NEEDS_THREE_SEEDS")
    if verdict:
        if (
            "P6_FAIL_TEST_READ" in verdict
            or "P6_FAIL_VAL_FIT" in verdict
            or "P6_FAIL_POSTHOC_VAL_FIT" in verdict
            or "P6_FAIL_DUPLICATE_FORMAL_SEED" in verdict
        ):
            verdict.append("P6_FAIL_REGRESSION")
        return verdict
    if mean_val_logloss is None or std_val_logloss is None:
        verdict.append("P6_FAIL_REGRESSION")
        return verdict

    ece_ok_for_strong = mean_ece is None or p4_mean_ece is None or mean_ece <= p4_mean_ece + 0.003
    if mean_val_logloss <= p4_mean_val_logloss:
        verdict.append("P6_BEATS_P4")
        verdict.append("P6_SEQUENCE_OBJECTIVE_VALIDATED")
    if mean_val_logloss <= 0.93870 and std_val_logloss <= 0.00030 and ece_ok_for_strong:
        verdict.append("P6_LOGLOSS_STRONG_PASS")
        verdict.append("P6_STRONG_PASS")
    if mean_val_logloss <= 0.9392418 and std_val_logloss <= 0.00035:
        verdict.append("P6_MINIMUM_PASS")
    if mean_val_logloss > 0.9395418 or std_val_logloss > 0.00050:
        verdict.append("P6_FAIL_REGRESSION")
    if mean_val_logloss > p4_mean_val_logloss and "P6_FAIL_REGRESSION" not in verdict:
        verdict.append("P6_BEHIND_P4_MEAN")
    if mean_argmax_draw_count == 0 or (mean_draw_recall is not None and mean_draw_recall <= 0.005):
        verdict.append("DRAW_STILL_COLLAPSED")
    if not verdict:
        verdict.append("P6_NEEDS_REVIEW")
    else:
        verdict.extend(["P6_NO_TEST_USED", "P6_NO_VAL_FIT", "P6_AUDIT_FLAGS_PASS"])
    return verdict


def load_p4_baselines(path: str | Path | None) -> tuple[float, float | None]:
    if not path:
        return P4_MEAN_VAL_LOGLOSS, P4_MEAN_ECE
    report_path = Path(path)
    if not report_path.exists():
        return P4_MEAN_VAL_LOGLOSS, P4_MEAN_ECE
    report = json.loads(report_path.read_text(encoding="utf-8"))
    best = report.get("best_model", {}) or {}
    mean_ll = _number(best.get("mean_val_logloss"))
    mean_ece = _number(best.get("mean_ECE"))
    if mean_ll is None:
        mean_ll = _number(report.get("baselines", {}).get("p4_mean_val_logloss")) or P4_MEAN_VAL_LOGLOSS
    return mean_ll, mean_ece if mean_ece is not None else P4_MEAN_ECE


def write_markdown_report(payload: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# P6 Residual Patch-iTransformer",
        "",
        "## Audit",
        "",
        f"- test_ids_used: `{payload.get('test_ids_used')}`",
        f"- validation_fit_used: `{payload.get('validation_fit_used')}`",
        f"- posthoc_val_fit_used: `{payload.get('posthoc_val_fit_used')}`",
        f"- raw_report_count: `{payload.get('raw_report_count')}`",
        f"- ignored_report_count: `{payload.get('ignored_report_count')}`",
        f"- formal_report_count: `{payload.get('formal_report_count')}`",
        f"- formal_seed_count: `{payload.get('formal_seed_count')}`",
        "",
        "## Comparison",
        "",
        "| Variant | formal reports | seeds | aux lambda | final lambda | coupling | scale | mean val_logloss | std | mean ECE | draw NLL | draw recall | draw precision | draw top2 | draw ECE |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in payload.get("variants", []):
        lines.append(
            f"| {variant.get('name')} | {variant.get('formal_report_count')} | {variant.get('seed_count')} | "
            f"{_fmt(variant.get('lambda_draw_bce'))} | {_fmt(variant.get('lambda_final_draw_bce'))} | "
            f"{variant.get('enable_draw_logit_coupling')} | {_fmt(variant.get('draw_coupling_scale'))} | "
            f"{_fmt(variant.get('mean_val_logloss'))} | {_fmt(variant.get('std_val_logloss'))} | "
            f"{_fmt(variant.get('mean_ece'))} | {_fmt(variant.get('mean_draw_class_nll'))} | "
            f"{_fmt(variant.get('mean_draw_recall'))} | {_fmt(variant.get('mean_draw_precision'))} | "
            f"{_fmt(variant.get('mean_draw_top2_recall'))} | {_fmt(variant.get('mean_classwise_ece_draw'))} |"
        )
    if payload.get("ignored_reports"):
        lines.extend(["", "## Ignored Reports", "", "| Report | Reason |", "|---|---|"])
        for item in payload.get("ignored_reports", []):
            lines.append(f"| {item.get('path')} | {item.get('reason')} |")
    lines.extend(["", "## Verdict", ""])
    for item in payload.get("verdict", []):
        lines.append(f"- {item}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_variant(
    name: str,
    reports: list[dict[str, Any]],
    expected_seeds: list[int] | None = None,
) -> dict[str, Any]:
    by_seed: dict[Any, dict[str, Any]] = {}
    duplicate_formal_seeds: list[Any] = []
    missing_seed_count = 0
    for idx, report in enumerate(reports):
        seed = report.get("config", {}).get("seed")
        if seed is None:
            seed = f"missing_seed_{idx}"
            missing_seed_count += 1
        if seed in by_seed:
            duplicate_formal_seeds.append(seed)
            continue
        by_seed[seed] = report
    unique_reports = list(by_seed.values())
    vals = [_number(report.get("best_val_logloss")) for report in unique_reports]
    vals = [value for value in vals if value is not None]
    val_metrics = [report.get("val_metrics", {}) for report in unique_reports]
    data_sections = [report.get("data", {}) for report in unique_reports]
    first = reports[0]
    test_ids_used = any(data.get("test_ids_used") is not False for data in data_sections)
    validation_fit_used = any(data.get("validation_fit_used") is not False for data in data_sections)
    posthoc_val_fit_used = any(data.get("posthoc_val_fit_used") is not False for data in data_sections)
    expected_set = set(expected_seeds or [])
    observed_int_seeds = {seed for seed in by_seed if isinstance(seed, int)}
    missing_expected_seeds = sorted(expected_set - observed_int_seeds)
    return {
        "name": name,
        "status": "invalid" if duplicate_formal_seeds else ("complete" if len(vals) >= 3 else "incomplete"),
        "seed_count": len(vals),
        "report_count": len(reports),
        "formal_report_count": len(reports),
        "duplicate_seeds": duplicate_formal_seeds,
        "duplicate_formal_seeds": duplicate_formal_seeds,
        "missing_seed_count": missing_seed_count,
        "missing_expected_seeds": missing_expected_seeds,
        "seeds": list(by_seed.keys()),
        "run_paths": [report.get("run_path") or report.get("_run_path") for report in unique_reports],
        "feature_groups": first.get("config", {}).get("feature_groups"),
        "lambda_draw_bce": _number(first.get("config", {}).get("lambda_draw_bce")) or 0.0,
        "lambda_final_draw_bce": _number(first.get("config", {}).get("lambda_final_draw_bce")) or 0.0,
        "enable_draw_logit_coupling": bool(first.get("config", {}).get("enable_draw_logit_coupling", False)),
        "draw_coupling_scale": _number(first.get("config", {}).get("draw_coupling_scale")) or 0.0,
        "mean_val_logloss": _mean(vals),
        "std_val_logloss": _std(vals),
        "best_val_logloss": min(vals) if vals else None,
        "mean_ece": _mean_metric(val_metrics, "ece"),
        "mean_draw_class_nll": _mean_metric(val_metrics, "draw_class_nll"),
        "mean_draw_recall": _mean_metric(val_metrics, "draw_recall"),
        "mean_draw_precision": _mean_metric(val_metrics, "draw_precision"),
        "mean_draw_top2_recall": _mean_metric(val_metrics, "draw_top2_recall"),
        "mean_argmax_draw_count": _mean_metric(val_metrics, "argmax_draw_count"),
        "mean_p_draw_on_true_draw": _mean_metric(val_metrics, "mean_p_draw_on_true_draw"),
        "mean_p_draw_on_non_draw": _mean_metric(val_metrics, "mean_p_draw_on_non_draw"),
        "mean_classwise_ece_draw": _mean_metric(val_metrics, "classwise_ece_draw"),
        "test_ids_used": test_ids_used,
        "validation_fit_used": validation_fit_used,
        "posthoc_val_fit_used": posthoc_val_fit_used,
        "reports": [report.get("_report_path") for report in unique_reports],
    }


def _ignore_reason(
    report_path: Path,
    report: dict[str, Any],
    report_variant: str,
    requested_variant: str | None,
    exclude_smoke: bool,
) -> str | None:
    if requested_variant and report_variant != requested_variant:
        return "variant_mismatch"
    run_name = report_path.parent.name.lower()
    run_mode = str(report.get("run_mode") or report.get("config", {}).get("run_mode") or "").lower()
    if exclude_smoke and (run_mode == "smoke" or "smoke" in run_name):
        return "smoke_run"
    return None


def _variant_name(run_name: str, report: dict[str, Any]) -> str:
    config = report.get("config", {})
    feature_groups = str(config.get("feature_groups", "euro")).replace(",", "_")
    diff = float(config.get("diff_loss_weight", 0.30))
    consistency = float(config.get("consistency_loss_weight", 0.10))
    delta = float(config.get("delta_l2_weight", 0.01))
    if "no_goal_diff" in run_name or (diff == 0 and consistency == 0):
        suffix = "no_goal_diff"
    elif "no_delta_l2" in run_name or delta == 0:
        suffix = "no_delta_l2"
    elif "no_consistency" in run_name or consistency == 0:
        suffix = "no_consistency"
    else:
        suffix = "default"
    lambda_draw = _number(config.get("lambda_draw_bce")) or 0.0
    if lambda_draw > 0:
        suffix = f"{suffix}_draw_aux_{_lambda_suffix(lambda_draw)}"
    lambda_final_draw = _number(config.get("lambda_final_draw_bce")) or 0.0
    coupling_enabled = bool(config.get("enable_draw_logit_coupling", False))
    coupling_scale = _number(config.get("draw_coupling_scale")) or 0.0
    if coupling_enabled:
        if abs(lambda_final_draw - 0.005) < 1e-9 and abs(coupling_scale - 0.25) < 1e-9:
            suffix = f"{suffix}_draw_logit_coupled_weak"
        else:
            suffix = f"{suffix}_draw_logit_coupled_{_lambda_suffix(coupling_scale)}"
    elif lambda_final_draw > 0:
        suffix = f"{suffix}_final_draw_bce_{_lambda_suffix_3(lambda_final_draw)}"
    return f"{feature_groups}_{suffix}"


def _lambda_suffix(value: float) -> str:
    known = {
        0.005: "0005",
        0.01: "001",
        0.02: "002",
    }
    for key, suffix in known.items():
        if abs(float(value) - key) < 1e-9:
            return suffix
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text.replace("0.", "").replace(".", "")


def _lambda_suffix_3(value: float) -> str:
    known = {
        0.005: "005",
        0.01: "010",
    }
    for key, suffix in known.items():
        if abs(float(value) - key) < 1e-9:
            return suffix
    return _lambda_suffix(value)


def _mean_metric(items: list[dict[str, Any]], key: str) -> float | None:
    values = [_number(item.get(key)) for item in items]
    return _mean([value for value in values if value is not None])


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


def _fmt(value: Any) -> str:
    number = _number(value)
    return "N/A" if number is None else f"{number:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate P6 residual Patch-iTransformer report")
    parser.add_argument("--root", "--runs-root", dest="root", default="runs/p6_residual_patch_itransformer")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--expected-seeds", default=None)
    parser.add_argument("--exclude-smoke", dest="exclude_smoke", action="store_true", default=True)
    parser.add_argument("--include-smoke", dest="exclude_smoke", action="store_false")
    parser.add_argument("--p4-report", default="runs/p4_residual_goal_diff/p4_report.json")
    parser.add_argument("--out-json", default="runs/p6_residual_patch_itransformer/p6_report.json")
    parser.add_argument("--out-md", default="runs/p6_residual_patch_itransformer/p6_report.md")
    args = parser.parse_args()
    expected_seeds = [int(seed.strip()) for seed in args.expected_seeds.split(",") if seed.strip()] if args.expected_seeds else None

    p4_mean, p4_ece = load_p4_baselines(args.p4_report)
    payload = aggregate_p6_reports(
        args.root,
        p4_mean_val_logloss=p4_mean,
        p4_mean_ece=p4_ece,
        inputs={"p4_report": args.p4_report, "out_json": args.out_json, "out_md": args.out_md},
        variant=args.variant,
        expected_seeds=expected_seeds,
        exclude_smoke=args.exclude_smoke,
    )
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    write_markdown_report(payload, out_md)
    print(f"P6 report json: {out_json}")
    print(f"P6 report markdown: {out_md}")
    print(f"Best P6 variant: {payload.get('best_variant', {}).get('name', 'N/A')}")
    print(f"Verdict: {', '.join(payload.get('verdict', []))}")


if __name__ == "__main__":
    main()
