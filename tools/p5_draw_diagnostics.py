"""P5 draw / goal-diff diagnostics for P4 validation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


PROB_FIELDS = ("p_home", "p_draw", "p_away")
DIFF_PROB_FIELDS = ("p_home_from_diff", "p_draw_from_diff", "p_away_from_diff")
BUCKET_FIELDS = tuple(f"p_bucket_{idx}" for idx in range(7))
DEFAULT_THRESHOLDS = (0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30)


def load_and_merge_prediction_rows(one_x_two_path: str | Path, goal_diff_path: str | Path) -> list[dict[str, Any]]:
    """Load P4 validation prediction CSVs and merge by match_id in 1X2 row order."""
    one_x_two_rows = _read_one_x_two_rows(one_x_two_path)
    goal_diff_by_id = _read_goal_diff_rows_by_id(goal_diff_path)
    occurrence_index: dict[str, int] = {}
    merged = []
    for row in one_x_two_rows:
        match_id = row["match_id"]
        if match_id not in goal_diff_by_id:
            raise ValueError(f"missing goal-diff prediction row for match_id={match_id}")
        idx = occurrence_index.get(match_id, 0)
        if idx >= len(goal_diff_by_id[match_id]):
            raise ValueError(f"missing goal-diff occurrence {idx + 1} for match_id={match_id}")
        merged.append({**row, **goal_diff_by_id[match_id][idx]})
        occurrence_index[match_id] = idx + 1
    for match_id, rows in goal_diff_by_id.items():
        consumed = occurrence_index.get(match_id, 0)
        if consumed != len(rows):
            raise ValueError(
                f"unconsumed goal-diff prediction rows for match_id={match_id}: "
                f"consumed={consumed}, available={len(rows)}"
            )
    return merged


def compute_draw_threshold_curve(rows: list[dict[str, Any]], thresholds=DEFAULT_THRESHOLDS) -> list[dict[str, Any]]:
    """Summarize draw recall/precision if p_draw is inspected at fixed thresholds."""
    true_draw_count = sum(1 for row in rows if _as_int(row["y_true"]) == 1)
    curve = []
    for threshold in thresholds:
        threshold = float(threshold)
        selected = [row for row in rows if _as_float(row["p_draw"]) >= threshold]
        true_selected = [row for row in selected if _as_int(row["y_true"]) == 1]
        decision_correct = 0
        for row in rows:
            pred = 1 if _as_float(row["p_draw"]) >= threshold else _final_argmax(row)
            decision_correct += int(pred == _as_int(row["y_true"]))
        draw_precision = _safe_div(len(true_selected), len(selected))
        curve.append(
            {
                "threshold": _round(threshold),
                "predicted_draw_count": len(selected),
                "draw_recall": _safe_div(len(true_selected), true_draw_count),
                "precision": draw_precision,
                "draw_precision": draw_precision,
                "decision_accuracy": _safe_div(decision_correct, len(rows)),
                "logloss_unchanged": True,
            }
        )
    return curve


def compute_p_draw_calibration_bins(rows: list[dict[str, Any]], n_bins: int = 10) -> list[dict[str, Any]]:
    """Bin final p_draw and report empirical draw rate per bin."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    bins = []
    for idx in range(n_bins):
        low = idx / n_bins
        high = (idx + 1) / n_bins
        selected = [
            row
            for row in rows
            if _as_float(row["p_draw"]) >= low
            and (_as_float(row["p_draw"]) < high or (idx == n_bins - 1 and _as_float(row["p_draw"]) <= high))
        ]
        if selected:
            draw_rate = _safe_div(sum(1 for row in selected if _as_int(row["y_true"]) == 1), len(selected))
            mean_p_draw = _round(sum(_as_float(row["p_draw"]) for row in selected) / len(selected))
        else:
            draw_rate = None
            mean_p_draw = None
        bins.append(
            {
                "bin": idx,
                "low": _round(low),
                "high": _round(high),
                "count": len(selected),
                "draw_rate": draw_rate,
                "mean_p_draw": mean_p_draw,
            }
        )
    return bins


def compute_final_vs_diff_draw_gap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare final p_draw against p_draw implied by the auxiliary goal-diff head."""
    gaps = [_as_float(row["p_draw"]) - _as_float(row["p_draw_from_diff"]) for row in rows]
    true_draw_gaps = [
        _as_float(row["p_draw"]) - _as_float(row["p_draw_from_diff"])
        for row in rows
        if _as_int(row["y_true"]) == 1
    ]
    return {
        "mean_gap": _mean(gaps),
        "mean_abs_gap": _mean([abs(gap) for gap in gaps]),
        "true_draw_gap": _mean(true_draw_gaps),
        "n_rows": len(rows),
        "n_true_draw": len(true_draw_gaps),
    }


def compute_draw_probability_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize final p_draw on all, true-draw, and non-draw rows."""
    true_draw = [row for row in rows if _as_int(row["y_true"]) == 1]
    non_draw = [row for row in rows if _as_int(row["y_true"]) != 1]
    return {
        "mean_p_draw": _mean([_as_float(row["p_draw"]) for row in rows]),
        "mean_p_draw_on_true_draw": _mean([_as_float(row["p_draw"]) for row in true_draw]),
        "mean_p_draw_on_non_draw": _mean([_as_float(row["p_draw"]) for row in non_draw]),
    }


def compute_argmax_draw_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report final and goal-diff-implied argmax draw counts and true-draw recall."""
    true_draw_count = sum(1 for row in rows if _as_int(row["y_true"]) == 1)
    final_draw_rows = [row for row in rows if _final_argmax(row) == 1]
    diff_draw_rows = [row for row in rows if _diff_argmax(row) == 1]
    return {
        "n_rows": len(rows),
        "true_draw_count": true_draw_count,
        "final_argmax_draw_count": len(final_draw_rows),
        "final_draw_recall": _safe_div(
            sum(1 for row in final_draw_rows if _as_int(row["y_true"]) == 1),
            true_draw_count,
        ),
        "diff_argmax_draw_count": len(diff_draw_rows),
        "diff_draw_recall": _safe_div(
            sum(1 for row in diff_draw_rows if _as_int(row["y_true"]) == 1),
            true_draw_count,
        ),
    }


def compute_run_diagnostics(
    run_dir: str | Path,
    thresholds=DEFAULT_THRESHOLDS,
    n_bins: int = 10,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    rows = load_and_merge_prediction_rows(
        run_dir / "val_predictions.csv",
        run_dir / "val_goal_diff_predictions.csv",
    )
    return {
        "run": run_dir.name,
        "variant": _variant_from_run_name(run_dir.name),
        "paths": {
            "val_predictions": str(run_dir / "val_predictions.csv"),
            "val_goal_diff_predictions": str(run_dir / "val_goal_diff_predictions.csv"),
        },
        "diagnostics": {
            "n_rows": len(rows),
            "argmax_draw_stats": compute_argmax_draw_stats(rows),
            "draw_threshold_curve": compute_draw_threshold_curve(rows, thresholds=thresholds),
            "p_draw_calibration_bins": compute_p_draw_calibration_bins(rows, n_bins=n_bins),
            "draw_probability_summary": compute_draw_probability_summary(rows),
            "final_vs_diff_draw_gap": compute_final_vs_diff_draw_gap(rows),
        },
    }


def collect_run_diagnostics(
    p4_root: str | Path,
    thresholds=DEFAULT_THRESHOLDS,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    runs_root = Path(p4_root) / "p4_1_runs"
    if not runs_root.exists():
        return []
    diagnostics = []
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        if (run_dir / "val_predictions.csv").exists() and (run_dir / "val_goal_diff_predictions.csv").exists():
            diagnostics.append(compute_run_diagnostics(run_dir, thresholds=thresholds, n_bins=n_bins))
    return diagnostics


def build_report_payload(
    p4_root: str | Path,
    run_diagnostics: list[dict[str, Any]],
    p4_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    p4_report = p4_report or {}
    models = {
        run["run"]: {
            "variant": run.get("variant"),
            "paths": run.get("paths", {}),
            "diagnostics": run.get("diagnostics", {}),
        }
        for run in run_diagnostics
    }
    verdict = _build_verdict(run_diagnostics, p4_report)
    return {
        "inputs": {
            "p4_root": str(p4_root),
            "p4_report": str(Path(p4_root) / "p4_report.json"),
            "run_count": len(run_diagnostics),
            "test_ids_used": False,
        },
        "models": models,
        "diagnostics": {
            "run_count": len(run_diagnostics),
            "variants": sorted({run.get("variant") for run in run_diagnostics if run.get("variant")}),
            "variant_summaries": aggregate_variant_diagnostics(run_diagnostics),
            "p4_verdict": list(p4_report.get("verdict", [])),
        },
        "verdict": verdict,
        "next_actions": _build_next_actions(verdict),
        "test_ids_used": False,
    }


def aggregate_variant_diagnostics(run_diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate P5 diagnostics across seeds by P4 variant."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in run_diagnostics:
        variant = run.get("variant")
        if variant:
            grouped.setdefault(variant, []).append(run)
    summaries = {}
    for variant, runs in sorted(grouped.items()):
        argmax_items = [run.get("diagnostics", {}).get("argmax_draw_stats", {}) for run in runs]
        prob_items = [run.get("diagnostics", {}).get("draw_probability_summary", {}) for run in runs]
        gap_items = [run.get("diagnostics", {}).get("final_vs_diff_draw_gap", {}) for run in runs]
        summaries[variant] = {
            "run_count": len(runs),
            "mean_final_argmax_draw_count": _mean_metric(argmax_items, "final_argmax_draw_count"),
            "mean_final_draw_recall": _mean_metric(argmax_items, "final_draw_recall"),
            "mean_diff_argmax_draw_count": _mean_metric(argmax_items, "diff_argmax_draw_count"),
            "mean_diff_draw_recall": _mean_metric(argmax_items, "diff_draw_recall"),
            "mean_p_draw": _mean_metric(prob_items, "mean_p_draw"),
            "mean_p_draw_on_true_draw": _mean_metric(prob_items, "mean_p_draw_on_true_draw"),
            "mean_p_draw_on_non_draw": _mean_metric(prob_items, "mean_p_draw_on_non_draw"),
            "mean_gap": _mean_metric(gap_items, "mean_gap"),
            "mean_abs_gap": _mean_metric(gap_items, "mean_abs_gap"),
            "mean_true_draw_gap": _mean_metric(gap_items, "true_draw_gap"),
        }
    return summaries


def write_markdown_report(payload: dict[str, Any], out_md: str | Path) -> None:
    lines = [
        "# P5 Draw / Goal-Diff Diagnostics",
        "",
        "Validation-only diagnostic report. No parameters were fitted and no test split was read.",
        "",
        "| Run | Variant | rows | final_argmax_draw | final_draw_recall | diff_argmax_draw | diff_draw_recall | mean_gap | mean_abs_gap | true_draw_gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run_name, model in payload.get("models", {}).items():
        diag = model.get("diagnostics", {})
        argmax = diag.get("argmax_draw_stats", {})
        gap = diag.get("final_vs_diff_draw_gap", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    run_name,
                    str(model.get("variant")),
                    str(argmax.get("n_rows", diag.get("n_rows", "N/A"))),
                    str(argmax.get("final_argmax_draw_count", "N/A")),
                    _fmt(argmax.get("final_draw_recall")),
                    str(argmax.get("diff_argmax_draw_count", "N/A")),
                    _fmt(argmax.get("diff_draw_recall")),
                    _fmt(gap.get("mean_gap")),
                    _fmt(gap.get("mean_abs_gap")),
                    _fmt(gap.get("true_draw_gap")),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Variant Summary", ""])
    lines.append(
        "| Variant | runs | final_argmax_draw | final_draw_recall | diff_argmax_draw | diff_draw_recall | mean_p_draw | mean_p_draw_true | mean_p_draw_non_draw | mean_abs_gap |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for variant, summary in payload.get("diagnostics", {}).get("variant_summaries", {}).items():
        lines.append(
            f"| {variant} | {summary.get('run_count')} | "
            f"{_fmt(summary.get('mean_final_argmax_draw_count'))} | "
            f"{_fmt(summary.get('mean_final_draw_recall'))} | "
            f"{_fmt(summary.get('mean_diff_argmax_draw_count'))} | "
            f"{_fmt(summary.get('mean_diff_draw_recall'))} | "
            f"{_fmt(summary.get('mean_p_draw'))} | "
            f"{_fmt(summary.get('mean_p_draw_on_true_draw'))} | "
            f"{_fmt(summary.get('mean_p_draw_on_non_draw'))} | "
            f"{_fmt(summary.get('mean_abs_gap'))} |"
        )

    lines.extend(["", "## Threshold Curves", ""])
    for run_name, model in payload.get("models", {}).items():
        lines.append(f"### {run_name}")
        lines.append("")
        lines.append("| threshold | predicted_draw_count | draw_recall | draw_precision | decision_accuracy | logloss_unchanged |")
        lines.append("|---:|---:|---:|---:|---:|---|")
        for item in model.get("diagnostics", {}).get("draw_threshold_curve", []):
            lines.append(
                f"| {_fmt(item.get('threshold'))} | {item.get('predicted_draw_count')} | "
                f"{_fmt(item.get('draw_recall'))} | {_fmt(item.get('draw_precision'))} | "
                f"{_fmt(item.get('decision_accuracy'))} | "
                f"{item.get('logloss_unchanged')} |"
            )
        lines.append("")

    lines.extend(["## Verdict", ""])
    for item in payload.get("verdict", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Next Actions", ""])
    for item in payload.get("next_actions", []):
        lines.append(f"- {item}")
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_one_x_two_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "match_id": row["match_id"],
                    "y_true": _as_int(row["y_true"]),
                    "p_home": _as_float(row["p_home"]),
                    "p_draw": _as_float(row["p_draw"]),
                    "p_away": _as_float(row["p_away"]),
                    "pred_class": _as_int(row["pred_class"]),
                    "correct": _as_int(row.get("correct", 0)),
                }
            )
    return rows


def _read_goal_diff_rows_by_id(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            match_id = row["match_id"]
            parsed = {"match_id": match_id, "y_goal_diff": _as_int(row["y_goal_diff"])}
            for field in BUCKET_FIELDS + DIFF_PROB_FIELDS:
                if field in row:
                    parsed[field] = _as_float(row[field])
            if "pred_bucket" in row:
                parsed["pred_bucket"] = _as_int(row["pred_bucket"])
            if "correct_bucket" in row:
                parsed["correct_bucket"] = _as_int(row["correct_bucket"])
            rows.setdefault(match_id, []).append(parsed)
    return rows


def _build_verdict(run_diagnostics: list[dict[str, Any]], p4_report: dict[str, Any]) -> list[str]:
    verdict = set(p4_report.get("verdict", []))
    if "DRAW_STILL_COLLAPSED" in verdict:
        verdict.add("P5_DIAGNOSE_DRAW_STILL_COLLAPSED")
    if run_diagnostics and all(
        run.get("diagnostics", {}).get("argmax_draw_stats", {}).get("final_argmax_draw_count", 0) <= 1
        for run in run_diagnostics
    ):
        verdict.add("P5_FINAL_ARGMAX_DRAW_COLLAPSE")
    if run_diagnostics and all(
        run.get("diagnostics", {}).get("argmax_draw_stats", {}).get("diff_argmax_draw_count", 0) == 0
        for run in run_diagnostics
    ):
        verdict.add("P5_DIFF_ARGMAX_DRAW_COLLAPSE")
    if any(
        _number(run.get("diagnostics", {}).get("final_vs_diff_draw_gap", {}).get("mean_abs_gap")) is not None
        and _number(run.get("diagnostics", {}).get("final_vs_diff_draw_gap", {}).get("mean_abs_gap")) > 0.05
        for run in run_diagnostics
    ):
        verdict.add("P5_FINAL_VS_DIFF_DRAW_GAP_VISIBLE")
    if not verdict:
        verdict.add("P5_DIAGNOSTICS_ONLY")
    return sorted(verdict)


def _build_next_actions(verdict: list[str]) -> list[str]:
    actions = [
        "Review validation-only draw threshold curves and calibration bins before changing P4 training.",
        "Compare final p_draw against p_draw_from_diff to localize whether collapse is in the final 1X2 head or the goal-diff mapping.",
    ]
    if "P5_FINAL_ARGMAX_DRAW_COLLAPSE" in verdict:
        actions.append("Inspect runs with zero final argmax draws for draw probability ranking compression.")
    return actions


def _variant_from_run_name(run_name: str) -> str:
    name = re.sub(r"^p4_", "", run_name)
    name = re.sub(r"_seed\d+$", "", name)
    return name


def _final_argmax(row: dict[str, Any]) -> int:
    if "pred_class" in row:
        return _as_int(row["pred_class"])
    return _argmax([_as_float(row[field]) for field in PROB_FIELDS])


def _diff_argmax(row: dict[str, Any]) -> int:
    if not all(field in row for field in DIFF_PROB_FIELDS):
        return 1 if _as_float(row["p_draw_from_diff"]) >= (1.0 / 3.0) else 0
    return _argmax([_as_float(row[field]) for field in DIFF_PROB_FIELDS])


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda idx: values[idx])


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return _round(sum(values) / len(values))


def _mean_metric(items: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for item in items:
        number = _number(item.get(key))
        if number is not None:
            values.append(number)
    return _mean(values)


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _number(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float:
    return float(value)


def _as_int(value: Any) -> int:
    return int(float(value))


def _fmt(value: Any) -> str:
    number = _number(value)
    return "N/A" if number is None else f"{number:.6f}"


def _load_p4_report(p4_root: Path) -> dict[str, Any]:
    report_path = p4_root / "p4_report.json"
    if not report_path.exists():
        return {}
    return json.loads(report_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="P5 validation-only draw / goal-diff diagnostics")
    parser.add_argument("--p4-root", default="runs/p4_residual_goal_diff")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--threshold", action="append", type=float, dest="thresholds")
    args = parser.parse_args()

    p4_root = Path(args.p4_root)
    thresholds = args.thresholds if args.thresholds else DEFAULT_THRESHOLDS
    run_diagnostics = collect_run_diagnostics(p4_root, thresholds=thresholds, n_bins=args.n_bins)
    payload = build_report_payload(
        p4_root=p4_root,
        run_diagnostics=run_diagnostics,
        p4_report=_load_p4_report(p4_root),
    )

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown_report(payload, out_md)
    print(f"P5 draw diagnostics json: {out_json}")
    print(f"P5 draw diagnostics markdown: {out_md}")
    print(f"Runs diagnosed: {len(run_diagnostics)}")


if __name__ == "__main__":
    main()
