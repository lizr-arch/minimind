"""Train-only draw-anchor mix calibration and locked evaluation."""

from __future__ import annotations

import argparse
import os
import json
import math
import hashlib
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p5_draw_segmentation import load_prediction_rows


GLOBAL_ALPHAS = [0.00, 0.03, 0.05, 0.08, 0.10, 0.15]
GATED_ALPHAS = [0.03, 0.05, 0.08, 0.10]
MARGIN_QUANTILES = [0.20, 0.30, 0.40, 0.50]
SUPPRESSION_QUANTILES = [0.50, 0.60, 0.70, 0.80]
INNER_SPLIT_NAME = "stable_hash_70_15_15"


def stable_bucket(value: str) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % 100


def assign_train_inner_splits(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assigned = []
    for idx, row in enumerate(rows):
        key = row.get("match_id", idx)
        bucket = stable_bucket(str(key))
        if bucket < 70:
            split = "train_base_fit"
        elif bucket < 85:
            split = "train_cal_fit"
        else:
            split = "train_cal_eval"
        assigned.append({**row, "inner_split": split, "inner_bucket": bucket})
    return assigned


def validate_fit_request(fit_split: str, forbid_val_fit: bool, no_test: bool) -> None:
    if not no_test:
        raise ValueError("--no-test is required")
    fit_split = fit_split.lower().strip()
    if fit_split == "test":
        raise ValueError("test split fitting is forbidden")
    if fit_split == "val" and forbid_val_fit:
        raise ValueError("validation fit is forbidden")
    if fit_split != "train":
        raise ValueError("fit_split must be train for parameter selection")


def validate_eval_request(eval_split: str, no_refit_on_val: bool, no_test: bool) -> None:
    if not no_test:
        raise ValueError("--no-test is required")
    eval_split = eval_split.lower().strip()
    if eval_split == "test":
        raise ValueError("test split evaluation is forbidden")
    if eval_split == "val" and not no_refit_on_val:
        raise ValueError("locked validation eval requires --no-refit-on-val")
    if eval_split not in {"train", "val"}:
        raise ValueError("eval_split must be one of: train, val")


def apply_anchor_mix(rows: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    family = params.get("family")
    alpha = float(params.get("alpha", 0.0))
    mixed = []
    for row in rows:
        gate = _gate(row, params)
        effective_alpha = alpha if gate else 0.0
        p_new_draw = (1.0 - effective_alpha) * row["p_draw"] + effective_alpha * row["p_anchor_draw"]
        p_new_draw = min(max(p_new_draw, 1e-12), 1.0 - 1e-12)
        scale = (1.0 - p_new_draw) / max(1.0 - row["p_draw"], 1e-8)
        updated = {
            **row,
            "p_home": row["p_home"] * scale,
            "p_draw": p_new_draw,
            "p_away": row["p_away"] * scale,
            "mix_family": family,
            "mix_alpha": alpha,
            "mix_gate": bool(gate),
        }
        total = updated["p_home"] + updated["p_draw"] + updated["p_away"]
        updated["p_home"] /= total
        updated["p_draw"] /= total
        updated["p_away"] /= total
        mixed.append(updated)
    return mixed


def build_candidate_grid(rows: list[dict[str, Any]], candidate_names: list[str]) -> list[dict[str, Any]]:
    source_rows = [row for row in rows if row.get("inner_split") == "train_cal_fit"]
    candidates: list[dict[str, Any]] = []
    names = {name.strip() for name in candidate_names if name.strip()}
    if "global_anchor_mix" in names:
        candidates.extend({"family": "global_anchor_mix", "alpha": alpha} for alpha in GLOBAL_ALPHAS)
    if "margin_gated_anchor_mix" in names:
        if not source_rows:
            raise ValueError("train_cal_fit rows are required for margin_gated_anchor_mix")
        margins = [_anchor_favorite_margin(row) for row in source_rows]
        for threshold in _quantiles(margins, MARGIN_QUANTILES):
            for alpha in GATED_ALPHAS:
                candidates.append(
                    {
                        "family": "margin_gated_anchor_mix",
                        "alpha": alpha,
                        "margin_threshold": threshold,
                        "threshold_source": "train_cal_fit",
                    }
                )
    if "suppression_gated_anchor_mix" in names:
        if not source_rows:
            raise ValueError("train_cal_fit rows are required for suppression_gated_anchor_mix")
        suppressions = [_draw_suppression(row) for row in source_rows]
        for threshold in _quantiles(suppressions, SUPPRESSION_QUANTILES):
            for alpha in GATED_ALPHAS:
                candidates.append(
                    {
                        "family": "suppression_gated_anchor_mix",
                        "alpha": alpha,
                        "suppression_threshold": threshold,
                        "threshold_source": "train_cal_fit",
                    }
                )
    return candidates


def select_anchor_mix(
    rows: list[dict[str, Any]],
    fit_split: str,
    forbid_val_fit: bool,
    candidate_names: list[str],
    no_test: bool = True,
) -> dict[str, Any]:
    validate_fit_request(fit_split=fit_split, forbid_val_fit=forbid_val_fit, no_test=no_test)
    assigned = rows if all("inner_split" in row for row in rows) else assign_train_inner_splits(rows)
    eval_rows = [row for row in assigned if row["inner_split"] == "train_cal_eval"]
    if not eval_rows:
        raise ValueError("train_cal_eval rows are required for train-only selection")
    baseline = compute_metrics(eval_rows)
    candidates = build_candidate_grid(assigned, candidate_names)
    candidate_reports = []
    for candidate in candidates:
        metrics = compute_metrics(apply_anchor_mix(eval_rows, candidate))
        report = {
            "params": candidate,
            "metrics": _with_deltas(metrics, baseline),
            "passes_gate": _passes_gate(metrics, baseline),
        }
        candidate_reports.append(report)

    passing = [item for item in candidate_reports if item["passes_gate"]]
    ranked = passing or candidate_reports
    selected = min(ranked, key=lambda item: (item["metrics"]["overall_logloss"], -item["metrics"]["draw_recall"]))
    params = {**selected["params"], "source": "train_only_selection"}
    return {
        "mode": "train_only_selection",
        "inner_split": INNER_SPLIT_NAME,
        "fit_split": fit_split,
        "selection_split": "train_cal_eval",
        "baseline_metrics": baseline,
        "candidates": candidate_reports,
        "passed_candidates": len(passing),
        "selected": selected,
        "params": params,
        "test_ids_used": False,
        "validation_fit_used": False,
    }


def evaluate_locked_params(
    rows: list[dict[str, Any]],
    locked_params: dict[str, Any],
    eval_split: str,
    no_refit_on_val: bool,
    no_test: bool = True,
) -> dict[str, Any]:
    validate_eval_request(eval_split=eval_split, no_refit_on_val=no_refit_on_val, no_test=no_test)
    metrics = compute_metrics(apply_anchor_mix(rows, locked_params))
    baseline = compute_metrics(rows)
    return {
        "mode": "locked_eval",
        "eval_split": eval_split,
        "params": locked_params,
        "baseline_metrics": baseline,
        "metrics": _with_deltas(metrics, baseline),
        "test_ids_used": False,
        "validation_fit_used": False,
    }


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot compute metrics on empty rows")
    hard_draw_rows = [row for row in rows if _argmax(row) == 1]
    true_draw_rows = [row for row in rows if row["y_true"] == 1]
    true_draw_hard = [row for row in hard_draw_rows if row["y_true"] == 1]
    return {
        "overall_logloss": _mean([_row_logloss(row) for row in rows]),
        "draw_class_nll": _mean([_binary_draw_nll(row) for row in rows]),
        "draw_recall": _safe_div(len(true_draw_hard), len(true_draw_rows)),
        "draw_precision": _safe_div(len(true_draw_hard), len(hard_draw_rows)),
        "draw_top2_recall": _safe_div(sum(1 for row in true_draw_rows if _draw_in_top2(row)), len(true_draw_rows)),
        "expected_draw_count": sum(row["p_draw"] for row in rows),
        "hard_draw_count": len(hard_draw_rows),
        "top_label_ece": _top_label_ece(rows),
        "classwise_ece_draw": _classwise_ece_draw(rows),
        "home_away_logloss_on_non_draw": _home_away_logloss_on_non_draw(rows),
    }


def write_markdown_report(report: dict[str, Any], out_md: str | Path) -> None:
    lines = [
        "# P5 Draw Anchor Mix",
        "",
        f"Mode: `{report.get('mode')}`",
        f"Test ids used: `{report.get('test_ids_used')}`",
        f"Validation fit used: `{report.get('validation_fit_used')}`",
        "",
    ]
    if report.get("mode") == "train_only_selection":
        lines.append(f"Passed candidates: `{report.get('passed_candidates')}`")
        lines.append(f"Selected params: `{json.dumps(report.get('params', {}), sort_keys=True)}`")
    else:
        lines.append(f"Locked params: `{json.dumps(report.get('params', {}), sort_keys=True)}`")
    lines.extend(["", "## Metrics", ""])
    for key, value in report.get("metrics", report.get("selected", {}).get("metrics", {})).items():
        lines.append(f"- `{key}`: `{value}`")
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _with_deltas(metrics: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    enriched = dict(metrics)
    enriched["delta_logloss_vs_uncalibrated"] = metrics["overall_logloss"] - baseline["overall_logloss"]
    enriched["delta_draw_class_nll"] = metrics["draw_class_nll"] - baseline["draw_class_nll"]
    enriched["delta_home_away_logloss_on_non_draw"] = (
        metrics["home_away_logloss_on_non_draw"] - baseline["home_away_logloss_on_non_draw"]
    )
    return enriched


def _passes_gate(metrics: dict[str, float], baseline: dict[str, float]) -> bool:
    baseline_recall = baseline["draw_recall"]
    return (
        metrics["overall_logloss"] <= baseline["overall_logloss"] + 0.00020
        and metrics["draw_class_nll"] <= baseline["draw_class_nll"] - 0.003
        and metrics["draw_recall"] >= max(0.010, 3.0 * baseline_recall)
        and metrics["classwise_ece_draw"] <= baseline["classwise_ece_draw"] + 0.005
        and metrics["top_label_ece"] <= baseline["top_label_ece"] + 0.003
        and metrics["home_away_logloss_on_non_draw"] <= baseline["home_away_logloss_on_non_draw"] + 0.00030
    )


def _gate(row: dict[str, Any], params: dict[str, Any]) -> bool:
    family = params.get("family")
    if family == "global_anchor_mix":
        return True
    if family == "margin_gated_anchor_mix":
        return _anchor_favorite_margin(row) <= float(params["margin_threshold"])
    if family == "suppression_gated_anchor_mix":
        return _draw_suppression(row) >= float(params["suppression_threshold"])
    raise ValueError(f"unknown anchor mix family: {family}")


def _quantiles(values: list[float], quantiles: list[float]) -> list[float]:
    if not values:
        raise ValueError("cannot compute quantiles on empty values")
    values = sorted(values)
    results = []
    for q in quantiles:
        pos = (len(values) - 1) * q
        low = int(math.floor(pos))
        high = int(math.ceil(pos))
        if low == high:
            value = values[low]
        else:
            weight = pos - low
            value = values[low] * (1.0 - weight) + values[high] * weight
        results.append(round(float(value), 12))
    return results


def _anchor_favorite_margin(row: dict[str, Any]) -> float:
    return abs(row["p_anchor_home"] - row["p_anchor_away"])


def _draw_suppression(row: dict[str, Any]) -> float:
    return _logit(row["p_anchor_draw"]) - _logit(row["p_draw"])


def _row_logloss(row: dict[str, Any]) -> float:
    return -math.log(max([row["p_home"], row["p_draw"], row["p_away"]][int(row["y_true"])], 1e-12))


def _binary_draw_nll(row: dict[str, Any]) -> float:
    p_draw = min(max(row["p_draw"], 1e-12), 1.0 - 1e-12)
    return -math.log(p_draw if int(row["y_true"]) == 1 else 1.0 - p_draw)


def _home_away_logloss_on_non_draw(rows: list[dict[str, Any]]) -> float:
    values = []
    for row in rows:
        if int(row["y_true"]) == 1:
            continue
        den = max(row["p_home"] + row["p_away"], 1e-12)
        prob = row["p_home"] / den if int(row["y_true"]) == 0 else row["p_away"] / den
        values.append(-math.log(max(prob, 1e-12)))
    return _mean(values) if values else 0.0


def _argmax(row: dict[str, Any]) -> int:
    probs = [row["p_home"], row["p_draw"], row["p_away"]]
    return max(range(3), key=lambda idx: probs[idx])


def _draw_in_top2(row: dict[str, Any]) -> bool:
    ranked = sorted(range(3), key=lambda idx: [row["p_home"], row["p_draw"], row["p_away"]][idx], reverse=True)
    return 1 in ranked[:2]


def _top_label_ece(rows: list[dict[str, Any]], n_bins: int = 10) -> float:
    total = len(rows)
    ece = 0.0
    for idx in range(n_bins):
        low = idx / n_bins
        high = (idx + 1) / n_bins
        selected = []
        for row in rows:
            conf = max(row["p_home"], row["p_draw"], row["p_away"])
            if conf >= low and (conf < high or idx == n_bins - 1):
                selected.append((row, conf))
        if not selected:
            continue
        acc = _mean([1.0 if _argmax(row) == int(row["y_true"]) else 0.0 for row, _ in selected])
        avg_conf = _mean([conf for _, conf in selected])
        ece += (len(selected) / total) * abs(acc - avg_conf)
    return round(ece, 6)


def _classwise_ece_draw(rows: list[dict[str, Any]], n_bins: int = 10) -> float:
    total = len(rows)
    ece = 0.0
    for idx in range(n_bins):
        low = idx / n_bins
        high = (idx + 1) / n_bins
        selected = [
            row
            for row in rows
            if row["p_draw"] >= low and (row["p_draw"] < high or idx == n_bins - 1)
        ]
        if not selected:
            continue
        draw_rate = _mean([1.0 if int(row["y_true"]) == 1 else 0.0 for row in selected])
        avg_draw = _mean([row["p_draw"] for row in selected])
        ece += (len(selected) / total) * abs(draw_rate - avg_draw)
    return round(ece, 6)


def _logit(value: float) -> float:
    clipped = min(max(value, 1e-8), 1.0 - 1e-8)
    return math.log(clipped / (1.0 - clipped))


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="P5 train-only draw-anchor mix")
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--base-variant", required=True)
    parser.add_argument("--fit-split")
    parser.add_argument("--inner-split", default=INNER_SPLIT_NAME)
    parser.add_argument("--candidates", default="global_anchor_mix,margin_gated_anchor_mix,suppression_gated_anchor_mix")
    parser.add_argument("--locked-params")
    parser.add_argument("--eval-split")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--no-test", action="store_true")
    parser.add_argument("--forbid-val-fit", action="store_true")
    parser.add_argument("--no-refit-on-val", action="store_true")
    args = parser.parse_args()

    if args.locked_params:
        validate_eval_request(args.eval_split or "val", args.no_refit_on_val, args.no_test)
    else:
        validate_fit_request(args.fit_split or "", args.forbid_val_fit, args.no_test)
        if args.inner_split != INNER_SPLIT_NAME:
            raise ValueError(f"unsupported inner split: {args.inner_split}")

    rows = load_prediction_rows(args.predictions_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.locked_params:
        params = json.loads(Path(args.locked_params).read_text(encoding="utf-8"))
        report = evaluate_locked_params(
            rows,
            locked_params=params,
            eval_split=args.eval_split or "val",
            no_refit_on_val=args.no_refit_on_val,
            no_test=args.no_test,
        )
    else:
        report = select_anchor_mix(
            rows,
            fit_split=args.fit_split or "",
            forbid_val_fit=args.forbid_val_fit,
            candidate_names=args.candidates.split(","),
            no_test=args.no_test,
        )
        (out_dir / "params.json").write_text(json.dumps(report["params"], indent=2), encoding="utf-8")

    report["base_variant"] = args.base_variant
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, out_dir / "report.md")
    print(f"P5 draw anchor mix report: {out_dir / 'report.json'}")
    print("test_ids_used=false")
    print(f"validation_fit_used={str(report.get('validation_fit_used')).lower()}")


if __name__ == "__main__":
    main()
