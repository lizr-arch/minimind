"""P6.3 draw decision/calibration probe.

This tool evaluates pre-registered decision policies on validation predictions.
It does not fit probabilities, choose a policy, or read the test split.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROB_COLS = ("p_home", "p_draw", "p_away")
DEFAULT_POLICIES = [
    "argmax",
    "draw_top2_margin_le_0_03",
    "draw_top2_margin_le_0_05",
    "draw_top2_margin_le_0_10",
    "draw_top2_margin_le_0_20",
    "draw_top2_any",
]


def apply_draw_decision_policy(probs: list[float], policy: str) -> int:
    if len(probs) != 3:
        raise ValueError("1X2 probabilities must have length 3")
    argmax = max(range(3), key=lambda idx: probs[idx])
    if policy == "argmax":
        return argmax
    sorted_indices = sorted(range(3), key=lambda idx: probs[idx], reverse=True)
    draw_rank = sorted_indices.index(1) + 1
    if policy == "draw_top2_any":
        return 1 if draw_rank <= 2 else argmax
    if policy.startswith("draw_top2_margin_le_"):
        threshold = _parse_margin_threshold(policy)
        if draw_rank <= 2 and (probs[sorted_indices[0]] - probs[1]) <= threshold:
            return 1
        return argmax
    raise ValueError(f"Unknown draw decision policy: {policy}")


def evaluate_policy_on_predictions(path: str | Path, policy: str) -> dict[str, Any]:
    rows = _load_rows(path)
    preds = [apply_draw_decision_policy(row["probs"], policy) for row in rows]
    labels = [row["y_true"] for row in rows]
    argmax_preds = [max(range(3), key=lambda idx: row["probs"][idx]) for row in rows]
    n = len(rows)
    correct = [int(pred == label) for pred, label in zip(preds, labels)]
    argmax_correct = [int(pred == label) for pred, label in zip(argmax_preds, labels)]
    true_draw = [label == 1 for label in labels]
    pred_draw = [pred == 1 for pred in preds]
    argmax_draw = [pred == 1 for pred in argmax_preds]
    tp_draw = sum(1 for td, pd in zip(true_draw, pred_draw) if td and pd)
    pred_draw_count = sum(1 for item in pred_draw if item)
    true_draw_count = sum(1 for item in true_draw if item)
    draw_precision = tp_draw / pred_draw_count if pred_draw_count else 0.0
    draw_recall = tp_draw / true_draw_count if true_draw_count else 0.0
    draw_f1 = (
        2 * draw_precision * draw_recall / (draw_precision + draw_recall)
        if draw_precision + draw_recall > 0
        else 0.0
    )
    argmax_accuracy = sum(argmax_correct) / n if n else 0.0
    accuracy = sum(correct) / n if n else 0.0
    return {
        "path": str(path),
        "policy": policy,
        "row_count": n,
        "accuracy": accuracy,
        "argmax_accuracy": argmax_accuracy,
        "accuracy_delta_vs_argmax": accuracy - argmax_accuracy,
        "draw_recall": draw_recall,
        "draw_precision": draw_precision,
        "draw_f1": draw_f1,
        "true_draw_count": true_draw_count,
        "draw_decision_count": pred_draw_count,
        "argmax_draw_count": sum(1 for item in argmax_draw if item),
        "non_draw_to_draw_count": sum(1 for ad, pd in zip(argmax_draw, pred_draw) if not ad and pd),
        "selected_policy": None,
    }


def run_policy_probe(
    runs_root: str | Path,
    run_prefix: str,
    expected_seeds: list[int],
    policies: list[str] | None = None,
) -> dict[str, Any]:
    policies = policies or DEFAULT_POLICIES
    root = Path(runs_root)
    per_policy = []
    for policy in policies:
        seed_metrics = []
        for seed in expected_seeds:
            pred_path = root / f"{run_prefix}{seed}" / "val_predictions.csv"
            if pred_path.exists():
                metrics = evaluate_policy_on_predictions(pred_path, policy)
                metrics["seed"] = seed
                seed_metrics.append(metrics)
        per_policy.append(_aggregate_policy(policy, seed_metrics))
    return {
        "phase": "P6.3 draw decision/calibration probe",
        "runs_root": str(runs_root),
        "run_prefix": run_prefix,
        "expected_seeds": expected_seeds,
        "selected_policy": None,
        "posthoc_val_fit_used": False,
        "test_ids_used": False,
        "policies": per_policy,
    }


def write_markdown_report(payload: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# P6.3 Draw Decision Probe",
        "",
        f"- selected_policy: `{payload.get('selected_policy')}`",
        f"- test_ids_used: `{payload.get('test_ids_used')}`",
        f"- posthoc_val_fit_used: `{payload.get('posthoc_val_fit_used')}`",
        "",
        "| Policy | seeds | accuracy | delta vs argmax | draw recall | draw precision | draw F1 | draw decisions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload.get("policies", []):
        lines.append(
            f"| {item.get('policy')} | {item.get('seed_count')} | {_fmt(item.get('accuracy'))} | "
            f"{_fmt(item.get('accuracy_delta_vs_argmax'))} | {_fmt(item.get('draw_recall'))} | "
            f"{_fmt(item.get('draw_precision'))} | {_fmt(item.get('draw_f1'))} | "
            f"{_fmt(item.get('draw_decision_count'))} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_policy(policy: str, seed_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "policy": policy,
        "seed_count": len(seed_metrics),
        "accuracy": _mean([item["accuracy"] for item in seed_metrics]),
        "argmax_accuracy": _mean([item["argmax_accuracy"] for item in seed_metrics]),
        "accuracy_delta_vs_argmax": _mean([item["accuracy_delta_vs_argmax"] for item in seed_metrics]),
        "draw_recall": _mean([item["draw_recall"] for item in seed_metrics]),
        "draw_precision": _mean([item["draw_precision"] for item in seed_metrics]),
        "draw_f1": _mean([item["draw_f1"] for item in seed_metrics]),
        "true_draw_count": sum(int(item["true_draw_count"]) for item in seed_metrics),
        "draw_decision_count": _mean([item["draw_decision_count"] for item in seed_metrics]),
        "argmax_draw_count": _mean([item["argmax_draw_count"] for item in seed_metrics]),
        "non_draw_to_draw_count": _mean([item["non_draw_to_draw_count"] for item in seed_metrics]),
        "selected_policy": None,
        "seed_metrics": seed_metrics,
    }


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "match_id": row["match_id"],
                    "y_true": int(row["y_true"]),
                    "probs": [float(row[col]) for col in PROB_COLS],
                }
            )
    return rows


def _parse_margin_threshold(policy: str) -> float:
    raw = policy.removeprefix("draw_top2_margin_le_")
    return float(raw.replace("_", "."))


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "N/A"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P6.3 draw decision/calibration probe")
    parser.add_argument("--runs-root", default="runs/p6_residual_patch_itransformer")
    parser.add_argument("--run-prefix", default="p6_euro_default_seed")
    parser.add_argument("--expected-seeds", default="42,123,2025")
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--out-json", default="runs/p6_3_draw_decision/p6_3_draw_decision_probe.json")
    parser.add_argument("--out-md", default="runs/p6_3_draw_decision/p6_3_draw_decision_probe.md")
    args = parser.parse_args()
    seeds = [int(seed.strip()) for seed in args.expected_seeds.split(",") if seed.strip()]
    policies = [policy.strip() for policy in args.policies.split(",") if policy.strip()]
    payload = run_policy_probe(args.runs_root, args.run_prefix, seeds, policies)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown_report(payload, out_md)
    print(f"P6.3 draw decision probe json: {out_json}")
    print(f"P6.3 draw decision probe markdown: {out_md}")


if __name__ == "__main__":
    main()
