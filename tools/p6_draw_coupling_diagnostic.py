"""Diagnostic for whether draw signal is close to final 1X2 argmax."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROB_COLS = ("p_home", "p_draw", "p_away")
MARGIN_BINS = (
    ("le_0_01", 0.01),
    ("le_0_03", 0.03),
    ("le_0_05", 0.05),
    ("le_0_10", 0.10),
    ("gt_0_10", float("inf")),
)


def diagnose_prediction_file(path: str | Path) -> dict[str, Any]:
    rows = _load_prediction_rows(path)
    true_draw_rows = [row for row in rows if row["y_true"] == 1]
    rank_hist = {"rank1": 0, "rank2": 0, "rank3": 0}
    margin_hist = {name: 0 for name, _ in MARGIN_BINS}
    margins = []

    for row in true_draw_rows:
        probs = [row[col] for col in PROB_COLS]
        p_draw = row["p_draw"]
        sorted_indices = sorted(range(3), key=lambda idx: probs[idx], reverse=True)
        draw_rank = sorted_indices.index(1) + 1
        rank_hist[f"rank{draw_rank}"] += 1
        margin = max(probs) - p_draw
        margins.append(margin)
        for name, upper in MARGIN_BINS:
            if margin <= upper:
                margin_hist[name] += 1
                break

    aux_probability_available = bool(rows and "p_aux_draw" in rows[0])
    return {
        "path": str(path),
        "row_count": len(rows),
        "true_draw_count": len(true_draw_rows),
        "draw_rank_histogram": rank_hist,
        "draw_margin_histogram": margin_hist,
        "mean_draw_margin_to_top": _mean(margins),
        "median_draw_margin_to_top": _median(margins),
        "mean_p_draw_on_true_draw": _mean([row["p_draw"] for row in true_draw_rows]),
        "aux_probability_available": aux_probability_available,
        "selected_lambda": None,
    }


def build_diagnostic(
    p6_root: str | Path,
    p6_1_root: str | Path,
    expected_seeds: list[int],
) -> dict[str, Any]:
    roots = [
        ("p6_baseline", Path(p6_root), "p6_euro_default_seed"),
        ("p6_1_draw_aux_0005", Path(p6_1_root), "p6_euro_default_draw_aux_0005_seed"),
        ("p6_1_draw_aux_001", Path(p6_1_root), "p6_euro_default_draw_aux_001_seed"),
        ("p6_1_draw_aux_002", Path(p6_1_root), "p6_euro_default_draw_aux_002_seed"),
    ]
    variants = []
    for name, root, prefix in roots:
        seed_reports = []
        for seed in expected_seeds:
            pred_path = root / f"{prefix}{seed}" / "val_predictions.csv"
            if pred_path.exists():
                report = diagnose_prediction_file(pred_path)
                report["seed"] = seed
                seed_reports.append(report)
        variants.append(_aggregate_variant(name, seed_reports))
    return {
        "phase": "P6 draw coupling diagnostic",
        "expected_seeds": expected_seeds,
        "selected_lambda": None,
        "variants": variants,
    }


def write_markdown_report(payload: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# P6 Draw Coupling Diagnostic",
        "",
        "| Variant | seeds | true draws | rank1 | rank2 | rank3 | mean margin | median margin | mean p_draw true draw |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload.get("variants", []):
        rank = item.get("draw_rank_histogram", {})
        lines.append(
            f"| {item.get('name')} | {item.get('seed_count')} | {item.get('true_draw_count')} | "
            f"{rank.get('rank1')} | {rank.get('rank2')} | {rank.get('rank3')} | "
            f"{_fmt(item.get('mean_draw_margin_to_top'))} | {_fmt(item.get('median_draw_margin_to_top'))} | "
            f"{_fmt(item.get('mean_p_draw_on_true_draw'))} |"
        )
    lines.extend(["", "selected_lambda: `None`", ""])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _aggregate_variant(name: str, seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    rank = {"rank1": 0, "rank2": 0, "rank3": 0}
    margin = {item[0]: 0 for item in MARGIN_BINS}
    for report in seed_reports:
        for key in rank:
            rank[key] += int(report["draw_rank_histogram"].get(key, 0))
        for key in margin:
            margin[key] += int(report["draw_margin_histogram"].get(key, 0))
    return {
        "name": name,
        "seed_count": len(seed_reports),
        "true_draw_count": sum(int(report["true_draw_count"]) for report in seed_reports),
        "draw_rank_histogram": rank,
        "draw_margin_histogram": margin,
        "mean_draw_margin_to_top": _mean([report["mean_draw_margin_to_top"] for report in seed_reports]),
        "median_draw_margin_to_top": _mean([report["median_draw_margin_to_top"] for report in seed_reports]),
        "mean_p_draw_on_true_draw": _mean([report["mean_p_draw_on_true_draw"] for report in seed_reports]),
        "aux_probability_available": any(bool(report["aux_probability_available"]) for report in seed_reports),
        "selected_lambda": None,
    }


def _load_prediction_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed = {"match_id": row["match_id"], "y_true": int(row["y_true"])}
            for col in PROB_COLS:
                parsed[col] = float(row[col])
            if "p_aux_draw" in row and row["p_aux_draw"] not in ("", None):
                parsed["p_aux_draw"] = float(row["p_aux_draw"])
            rows.append(parsed)
    return rows


def _mean(values: list[float | None]) -> float:
    cleaned = [float(value) for value in values if value is not None]
    return float(sum(cleaned) / len(cleaned)) if cleaned else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2.0)


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "N/A"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose final p_draw coupling opportunities")
    parser.add_argument("--p6-root", default="runs/p6_residual_patch_itransformer")
    parser.add_argument("--p6-1-root", default="runs/p6_1_draw_aux")
    parser.add_argument("--expected-seeds", default="42,123,2025")
    parser.add_argument("--expected-val-rows", type=int, default=4747)
    parser.add_argument("--out-json", default="runs/p6_1_draw_aux/p6_1_draw_coupling_diagnostic.json")
    parser.add_argument("--out-md", default="runs/p6_1_draw_aux/p6_1_draw_coupling_diagnostic.md")
    args = parser.parse_args()
    seeds = [int(seed.strip()) for seed in args.expected_seeds.split(",") if seed.strip()]
    payload = build_diagnostic(args.p6_root, args.p6_1_root, seeds)
    for variant in payload["variants"]:
        expected_rows = args.expected_val_rows * int(variant.get("seed_count", 0))
        variant["expected_row_count"] = expected_rows
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown_report(payload, out_md)
    print(f"P6 draw coupling diagnostic json: {out_json}")
    print(f"P6 draw coupling diagnostic markdown: {out_md}")


if __name__ == "__main__":
    main()
