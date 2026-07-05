"""Train-only draw-prone segmentation over exported P4 prediction CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable


REQUIRED_COLUMNS = {
    "match_id",
    "y_true",
    "p_home",
    "p_draw",
    "p_away",
    "p_anchor_home",
    "p_anchor_draw",
    "p_anchor_away",
    "p_home_from_diff",
    "p_draw_from_diff",
    "p_away_from_diff",
}

SEGMENT_NAMES = [
    "anchor_draw_prob_decile",
    "anchor_favorite_margin_decile",
    "final_favorite_margin_decile",
    "final_p_draw_decile",
    "draw_suppression_decile",
    "p_diff_draw_decile",
    "final_vs_diff_draw_gap_decile",
]


def validate_segmentation_request(split: str, allow_val_diagnostic: bool) -> None:
    split = split.lower().strip()
    if split == "test":
        raise ValueError("test split segmentation is forbidden")
    if split == "val" and not allow_val_diagnostic:
        raise ValueError("validation diagnostic requires --allow-val-diagnostic")
    if split not in {"train", "val"}:
        raise ValueError("split must be one of: train, val")


def load_prediction_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"predictions CSV missing required columns: {missing}")
        return [_parse_row(row) for row in reader]


def build_segmentation_report(
    rows: list[dict[str, Any]],
    split: str,
    n_bins: int = 10,
    allow_val_diagnostic: bool = False,
) -> dict[str, Any]:
    validate_segmentation_request(split, allow_val_diagnostic=allow_val_diagnostic)
    if not rows:
        raise ValueError("cannot segment empty predictions")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    metrics: dict[str, tuple[str, Callable[[dict[str, Any]], float]]] = {
        "anchor_draw_prob_decile": ("anchor_draw_prob", lambda row: row["p_anchor_draw"]),
        "anchor_favorite_margin_decile": ("anchor_favorite_margin", _anchor_favorite_margin),
        "final_favorite_margin_decile": ("final_favorite_margin", _final_favorite_margin),
        "final_p_draw_decile": ("final_p_draw", lambda row: row["p_draw"]),
        "draw_suppression_decile": ("draw_suppression", _draw_suppression),
        "p_diff_draw_decile": ("p_diff_draw", lambda row: row["p_draw_from_diff"]),
        "final_vs_diff_draw_gap_decile": ("final_vs_diff_draw_gap", _final_vs_diff_draw_gap),
    }
    segments = {}
    for name in SEGMENT_NAMES:
        metric_name, getter = metrics[name]
        segments[name] = {
            "metric": metric_name,
            "threshold_source": "input_split",
            "bins": _build_rank_bins(rows, getter, n_bins),
        }

    return {
        "split": split,
        "n_rows": len(rows),
        "threshold_source": "input_split",
        "test_ids_used": False,
        "segments": segments,
    }


def write_markdown_report(report: dict[str, Any], out_md: str | Path) -> None:
    lines = [
        "# P5 Draw Segmentation",
        "",
        f"Split: `{report['split']}`",
        f"Rows: `{report['n_rows']}`",
        "",
        "All thresholds are computed from the input split. This report is diagnostic-only for validation splits.",
        "",
    ]
    for segment_name, segment in report.get("segments", {}).items():
        lines.extend([f"## {segment_name}", ""])
        lines.append(
            "| bin | n | true_draw_rate | mean_final_p_draw | mean_anchor_p_draw | mean_p_diff_draw | draw_calibration_gap | final_logloss | draw_class_nll | home_away_logloss_on_non_draw | argmax_draw_count |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for item in segment.get("bins", []):
            lines.append(
                f"| {item['bin']} | {item['n']} | {_fmt(item['true_draw_rate'])} | "
                f"{_fmt(item['mean_final_p_draw'])} | {_fmt(item['mean_anchor_p_draw'])} | "
                f"{_fmt(item['mean_p_diff_draw'])} | {_fmt(item['draw_calibration_gap'])} | "
                f"{_fmt(item['final_logloss'])} | {_fmt(item['draw_class_nll'])} | "
                f"{_fmt(item['home_away_logloss_on_non_draw'])} | {item['argmax_draw_count']} |"
            )
        lines.append("")
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")


def _build_rank_bins(rows: list[dict[str, Any]], getter: Callable[[dict[str, Any]], float], n_bins: int) -> list[dict[str, Any]]:
    ranked = sorted(((getter(row), idx, row) for idx, row in enumerate(rows)), key=lambda item: (item[0], item[1]))
    bins = []
    for bin_idx in range(n_bins):
        start = len(ranked) * bin_idx // n_bins
        end = len(ranked) * (bin_idx + 1) // n_bins
        selected = [item[2] for item in ranked[start:end]]
        values = [item[0] for item in ranked[start:end]]
        bins.append(
            {
                "bin": bin_idx,
                "low": _round(min(values)) if values else None,
                "high": _round(max(values)) if values else None,
                **_compute_bin_metrics(selected),
            }
        )
    return bins


def _compute_bin_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    true_draw_rate = _mean([1.0 if row["y_true"] == 1 else 0.0 for row in rows])
    mean_final_p_draw = _mean([row["p_draw"] for row in rows])
    return {
        "n": n,
        "true_draw_rate": true_draw_rate,
        "mean_final_p_draw": mean_final_p_draw,
        "mean_anchor_p_draw": _mean([row["p_anchor_draw"] for row in rows]),
        "mean_p_diff_draw": _mean([row["p_draw_from_diff"] for row in rows]),
        "draw_calibration_gap": None if true_draw_rate is None or mean_final_p_draw is None else _round(mean_final_p_draw - true_draw_rate),
        "final_logloss": _mean([_row_logloss(row) for row in rows]),
        "draw_class_nll": _mean([_binary_draw_nll(row) for row in rows]),
        "home_away_logloss_on_non_draw": _home_away_logloss_on_non_draw(rows),
        "argmax_draw_count": sum(1 for row in rows if _argmax_final(row) == 1),
    }


def _parse_row(row: dict[str, str]) -> dict[str, Any]:
    parsed = {"match_id": row["match_id"], "y_true": int(float(row["y_true"]))}
    for key in REQUIRED_COLUMNS - {"match_id", "y_true"}:
        parsed[key] = float(row[key])
    if "pred_class" in row and row["pred_class"] != "":
        parsed["pred_class"] = int(float(row["pred_class"]))
    return parsed


def _anchor_favorite_margin(row: dict[str, Any]) -> float:
    return abs(row["p_anchor_home"] - row["p_anchor_away"])


def _final_favorite_margin(row: dict[str, Any]) -> float:
    return abs(row["p_home"] - row["p_away"])


def _final_vs_diff_draw_gap(row: dict[str, Any]) -> float:
    return row["p_draw"] - row["p_draw_from_diff"]


def _draw_suppression(row: dict[str, Any]) -> float:
    return _logit(row["p_anchor_draw"]) - _logit(row["p_draw"])


def _row_logloss(row: dict[str, Any]) -> float:
    probs = [row["p_home"], row["p_draw"], row["p_away"]]
    return -math.log(max(probs[row["y_true"]], 1e-12))


def _binary_draw_nll(row: dict[str, Any]) -> float:
    p_draw = min(max(row["p_draw"], 1e-12), 1.0 - 1e-12)
    return -math.log(p_draw if row["y_true"] == 1 else 1.0 - p_draw)


def _home_away_logloss_on_non_draw(rows: list[dict[str, Any]]) -> float | None:
    values = []
    for row in rows:
        if row["y_true"] == 1:
            continue
        den = max(row["p_home"] + row["p_away"], 1e-12)
        prob = row["p_home"] / den if row["y_true"] == 0 else row["p_away"] / den
        values.append(-math.log(max(prob, 1e-12)))
    return _mean(values)


def _argmax_final(row: dict[str, Any]) -> int:
    if "pred_class" in row:
        return int(row["pred_class"])
    probs = [row["p_home"], row["p_draw"], row["p_away"]]
    return max(range(3), key=lambda idx: probs[idx])


def _logit(value: float) -> float:
    clipped = min(max(value, 1e-8), 1.0 - 1e-8)
    return math.log(clipped / (1.0 - clipped))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return _round(sum(values) / len(values))


def _round(value: float) -> float:
    return round(float(value), 6)


def _fmt(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="P5 train-only draw segmentation")
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--allow-val-diagnostic", action="store_true")
    args = parser.parse_args()

    validate_segmentation_request(args.split, args.allow_val_diagnostic)
    rows = load_prediction_rows(args.predictions_csv)
    report = build_segmentation_report(
        rows,
        split=args.split,
        n_bins=args.n_bins,
        allow_val_diagnostic=args.allow_val_diagnostic,
    )

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, out_md)
    print(f"P5 draw segmentation json: {out_json}")
    print(f"P5 draw segmentation markdown: {out_md}")
    print("test_ids_used=false")


if __name__ == "__main__":
    main()
