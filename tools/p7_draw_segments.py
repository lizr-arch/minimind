"""Frozen draw-regime segmentation diagnostics for P7/P6 prediction artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.odds_metrics import logloss_from_probs
from tools.p4_goal_diff_utils import extract_euro_anchor_probs


REQUIRED_SEGMENT_FIELDS = {
    "segment_type",
    "segment_value",
    "n_matches",
    "true_draw_rate",
    "mean_p_draw_all",
    "mean_p_draw_true_draw",
    "draw_underpricing",
    "draw_nll",
    "draw_recall_argmax",
    "draw_top2",
    "draw_rank1_count",
    "draw_rank2_count",
    "draw_rank3_count",
    "mean_draw_margin_to_top",
    "overall_logloss",
    "classwise_nll_home",
    "classwise_nll_draw",
    "classwise_nll_away",
    "ece",
}

EURO_MAP = {"home": 0, "draw": 1, "away": 2}


def _bucket(value: float, edges: list[float], labels: list[str]) -> str:
    if not math.isfinite(value):
        return "unknown"
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


def _valid_minutes(row: dict[str, Any]) -> list[float]:
    out: list[float] = []
    for event in row.get("odds_timeline", []) or row.get("raw_timeline", []) or []:
        try:
            minute = float(event.get("minutes_before_kickoff"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(minute):
            out.append(minute)
    return out


def _anchor_probs(row: dict[str, Any]) -> tuple[float, float, float]:
    anchor, _ = extract_euro_anchor_probs(row)
    values = [float(x) for x in anchor.tolist()]
    total = sum(values)
    if total <= 0 or not all(math.isfinite(x) for x in values):
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return (values[0] / total, values[1] / total, values[2] / total)


def _entropy(probs: tuple[float, float, float]) -> float:
    return float(-sum(p * math.log(max(p, 1e-12)) for p in probs))


def bin_draw_regime(row: dict[str, Any]) -> dict[str, str]:
    p_home, p_draw, p_away = _anchor_probs(row)
    favorite_strength = abs(p_home - p_away)
    home_away_edge = p_home - p_away
    entropy = _entropy((p_home, p_draw, p_away))
    minutes = _valid_minutes(row)
    event_count = len(row.get("odds_timeline", []) or row.get("raw_timeline", []) or [])
    if minutes:
        earliest = max(minutes)
        latest = min(minutes)
        coverage = max(0.0, earliest - latest)
    else:
        earliest = latest = coverage = 0.0

    favorite_bin = _bucket(favorite_strength, [0.08, 0.18, 0.35], ["fav_balanced", "fav_mild", "fav_clear", "fav_strong"])
    draw_bin = _bucket(p_draw, [0.22, 0.28, 0.34], ["draw_low", "draw_mid", "draw_high", "draw_very_high"])
    edge_bin = _bucket(home_away_edge, [-0.15, -0.05, 0.05, 0.15], ["away_edge", "away_lean", "balanced", "home_lean", "home_edge"])
    entropy_bin = _bucket(entropy, [0.85, 1.02], ["entropy_low", "entropy_mid", "entropy_high"])
    coverage_bin = _bucket(coverage, [120.0, 720.0, 1440.0], ["coverage_short", "coverage_mid", "coverage_long", "coverage_very_long"])
    event_bin = _bucket(float(event_count), [5.0, 20.0, 60.0], ["events_very_low", "events_low", "events_mid", "events_high"])
    earliest_bin = _bucket(earliest, [120.0, 720.0, 1440.0], ["earliest_late", "earliest_same_day", "earliest_prev_day", "earliest_early"])
    latest_bin = _bucket(latest, [5.0, 60.0, 240.0], ["latest_closing", "latest_hour", "latest_4h", "latest_early"])
    odds_regime = "|".join([favorite_bin, draw_bin, entropy_bin, coverage_bin])
    return {
        "league": str(row.get("league_id") or "unknown"),
        "favorite_strength_bin": favorite_bin,
        "implied_draw_prior_bin": draw_bin,
        "home_away_edge_bin": edge_bin,
        "odds_entropy_bin": entropy_bin,
        "time_coverage_bin": coverage_bin,
        "event_count_bin": event_bin,
        "earliest_event_minutes_bin": earliest_bin,
        "latest_event_minutes_bin": latest_bin,
        "odds_regime_bin": odds_regime,
    }


def _ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    confidence, preds = probs.max(dim=-1)
    correct = (preds == labels).float()
    ece = 0.0
    for i in range(n_bins):
        low = i / n_bins
        high = (i + 1) / n_bins
        mask = (confidence >= low) & (confidence <= high if i == n_bins - 1 else confidence < high)
        if int(mask.sum().item()) == 0:
            continue
        ece += float(mask.float().mean().item()) * abs(
            float(confidence[mask].mean().item()) - float(correct[mask].mean().item())
        )
    return float(ece)


def _segment_metrics(segment_type: str, segment_value: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    labels = torch.tensor([int(item["y_true"]) for item in items], dtype=torch.long)
    probs = torch.tensor([[float(item["p_home"]), float(item["p_draw"]), float(item["p_away"])] for item in items])
    probs = probs.clamp_min(1e-8)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    preds = probs.argmax(dim=-1)
    true_draw = labels == 1
    pred_draw = preds == 1
    n_true_draw = int(true_draw.sum().item())
    top2 = torch.topk(probs, k=2, dim=-1).indices
    draw_rank = (torch.argsort(probs, dim=-1, descending=True) == 1).nonzero(as_tuple=False)[:, 1] + 1
    draw_margin = probs.max(dim=-1).values - probs[:, 1]

    result = {
        "segment_type": segment_type,
        "segment_value": segment_value,
        "n_matches": len(items),
        "true_draw_rate": float(true_draw.float().mean().item()) if len(items) else 0.0,
        "mean_p_draw_all": float(probs[:, 1].mean().item()) if len(items) else 0.0,
        "mean_p_draw_true_draw": float(probs[true_draw, 1].mean().item()) if n_true_draw else 0.0,
        "draw_underpricing": 0.0,
        "draw_nll": float((-torch.log(probs[true_draw, 1])).mean().item()) if n_true_draw else 0.0,
        "draw_recall_argmax": float((pred_draw & true_draw).sum().item() / n_true_draw) if n_true_draw else 0.0,
        "draw_top2": float((top2[true_draw] == 1).any(dim=-1).float().mean().item()) if n_true_draw else 0.0,
        "draw_rank1_count": int((draw_rank == 1).sum().item()),
        "draw_rank2_count": int((draw_rank == 2).sum().item()),
        "draw_rank3_count": int((draw_rank == 3).sum().item()),
        "mean_draw_margin_to_top": float(draw_margin.mean().item()) if len(items) else 0.0,
        "overall_logloss": logloss_from_probs(probs, labels) if len(items) else 0.0,
        "ece": _ece(probs, labels) if len(items) else 0.0,
    }
    result["draw_underpricing"] = result["true_draw_rate"] - result["mean_p_draw_all"]
    for class_idx, name in [(0, "home"), (1, "draw"), (2, "away")]:
        mask = labels == class_idx
        result[f"classwise_nll_{name}"] = float((-torch.log(probs[mask, class_idx])).mean().item()) if int(mask.sum().item()) else 0.0
    return result


def build_draw_segment_report(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    run_id: str,
) -> dict[str, Any]:
    row_by_id = {str(row.get("match_id")): row for row in rows}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    unmatched_predictions = 0
    for pred in predictions:
        match_id = str(pred.get("match_id"))
        row = row_by_id.get(match_id)
        if row is None:
            unmatched_predictions += 1
            continue
        bins = bin_draw_regime(row)
        for segment_type, segment_value in bins.items():
            groups[(segment_type, segment_value)].append(pred)
    segments = [
        _segment_metrics(segment_type, segment_value, items)
        for (segment_type, segment_value), items in sorted(groups.items())
    ]
    return {
        "phase": "P7 frozen draw regime diagnostics",
        "run_id": run_id,
        "n_predictions": len(predictions),
        "n_rows": len(rows),
        "unmatched_predictions": unmatched_predictions,
        "segment_count": len(segments),
        "segments": segments,
    }


def write_draw_segment_outputs(report: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(report.get("run_id", "unknown"))
    json_path = out_dir / f"p7_draw_segments_{run_id}.json"
    csv_path = out_dir / f"p7_draw_segments_{run_id}.csv"
    md_path = out_dir / f"p7_draw_segment_summary_{run_id}.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    segments = report.get("segments", [])
    fields = sorted(REQUIRED_SEGMENT_FIELDS)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(segments)
    lines = [
        f"# P7 Draw Segment Summary: {run_id}",
        "",
        f"- predictions: {report.get('n_predictions', 0)}",
        f"- segments: {report.get('segment_count', 0)}",
        f"- unmatched_predictions: {report.get('unmatched_predictions', 0)}",
        "",
        "| segment | value | n | draw_rate | mean_p_draw | draw_recall | draw_top2 | draw_nll |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in segments[:50]:
        lines.append(
            f"| {item['segment_type']} | {item['segment_value']} | {item['n_matches']} | "
            f"{item['true_draw_rate']:.4f} | {item['mean_p_draw_all']:.4f} | "
            f"{item['draw_recall_argmax']:.4f} | {item['draw_top2']:.4f} | {item['draw_nll']:.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": md_path}


def read_predictions_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_rows_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build frozen P7 draw segment diagnostics")
    parser.add_argument("--data", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", default="diagnostics")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build_draw_segment_report(
        read_rows_jsonl(Path(args.data)),
        read_predictions_csv(Path(args.predictions)),
        run_id=args.run_id,
    )
    outputs = write_draw_segment_outputs(report, Path(args.out_dir))
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
