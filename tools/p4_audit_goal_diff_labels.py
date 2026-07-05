"""Audit P4 goal-diff labels and Euro anchor availability."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.odds_patch_itransformer_v2 import extract_cross_market_feature
from tools.p4_goal_diff_utils import extract_euro_anchor_probs, goal_diff_to_bucket
from tools.p4_train_residual_goal_diff import load_rows_for_ids, load_split_ids


def build_split_audit_payload(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "train": build_audit_payload(train_rows),
        "val": build_audit_payload(val_rows),
        "test_ids_used": False,
    }


def build_audit_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bucket_counts: Counter[int] = Counter()
    label_counts: Counter[str] = Counter()
    asian_lines: Counter[str] = Counter()
    labelled_rows = 0
    missing_score_rows = 0
    asian_present_rows = 0
    anchor_fallback_count = 0
    negative_time_valid_euro_count = 0

    for row in rows:
        label = row.get("label", {})
        result = label.get("euro_result")
        if result:
            label_counts[str(result)] += 1
        if _row_has_asian(row):
            asian_present_rows += 1
            for line in _asian_lines(row):
                asian_lines[str(line)] += 1
        _, diag = extract_euro_anchor_probs(row)
        anchor_fallback_count += int(diag["used_fallback"])
        negative_time_valid_euro_count += int(diag["negative_time_valid_euro_count"])
        if result not in {"home", "draw", "away"}:
            continue
        if "home_goals" not in label or "away_goals" not in label:
            missing_score_rows += 1
            continue
        try:
            goal_diff = int(label["home_goals"]) - int(label["away_goals"])
        except (TypeError, ValueError):
            missing_score_rows += 1
            continue
        labelled_rows += 1
        bucket_counts[goal_diff_to_bucket(goal_diff)] += 1

    return {
        "rows": len(rows),
        "labelled_rows": labelled_rows,
        "missing_score_rows": missing_score_rows,
        "label_counts": dict(label_counts),
        "goal_diff_bucket_counts": {str(idx): bucket_counts.get(idx, 0) for idx in range(7)},
        "asian_present_rate": asian_present_rows / len(rows) if rows else 0.0,
        "asian_line_top_values": [
            {"line": line, "count": count}
            for line, count in asian_lines.most_common(10)
        ],
        "anchor_fallback_count": anchor_fallback_count,
        "negative_time_valid_euro_count": negative_time_valid_euro_count,
    }


def _timeline(row: dict[str, Any]) -> list[dict[str, Any]]:
    return row.get("raw_timeline", []) or row.get("odds_timeline", []) or []


def _row_has_asian(row: dict[str, Any]) -> bool:
    return any(extract_cross_market_feature(event, "has_asian") > 0 for event in _timeline(row))


def _asian_lines(row: dict[str, Any]) -> list[float]:
    lines = []
    for event in _timeline(row):
        if extract_cross_market_feature(event, "has_asian") <= 0:
            continue
        try:
            line = float(event.get("asian_line", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        lines.append(round(line, 2))
    return lines


def _legacy_build_audit_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bucket_counts: Counter[int] = Counter()
    label_counts: Counter[str] = Counter()
    valid_label_rows = 0
    anchor_fallback_count = 0
    negative_time_valid_euro_count = 0

    for row in rows:
        label = row.get("label", {})
        result = label.get("euro_result")
        if result:
            label_counts[str(result)] += 1
        if result in {"home", "draw", "away"} and "home_goals" in label and "away_goals" in label:
            try:
                goal_diff = int(label["home_goals"]) - int(label["away_goals"])
            except (TypeError, ValueError):
                continue
            valid_label_rows += 1
            bucket_counts[goal_diff_to_bucket(goal_diff)] += 1
            _, diag = extract_euro_anchor_probs(row)
            anchor_fallback_count += int(diag["used_fallback"])
            negative_time_valid_euro_count += int(diag["negative_time_valid_euro_count"])

    return {
        "rows": len(rows),
        "valid_label_rows": valid_label_rows,
        "label_counts": dict(label_counts),
        "goal_diff_bucket_counts": {str(idx): bucket_counts.get(idx, 0) for idx in range(7)},
        "anchor_fallback_count": anchor_fallback_count,
        "negative_time_valid_euro_count": negative_time_valid_euro_count,
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# P4 Goal-Diff Label Audit",
        "",
        f"- test_ids_used: {payload.get('test_ids_used', False)}",
    ]
    for split in ["train", "val"]:
        section = payload.get(split, {})
        lines.extend([
            "",
            f"## {split}",
            "",
            f"- rows: {section.get('rows', 0)}",
            f"- labelled_rows: {section.get('labelled_rows', 0)}",
            f"- missing_score_rows: {section.get('missing_score_rows', 0)}",
            f"- asian_present_rate: {section.get('asian_present_rate', 0.0):.6f}",
            f"- anchor_fallback_count: {section.get('anchor_fallback_count', 0)}",
            f"- negative_time_valid_euro_count: {section.get('negative_time_valid_euro_count', 0)}",
            "",
            "### Goal-Diff Buckets",
            "",
        ])
        for bucket, count in section.get("goal_diff_bucket_counts", {}).items():
            lines.append(f"- {bucket}: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit P4 goal-diff labels and anchor availability")
    parser.add_argument("--data", required=True)
    parser.add_argument("--train-ids", required=True)
    parser.add_argument("--val-ids", required=True)
    parser.add_argument("--out-dir", default="runs/p4_residual_goal_diff/audit")
    parser.add_argument("--out-json")
    parser.add_argument("--out-md")
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    json_path = Path(args.out_json) if args.out_json else out_dir / "goal_diff_audit.json"
    md_path = Path(args.out_md) if args.out_md else out_dir / "goal_diff_audit.md"
    if not args.allow_overwrite and (json_path.exists() or md_path.exists()):
        raise FileExistsError(f"Refusing to overwrite existing audit outputs in {out_dir}")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    train_rows = load_rows_for_ids(args.data, load_split_ids(args.train_ids))
    val_rows = load_rows_for_ids(args.data, load_split_ids(args.val_ids))
    payload = build_split_audit_payload(train_rows, val_rows)
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    write_markdown(payload, md_path)
    print(f"P4 audit markdown: {md_path}")
    print(f"P4 audit json: {json_path}")


if __name__ == "__main__":
    main()
