"""
Generate label mapping report for OddsMind training data.

Read-only script — does not modify any files.

Usage:
    python tools/report_odds_labels.py --input data/odds_real/titan007_pure_v3_time.jsonl --output docs/review/label_mapping_report.md
"""
import json
import argparse
from pathlib import Path
from collections import Counter

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.odds_labels import (
    encode_euro_label,
    encode_asian_label,
    normalize_asian_label_3class,
    ASIAN_3CLASS_TO_ID,
    ASIAN_5CLASS_TO_ID,
    ASIAN_5_TO_3,
    EURO_RESULT_TO_ID,
)


def main():
    p = argparse.ArgumentParser(description="Generate label mapping report")
    p.add_argument("--input", required=True, help="Path to training JSONL")
    p.add_argument("--output", required=True, help="Path to output report")
    p.add_argument("--asian-label-mode", default="3class", choices=["3class", "5class"])
    args = p.parse_args()

    with open(args.input, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f]

    # Collect labels
    euro_raw = Counter()
    asian_raw = Counter()
    asian_3class = Counter()
    label_status_counter = Counter()
    unknown_euro = []
    unknown_asian = []
    null_labels = []

    for s in samples:
        mid = s.get("match_id", "?")
        label = s.get("label", {})
        euro = label.get("euro_result", "")
        asian = label.get("asian_result", "")
        label_status = label.get("asian_label_status", "ok")
        label_status_counter[label_status] += 1

        if not euro or not asian:
            null_labels.append(mid)
            continue

        euro_raw[euro] += 1
        asian_raw[asian] += 1

        # Check euro
        try:
            encode_euro_label(euro)
        except ValueError:
            unknown_euro.append((mid, euro))

        # Check asian
        try:
            encode_asian_label(asian, args.asian_label_mode)
        except ValueError:
            unknown_asian.append((mid, asian))

        # 3class normalization
        try:
            asian_3class[normalize_asian_label_3class(asian)] += 1
        except ValueError:
            pass

    # Generate report
    lines = []
    lines.append("# Label Mapping Report — P1.0B")
    lines.append("")
    lines.append(f"> Input: `{args.input}`")
    lines.append(f"> Total samples: {len(samples)}")
    lines.append(f"> Asian label mode: {args.asian_label_mode}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Euro
    lines.append("## 1. Euro Label Distribution")
    lines.append("")
    lines.append("| Label | Count | Pct |")
    lines.append("|-------|-------|-----|")
    for label in ["home", "draw", "away"]:
        cnt = euro_raw.get(label, 0)
        pct = cnt / len(samples) * 100 if samples else 0
        lines.append(f"| {label} | {cnt} | {pct:.1f}% |")
    lines.append("")

    # Asian raw
    lines.append("## 2. Asian Raw Label Distribution")
    lines.append("")
    lines.append("| Label | Count | Pct |")
    lines.append("|-------|-------|-----|")
    for label in sorted(asian_raw.keys(), key=lambda x: asian_raw[x], reverse=True):
        cnt = asian_raw[label]
        pct = cnt / len(samples) * 100 if samples else 0
        lines.append(f"| {label} | {cnt} | {pct:.1f}% |")
    lines.append("")

    # Asian 3class
    lines.append("## 3. Asian 3-Class Distribution")
    lines.append("")
    lines.append("| Label | Count | Pct |")
    lines.append("|-------|-------|-----|")
    for label in ["upper", "push", "lower"]:
        cnt = asian_3class.get(label, 0)
        pct = cnt / len(samples) * 100 if samples else 0
        lines.append(f"| {label} | {cnt} | {pct:.1f}% |")
    lines.append("")

    # Unknown labels
    lines.append("## 4. Unknown Labels")
    lines.append("")
    lines.append(f"- Unknown euro labels: {len(unknown_euro)}")
    lines.append(f"- Unknown asian labels: {len(unknown_asian)}")
    lines.append(f"- Null/empty labels: {len(null_labels)}")
    if unknown_euro[:5]:
        lines.append(f"- Sample unknown euro: {unknown_euro[:5]}")
    if unknown_asian[:5]:
        lines.append(f"- Sample unknown asian: {unknown_asian[:5]}")
    lines.append("")

    # Asian label status (v4 data)
    if label_status_counter:
        lines.append("## 4b. Asian Label Status")
        lines.append("")
        lines.append("| Status | Count | Pct |")
        lines.append("|--------|-------|-----|")
        for status, cnt in sorted(label_status_counter.items(), key=lambda x: -x[1]):
            pct = cnt / len(samples) * 100 if samples else 0
            lines.append(f"| {status} | {cnt} | {pct:.1f}% |")
        lines.append("")

    # Mapping tables
    lines.append("## 5. Mapping Tables")
    lines.append("")
    lines.append("### Euro (3-class)")
    lines.append("")
    lines.append("| Raw | ID |")
    lines.append("|-----|-----|")
    for label, idx in EURO_RESULT_TO_ID.items():
        lines.append(f"| {label} | {idx} |")
    lines.append("")

    lines.append("### Asian 3-Class")
    lines.append("")
    lines.append("| Raw | ID |")
    lines.append("|-----|-----|")
    for label, idx in ASIAN_3CLASS_TO_ID.items():
        lines.append(f"| {label} | {idx} |")
    lines.append("")

    lines.append("### Asian 5-Class")
    lines.append("")
    lines.append("| Raw | ID |")
    lines.append("|-----|-----|")
    for label, idx in ASIAN_5CLASS_TO_ID.items():
        if not label.startswith("upper_"):  # skip backward compat aliases
            lines.append(f"| {label} | {idx} |")
    lines.append("")

    lines.append("### Asian 5→3 Collapse")
    lines.append("")
    lines.append("| 5-Class | 3-Class |")
    lines.append("|---------|---------|")
    for label, norm in ASIAN_5_TO_3.items():
        if not label.startswith("upper_"):
            lines.append(f"| {label} | {norm} |")
    lines.append("")

    # Notes
    lines.append("## 6. Notes")
    lines.append("")
    lines.append("- **Default training mode**: 3class")
    lines.append("- **Why 3class**: Sample count is modest (~1500). half_win/half_loss may be sparse.")
    lines.append("- **5class available**: `--asian-label-mode 5class` for finer granularity.")
    lines.append("- **No silent fallback**: Unknown labels raise ValueError at encode time.")
    lines.append("- **P1.0C allowed**: Time-based train/val/test split repair.")
    lines.append("")

    # Write
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
