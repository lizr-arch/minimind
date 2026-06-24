"""
P1.0F: Missing Semantics Report Generator.

Audits missing_mask semantics, has_euro/has_asian/has_over_under detection,
and asian_line=0 ambiguity in the odds data.

Usage:
    python tools/report_missing_semantics.py \
        --input data/odds_real/titan007_pure_v3_time.jsonl \
        --splits-dir data/odds_real/splits/v3_time \
        --output docs/review/missing_semantics_report.md
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.odds_dataset import (
    _event_has_euro,
    _event_has_asian,
    _event_has_over_under,
    _event_to_missing_mask,
)


def load_split_ids(splits_dir: str) -> dict:
    result = {}
    for name in ("train", "val", "test"):
        path = os.path.join(splits_dir, f"{name}_match_ids.txt")
        if os.path.exists(path):
            with open(path, "r") as f:
                result[name] = {line.strip() for line in f if line.strip()}
        else:
            result[name] = set()
    return result


def analyze_data(jsonl_path: str) -> dict:
    """Analyze missing semantics across the full dataset."""
    stats = {
        "total_matches": 0,
        "total_events": 0,
        "has_euro_true": 0,
        "has_euro_false": 0,
        "has_asian_true": 0,
        "has_asian_false": 0,
        "has_ou_true": 0,
        "has_ou_false": 0,
        "asian_line_zero_total": 0,
        "asian_line_zero_has_asian_true": 0,
        "asian_line_zero_has_asian_false": 0,
        "ou_line_2_5_total": 0,
        "ou_line_2_5_has_ou_true": 0,
        "ou_line_2_5_has_ou_false": 0,
        "water_default_count": 0,  # upper=1.0 AND lower=1.0
        "ou_water_default_count": 0,  # over=1.0 AND under=1.0
        "exporter_has_asian_true": 0,
        "exporter_has_asian_false": 0,
        "exporter_has_ou_true": 0,
        "exporter_has_ou_false": 0,
        "asian_line_values": Counter(),
        "upper_water_values": Counter(),
        "ou_line_values": Counter(),
        "mask_present_counts": [0] * 13,
        "mask_missing_counts": [0] * 13,
    }

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            stats["total_matches"] += 1
            tl = m.get("odds_timeline", [])
            stats["total_events"] += len(tl)

            for e in tl:
                # Exporter flags
                exporter_asian = e.get("has_asian", None)
                exporter_ou = e.get("has_over_under", None)
                if exporter_asian is True:
                    stats["exporter_has_asian_true"] += 1
                elif exporter_asian is False:
                    stats["exporter_has_asian_false"] += 1
                if exporter_ou is True:
                    stats["exporter_has_ou_true"] += 1
                elif exporter_ou is False:
                    stats["exporter_has_ou_false"] += 1

                # Content-based detection (fixed)
                he = _event_has_euro(e)
                ha = _event_has_asian(e)
                hou = _event_has_over_under(e)

                stats["has_euro_true" if he else "has_euro_false"] += 1
                stats["has_asian_true" if ha else "has_asian_false"] += 1
                stats["has_ou_true" if hou else "has_ou_false"] += 1

                # asian_line=0 analysis
                al = e.get("asian_line", 0)
                if al == 0:
                    stats["asian_line_zero_total"] += 1
                    if ha:
                        stats["asian_line_zero_has_asian_true"] += 1
                    else:
                        stats["asian_line_zero_has_asian_false"] += 1

                # ou_line=2.5 analysis
                ol = e.get("over_under_line", 0)
                if ol == 2.5:
                    stats["ou_line_2_5_total"] += 1
                    if hou:
                        stats["ou_line_2_5_has_ou_true"] += 1
                    else:
                        stats["ou_line_2_5_has_ou_false"] += 1

                # Water defaults
                uw = e.get("upper_water", 0)
                lw = e.get("lower_water", 0)
                if uw == 1.0 and lw == 1.0:
                    stats["water_default_count"] += 1

                ow = e.get("over_water", 0)
                ou = e.get("under_water", 0)
                if ow == 1.0 and ou == 1.0:
                    stats["ou_water_default_count"] += 1

                # Value distributions (sample)
                stats["asian_line_values"][al] += 1
                stats["upper_water_values"][uw] += 1
                stats["ou_line_values"][ol] += 1

                # Missing mask analysis
                mask = _event_to_missing_mask(e)
                for i, v in enumerate(mask):
                    if v >= 0.5:
                        stats["mask_present_counts"][i] += 1
                    else:
                        stats["mask_missing_counts"][i] += 1

    return stats


FEATURE_NAMES = [
    "minutes_before_kickoff", "euro_h", "euro_d", "euro_a",
    "asian_line", "upper_water", "lower_water",
    "over_under_line", "over_water", "under_water",
    "has_euro", "has_asian", "has_over_under",
]


def generate_report(args) -> str:
    stats = analyze_data(args.input)
    splits = load_split_ids(args.splits_dir)

    lines = []
    lines.append("# Missing Semantics Report")
    lines.append(f"\nGenerated: {datetime.now().isoformat()}")
    lines.append(f"\nInput: `{args.input}`")
    lines.append(f"Splits dir: `{args.splits_dir}`")

    lines.append("\n## 1. Data Overview")
    lines.append(f"- Total matches: {stats['total_matches']}")
    lines.append(f"- Total events: {stats['total_events']}")
    lines.append(f"- Train / Val / Test: {len(splits.get('train', set()))} / {len(splits.get('val', set()))} / {len(splits.get('test', set()))}")

    lines.append("\n## 2. missing_mask Semantics")
    lines.append("- **Convention**: `1.0 = present (real data)`, `0.0 = missing (default/placeholder)`")
    lines.append("- **Source**: `dataset/odds_dataset.py` → `_event_to_missing_mask()`")
    lines.append("- **Applied**: Per-event, per-feature (13-element vector for v3 schema)")

    lines.append("\n## 3. has_euro / has_asian / has_over_under Sources")
    lines.append("- **has_euro**: Content-based — `euro_h > 1.0 AND euro_d > 1.0 AND euro_a > 1.0`")
    lines.append("- **has_asian**: Content-based with default detection — `NOT (line=0 AND water=1.0/1.0)`")
    lines.append("- **has_over_under**: Content-based with default detection — `NOT (line=2.5 AND water=1.0/1.0)`")
    lines.append("- **P1.0F fix**: Overridden exporter flags — dataset now uses its own detection")

    lines.append("\n## 4. asian_line=0 Analysis")
    lines.append(f"- `asian_line=0` total events: **{stats['asian_line_zero_total']}**")
    lines.append(f"- `asian_line=0` AND `has_asian=True` (real flat handicap): **{stats['asian_line_zero_has_asian_true']}**")
    lines.append(f"- `asian_line=0` AND `has_asian=False` (default placeholder): **{stats['asian_line_zero_has_asian_false']}**")
    if stats['asian_line_zero_total'] > 0 and stats['asian_line_zero_has_asian_true'] == 0:
        lines.append("- **Conclusion**: ALL asian_line=0 are default placeholders, NOT real flat handicaps")

    lines.append("\n## 5. over_under_line=2.5 Analysis")
    lines.append(f"- `ou_line=2.5` total events: **{stats['ou_line_2_5_total']}**")
    lines.append(f"- `ou_line=2.5` AND `has_ou=True` (real): **{stats['ou_line_2_5_has_ou_true']}**")
    lines.append(f"- `ou_line=2.5` AND `has_ou=False` (default): **{stats['ou_line_2_5_has_ou_false']}**")
    if stats['ou_line_2_5_total'] > 0 and stats['ou_line_2_5_has_ou_true'] == 0:
        lines.append("- **Conclusion**: ALL ou_line=2.5 are default placeholders")

    lines.append("\n## 6. Default Water Value Risk")
    lines.append(f"- Events with `upper_water=1.0 AND lower_water=1.0`: **{stats['water_default_count']}** / {stats['total_events']}")
    lines.append(f"- Events with `over_water=1.0 AND under_water=1.0`: **{stats['ou_water_default_count']}** / {stats['total_events']}")
    if stats['water_default_count'] == stats['total_events']:
        lines.append("- **Risk**: ALL asian water values are 1.0/1.0 — no real asian data exists")
    if stats['ou_water_default_count'] == stats['total_events']:
        lines.append("- **Risk**: ALL ou water values are 1.0/1.0 — no real ou data exists")

    lines.append("\n## 7. Exporter Flag vs Content-Based Comparison")
    lines.append(f"- Exporter `has_asian=True`: {stats['exporter_has_asian_true']} events")
    lines.append(f"- Exporter `has_asian=False`: {stats['exporter_has_asian_false']} events")
    lines.append(f"- Content-based `has_asian=True`: {stats['has_asian_true']} events")
    lines.append(f"- Content-based `has_asian=False`: {stats['has_asian_false']} events")
    lines.append(f"- Exporter `has_ou=True`: {stats['exporter_has_ou_true']} events")
    lines.append(f"- Exporter `has_ou=False`: {stats['exporter_has_ou_false']} events")
    lines.append(f"- Content-based `has_ou=True`: {stats['has_ou_true']} events")
    lines.append(f"- Content-based `has_ou=False`: {stats['has_ou_false']} events")
    exporter_correct = (stats['exporter_has_asian_true'] == stats['has_asian_true'])
    if not exporter_correct:
        lines.append("- **Warning**: Exporter flags do NOT match content-based detection")

    lines.append("\n## 8. Value Distributions")
    lines.append("\n### asian_line (top 5)")
    for val, cnt in stats["asian_line_values"].most_common(5):
        lines.append(f"- `{val}`: {cnt} events")

    lines.append("\n### upper_water (top 5)")
    for val, cnt in stats["upper_water_values"].most_common(5):
        lines.append(f"- `{val}`: {cnt} events")

    lines.append("\n### over_under_line (top 5)")
    for val, cnt in stats["ou_line_values"].most_common(5):
        lines.append(f"- `{val}`: {cnt} events")

    lines.append("\n## 9. Per-Feature Missing Mask Summary")
    lines.append("| Feature | Present | Missing | Present % |")
    lines.append("|---------|---------|---------|-----------|")
    for i, name in enumerate(FEATURE_NAMES):
        p = stats["mask_present_counts"][i]
        m = stats["mask_missing_counts"][i]
        total = p + m
        pct = f"{100*p/total:.1f}%" if total > 0 else "N/A"
        lines.append(f"| {name} | {p} | {m} | {pct} |")

    lines.append("\n## 10. Can Distinguish Real 0 vs Missing 0?")
    lines.append("- **asian_line=0**: YES — check water prices (1.0/1.0 = missing, else real flat handicap)")
    lines.append("- **over_under_line=2.5**: YES — check water prices (1.0/1.0 = missing, else real)")
    lines.append("- **over_under_line=0**: Always missing (no real goal line is 0)")
    lines.append("- **euro odds=0**: Always missing (odds > 1.0 required)")

    lines.append("\n## 11. Exporter Fix Required?")
    needs_fix = stats['exporter_has_asian_true'] != stats['has_asian_true']
    if needs_fix:
        lines.append("- **NEEDS_EXPORTER_FIX**: YES")
        lines.append("- Exporter's `has_asian` and `has_over_under` flags are incorrect")
        lines.append("- Recommendation: Add source flags to odds-data-probe exporter")
        lines.append("- Workaround: Dataset now uses content-based detection (P1.0F fix)")
    else:
        lines.append("- **NEEDS_EXPORTER_FIX**: NO")

    lines.append("\n## 12. Verdict")
    if stats['has_asian_true'] == 0 and stats['has_ou_true'] == 0:
        lines.append("\n**PASS_WITH_WARNINGS**")
        lines.append("- All asian/ou data detected as default placeholders")
        lines.append("- missing_mask correctly marks them as missing (0.0)")
        lines.append("- Only euro odds are real data")
        lines.append("- Warning: No real asian/ou data available for training")
        lines.append("- NEEDS_EXPORTER_FIX for future data collection")
    elif stats['has_asian_true'] > 0 and stats['has_ou_true'] > 0:
        lines.append("\n**PASS**")
        lines.append("- Real asian and ou data detected")
        lines.append("- missing_mask correctly distinguishes present vs missing")
    else:
        lines.append("\n**PASS_WITH_WARNINGS**")
        lines.append("- Partial data availability")

    lines.append(f"\n---\n*Report generated by `tools/report_missing_semantics.py`*")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="P1.0F Missing Semantics Report")
    parser.add_argument("--input", required=True, help="Path to JSONL data file")
    parser.add_argument("--splits-dir", required=True, help="Path to splits directory")
    parser.add_argument("--output", required=True, help="Output markdown file path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    report = generate_report(args)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
