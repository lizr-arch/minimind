"""
Bucket Occupancy Report: 验证 time bucketing 是否真实有效。

检查事件在 10 个时间桶中的分布，检测 closing-only 退化和 minutes_before_kickoff=0 问题。

Usage:
    python tools/p3_bucket_occupancy_report.py \
        --data data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl \
        --ids data/odds_real/splits_v6/train_match_ids.txt \
        --out runs/p3_bucket_occupancy_report.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from statistics import mean, median

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BUCKET_EDGES = [
    (float('inf'), 168, 'opening'),
    (168, 72, '7d'), (72, 24, '3d'), (24, 12, '1d'),
    (12, 6, '12h'), (6, 3, '6h'), (3, 1, '3h'),
    (1, 0.5, '1h'), (0.5, 0, '30m'), (0, -float('inf'), 'closing'),
]
BUCKET_NAMES = [name for _, _, name in BUCKET_EDGES]


def load_match_ids(ids_path):
    with open(ids_path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def classify_bucket(minutes_before_kickoff):
    hours = minutes_before_kickoff / 60.0
    for hi, lo, name in BUCKET_EDGES:
        if hours <= hi and hours > lo:
            return name
    return 'closing'


def analyze(args):
    match_ids = load_match_ids(args.ids)
    data_by_id = {}
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            mid = row.get("match_id")
            if mid in match_ids:
                data_by_id[mid] = row

    bucket_event_counts = {name: 0 for name in BUCKET_NAMES}
    bucket_match_coverage = {name: 0 for name in BUCKET_NAMES}
    non_empty_per_match = []
    num_events = 0
    closing_events = 0
    minutes_zero_events = 0

    for mid, row in data_by_id.items():
        timeline = row.get("raw_timeline", []) or row.get("odds_timeline", [])
        match_buckets = set()

        for event in timeline:
            minutes = event.get("minutes_before_kickoff")
            if minutes is None:
                continue
            num_events += 1
            bucket = classify_bucket(minutes)
            bucket_event_counts[bucket] += 1
            match_buckets.add(bucket)
            if bucket == 'closing':
                closing_events += 1
            if minutes == 0:
                minutes_zero_events += 1

        for b in match_buckets:
            bucket_match_coverage[b] += 1
        non_empty_per_match.append(len(match_buckets))

    num_matches = len(data_by_id)
    closing_ratio = closing_events / num_events if num_events > 0 else 0.0
    minutes_zero_ratio = minutes_zero_events / num_events if num_events > 0 else 0.0
    avg_non_empty = mean(non_empty_per_match) if non_empty_per_match else 0.0
    median_non_empty = median(non_empty_per_match) if non_empty_per_match else 0.0

    warnings = []
    if closing_ratio > 0.8:
        warnings.append(f"WARN: time bucketing likely collapsed to closing (closing_event_ratio={closing_ratio:.3f})")
    if avg_non_empty < 2:
        warnings.append(f"WARN: insufficient temporal coverage (avg_non_empty_buckets={avg_non_empty:.1f})")
    if minutes_zero_ratio > 0.9:
        warnings.append(f"BLOCKER: timeline not usable for patching — minutes_before_kickoff mostly zero ({minutes_zero_ratio:.1%})")

    report = {
        "num_matches": num_matches,
        "num_events": num_events,
        "bucket_names": BUCKET_NAMES,
        "bucket_event_counts": bucket_event_counts,
        "bucket_match_coverage": bucket_match_coverage,
        "avg_non_empty_buckets_per_match": round(avg_non_empty, 2),
        "median_non_empty_buckets_per_match": round(median_non_empty, 2),
        "closing_event_ratio": round(closing_ratio, 6),
        "all_events_minutes_zero_ratio": round(minutes_zero_ratio, 6),
        "warnings": warnings,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"=== Bucket Occupancy Report ===")
    print(f"Matches: {num_matches}")
    print(f"Events: {num_events}")
    print(f"Closing event ratio: {closing_ratio:.3f}")
    print(f"Avg non-empty buckets: {avg_non_empty:.1f}")
    print(f"Median non-empty buckets: {median_non_empty:.1f}")
    print(f"Minutes-zero ratio: {minutes_zero_ratio:.1%}")
    print()
    print("Bucket distribution:")
    for name in BUCKET_NAMES:
        count = bucket_event_counts[name]
        coverage = bucket_match_coverage[name]
        print(f"  {name:>10}: {count:>6} events, {coverage:>4} matches")
    if warnings:
        print()
        print("Warnings:")
        for w in warnings:
            print(f"  {w}")
    print()
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bucket Occupancy Report")
    parser.add_argument("--data", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    analyze(args)
