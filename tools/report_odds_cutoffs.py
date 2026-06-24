"""
Generate cutoff availability report for OddsMind training data.

Read-only script — does not modify any files.

Usage:
    python tools/report_odds_cutoffs.py \
        --input data/odds_real/titan007_pure_v3_time.jsonl \
        --splits-dir data/odds_real/splits/v3_time \
        --output docs/review/cutoff_report.md \
        --cutoff-buckets "1440,720,360,180,120,60,30,15,5,1"
"""
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.odds_cutoff import filter_timeline_by_cutoff, assert_cutoff_integrity


def load_ids(path):
    with open(path) as f:
        return {l.strip() for l in f if l.strip()}


def compute_cutoff_stats(samples, cutoff):
    """Compute stats for a single cutoff."""
    non_empty = 0
    vis_counts = []
    for s in samples:
        vis = filter_timeline_by_cutoff(s['odds_timeline'], cutoff)
        if vis:
            non_empty += 1
            vis_counts.append(len(vis))
    avg = sum(vis_counts) / len(vis_counts) if vis_counts else 0
    vis_counts.sort()
    n = len(vis_counts)
    med = vis_counts[n // 2] if vis_counts else 0
    p25 = vis_counts[n // 4] if vis_counts else 0
    p75 = vis_counts[3 * n // 4] if vis_counts else 0
    return {"non_empty": non_empty, "avg": avg, "median": med, "p25": p25, "p75": p75}


def check_cutoff_integrity(samples, cutoff_buckets):
    """Check for cutoff violations."""
    violations = 0
    for s in samples:
        for cutoff in cutoff_buckets:
            vis = filter_timeline_by_cutoff(s['odds_timeline'], cutoff)
            for e in vis:
                if e['minutes_before_kickoff'] < cutoff:
                    violations += 1
    return violations


def main():
    p = argparse.ArgumentParser(description="Generate cutoff availability report")
    p.add_argument("--input", required=True, help="Path to training JSONL")
    p.add_argument("--splits-dir", required=True, help="Path to splits directory")
    p.add_argument("--output", required=True, help="Path to output report")
    p.add_argument("--cutoff-buckets", default="1440,720,360,180,120,60,30,15,5,1")
    args = p.parse_args()

    cutoff_buckets = [int(c.strip()) for c in args.cutoff_buckets.split(",") if c.strip()]

    # Load data
    with open(args.input, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f]

    # Load splits
    splits_dir = Path(args.splits_dir)
    train_ids = load_ids(splits_dir / "train_match_ids.txt")
    val_ids = load_ids(splits_dir / "val_match_ids.txt")
    test_ids = load_ids(splits_dir / "test_match_ids.txt")

    split_samples = {
        "train": [s for s in samples if s["match_id"] in train_ids],
        "val": [s for s in samples if s["match_id"] in val_ids],
        "test": [s for s in samples if s["match_id"] in test_ids],
    }

    # Generate report
    lines = []
    lines.append("# Cutoff Report — P1.0D")
    lines.append("")
    lines.append(f"> Generated: {datetime.now().isoformat()}")
    lines.append(f"> Input: `{args.input}`")
    lines.append(f"> Cutoff buckets: {cutoff_buckets}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Summary
    lines.append("## 1. Summary")
    lines.append("")
    lines.append("| Split | Match IDs | Samples |")
    lines.append("|-------|-----------|---------|")
    for name in ["train", "val", "test"]:
        ids = train_ids if name == "train" else (val_ids if name == "val" else test_ids)
        lines.append(f"| {name} | {len(ids)} | {len(split_samples[name])} |")
    lines.append("")

    # Cutoff bucket stats for each split
    for split_name in ["train", "val", "test"]:
        split = split_samples[split_name]
        lines.append(f"## 2. {split_name.capitalize()} Cutoff Availability")
        lines.append("")
        lines.append("| Cutoff | Non-empty | Pct | Avg vis | Median vis | P25 | P75 |")
        lines.append("|--------|-----------|-----|---------|------------|-----|-----|")
        for cutoff in cutoff_buckets:
            stats = compute_cutoff_stats(split, cutoff)
            pct = stats["non_empty"] / len(split) * 100 if split else 0
            lines.append(f"| {cutoff} | {stats['non_empty']} | {pct:.1f}% | {stats['avg']:.1f} | {stats['median']} | {stats['p25']} | {stats['p75']} |")
        lines.append("")

    # T-30 to T-1 detail for train
    lines.append("## 3. Train T-30 to T-1 Detail")
    lines.append("")
    lines.append("| Cutoff | Non-empty | Pct | Avg vis | Median vis |")
    lines.append("|--------|-----------|-----|---------|------------|")
    for cutoff in range(30, 0, -1):
        stats = compute_cutoff_stats(split_samples["train"], cutoff)
        pct = stats["non_empty"] / len(split_samples["train"]) * 100 if split_samples["train"] else 0
        lines.append(f"| {cutoff} | {stats['non_empty']} | {pct:.1f}% | {stats['avg']:.1f} | {stats['median']} |")
    lines.append("")

    # Integrity check
    lines.append("## 4. Cutoff Integrity Check")
    lines.append("")
    violations = check_cutoff_integrity(split_samples["train"], cutoff_buckets)
    lines.append(f"- Train violations: {violations} {'PASS' if violations == 0 else 'FAIL'}")
    violations = check_cutoff_integrity(split_samples["val"], cutoff_buckets)
    lines.append(f"- Val violations: {violations} {'PASS' if violations == 0 else 'FAIL'}")
    violations = check_cutoff_integrity(split_samples["test"], cutoff_buckets)
    lines.append(f"- Test violations: {violations} {'PASS' if violations == 0 else 'FAIL'}")
    lines.append("")

    # Overlap check
    lines.append("## 5. Split Overlap Check")
    lines.append("")
    tv = train_ids & val_ids
    tt = train_ids & test_ids
    vt = val_ids & test_ids
    lines.append(f"- train ∩ val: {len(tv)} {'PASS' if len(tv) == 0 else 'FAIL'}")
    lines.append(f"- train ∩ test: {len(tt)} {'PASS' if len(tt) == 0 else 'FAIL'}")
    lines.append(f"- val ∩ test: {len(vt)} {'PASS' if len(vt) == 0 else 'FAIL'}")
    lines.append("")

    # Write
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
