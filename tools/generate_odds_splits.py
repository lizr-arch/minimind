"""
Generate train/val/test split files for OddsMind training data.

Splits by match_id grouped by kickoff_time (time-based, not random).
All samples from the same match_id stay in the same split.

Usage:
    python tools/generate_odds_splits.py \
        --input data/odds_real/titan007_pure_v3_time.jsonl \
        --output-dir data/odds_real/splits/v3_time \
        --report docs/review/split_report.md
"""
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Set


def load_and_validate(input_path: str) -> tuple:
    """Load samples, group by match_id, validate consistency.
    
    Returns (samples, grouped, invalid_reasons).
    grouped: dict of match_id -> {kickoff_time, samples, league_id}
    """
    with open(input_path, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f]

    grouped = defaultdict(lambda: {"kickoff_times": set(), "samples": [], "league_ids": set()})
    invalid_reasons = []

    for s in samples:
        mid = s["match_id"]
        ko = s.get("kickoff_time", "")
        status = s.get("time_axis_status", "")
        league = s.get("league_id", "")

        g = grouped[mid]
        g["samples"].append(s)
        g["league_ids"].add(league)
        if ko:
            g["kickoff_times"].add(ko)

    # Validate: no multiple kickoff_times per match_id
    for mid, g in grouped.items():
        if len(g["kickoff_times"]) > 1:
            invalid_reasons.append(f"match_id={mid} has {len(g['kickoff_times'])} kickoff_times: {g['kickoff_times']}")
        if not g["kickoff_times"]:
            invalid_reasons.append(f"match_id={mid} has no kickoff_time")

    return samples, dict(grouped), invalid_reasons


def generate_splits(grouped: dict, train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, Set[str]]:
    """Split match_ids by kickoff_time ascending."""
    # Get match_id -> kickoff_time mapping (pick first if multiple, but we validated above)
    match_ko = {}
    for mid, g in grouped.items():
        if g["kickoff_times"]:
            match_ko[mid] = min(g["kickoff_times"])  # should be only one

    # Sort by kickoff_time ascending
    sorted_matches = sorted(match_ko.items(), key=lambda x: x[1])
    n = len(sorted_matches)

    if n == 0:
        return {"train": set(), "val": set(), "test": set()}

    n_train = max(1, round(n * train_ratio))
    n_val = max(1, round(n * val_ratio))
    n_test = n - n_train - n_val

    if n_test < 1:
        if n_val > 1:
            n_val -= 1
            n_test = 1
        elif n_train > 1:
            n_train -= 1
            n_test = 1

    train_ids = {m[0] for m in sorted_matches[:n_train]}
    val_ids = {m[0] for m in sorted_matches[n_train:n_train + n_val]}
    test_ids = {m[0] for m in sorted_matches[n_train + n_val:]}

    assert train_ids.isdisjoint(val_ids), "train/val overlap"
    assert train_ids.isdisjoint(test_ids), "train/test overlap"
    assert val_ids.isdisjoint(test_ids), "val/test overlap"

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def get_split_stats(samples: list, match_ids: Set[str]) -> dict:
    """Compute statistics for a split."""
    split_samples = [s for s in samples if s["match_id"] in match_ids]
    if not split_samples:
        return {"sample_count": 0}

    kos = [s["kickoff_time"] for s in split_samples if s.get("kickoff_time")]
    tl_lens = [len(s.get("odds_timeline", [])) for s in split_samples]
    euro_labels = Counter(s["label"]["euro_result"] for s in split_samples)
    asian_labels = Counter(s["label"]["asian_result"] for s in split_samples)
    leagues = Counter(s.get("league_id", "?") for s in split_samples)
    bookmakers = Counter(s.get("bookmaker_id", "?") for s in split_samples)

    tl_lens.sort()
    n = len(tl_lens)

    return {
        "match_count": len(match_ids),
        "sample_count": len(split_samples),
        "time_range": (min(kos), max(kos)) if kos else ("N/A", "N/A"),
        "timeline_len": {
            "min": tl_lens[0] if tl_lens else 0,
            "p25": tl_lens[n // 4] if tl_lens else 0,
            "median": tl_lens[n // 2] if tl_lens else 0,
            "p75": tl_lens[3 * n // 4] if tl_lens else 0,
            "max": tl_lens[-1] if tl_lens else 0,
        },
        "euro_labels": dict(euro_labels),
        "asian_labels": dict(asian_labels),
        "leagues": dict(leagues),
        "bookmakers": dict(bookmakers),
    }


def generate_report(output_path: str, input_path: str, samples: list, grouped: dict,
                    splits: Dict[str, Set[str]], invalid_reasons: list):
    """Generate split_report.md."""
    lines = []
    lines.append("# Split Report — P1.0C")
    lines.append("")
    lines.append(f"> Generated: {datetime.now().isoformat()}")
    lines.append(f"> Input: `{input_path}`")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Summary
    lines.append("## 1. Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total samples | {len(samples)} |")
    lines.append(f"| Unique match_ids | {len(grouped)} |")
    lines.append(f"| Invalid match_ids | {len(invalid_reasons)} |")
    lines.append("")

    # Split counts
    lines.append("## 2. Split Counts")
    lines.append("")
    lines.append("| Split | Match IDs | Samples |")
    lines.append("|-------|-----------|---------|")
    for split_name in ["train", "val", "test"]:
        ids = splits[split_name]
        stats = get_split_stats(samples, ids)
        lines.append(f"| {split_name} | {stats['match_count']} | {stats['sample_count']} |")
    lines.append("")

    # Time ranges
    lines.append("## 3. Time Ranges")
    lines.append("")
    lines.append("| Split | Min kickoff | Max kickoff |")
    lines.append("|-------|-------------|-------------|")
    for split_name in ["train", "val", "test"]:
        stats = get_split_stats(samples, splits[split_name])
        tmin, tmax = stats["time_range"]
        lines.append(f"| {split_name} | {tmin} | {tmax} |")
    lines.append("")

    # Overlap check
    lines.append("## 4. Overlap Check")
    lines.append("")
    train, val, test = splits["train"], splits["val"], splits["test"]
    tv = train & val
    tt = train & test
    vt = val & test
    lines.append(f"- train ∩ val: {len(tv)} {'PASS' if len(tv) == 0 else 'FAIL'}")
    lines.append(f"- train ∩ test: {len(tt)} {'PASS' if len(tt) == 0 else 'FAIL'}")
    lines.append(f"- val ∩ test: {len(vt)} {'PASS' if len(vt) == 0 else 'FAIL'}")
    lines.append("")

    # Per-split details
    for split_name in ["train", "val", "test"]:
        stats = get_split_stats(samples, splits[split_name])
        lines.append(f"## 5. {split_name.capitalize()} Distribution")
        lines.append("")

        lines.append("### Euro Labels")
        lines.append("")
        lines.append("| Label | Count | Pct |")
        lines.append("|-------|-------|-----|")
        total = stats["sample_count"]
        for label in ["home", "draw", "away"]:
            cnt = stats["euro_labels"].get(label, 0)
            pct = cnt / total * 100 if total else 0
            lines.append(f"| {label} | {cnt} | {pct:.1f}% |")
        lines.append("")

        lines.append("### Asian Labels")
        lines.append("")
        lines.append("| Label | Count | Pct |")
        lines.append("|-------|-------|-----|")
        for label in ["full_win", "full_loss", "push", "half_win", "half_loss"]:
            cnt = stats["asian_labels"].get(label, 0)
            pct = cnt / total * 100 if total else 0
            lines.append(f"| {label} | {cnt} | {pct:.1f}% |")
        lines.append("")

        lines.append("### Leagues")
        lines.append("")
        lines.append("| League | Count |")
        lines.append("|--------|-------|")
        for league, cnt in sorted(stats["leagues"].items(), key=lambda x: -x[1]):
            lines.append(f"| {league} | {cnt} |")
        lines.append("")

        lines.append("### Bookmakers")
        lines.append("")
        lines.append("| Bookmaker | Count |")
        lines.append("|-----------|-------|")
        for bm, cnt in sorted(stats["bookmakers"].items(), key=lambda x: -x[1]):
            lines.append(f"| {bm} | {cnt} |")
        lines.append("")

        lines.append("### Timeline Length")
        lines.append("")
        lines.append(f"min={stats['timeline_len']['min']}, p25={stats['timeline_len']['p25']}, "
                     f"median={stats['timeline_len']['median']}, p75={stats['timeline_len']['p75']}, "
                     f"max={stats['timeline_len']['max']}")
        lines.append("")

    # Multi-bookmaker leak check
    lines.append("## 6. Multi-Bookmaker Leak Check")
    lines.append("")
    multi_bm = {mid: g for mid, g in grouped.items() if len(g["samples"]) > 1}
    leak_count = 0
    for mid, g in multi_bm.items():
        sample_splits = set()
        for split_name, ids in splits.items():
            if mid in ids:
                sample_splits.add(split_name)
        if len(sample_splits) > 1:
            leak_count += 1
    lines.append(f"- Matches with multiple bookmaker samples: {len(multi_bm)}")
    lines.append(f"- Matches leaking across splits: {leak_count} {'PASS' if leak_count == 0 else 'FAIL'}")
    lines.append("")

    # Invalid reasons
    if invalid_reasons:
        lines.append("## 7. Invalid Match IDs")
        lines.append("")
        for reason in invalid_reasons[:20]:
            lines.append(f"- {reason}")
        lines.append("")

    # Write
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written to {out_path}")


def main():
    p = argparse.ArgumentParser(description="Generate train/val/test splits")
    p.add_argument("--input", required=True, help="Path to training JSONL")
    p.add_argument("--output-dir", required=True, help="Output directory for split files")
    p.add_argument("--report", default="", help="Path to split report")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--group-key", default="match_id")
    p.add_argument("--sort-key", default="kickoff_time")
    args = p.parse_args()

    # Load and validate
    print(f"Loading {args.input}...")
    samples, grouped, invalid_reasons = load_and_validate(args.input)
    print(f"  Total samples: {len(samples)}")
    print(f"  Unique match_ids: {len(grouped)}")
    print(f"  Invalid: {len(invalid_reasons)}")

    # Generate splits
    splits = generate_splits(grouped, args.train_ratio, args.val_ratio, args.test_ratio)
    print(f"  Train: {len(splits['train'])} match_ids")
    print(f"  Val: {len(splits['val'])} match_ids")
    print(f"  Test: {len(splits['test'])} match_ids")

    # Write split files
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        path = out_dir / f"{split_name}_match_ids.txt"
        with open(path, "w") as f:
            for mid in sorted(splits[split_name]):
                f.write(mid + "\n")
        print(f"  Written {path}")

    # Write manifest
    manifest = {
        "input": args.input,
        "generated_at": datetime.now().isoformat(),
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "group_key": args.group_key,
        "sort_key": args.sort_key,
        "total_samples": len(samples),
        "unique_match_ids": len(grouped),
        "invalid_match_ids": len(invalid_reasons),
        "train_match_ids": len(splits["train"]),
        "val_match_ids": len(splits["val"]),
        "test_match_ids": len(splits["test"]),
    }
    manifest_path = out_dir / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Written {manifest_path}")

    # Generate report
    if args.report:
        generate_report(args.report, args.input, samples, grouped, splits, invalid_reasons)


if __name__ == "__main__":
    main()
