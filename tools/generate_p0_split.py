"""
P0 Split Generator — time-based grouped split for large dataset.

Reads pipeline_export.jsonl, generates train/val/test split files
and a split summary JSON. Grouped by match_id to prevent leakage.

Usage:
    python tools/generate_p0_split.py \
        --data data/training/pipeline_export.jsonl \
        --out-dir data/training/splits_p0 \
        --train-ratio 0.70 --val-ratio 0.15 --seed 42
"""

import argparse, json, os, random, sys
from collections import defaultdict
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="P0 Split Generator")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.data):
        print(f"ERROR: {args.data} not found")
        sys.exit(1)

    random.seed(args.seed)

    # Load all matches
    matches = []
    errors = 0
    with open(args.data, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip(): continue
            try:
                m = json.loads(line)
                m["_kickoff_dt"] = m["kickoff_time"].replace("Z", "+00:00")
                matches.append(m)
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  WARNING: line {i} parse error: {e}")
    print(f"Loaded {len(matches)} matches ({errors} errors)")

    # Sort by kickoff time
    matches.sort(key=lambda m: m["_kickoff_dt"])

    # Group by match_id (though each should be unique in this dataset)
    n = len(matches)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    train_matches = matches[:n_train]
    val_matches = matches[n_train:n_train + n_val]
    test_matches = matches[n_train + n_val:]

    # Extract match_ids (unique per group)
    train_ids = set(m["match_id"] for m in train_matches)
    val_ids = set(m["match_id"] for m in val_matches)
    test_ids = set(m["match_id"] for m in test_matches)

    # Verify no overlap
    assert train_ids.isdisjoint(val_ids), "Train/Val overlap"
    assert train_ids.isdisjoint(test_ids), "Train/Test overlap"
    assert val_ids.isdisjoint(test_ids), "Val/Test overlap"
    print("Split integrity: PASS (no overlap)")

    # Write split files
    def write_ids(path, ids):
        with open(path, "w") as f:
            for mid in sorted(ids):
                f.write(mid + "\n")

    write_ids(os.path.join(args.out_dir, "train_match_ids.txt"), train_ids)
    write_ids(os.path.join(args.out_dir, "val_match_ids.txt"), val_ids)
    write_ids(os.path.join(args.out_dir, "test_match_ids.txt"), test_ids)

    # League distribution
    def league_dist(match_list):
        dist = defaultdict(int)
        for m in match_list:
            dist[m.get("league_id", "?")] += 1
        return dict(sorted(dist.items(), key=lambda x: -x[1]))

    # Time ranges
    def time_range(match_list):
        times = sorted(m["_kickoff_dt"] for m in match_list)
        return times[0][:19] if times else "N/A", times[-1][:19] if times else "N/A"

    train_tr = time_range(train_matches)
    val_tr = time_range(val_matches)
    test_tr = time_range(test_matches)

    summary = {
        "total_matches": n,
        "train_matches": len(train_ids),
        "val_matches": len(val_ids),
        "test_matches": len(test_ids),
        "train_time_range": list(train_tr),
        "val_time_range": list(val_tr),
        "test_time_range": list(test_tr),
        "train_league_distribution": league_dist(train_matches),
        "val_league_distribution": league_dist(val_matches),
        "test_league_distribution": league_dist(test_matches),
        "overlap_check": "pass",
        "seed": args.seed,
    }

    with open(os.path.join(args.out_dir, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSplit summary:")
    print(f"  Train: {len(train_ids)} matches, {train_tr[0]} -> {train_tr[1]}")
    print(f"  Val:   {len(val_ids)} matches, {val_tr[0]} -> {val_tr[1]}")
    print(f"  Test:  {len(test_ids)} matches, {test_tr[0]} -> {test_tr[1]}")
    print(f"  Top leagues (train): {list(league_dist(train_matches).items())[:8]}")
    print(f"\nSaved to {args.out_dir}/")


if __name__ == "__main__":
    main()
