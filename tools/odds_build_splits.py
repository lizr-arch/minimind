"""
OddsMind Split Builder (P0.4)

Builds train / val / test split files from a JSONL match file.

Outputs:
    train_match_ids.txt   — one match_id per line
    val_match_ids.txt
    test_match_ids.txt
    split_summary.json    — metadata and time ranges

Usage:
    python tools/odds_build_splits.py \
        --data data/odds_fixtures/sample_odds_matches_5class.jsonl \
        --out-dir data/odds_fixtures/splits_p0_4 \
        --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.odds_split import (
    load_match_metadata,
    grouped_time_split,
    get_time_range,
)


def main():
    parser = argparse.ArgumentParser(description="OddsMind Split Builder")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to JSONL match file")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Output directory for split files")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    # Load & split
    matches = load_match_metadata(args.data)
    splits = grouped_time_split(
        matches,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # Write match ID files
    for name in ("train", "val", "test"):
        path = os.path.join(args.out_dir, f"{name}_match_ids.txt")
        ids = sorted(splits[name])
        with open(path, "w", encoding="utf-8") as f:
            for mid in ids:
                f.write(mid + "\n")
        print(f"  {name}: {len(ids)} matches → {path}")

    # Write summary JSON
    summary = {
        "total_matches": len(matches),
        "train_matches": len(splits["train"]),
        "val_matches": len(splits["val"]),
        "test_matches": len(splits["test"]),
        "train_time_range": list(get_time_range(matches, splits["train"])),
        "val_time_range": list(get_time_range(matches, splits["val"])),
        "test_time_range": list(get_time_range(matches, splits["test"])),
        "overlap_check": (
            "pass"
            if (splits["train"].isdisjoint(splits["val"])
                and splits["train"].isdisjoint(splits["test"])
                and splits["val"].isdisjoint(splits["test"]))
            else "FAIL"
        ),
    }

    summary_path = os.path.join(args.out_dir, "split_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary → {summary_path}")

    print("Done.")


if __name__ == "__main__":
    main()
