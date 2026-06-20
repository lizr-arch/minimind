"""
Import final_dataset to OddsMind canonical JSONL (P1.0)

Usage:
    python tools/odds_import_final_dataset.py \
        --input data/raw_jsonl/epl_2425_bet365 \
        --out data/odds_real/epl_2425_final.jsonl \
        --write
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.importers.final_dataset_jsonl import import_final_dataset


def main():
    parser = argparse.ArgumentParser(description="Import final_dataset to OddsMind JSONL")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--asian-label-mode", type=str, default="5class")
    parser.add_argument("--upper-side", type=str, default="home")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    result = import_final_dataset(
        data_dir=args.input,
        asian_label_mode=args.asian_label_mode,
        upper_side=args.upper_side,
        limit=args.limit,
    )

    report = result["report"]
    print("=== Import Summary ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    if not args.write:
        print("\n[Dry-run] Use --write to persist.")
        return

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for s in result["samples"]:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(result['samples'])} samples to {args.out}")


if __name__ == "__main__":
    main()
