"""
OddsMind CSV Import CLI (P0.8)

Usage:
    python tools/odds_import_csv.py --input raw.csv --preset football_data_b365 \
        --league-id EPL --bookmaker-id B365 --write --out out.jsonl
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.importers.field_mapping import load_field_mapping
from dataset.importers.football_data_csv import import_football_data_csv


def main():
    parser = argparse.ArgumentParser(description="OddsMind CSV Importer")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--report", type=str, default="")
    parser.add_argument("--preset", type=str, default="football_data_b365")
    parser.add_argument("--mapping", type=str, default="")
    parser.add_argument("--league-id", type=str, required=True)
    parser.add_argument("--season", type=str, default="")
    parser.add_argument("--bookmaker-id", type=str, default="B365")
    parser.add_argument("--default-minutes-before-kickoff", type=float, default=0)
    parser.add_argument("--asian-label-mode", type=str, default="5class")
    parser.add_argument("--upper-side", type=str, default="home")
    parser.add_argument("--allow-missing-asian", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    mapping = load_field_mapping(path=args.mapping or None, preset=args.preset)

    result = import_football_data_csv(
        csv_path=args.input,
        mapping=mapping,
        league_id=args.league_id,
        season=args.season,
        bookmaker_id=args.bookmaker_id,
        upper_side=args.upper_side,
        asian_label_mode=args.asian_label_mode,
        default_minutes_before_kickoff=args.default_minutes_before_kickoff,
        allow_missing_asian=args.allow_missing_asian,
        limit=args.limit,
    )

    report = result["report"]
    samples = result["samples"]

    print("=== Import Summary ===")
    for k, v in report.items():
        if k == "sample_preview":
            print(f"  {k}: {len(v)} items")
        elif k == "label_distribution":
            print(f"  {k}:")
            for sk, sv in v.items():
                print(f"    {sk}: {sv}")
        else:
            print(f"  {k}: {v}")

    if not args.write:
        print("\n[Dry-run] No files written. Use --write to persist.")
        return

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(samples)} samples to {args.out}")

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved to {args.report}")


if __name__ == "__main__":
    main()
