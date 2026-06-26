"""
Extract Phase 1a flat features from v4/v6 timeline JSONL.

Input:  pipeline_export.jsonl (v4 timeline format)
Output: Phase 1a flat JSONL with open/close/delta per market

Usage:
    python extract_phase1a.py \
        --input data/training/pipeline_export.jsonl \
        --output data/training/phase1a.jsonl \
        --stats
"""

import json, sys, argparse
from collections import Counter


def find_open(timeline):
    """First event in descending mbk order = farthest from kickoff = open."""
    return timeline[0] if timeline else None


def find_close(timeline):
    """Last event in descending mbk order = closest to kickoff = close."""
    return timeline[-1] if timeline else None


def extract_open_close(timeline):
    """Extract open/close values for all three markets from a timeline."""
    open_evt = find_open(timeline)
    close_evt = find_close(timeline)
    result = {}
    
    # ── Euro (always present) ──
    for prefix, evt in [("open", open_evt), ("close", close_evt)]:
        if evt and evt.get("has_euro", True):
            result[f"{prefix}_euro_h"] = evt.get("euro_h")
            result[f"{prefix}_euro_d"] = evt.get("euro_d")
            result[f"{prefix}_euro_a"] = evt.get("euro_a")
        else:
            for k in ["euro_h", "euro_d", "euro_a"]:
                result[f"{prefix}_{k}"] = None
    
    # ── Asian (may be missing) ──
    for prefix, evt in [("open", open_evt), ("close", close_evt)]:
        if evt and evt.get("has_asian", False):
            result[f"{prefix}_asian_line"] = evt.get("asian_line")
            result[f"{prefix}_asian_upper_water"] = evt.get("upper_water")
            result[f"{prefix}_asian_lower_water"] = evt.get("lower_water")
        else:
            for k in ["asian_line", "asian_upper_water", "asian_lower_water"]:
                result[f"{prefix}_{k}"] = None
    
    # ── Over/Under (may be missing) ──
    for prefix, evt in [("open", open_evt), ("close", close_evt)]:
        if evt and evt.get("has_over_under", False):
            result[f"{prefix}_ou_line"] = evt.get("over_under_line")
            result[f"{prefix}_ou_over_water"] = evt.get("over_water")
            result[f"{prefix}_ou_under_water"] = evt.get("under_water")
        else:
            for k in ["ou_line", "ou_over_water", "ou_under_water"]:
                result[f"{prefix}_{k}"] = None
    
    # ── Deltas ──
    for field in ["euro_h", "euro_d", "euro_a",
                  "asian_line", "asian_upper_water", "asian_lower_water",
                  "ou_line", "ou_over_water", "ou_under_water"]:
        open_val = result.get(f"open_{field}")
        close_val = result.get(f"close_{field}")
        if open_val is not None and close_val is not None:
            result[f"delta_{field}"] = round(close_val - open_val, 4)
        else:
            result[f"delta_{field}"] = None
    
    return result


def extract_phase1a(input_path, output_path, limit=0):
    """Convert timeline JSONL to Phase 1a flat JSONL."""
    stats = Counter()
    written = 0
    
    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        
        for i, line in enumerate(fin, 1):
            if limit and i > limit: break
            
            rec = json.loads(line)
            tl = rec.get("odds_timeline", [])
            if not tl: continue
            
            flat = extract_open_close(tl)
            
            sample = {
                "match_id": rec["match_id"],
                "league_id": rec["league_id"],
                "kickoff_time": rec.get("kickoff_time", ""),
                "bookmaker_id": rec.get("bookmaker_id", ""),
                "bookmaker_type": rec.get("bookmaker_type", ""),
                **flat,
                "label": {
                    "euro_result": rec["label"].get("euro_result"),
                    "asian_result": rec["label"].get("asian_result"),
                    "home_goals": rec["label"].get("home_goals", 0),
                    "away_goals": rec["label"].get("away_goals", 0),
                },
            }
            
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1
            
            for k in ["close_euro_h", "close_asian_line", "close_ou_line"]:
                if flat.get(k) is not None:
                    stats[f"has_{k}"] += 1
            stats["total"] += 1
    
    return written, stats


def main():
    p = argparse.ArgumentParser(description="Extract Phase 1a features")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--stats", action="store_true")
    args = p.parse_args()
    
    written, stats = extract_phase1a(args.input, args.output, args.limit)
    print(f"Written: {written} samples")
    
    if args.stats:
        total = stats["total"]
        print(f"\n=== Coverage ===")
        for k in ["close_euro_h", "close_asian_line", "close_ou_line"]:
            n = stats.get(f"has_{k}", 0)
            pct = n / max(total, 1) * 100
            print(f"  {k}: {n}/{total} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
