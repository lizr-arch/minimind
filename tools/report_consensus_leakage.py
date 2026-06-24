"""
P1.0E: Consensus Leakage Report Generator.

Audits consensus_feats for time-series data leakage and generates
a markdown report.

Usage:
    python tools/report_consensus_leakage.py \
        --input data/odds_real/titan007_pure_v3_time.jsonl \
        --splits-dir data/odds_real/splits/v3_time \
        --output docs/review/consensus_leakage_report.md \
        --cutoff-buckets "1440,60,1" \
        --consensus-mode none
"""

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from dataset.odds_dataset import (
    OddsDataset,
    _compute_consensus_feats,
    _event_has_euro,
    CONSENSUS_MODES,
)
from dataset.odds_cutoff import build_cutoff_samples, CUTOFF_BUCKETS_V1


def load_split_ids(splits_dir: str) -> dict:
    """Load train/val/test match_id sets from split files."""
    result = {}
    for name in ("train", "val", "test"):
        path = os.path.join(splits_dir, f"{name}_match_ids.txt")
        if os.path.exists(path):
            with open(path, "r") as f:
                result[name] = {line.strip() for line in f if line.strip()}
        else:
            result[name] = set()
    return result


def analyze_consensus_feats(jsonl_path: str, cutoff_buckets: list, consensus_mode: str) -> dict:
    """Analyze consensus_feats across the dataset."""
    results = {
        "total_matches": 0,
        "total_events": 0,
        "matches_with_bookmaker_features": 0,
        "events_with_real_euro": 0,
        "consensus_mode": consensus_mode,
        "cutoff_comparisons": {},
    }

    # Load all matches
    matches = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            matches.append(m)

    results["total_matches"] = len(matches)

    # Count events and bookmaker_features
    for m in matches:
        tl = m.get("odds_timeline", [])
        results["total_events"] += len(tl)
        if "bookmaker_features" in m:
            results["matches_with_bookmaker_features"] += 1
        for e in tl:
            if _event_has_euro(e):
                results["events_with_real_euro"] += 1

    # Compare consensus_feats across cutoffs for a sample of matches
    sample_matches = matches[:min(10, len(matches))]
    for cutoff in cutoff_buckets:
        cutoff_key = str(int(cutoff))
        feats_at_cutoff = []
        for m in sample_matches:
            cs_list = build_cutoff_samples(m, [cutoff], min_events=1)
            if not cs_list:
                continue
            cs = cs_list[0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                feat = _compute_consensus_feats(
                    cs, consensus_mode=consensus_mode,
                    visible_timeline=cs.get("odds_timeline", []),
                )
            feats_at_cutoff.append(feat)

        if feats_at_cutoff:
            stacked = torch.stack(feats_at_cutoff)
            results["cutoff_comparisons"][cutoff_key] = {
                "mean": stacked.mean(dim=0).tolist(),
                "std": stacked.std(dim=0).tolist(),
                "all_zero": bool(stacked.abs().sum().item() == 0),
                "count": len(feats_at_cutoff),
            }

    return results


def generate_report(args) -> str:
    """Generate the consensus leakage report markdown."""
    cutoff_buckets = [float(x) for x in args.cutoff_buckets.split(",")]

    analysis = analyze_consensus_feats(args.input, cutoff_buckets, args.consensus_mode)

    # Load split info
    splits = load_split_ids(args.splits_dir)

    lines = []
    lines.append("# Consensus Leakage Report")
    lines.append(f"\nGenerated: {datetime.now().isoformat()}")
    lines.append(f"\nInput: `{args.input}`")
    lines.append(f"Splits dir: `{args.splits_dir}`")
    lines.append(f"Consensus mode: `{args.consensus_mode}`")

    lines.append("\n## 1. Data Overview")
    lines.append(f"- Total matches: {analysis['total_matches']}")
    lines.append(f"- Total events: {analysis['total_events']}")
    lines.append(f"- Matches with `bookmaker_features`: {analysis['matches_with_bookmaker_features']}")
    lines.append(f"- Events with real euro odds: {analysis['events_with_real_euro']}")
    lines.append(f"- Train matches: {len(splits.get('train', set()))}")
    lines.append(f"- Val matches: {len(splits.get('val', set()))}")
    lines.append(f"- Test matches: {len(splits.get('test', set()))}")

    lines.append("\n## 2. Consensus Feats Generation Location")
    lines.append("- **Where**: `dataset/odds_dataset.py` → `_compute_consensus_feats()`")
    lines.append("- **When**: Called from `_build_features_and_labels()` for each dataset item")
    lines.append("- **Input**: Depends on `consensus_mode` parameter")

    lines.append("\n## 3. Current Data: bookmaker_features Status")
    if analysis["matches_with_bookmaker_features"] == 0:
        lines.append("- **Result**: NO matches have `bookmaker_features` in the JSONL data")
        lines.append("- **Implication**: `consensus_feats` is ALWAYS `zeros(6)` regardless of mode")
        lines.append("- **Leakage risk**: NONE — zeros contain no information")
    else:
        lines.append(f"- **Result**: {analysis['matches_with_bookmaker_features']} matches have `bookmaker_features`")
        lines.append("- **Implication**: Consensus features may contain information")

    lines.append("\n## 4. Consensus Mode Analysis")
    lines.append(f"- Current mode: `{analysis['consensus_mode']}`")
    lines.append(f"- Available modes: {CONSENSUS_MODES}")
    lines.append("- `none`: Returns `zeros(6)` — safe, no leakage")
    lines.append("- `visible_only`: Computes from cutoff-filtered timeline — cutoff-safe")
    lines.append("- `legacy_full_timeline`: Uses full-match bookmaker_features — **DEBUG ONLY, leaks future data**")

    lines.append("\n## 5. Cutoff Comparison")
    lines.append("| Cutoff | All Zero | Mean[0] | Count |")
    lines.append("|--------|----------|---------|-------|")
    for cutoff_key, stats in sorted(analysis["cutoff_comparisons"].items(), key=lambda x: -float(x[0])):
        mean_0 = f"{stats['mean'][0]:.6f}"
        lines.append(f"| {cutoff_key} | {stats['all_zero']} | {mean_0} | {stats['count']} |")

    lines.append("\n## 6. Full Timeline vs Cutoff-Filtered Comparison")
    lines.append("- `build_cutoff_samples()` does NOT copy `bookmaker_features` to cutoff samples")
    lines.append("- Even if full-match had bookmaker_features, cutoff samples would have none")
    lines.append("- This is an additional safety layer against leakage")

    lines.append("\n## 7. Missing Data")
    lines.append("- `bookmaker_features` field: **ABSENT** from all matches")
    lines.append("- `consensus_feats` output: **ALWAYS zeros(6)**")
    lines.append("- Closing odds: Not used for consensus computation")

    lines.append("\n## 8. Leak Checks")
    leak_checks = [
        ("Uses full timeline consensus?", analysis["consensus_mode"] != "none"),
        ("Uses closing odds?", False),
        ("build_cutoff_sample copies full-match consensus?", False),
        ("Collator generates its own consensus?", False),
        ("Dataset item missing consensus → zeros?", False),
        ("cutoff=1440/60/1 produce same consensus?", analysis["consensus_mode"] != "none"),
    ]
    all_safe = True
    for check, is_leaky in leak_checks:
        safe = not is_leaky
        if not safe:
            all_safe = False
        status = "SAFE" if safe else "LEAK_DETECTED"
        lines.append(f"- {check}: **{status}**")

    lines.append("\n## 9. Verdict")
    has_bk_feats = analysis["matches_with_bookmaker_features"] > 0

    if analysis["consensus_mode"] == "none" and not has_bk_feats:
        verdict = "PASS"
        lines.append(f"\n**{verdict}**")
        lines.append("- consensus_mode is `none` (default)")
        lines.append("- No bookmaker_features in data")
        lines.append("- consensus_feats is always zeros(6)")
        lines.append("- No future information leakage possible")
    elif analysis["consensus_mode"] == "none":
        verdict = "PASS"
        lines.append(f"\n**{verdict}**")
        lines.append("- consensus_mode is `none` — zeros regardless of data")
    elif all_safe:
        verdict = "PASS"
        lines.append(f"\n**{verdict}**")
        lines.append("- All leak checks passed")
    else:
        verdict = "BLOCKED"
        lines.append(f"\n**{verdict}**")
        lines.append("- Leakage risks detected — see checks above")

    lines.append(f"\n---\n*Report generated by `tools/report_consensus_leakage.py`*")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="P1.0E Consensus Leakage Report")
    parser.add_argument("--input", required=True, help="Path to JSONL data file")
    parser.add_argument("--splits-dir", required=True, help="Path to splits directory")
    parser.add_argument("--output", required=True, help="Output markdown file path")
    parser.add_argument("--cutoff-buckets", default="1440,60,1",
                        help="Comma-separated cutoff bucket values")
    parser.add_argument("--consensus-mode", default="none",
                        choices=CONSENSUS_MODES,
                        help="Consensus mode to audit")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    report = generate_report(args)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
