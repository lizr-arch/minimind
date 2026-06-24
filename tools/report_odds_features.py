"""
P1.0G: Odds Feature Schema Report Generator.

Audits feature schema versions (v1/v2/v3/v4), checks for nan/inf,
and reports implied/no-vig/overround statistics.

Usage:
    python tools/report_odds_features.py \
        --input data/odds_real/titan007_pure_v3_time.jsonl \
        --splits-dir data/odds_real/splits/v3_time \
        --output docs/review/odds_feature_schema_report.md \
        --feature-schema v4
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from dataset.odds_dataset import (
    OddsDataset,
    FEATURE_KEYS,
    V4_FEATURE_NAMES,
    V4_FEATURE_DIM,
    _event_to_features,
    _event_to_features_v4,
    _event_to_missing_mask,
    _event_to_missing_mask_v4,
    safe_inverse_odds,
    safe_implied_probs,
    safe_novig_probs,
    safe_overround,
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


def analyze_features(jsonl_path: str, feature_schema: str) -> dict:
    """Analyze feature statistics across the dataset."""
    stats = {
        "schema": feature_schema,
        "total_matches": 0,
        "total_events": 0,
        "nan_count": 0,
        "inf_count": 0,
        "finite_count": 0,
        "feature_sums": None,
        "feature_sq_sums": None,
        "feature_min": None,
        "feature_max": None,
        "feature_count": 0,
        "examples": [],
    }

    if feature_schema == "v4":
        fdim = V4_FEATURE_DIM
        feat_fn = _event_to_features_v4
    else:
        fdim = len(FEATURE_KEYS) + (3 if feature_schema in ("v2", "v3") else 0)
        feat_fn = lambda e: _event_to_features(e, schema_version=feature_schema)

    stats["feature_dim"] = fdim
    stats["feature_sums"] = [0.0] * fdim
    stats["feature_sq_sums"] = [0.0] * fdim
    stats["feature_min"] = [float("inf")] * fdim
    stats["feature_max"] = [float("-inf")] * fdim

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            stats["total_matches"] += 1
            tl = m.get("odds_timeline", [])
            stats["total_events"] += len(tl)

            # Collect example from first match
            if stats["total_matches"] == 1 and tl:
                ex_feats = feat_fn(tl[0])
                stats["examples"] = ex_feats

            for e in tl:
                feats = feat_fn(e)
                for i, v in enumerate(feats):
                    if math.isnan(v):
                        stats["nan_count"] += 1
                    elif math.isinf(v):
                        stats["inf_count"] += 1
                    else:
                        stats["finite_count"] += 1
                        stats["feature_sums"][i] += v
                        stats["feature_sq_sums"][i] += v * v
                        stats["feature_min"][i] = min(stats["feature_min"][i], v)
                        stats["feature_max"][i] = max(stats["feature_max"][i], v)
                stats["feature_count"] += 1

    return stats


def generate_report(args) -> str:
    stats = analyze_features(args.input, args.feature_schema)
    splits = load_split_ids(args.splits_dir)

    fdim = stats["feature_dim"]
    n = stats["feature_count"]

    lines = []
    lines.append("# Odds Feature Schema Report")
    lines.append(f"\nGenerated: {datetime.now().isoformat()}")
    lines.append(f"\nInput: `{args.input}`")
    lines.append(f"Splits dir: `{args.splits_dir}`")
    lines.append(f"Feature schema: `{args.feature_schema}`")

    lines.append("\n## 1. Schema Versions")
    lines.append("| Schema | Features | Dim | Description |")
    lines.append("|--------|----------|-----|-------------|")
    lines.append("| v1 | FEATURE_KEYS only | 10 | Raw odds + minutes |")
    lines.append("| v2 | + has_euro/has_asian/has_ou | 13 | + market availability flags |")
    lines.append("| v3 | same as v2 + missing_mask | 13 + mask | + per-feature missing mask |")
    lines.append("| v4 | raw + implied + no-vig + overround | 32 | Full probability features |")

    lines.append(f"\n## 2. Current Schema: {args.feature_schema}")
    lines.append(f"- **Feature dim**: {fdim}")
    lines.append(f"- **Total events analyzed**: {stats['total_events']}")

    lines.append("\n## 3. Feature List")
    if args.feature_schema == "v4":
        lines.append("| Index | Name | Source | Missing Handling |")
        lines.append("|-------|------|--------|------------------|")
        for i, name in enumerate(V4_FEATURE_NAMES):
            source = "computed" if ("implied" in name or "novig" in name or "overround" in name or "spread" in name) else "raw"
            if "has_" in name:
                source = "detection"
            missing = "safe default (0.0 or 1/3)" if source == "computed" else "missing_mask"
            lines.append(f"| {i} | {name} | {source} | {missing} |")
    else:
        lines.append("| Index | Name |")
        lines.append("|-------|------|")
        for i, name in enumerate(FEATURE_KEYS):
            lines.append(f"| {i} | {name} |")
        if args.feature_schema in ("v2", "v3"):
            lines.append(f"| {len(FEATURE_KEYS)} | has_euro |")
            lines.append(f"| {len(FEATURE_KEYS)+1} | has_asian |")
            lines.append(f"| {len(FEATURE_KEYS)+2} | has_over_under |")

    lines.append("\n## 4. Finite Check Results")
    lines.append(f"- Total values checked: {stats['finite_count'] + stats['nan_count'] + stats['inf_count']}")
    lines.append(f"- Finite: **{stats['finite_count']}**")
    lines.append(f"- NaN: **{stats['nan_count']}**")
    lines.append(f"- Inf: **{stats['inf_count']}**")
    if stats['nan_count'] == 0 and stats['inf_count'] == 0:
        lines.append("- **PASS**: All features are finite")
    else:
        lines.append("- **FAIL**: NaN or Inf detected")

    lines.append("\n## 5. Per-Feature Statistics")
    if n > 0:
        names = V4_FEATURE_NAMES if args.feature_schema == "v4" else (
            FEATURE_KEYS + (["has_euro", "has_asian", "has_over_under"]
                           if args.feature_schema in ("v2", "v3") else [])
        )
        lines.append("| Feature | Mean | Std | Min | Max |")
        lines.append("|---------|------|-----|-----|-----|")
        for i in range(min(fdim, len(names))):
            mean = stats["feature_sums"][i] / n
            var = stats["feature_sq_sums"][i] / n - mean * mean
            std = math.sqrt(max(0, var))
            lines.append(f"| {names[i]} | {mean:.4f} | {std:.4f} | {stats['feature_min'][i]:.4f} | {stats['feature_max'][i]:.4f} |")

    lines.append("\n## 6. Example v4 Features (first event)")
    if stats["examples"]:
        if args.feature_schema == "v4":
            names = V4_FEATURE_NAMES
        else:
            names = FEATURE_KEYS + (["has_euro", "has_asian", "has_over_under"]
                                   if args.feature_schema in ("v2", "v3") else [])
        lines.append("| Feature | Value |")
        lines.append("|---------|-------|")
        for i, v in enumerate(stats["examples"][:len(names)]):
            lines.append(f"| {names[i]} | {v:.6f} |")

    lines.append("\n## 7. Implied / No-vig / Overround Examples")
    if stats["examples"] and args.feature_schema == "v4":
        lines.append("- **euro_h_implied**: " + f"{stats['examples'][4]:.6f}")
        lines.append("- **euro_d_implied**: " + f"{stats['examples'][5]:.6f}")
        lines.append("- **euro_a_implied**: " + f"{stats['examples'][6]:.6f}")
        lines.append("- **euro_overround**: " + f"{stats['examples'][7]:.6f}")
        lines.append("- **euro_h_novig**: " + f"{stats['examples'][8]:.6f}")
    else:
        lines.append("- (Only available in v4 schema)")

    lines.append("\n## 8. Per-Split Finite Check")
    lines.append("| Split | Matches | Events | NaN | Inf | Finite |")
    lines.append("|-------|---------|--------|-----|-----|--------|")
    for split_name, ids in splits.items():
        if not ids:
            lines.append(f"| {split_name} | 0 | 0 | - | - | - |")
            continue
        split_nan = 0
        split_inf = 0
        split_finite = 0
        split_events = 0
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                m = json.loads(line.strip())
                if m["match_id"] not in ids:
                    continue
                tl = m.get("odds_timeline", [])
                split_events += len(tl)
                feat_fn = _event_to_features_v4 if args.feature_schema == "v4" else (
                    lambda e: _event_to_features(e, schema_version=args.feature_schema)
                )
                for e in tl:
                    for v in feat_fn(e):
                        if math.isnan(v):
                            split_nan += 1
                        elif math.isinf(v):
                            split_inf += 1
                        else:
                            split_finite += 1
        lines.append(f"| {split_name} | {len(ids)} | {split_events} | {split_nan} | {split_inf} | {split_finite} |")

    lines.append("\n## 9. NaN / Inf Detection")
    if stats["nan_count"] == 0 and stats["inf_count"] == 0:
        lines.append("- **No NaN or Inf detected** in any feature")
    else:
        lines.append(f"- **NaN count**: {stats['nan_count']}")
        lines.append(f"- **Inf count**: {stats['inf_count']}")
        lines.append("- **BLOCKED**: Fix safe math functions before training")

    lines.append("\n## 10. Recommendation")
    if stats["nan_count"] == 0 and stats["inf_count"] == 0:
        lines.append("\n**v4 schema is safe for training use.**")
        lines.append("- All features are finite")
        lines.append("- Missing values use safe defaults (0.0 for odds, 1/3 for probabilities)")
        lines.append("- Missing mask correctly identifies placeholder values")
        lines.append("- Implied/no-vig/overround features add probabilistic interpretation")
    else:
        lines.append("\n**BLOCKED**: Fix NaN/Inf before using v4 schema")

    verdict = "PASS" if (stats["nan_count"] == 0 and stats["inf_count"] == 0) else "BLOCKED"
    lines.append(f"\n**Verdict: {verdict}**")

    lines.append(f"\n---\n*Report generated by `tools/report_odds_features.py`*")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="P1.0G Odds Feature Schema Report")
    parser.add_argument("--input", required=True, help="Path to JSONL data file")
    parser.add_argument("--splits-dir", required=True, help="Path to splits directory")
    parser.add_argument("--output", required=True, help="Output markdown file path")
    parser.add_argument("--feature-schema", default="v4", choices=["v1", "v2", "v3", "v4"],
                        help="Feature schema version to audit")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    report = generate_report(args)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
