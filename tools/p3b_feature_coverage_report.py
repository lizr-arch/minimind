"""
P3b Feature Coverage Report: 验证 25 个 cross-market 特征的数据覆盖率。

检查每个特征的非零率、均值、标准差，特别关注 Asian/OU 特征是否大量为 0。
如果 Asian/OU 特征几乎全 0，说明 P3b 的 cross-market 信息无效。

Usage:
    python tools/p3b_feature_coverage_report.py \
        --data data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl \
        --ids data/odds_real/splits_v6/train_match_ids.txt \
        --out runs/p3b_feature_coverage_report.json
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.odds_patch_itransformer_v2 import (
    FEATURE_DEFS,
    extract_cross_market_feature,
)


def load_match_ids(ids_path):
    with open(ids_path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def load_jsonl_data(data_path, match_ids):
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            if row.get("match_id") in match_ids:
                records.append(row)
    return records


def analyze(args):
    match_ids = load_match_ids(args.ids)
    records = load_jsonl_data(args.data, match_ids)
    print(f"Loaded {len(records)} matches for {len(match_ids)} IDs")

    feature_names = [fd[0] for fd in FEATURE_DEFS]
    feature_values = defaultdict(list)

    num_events = 0
    for row in records:
        timeline = row.get("raw_timeline", []) or row.get("odds_timeline", [])
        for event in timeline:
            num_events += 1
            for fname in feature_names:
                val = extract_cross_market_feature(event, fname)
                if math.isfinite(val):
                    feature_values[fname].append(val)

    print(f"Processed {num_events} events across {len(records)} matches")

    results = []
    warnings = []

    for fname in feature_names:
        vals = feature_values.get(fname, [])
        n = len(vals)
        if n == 0:
            results.append({
                "feature_name": fname,
                "non_zero_ratio": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "num_values": 0,
                "warning": ["NO_VALUES"],
            })
            warnings.append(f"BLOCKER: {fname} has no values")
            continue

        vt = vals
        non_zero = sum(1 for v in vt if v != 0.0)
        mean_v = sum(vt) / n
        std_v = (sum((v - mean_v) ** 2 for v in vt) / n) ** 0.5
        min_v = min(vt)
        max_v = max(vt)
        nzr = non_zero / n

        feat_warnings = []
        if fname.startswith("has_") and nzr < 0.01:
            feat_warnings.append(f"WARN: {fname} non_zero_ratio={nzr:.4f} < 1%")
            warnings.append(f"WARN: {fname} almost all zero ({nzr:.1%})")
        if fname in ("asian_line", "upper_water", "lower_water",
                     "asian_upper_implied", "asian_lower_implied", "asian_margin") and nzr < 0.05:
            feat_warnings.append(f"WARN: Asian feature {fname} non_zero_ratio={nzr:.4f}")
            warnings.append(f"WARN: Asian {fname} mostly zero ({nzr:.1%})")
        if fname in ("over_under_line", "over_water", "under_water",
                     "over_implied", "under_implied", "ou_margin") and nzr < 0.05:
            feat_warnings.append(f"WARN: OU feature {fname} non_zero_ratio={nzr:.4f}")
            warnings.append(f"WARN: OU {fname} mostly zero ({nzr:.1%})")

        results.append({
            "feature_name": fname,
            "non_zero_ratio": round(nzr, 6),
            "mean": round(mean_v, 6),
            "std": round(std_v, 6),
            "min": round(min_v, 6),
            "max": round(max_v, 6),
            "num_values": n,
            "warning": feat_warnings,
        })

    has_euro_nzr = next((r["non_zero_ratio"] for r in results if r["feature_name"] == "has_euro"), 0)
    has_asian_nzr = next((r["non_zero_ratio"] for r in results if r["feature_name"] == "has_asian"), 0)
    has_ou_nzr = next((r["non_zero_ratio"] for r in results if r["feature_name"] == "has_ou"), 0)

    summary = {
        "num_matches": len(records),
        "num_events": num_events,
        "has_euro_non_zero_ratio": has_euro_nzr,
        "has_asian_non_zero_ratio": has_asian_nzr,
        "has_ou_non_zero_ratio": has_ou_nzr,
        "features": results,
        "warnings": warnings,
    }

    if has_asian_nzr < 0.05:
        summary["verdict"] = "BLOCKED"
        summary["verdict_reason"] = f"Asian features mostly zero ({has_asian_nzr:.1%}), P3b cross-market invalid"
    elif has_ou_nzr < 0.05:
        summary["verdict"] = "BLOCKED"
        summary["verdict_reason"] = f"OU features mostly zero ({has_ou_nzr:.1%}), P3b cross-market invalid"
    elif has_asian_nzr < 0.3 or has_ou_nzr < 0.3:
        summary["verdict"] = "PASS_WITH_WARNINGS"
        summary["verdict_reason"] = f"Low coverage: asian={has_asian_nzr:.1%}, ou={has_ou_nzr:.1%}"
    else:
        summary["verdict"] = "PASS"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== P3b Feature Coverage Report ===")
    print(f"Events: {num_events}")
    print(f"has_euro  non-zero: {has_euro_nzr:.1%}")
    print(f"has_asian non-zero: {has_asian_nzr:.1%}")
    print(f"has_ou    non-zero: {has_ou_nzr:.1%}")
    print(f"\nKey features:")
    for r in results:
        if r["feature_name"] in ("has_euro", "has_asian", "has_ou",
                                  "asian_line", "upper_water", "lower_water",
                                  "over_under_line", "over_water", "under_water"):
            print(f"  {r['feature_name']:>25}: nzr={r['non_zero_ratio']:.4f}  "
                  f"mean={r['mean']:.4f}  std={r['std']:.4f}")
    if warnings:
        print(f"\nWarnings:")
        for w in warnings:
            print(f"  {w}")
    print(f"\nVerdict: {summary['verdict']}")
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P3b Feature Coverage Report")
    parser.add_argument("--data", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    analyze(args)
