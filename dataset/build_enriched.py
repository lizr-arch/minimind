"""
Build enriched JSONL: original Bet365 timeline + multi-bookmaker features.

Usage:
    python dataset/build_enriched.py
"""

import json, sys, os
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset.odds_bookmaker_features import extract_bookmaker_features, get_feature_names

SRC = "data/odds_real/all_bookmakers_11679.jsonl"
DST = "data/odds_real/enriched_bookmaker_features.jsonl"


def main():
    # Group by match_id
    per_match = defaultdict(dict)
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            mid = s["match_id"]
            bk = s["bookmaker_id"]
            closing = s["odds_timeline"][-1]
            per_match[mid][bk] = closing

    # Keep Bet365 version as base, add features
    with open(SRC, encoding="utf-8") as f:
        all_samples = [json.loads(l) for l in f if l.strip()]

    # Find one Bet365 sample per match
    bet365_samples = {}
    for s in all_samples:
        if s["bookmaker_id"] == "Bet365":
            mid = s["match_id"]
            if mid not in bet365_samples:
                bet365_samples[mid] = s

    # Build enriched samples
    enriched = []
    for mid, b365 in bet365_samples.items():
        bk_data = per_match.get(mid, {})
        feats = extract_bookmaker_features(bk_data)
        b365["bookmaker_features"] = feats
        enriched.append(b365)

    os.makedirs(os.path.dirname(DST), exist_ok=True)
    with open(DST, "w", encoding="utf-8") as f:
        for s in enriched:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Enriched: {len(enriched)} matches")
    feat_names = get_feature_names()
    print(f"Feature dim: {len(feat_names)}")

    # Show sample
    if enriched:
        s = enriched[0]
        n_feats = len(s.get("bookmaker_features", {}))
        fkeys = list(s.get("bookmaker_features", {}).keys())[:5]
        print(f"Sample: {s['match_id']}, {len(s['odds_timeline'])} events, {n_feats} features")
        for k in fkeys:
            print(f"  {k}: {s['bookmaker_features'][k]:.4f}")


if __name__ == "__main__":
    main()
