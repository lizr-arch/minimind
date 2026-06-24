"""
Master Dataset Importer (P1.2 — Multi-Bookmaker fix)

Imports master_dataset (football_data open/close + Titan007 movement)
into OddsMind canonical JSONL.

Key fix: groups events by (match, bookmaker) — one sample per bookmaker
per match. No longer hardcodes "Bet365" or merges bookmakers.

Replace: D:\code\git\betmind\minimind\dataset\importers\master_dataset.py
"""

import json, os, sys
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dataset.asian_handicap import settle_asian_5class, settle_asian_3class, get_upper_lower_goals
from dataset.odds_import_schema import validate_imported_match


def _events_identical(a: dict, b: dict) -> bool:
    keys = ["euro_h", "euro_d", "euro_a", "asian_line", "upper_water", "lower_water"]
    return all(abs(a.get(k, 0) - b.get(k, 0)) < 1e-5 for k in keys)


def import_master_dataset(
    data_dir: str,
    asian_label_mode: str = "5class",
    upper_side: str = "home",
    min_events: int = 1,
    limit: int = 0,
) -> dict:
    """Import master_dataset directory with multi-bookmaker support."""

    # ── Load matches ──
    match_path = os.path.join(data_dir, "matches.jsonl")
    matches: Dict[str, dict] = {}
    norm_to_mid: Dict[str, str] = {}
    with open(match_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            mid = m["source_match_id"]
            matches[mid] = m
            date = m.get("kickoff_time", "")[:10]
            home = m.get("home_team", "").lower().replace(" ", "")
            away = m.get("away_team", "").lower().replace(" ", "")
            norm_to_mid[f"{date}_{home}_{away}"] = mid

    # ── Load 1X2 events, grouped by (match_id, bookmaker) ──
    eu_events: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    eu_path = os.path.join(data_dir, "odds_1x2_events.jsonl")
    all_bookmakers: set = set()
    with open(eu_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            mid = e["source_match_id"]
            bk = e.get("bookmaker", "Bet365")
            all_bookmakers.add(bk)

            if mid in matches:
                eu_events[mid][bk].append(e)
            else:
                # Fuzzy match for Titan007 IDs
                mid_parts = mid.split("-")
                if len(mid_parts) >= 8:
                    date = f"{mid_parts[4]}-{mid_parts[3]}-{mid_parts[2]}"
                    home = "-".join(mid_parts[5:-1]).lower().replace(" ", "").replace("-", "")
                    away = mid_parts[-1].lower().replace(" ", "").replace("-", "")
                    norm_key = f"{date}_{home}_{away}"
                    if norm_key in norm_to_mid:
                        eu_events[norm_to_mid[norm_key]][bk].append(e)

    # ── Load AH events, grouped by (match_id, bookmaker) ──
    ah_events: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    ah_path = os.path.join(data_dir, "odds_ah_events.jsonl")
    with open(ah_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            mid = e["source_match_id"]
            bk = e.get("bookmaker", "Bet365")
            all_bookmakers.add(bk)

            if mid in matches:
                ah_events[mid][bk].append(e)
            else:
                mid_parts = mid.split("-")
                if len(mid_parts) >= 8:
                    date = f"{mid_parts[5]}-{mid_parts[4]}-{mid_parts[3]}"
                    home = "-".join(mid_parts[6:-1]).lower().replace(" ", "")
                    away = mid_parts[-1].lower().replace(" ", "")
                    norm_key = f"{date}_{home}_{away}"
                    if norm_key in norm_to_mid:
                        ah_events[norm_to_mid[norm_key]][bk].append(e)

    # ── Build samples — one per (match, bookmaker) ──
    samples = []
    report = {
        "total_matches": len(matches),
        "total_bookmakers": len(all_bookmakers),
        "bookmakers": sorted(all_bookmakers),
        "imported_samples": 0,
        "skipped_matches": 0,
        "errors": {"no_odds": 0, "no_asian": 0, "inplay_filtered": 0, "validation_failed": 0},
        "label_distribution": {"euro_result": {}, "asian_result": {}},
        "per_bookmaker": {},
    }

    count = 0
    for source_mid, match in matches.items():
        if limit and count >= limit:
            break
        count += 1

        # Collect all bookmakers that have data for this match
        match_bms = set(eu_events.get(source_mid, {}).keys()) | set(ah_events.get(source_mid, {}).keys())
        if not match_bms:
            report["errors"]["no_odds"] += 1
            report["skipped_matches"] += 1
            continue

        match_has_sample = False
        for bk in match_bms:
            eu = eu_events.get(source_mid, {}).get(bk, [])
            ah = ah_events.get(source_mid, {}).get(bk, [])

            # Filter in-play
            eu = [e for e in eu if e.get("minutes_before_kickoff", 0) >= 0 and not e.get("in_play")]
            ah = [e for e in ah if e.get("minutes_before_kickoff", 0) >= 0 and not e.get("in_play")]
            if not eu:
                continue  # Skip this bookmaker for this match

            # Sort by minutes_before_kickoff DESC
            eu.sort(key=lambda e: e.get("minutes_before_kickoff", 0), reverse=True)
            ah.sort(key=lambda e: e.get("minutes_before_kickoff", 0), reverse=True)

            # Build AH lookup
            ah_by_time = {}
            for ae in ah:
                ah_by_time[ae.get("minutes_before_kickoff", 0)] = ae

            # Build timeline
            odds_timeline = []
            for eu_ev in eu:
                t = eu_ev.get("minutes_before_kickoff", 0)
                event = {
                    "minutes_before_kickoff": t,
                    "euro_h": eu_ev["home_odds"],
                    "euro_d": eu_ev["draw_odds"],
                    "euro_a": eu_ev["away_odds"],
                }
                ahev = ah_by_time.get(t)
                if ahev:
                    event["asian_line"] = ahev["line"]
                    event["upper_water"] = ahev["home_price"]
                    event["lower_water"] = ahev["away_price"]
                else:
                    prev_ah = None
                    for at in sorted(ah_by_time.keys(), reverse=True):
                        if at >= t:
                            prev_ah = ah_by_time[at]
                        else:
                            break
                    if prev_ah:
                        event["asian_line"] = prev_ah["line"]
                        event["upper_water"] = prev_ah["home_price"]
                        event["lower_water"] = prev_ah["away_price"]
                    else:
                        event["asian_line"] = 0.0
                        event["upper_water"] = 1.0
                        event["lower_water"] = 1.0
                odds_timeline.append(event)

            # Deduplicate
            deduped = []
            for ev in odds_timeline:
                if deduped and _events_identical(deduped[-1], ev):
                    deduped[-1]["minutes_before_kickoff"] = ev["minutes_before_kickoff"]
                    continue
                deduped.append(ev)
            odds_timeline = deduped

            if len(odds_timeline) < min_events:
                continue

            # Results
            hg = match.get("home_goals", 0) or 0
            ag = match.get("away_goals", 0) or 0
            if hg > ag:
                euro_result = "home"
            elif hg < ag:
                euro_result = "away"
            else:
                euro_result = "draw"

            closing = odds_timeline[-1]
            asian_result = "push"
            has_asian = closing.get("asian_line", 0) != 0 or bool(ah)
            if has_asian and ah:
                ug, lg = get_upper_lower_goals(hg, ag, upper_side)
                al = closing["asian_line"]
                try:
                    if asian_label_mode == "5class":
                        asian_result = settle_asian_5class(ug, lg, al)
                    else:
                        asian_result = settle_asian_3class(ug, lg, al)
                except (ValueError, KeyError):
                    asian_result = "push"

            sample = {
                "match_id": source_mid,
                "league_id": match.get("league_id", "unknown"),
                "bookmaker_id": bk,
                "kickoff_time": match.get("kickoff_time", ""),
                "odds_timeline": odds_timeline,
                "label": {
                    "euro_result": euro_result,
                    "asian_result": asian_result,
                    "home_goals": hg,
                    "away_goals": ag,
                    "upper_side": upper_side,
                },
            }

            errs = validate_imported_match(sample)
            if errs:
                report["errors"]["validation_failed"] += 1
                continue

            samples.append(sample)
            match_has_sample = True
            report["imported_samples"] += 1
            report["label_distribution"]["euro_result"][euro_result] = \
                report["label_distribution"]["euro_result"].get(euro_result, 0) + 1
            report["label_distribution"]["asian_result"][asian_result] = \
                report["label_distribution"]["asian_result"].get(asian_result, 0) + 1
            report["per_bookmaker"][bk] = report["per_bookmaker"].get(bk, 0) + 1

        if not match_has_sample:
            report["skipped_matches"] += 1

    if samples:
        lens = [len(s["odds_timeline"]) for s in samples]
        report["timeline_stats"] = {
            "avg_len": sum(lens) / len(lens),
            "min_len": min(lens),
            "max_len": max(lens),
        }

    return {"samples": samples, "report": report}


# ── CLI ──
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--asian-label-mode", default="5class")
    parser.add_argument("--upper-side", default="home")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    result = import_master_dataset(
        data_dir=args.input,
        asian_label_mode=args.asian_label_mode,
        upper_side=args.upper_side,
        limit=args.limit,
    )
    r = result["report"]
    print("=== Import Summary ===")
    for k, v in r.items():
        print(f"  {k}: {v}")

    if args.write and args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for s in result["samples"]:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(result['samples'])} samples to {args.out}")
