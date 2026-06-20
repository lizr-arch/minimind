"""
Multi-File JSONL Importer (P1.0)

Converts final_dataset format (matches.jsonl + odds_1x2_events.jsonl +
odds_ah_events.jsonl) to OddsMind canonical JSONL.

Matches are joined by source_match_id. Odds events are sorted by
minutes_before_kickoff to build the timeline.
"""

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.asian_handicap import settle_asian_5class, settle_asian_3class, get_upper_lower_goals
from dataset.odds_import_schema import validate_imported_match

FTR_MAP = {"H": "home", "D": "draw", "A": "away"}


def import_final_dataset(
    data_dir: str,
    asian_label_mode: str = "5class",
    upper_side: str = "home",
    limit: int = 0,
) -> dict:
    """
    Import a final_dataset directory.

    Args:
        data_dir: path containing matches.jsonl, odds_1x2_events.jsonl, odds_ah_events.jsonl
        asian_label_mode: "3class" or "5class"
        upper_side: "home" or "away"
        limit: max matches to import (0 = all)

    Returns:
        dict with keys: samples, report
    """
    # Load matches
    matches = {}
    with open(os.path.join(data_dir, "matches.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            mid = m["source_match_id"]
            matches[mid] = m

    # Load 1X2 events
    euro_events: Dict[str, List[dict]] = defaultdict(list)
    with open(os.path.join(data_dir, "odds_1x2_events.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            euro_events[e["source_match_id"]].append(e)

    # Load AH events
    ah_events: Dict[str, List[dict]] = defaultdict(list)
    ah_path = os.path.join(data_dir, "odds_ah_events.jsonl")
    if os.path.exists(ah_path):
        with open(ah_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                ah_events[e["source_match_id"]].append(e)

    samples = []
    report = {
        "total_matches": len(matches),
        "imported_matches": 0,
        "skipped_matches": 0,
        "errors": {"no_odds": 0, "no_asian": 0, "validation_failed": 0, "inplay": 0},
        "label_distribution": {"euro_result": {}, "asian_result": {}},
        "timeline_stats": {"avg_len": 0.0, "min_len": 0, "max_len": 0},
    }

    count = 0
    for source_mid, match in matches.items():
        if limit and count >= limit:
            break
        count += 1

        eu_events = euro_events.get(source_mid, [])
        ah_evs = ah_events.get(source_mid, [])

        if not eu_events:
            report["errors"]["no_odds"] += 1
            report["skipped_matches"] += 1
            continue

        # Filter out in-play events (minutes_before_kickoff < 0)
        eu_events = [e for e in eu_events if e.get("minutes_before_kickoff", 0) >= 0]
        ah_evs = [e for e in ah_evs if e.get("minutes_before_kickoff", 0) >= 0]

        if not eu_events:
            report["errors"]["inplay"] += 1
            report["skipped_matches"] += 1
            continue

        # Sort by minutes_before_kickoff descending (earliest first)
        eu_events.sort(key=lambda e: e["minutes_before_kickoff"], reverse=True)
        ah_evs.sort(key=lambda e: e["minutes_before_kickoff"], reverse=True)

        # Build AH lookup by timestamp
        ah_by_time = {}
        for ae in ah_evs:
            t = ae["minutes_before_kickoff"]
            if t not in ah_by_time:  # keep first occurrence at each time
                ah_by_time[t] = ae

        # Build timeline
        odds_timeline = []
        for eu in eu_events:
            t = eu["minutes_before_kickoff"]
            event = {
                "minutes_before_kickoff": t,
                "euro_h": eu["home_odds"],
                "euro_d": eu["draw_odds"],
                "euro_a": eu["away_odds"],
            }
            # Try to match an AH event at the same time
            ah = ah_by_time.get(t)
            if ah:
                event["asian_line"] = ah["line"]
                event["upper_water"] = ah["home_price"]
                event["lower_water"] = ah["away_price"]
            else:
                event["asian_line"] = 0.0
                event["upper_water"] = 1.0
                event["lower_water"] = 1.0
            odds_timeline.append(event)

        # Deduplicate consecutive identical events
        deduped = []
        for ev in odds_timeline:
            if deduped and all(
                abs(ev[k] - deduped[-1][k]) < 1e-6
                for k in ["euro_h", "euro_d", "euro_a", "asian_line", "upper_water", "lower_water"]
            ):
                continue
            deduped.append(ev)
        odds_timeline = deduped

        if not odds_timeline:
            report["skipped_matches"] += 1
            continue

        # Euro result
        hg = match.get("home_goals", 0) or 0
        ag = match.get("away_goals", 0) or 0
        if hg > ag:
            euro_result = "home"
        elif hg < ag:
            euro_result = "away"
        else:
            euro_result = "draw"

        # Asian result from closing event
        closing = odds_timeline[-1]
        asian_result = "push"
        has_asian = ah_evs and closing.get("asian_line", 0) != 0
        if has_asian:
            ug, lg = get_upper_lower_goals(hg, ag, upper_side)
            al = closing["asian_line"]
            if asian_label_mode == "5class":
                asian_result = settle_asian_5class(ug, lg, al)
            else:
                asian_result = settle_asian_3class(ug, lg, al)

        # Build sample
        sample = {
            "match_id": source_mid,
            "league_id": match.get("league_id", "unknown"),
            "bookmaker_id": "Bet365",
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
            report["skipped_matches"] += 1
            continue

        samples.append(sample)
        report["imported_matches"] += 1
        report["label_distribution"]["euro_result"][euro_result] = \
            report["label_distribution"]["euro_result"].get(euro_result, 0) + 1
        report["label_distribution"]["asian_result"][asian_result] = \
            report["label_distribution"]["asian_result"].get(asian_result, 0) + 1

    if samples:
        lens = [len(s["odds_timeline"]) for s in samples]
        report["timeline_stats"] = {
            "avg_len": sum(lens) / len(lens),
            "min_len": min(lens),
            "max_len": max(lens),
        }

    return {"samples": samples, "report": report}
