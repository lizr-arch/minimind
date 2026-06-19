"""
Football-Data CSV Importer (P0.8)

Converts Football-Data.co.uk style CSV rows to OddsMind JSONL samples.

Handles:
  - Euro odds (B365H/B365D/B365A or similar columns via FieldMapping)
  - Asian handicap (B365AH/B365AHH/B365AHA)
  - Match results (FTR, FTHG, FTAG)
  - Builds match_id, kickoff_time, odds_timeline, label
"""

import csv
import os
import sys
from typing import Dict, List, Optional, Set

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dataset.odds_import_schema import (
    build_match_id,
    normalize_team_name,
    validate_imported_match,
)
from dataset.importers.field_mapping import FieldMapping
from dataset.asian_handicap import settle_asian_5class, settle_asian_3class, get_upper_lower_goals

FTR_MAP = {"H": "home", "D": "draw", "A": "away"}


def import_football_data_csv(
    csv_path: str,
    mapping: FieldMapping,
    league_id: str,
    season: str,
    bookmaker_id: str,
    upper_side: str = "home",
    asian_label_mode: str = "5class",
    default_minutes_before_kickoff: float = 0,
    allow_missing_asian: bool = False,
    limit: int = 0,
) -> dict:
    """
    Import a Football-Data style CSV.

    Returns:
        dict with keys: samples, report
    """
    samples = []
    report = {
        "total_rows": 0,
        "imported_matches": 0,
        "skipped_rows": 0,
        "errors": {
            "missing_required_fields": 0,
            "invalid_odds": 0,
            "invalid_score": 0,
            "missing_asian_fields": 0,
            "ambiguous_upper_side": 0,
            "validation_failed": 0,
        },
        "label_distribution": {"euro_result": {}, "asian_result": {}},
        "time_range": {"min_kickoff_time": "", "max_kickoff_time": ""},
        "sample_preview": [],
    }

    required_euro = [mapping.euro_h, mapping.euro_d, mapping.euro_a]
    required_match = [mapping.date, mapping.home_team, mapping.away_team,
                      mapping.home_goals, mapping.away_goals]
    asian_fields = [mapping.asian_line, mapping.upper_water, mapping.lower_water]

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            report["total_rows"] += 1
            if limit and report["total_rows"] > limit:
                break

            # Check required fields
            missing = [f for f in required_euro + required_match if not row.get(f)]
            if missing:
                report["errors"]["missing_required_fields"] += 1
                report["skipped_rows"] += 1
                continue

            # Parse date/time
            date_str = row[mapping.date].strip()
            time_str = (row.get(mapping.time) or "15:00").strip()
            kickoff = f"{date_str}T{time_str}:00Z"

            # Teams
            home = normalize_team_name(row[mapping.home_team])
            away = normalize_team_name(row[mapping.away_team])

            # Scores
            try:
                hg = int(row[mapping.home_goals])
                ag = int(row[mapping.away_goals])
            except (ValueError, KeyError):
                report["errors"]["invalid_score"] += 1
                report["skipped_rows"] += 1
                continue

            # Euro odds
            try:
                euro_h = float(row[mapping.euro_h])
                euro_d = float(row[mapping.euro_d])
                euro_a = float(row[mapping.euro_a])
            except (ValueError, KeyError):
                report["errors"]["invalid_odds"] += 1
                report["skipped_rows"] += 1
                continue

            # Euro result
            ftr = row.get(mapping.ftr, "")
            euro_result = FTR_MAP.get(ftr.strip().upper(), "")
            if not euro_result:
                # Infer from scores
                if hg > ag:
                    euro_result = "home"
                elif hg < ag:
                    euro_result = "away"
                else:
                    euro_result = "draw"

            # Build odds event
            event = {
                "minutes_before_kickoff": default_minutes_before_kickoff,
                "euro_h": euro_h,
                "euro_d": euro_d,
                "euro_a": euro_a,
            }

            # Asian handicap
            asian_result = ""
            has_asian = all(row.get(f) for f in asian_fields)
            if has_asian:
                try:
                    asian_line = float(row[mapping.asian_line])
                    uw = float(row[mapping.upper_water])
                    lw = float(row[mapping.lower_water])
                    event["asian_line"] = asian_line
                    event["upper_water"] = uw
                    event["lower_water"] = lw

                    # Compute asian label
                    ug, lg = get_upper_lower_goals(hg, ag, upper_side)
                    if asian_label_mode == "5class":
                        asian_result = settle_asian_5class(ug, lg, asian_line)
                    else:
                        asian_result = settle_asian_3class(ug, lg, asian_line)
                except (ValueError, KeyError):
                    report["errors"]["invalid_odds"] += 1
                    event["asian_line"] = 0.0
                    event["upper_water"] = 1.0
                    event["lower_water"] = 1.0
                    has_asian = False

            if not has_asian:
                if not allow_missing_asian:
                    report["errors"]["missing_asian_fields"] += 1
                    report["skipped_rows"] += 1
                    continue
                event["asian_line"] = 0.0
                event["upper_water"] = 1.0
                event["lower_water"] = 1.0
                asian_result = "push"

            # Build sample
            match_id = build_match_id(league_id, season, kickoff, home, away)
            sample = {
                "match_id": match_id,
                "league_id": league_id,
                "bookmaker_id": bookmaker_id,
                "kickoff_time": kickoff,
                "odds_timeline": [event],
                "label": {
                    "euro_result": euro_result,
                    "asian_result": asian_result,
                    "home_goals": hg,
                    "away_goals": ag,
                    "upper_side": upper_side,
                },
            }

            # Validate
            errs = validate_imported_match(sample)
            if errs:
                report["errors"]["validation_failed"] += 1
                report["skipped_rows"] += 1
                continue

            samples.append(sample)
            report["imported_matches"] += 1

            # Track distributions
            report["label_distribution"]["euro_result"][euro_result] = \
                report["label_distribution"]["euro_result"].get(euro_result, 0) + 1
            report["label_distribution"]["asian_result"][asian_result] = \
                report["label_distribution"]["asian_result"].get(asian_result, 0) + 1

            if len(report["sample_preview"]) < 5:
                report["sample_preview"].append({
                    "match_id": match_id,
                    "kickoff_time": kickoff,
                    "home_team": home,
                    "away_team": away,
                })

    # Time range
    if samples:
        kicks = [s["kickoff_time"] for s in samples]
        report["time_range"]["min_kickoff_time"] = min(kicks)
        report["time_range"]["max_kickoff_time"] = max(kicks)

    return {"samples": samples, "report": report}
