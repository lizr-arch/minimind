"""
Football-Data CSV Importer (P0.8 / P0.9R)

Converts Football-Data.co.uk style CSV rows to OddsMind JSONL samples.
P0.9R adds open+closing dual-timeline support.

Handles:
  - Euro odds (open B365H/D/A, closing B365CH/CD/CA)
  - Asian handicap (open AHh/B365AHH/B365AHA, closing AHCh/B365CAHH/B365CAHA)
  - Match results (FTR, FTHG, FTAG)
  - Builds match_id, kickoff_time, dual-event odds_timeline, label
"""

import csv
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dataset.odds_import_schema import (
    build_match_id,
    normalize_team_name,
    validate_imported_match,
)
from dataset.importers.field_mapping import FieldMapping
from dataset.asian_handicap import settle_asian_5class, settle_asian_3class, get_upper_lower_goals

FTR_MAP = {"H": "home", "D": "draw", "A": "away"}


def _safe_float(val):
    """Convert string to float, returning None on failure."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


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
    Import a Football-Data style CSV with open + closing odds.

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
            "validation_failed": 0,
        },
        "label_distribution": {"euro_result": {}, "asian_result": {}},
        "time_range": {"min_kickoff_time": "", "max_kickoff_time": ""},
        "sample_preview": [],
        "timeline_type": "single",
    }

    required_euro = [mapping.euro_h, mapping.euro_d, mapping.euro_a]
    required_match = [mapping.date, mapping.home_team, mapping.away_team,
                      mapping.home_goals, mapping.away_goals]
    open_asian_fields = [mapping.asian_line, mapping.upper_water, mapping.lower_water]
    has_closing_cols = bool(mapping.closing_euro_h)

    if has_closing_cols:
        report["timeline_type"] = "open+closing"

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

            # Parse date/time (football-data format: DD/MM/YYYY)
            date_str = row[mapping.date].strip()
            time_str = (row.get(mapping.time) or "15:00").strip()
            # Normalize to ISO format
            parts = date_str.split("/")
            if len(parts) == 3:
                date_str = f"{parts[2]}-{parts[1]}-{parts[0]}"
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

            # Euro result
            ftr = row.get(mapping.ftr, "")
            euro_result = FTR_MAP.get(ftr.strip().upper(), "")
            if not euro_result:
                if hg > ag:
                    euro_result = "home"
                elif hg < ag:
                    euro_result = "away"
                else:
                    euro_result = "draw"

            # ── Build timeline ──
            odds_timeline = []

            # Open event
            euro_h = _safe_float(row[mapping.euro_h])
            euro_d = _safe_float(row[mapping.euro_d])
            euro_a = _safe_float(row[mapping.euro_a])
            if euro_h is None or euro_d is None or euro_a is None:
                report["errors"]["invalid_odds"] += 1
                report["skipped_rows"] += 1
                continue

            open_ev = {
                "minutes_before_kickoff": 10080,
                "euro_h": euro_h,
                "euro_d": euro_d,
                "euro_a": euro_a,
            }
            has_open_asian = all(row.get(f) for f in open_asian_fields)
            if has_open_asian:
                al = _safe_float(row[mapping.asian_line])
                uw = _safe_float(row[mapping.upper_water])
                lw = _safe_float(row[mapping.lower_water])
                if al is not None and uw is not None and lw is not None:
                    open_ev["asian_line"] = al
                    open_ev["upper_water"] = uw
                    open_ev["lower_water"] = lw
            open_ev.setdefault("asian_line", 0.0)
            open_ev.setdefault("upper_water", 1.0)
            open_ev.setdefault("lower_water", 1.0)
            odds_timeline.append(open_ev)

            # Closing event (if columns exist)
            has_asian = has_open_asian
            calc_asian_line = open_ev["asian_line"]

            if has_closing_cols:
                ch = _safe_float(row.get(mapping.closing_euro_h, ""))
                cd = _safe_float(row.get(mapping.closing_euro_d, ""))
                ca = _safe_float(row.get(mapping.closing_euro_a, ""))
                if ch and cd and ca:
                    close_ev = {
                        "minutes_before_kickoff": 0,
                        "euro_h": ch,
                        "euro_d": cd,
                        "euro_a": ca,
                    }
                    has_close_asian = all(row.get(f) for f in [
                        mapping.closing_asian_line, mapping.closing_upper_water, mapping.closing_lower_water])
                    if has_close_asian:
                        al2 = _safe_float(row[mapping.closing_asian_line])
                        uw2 = _safe_float(row[mapping.closing_upper_water])
                        lw2 = _safe_float(row[mapping.closing_lower_water])
                        if al2 is not None and uw2 is not None and lw2 is not None:
                            close_ev["asian_line"] = al2
                            close_ev["upper_water"] = uw2
                            close_ev["lower_water"] = lw2
                            calc_asian_line = al2
                            has_asian = True
                    close_ev.setdefault("asian_line", 0.0)
                    close_ev.setdefault("upper_water", 1.0)
                    close_ev.setdefault("lower_water", 1.0)
                    odds_timeline.append(close_ev)
                    # If open event had same values as closing, remove duplicate
                    if len(odds_timeline) == 2 and odds_timeline[0]["euro_h"] == odds_timeline[1]["euro_h"]:
                        pass  # keep both, they represent different time snapshots

            # Compute asian label
            asian_result = "push"
            if has_asian:
                ug, lg = get_upper_lower_goals(hg, ag, upper_side)
                if asian_label_mode == "5class":
                    asian_result = settle_asian_5class(ug, lg, calc_asian_line)
                else:
                    asian_result = settle_asian_3class(ug, lg, calc_asian_line)
            else:
                if not allow_missing_asian:
                    report["errors"]["missing_asian_fields"] += 1
                    report["skipped_rows"] += 1
                    continue

            # Build sample
            match_id = build_match_id(league_id, season, kickoff, home, away)
            sample = {
                "match_id": match_id,
                "league_id": league_id,
                "bookmaker_id": bookmaker_id,
                "kickoff_time": kickoff,
                "odds_timeline": odds_timeline,
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

    if samples:
        kicks = [s["kickoff_time"] for s in samples]
        report["time_range"]["min_kickoff_time"] = min(kicks)
        report["time_range"]["max_kickoff_time"] = max(kicks)

    return {"samples": samples, "report": report}
