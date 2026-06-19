"""
OddsMind Import Schema & Validation (P0.8)

Validates imported match samples against the OddsMind canonical format.
All functions return lists of error strings (empty = valid).
"""

from datetime import datetime
from typing import List, Optional

REQUIRED_OUTPUT_FIELDS = [
    "match_id", "league_id", "bookmaker_id", "kickoff_time",
    "odds_timeline", "label",
]

VALID_EURO_RESULTS = {"home", "draw", "away"}
VALID_ASIAN_5CLASS = {"upper_full_win", "upper_half_win", "push",
                      "upper_half_loss", "upper_full_loss"}
VALID_ASIAN_3CLASS = {"upper", "push", "lower"}
VALID_ASIAN_LINES = {round(i * 0.25, 2) for i in range(-20, 21)}  # -5.0 to +5.0


def validate_odds_event(event: dict) -> List[str]:
    """Validate a single odds event dict. Returns list of error strings."""
    errors = []
    for f in ["euro_h", "euro_d", "euro_a"]:
        v = event.get(f)
        if v is None or not isinstance(v, (int, float)) or v <= 0:
            errors.append(f"Invalid {f}: {v}")

    for f in ["upper_water", "lower_water"]:
        v = event.get(f)
        if v is None or not isinstance(v, (int, float)) or v <= 0:
            errors.append(f"Invalid {f}: {v}")

    al = event.get("asian_line")
    if al is None or not isinstance(al, (int, float)):
        errors.append(f"Missing asian_line")
    elif round(al * 4) / 4 not in VALID_ASIAN_LINES:
        errors.append(f"asian_line {al} not a valid 0.25 multiple")

    mbk = event.get("minutes_before_kickoff")
    if mbk is None or not isinstance(mbk, (int, float)) or mbk < 0:
        errors.append(f"minutes_before_kickoff must be >= 0, got {mbk}")

    return errors


def validate_imported_match(sample: dict) -> List[str]:
    """Validate a full match sample dict. Returns list of error strings."""
    errors = []
    for f in REQUIRED_OUTPUT_FIELDS:
        if f not in sample:
            errors.append(f"Missing required field: {f}")

    timeline = sample.get("odds_timeline", [])
    if not timeline:
        errors.append("odds_timeline is empty")
    else:
        for i, event in enumerate(timeline):
            ev_errors = validate_odds_event(event)
            for e in ev_errors:
                errors.append(f"event[{i}]: {e}")

        # Check sort order (descending)
        mbks = [e["minutes_before_kickoff"] for e in timeline]
        if mbks != sorted(mbks, reverse=True):
            errors.append("odds_timeline not sorted descending by minutes_before_kickoff")

    label = sample.get("label", {})
    er = label.get("euro_result")
    if er not in VALID_EURO_RESULTS:
        errors.append(f"Invalid euro_result: {er}")

    ar = label.get("asian_result")
    if ar not in VALID_ASIAN_5CLASS and ar not in VALID_ASIAN_3CLASS:
        errors.append(f"Invalid asian_result: {ar}")

    hg = label.get("home_goals")
    ag = label.get("away_goals")
    if hg is not None and (not isinstance(hg, int) or hg < 0):
        errors.append(f"Invalid home_goals: {hg}")
    if ag is not None and (not isinstance(ag, int) or ag < 0):
        errors.append(f"Invalid away_goals: {ag}")

    try:
        parse_kickoff_time(sample["kickoff_time"])
    except Exception as e:
        errors.append(f"Invalid kickoff_time: {e}")

    return errors


def parse_kickoff_time(value: str) -> datetime:
    """Parse ISO-8601 datetime, handling Z suffix."""
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def normalize_team_name(value: str) -> str:
    """Normalize team name: strip, title-case."""
    return value.strip().title()


def build_match_id(
    league_id: str,
    season: Optional[str],
    kickoff_time: str,
    home_team: str,
    away_team: str,
) -> str:
    """Build a canonical match_id."""
    dt = parse_kickoff_time(kickoff_time)
    date_str = dt.strftime("%Y-%m-%d")
    parts = [league_id]
    if season:
        parts.append(season.replace("/", "-"))
    parts.append(date_str)
    parts.append(home_team.replace(" ", ""))
    parts.append(away_team.replace(" ", ""))
    return "_".join(parts)
