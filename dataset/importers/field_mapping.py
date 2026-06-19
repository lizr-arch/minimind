"""
OddsMind Field Mapping (P0.8)

Maps CSV column names to OddsMind canonical fields.
Supports JSON-based mapping files and built-in presets.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class FieldMapping:
    """Maps source column names to OddsMind fields."""
    league_id: str = ""
    date: str = ""
    time: str = ""
    home_team: str = ""
    away_team: str = ""
    home_goals: str = ""
    away_goals: str = ""
    euro_h: str = ""
    euro_d: str = ""
    euro_a: str = ""
    asian_line: str = ""
    upper_water: str = ""
    lower_water: str = ""
    season: str = ""
    ftr: str = ""  # full-time result string e.g. "H"/"D"/"A"
    # P0.9R closing odds (for dual-timeline open+closing)
    closing_euro_h: str = ""
    closing_euro_d: str = ""
    closing_euro_a: str = ""
    closing_asian_line: str = ""
    closing_upper_water: str = ""
    closing_lower_water: str = ""


# Built-in presets
PRESETS: Dict[str, FieldMapping] = {
    "football_data_b365": FieldMapping(
        league_id="Div",
        date="Date",
        time="Time",
        home_team="HomeTeam",
        away_team="AwayTeam",
        home_goals="FTHG",
        away_goals="FTAG",
        euro_h="B365H",
        euro_d="B365D",
        euro_a="B365A",
        asian_line="AHh",
        upper_water="B365AHH",
        lower_water="B365AHA",
        ftr="FTR",
        # P0.9R closing odds extensions
        closing_euro_h="B365CH",
        closing_euro_d="B365CD",
        closing_euro_a="B365CA",
        closing_asian_line="AHCh",
        closing_upper_water="B365CAHH",
        closing_lower_water="B365CAHA",
    ),
    "generic_single_snapshot": FieldMapping(
        league_id="league",
        date="date",
        time="time",
        home_team="home",
        away_team="away",
        home_goals="home_goals",
        away_goals="away_goals",
        euro_h="euro_h",
        euro_d="euro_d",
        euro_a="euro_a",
        asian_line="asian_line",
        upper_water="upper_water",
        lower_water="lower_water",
    ),
}


def load_field_mapping(path: Optional[str] = None, preset: str = "") -> FieldMapping:
    """
    Load a field mapping from a preset or JSON file.

    Args:
        path: path to a JSON mapping file.
        preset: name of a built-in preset.
    """
    if path:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return FieldMapping(**{k: data.get(k, "") for k in FieldMapping.__dataclass_fields__})
    if preset and preset in PRESETS:
        return PRESETS[preset]
    raise ValueError(f"Unknown preset: {preset}. Use one of: {list(PRESETS.keys())}")
