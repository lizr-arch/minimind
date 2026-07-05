import csv
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.p7_draw_segments import (
    REQUIRED_SEGMENT_FIELDS,
    bin_draw_regime,
    build_draw_segment_report,
    write_draw_segment_outputs,
)


def _row(match_id, league_id, label, odds):
    return {
        "match_id": match_id,
        "league_id": league_id,
        "odds_timeline": [
            {
                "minutes_before_kickoff": 90.0,
                "has_euro": True,
                "euro_h": odds[0],
                "euro_d": odds[1],
                "euro_a": odds[2],
                "has_asian": True,
                "has_over_under": False,
            }
        ],
        "label": {"euro_result": label, "home_goals": 1, "away_goals": 1},
    }


def test_draw_segment_schema_required_fields():
    rows = [
        _row("m1", "league_a", "draw", (2.4, 3.1, 3.0)),
        _row("m2", "league_a", "home", (1.7, 3.4, 5.0)),
    ]
    predictions = [
        {"match_id": "m1", "y_true": 1, "p_home": 0.40, "p_draw": 0.30, "p_away": 0.30},
        {"match_id": "m2", "y_true": 0, "p_home": 0.55, "p_draw": 0.25, "p_away": 0.20},
    ]

    report = build_draw_segment_report(rows, predictions, run_id="unit")

    assert report["run_id"] == "unit"
    assert report["segments"]
    assert set(report["segments"][0]) >= REQUIRED_SEGMENT_FIELDS


def test_draw_segment_bins_are_deterministic():
    row = _row("m1", "league_a", "draw", (1.8, 3.2, 4.8))

    first = bin_draw_regime(row)
    second = bin_draw_regime(row)

    assert first == second
    assert set(first) >= {
        "league",
        "favorite_strength_bin",
        "implied_draw_prior_bin",
        "home_away_edge_bin",
        "odds_entropy_bin",
        "time_coverage_bin",
        "event_count_bin",
        "earliest_event_minutes_bin",
        "latest_event_minutes_bin",
        "odds_regime_bin",
    }


def test_write_draw_segment_outputs(tmp_path):
    report = {
        "run_id": "unit",
        "segments": [
            {
                field: 0 for field in REQUIRED_SEGMENT_FIELDS if field not in {"segment_type", "segment_value"}
            }
            | {"segment_type": "league", "segment_value": "league_a"}
        ],
    }

    outputs = write_draw_segment_outputs(report, tmp_path)

    assert outputs["json"].exists()
    assert outputs["csv"].exists()
    assert outputs["markdown"].exists()
    assert json.loads(outputs["json"].read_text(encoding="utf-8"))["run_id"] == "unit"
    with outputs["csv"].open("r", encoding="utf-8", newline="") as f:
        assert next(csv.DictReader(f))["segment_type"] == "league"
