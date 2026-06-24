"""Tests for P1.0G odds normalization / v4 feature schema."""
import sys
import os
import json
import tempfile
import math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
from dataset.odds_dataset import (
    OddsDataset,
    safe_inverse_odds,
    safe_implied_probs,
    safe_novig_probs,
    safe_overround,
    safe_2way_implied,
    safe_2way_novig,
    safe_overround_2way,
    _event_to_features_v4,
    _event_to_missing_mask_v4,
    V4_FEATURE_DIM,
    V4_FEATURE_NAMES,
)


# ── 15. safe_inverse_odds normal odds ────────────────────────────────

def test_safe_inverse_odds_normal():
    """Normal odds > 1 return correct inverse."""
    assert safe_inverse_odds(2.0) == 0.5
    assert abs(safe_inverse_odds(3.5) - (1.0 / 3.5)) < 1e-9
    assert safe_inverse_odds(1.01) == 1.0 / 1.01


# ── 16. safe_inverse_odds missing / 0 / <=1 ─────────────────────────

def test_safe_inverse_odds_zero():
    assert safe_inverse_odds(0) == 0.0

def test_safe_inverse_odds_negative():
    assert safe_inverse_odds(-1.0) == 0.0

def test_safe_inverse_odds_one():
    assert safe_inverse_odds(1.0) == 0.0

def test_safe_inverse_odds_no_nan_inf():
    """No input should produce nan or inf."""
    for x in [0, -1, 0.5, 1.0, 1.01, 100.0]:
        result = safe_inverse_odds(x)
        assert math.isfinite(result), f"safe_inverse_odds({x}) = {result}"


# ── 17. safe_novig_probs normal 3-way ────────────────────────────────

def test_safe_novig_probs_normal():
    """Normal odds produce probabilities summing to ~1."""
    h, d, a = safe_novig_probs(2.0, 3.5, 4.0)
    assert abs(h + d + a - 1.0) < 1e-9
    assert h > 0 and d > 0 and a > 0


# ── 18. safe_novig_probs overround=0 / missing odds ──────────────────

def test_safe_novig_probs_all_missing():
    """All odds <= 1 returns uniform distribution."""
    h, d, a = safe_novig_probs(0, 0, 0)
    assert abs(h - 1.0 / 3) < 1e-9
    assert abs(d - 1.0 / 3) < 1e-9
    assert abs(a - 1.0 / 3) < 1e-9

def test_safe_novig_probs_no_nan_inf():
    """No input combination produces nan or inf."""
    test_cases = [
        (0, 0, 0), (1.0, 1.0, 1.0), (0.5, 0.5, 0.5),
        (2.0, 0, 4.0), (1.01, 3.5, 4.0),
    ]
    for h, d, a in test_cases:
        result = safe_novig_probs(h, d, a)
        for v in result:
            assert math.isfinite(v), f"safe_novig_probs({h},{d},{a}) produced {result}"


# ── 19. v4 feature all finite ────────────────────────────────────────

def test_v4_features_all_finite():
    """v4 features from default event must all be finite."""
    event = {
        "minutes_before_kickoff": 60,
        "euro_h": 2.0, "euro_d": 3.5, "euro_a": 4.0,
        "asian_line": 0.0, "upper_water": 1.0, "lower_water": 1.0,
        "over_under_line": 2.5, "over_water": 1.0, "under_water": 1.0,
    }
    feats = _event_to_features_v4(event)
    for i, v in enumerate(feats):
        assert math.isfinite(v), f"v4 feature[{i}] ({V4_FEATURE_NAMES[i]}) = {v} is not finite"

def test_v4_features_real_data():
    """v4 features from realistic event must all be finite."""
    event = {
        "minutes_before_kickoff": 120,
        "euro_h": 1.95, "euro_d": 3.6, "euro_a": 3.8,
        "asian_line": -0.5, "upper_water": 0.85, "lower_water": 1.05,
        "over_under_line": 2.5, "over_water": 0.9, "under_water": 1.1,
    }
    feats = _event_to_features_v4(event)
    for i, v in enumerate(feats):
        assert math.isfinite(v), f"v4 feature[{i}] ({V4_FEATURE_NAMES[i]}) = {v} is not finite"


# ── 20. v4 missing_mask dim matches feature_dim ─────────────────────

def test_v4_missing_mask_dim():
    """v4 missing_mask must have 32 elements matching V4_FEATURE_DIM."""
    event = {
        "minutes_before_kickoff": 60,
        "euro_h": 2.0, "euro_d": 3.5, "euro_a": 4.0,
        "asian_line": 0.0, "upper_water": 1.0, "lower_water": 1.0,
        "over_under_line": 2.5, "over_water": 1.0, "under_water": 1.0,
    }
    mask = _event_to_missing_mask_v4(event)
    assert len(mask) == V4_FEATURE_DIM
    assert len(mask) == 32


# ── 21. v4 feature_dim stable ────────────────────────────────────────

def test_v4_feature_dim():
    """V4_FEATURE_DIM must be 32."""
    assert V4_FEATURE_DIM == 32
    assert len(V4_FEATURE_NAMES) == 32


# ── 22. cutoff sample v4 features don't see future events ────────────

def test_v4_cutoff_sample_respects_cutoff():
    """v4 features from a cutoff sample must only reflect events at or before cutoff."""
    timeline = [
        {"minutes_before_kickoff": 120, "euro_h": 2.0, "euro_d": 3.5, "euro_a": 4.0,
         "asian_line": -0.25, "upper_water": 0.9, "lower_water": 1.0,
         "over_under_line": 2.5, "over_water": 0.9, "under_water": 1.0},
        {"minutes_before_kickoff": 60, "euro_h": 2.1, "euro_d": 3.4, "euro_a": 3.9,
         "asian_line": -0.5, "upper_water": 0.85, "lower_water": 1.05,
         "over_under_line": 2.5, "over_water": 0.88, "under_water": 1.12},
        {"minutes_before_kickoff": 30, "euro_h": 2.2, "euro_d": 3.3, "euro_a": 3.8,
         "asian_line": -0.75, "upper_water": 0.8, "lower_water": 1.1,
         "over_under_line": 3.0, "over_water": 0.85, "under_water": 1.15},
    ]
    # cutoff=60 should exclude the T-30 event
    from dataset.odds_cutoff import filter_timeline_by_cutoff
    filtered = filter_timeline_by_cutoff(timeline, 60)
    assert len(filtered) == 2
    for e in filtered:
        assert e["minutes_before_kickoff"] >= 60

    # Build v4 features from filtered timeline
    for e in filtered:
        feats = _event_to_features_v4(e)
        for v in feats:
            assert math.isfinite(v)


# ── Additional v4 tests ──────────────────────────────────────────────

def test_v4_overround_nonnegative_for_real_odds():
    """Overround should be non-negative for real odds > 1."""
    event = {
        "minutes_before_kickoff": 60,
        "euro_h": 2.0, "euro_d": 3.5, "euro_a": 4.0,
        "asian_line": -0.5, "upper_water": 0.9, "lower_water": 1.0,
        "over_under_line": 2.5, "over_water": 0.9, "under_water": 1.0,
    }
    feats = _event_to_features_v4(event)
    # euro_overround at index 7
    assert feats[7] >= 0, f"euro_overround = {feats[7]}"

def test_v4_has_euro_reflects_detection():
    """v4 has_euro feature should match _event_has_euro."""
    from dataset.odds_dataset import _event_has_euro
    # Real euro
    event_real = {"euro_h": 2.0, "euro_d": 3.5, "euro_a": 4.0,
                  "asian_line": 0.0, "upper_water": 1.0, "lower_water": 1.0,
                  "over_under_line": 2.5, "over_water": 1.0, "under_water": 1.0}
    feats = _event_to_features_v4(event_real)
    assert feats[11] == (1.0 if _event_has_euro(event_real) else 0.0)

def test_v4_dataset_item_schema():
    """Dataset with v4 schema produces items with correct feature dim."""
    data = {
        "match_id": "test_m1",
        "league_id": "epl",
        "bookmaker_id": "Bet365",
        "kickoff_time": "2023-04-10T15:00:00",
        "time_axis_status": "ok",
        "odds_timeline": [
            {"minutes_before_kickoff": 60, "euro_h": 2.0, "euro_d": 3.5, "euro_a": 4.0,
             "asian_line": 0.0, "upper_water": 1.0, "lower_water": 1.0,
             "over_under_line": 2.5, "over_water": 1.0, "under_water": 1.0},
        ],
        "label": {"euro_result": "home", "asian_result": "full_win", "home_goals": 2, "away_goals": 1},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(data) + "\n")
        tmp_path = f.name
    try:
        ds = OddsDataset(tmp_path, feature_schema_version="v4", cutoff_mode="none")
        item = ds[0]
        assert item["features"].shape[-1] == 32
        assert "missing_mask" in item
        assert item["missing_mask"].shape[-1] == 32
    finally:
        os.unlink(tmp_path)
