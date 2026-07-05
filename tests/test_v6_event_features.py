"""Tests for Phase 1b Step 1: v6_event feature schema (31-dim per-event features)."""

import json
import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.odds_dataset import (
    OddsDataset,
    _event_to_features_v6,
    _event_to_missing_mask_v6,
    V6_EVENT_FEATURE_DIM,
    V6_EVENT_FEATURE_NAMES,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_v6_event(
    minutes=100.0,
    euro_h=2.0, euro_d=3.0, euro_a=4.0,
    asian_line=-0.5, upper_water=1.9, lower_water=1.9,
    ou_line=2.5, ou_over=1.9, ou_under=1.9,
    snapshot_type="movement",
    market_updated="1x2",
    euro_source="raw_update",
    has_euro=True, has_asian=True, has_over_under=True,
    **kwargs,
):
    """Create a single event dict with v6_event fields."""
    event = {
        "minutes_before_kickoff": minutes,
        "snapshot_type": snapshot_type,
        "market_updated": market_updated,
        "euro_h": euro_h, "euro_d": euro_d, "euro_a": euro_a,
        "has_euro": has_euro,
        "euro_source": euro_source,
        "asian_line": asian_line,
        "upper_water": upper_water, "lower_water": lower_water,
        "has_asian": has_asian,
        "asian_source": "raw_update" if has_asian else "missing",
        "over_under_line": ou_line if has_over_under else None,
        "over_water": ou_over if has_over_under else None,
        "under_water": ou_under if has_over_under else None,
        "has_over_under": has_over_under,
        "over_under_source": "raw_update" if has_over_under else "missing",
    }
    event.update(kwargs)
    return event


def _make_v6_sample(match_id="test_001", num_events=5, **kwargs):
    """Create a complete sample dict for v6_event testing."""
    timeline = []
    for i in range(num_events):
        minutes = 100.0 - i * 10.0  # descending
        timeline.append(_make_v6_event(minutes=minutes, **kwargs))

    return {
        "match_id": match_id,
        "league_id": "EPL",
        "bookmaker_id": "Bet365",
        "odds_timeline": timeline,
        "label": {
            "euro_result": "home",
            "asian_result": "upper",
            "home_goals": 2,
            "away_goals": 1,
        },
    }


def _tmp_jsonl(samples):
    """Write samples to a temporary JSONL file and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for s in samples:
        f.write(json.dumps(s) + "\n")
    f.close()
    return f.name


# ── T1: Feature dimension correctness ───────────────────────────────────

def test_v6_feature_dim():
    """V6_EVENT_FEATURE_DIM should be 31."""
    assert V6_EVENT_FEATURE_DIM == 31


def test_v6_feature_names_length():
    """V6_FEATURE_NAMES should have 31 entries."""
    assert len(V6_EVENT_FEATURE_NAMES) == 31


def test_v6_event_to_features_shape():
    """_event_to_features_v6 should return a list of 31 floats."""
    event = _make_v6_event()
    features = _event_to_features_v6(event, prev_event=None)
    assert len(features) == 31
    assert all(isinstance(f, float) for f in features)


def test_v6_dataset_item_shape():
    """OddsDataset with v6_event should output [T, 31] features."""
    sample = _make_v6_sample(num_events=10)
    path = _tmp_jsonl([sample])
    try:
        ds = OddsDataset(path, max_seq_len=64, feature_schema_version="v6_event")
        item = ds[0]
        assert item["features"].shape == (10, 31)  # 10 events, 31 features
        assert item["missing_mask"].shape == (10, 31)
    finally:
        os.unlink(path)


def test_v6_dataset_padded_shape():
    """OddsDataset with v6_event should pad to max_seq_len."""
    sample = _make_v6_sample(num_events=5)
    path = _tmp_jsonl([sample])
    try:
        ds = OddsDataset(path, max_seq_len=64, feature_schema_version="v6_event")
        item = ds[0]
        assert item["features"].shape == (5, 31)  # only 5 events, no padding in dataset
    finally:
        os.unlink(path)


# ── T2: Missing value handling ──────────────────────────────────────────

def test_v6_ou_missing_mask():
    """When OU is missing (ou_line=None, over_under_source='missing'), mask bits=1.0."""
    event = _make_v6_event(has_over_under=False, ou_line=None, ou_over=None, ou_under=None)
    # Force over_under_source to missing
    event["over_under_source"] = "missing"
    features = _event_to_features_v6(event, prev_event=None)
    mask = _event_to_missing_mask_v6(event)

    # has_ou should be 0.0
    assert features[26] == 0.0  # has_ou

    # OU features should be 0.0
    assert features[6] == 0.0   # ou_line
    assert features[7] == 0.0   # ou_over_water
    assert features[8] == 0.0   # ou_under_water

    # OU mask bits should be 1.0 (missing)
    assert mask[6] == 1.0
    assert mask[7] == 1.0
    assert mask[8] == 1.0


def test_v6_euro_missing_mask():
    """When Euro is missing (euro values <=1.0), has_euro=0.0 and mask bits=1.0."""
    event = _make_v6_event(euro_h=0.0, euro_d=0.0, euro_a=0.0)
    features = _event_to_features_v6(event, prev_event=None)
    mask = _event_to_missing_mask_v6(event)

    # has_euro should be 0.0
    assert features[24] == 0.0  # has_euro

    # Euro features should be 0.0
    assert features[0] == 0.0   # euro_h
    assert features[1] == 0.0   # euro_d
    assert features[2] == 0.0   # euro_a

    # Euro mask bits should be 1.0 (missing)
    assert mask[0] == 1.0
    assert mask[1] == 1.0
    assert mask[2] == 1.0


def test_v6_asian_missing_mask():
    """When Asian is missing (asian_source='missing'), has_asian=0.0 and mask bits=1.0."""
    event = _make_v6_event(has_asian=False, asian_line=0, upper_water=0, lower_water=0)
    # Force asian_source to missing
    event["asian_source"] = "missing"
    features = _event_to_features_v6(event, prev_event=None)
    mask = _event_to_missing_mask_v6(event)

    # has_asian should be 0.0
    assert features[25] == 0.0  # has_asian

    # Asian features should be 0.0
    assert features[3] == 0.0   # asian_line
    assert features[4] == 0.0   # upper_water
    assert features[5] == 0.0   # lower_water

    # Asian mask bits should be 1.0 (missing)
    assert mask[3] == 1.0
    assert mask[4] == 1.0
    assert mask[5] == 1.0


def test_v6_derived_features_no_mask():
    """Derived features (indices 9-30) should have mask=0.0."""
    event = _make_v6_event()
    mask = _event_to_missing_mask_v6(event)

    for i in range(9, 31):
        assert mask[i] == 0.0, f"mask[{i}] should be 0.0 for derived features"


# ── T3: Time delta calculation ──────────────────────────────────────────

def test_v6_time_delta_first_event():
    """First event (prev_event=None) should have time_delta_prev=0.0."""
    event = _make_v6_event(minutes=100.0)
    features = _event_to_features_v6(event, prev_event=None)
    assert features[18] == 0.0  # time_delta_prev


def test_v6_time_delta_calculation():
    """Time delta should be abs(minutes_curr - minutes_prev)."""
    event = _make_v6_event(minutes=80.0)
    prev_event = _make_v6_event(minutes=100.0)
    features = _event_to_features_v6(event, prev_event=prev_event)
    assert features[18] == 20.0  # abs(80 - 100) = 20


# ── T4: Backward compatibility ──────────────────────────────────────────

def test_v5_schema_unchanged():
    """Loading v6 data with v5 schema should still work (fallback)."""
    sample = _make_v6_sample(num_events=3)
    path = _tmp_jsonl([sample])
    try:
        ds = OddsDataset(path, max_seq_len=64, feature_schema_version="v5")
        item = ds[0]
        # v5 has 35 features
        assert item["features"].shape[1] == 35
    finally:
        os.unlink(path)


def test_v4_schema_unchanged():
    """Loading v6 data with v4 schema should still work (fallback)."""
    sample = _make_v6_sample(num_events=3)
    path = _tmp_jsonl([sample])
    try:
        ds = OddsDataset(path, max_seq_len=64, feature_schema_version="v4")
        item = ds[0]
        # v4 has 32 features
        assert item["features"].shape[1] == 32
    finally:
        os.unlink(path)


# ── T5: Sequence truncation ─────────────────────────────────────────────

def test_v6_sequence_truncation():
    """With max_seq_len=10, only the last 10 events should be kept."""
    sample = _make_v6_sample(num_events=50)
    path = _tmp_jsonl([sample])
    try:
        ds = OddsDataset(path, max_seq_len=10, feature_schema_version="v6_event")
        item = ds[0]
        assert item["features"].shape[0] == 10
    finally:
        os.unlink(path)


# ── Additional: Feature value correctness ───────────────────────────────

def test_v6_snapshot_is_opening():
    """snapshot_is_opening should be 1.0 for opening events, 0.0 otherwise."""
    event_opening = _make_v6_event(snapshot_type="opening")
    event_movement = _make_v6_event(snapshot_type="movement")

    feat_opening = _event_to_features_v6(event_opening, prev_event=None)
    feat_movement = _event_to_features_v6(event_movement, prev_event=None)

    assert feat_opening[19] == 1.0   # snapshot_is_opening
    assert feat_movement[19] == 0.0


def test_v6_market_flags():
    """Market flags should be set correctly based on market_updated."""
    event_euro = _make_v6_event(market_updated="1x2")
    event_asian = _make_v6_event(market_updated="asian")
    event_ou = _make_v6_event(market_updated="over_under")

    feat_euro = _event_to_features_v6(event_euro, prev_event=None)
    feat_asian = _event_to_features_v6(event_asian, prev_event=None)
    feat_ou = _event_to_features_v6(event_ou, prev_event=None)

    assert feat_euro[20] == 1.0 and feat_euro[21] == 0.0 and feat_euro[22] == 0.0
    assert feat_asian[20] == 0.0 and feat_asian[21] == 1.0 and feat_asian[22] == 0.0
    assert feat_ou[20] == 0.0 and feat_ou[21] == 0.0 and feat_ou[22] == 1.0


def test_v6_source_is_raw():
    """source_is_raw should be 1.0 only when euro_source='raw_update'."""
    event_raw = _make_v6_event(euro_source="raw_update")
    event_fill = _make_v6_event(euro_source="forward_fill")

    feat_raw = _event_to_features_v6(event_raw, prev_event=None)
    feat_fill = _event_to_features_v6(event_fill, prev_event=None)

    assert feat_raw[23] == 1.0
    assert feat_fill[23] == 0.0


def test_v6_change_features_first_event():
    """First event should have all change features = 0.0."""
    event = _make_v6_event()
    features = _event_to_features_v6(event, prev_event=None)

    assert features[27] == 0.0  # euro_h_change
    assert features[28] == 0.0  # asian_line_change
    assert features[29] == 0.0  # upper_water_change
    assert features[30] == 0.0  # ou_line_change


def test_v6_change_features_calculation():
    """Change features should be curr - prev."""
    event = _make_v6_event(euro_h=1.8, asian_line=-0.75, upper_water=2.0, ou_line=2.5)
    prev_event = _make_v6_event(euro_h=2.0, asian_line=-0.5, upper_water=1.9, ou_line=2.25)

    features = _event_to_features_v6(event, prev_event=prev_event)

    assert features[27] == pytest.approx(-0.2)   # 1.8 - 2.0
    assert features[28] == pytest.approx(-0.25)  # -0.75 - (-0.5)
    assert features[29] == pytest.approx(0.1)    # 2.0 - 1.9
    assert features[30] == pytest.approx(0.25)   # 2.5 - 2.25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
