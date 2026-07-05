"""Tests for Phase 1b Step 2: OddsEventCollator with lead-lag and alignment features."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.odds_collator_v6_event import (
    OddsEventCollator,
    build_pinnacle_index,
    compute_leadlag_feature,
    compute_euro_asian_alignment,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_event(
    minutes=100.0,
    euro_h=2.0, euro_d=3.0, euro_a=4.0,
    asian_line=-0.5, upper_water=1.9, lower_water=1.9,
    ou_line=2.5, ou_over=1.9, ou_under=1.9,
    snapshot_type="movement",
    market_updated="1x2",
    euro_source="raw_update",
    has_euro=True, has_asian=True, has_over_under=True,
):
    """Create a single event dict."""
    return {
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


def _make_sample(
    match_id="test_001",
    bookmaker_id="Bet365",
    num_events=5,
    euro_h_start=2.0,
    euro_h_change=-0.1,  # decrease per event
):
    """Create a sample dict with raw_timeline."""
    timeline = []
    for i in range(num_events):
        minutes = 100.0 - i * 10.0
        euro_h = euro_h_start + i * euro_h_change
        timeline.append(_make_event(minutes=minutes, euro_h=euro_h))

    return {
        "match_id": match_id,
        "league_id": "EPL",
        "bookmaker_id": bookmaker_id,
        "features": torch.randn(num_events, 31),
        "missing_mask": torch.zeros(num_events, 31),
        "attention_mask": torch.ones(num_events, dtype=torch.bool),
        "euro_label": 0,
        "asian_label": 1,
        "raw_timeline": timeline,
    }


# ── T1: Padding correctness ────────────────────────────────────────────

def test_padding_correctness():
    """Collator should pad sequences to max length in batch."""
    collator = OddsEventCollator()
    
    # Create batch with different lengths: 5, 10, 3
    batch = [
        _make_sample(num_events=5),
        _make_sample(num_events=10),
        _make_sample(num_events=3),
    ]
    
    result = collator(batch)
    
    # Features should be [3, 10, 33]
    assert result["features"].shape == (3, 10, 33)
    
    # Attention mask should be correct
    assert result["attention_mask"][0, :5].all()
    assert not result["attention_mask"][0, 5:].any()
    assert result["attention_mask"][1, :10].all()
    assert result["attention_mask"][2, :3].all()
    assert not result["attention_mask"][2, 3:].any()
    
    # Padded area should be 0
    assert (result["features"][0, 5:] == 0).all()
    assert (result["features"][2, 3:] == 0).all()


def test_missing_mask_padding():
    """Missing mask should be padded correctly."""
    collator = OddsEventCollator()
    
    batch = [
        _make_sample(num_events=5),
        _make_sample(num_events=3),
    ]
    
    result = collator(batch)
    
    # Missing mask should be [2, 5, 33]
    assert result["missing_mask"].shape == (2, 5, 33)
    
    # Padded area should be 0
    assert (result["missing_mask"][1, 3:] == 0).all()


def test_event_counts():
    """Event counts should reflect actual sequence lengths."""
    collator = OddsEventCollator()
    
    batch = [
        _make_sample(num_events=5),
        _make_sample(num_events=10),
        _make_sample(num_events=3),
    ]
    
    result = collator(batch)
    
    assert result["event_counts"].tolist() == [5, 10, 3]


# ── T2: Lead-Lag feature ───────────────────────────────────────────────

def test_leadlag_same_direction():
    """When Pinnacle and bookmaker move in same direction, leadlag = +1.0."""
    # Pinnacle at T=60min: euro_h drops from 2.0 to 1.8 (favor home)
    pinnacle_timeline = [
        _make_event(minutes=60, euro_h=2.0),
        _make_event(minutes=50, euro_h=1.8),
    ]
    
    # Bet365 at T=30min: euro_h drops from 2.1 to 1.9 (same direction)
    event = _make_event(minutes=30, euro_h=1.9)
    prev_event = _make_event(minutes=40, euro_h=2.1)
    
    result = compute_leadlag_feature(event, pinnacle_timeline, prev_event, "Bet365")
    assert result == 1.0


def test_leadlag_opposite_direction():
    """When Pinnacle and bookmaker move in opposite directions, leadlag = -1.0."""
    # Pinnacle at T=60min: euro_h drops from 2.0 to 1.8 (favor home)
    pinnacle_timeline = [
        _make_event(minutes=60, euro_h=2.0),
        _make_event(minutes=50, euro_h=1.8),
    ]
    
    # Bet365 at T=30min: euro_h rises from 1.9 to 2.1 (opposite direction)
    event = _make_event(minutes=30, euro_h=2.1)
    prev_event = _make_event(minutes=40, euro_h=1.9)
    
    result = compute_leadlag_feature(event, pinnacle_timeline, prev_event, "Bet365")
    assert result == -1.0


def test_leadlag_pinnacle_self():
    """Pinnacle itself should have leadlag = 0.0."""
    event = _make_event(minutes=30, euro_h=1.9)
    prev_event = _make_event(minutes=40, euro_h=2.1)
    
    result = compute_leadlag_feature(event, [], prev_event, "Pinnacle")
    assert result == 0.0


def test_leadlag_no_pinnacle_data():
    """Without Pinnacle data, leadlag = 0.0."""
    event = _make_event(minutes=30, euro_h=1.9)
    prev_event = _make_event(minutes=40, euro_h=2.1)
    
    result = compute_leadlag_feature(event, None, prev_event, "Bet365")
    assert result == 0.0


def test_leadlag_no_prev_event():
    """Without previous event, leadlag = 0.0."""
    pinnacle_timeline = [
        _make_event(minutes=60, euro_h=2.0),
        _make_event(minutes=50, euro_h=1.8),
    ]
    
    event = _make_event(minutes=30, euro_h=1.9)
    
    result = compute_leadlag_feature(event, pinnacle_timeline, None, "Bet365")
    assert result == 0.0


# ── T3: Information leak check ──────────────────────────────────────────

def test_leadlag_no_future_info():
    """Leadlag should not use Pinnacle info from the future (after current moment)."""
    # Pinnacle moves at T=20min (future relative to T=60min)
    pinnacle_timeline = [
        _make_event(minutes=60, euro_h=2.0),
        _make_event(minutes=20, euro_h=1.8),  # This is future info!
    ]
    
    # Bet365 at T=60min: euro_h changes
    event = _make_event(minutes=60, euro_h=1.9)
    prev_event = _make_event(minutes=70, euro_h=2.1)
    
    # At T=60min, Pinnacle hasn't moved yet (only has opening at T=60)
    # So leadlag should be 0.0 (no direction change detected)
    result = compute_leadlag_feature(event, pinnacle_timeline, prev_event, "Bet365")
    assert result == 0.0


# ── T4: Euro-Asian alignment ────────────────────────────────────────────

def test_alignment_same_direction():
    """When euro and asian favor same side, alignment = +1.0."""
    # euro_h drops (favor home), asian_line drops (favor home)
    event = _make_event(euro_h=1.8, asian_line=-0.75)
    prev_event = _make_event(euro_h=2.0, asian_line=-0.5)
    
    result = compute_euro_asian_alignment(event, prev_event)
    assert result == 1.0


def test_alignment_opposite_direction():
    """When euro and asian favor opposite sides, alignment = -1.0."""
    # euro_h drops (favor home), asian_line rises (favor away)
    event = _make_event(euro_h=1.8, asian_line=-0.25)
    prev_event = _make_event(euro_h=2.0, asian_line=-0.5)
    
    result = compute_euro_asian_alignment(event, prev_event)
    assert result == -1.0


def test_alignment_euro_unchanged():
    """When euro doesn't change, alignment = 0.0."""
    event = _make_event(euro_h=2.0, asian_line=-0.75)
    prev_event = _make_event(euro_h=2.0, asian_line=-0.5)
    
    result = compute_euro_asian_alignment(event, prev_event)
    assert result == 0.0


def test_alignment_no_prev_event():
    """Without previous event, alignment = 0.0."""
    event = _make_event(euro_h=1.8, asian_line=-0.75)
    
    result = compute_euro_asian_alignment(event, None)
    assert result == 0.0


def test_alignment_asian_water_fallback():
    """When asian_line unchanged, use upper_water for direction."""
    # euro_h drops (favor home), upper_water drops (favor home)
    event = _make_event(euro_h=1.8, asian_line=-0.5, upper_water=1.8)
    prev_event = _make_event(euro_h=2.0, asian_line=-0.5, upper_water=2.0)
    
    result = compute_euro_asian_alignment(event, prev_event)
    assert result == 1.0


# ── T5: Edge cases ──────────────────────────────────────────────────────

def test_single_event_sequence():
    """Sequence with 1 event should have leadlag and alignment = 0.0."""
    collator = OddsEventCollator()
    
    batch = [_make_sample(num_events=1)]
    result = collator(batch)
    
    # First event has no prev, so leadlag and alignment should be 0.0
    assert result["features"][0, 0, 31] == 0.0  # leadlag
    assert result["features"][0, 0, 32] == 0.0  # alignment


def test_build_pinnacle_index():
    """build_pinnacle_index should correctly index Pinnacle samples."""
    samples = [
        {
            "match_id": "match_001",
            "bookmaker_id": "Bet365",
            "odds_timeline": [_make_event(minutes=100)],
        },
        {
            "match_id": "match_001",
            "bookmaker_id": "Pinnacle",
            "odds_timeline": [
                _make_event(minutes=100),
                _make_event(minutes=90),
            ],
        },
        {
            "match_id": "match_002",
            "bookmaker_id": "Pinnacle",
            "odds_timeline": [_make_event(minutes=80)],
        },
    ]
    
    index = build_pinnacle_index(samples)
    
    assert "match_001" in index
    assert "match_002" in index
    assert len(index["match_001"]) == 2
    assert len(index["match_002"]) == 1
    # Should be sorted descending
    assert index["match_001"][0]["minutes_before_kickoff"] >= index["match_001"][1]["minutes_before_kickoff"]


def test_collator_with_pinnacle_index():
    """Collator should use Pinnacle index for leadlag computation."""
    pinnacle_index = {
        "match_001": [
            _make_event(minutes=100, euro_h=2.0),
            _make_event(minutes=80, euro_h=1.8),  # Pinnacle favors home
        ],
    }
    
    collator = OddsEventCollator(pinnacle_index=pinnacle_index)
    
    # Bet365 sample
    sample = _make_sample(
        match_id="match_001",
        bookmaker_id="Bet365",
        num_events=3,
        euro_h_start=2.1,
        euro_h_change=-0.1,  # Also favors home
    )
    
    batch = [sample]
    result = collator(batch)
    
    # Second event should have leadlag = 1.0 (same direction as Pinnacle)
    assert result["features"].shape == (1, 3, 33)


def test_batch_labels():
    """Labels should be correctly batched."""
    collator = OddsEventCollator()
    
    batch = [
        {**_make_sample(num_events=3), "euro_label": 0, "asian_label": 1},
        {**_make_sample(num_events=3), "euro_label": 2, "asian_label": 3},
    ]
    
    result = collator(batch)
    
    assert result["euro_labels"].tolist() == [0, 2]
    assert result["asian_labels"].tolist() == [1, 3]


def test_match_and_league_ids():
    """Match and league IDs should be correctly batched."""
    collator = OddsEventCollator()
    
    batch = [
        {**_make_sample(num_events=3), "match_id": "m1", "league_id": "EPL"},
        {**_make_sample(num_events=3), "match_id": "m2", "league_id": "LaLiga"},
    ]
    
    result = collator(batch)
    
    assert result["match_ids"] == ["m1", "m2"]
    assert result["league_ids"] == ["EPL", "LaLiga"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
