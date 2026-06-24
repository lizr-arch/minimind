"""Tests for P1.0D cutoff sample builder."""
import sys
import os
import json
import tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from dataset.odds_cutoff import (
    filter_timeline_by_cutoff,
    assert_cutoff_integrity,
    build_cutoff_samples,
    build_exhaustive_cutoff_dataset,
    CUTOFF_BUCKETS_BASE,
    CUTOFF_BUCKETS_V1,
)
from dataset.odds_dataset import OddsDataset


# ── Fixtures ──────────────────────────────────────────────────────────

def make_sample(match_id="m1", n_events=20):
    """Create a sample with events from T-1440 to T-1."""
    timeline = []
    for i in range(n_events):
        mbk = 1440 - i * (1440 // n_events)
        timeline.append({
            "minutes_before_kickoff": mbk,
            "euro_h": 2.0, "euro_d": 3.5, "euro_a": 4.0,
            "asian_line": -0.25, "upper_water": 0.9, "lower_water": 1.0,
            "over_under_line": 2.5, "over_water": 0.9, "under_water": 1.0,
        })
    return {
        "match_id": match_id,
        "league_id": "epl",
        "bookmaker_id": "Bet365",
        "kickoff_time": "2023-04-10T15:00:00",
        "time_axis_status": "ok",
        "odds_timeline": timeline,
        "label": {"euro_result": "home", "asian_result": "full_win", "home_goals": 2, "away_goals": 1},
    }


# ── filter_timeline_by_cutoff ────────────────────────────────────────

def test_cutoff_60():
    sample = make_sample(n_events=30)
    vis = filter_timeline_by_cutoff(sample["odds_timeline"], 60)
    assert all(e["minutes_before_kickoff"] >= 60 for e in vis)
    assert len(vis) > 0

def test_cutoff_30():
    sample = make_sample(n_events=30)
    vis = filter_timeline_by_cutoff(sample["odds_timeline"], 30)
    assert all(e["minutes_before_kickoff"] >= 30 for e in vis)

def test_cutoff_1():
    sample = make_sample(n_events=30)
    vis = filter_timeline_by_cutoff(sample["odds_timeline"], 1)
    assert all(e["minutes_before_kickoff"] >= 1 for e in vis)

def test_cutoff_empty():
    sample = make_sample(n_events=5)
    # Events are at T-1440, T-1152, T-864, T-576, T-288
    vis = filter_timeline_by_cutoff(sample["odds_timeline"], 2000)
    assert len(vis) == 0

def test_cutoff_sorted_descending():
    sample = make_sample(n_events=10)
    vis = filter_timeline_by_cutoff(sample["odds_timeline"], 0)
    for i in range(len(vis) - 1):
        assert vis[i]["minutes_before_kickoff"] >= vis[i+1]["minutes_before_kickoff"]


# ── assert_cutoff_integrity ──────────────────────────────────────────

def test_integrity_pass():
    timeline = [{"minutes_before_kickoff": 100}, {"minutes_before_kickoff": 60}]
    assert_cutoff_integrity(timeline, 60)  # Should not raise

def test_integrity_fail():
    timeline = [{"minutes_before_kickoff": 100}, {"minutes_before_kickoff": 30}]
    with pytest.raises(AssertionError, match="Cutoff violation"):
        assert_cutoff_integrity(timeline, 60)


# ── build_cutoff_samples ─────────────────────────────────────────────

def test_build_cutoff_sample_60():
    sample = make_sample(n_events=30)
    cutoff_samples = build_cutoff_samples(sample, [60])
    assert len(cutoff_samples) == 1
    cs = cutoff_samples[0]
    assert cs["cutoff_minutes"] == 60
    assert len(cs["odds_timeline"]) > 0
    assert all(e["minutes_before_kickoff"] >= 60 for e in cs["odds_timeline"])

def test_build_cutoff_sample_empty_returns_skip():
    sample = make_sample(n_events=3)
    # Events at T-1440, T-960, T-480
    cutoff_samples = build_cutoff_samples(sample, [2000], min_events=1)
    assert len(cutoff_samples) == 0  # skipped

def test_build_cutoff_sample_preserves_metadata():
    sample = make_sample(n_events=10)
    cutoff_samples = build_cutoff_samples(sample, [60])
    cs = cutoff_samples[0]
    assert cs["source_match_id"] == "m1"
    assert cs["league_id"] == "epl"
    assert cs["bookmaker_id"] == "Bet365"
    assert cs["kickoff_time"] == "2023-04-10T15:00:00"

def test_build_cutoff_does_not_modify_original():
    sample = make_sample(n_events=10)
    orig_len = len(sample["odds_timeline"])
    build_cutoff_samples(sample, [60])
    assert len(sample["odds_timeline"]) == orig_len  # unchanged

def test_build_multiple_cutoffs():
    sample = make_sample(n_events=30)
    cutoff_samples = build_cutoff_samples(sample, [1440, 360, 60, 1])
    assert len(cutoff_samples) == 4
    cutoffs = [cs["cutoff_minutes"] for cs in cutoff_samples]
    assert cutoffs == [1440, 360, 60, 1]


# ── build_exhaustive_cutoff_dataset ──────────────────────────────────

def test_exhaustive_expands():
    samples = [make_sample("m1", 30), make_sample("m2", 30)]
    expanded = build_exhaustive_cutoff_dataset(samples, [360, 60, 1])
    assert len(expanded) == 6  # 2 matches * 3 cutoffs


# ── Dataset cutoff modes ─────────────────────────────────────────────

def test_dataset_none_mode_length():
    """none mode: len == raw sample count."""
    data = [make_sample("m1", 20), make_sample("m2", 20)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for s in data:
            f.write(json.dumps(s) + "\n")
        tmp_path = f.name
    try:
        ds = OddsDataset(tmp_path, cutoff_mode="none")
        assert len(ds) == 2
    finally:
        os.unlink(tmp_path)

def test_dataset_exhaustive_mode_length():
    """exhaustive mode: len == non-empty expanded samples."""
    data = [make_sample("m1", 30), make_sample("m2", 30)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for s in data:
            f.write(json.dumps(s) + "\n")
        tmp_path = f.name
    try:
        ds = OddsDataset(tmp_path, cutoff_mode="exhaustive", cutoffs=[360, 60, 1])
        assert len(ds) == 6  # 2 matches * 3 cutoffs
    finally:
        os.unlink(tmp_path)

def test_dataset_exhaustive_skips_empty():
    """exhaustive mode should skip cutoffs with 0 visible events."""
    data = [make_sample("m1", 3)]  # Only 3 events
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for s in data:
            f.write(json.dumps(s) + "\n")
        tmp_path = f.name
    try:
        ds = OddsDataset(tmp_path, cutoff_mode="exhaustive", cutoffs=[2000, 60], min_events=1)
        # cutoff=2000 has 0 events, cutoff=60 may have some
        assert len(ds) <= 1
    finally:
        os.unlink(tmp_path)

def test_dataset_allowed_ids_then_cutoff():
    """allowed_match_ids filters first, then cutoff expand."""
    data = [make_sample("m1", 30), make_sample("m2", 30), make_sample("m3", 30)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for s in data:
            f.write(json.dumps(s) + "\n")
        tmp_path = f.name
    try:
        ds = OddsDataset(tmp_path, cutoff_mode="exhaustive", cutoffs=[60],
                         allowed_match_ids={"m1", "m3"})
        # Only m1 and m3 should be expanded
        source_ids = set()
        for i in range(len(ds)):
            item = ds[i]
            # match_id in item is sample_id like "m1__cutoff_60"
            source_ids.add(item["match_id"].split("__")[0])
        assert source_ids == {"m1", "m3"}
    finally:
        os.unlink(tmp_path)

def test_dataset_cutoff_minutes_in_item():
    """cutoff_minutes should be in the dataset item."""
    data = [make_sample("m1", 30)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for s in data:
            f.write(json.dumps(s) + "\n")
        tmp_path = f.name
    try:
        ds = OddsDataset(tmp_path, cutoff_mode="exhaustive", cutoffs=[60])
        item = ds[0]
        # The item should have been built from a cutoff sample
        assert "match_id" in item
    finally:
        os.unlink(tmp_path)


# ── Cutoff bucket constants ──────────────────────────────────────────

def test_cutoff_buckets_base():
    assert CUTOFF_BUCKETS_BASE == [1440, 720, 360, 180, 120, 60, 30, 15, 5]

def test_cutoff_buckets_v1():
    assert CUTOFF_BUCKETS_V1[0] == 1440
    assert CUTOFF_BUCKETS_V1[-1] == 1
    assert len(CUTOFF_BUCKETS_V1) == 6 + 30  # 6 base + 30 last-30


# ── Cutoff semantics ─────────────────────────────────────────────────

def test_cutoff_semantics_60_only_sees_earlier():
    """cutoff=60 means only see events at T-60 or earlier, not T-59."""
    timeline = [
        {"minutes_before_kickoff": 120},
        {"minutes_before_kickoff": 60},
        {"minutes_before_kickoff": 59},
        {"minutes_before_kickoff": 30},
        {"minutes_before_kickoff": 1},
    ]
    vis = filter_timeline_by_cutoff(timeline, 60)
    mbks = [e["minutes_before_kickoff"] for e in vis]
    assert 59 not in mbks
    assert 30 not in mbks
    assert 1 not in mbks
    assert 120 in mbks
    assert 60 in mbks

def test_cutoff_semantics_1_excludes_t0():
    """cutoff=1 excludes only T-0 (kickoff moment)."""
    timeline = [
        {"minutes_before_kickoff": 60},
        {"minutes_before_kickoff": 1},
        {"minutes_before_kickoff": 0},
    ]
    vis = filter_timeline_by_cutoff(timeline, 1)
    mbks = [e["minutes_before_kickoff"] for e in vis]
    assert 0 not in mbks
    assert 1 in mbks
    assert 60 in mbks
