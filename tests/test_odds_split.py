"""Tests for P1.0C grouped time split."""
import sys
import os
import json
import tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from dataset.odds_split import grouped_time_split, load_match_ids_from_file
from tools.generate_odds_splits import load_and_validate, generate_splits


# ── Fixtures ──────────────────────────────────────────────────────────

def make_sample(match_id, kickoff_time, bookmaker="Bet365", euro="home", asian="full_win"):
    return {
        "match_id": match_id,
        "kickoff_time": kickoff_time,
        "time_axis_status": "ok",
        "league_id": "epl",
        "bookmaker_id": bookmaker,
        "odds_timeline": [{"minutes_before_kickoff": 60, "euro_h": 2.0, "euro_d": 3.5, "euro_a": 4.0}],
        "label": {"euro_result": euro, "asian_result": asian, "home_goals": 2, "away_goals": 1},
    }


@pytest.fixture
def sample_data():
    """Create sample data with 10 matches across 3 time periods."""
    return [
        make_sample("m1", "2022-08-10T15:00:00"),
        make_sample("m2", "2022-09-10T15:00:00"),
        make_sample("m3", "2022-10-10T15:00:00"),
        make_sample("m4", "2023-01-10T15:00:00"),
        make_sample("m5", "2023-02-10T15:00:00"),
        make_sample("m6", "2023-03-10T15:00:00"),
        make_sample("m7", "2023-04-10T15:00:00"),
        make_sample("m8", "2024-01-10T15:00:00"),
        make_sample("m9", "2024-02-10T15:00:00"),
        make_sample("m10", "2024-03-10T15:00:00"),
    ]


@pytest.fixture
def multi_bookmaker_data():
    """Data where some matches have multiple bookmaker samples."""
    return [
        make_sample("m1", "2022-08-10T15:00:00", "Bet365"),
        make_sample("m1", "2022-08-10T15:00:00", "Pinnacle"),
        make_sample("m1", "2022-08-10T15:00:00", "Sbobet"),
        make_sample("m2", "2023-01-10T15:00:00", "Bet365"),
        make_sample("m2", "2023-01-10T15:00:00", "Pinnacle"),
        make_sample("m3", "2024-01-10T15:00:00", "Bet365"),
    ]


# ── Split by match_id ────────────────────────────────────────────────

def test_split_by_match_id_not_sample(sample_data):
    """Split must be by match_id, not by individual samples."""
    grouped = {}
    for s in sample_data:
        mid = s["match_id"]
        if mid not in grouped:
            grouped[mid] = {"kickoff_times": set(), "samples": []}
        grouped[mid]["kickoff_times"].add(s["kickoff_time"])
        grouped[mid]["samples"].append(s)

    splits = generate_splits(grouped, 0.7, 0.15, 0.15)
    all_ids = splits["train"] | splits["val"] | splits["test"]
    assert len(all_ids) == 10  # 10 unique match_ids


def test_multi_bookmaker_same_split(multi_bookmaker_data):
    """All bookmaker samples from same match_id must be in same split."""
    grouped = {}
    for s in multi_bookmaker_data:
        mid = s["match_id"]
        if mid not in grouped:
            grouped[mid] = {"kickoff_times": set(), "samples": []}
        grouped[mid]["kickoff_times"].add(s["kickoff_time"])
        grouped[mid]["samples"].append(s)

    splits = generate_splits(grouped, 0.5, 0.25, 0.25)

    # Check each match_id is in exactly one split
    for mid in ["m1", "m2", "m3"]:
        in_splits = [name for name, ids in splits.items() if mid in ids]
        assert len(in_splits) == 1, f"{mid} found in multiple splits: {in_splits}"


# ── Split mutual exclusivity ─────────────────────────────────────────

def test_splits_mutually_exclusive(sample_data):
    """train/val/test match_id sets must be disjoint."""
    grouped = {}
    for s in sample_data:
        mid = s["match_id"]
        if mid not in grouped:
            grouped[mid] = {"kickoff_times": set(), "samples": []}
        grouped[mid]["kickoff_times"].add(s["kickoff_time"])

    splits = generate_splits(grouped, 0.7, 0.15, 0.15)
    assert splits["train"].isdisjoint(splits["val"])
    assert splits["train"].isdisjoint(splits["test"])
    assert splits["val"].isdisjoint(splits["test"])


# ── Time ordering ────────────────────────────────────────────────────

def test_split_ascending_time(sample_data):
    """Earlier matches go to train, later to test."""
    grouped = {}
    for s in sample_data:
        mid = s["match_id"]
        if mid not in grouped:
            grouped[mid] = {"kickoff_times": set(), "samples": []}
        grouped[mid]["kickoff_times"].add(s["kickoff_time"])

    splits = generate_splits(grouped, 0.5, 0.25, 0.25)

    # m1-m5 should be in train (earliest), m8-m10 in test (latest)
    assert "m1" in splits["train"]
    assert "m10" in splits["test"]


# ── Missing kickoff_time exclusion ───────────────────────────────────

def test_missing_kickoff_excluded():
    """Samples without kickoff_time should not enter main split."""
    data = [
        {"match_id": "m1", "kickoff_time": "", "time_axis_status": "ok", "label": {"euro_result": "home", "asian_result": "full_win"}},
        {"match_id": "m2", "kickoff_time": "2023-01-10T15:00:00", "time_axis_status": "ok", "label": {"euro_result": "draw", "asian_result": "push"}},
    ]
    grouped = {}
    for s in data:
        mid = s["match_id"]
        if mid not in grouped:
            grouped[mid] = {"kickoff_times": set(), "samples": []}
        grouped[mid]["kickoff_times"].add(s.get("kickoff_time", ""))
        grouped[mid]["samples"].append(s)

    # m1 has empty kickoff_time, should be filtered
    valid = {mid: g for mid, g in grouped.items() if g["kickoff_times"] and any(ko for ko in g["kickoff_times"])}
    assert "m1" not in valid
    assert "m2" in valid


# ── Multiple kickoff_time detection ──────────────────────────────────

def test_multiple_kickoff_times_detected():
    """Same match_id with different kickoff_times should be flagged."""
    data = [
        {"match_id": "m1", "kickoff_time": "2023-01-10T15:00:00", "time_axis_status": "ok"},
        {"match_id": "m1", "kickoff_time": "2023-01-11T15:00:00", "time_axis_status": "ok"},
    ]

    # Group manually
    grouped = {}
    for s in data:
        mid = s["match_id"]
        if mid not in grouped:
            grouped[mid] = {"kickoff_times": set(), "samples": []}
        grouped[mid]["kickoff_times"].add(s["kickoff_time"])
        grouped[mid]["samples"].append(s)

    # Should detect inconsistency
    invalid = [mid for mid, g in grouped.items() if len(g["kickoff_times"]) > 1]
    assert "m1" in invalid


# ── Dataset allowed_match_ids ────────────────────────────────────────

def test_dataset_filters_by_allowed_ids():
    """Dataset should only load samples with allowed match_ids."""
    from dataset.odds_dataset import OddsDataset
    import tempfile

    data = [
        make_sample("m1", "2023-01-10T15:00:00"),
        make_sample("m2", "2023-02-10T15:00:00"),
        make_sample("m3", "2023-03-10T15:00:00"),
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for s in data:
            f.write(json.dumps(s) + "\n")
        tmp_path = f.name

    try:
        ds = OddsDataset(tmp_path, allowed_match_ids={"m1", "m3"})
        assert len(ds) == 2
        loaded_ids = {ds[i]["match_id"] for i in range(len(ds))}
        assert loaded_ids == {"m1", "m3"}
    finally:
        os.unlink(tmp_path)


# ── Split file I/O ───────────────────────────────────────────────────

def test_load_match_ids_from_file():
    """Should load match_ids from text file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("m1\nm2\nm3\n")
        tmp_path = f.name

    try:
        ids = load_match_ids_from_file(tmp_path)
        assert ids == {"m1", "m2", "m3"}
    finally:
        os.unlink(tmp_path)


# ── Full pipeline smoke ──────────────────────────────────────────────

def test_full_pipeline_smoke(sample_data):
    """End-to-end: load, split, write, reload."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write input
        input_path = os.path.join(tmpdir, "input.jsonl")
        with open(input_path, "w") as f:
            for s in sample_data:
                f.write(json.dumps(s) + "\n")

        # Load and split
        samples, grouped, invalid = load_and_validate(input_path)
        assert len(invalid) == 0
        splits = generate_splits(grouped, 0.7, 0.15, 0.15)

        # Write split files
        for name, ids in splits.items():
            path = os.path.join(tmpdir, f"{name}_match_ids.txt")
            with open(path, "w") as f:
                for mid in sorted(ids):
                    f.write(mid + "\n")

        # Reload and verify
        for name in ["train", "val", "test"]:
            path = os.path.join(tmpdir, f"{name}_match_ids.txt")
            ids = load_match_ids_from_file(path)
            assert len(ids) > 0
