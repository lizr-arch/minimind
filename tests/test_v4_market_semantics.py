"""Tests for P1.1B v4 market semantics: source detection + asian_label_mask."""
import sys
import os
import json
import tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
from dataset.odds_dataset import (
    OddsDataset,
    _event_has_asian,
    _event_has_over_under,
    _event_to_features_v4,
    _event_to_missing_mask_v4,
)
from dataset.odds_collator import OddsCollator


# ── Helpers ───────────────────────────────────────────────────────────

def _make_v4_event(**overrides):
    event = {
        "minutes_before_kickoff": 60,
        "euro_h": 2.0, "euro_d": 3.5, "euro_a": 4.0,
        "euro_source": "raw_update",
        "euro_h_missing": False, "euro_d_missing": False, "euro_a_missing": False,
        "asian_line": -0.5, "upper_water": 0.9, "lower_water": 1.0,
        "asian_source": "raw_update",
        "asian_line_missing": False, "upper_water_missing": False, "lower_water_missing": False,
        "has_asian": True,
        "over_under_line": 2.5, "over_water": 0.9, "under_water": 1.1,
        "over_under_source": "raw_update",
        "over_under_line_missing": False, "over_water_missing": False, "under_water_missing": False,
        "has_over_under": True,
        "has_euro": True,
    }
    event.update(overrides)
    return event


def _make_v4_sample(match_id="m1", asian_label_status="ok", asian_result="full_win", n_events=5):
    timeline = []
    for i in range(n_events):
        e = _make_v4_event(minutes_before_kickoff=1440 - i * 100)
        timeline.append(e)
    return {
        "match_id": match_id,
        "league_id": "epl",
        "bookmaker_id": "Bet365",
        "kickoff_time": "2023-04-10T15:00:00",
        "time_axis_status": "ok",
        "odds_timeline": timeline,
        "label": {
            "euro_result": "home",
            "asian_result": asian_result,
            "asian_label_status": asian_label_status,
            "home_goals": 2, "away_goals": 1,
        },
    }


def _tmp_jsonl(samples):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for s in samples:
        f.write(json.dumps(s) + "\n")
    f.close()
    return f.name


# ── 1. v4 asian_line=0 + has_asian=true → real flat handicap ─────────

def test_v4_asian_line_zero_with_has_asian_true():
    """asian_line=0 with asian_source=raw_update is a real flat handicap."""
    e = _make_v4_event(asian_line=0.0, upper_water=0.9, lower_water=1.0,
                       asian_source="raw_update", has_asian=True)
    assert _event_has_asian(e) is True


# ── 2. v4 asian_line=0 + has_asian=false → missing ──────────────────

def test_v4_asian_line_zero_with_has_asian_false():
    """asian_line=0 with asian_source=missing is missing."""
    e = _make_v4_event(asian_line=0.0, upper_water=0.0, lower_water=0.0,
                       asian_source="missing", has_asian=False)
    assert _event_has_asian(e) is False


# ── 3. asian_source=raw_update → present ────────────────────────────

def test_v4_asian_source_raw_update():
    e = _make_v4_event(asian_source="raw_update")
    assert _event_has_asian(e) is True


# ── 4. asian_source=forward_fill → present ──────────────────────────

def test_v4_asian_source_forward_fill():
    e = _make_v4_event(asian_source="forward_fill")
    assert _event_has_asian(e) is True


# ── 5. asian_source=missing → missing ───────────────────────────────

def test_v4_asian_source_missing():
    e = _make_v4_event(asian_source="missing", has_asian=False)
    assert _event_has_asian(e) is False


# ── 6. over_under_source=raw_update → present ───────────────────────

def test_v4_ou_source_raw_update():
    e = _make_v4_event(over_under_source="raw_update")
    assert _event_has_over_under(e) is True


# ── 7. over_under_source=forward_fill → present ─────────────────────

def test_v4_ou_source_forward_fill():
    e = _make_v4_event(over_under_source="forward_fill")
    assert _event_has_over_under(e) is True


# ── 8. over_under_source=missing → missing ──────────────────────────

def test_v4_ou_source_missing():
    e = _make_v4_event(over_under_source="missing", has_over_under=False)
    assert _event_has_over_under(e) is False


# ── 9. asian_label_status=missing_handicap → no silent fallback ─────

def test_missing_handicap_no_silent_fallback():
    """missing_handicap must produce asian_label=-100 and mask=0.0, not class 0."""
    sample = _make_v4_sample(asian_label_status="missing_handicap", asian_result=None)
    path = _tmp_jsonl([sample])
    try:
        ds = OddsDataset(path, feature_schema_version="v4", cutoff_mode="none")
        item = ds[0]
        assert item["asian_label"] == -100, f"Expected -100, got {item['asian_label']}"
        assert item["asian_label_mask"] == 0.0, f"Expected 0.0, got {item['asian_label_mask']}"
    finally:
        os.unlink(path)


# ── 10. Dataset + Collator output asian_label_mask ───────────────────

def test_collator_outputs_asian_label_mask():
    """Collator must output asian_label_mask with shape [B]."""
    s1 = _make_v4_sample("m1", asian_label_status="ok", asian_result="full_win")
    s2 = _make_v4_sample("m2", asian_label_status="missing_handicap", asian_result=None)
    path = _tmp_jsonl([s1, s2])
    try:
        ds = OddsDataset(path, feature_schema_version="v4", cutoff_mode="none")
        collator = OddsCollator()
        batch = collator([ds[0], ds[1]])
        assert "asian_label_mask" in batch
        assert batch["asian_label_mask"].shape == (2,)
        assert batch["asian_label_mask"][0].item() == 1.0  # ok
        assert batch["asian_label_mask"][1].item() == 0.0  # missing_handicap
    finally:
        os.unlink(path)


# ── 11. v4 features all finite ───────────────────────────────────────

def test_v4_features_all_finite_with_source_fields():
    """v4 features with source fields must be all finite."""
    e = _make_v4_event()
    feats = _event_to_features_v4(e)
    import math
    for i, v in enumerate(feats):
        assert math.isfinite(v), f"Feature[{i}] = {v} is not finite"


# ── 12. v4 split overlap = 0 ────────────────────────────────────────
# (covered by generate_odds_splits + gate script)


# ── 13. v4 cutoff integrity ──────────────────────────────────────────

def test_v4_cutoff_integrity():
    """Cutoff samples from v4 data must respect cutoff boundary."""
    from dataset.odds_cutoff import build_cutoff_samples, assert_cutoff_integrity
    sample = _make_v4_sample(n_events=20)
    for cutoff in [1440, 360, 60, 1]:
        cs_list = build_cutoff_samples(sample, [cutoff], min_events=1)
        for cs in cs_list:
            assert_cutoff_integrity(cs["odds_timeline"], cutoff, sample_id=cs.get("sample_id", ""))


# ── Additional v4 source detection tests ─────────────────────────────

def test_v4_asian_line_zero_raw_update_real_water():
    """asian_line=0 with raw_update and non-1.0 water = real flat handicap."""
    e = _make_v4_event(asian_line=0.0, upper_water=0.85, lower_water=1.05,
                       asian_source="raw_update", has_asian=True)
    assert _event_has_asian(e) is True
    mask = _event_to_missing_mask_v4(e)
    assert mask[12] == 1.0  # asian_line present

def test_v4_missing_mask_reflects_source():
    """Missing mask must reflect source-based detection."""
    # Missing event
    e_missing = _make_v4_event(asian_source="missing", has_asian=False,
                               asian_line=0.0, upper_water=0.0, lower_water=0.0)
    mask = _event_to_missing_mask_v4(e_missing)
    assert mask[12] == 0.0  # asian_line missing
    assert mask[13] == 0.0  # upper_water missing

    # Present event
    e_present = _make_v4_event(asian_source="raw_update", has_asian=True)
    mask = _event_to_missing_mask_v4(e_present)
    assert mask[12] == 1.0  # asian_line present
    assert mask[13] == 1.0  # upper_water present

def test_ok_label_has_mask_one():
    """ok label status produces asian_label_mask=1.0."""
    sample = _make_v4_sample(asian_label_status="ok", asian_result="full_win")
    path = _tmp_jsonl([sample])
    try:
        ds = OddsDataset(path, feature_schema_version="v4", cutoff_mode="none")
        item = ds[0]
        assert item["asian_label_mask"] == 1.0
        assert item["asian_label"] == 0  # full_win → 0
    finally:
        os.unlink(path)
