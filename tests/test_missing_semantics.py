"""Tests for P1.0F missing_mask / asian_line=0 semantic repair."""
import sys
import os
import json
import tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
from dataset.odds_dataset import (
    OddsDataset,
    _event_has_euro,
    _event_has_asian,
    _event_has_over_under,
    _event_to_missing_mask,
    _event_to_features,
)


# ── Helpers ───────────────────────────────────────────────────────────

def _make_event(**overrides):
    """Create an event with defaults, then apply overrides."""
    event = {
        "minutes_before_kickoff": 60,
        "euro_h": 2.0, "euro_d": 3.5, "euro_a": 4.0,
        "asian_line": 0.0, "upper_water": 1.0, "lower_water": 1.0,
        "over_under_line": 2.5, "over_water": 1.0, "under_water": 1.0,
    }
    event.update(overrides)
    return event


# ── 8. asian_line=0 + real water → has_asian=True ────────────────────

def test_asian_flat_handicap_with_real_water():
    """asian_line=0 with non-1.0 water prices is a real flat handicap."""
    event = _make_event(asian_line=0.0, upper_water=0.9, lower_water=1.0)
    assert _event_has_asian(event) is True


# ── 9. asian_line=0 + water=1.0/1.0 → has_asian=False ──────────────

def test_asian_default_placeholder_detected():
    """asian_line=0 with water=1.0/1.0 is a default placeholder (missing)."""
    event = _make_event(asian_line=0.0, upper_water=1.0, lower_water=1.0)
    assert _event_has_asian(event) is False


# ── 10. missing_mask shape aligns with features ──────────────────────

def test_missing_mask_shape_matches_features():
    """missing_mask shape must match feature shape [T, feature_dim]."""
    event = _make_event(euro_h=2.0, euro_d=3.5, euro_a=4.0,
                        asian_line=0.5, upper_water=0.9, lower_water=1.0,
                        over_under_line=2.5, over_water=0.85, under_water=1.05)
    features = _event_to_features(event, schema_version="v3")
    mask = _event_to_missing_mask(event)
    assert len(features) == len(mask)


# ── 11. missing_mask 1=present, 0=missing ────────────────────────────

def test_missing_mask_present_vs_missing():
    """Euro present → 1.0; asian default → 0.0."""
    # Event with real euro, default asian, default ou
    event = _make_event(euro_h=2.0, euro_d=3.5, euro_a=4.0,
                        asian_line=0.0, upper_water=1.0, lower_water=1.0,
                        over_under_line=2.5, over_water=1.0, under_water=1.0)
    mask = _event_to_missing_mask(event)
    # minutes: always present (1.0)
    assert mask[0] == 1.0
    # euro_h/d/a: present (1.0)
    assert mask[1] == 1.0
    assert mask[2] == 1.0
    assert mask[3] == 1.0
    # asian_line, upper_water, lower_water: missing (0.0)
    assert mask[4] == 0.0
    assert mask[5] == 0.0
    assert mask[6] == 0.0
    # over_under_line, over_water, under_water: missing (0.0)
    assert mask[7] == 0.0
    assert mask[8] == 0.0
    assert mask[9] == 0.0
    # has_euro/has_asian/has_ou: always present (1.0)
    assert mask[10] == 1.0
    assert mask[11] == 1.0
    assert mask[12] == 1.0


# ── 12. Missing euro not misjudged as present ────────────────────────

def test_missing_euro_not_present():
    """euro_h=0 → has_euro=False, mask=0.0 for euro fields."""
    event = _make_event(euro_h=0, euro_d=0, euro_a=0)
    assert _event_has_euro(event) is False
    mask = _event_to_missing_mask(event)
    assert mask[1] == 0.0  # euro_h
    assert mask[2] == 0.0  # euro_d
    assert mask[3] == 0.0  # euro_a


# ── 13. Missing asian not misjudged by default water ─────────────────

def test_missing_asian_not_present_by_default_water():
    """Default water=1.0/1.0 with line=0 → has_asian=False."""
    event = _make_event(asian_line=0.0, upper_water=1.0, lower_water=1.0)
    assert _event_has_asian(event) is False
    mask = _event_to_missing_mask(event)
    assert mask[4] == 0.0  # asian_line
    assert mask[5] == 0.0  # upper_water
    assert mask[6] == 0.0  # lower_water


# ── 14. Missing ou not misjudged by default 2.5 ─────────────────────

def test_missing_ou_not_present_by_default():
    """ou=2.5, water=1.0/1.0 → has_ou=False."""
    event = _make_event(over_under_line=2.5, over_water=1.0, under_water=1.0)
    assert _event_has_over_under(event) is False
    mask = _event_to_missing_mask(event)
    assert mask[7] == 0.0  # over_under_line
    assert mask[8] == 0.0  # over_water
    assert mask[9] == 0.0  # under_water


# ── Additional edge cases ────────────────────────────────────────────

def test_asian_real_nonzero_line():
    """asian_line=-0.5 with real water → has_asian=True."""
    event = _make_event(asian_line=-0.5, upper_water=0.85, lower_water=1.05)
    assert _event_has_asian(event) is True
    mask = _event_to_missing_mask(event)
    assert mask[4] == 1.0
    assert mask[5] == 1.0
    assert mask[6] == 1.0


def test_ou_real_non_default():
    """ou_line=3.0 with non-1.0 water → has_ou=True."""
    event = _make_event(over_under_line=3.0, over_water=0.9, under_water=1.1)
    assert _event_has_over_under(event) is True
    mask = _event_to_missing_mask(event)
    assert mask[7] == 1.0
    assert mask[8] == 1.0
    assert mask[9] == 1.0


def test_ou_line_zero_always_missing():
    """ou_line=0 is always missing regardless of water."""
    event = _make_event(over_under_line=0, over_water=0.9, under_water=1.1)
    assert _event_has_over_under(event) is False


def test_has_asian_flag_in_features_v2():
    """v2 features should reflect the corrected has_asian detection."""
    # Default placeholder
    event_default = _make_event()
    feats_default = _event_to_features(event_default, schema_version="v2")
    assert feats_default[11] == 0.0  # has_asian = False

    # Real asian
    event_real = _make_event(asian_line=-0.5, upper_water=0.9, lower_water=1.0)
    feats_real = _event_to_features(event_real, schema_version="v2")
    assert feats_real[11] == 1.0  # has_asian = True


def test_has_ou_flag_in_features_v2():
    """v2 features should reflect the corrected has_ou detection."""
    # Default placeholder
    event_default = _make_event()
    feats_default = _event_to_features(event_default, schema_version="v2")
    assert feats_default[12] == 0.0  # has_ou = False

    # Real ou
    event_real = _make_event(over_under_line=3.0, over_water=0.9, under_water=1.1)
    feats_real = _event_to_features(event_real, schema_version="v2")
    assert feats_real[12] == 1.0  # has_ou = True


def test_all_default_event_full_mask():
    """An event with all defaults: only minutes and has_* flags are present."""
    event = _make_event(euro_h=0, euro_d=0, euro_a=0,
                        asian_line=0.0, upper_water=1.0, lower_water=1.0,
                        over_under_line=2.5, over_water=1.0, under_water=1.0)
    mask = _event_to_missing_mask(event)
    expected = [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 1.0, 1.0]
    for i, (a, b) in enumerate(zip(mask, expected)):
        assert a == b, f"mask[{i}] = {a}, expected {b}"
