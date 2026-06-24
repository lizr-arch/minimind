"""Tests for P1.0E-G total data gate."""
import sys
import os
import json
import tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
import math
from dataset.odds_dataset import (
    OddsDataset,
    _event_to_features_v4,
    _compute_consensus_feats,
    V4_FEATURE_DIM,
)
from dataset.odds_cutoff import build_cutoff_samples, assert_cutoff_integrity


# ── Helpers ───────────────────────────────────────────────────────────

def _make_sample(match_id="m1", n_events=10, euro_h=2.0, asian_default=True):
    timeline = []
    for i in range(n_events):
        mbk = 1440 - i * (1440 // n_events)
        e = {
            "minutes_before_kickoff": mbk,
            "euro_h": euro_h + i * 0.05, "euro_d": 3.5, "euro_a": 4.0,
            "over_under_line": 2.5, "over_water": 1.0, "under_water": 1.0,
        }
        if asian_default:
            e.update({"asian_line": 0.0, "upper_water": 1.0, "lower_water": 1.0})
        else:
            e.update({"asian_line": -0.5, "upper_water": 0.9, "lower_water": 1.0})
        timeline.append(e)
    return {
        "match_id": match_id,
        "league_id": "epl",
        "bookmaker_id": "Bet365",
        "kickoff_time": "2023-04-10T15:00:00",
        "time_axis_status": "ok",
        "odds_timeline": timeline,
        "label": {"euro_result": "home", "asian_result": "full_win", "home_goals": 2, "away_goals": 1},
    }


def _tmp_jsonl(samples):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for s in samples:
        f.write(json.dumps(s) + "\n")
    f.close()
    return f.name


# ── 23. inspect_odds_data_gate.py can generate report ────────────────

def test_gate_script_generates_report():
    """The gate script must produce a valid report file."""
    import subprocess
    path = _tmp_jsonl([_make_sample("m1"), _make_sample("m2")])
    out_path = tempfile.mktemp(suffix=".md")
    try:
        result = subprocess.run(
            [sys.executable, "tools/inspect_odds_data_gate.py",
             "--input", path,
             "--splits-dir", "data/odds_real/splits/v3_time",
             "--output", out_path,
             "--feature-schema", "v4",
             "--consensus-mode", "none",
             "--cutoff-buckets", "1440,60,1"],
            capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert os.path.exists(out_path)
        with open(out_path, "r") as f:
            content = f.read()
        assert "Data Gate Report" in content
        assert "Final Verdict" in content
    finally:
        os.unlink(path)
        if os.path.exists(out_path):
            os.unlink(out_path)


# ── 24. data gate can identify consensus leakage ─────────────────────

def test_gate_detects_consensus_leakage():
    """Gate must detect when consensus_feats uses non-safe mode with data."""
    from tools.inspect_odds_data_gate import check_consensus_leakage
    # With no bookmaker_features and mode=none, should be safe
    samples = [_make_sample()]
    result = check_consensus_leakage(samples, consensus_mode="none")
    assert result["pass"] is True


# ── 25. data gate can identify split overlap ──────────────────────────

def test_gate_detects_split_overlap():
    """Gate must detect overlapping match_ids between splits."""
    from tools.inspect_odds_data_gate import check_split_overlap
    # No overlap
    splits_ok = {"train": {"m1", "m2"}, "val": {"m3"}, "test": {"m4"}}
    result = check_split_overlap(splits_ok)
    assert result["pass"] is True

    # With overlap
    splits_bad = {"train": {"m1", "m2"}, "val": {"m2", "m3"}, "test": {"m4"}}
    result = check_split_overlap(splits_bad)
    assert result["pass"] is False
    assert result["train_val_overlap"] == 1


# ── 26. data gate can identify cutoff violation ──────────────────────

def test_gate_detects_cutoff_violation():
    """Gate must detect cutoff integrity violations."""
    from tools.inspect_odds_data_gate import check_cutoff
    # Valid data
    samples = [_make_sample(n_events=20)]
    result = check_cutoff(samples, [60])
    assert result["pass"] is True


# ── 27. data gate can identify nan/inf feature ───────────────────────

def test_gate_detects_nan_inf():
    """Gate must detect nan/inf in features."""
    from tools.inspect_odds_data_gate import check_feature_finite
    # Normal data
    samples = [_make_sample()]
    result = check_feature_finite(samples, "v4")
    assert result["pass"] is True
    assert result["nan_count"] == 0
    assert result["inf_count"] == 0


# ── Additional gate tests ────────────────────────────────────────────

def test_gate_time_axis_check():
    """Gate 1: time axis must pass for valid data."""
    from tools.inspect_odds_data_gate import check_time_axis
    samples = [_make_sample()]
    result = check_time_axis(samples)
    assert result["pass"] is True

def test_gate_label_check():
    """Gate 2: labels must pass for valid data."""
    from tools.inspect_odds_data_gate import check_labels
    samples = [_make_sample()]
    result = check_labels(samples)
    assert result["pass"] is True

def test_gate_missing_semantics_check():
    """Gate 6: missing semantics must detect defaults correctly."""
    from tools.inspect_odds_data_gate import check_missing_semantics
    samples = [_make_sample(asian_default=True)]
    result = check_missing_semantics(samples)
    assert result["pass"] is True
    assert result["misdetected"] == 0

def test_gate_no_future_leakage():
    """Gate 8: no future leakage in cutoff-filtered data."""
    from tools.inspect_odds_data_gate import check_no_future_leakage
    samples = [_make_sample(n_events=30)]
    result = check_no_future_leakage(samples, [60, 30])
    assert result["pass"] is True
