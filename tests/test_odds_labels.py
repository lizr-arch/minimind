"""Tests for P1.0B label mapping contract."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from dataset.odds_labels import (
    encode_euro_label,
    encode_asian_label,
    normalize_asian_label_3class,
    get_asian_num_classes,
    EURO_RESULT_TO_ID,
    ASIAN_3CLASS_TO_ID,
    ASIAN_5CLASS_TO_ID,
    ASIAN_5_TO_3,
)


# ── Euro encode ───────────────────────────────────────────────────────

def test_euro_home():
    assert encode_euro_label("home") == 0

def test_euro_draw():
    assert encode_euro_label("draw") == 1

def test_euro_away():
    assert encode_euro_label("away") == 2

def test_euro_unknown_raises():
    with pytest.raises(ValueError, match="Unknown euro label"):
        encode_euro_label("win")


# ── Asian 3class encode ──────────────────────────────────────────────

def test_asian_3class_full_win():
    assert encode_asian_label("full_win", "3class") == 0

def test_asian_3class_half_win():
    assert encode_asian_label("half_win", "3class") == 0

def test_asian_3class_push():
    assert encode_asian_label("push", "3class") == 1

def test_asian_3class_half_loss():
    assert encode_asian_label("half_loss", "3class") == 2

def test_asian_3class_full_loss():
    assert encode_asian_label("full_loss", "3class") == 2

def test_asian_3class_upper_direct():
    assert encode_asian_label("upper", "3class") == 0

def test_asian_3class_lower_direct():
    assert encode_asian_label("lower", "3class") == 2


# ── Asian 5class encode ──────────────────────────────────────────────

def test_asian_5class_full_win():
    assert encode_asian_label("full_win", "5class") == 0

def test_asian_5class_half_win():
    assert encode_asian_label("half_win", "5class") == 1

def test_asian_5class_push():
    assert encode_asian_label("push", "5class") == 2

def test_asian_5class_half_loss():
    assert encode_asian_label("half_loss", "5class") == 3

def test_asian_5class_full_loss():
    assert encode_asian_label("full_loss", "5class") == 4


# ── Error cases ──────────────────────────────────────────────────────

def test_asian_unknown_raises_3class():
    with pytest.raises(ValueError, match="Unknown asian label"):
        encode_asian_label("win", "3class")

def test_asian_unknown_raises_5class():
    with pytest.raises(ValueError, match="Unknown asian label"):
        encode_asian_label("upper", "5class")

def test_asian_invalid_mode_raises():
    with pytest.raises(ValueError, match="Invalid asian label mode"):
        encode_asian_label("full_win", "2class")


# ── Normalize ────────────────────────────────────────────────────────

def test_normalize_full_win():
    assert normalize_asian_label_3class("full_win") == "upper"

def test_normalize_half_win():
    assert normalize_asian_label_3class("half_win") == "upper"

def test_normalize_push():
    assert normalize_asian_label_3class("push") == "push"

def test_normalize_half_loss():
    assert normalize_asian_label_3class("half_loss") == "lower"

def test_normalize_full_loss():
    assert normalize_asian_label_3class("full_loss") == "lower"

def test_normalize_upper_passthrough():
    assert normalize_asian_label_3class("upper") == "upper"

def test_normalize_unknown_raises():
    with pytest.raises(ValueError, match="Unknown asian label"):
        normalize_asian_label_3class("win")


# ── get_asian_num_classes ────────────────────────────────────────────

def test_num_classes_3class():
    assert get_asian_num_classes("3class") == 3

def test_num_classes_5class():
    assert get_asian_num_classes("5class") == 5

def test_num_classes_invalid():
    with pytest.raises(ValueError):
        get_asian_num_classes("2class")


# ── Mapping completeness ─────────────────────────────────────────────

def test_all_5class_labels_have_3class_mapping():
    """Every 5-class label must map to a 3-class label."""
    for label in ASIAN_5CLASS_TO_ID:
        if label in ("upper_full_win", "upper_half_win", "upper_full_loss", "upper_half_loss"):
            continue  # backward compat aliases
        assert label in ASIAN_5_TO_3, f"5-class label '{label}' missing from ASIAN_5_TO_3"

def test_all_3class_labels_valid():
    """3-class labels must be upper/push/lower."""
    assert set(ASIAN_3CLASS_TO_ID.keys()) == {"upper", "push", "lower"}
