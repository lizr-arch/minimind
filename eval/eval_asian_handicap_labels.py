"""
P0.3 Asian Handicap Label Tests

Comprehensive unit tests for:
  - split_asian_line
  - settle_half_line
  - settle_asian_5class
  - settle_asian_3class
  - get_upper_lower_goals

Usage:
    python eval/eval_asian_handicap_labels.py
"""

import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.asian_handicap import (
    split_asian_line,
    settle_half_line,
    settle_asian_5class,
    settle_asian_3class,
    get_upper_lower_goals,
)

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  -- {detail}")


# ── Test 1: split_asian_line ──────────────────────────────────────────

def test_split_asian_line():
    print("\n--- Test 1: split_asian_line ---")

    # Integer / half lines → both halves identical
    check("0 -> [0, 0]", split_asian_line(0) == (0, 0))
    check("-0.5 -> [-0.5, -0.5]", split_asian_line(-0.5) == (-0.5, -0.5))
    check("+0.5 -> [0.5, 0.5]", split_asian_line(0.5) == (0.5, 0.5))
    check("-1.0 -> [-1.0, -1.0]", split_asian_line(-1.0) == (-1.0, -1.0))
    check("+1.0 -> [1.0, 1.0]", split_asian_line(1.0) == (1.0, 1.0))
    check("-1.5 -> [-1.5, -1.5]", split_asian_line(-1.5) == (-1.5, -1.5))

    # Quarter lines → two adjacent half lines (more favourable first)
    check("-0.25 -> [0, -0.5]", split_asian_line(-0.25) == (0.0, -0.5))
    check("+0.25 -> [0.5, 0]", split_asian_line(0.25) == (0.5, 0.0))
    check("-0.75 -> [-0.5, -1.0]", split_asian_line(-0.75) == (-0.5, -1.0))
    check("+0.75 -> [1.0, 0.5]", split_asian_line(0.75) == (1.0, 0.5))
    check("-1.25 -> [-1.0, -1.5]", split_asian_line(-1.25) == (-1.0, -1.5))
    check("+1.25 -> [1.5, 1.0]", split_asian_line(1.25) == (1.5, 1.0))
    check("-1.75 -> [-1.5, -2.0]", split_asian_line(-1.75) == (-1.5, -2.0))
    check("+1.75 -> [2.0, 1.5]", split_asian_line(1.75) == (2.0, 1.5))

    # Invalid lines
    try:
        split_asian_line(0.3)
        check("invalid 0.3 raises ValueError", False, "should have raised")
    except ValueError:
        check("invalid 0.3 raises ValueError", True)


# ── Test 2: settle_half_line ──────────────────────────────────────────

def test_settle_half_line():
    print("\n--- Test 2: settle_half_line ---")

    check("margin=1, line=-0.5 -> win",
          settle_half_line(1, -0.5) == "win")
    check("margin=0, line=-0.5 -> loss",
          settle_half_line(0, -0.5) == "loss")
    check("margin=1, line=-1.0 -> push",
          settle_half_line(1, -1.0) == "push")
    check("margin=0, line=0 -> push",
          settle_half_line(0, 0) == "push")
    check("margin=0, line=+0.5 -> win",
          settle_half_line(0, 0.5) == "win")
    check("margin=-1, line=+0.5 -> loss",
          settle_half_line(-1, 0.5) == "loss")
    check("margin=-1, line=+1.0 -> push",
          settle_half_line(-1, 1.0) == "push")


# ── Test 3: settle_asian_5class ───────────────────────────────────────

def test_settle_asian_5class():
    print("\n--- Test 3: settle_asian_5class ---")

    # Integer / half lines (both halves same → full_win or full_loss or push)
    check("upper 2:1, line=-0.5 -> upper_full_win",
          settle_asian_5class(2, 1, -0.5) == "upper_full_win")
    check("upper 1:1, line=-0.5 -> upper_full_loss",
          settle_asian_5class(1, 1, -0.5) == "upper_full_loss")
    check("upper 2:1, line=-1.0 -> push",
          settle_asian_5class(2, 1, -1.0) == "push")
    check("upper 0:0, line=0 -> push",
          settle_asian_5class(0, 0, 0) == "push")
    check("upper 3:0, line=-2.0 -> upper_full_win",
          settle_asian_5class(3, 0, -2.0) == "upper_full_win")

    # Quarter line: -0.75, win by 1 → [win on -0.5, push on -1.0] → upper_half_win
    check("upper 1:0, line=-0.75 -> upper_half_win",
          settle_asian_5class(1, 0, -0.75) == "upper_half_win")

    # Quarter line: -0.75, win by 2 → [win on -0.5, win on -1.0] → upper_full_win
    check("upper 2:0, line=-0.75 -> upper_full_win",
          settle_asian_5class(2, 0, -0.75) == "upper_full_win")

    # Quarter line: -1.25, win by 1 → [push on -1.0, loss on -1.5] → upper_half_loss
    check("upper 1:0, line=-1.25 -> upper_half_loss",
          settle_asian_5class(1, 0, -1.25) == "upper_half_loss")

    # Quarter line: -0.25, draw → [push on 0, loss on -0.5] → upper_half_loss
    check("upper 0:0, line=-0.25 -> upper_half_loss",
          settle_asian_5class(0, 0, -0.25) == "upper_half_loss")

    # Quarter line: +0.25, draw → [win on +0.5, push on 0] → upper_half_win
    check("upper 0:0, line=+0.25 -> upper_half_win",
          settle_asian_5class(0, 0, 0.25) == "upper_half_win")

    # Quarter line: +0.5, draw → upper_full_win (both halves = +0.5, both win)
    check("upper 0:0, line=+0.5 -> upper_full_win",
          settle_asian_5class(0, 0, 0.5) == "upper_full_win")

    # Upper side is away: away 2, home 1, line=-0.5 → away=upper → margin=1 → win
    check("away(upper) 2:1, line=-0.5 -> upper_full_win",
          settle_asian_5class(2, 1, -0.5) == "upper_full_win")

    # +1.25, lose by 1 → upper_half_win (win on +1.5, push on +1.0)
    check("upper 0:1, line=+1.25 -> upper_half_win",
          settle_asian_5class(0, 1, 1.25) == "upper_half_win")

    # +1, lose by 1 → push
    check("upper 0:1, line=+1.0 -> push",
          settle_asian_5class(0, 1, 1.0) == "push")

    # -0.75, win by 0 → [loss on -0.5, loss on -1.0] → upper_full_loss
    check("upper 0:0, line=-0.75 -> upper_full_loss",
          settle_asian_5class(0, 0, -0.75) == "upper_full_loss")

    # -1.75, win by 2 → [win on -1.5, push on -2.0] → upper_half_win
    check("upper 2:0, line=-1.75 -> upper_half_win",
          settle_asian_5class(2, 0, -1.75) == "upper_half_win")

    # +0.75, win by 1 → [win on +1.0, win on +0.5] → upper_full_win
    check("upper 1:0, line=+0.75 -> upper_full_win",
          settle_asian_5class(1, 0, 0.75) == "upper_full_win")


# ── Test 4: settle_asian_3class ───────────────────────────────────────

def test_settle_asian_3class():
    print("\n--- Test 4: settle_asian_3class ---")

    check("upper_full_win -> upper",
          settle_asian_3class(2, 0, -0.5) == "upper")
    check("upper_half_win -> upper",
          settle_asian_3class(1, 0, -0.75) == "upper")
    check("push -> push",
          settle_asian_3class(2, 1, -1.0) == "push")
    check("upper_half_loss -> lower",
          settle_asian_3class(1, 0, -1.25) == "lower")
    check("upper_full_loss -> lower",
          settle_asian_3class(0, 0, -0.5) == "lower")


# ── Test 5: get_upper_lower_goals ─────────────────────────────────────

def test_get_upper_lower_goals():
    print("\n--- Test 5: get_upper_lower_goals ---")

    check("home upper, 2:1",
          get_upper_lower_goals(2, 1, "home") == (2, 1))
    check("away upper, 1:2",
          get_upper_lower_goals(1, 2, "away") == (2, 1))
    check("home upper, 0:0",
          get_upper_lower_goals(0, 0, "home") == (0, 0))

    try:
        get_upper_lower_goals(1, 0, "neutral")
        check("invalid upper_side raises ValueError", False)
    except ValueError:
        check("invalid upper_side raises ValueError", True)


# ── Test 6: Fixture labels match calculation ──────────────────────────

def test_fixture_labels():
    """Verify that the asian_result labels in the fixture data are correct
    according to settle_asian_5class."""
    print("\n--- Test 6: Fixture label consistency ---")
    import json

    with open("data/odds_fixtures/sample_odds_matches_5class.jsonl", "r") as f:
        samples = [json.loads(l) for l in f if l.strip()]

    for s in samples:
        mid = s["match_id"]
        upper_side = s.get("upper_side", "home")
        hg = s.get("home_goals", 0)
        ag = s.get("away_goals", 0)
        upper_g, lower_g = get_upper_lower_goals(hg, ag, upper_side)

        # Use the last asian_line in the timeline (most recent)
        asian_line = s["odds_timeline"][-1]["asian_line"]

        computed = settle_asian_5class(upper_g, lower_g, asian_line)
        fixture_label = s["label"]["asian_result"]

        check(f"{mid}: computed={computed} vs fixture={fixture_label}",
              computed == fixture_label,
              f"upper={upper_g} lower={lower_g} line={asian_line}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    global PASS, FAIL
    PASS = 0
    FAIL = 0

    print("=" * 60)
    print("P0.3 Asian Handicap Label Tests")
    print("=" * 60)

    test_split_asian_line()
    test_settle_half_line()
    test_settle_asian_5class()
    test_settle_asian_3class()
    test_get_upper_lower_goals()
    test_fixture_labels()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL == 0:
        print("ASIAN HANDICAP LABEL TESTS: ALL CHECKS PASSED")
    else:
        print(f"ASIAN HANDICAP LABEL TESTS: {FAIL} FAILURE(S)")
    print("=" * 60)

    return FAIL == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
