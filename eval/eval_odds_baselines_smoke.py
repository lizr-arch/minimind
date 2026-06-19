"""
OddsMind P0.5A Baseline Smoke Test

Verifies all deterministic baselines and probability metrics.

Usage:
    python eval/eval_odds_baselines_smoke.py
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from eval.odds_baselines import (
    lowest_odds_euro,
    implied_prob_euro,
    low_water_asian_3class,
    low_water_asian_5class,
    uniform_baseline,
    euro_probs_to_tensor,
    asian_probs_to_tensor,
)
from eval.odds_metrics import (
    accuracy_from_probs,
    logloss_from_probs,
    brier_from_probs,
)

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  -- {detail}")


def make_events(euro_h, euro_d, euro_a, asian_line=0.0, upper_water=0.9, lower_water=0.9,
                minutes=60):
    return [{
        "minutes_before_kickoff": minutes,
        "euro_h": euro_h, "euro_d": euro_d, "euro_a": euro_a,
        "asian_line": asian_line, "upper_water": upper_water, "lower_water": lower_water,
    }]


# ── Test 1: lowest_odds_euro ──────────────────────────────────────────

def test_lowest_odds_euro():
    print("\n--- Test 1: lowest_odds_euro ---")
    e = make_events(2.0, 3.5, 4.0)
    probs = lowest_odds_euro(e)
    check("home lowest → home", probs["home"] == 1.0 and probs["draw"] == 0.0 and probs["away"] == 0.0)

    e2 = make_events(3.0, 2.0, 4.0)
    probs2 = lowest_odds_euro(e2)
    check("draw lowest → draw", probs2["draw"] == 1.0)

    e3 = make_events(4.0, 3.0, 2.0)
    probs3 = lowest_odds_euro(e3)
    check("away lowest → away", probs3["away"] == 1.0)

    # Empty
    try:
        lowest_odds_euro([])
        check("empty → ValueError", False)
    except ValueError:
        check("empty → ValueError", True)


# ── Test 2: implied_prob_euro ─────────────────────────────────────────

def test_implied_prob_euro():
    print("\n--- Test 2: implied_prob_euro ---")
    e = make_events(2.0, 4.0, 4.0)
    probs = implied_prob_euro(e)
    total = probs["home"] + probs["draw"] + probs["away"]
    check("sum ≈ 1", abs(total - 1.0) < 0.001, f"sum={total}")
    check("home > draw", probs["home"] > probs["draw"])
    check("draw ≈ away", abs(probs["draw"] - probs["away"]) < 0.01)

    # Illegal odds
    try:
        implied_prob_euro(make_events(0, 3.0, 4.0))
        check("zero odds → ValueError", False)
    except ValueError:
        check("zero odds → ValueError", True)

    try:
        implied_prob_euro(make_events(-1, 3.0, 4.0))
        check("negative odds → ValueError", False)
    except ValueError:
        check("negative odds → ValueError", True)


# ── Test 3: low_water_asian_3class ────────────────────────────────────

def test_low_water_asian_3class():
    print("\n--- Test 3: low_water_asian_3class ---")
    e = make_events(2.0, 3.0, 4.0, upper_water=0.85, lower_water=1.05)
    probs = low_water_asian_3class(e)
    check("upper water lower → upper", probs["upper"] == 1.0)

    e2 = make_events(2.0, 3.0, 4.0, upper_water=1.05, lower_water=0.85)
    probs2 = low_water_asian_3class(e2)
    check("lower water lower → lower", probs2["lower"] == 1.0)

    e3 = make_events(2.0, 3.0, 4.0, upper_water=0.900, lower_water=0.902)
    probs3 = low_water_asian_3class(e3, eps=0.005)
    check("close waters → push", probs3["push"] == 1.0)


# ── Test 4: low_water_asian_5class ────────────────────────────────────

def test_low_water_asian_5class():
    print("\n--- Test 4: low_water_asian_5class ---")
    e = make_events(2.0, 3.0, 4.0, upper_water=0.85, lower_water=1.05)
    probs = low_water_asian_5class(e)
    check("upper water lower → upper_full_win", probs["upper_full_win"] == 1.0)
    check("half_win=0", probs["upper_half_win"] == 0.0)
    check("half_loss=0", probs["upper_half_loss"] == 0.0)

    e2 = make_events(2.0, 3.0, 4.0, upper_water=1.05, lower_water=0.85)
    probs2 = low_water_asian_5class(e2)
    check("lower water lower → upper_full_loss", probs2["upper_full_loss"] == 1.0)

    total = sum(probs2.values())
    check("sum ≈ 1", abs(total - 1.0) < 0.001)


# ── Test 5: uniform_baseline ──────────────────────────────────────────

def test_uniform():
    print("\n--- Test 5: uniform_baseline ---")
    u3 = uniform_baseline(3)
    check("3class sum≈1", abs(sum(u3.values()) - 1.0) < 0.001)
    check("3class each ≈ 1/3", all(abs(v - 1/3) < 0.001 for v in u3.values()))

    u5 = uniform_baseline(5)
    check("5class sum≈1", abs(sum(u5.values()) - 1.0) < 0.001)
    check("5class each ≈ 0.2", all(abs(v - 0.2) < 0.001 for v in u5.values()))


# ── Test 6: probs to tensor ───────────────────────────────────────────

def test_probs_to_tensor():
    print("\n--- Test 6: probs_to_tensor ---")
    eur = implied_prob_euro(make_events(2.0, 3.0, 4.0))
    t = euro_probs_to_tensor(eur)
    check("euro tensor shape", t.shape == (3,))
    check("euro tensor sum ≈ 1", abs(t.sum().item() - 1.0) < 0.001)

    asn3 = low_water_asian_3class(make_events(2.0, 3.0, 4.0))
    t3 = asian_probs_to_tensor(asn3, 3)
    check("asian 3 tensor shape", t3.shape == (3,))

    asn5 = low_water_asian_5class(make_events(2.0, 3.0, 4.0))
    t5 = asian_probs_to_tensor(asn5, 5)
    check("asian 5 tensor shape", t5.shape == (5,))


# ── Test 7: prob metrics 3class ──────────────────────────────────────

def test_prob_metrics_3class():
    print("\n--- Test 7: prob metrics 3class ---")
    probs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    labels = torch.tensor([0, 1, 2])
    acc = accuracy_from_probs(probs, labels)
    check("perfect acc=1", abs(acc - 1.0) < 0.01)

    ll = logloss_from_probs(probs, labels)
    check("logloss finite", math.isfinite(ll))

    br = brier_from_probs(probs, labels, 3)
    check("brier finite", math.isfinite(br))
    check("brier ≈ 0 for perfect", br < 0.01, f"got {br}")


# ── Test 8: prob metrics 5class ──────────────────────────────────────

def test_prob_metrics_5class():
    print("\n--- Test 8: prob metrics 5class ---")
    probs = torch.rand(10, 5)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    labels = torch.randint(0, 5, (10,))
    acc = accuracy_from_probs(probs, labels)
    check("acc 0-1", 0 <= acc <= 1)
    ll = logloss_from_probs(probs, labels)
    check("logloss finite", math.isfinite(ll))
    br = brier_from_probs(probs, labels, 5)
    check("brier finite", math.isfinite(br))


# ── Test 9: eval_odds_baselines on fixture ────────────────────────────

def test_eval_baselines_script():
    print("\n--- Test 9: eval_odds_baselines script outputs JSON ---")
    import subprocess
    result = subprocess.run([
        sys.executable, "eval/eval_odds_baselines.py",
        "--data", "data/odds_fixtures/sample_odds_matches_5class.jsonl",
        "--split-match-ids", "data/odds_fixtures/splits_p0_4/test_match_ids.txt",
        "--cutoffs", "90,60,30",
        "--cutoff-mode", "exhaustive",
        "--asian-label-mode", "5class",
    ], capture_output=True, text=True)
    out = result.stdout.strip()
    try:
        data = json.loads(out)
        check("num_samples > 0", data.get("num_samples", 0) > 0)
        check("baselines key exists", "baselines" in data)
        check("by_cutoff key exists", "by_cutoff" in data)
        check("euro_lowest_odds exists", "euro_lowest_odds" in data["baselines"])
        check("euro_implied_prob exists", "euro_implied_prob" in data["baselines"])
        check("asian_low_water exists", "asian_low_water" in data["baselines"])
        check("uniform exists", "uniform" in data["baselines"])
    except Exception as e:
        check("parse JSON", False, str(e))
        print("  stdout:", out[:500])


# ── Main ──────────────────────────────────────────────────────────────

def main():
    global PASS, FAIL
    PASS = 0
    FAIL = 0

    print("=" * 60)
    print("OddsMind P0.5A Baseline Smoke Test")
    print("=" * 60)

    test_lowest_odds_euro()
    test_implied_prob_euro()
    test_low_water_asian_3class()
    test_low_water_asian_5class()
    test_uniform()
    test_probs_to_tensor()
    test_prob_metrics_3class()
    test_prob_metrics_5class()
    test_eval_baselines_script()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL == 0:
        print("BASELINE SMOKE TEST: ALL CHECKS PASSED")
    else:
        print(f"BASELINE SMOKE TEST: {FAIL} FAILURE(S)")
    print("=" * 60)
    return FAIL == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
