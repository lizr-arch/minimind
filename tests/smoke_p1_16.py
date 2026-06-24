"""P1.16 Integration Smoke Test

Covers:
  1. Cutoff assertion catches violations
  2. v3 missing_mask shapes are correct and backward compat holds
  3. No-vig baselines compute and return valid probabilities
  4. ECE computation on known-uniform predictions yields ~0
  5. Per-league grouping on fixture data works
  6. All three pooling modes pass a forward pass
  7. Existing eval_odds_smoke.py still passes unchanged (manual check)
"""

import json
import math
import os
import sys

sys.path.insert(0, ".")

import torch
from torch.utils.data import DataLoader

from dataset.odds_cutoff import assert_cutoff_integrity, filter_timeline_by_cutoff
from dataset.odds_dataset import (
    OddsDataset, _event_to_missing_mask, _event_has_euro,
    _event_has_asian, _event_has_over_under, _build_features_and_labels,
    EURO_MAP, ASIAN_MAP_5CLASS,
)
from dataset.odds_collator import OddsCollator
from model.model_oddsmind import OddsMindConfig, OddsMindModel
from eval.odds_baselines import (
    open_no_vig_euro, close_no_vig_euro,
    open_no_vig_asian, close_no_vig_asian,
    euro_probs_to_tensor, asian_probs_to_tensor,
)
from eval.odds_metrics import ece_from_probs, ece_from_logits

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIXTURE_PATH = "data/odds_fixtures/sample_odds_matches_5class.jsonl"

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


# ── Test 1: Cutoff assertion ────────────────────────────────────────────

def test_cutoff_assertion():
    print("\n=== Test 1: Cutoff assertion ===")

    # Good timeline: all events >= cutoff 60
    good_timeline = [
        {"minutes_before_kickoff": 120},
        {"minutes_before_kickoff": 90},
        {"minutes_before_kickoff": 60},
    ]
    try:
        assert_cutoff_integrity(good_timeline, 60, "test_good")
        check("good timeline passes assertion", True)
    except AssertionError as e:
        check("good timeline passes assertion", False, str(e))

    # Bad timeline: one event < cutoff
    bad_timeline = [
        {"minutes_before_kickoff": 90},
        {"minutes_before_kickoff": 45},  # violation for cutoff=60
        {"minutes_before_kickoff": 30},
    ]
    try:
        assert_cutoff_integrity(bad_timeline, 60, "test_bad")
        check("bad timeline raises assertion", False, "should have raised")
    except AssertionError:
        check("bad timeline raises assertion", True)

    # cutoff=0: always passes
    try:
        assert_cutoff_integrity(bad_timeline, 0, "test_zero")
        check("cutoff=0 always passes", True)
    except AssertionError:
        check("cutoff=0 always passes", False)

    # filter_timeline_by_cutoff produces valid timelines
    filtered = filter_timeline_by_cutoff(bad_timeline, 60)
    try:
        assert_cutoff_integrity(filtered, 60, "test_filtered")
        check("filter_timeline_by_cutoff output is valid", True)
    except AssertionError:
        check("filter_timeline_by_cutoff output is valid", False)

    # Verify filtered only contains events >= 60
    all_ok = all(e["minutes_before_kickoff"] >= 60 for e in filtered)
    check("filtered events all >= 60", all_ok, str(filtered))


# ── Test 2: Missing mask (v3 schema) ─────────────────────────────────────

def test_missing_mask():
    print("\n=== Test 2: Missing mask (v3 schema) ===")

    # Test individual event mask
    event_euro = {"minutes_before_kickoff": 90, "euro_h": 2.0, "euro_d": 3.0,
                   "euro_a": 4.0, "asian_line": 0.5, "upper_water": 0.95,
                   "lower_water": 0.85}
    mask = _event_to_missing_mask(event_euro)
    check("mask is list of 13", len(mask) == 13, f"got {len(mask)}")
    check("minutes_before_kickoff present", mask[0] == 1.0)
    check("euro_h present", mask[1] == 1.0)
    check("asian_line present", mask[4] == 1.0)
    check("over_under_line missing", mask[7] == 0.0)
    check("has_euro/asian/ou always present", all(m == 1.0 for m in mask[10:]))

    # Test event without asian
    event_no_asian = {"minutes_before_kickoff": 60, "euro_h": 1.8, "euro_d": 3.5,
                       "euro_a": 5.0}
    mask2 = _event_to_missing_mask(event_no_asian)
    check("asian_line missing without water", mask2[4] == 0.0)
    check("upper_water missing", mask2[5] == 0.0)
    check("lower_water missing", mask2[6] == 0.0)

    # Test asian_line=0 with water (real flat handicap)
    event_flat = {"minutes_before_kickoff": 30, "asian_line": 0.0,
                   "upper_water": 0.90, "lower_water": 0.90}
    mask3 = _event_to_missing_mask(event_flat)
    check("asian_line=0 with water → present", mask3[4] == 1.0)

    # Test asian_line=0 without water → missing
    event_zero_no_water = {"minutes_before_kickoff": 30, "asian_line": 0.0}
    mask4 = _event_to_missing_mask(event_zero_no_water)
    check("asian_line=0 without water → missing", mask4[4] == 0.0)

    # Test v3 dataset returns missing_mask
    ds = OddsDataset(FIXTURE_PATH, max_seq_len=64, feature_schema_version="v3",
                     asian_label_mode="5class")
    sample = ds[0]
    check("v3 sample has missing_mask", "missing_mask" in sample)
    if "missing_mask" in sample:
        fmask = sample["missing_mask"]
        check("missing_mask same shape as features", fmask.shape == sample["features"].shape)
        check("missing_mask values in [0,1]", (fmask >= 0).all() and (fmask <= 1).all())

    # Test backward compat: v1/v2 no missing_mask
    ds_v1 = OddsDataset(FIXTURE_PATH, max_seq_len=64, feature_schema_version="v1",
                        asian_label_mode="5class")
    check("v1 sample has no missing_mask", "missing_mask" not in ds_v1[0])

    ds_v2 = OddsDataset(FIXTURE_PATH, max_seq_len=64, feature_schema_version="v2",
                        asian_label_mode="5class")
    check("v2 sample has no missing_mask", "missing_mask" not in ds_v2[0])

    # Test collator with v3
    collator = OddsCollator()
    batch = collator([ds[i] for i in range(min(3, len(ds)))])
    check("v3 batch has missing_mask", "missing_mask" in batch)
    if "missing_mask" in batch:
        check("batch missing_mask shape", batch["missing_mask"].shape == batch["features"].shape)

    # Test collator with v1 (no missing_mask)
    batch_v1 = collator([ds_v1[i] for i in range(min(3, len(ds_v1)))])
    check("v1 batch has no missing_mask", "missing_mask" not in batch_v1)


# ── Test 3: No-vig baselines ────────────────────────────────────────────

def test_no_vig_baselines():
    print("\n=== Test 3: No-vig baselines ===")

    events = [
        {"minutes_before_kickoff": 1440, "euro_h": 2.50, "euro_d": 3.20,
         "euro_a": 2.80, "asian_line": 0.0, "upper_water": 0.95, "lower_water": 0.95},
        {"minutes_before_kickoff": 60, "euro_h": 2.10, "euro_d": 3.40,
         "euro_a": 3.50, "asian_line": -0.25, "upper_water": 0.90, "lower_water": 0.92},
    ]

    # Open no-vig (first event)
    open_probs = open_no_vig_euro(events)
    tensor = euro_probs_to_tensor(open_probs)
    check("open_no_vig sums to ~1", abs(tensor.sum().item() - 1.0) < 1e-5)
    check("open_no_vig uses first event", open_probs["home"] > 0)
    # First event: 1/2.5=0.4, 1/3.2=0.3125, 1/2.8=0.357, total=1.07
    # probs: 0.4/1.07≈0.374, 0.3125/1.07≈0.292, 0.357/1.07≈0.334
    check("open_no_vig home ≈ 0.37", abs(open_probs["home"] - 0.374) < 0.01)

    # Close no-vig (last event)
    close_probs = close_no_vig_euro(events)
    tensor2 = euro_probs_to_tensor(close_probs)
    check("close_no_vig sums to ~1", abs(tensor2.sum().item() - 1.0) < 1e-5)
    # Last event: 1/2.1=0.476, 1/3.4=0.294, 1/3.5=0.286, total=1.056
    check("close_no_vig home > open", close_probs["home"] > open_probs["home"])

    # Asian no-vig
    asn_close = close_no_vig_asian(events, num_classes=3)
    tasn = asian_probs_to_tensor(asn_close, 3)
    check("asian close no-vig sums to ~1", abs(tasn.sum().item() - 1.0) < 1e-5)
    # 1/0.90 + 1/0.92 = 1.111 + 1.087 = 2.198
    # upper=1.111/2.198≈0.506, lower=1.087/2.198≈0.494
    check("asian no-vig push=0", asn_close["push"] == 0.0)
    check("asian no-vig upper≈0.506", abs(asn_close["upper"] - 0.506) < 0.01)

    # Fallback: missing data → uniform
    empty = []
    try:
        open_no_vig_euro(empty)
        check("empty events raises", False, "should raise ValueError")
    except ValueError:
        check("empty events raises ValueError", True)


# ── Test 4: ECE calibration ─────────────────────────────────────────────

def test_ece():
    print("\n=== Test 4: ECE calibration ===")

    # Perfectly calibrated: confidence = accuracy
    # For 3 classes, if we always predict with prob [0.5, 0.3, 0.2] and are
    # right 50% of the time, ECE should be ~0
    n = 1000
    probs = torch.full((n, 3), 1.0 / 3.0)  # uniform, confidence=0.333
    labels = torch.randint(0, 3, (n,))       # random labels
    result = ece_from_probs(probs, labels, 3, n_bins=10)
    # With uniform probs and random labels, accuracy ≈ 1/3, conf = 1/3 → ECE ≈ 0
    check("uniform probs → ECE ≈ 0", abs(result["ece"]) < 0.05, f"got {result['ece']}")

    # Over-confident: always 1.0 but only ~1/3 correct
    over_probs = torch.zeros(n, 3)
    over_probs[:, 0] = 1.0  # always predict class 0 with 100% confidence
    result2 = ece_from_probs(over_probs, labels, 3)
    check("over-confident → ECE > 0", result2["ece"] > 0.3, f"got {result2['ece']}")

    # Test from logits
    logits = torch.zeros(n, 3)
    logits[:, 0] = 100.0  # softmax → nearly [1,0,0]
    result3 = ece_from_logits(logits, labels, 3)
    check("ece_from_logits works", result3["ece"] > 0.3, f"got {result3['ece']}")

    check("ece returns bins list", "bins" in result)
    check("ece returns 10 bins", len(result["bins"]) == 10)


# ── Test 5: Per-league grouping ─────────────────────────────────────────

def test_per_league():
    print("\n=== Test 5: Per-league grouping ===")

    ds = OddsDataset(FIXTURE_PATH, max_seq_len=64, feature_schema_version="v2",
                     asian_label_mode="5class")
    collator = OddsCollator()
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collator)

    all_league_ids = []
    for batch in loader:
        all_league_ids.extend(batch.get("league_ids", []))

    check("league_ids collected", len(all_league_ids) > 0, f"got {len(all_league_ids)}")
    unique_leagues = set(lid for lid in all_league_ids if lid)
    check("multiple leagues in fixture", len(unique_leagues) >= 2, f"got {unique_leagues}")

    # Verify league_id is in sample items
    sample = ds[0]
    check("sample has league_id key", "league_id" in sample, f"keys: {list(sample.keys())}")


# ── Test 6: Pooling modes forward pass ──────────────────────────────────

def test_pooling_modes():
    print("\n=== Test 6: Pooling modes forward pass ===")

    configs = [
        OddsMindConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                       pooling_mode="mean"),
        OddsMindConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                       pooling_mode="attention"),
        OddsMindConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                       pooling_mode="cls"),
    ]

    features = torch.randn(2, 8, 13, device=DEVICE)
    mask = torch.ones(2, 8, dtype=torch.bool, device=DEVICE)
    mask[0, 5:] = False

    for cfg in configs:
        mode = cfg.pooling_mode
        model = OddsMindModel(cfg).to(DEVICE)
        model.eval()
        with torch.no_grad():
            out = model(features, attention_mask=mask)
        check(f"{mode}: euro_logits shape", out["euro_logits"].shape == (2, 3),
              f"got {out['euro_logits'].shape}")
        check(f"{mode}: asian_logits shape", out["asian_logits"].shape == (2, 3),
              f"got {out['asian_logits'].shape}")
        check(f"{mode}: logits finite", torch.isfinite(out["euro_logits"]).all())

    # Test without attention_mask
    model_mean = OddsMindModel(configs[0]).to(DEVICE)
    model_mean.eval()
    with torch.no_grad():
        out = model_mean(features, attention_mask=None)
    check("no mask: euro_logits shape", out["euro_logits"].shape == (2, 3))


# ── Test 7: Model forward with missing_mask ──────────────────────────────

def test_model_missing_mask():
    print("\n=== Test 7: Model forward with missing_mask ===")

    cfg = OddsMindConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4)
    model = OddsMindModel(cfg).to(DEVICE)
    model.eval()

    features = torch.randn(2, 8, 13, device=DEVICE)
    mask = torch.ones(2, 8, dtype=torch.bool, device=DEVICE)
    missing_mask = torch.ones(2, 8, 13, device=DEVICE)
    missing_mask[0, 3:, 7:10] = 0.0  # over/under missing for later events

    with torch.no_grad():
        out = model(features, attention_mask=mask, missing_mask=missing_mask)
    check("forward with missing_mask works", "euro_logits" in out)
    check("logits finite with missing_mask", torch.isfinite(out["euro_logits"]).all())

    # Without missing_mask (backward compat)
    with torch.no_grad():
        out2 = model(features, attention_mask=mask)
    check("forward without missing_mask works", "euro_logits" in out2)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    global PASS, FAIL
    PASS = 0
    FAIL = 0

    print("=" * 60)
    print("P1.16 OddsMind Evaluation & Mask Hardening — Smoke Test")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    try:
        test_cutoff_assertion()
        test_missing_mask()
        test_no_vig_baselines()
        test_ece()
        test_per_league()
        test_pooling_modes()
        test_model_missing_mask()
    except Exception as e:
        print(f"\n[ERROR] Smoke test raised exception: {e}")
        import traceback
        traceback.print_exc()
        FAIL += 1

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL == 0:
        print("P1.16 SMOKE TEST: ALL CHECKS PASSED")
    else:
        print(f"P1.16 SMOKE TEST: {FAIL} FAILURE(S)")
    print("=" * 60)

    # Manual check reminder
    print("\n--- Manual Verification ---")
    print("  1. Run: python eval/eval_odds_smoke.py (should still pass)")
    print("  2. Run: python eval/eval_odds_baselines.py --data data/odds_fixtures/sample_odds_matches_5class.jsonl")
    print("  3. Check no MiniMind files modified")

    return FAIL == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
