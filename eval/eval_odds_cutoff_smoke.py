"""
OddsMind P0.2 Cutoff Smoke Test

Verifies cutoff sample generation, filtering rules, and integration
with the existing training / inference pipeline.

Usage:
    python eval/eval_odds_cutoff_smoke.py
"""

import json
import math
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader

from model.model_oddsmind import OddsMindConfig, OddsMindModel
from dataset.odds_dataset import OddsDataset, EURO_MAP, ASIAN_MAP
from dataset.odds_collator import OddsCollator
from dataset.odds_cutoff import (
    filter_timeline_by_cutoff,
    build_cutoff_samples,
    build_exhaustive_cutoff_dataset,
)

warnings.filterwarnings("ignore")

FIXTURE_PATH = "data/odds_fixtures/sample_odds_matches.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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


def load_raw_matches():
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ── Test 1: filter_timeline_by_cutoff ───────────────────────────────────

def test_filter_no_future_leak():
    """cutoff=90 must exclude T-60 and T-30 events."""
    print("\n--- Test 1: filter_timeline_by_cutoff — no future leak ---")
    matches = load_raw_matches()

    # Find a match with events at multiple time points
    for m in matches:
        timeline = m["odds_timeline"]
        times = sorted([e["minutes_before_kickoff"] for e in timeline], reverse=True)
        if any(t <= 60 for t in times) and any(t >= 90 for t in times):
            break

    # cutoff=90: all events must be >= 90
    f90 = filter_timeline_by_cutoff(timeline, 90)
    check("cutoff=90: all events >= 90",
          all(e["minutes_before_kickoff"] >= 90 for e in f90))
    check("cutoff=90: no T-60 or T-30",
          not any(e["minutes_before_kickoff"] < 90 for e in f90))

    # cutoff=60: all events must be >= 60
    f60 = filter_timeline_by_cutoff(timeline, 60)
    check("cutoff=60: all events >= 60",
          all(e["minutes_before_kickoff"] >= 60 for e in f60))
    check("cutoff=60: no T-30 events",
          not any(e["minutes_before_kickoff"] < 60 for e in f60))

    # cutoff=30: includes events at or above 30, and at least as many as cutoff=60
    f30 = filter_timeline_by_cutoff(timeline, 30)
    check("cutoff=30: all events >= 30",
          all(e["minutes_before_kickoff"] >= 30 for e in f30))
    check("cutoff=30: >= count of cutoff=60 events",
          len(f30) >= len(f60),
          f"f30={len(f30)} vs f60={len(f60)}")

    # cutoff=0: all events
    f0 = filter_timeline_by_cutoff(timeline, 0)
    check("cutoff=0: same count as original",
          len(f0) == len(timeline))

    return True


# ── Test 2: build_cutoff_samples ──────────────────────────────────────

def test_build_cutoff_samples():
    """Exhaustive mode generates correct number of samples with unique IDs."""
    print("\n--- Test 2: build_cutoff_samples — exhaustive generation ---")
    matches = load_raw_matches()
    match = matches[0]
    match_id = match["match_id"]
    original_label = match["label"]

    samples = build_cutoff_samples(match, [90, 60, 30], min_events=1)

    # Check we got at least some samples
    check("at least 1 sample generated", len(samples) > 0,
          f"got {len(samples)}")

    for s in samples:
        c = int(s["cutoff_minutes"])
        check(f"sample_id unique: {s['sample_id']}",
              s["sample_id"] == f"{match_id}__cutoff_{c}")
        check(f"label unchanged for cutoff={c}",
              s["label"] == original_label)
        check(f"source_match_id preserved: {s['sample_id']}",
              s["source_match_id"] == match_id)
        check(f"all events >= {c} for cutoff={c}",
              all(e["minutes_before_kickoff"] >= c for e in s["odds_timeline"]))

    # Verify uniqueness
    ids = [s["sample_id"] for s in samples]
    check("all sample_ids unique", len(ids) == len(set(ids)))

    return True


# ── Test 3: OddsDataset exhaustive mode ───────────────────────────────

def test_dataset_exhaustive():
    """Dataset in exhaustive mode produces correct samples."""
    print("\n--- Test 3: OddsDataset exhaustive mode ---")
    ds = OddsDataset(
        FIXTURE_PATH,
        max_seq_len=64,
        cutoffs=[90, 60, 30],
        cutoff_mode="exhaustive",
        min_events=1,
    )

    check("more samples than raw matches", len(ds) > ds.num_raw_matches,
          f"samples={len(ds)} > raw={ds.num_raw_matches}")
    check("has cutoff_counts", len(ds.cutoff_counts) > 0)

    # First sample should be a valid cutoff sample
    item = ds[0]
    check("features is 2D", item["features"].dim() == 2)
    check("features has 7 cols", item["features"].shape[-1] == 7)
    check("euro_label valid", 0 <= item["euro_label"] <= 2)
    check("asian_label valid", 0 <= item["asian_label"] <= 2)
    check("match_id contains cutoff", "__cutoff_" in item["match_id"])

    return ds


# ── Test 4: OddsDataset random mode ──────────────────────────────────

def test_dataset_random():
    """Dataset in random mode produces valid samples."""
    print("\n--- Test 4: OddsDataset random mode ---")
    ds = OddsDataset(
        FIXTURE_PATH,
        max_seq_len=64,
        cutoffs=[90, 60, 30],
        cutoff_mode="random",
        min_events=1,
        seed=123,
    )

    check("sample count = raw match count", len(ds) == ds.num_raw_matches,
          f"{len(ds)} == {ds.num_raw_matches}")

    # Get a few samples — they should have different lengths
    items = [ds[i] for i in range(min(4, len(ds)))]
    for i, item in enumerate(items):
        check(f"random sample {i} valid", item["features"].dim() == 2)

    return ds


# ── Test 5: Collator with cutoff samples ─────────────────────────────

def test_collator_cutoff():
    """Collator handles variable-length cutoff samples."""
    print("\n--- Test 5: Collator with cutoff samples ---")
    ds = OddsDataset(FIXTURE_PATH, max_seq_len=64,
                     cutoffs=[90, 60, 30], cutoff_mode="exhaustive")
    collator = OddsCollator()

    batch = collator([ds[0], ds[1], ds[2]])

    check("features batch_size=3", batch["features"].shape[0] == 3)
    check("attention_mask shape matches", batch["attention_mask"].shape == batch["features"].shape[:2])
    check("euro_labels shape", batch["euro_labels"].shape == (3,))
    check("asian_labels shape", batch["asian_labels"].shape == (3,))
    return True


# ── Test 6: Model forward with cutoff batch ──────────────────────────

def test_model_cutoff_forward():
    """Model forward works with cutoff batch."""
    print("\n--- Test 6: Model forward with cutoff batch ---")
    config = OddsMindConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=4)
    model = OddsMindModel(config).to(DEVICE)

    ds = OddsDataset(FIXTURE_PATH, max_seq_len=64,
                     cutoffs=[90, 60, 30], cutoff_mode="exhaustive")
    collator = OddsCollator()
    batch = collator([ds[0], ds[1]])

    features = batch["features"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    euro_labels = batch["euro_labels"].to(DEVICE)
    asian_labels = batch["asian_labels"].to(DEVICE)

    out = model(features, attention_mask=attention_mask,
                euro_labels=euro_labels, asian_labels=asian_labels)

    check("euro_logits shape", out["euro_logits"].shape == (2, 3))
    check("asian_logits shape", out["asian_logits"].shape == (2, 3))
    check("loss computed", "loss" in out)
    check("loss > 0", out["loss"].item() > 0)
    return True


# ── Test 7: Cutoff train 1 epoch ─────────────────────────────────────

def test_cutoff_train():
    """Training loop works in exhaustive cutoff mode."""
    print("\n--- Test 7: Cutoff train 1 epoch ---")
    config = OddsMindConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4)
    model = OddsMindModel(config).to(DEVICE)

    ds = OddsDataset(FIXTURE_PATH, max_seq_len=64,
                     cutoffs=[90, 60, 30], cutoff_mode="exhaustive")
    collator = OddsCollator()
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    losses = []

    for batch in loader:
        features = batch["features"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        euro_labels = batch["euro_labels"].to(DEVICE)
        asian_labels = batch["asian_labels"].to(DEVICE)

        optimizer.zero_grad()
        out = model(features, attention_mask=attention_mask,
                    euro_labels=euro_labels, asian_labels=asian_labels)
        out["loss"].backward()
        optimizer.step()
        losses.append(out["loss"].item())

    check("at least 1 batch", len(losses) > 0)
    check("all losses finite", all(math.isfinite(l) for l in losses))
    print(f"  losses: {[f'{l:.4f}' for l in losses]}")
    return True


# ── Test 8: Backward compatibility ───────────────────────────────────

def test_backward_compat():
    """P0.1 mode still works (no cutoffs)."""
    print("\n--- Test 8: Backward compatibility (P0.1 mode) ---")
    ds = OddsDataset(FIXTURE_PATH, max_seq_len=64)
    check("default mode = none", ds.cutoff_mode == "none")
    check("sample count = raw count", len(ds) == ds.num_raw_matches)

    item = ds[0]
    check("features valid", item["features"].dim() == 2)
    check("euro_label valid", 0 <= item["euro_label"] <= 2)
    return True


# ── Test 9: Inference still compatible ───────────────────────────────

def test_inference_compat():
    """Inference still works with original JSONL input."""
    print("\n--- Test 9: Inference compatibility ---")
    config = OddsMindConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=4)
    model = OddsMindModel(config).to(DEVICE)
    model.eval()

    # Use old P0.1 style (no cutoffs)
    ds = OddsDataset(FIXTURE_PATH, max_seq_len=64)
    collator = OddsCollator()
    sample = collator([ds[0]])

    with torch.no_grad():
        out = model(
            sample["features"].to(DEVICE),
            attention_mask=sample["attention_mask"].to(DEVICE),
        )

    euro_probs = torch.softmax(out["euro_logits"], dim=-1)[0].cpu()
    check("euro_probs sum ≈ 1", abs(euro_probs.sum().item() - 1.0) < 0.001,
          f"sum={euro_probs.sum().item():.4f}")

    asian_probs = torch.softmax(out["asian_logits"], dim=-1)[0].cpu()
    check("asian_probs sum ≈ 1", abs(asian_probs.sum().item() - 1.0) < 0.001,
          f"sum={asian_probs.sum().item():.4f}")

    return True


# ── Main ──────────────────────────────────────────────────────────────

def main():
    global PASS, FAIL
    PASS = 0
    FAIL = 0

    print("=" * 60)
    print("OddsMind P0.2 Cutoff Smoke Test")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    try:
        test_filter_no_future_leak()
        test_build_cutoff_samples()
        test_dataset_exhaustive()
        test_dataset_random()
        test_collator_cutoff()
        test_model_cutoff_forward()
        test_cutoff_train()
        test_backward_compat()
        test_inference_compat()
    except Exception as e:
        print(f"\n[ERROR] Smoke test raised exception: {e}")
        import traceback
        traceback.print_exc()
        FAIL += 1

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL == 0:
        print("CUTOFF SMOKE TEST: ALL CHECKS PASSED")
    else:
        print(f"CUTOFF SMOKE TEST: {FAIL} FAILURE(S)")
    print("=" * 60)

    return FAIL == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
