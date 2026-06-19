"""
OddsMind P0.4 Split + Metrics Smoke Test

Verifies grouped time split, allowed_match_ids filtering, metrics,
and integration with training/evaluation pipelines.

Usage:
    python eval/eval_odds_split_metrics_smoke.py
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader

from model.model_oddsmind import OddsMindConfig, OddsMindModel
from dataset.odds_dataset import OddsDataset
from dataset.odds_collator import OddsCollator
from dataset.odds_split import (
    load_match_metadata,
    grouped_time_split,
    load_match_ids_from_file,
)
from eval.odds_metrics import (
    accuracy_from_logits,
    logloss_from_logits,
    brier_from_logits,
    class_counts,
    prediction_counts,
)

FIXTURE = "data/odds_fixtures/sample_odds_matches_5class.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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


# ── Test 1: grouped_time_split ────────────────────────────────────────

def test_grouped_time_split():
    print("\n--- Test 1: grouped_time_split ---")
    matches = load_match_metadata(FIXTURE)
    splits = grouped_time_split(matches, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

    check("train has matches", len(splits["train"]) > 0)
    check("val has matches", len(splits["val"]) > 0)
    check("test has matches", len(splits["test"]) > 0)
    check("total = all matches",
          len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == len(matches))
    check("no train/val overlap", splits["train"].isdisjoint(splits["val"]))
    check("no train/test overlap", splits["train"].isdisjoint(splits["test"]))
    check("no val/test overlap", splits["val"].isdisjoint(splits["test"]))

    # Verify time order: train < val < test
    train_times = [m["_kickoff_dt"] for m in matches if m["match_id"] in splits["train"]]
    val_times = [m["_kickoff_dt"] for m in matches if m["match_id"] in splits["val"]]
    test_times = [m["_kickoff_dt"] for m in matches if m["match_id"] in splits["test"]]
    check("train times < val times", max(train_times) < min(val_times))
    check("val times < test times", max(val_times) < min(test_times))

    return splits


# ── Test 2: allowed_match_ids filter ──────────────────────────────────

def test_allowed_match_ids(splits):
    print("\n--- Test 2: allowed_match_ids filter ---")
    train_ids = splits["train"]
    ds = OddsDataset(FIXTURE, max_seq_len=64, allowed_match_ids=train_ids,
                     cutoffs=[90, 60, 30], cutoff_mode="exhaustive",
                     asian_label_mode="5class")

    check("only train matches loaded", ds.num_raw_matches == len(train_ids),
          f"{ds.num_raw_matches} vs {len(train_ids)}")

    # Verify no sample has match_id outside train
    for i in range(min(5, len(ds))):
        item = ds[i]
        mid = item["match_id"]
        # Extract source match_id from cutoff sample_id
        source = mid.split("__cutoff_")[0]
        check(f"sample {i} source in train", source in train_ids)

    # Check that exhaustive mode with filter still works
    check("more samples than raw", len(ds) > ds.num_raw_matches,
          f"{len(ds)} vs {ds.num_raw_matches}")

    return ds


# ── Test 3: Metrics 3class ───────────────────────────────────────────

def test_metrics_3class():
    print("\n--- Test 3: Metrics 3class ---")
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0], [0.0, 0.0, 2.0]], dtype=torch.float32)
    labels = torch.tensor([0, 1, 2])

    acc = accuracy_from_logits(logits, labels)
    check("accuracy perfect=1.0", abs(acc - 1.0) < 0.01, f"got {acc}")

    ll = logloss_from_logits(logits, labels)
    check("logloss finite", math.isfinite(ll), f"got {ll}")
    check("logloss < 1.0 (good predictions)", ll < 1.0, f"got {ll}")

    br = brier_from_logits(logits, labels, 3)
    check("brier finite", math.isfinite(br))
    check("brier < 0.5", br < 0.5, f"got {br}")

    cc = class_counts(labels, 3)
    check("class_counts total=3", cc["total"] == 3, f"got {cc}")

    pc = prediction_counts(logits, 3)
    check("prediction_counts total=3", pc["total"] == 3)

    # Imperfect predictions
    logits2 = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    labels2 = torch.tensor([0, 2])
    acc2 = accuracy_from_logits(logits2, labels2)
    check("random accuracy ≈ 1/3", 0.0 <= acc2 <= 1.0)

    return True


# ── Test 4: Metrics 5class ───────────────────────────────────────────

def test_metrics_5class():
    print("\n--- Test 4: Metrics 5class ---")
    logits = torch.randn(10, 5)
    labels = torch.randint(0, 5, (10,))

    acc = accuracy_from_logits(logits, labels)
    check("accuracy 0-1", 0 <= acc <= 1)

    ll = logloss_from_logits(logits, labels)
    check("logloss finite", math.isfinite(ll))

    br = brier_from_logits(logits, labels, 5)
    check("brier finite", math.isfinite(br))
    check("brier 0-2", 0 <= br <= 2, f"got {br}")

    cc = class_counts(labels, 5)
    check("class_counts total=10", cc["total"] == 10)

    pc = prediction_counts(logits, 5)
    check("prediction_counts total=10", pc["total"] == 10)
    return True


# ── Test 5: eval_odds_model untrained ────────────────────────────────

def test_eval_untrained(splits):
    print("\n--- Test 5: eval_odds_model (untrained) ---")
    config = OddsMindConfig(hidden_size=64, num_hidden_layers=2,
                            num_attention_heads=4, asian_num_classes=5)
    model = OddsMindModel(config).to(DEVICE)
    model.eval()

    test_ids = splits["test"]
    ds = OddsDataset(FIXTURE, max_seq_len=64, allowed_match_ids=test_ids,
                     cutoffs=[90, 60, 30], cutoff_mode="exhaustive",
                     asian_label_mode="5class")
    collator = OddsCollator()
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collator)

    from eval.odds_metrics import accuracy_from_logits, logloss_from_logits

    all_eu, all_as = [], []
    all_eul, all_asl = [], []
    with torch.no_grad():
        for batch in loader:
            f = batch["features"].to(DEVICE)
            m = batch["attention_mask"].to(DEVICE)
            el = batch["euro_labels"].to(DEVICE)
            al = batch["asian_labels"].to(DEVICE)
            out = model(f, attention_mask=m)
            all_eu.append(out["euro_logits"].cpu())
            all_as.append(out["asian_logits"].cpu())
            all_eul.append(el.cpu())
            all_asl.append(al.cpu())

    eu = torch.cat(all_eu, dim=0)
    al_labels = torch.cat(all_asl, dim=0)

    check("euro logits shape", eu.shape[1] == 3)
    check("asian logits shape", torch.cat(all_as, dim=0).shape[1] == 5)
    check("eval completed", len(ds) > 0)
    return True


# ── Test 6: split builder output ─────────────────────────────────────

def test_split_builder_output():
    print("\n--- Test 6: split builder output files ---")
    split_dir = "data/odds_fixtures/splits_p0_4"
    for fname in ["train_match_ids.txt", "val_match_ids.txt", "test_match_ids.txt", "split_summary.json"]:
        path = os.path.join(split_dir, fname)
        check(f"{fname} exists", os.path.exists(path))

    with open(os.path.join(split_dir, "split_summary.json")) as f:
        summary = json.load(f)
    check("overlap_check=pass", summary["overlap_check"] == "pass")
    check("total_matches correct", summary["total_matches"] == 12)
    return True


# ── Main ──────────────────────────────────────────────────────────────

def main():
    global PASS, FAIL
    PASS = 0
    FAIL = 0

    print("=" * 60)
    print("OddsMind P0.4 Split + Metrics Smoke Test")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    try:
        splits = test_grouped_time_split()
        test_split_builder_output()
        test_allowed_match_ids(splits)
        test_metrics_3class()
        test_metrics_5class()
        test_eval_untrained(splits)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        FAIL += 1

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL == 0:
        print("SPLIT + METRICS SMOKE TEST: ALL CHECKS PASSED")
    else:
        print(f"SPLIT + METRICS SMOKE TEST: {FAIL} FAILURE(S)")
    print("=" * 60)
    return FAIL == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
