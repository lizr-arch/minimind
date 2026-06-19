"""
OddsMind P0.5B Tabular Baseline Smoke Test

Tests feature extraction, tabular dataset, logistic/tiny_mlp models,
training, evaluation, and comparison.

Usage:
    python eval/eval_odds_tabular_baseline_smoke.py
"""

import json
import math
import os
import sys
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from dataset.odds_features import extract_features, feature_dim, FEATURE_NAMES
from dataset.odds_tabular_dataset import OddsTabularDataset
from model.odds_tabular_baselines import OddsLogisticRegression, OddsTinyMLP

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


FIXTURE = "data/odds_fixtures/sample_odds_matches_5class.jsonl"


# ── Test 1: Feature extraction ───────────────────────────────────────

def test_feature_extraction():
    print("\n--- Test 1: Feature extraction ---")
    dim = feature_dim()
    check("feature_dim > 0", dim > 0)
    check("FEATURE_NAMES length matches", len(FEATURE_NAMES) == dim)

    events = [{"minutes_before_kickoff": 1440, "euro_h": 2.0, "euro_d": 3.5, "euro_a": 4.0,
               "asian_line": -0.5, "upper_water": 0.9, "lower_water": 0.9},
              {"minutes_before_kickoff": 60, "euro_h": 1.8, "euro_d": 3.6, "euro_a": 4.5,
               "asian_line": -0.75, "upper_water": 0.85, "lower_water": 0.95}]
    feats = extract_features(events)
    check("shape matches dim", feats.shape == (dim,))
    check("all finite", torch.all(torch.isfinite(feats)).item())

    try:
        extract_features([])
        check("empty → ValueError", False)
    except ValueError:
        check("empty → ValueError", True)


# ── Test 2: Tabular dataset ──────────────────────────────────────────

def test_tabular_dataset():
    print("\n--- Test 2: Tabular dataset ---")
    ds = OddsTabularDataset(FIXTURE, cutoffs=[90, 60, 30], cutoff_mode="exhaustive",
                            asian_label_mode="5class")
    check("has samples", len(ds) > 0)
    item = ds[0]
    check("features key", "features" in item)
    check("features shape", item["features"].shape == (feature_dim(),))
    check("euro_label in range", 0 <= item["euro_label"] <= 2)
    check("asian_label in range", 0 <= item["asian_label"] <= 4)
    check("match_id present", "match_id" in item)


# ── Test 3: Logistic model forward ──────────────────────────────────

def test_logistic_forward():
    print("\n--- Test 3: Logistic forward ---")
    model = OddsLogisticRegression(asian_num_classes=5)
    x = torch.randn(4, feature_dim())
    out = model(x)
    check("euro_logits [4,3]", out["euro_logits"].shape == (4, 3))
    check("asian_logits [4,5]", out["asian_logits"].shape == (4, 5))


# ── Test 4: TinyMLP forward ─────────────────────────────────────────

def test_tiny_mlp_forward():
    print("\n--- Test 4: TinyMLP forward ---")
    model = OddsTinyMLP(hidden_size=32, asian_num_classes=3)
    x = torch.randn(4, feature_dim())
    out = model(x)
    check("euro_logits [4,3]", out["euro_logits"].shape == (4, 3))
    check("asian_logits [4,3]", out["asian_logits"].shape == (4, 3))


# ── Test 5: Logistic smoke train ────────────────────────────────────

def test_logistic_train():
    print("\n--- Test 5: Logistic smoke train ---")
    r = subprocess.run([
        sys.executable, "trainer/train_odds_tabular_baseline.py",
        "--data", FIXTURE, "--epochs", "1", "--batch-size", "4",
        "--model-type", "logistic", "--device", "cpu",
        "--cutoffs", "90,60,30", "--cutoff-mode", "exhaustive",
        "--asian-label-mode", "5class",
        "--train-match-ids", "data/odds_fixtures/splits_p0_4/train_match_ids.txt",
        "--val-match-ids", "data/odds_fixtures/splits_p0_4/val_match_ids.txt",
        "--eval-every-epoch",
        "--out-dir", "runs/oddsmind_p0_5b_logistic_smoke",
    ], capture_output=True, text=True)
    check("logistic train exit 0", r.returncode == 0, r.stderr[-200:] if r.stderr else "")
    return r.returncode == 0


# ── Test 6: TinyMLP smoke train ─────────────────────────────────────

def test_tiny_mlp_train():
    print("\n--- Test 6: TinyMLP smoke train ---")
    r = subprocess.run([
        sys.executable, "trainer/train_odds_tabular_baseline.py",
        "--data", FIXTURE, "--epochs", "1", "--batch-size", "4",
        "--model-type", "tiny_mlp", "--hidden-size", "32", "--device", "cpu",
        "--cutoffs", "90,60,30", "--cutoff-mode", "exhaustive",
        "--asian-label-mode", "5class",
        "--train-match-ids", "data/odds_fixtures/splits_p0_4/train_match_ids.txt",
        "--val-match-ids", "data/odds_fixtures/splits_p0_4/val_match_ids.txt",
        "--eval-every-epoch",
        "--out-dir", "runs/oddsmind_p0_5b_tiny_mlp_smoke",
    ], capture_output=True, text=True)
    check("tiny_mlp train exit 0", r.returncode == 0, r.stderr[-200:] if r.stderr else "")
    return r.returncode == 0


# ── Test 7: Tabular eval ────────────────────────────────────────────

def test_tabular_eval():
    print("\n--- Test 7: Tabular eval ---")
    r = subprocess.run([
        sys.executable, "eval/eval_odds_tabular_baseline.py",
        "--data", FIXTURE, "--untrained", "--model-type", "logistic",
        "--split-match-ids", "data/odds_fixtures/splits_p0_4/test_match_ids.txt",
        "--cutoffs", "90,60,30", "--cutoff-mode", "exhaustive",
        "--asian-label-mode", "5class", "--device", "cpu",
        "--out-json", "runs/oddsmind_p0_5b_logistic_smoke/tabular_eval.json",
    ], capture_output=True, text=True)
    check("tabular eval exit 0", r.returncode == 0, r.stderr[-200:] if r.stderr else "")
    try:
        data = json.loads(r.stdout.strip())
        check("eval has euro", "euro" in data)
        check("eval has asian", "asian" in data)
    except:
        check("parse JSON", False)
    return r.returncode == 0


# ── Test 8: Comparison script ───────────────────────────────────────

def test_comparison():
    print("\n--- Test 8: Comparison script ---")
    r = subprocess.run([
        sys.executable, "eval/compare_odds_evals.py",
        "--eval-json", "tabular=runs/oddsmind_p0_5b_logistic_smoke/tabular_eval.json",
        "--eval-json", "deterministic=runs/oddsmind_p0_5a_baseline_smoke/baseline_eval.json",
        "--out-json", "runs/oddsmind_p0_5b_logistic_smoke/comparison.json",
    ], capture_output=True, text=True)
    check("comparison exit 0", r.returncode == 0, r.stderr[-200:] if r.stderr else "")
    try:
        data = json.loads(r.stdout.strip())
        check("models key exists", "models" in data)
    except:
        check("parse JSON", False)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    global PASS, FAIL
    PASS = 0
    FAIL = 0

    print("=" * 60)
    print("OddsMind P0.5B Tabular Baseline Smoke Test")
    print("=" * 60)

    test_feature_extraction()
    test_tabular_dataset()
    test_logistic_forward()
    test_tiny_mlp_forward()
    test_logistic_train()
    test_tiny_mlp_train()
    test_tabular_eval()
    test_comparison()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL == 0:
        print("TABULAR BASELINE SMOKE TEST: ALL CHECKS PASSED")
    else:
        print(f"TABULAR BASELINE SMOKE TEST: {FAIL} FAILURE(S)")
    print("=" * 60)
    return FAIL == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
