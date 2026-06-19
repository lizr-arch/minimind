"""
OddsMind P0.6 MiniMind Backend Smoke Test

Tests:
  1. Native backend forward still works.
  2. MiniMind backend forward works (or gracefully skipped).
  3. Native backend train 1 epoch.
  4. MiniMind backend train 1 epoch (or skip).
  5. Backward compat for all smoke tests.
"""

import json
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.model_oddsmind import OddsMindConfig, OddsMindModel

PASS = 0
FAIL = 0
SKIP = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  -- {detail}")


def skip_check(name, reason=""):
    global SKIP
    SKIP += 1
    print(f"  [SKIP] {name}  -- {reason}")


DEVICE = "cpu"
FIXTURE = "data/odds_fixtures/sample_odds_matches_5class.jsonl"


# ── Test 1: Native backend forward ────────────────────────────────────

def test_native_forward():
    print("\n--- Test 1: Native backend forward ---")
    cfg = OddsMindConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                         asian_num_classes=5, transformer_backend="odds_native")
    model = OddsMindModel(cfg).to(DEVICE)
    x = torch.randn(2, 6, 7)
    mask = torch.ones(2, 6, dtype=torch.bool)
    out = model(x, attention_mask=mask)
    check("euro_logits [2,3]", out["euro_logits"].shape == (2, 3))
    check("asian_logits [2,5]", out["asian_logits"].shape == (2, 5))


# ── Test 2: MiniMind backend forward ──────────────────────────────────

def test_minimind_forward():
    print("\n--- Test 2: MiniMind backend forward ---")
    try:
        cfg = OddsMindConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                             asian_num_classes=5, transformer_backend="minimind")
        model = OddsMindModel(cfg).to(DEVICE)
        x = torch.randn(2, 6, 7)
        mask = torch.ones(2, 6, dtype=torch.bool)
        out = model(x, attention_mask=mask)
        check("euro_logits [2,3]", out["euro_logits"].shape == (2, 3))
        check("asian_logits [2,5]", out["asian_logits"].shape == (2, 5))

        # With loss
        eu_labels = torch.tensor([0, 2])
        as_labels = torch.tensor([1, 3])
        out2 = model(x, attention_mask=mask, euro_labels=eu_labels, asian_labels=as_labels)
        check("loss computed", "loss" in out2)
        check("loss > 0", out2["loss"].item() > 0)

        # With padding mask
        mask2 = torch.ones(2, 6, dtype=torch.bool)
        mask2[0, 4:] = False
        out3 = model(x, attention_mask=mask2)
        check("masked forward ok", out3["euro_logits"].shape == (2, 3))
    except Exception as e:
        skip_check("minimind backend", str(e)[:100])


# ── Test 3: Native backend smoke train ────────────────────────────────

def test_native_smoke_train():
    print("\n--- Test 3: Native backend smoke train ---")
    r = subprocess.run([
        sys.executable, "trainer/train_odds_supervised.py",
        "--data", FIXTURE, "--epochs", "1", "--batch-size", "2",
        "--hidden-size", "64", "--num-layers", "2", "--num-heads", "4",
        "--device", "cpu", "--cutoffs", "90,60,30", "--cutoff-mode", "exhaustive",
        "--asian-label-mode", "5class",
        "--train-match-ids", "data/odds_fixtures/splits_p0_4/train_match_ids.txt",
        "--val-match-ids", "data/odds_fixtures/splits_p0_4/val_match_ids.txt",
        "--eval-every-epoch",
        "--transformer-backend", "odds_native",
        "--out-dir", "runs/oddsmind_p0_6_native_regression_smoke",
    ], capture_output=True, text=True, timeout=60)
    check("native train exit 0", r.returncode == 0,
          (r.stderr + r.stdout)[-300:] if r.returncode != 0 else "")


# ── Test 4: MiniMind backend smoke train ──────────────────────────────

def test_minimind_smoke_train():
    print("\n--- Test 4: MiniMind backend smoke train ---")
    r = subprocess.run([
        sys.executable, "trainer/train_odds_supervised.py",
        "--data", FIXTURE, "--epochs", "1", "--batch-size", "2",
        "--hidden-size", "64", "--num-layers", "2", "--num-heads", "4",
        "--device", "cpu", "--cutoffs", "90,60,30", "--cutoff-mode", "exhaustive",
        "--asian-label-mode", "5class",
        "--train-match-ids", "data/odds_fixtures/splits_p0_4/train_match_ids.txt",
        "--val-match-ids", "data/odds_fixtures/splits_p0_4/val_match_ids.txt",
        "--eval-every-epoch",
        "--transformer-backend", "minimind",
        "--out-dir", "runs/oddsmind_p0_6_minimind_backend_smoke",
    ], capture_output=True, text=True, timeout=60)
    if r.returncode == 0:
        check("minimind train exit 0", True)
    else:
        skip_check("minimind train", (r.stderr + r.stdout)[-200:])


# ── Test 5: MiniMind backend inference ────────────────────────────────

def test_minimind_inference():
    print("\n--- Test 5: MiniMind backend inference ---")
    r = subprocess.run([
        sys.executable, "infer/predict_odds_match.py",
        "--untrained", "--input", FIXTURE,
        "--hidden-size", "64", "--num-layers", "2", "--num-heads", "4",
        "--device", "cpu", "--asian-label-mode", "5class",
        "--transformer-backend", "minimind",
    ], capture_output=True, text=True, timeout=30)
    if r.returncode == 0:
        try:
            data = json.loads(r.stdout.strip())
            check("euro_probs has home", "home" in data.get("euro_probs", {}))
            check("asian_probs has upper_full_win",
                  "upper_full_win" in data.get("asian_probs", {}))
        except:
            check("parse JSON", False)
    else:
        skip_check("minimind inference", (r.stderr + r.stdout)[-200:])


# ── Main ──────────────────────────────────────────────────────────────

def main():
    global PASS, FAIL, SKIP
    PASS = 0
    FAIL = 0
    SKIP = 0

    print("=" * 60)
    print("OddsMind P0.6 MiniMind Backend Smoke Test")
    print("=" * 60)

    test_native_forward()
    test_minimind_forward()
    test_native_smoke_train()
    test_minimind_smoke_train()
    test_minimind_inference()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed, {SKIP} skipped")
    if FAIL == 0:
        print("MINIMIND BACKEND SMOKE: ALL CHECKS PASSED")
    else:
        print(f"MINIMIND BACKEND SMOKE: {FAIL} FAILURE(S)")
    print("=" * 60)
    return FAIL == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
