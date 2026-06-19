"""
OddsMind P0.7 Pretrain Smoke Test

Tests pretrain tasks, dataset, model, training, eval.
"""

import json
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from dataset.odds_pretrain_tasks import (
    build_masked_reconstruction_sample,
    build_next_event_sample,
)
from dataset.odds_pretrain_dataset import OddsPretrainDataset
from dataset.odds_pretrain_collator import OddsPretrainCollator
from model.odds_pretrain_heads import OddsReconstructionHead, OddsEventPredictionHead
from model.model_oddsmind_pretrain import OddsMindPretrainModel, OddsPretrainConfig

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


FIXTURE = "data/odds_fixtures/sample_odds_matches_5class.jsonl"


# ── Test 1: Masked reconstruction task ────────────────────────────────

def test_masked_task():
    print("\n--- Test 1: Masked reconstruction task ---")
    feats = torch.randn(5, 7)
    out = build_masked_reconstruction_sample(feats, mask_ratio=0.3, seed=42)
    check("corrupted shape", out["corrupted_features"].shape == (5, 7))
    check("target features shape", out["target_features"].shape == (5, 7))
    check("target_mask shape", out["target_mask"].shape == (5, 7))
    check("at least 1 masked", out["target_mask"].sum() >= 1)
    check("masked positions differ", not torch.allclose(out["corrupted_features"], out["target_features"]))


# ── Test 2: Next event task ──────────────────────────────────────────

def test_next_event_task():
    print("\n--- Test 2: Next event task ---")
    feats = torch.randn(4, 7)
    out = build_next_event_sample(feats)
    check("input shape [3,7]", out["input_features"].shape == (3, 7))
    check("target shape [7]", out["target_event"].shape == (7,))

    # Target must not be in input
    try:
        build_next_event_sample(torch.randn(1, 7))
        check("too few events raises", False)
    except ValueError:
        check("too few events raises", True)


# ── Test 3: Pretrain dataset ────────────────────────────────────────

def test_pretrain_dataset():
    print("\n--- Test 3: Pretrain dataset ---")
    ds = OddsPretrainDataset(FIXTURE, cutoffs=[90, 60, 30], cutoff_mode="exhaustive",
                             task="masked_reconstruction")
    check("has samples", len(ds) > 0)
    item = ds[0]
    check("masked features", "features" in item)
    check("target_features", "target_features" in item)
    check("target_mask", "target_mask" in item)


# ── Test 4: Collator ────────────────────────────────────────────────

def test_collator():
    print("\n--- Test 4: Collator ---")
    ds = OddsPretrainDataset(FIXTURE, cutoffs=[90, 60, 30], cutoff_mode="exhaustive",
                             task="masked_reconstruction")
    collator = OddsPretrainCollator()
    batch = collator([ds[0], ds[1]])
    check("features in batch", "features" in batch)
    check("target_features in batch", "target_features" in batch)
    check("target_mask in batch", "target_mask" in batch)
    check("attention_mask in batch", "attention_mask" in batch)


# ── Test 5: Pretrain model forward ──────────────────────────────────

def test_pretrain_model():
    print("\n--- Test 5: Pretrain model forward ---")
    cfg = OddsPretrainConfig(hidden_size=64, num_layers=2, num_heads=4,
                             task="masked_reconstruction")
    model = OddsMindPretrainModel(cfg)
    x = torch.randn(2, 5, 7)
    mask = torch.ones(2, 5, dtype=torch.bool)
    tgt_f = torch.randn(2, 5, 7)
    tgt_m = torch.zeros(2, 5, 7, dtype=torch.bool)
    tgt_m[0, 2, 1] = True
    out = model(features=x, attention_mask=mask, target_features=tgt_f, target_mask=tgt_m)
    check("loss scalar", "loss" in out and out["loss"].dim() == 0)
    check("loss > 0", out["loss"].item() > 0)

    # next_event
    cfg2 = OddsPretrainConfig(hidden_size=64, num_layers=2, num_heads=4, task="next_event")
    model2 = OddsMindPretrainModel(cfg2)
    out2 = model2(features=x[:, :4], attention_mask=mask[:, :4],
                  target_event=torch.randn(2, 7))
    check("next_event loss", "loss" in out2 and out2["loss"].dim() == 0)


# ── Test 6: Smoke train ─────────────────────────────────────────────

def test_smoke_train():
    print("\n--- Test 6: Smoke train masked_reconstruction ---")
    r = subprocess.run([
        sys.executable, "trainer/train_odds_pretrain.py",
        "--data", FIXTURE, "--task", "masked_reconstruction",
        "--epochs", "1", "--batch-size", "2",
        "--hidden-size", "64", "--num-layers", "2", "--num-heads", "4",
        "--device", "cpu", "--cutoffs", "90,60,30", "--cutoff-mode", "exhaustive",
        "--train-match-ids", "data/odds_fixtures/splits_p0_4/train_match_ids.txt",
        "--val-match-ids", "data/odds_fixtures/splits_p0_4/val_match_ids.txt",
        "--eval-every-epoch", "--transformer-backend", "odds_native",
        "--mask-ratio", "0.15",
        "--out-dir", "runs/oddsmind_p0_7_masked_pretrain_smoke",
    ], capture_output=True, text=True, timeout=60)
    check("masked train exit 0", r.returncode == 0, (r.stderr+r.stdout)[-300:] if r.returncode else "")

    print("\n--- Test 6b: Smoke train next_event ---")
    r2 = subprocess.run([
        sys.executable, "trainer/train_odds_pretrain.py",
        "--data", FIXTURE, "--task", "next_event",
        "--epochs", "1", "--batch-size", "2",
        "--hidden-size", "64", "--num-layers", "2", "--num-heads", "4",
        "--device", "cpu", "--cutoffs", "90,60,30", "--cutoff-mode", "exhaustive",
        "--train-match-ids", "data/odds_fixtures/splits_p0_4/train_match_ids.txt",
        "--transformer-backend", "odds_native",
        "--out-dir", "runs/oddsmind_p0_7_next_event_smoke",
    ], capture_output=True, text=True, timeout=60)
    check("next_event train exit 0", r2.returncode == 0, (r2.stderr+r2.stdout)[-300:] if r2.returncode else "")


# ── Test 7: Pretrain eval ───────────────────────────────────────────

def test_pretrain_eval():
    print("\n--- Test 7: Pretrain eval ---")
    r = subprocess.run([
        sys.executable, "eval/eval_odds_pretrain.py",
        "--data", FIXTURE, "--task", "masked_reconstruction", "--untrained",
        "--split-match-ids", "data/odds_fixtures/splits_p0_4/test_match_ids.txt",
        "--cutoffs", "90,60,30", "--cutoff-mode", "exhaustive",
        "--hidden-size", "64", "--num-layers", "2", "--num-heads", "4",
        "--device", "cpu", "--transformer-backend", "odds_native",
        "--mask-ratio", "0.15",
    ], capture_output=True, text=True, timeout=30)
    if r.returncode == 0:
        try:
            data = json.loads(r.stdout.strip())
            check("eval has loss", "loss" in data)
        except:
            check("parse JSON", False)
    else:
        check("eval exit 0", False, r.stderr[-200:])


# ── Test 8: Minimind backend pretrain (if available) ─────────────────

def test_minimind_pretrain():
    print("\n--- Test 8: Minimind backend pretrain ---")
    try:
        from model.oddsmind_minimind_adapter import MiniMindBlockAdapter
        r = subprocess.run([
            sys.executable, "trainer/train_odds_pretrain.py",
            "--data", FIXTURE, "--task", "masked_reconstruction",
            "--epochs", "1", "--batch-size", "2",
            "--hidden-size", "64", "--num-layers", "2", "--num-heads", "4",
            "--device", "cpu", "--cutoffs", "90,60,30", "--cutoff-mode", "exhaustive",
            "--transformer-backend", "minimind", "--mask-ratio", "0.15",
            "--out-dir", "runs/oddsmind_p0_7_minimind_masked_pretrain_smoke",
        ], capture_output=True, text=True, timeout=60)
        check("minimind pretrain exit 0", r.returncode == 0, (r.stderr+r.stdout)[-300:] if r.returncode else "")
    except ImportError:
        skip_check("minimind pretrain", "transformers not available")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    global PASS, FAIL, SKIP
    PASS = 0; FAIL = 0; SKIP = 0

    print("=" * 60)
    print("OddsMind P0.7 Pretrain Smoke Test")
    print("=" * 60)

    test_masked_task()
    test_next_event_task()
    test_pretrain_dataset()
    test_collator()
    test_pretrain_model()
    test_smoke_train()
    test_pretrain_eval()
    test_minimind_pretrain()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed, {SKIP} skipped")
    if FAIL == 0:
        print("PRETRAIN SMOKE TEST: ALL CHECKS PASSED")
    else:
        print(f"PRETRAIN SMOKE TEST: {FAIL} FAILURE(S)")
    print("=" * 60)
    return FAIL == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
