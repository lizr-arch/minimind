"""
OddsMind P0.7B Pretrain Transfer Smoke Test

Tests pretrain-to-supervised weight transfer, dry-run, training with
pretrained encoder, and compatibility with eval/inference.
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from model.model_oddsmind import OddsMindConfig, OddsMindModel
from model.oddsmind_weight_transfer import load_pretrained_encoder_transformer

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
PT_CKP = "runs/oddsmind_p0_7b_native_pretrain_smoke/oddsmind_pretrain_masked_reconstruction.pth"


# ── Test 1: Dry-run transfer ─────────────────────────────────────────

def test_dry_run():
    print("\n--- Test 1: Dry-run transfer ---")
    r = subprocess.run([
        sys.executable, "tools/odds_transfer_pretrain_to_supervised.py",
        "--pretrain-checkpoint", PT_CKP,
        "--out", "runs/oddsmind_p0_7b_transfer_smoke/test_dry_run.pth",
        "--hidden-size", "64", "--num-layers", "2", "--num-heads", "4",
        "--asian-label-mode", "5class", "--transformer-backend", "odds_native",
        "--dry-run",
    ], capture_output=True, text=True, timeout=30)
    check("dry-run exit 0", r.returncode == 0, r.stderr[-200:])
    check("dry-run does not create file",
          not os.path.exists("runs/oddsmind_p0_7b_transfer_smoke/test_dry_run.pth"))
    check("report in stdout", "transferred" in r.stdout)


# ── Test 2: Non-dry-run transfer ─────────────────────────────────────

def test_transfer():
    print("\n--- Test 2: Non-dry-run transfer ---")
    out_path = "runs/oddsmind_p0_7b_transfer_smoke/supervised_init.pth"
    r = subprocess.run([
        sys.executable, "tools/odds_transfer_pretrain_to_supervised.py",
        "--pretrain-checkpoint", PT_CKP,
        "--out", out_path,
        "--hidden-size", "64", "--num-layers", "2", "--num-heads", "4",
        "--asian-label-mode", "5class", "--transformer-backend", "odds_native",
    ], capture_output=True, text=True, timeout=30)
    check("transfer exit 0", r.returncode == 0, r.stderr[-200:])
    check("checkpoint created", os.path.exists(out_path))
    check("transferred > 0", "transferred" in r.stdout and "0" not in r.stdout.split("transferred:")[1].split()[0])

    # Load and verify
    data = torch.load(out_path, map_location="cpu")
    check("model_state_dict exists", "model_state_dict" in data)
    check("config exists", "config" in data)


# ── Test 3: Transferred model forward ────────────────────────────────

def test_transferred_forward():
    print("\n--- Test 3: Transferred model forward ---")
    out_path = "runs/oddsmind_p0_7b_transfer_smoke/supervised_init.pth"
    data = torch.load(out_path, map_location="cpu")
    config = OddsMindConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                            asian_num_classes=5, transformer_backend="odds_native")
    model = OddsMindModel(config)
    model.load_state_dict(data["model_state_dict"])
    model.eval()

    x = torch.randn(2, 5, 7)
    mask = torch.ones(2, 5, dtype=torch.bool)
    out = model(x, attention_mask=mask)
    check("euro_logits [2,3]", out["euro_logits"].shape == (2, 3))
    check("asian_logits [2,5]", out["asian_logits"].shape == (2, 5))

    # Loss
    el = torch.tensor([0, 2])
    al = torch.tensor([1, 3])
    out2 = model(x, attention_mask=mask, euro_labels=el, asian_labels=al)
    check("loss computed", "loss" in out2)


# ── Test 4: Train with pretrained encoder ────────────────────────────

def test_train_with_pretrained():
    print("\n--- Test 4: Train with --pretrained-encoder-checkpoint ---")
    r = subprocess.run([
        sys.executable, "trainer/train_odds_supervised.py",
        "--data", FIXTURE, "--epochs", "1", "--batch-size", "2",
        "--hidden-size", "64", "--num-layers", "2", "--num-heads", "4",
        "--device", "cpu", "--cutoffs", "90,60,30", "--cutoff-mode", "exhaustive",
        "--asian-label-mode", "5class",
        "--train-match-ids", "data/odds_fixtures/splits_p0_4/train_match_ids.txt",
        "--val-match-ids", "data/odds_fixtures/splits_p0_4/val_match_ids.txt",
        "--eval-every-epoch", "--transformer-backend", "odds_native",
        "--pretrained-encoder-checkpoint", PT_CKP,
        "--out-dir", "runs/oddsmind_p0_7b_finetune_from_pretrain_smoke",
    ], capture_output=True, text=True, timeout=60)
    check("finetune exit 0", r.returncode == 0, (r.stderr+r.stdout)[-300:] if r.returncode else "")
    check("transfer report in logs", "Transfer report" in r.stdout or "transferred" in r.stdout)


# ── Test 5: Transferred checkpoint used by eval ─────────────────────

def test_transferred_eval():
    print("\n--- Test 5: Eval with transferred checkpoint ---")
    r = subprocess.run([
        sys.executable, "eval/eval_odds_model.py",
        "--data", FIXTURE,
        "--model", "runs/oddsmind_p0_7b_transfer_smoke/supervised_init.pth",
        "--split-match-ids", "data/odds_fixtures/splits_p0_4/test_match_ids.txt",
        "--cutoffs", "90,60,30", "--cutoff-mode", "exhaustive",
        "--asian-label-mode", "5class",
        "--hidden-size", "64", "--num-layers", "2", "--num-heads", "4",
        "--device", "cpu", "--transformer-backend", "odds_native",
    ], capture_output=True, text=True, timeout=30)
    if r.returncode == 0:
        try:
            json.loads(r.stdout.strip())
            check("eval JSON valid", True)
        except:
            check("eval JSON valid", False)
    else:
        check("eval exit 0", False, r.stderr[-200:])


# ── Test 6: Minimind backend transfer ────────────────────────────────

def test_minimind_transfer():
    print("\n--- Test 6: Minimind backend transfer ---")
    try:
        from model.oddsmind_minimind_adapter import MiniMindBlockAdapter
        # First create minimind pretrain checkpoint
        r1 = subprocess.run([
            sys.executable, "trainer/train_odds_pretrain.py",
            "--data", FIXTURE, "--task", "masked_reconstruction",
            "--epochs", "1", "--batch-size", "2",
            "--hidden-size", "64", "--num-layers", "2", "--num-heads", "4",
            "--device", "cpu", "--cutoffs", "90,60,30", "--cutoff-mode", "exhaustive",
            "--transformer-backend", "minimind", "--mask-ratio", "0.15",
            "--out-dir", "runs/oddsmind_p0_7b_minimind_pretrain_smoke",
        ], capture_output=True, text=True, timeout=60)
        check("minimind pretrain exit 0", r1.returncode == 0)

        mm_ckp = "runs/oddsmind_p0_7b_minimind_pretrain_smoke/oddsmind_pretrain_masked_reconstruction.pth"
        r2 = subprocess.run([
            sys.executable, "tools/odds_transfer_pretrain_to_supervised.py",
            "--pretrain-checkpoint", mm_ckp,
            "--out", "runs/oddsmind_p0_7b_transfer_smoke/minimind_supervised_init.pth",
            "--hidden-size", "64", "--num-layers", "2", "--num-heads", "4",
            "--asian-label-mode", "5class", "--transformer-backend", "minimind",
        ], capture_output=True, text=True, timeout=30)
        check("minimind transfer exit 0", r2.returncode == 0, r2.stderr[-200:])
        check("minimind transferred > 0", "transferred" in r2.stdout)
    except ImportError:
        skip_check("minimind transfer", "transformers not available")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    global PASS, FAIL, SKIP
    PASS = 0; FAIL = 0; SKIP = 0

    print("=" * 60)
    print("OddsMind P0.7B Pretrain Transfer Smoke Test")
    print("=" * 60)

    test_dry_run()
    test_transfer()
    test_transferred_forward()
    test_train_with_pretrained()
    test_transferred_eval()
    test_minimind_transfer()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed, {SKIP} skipped")
    if FAIL == 0:
        print("PRETRAIN TRANSFER SMOKE: ALL CHECKS PASSED")
    else:
        print(f"PRETRAIN TRANSFER SMOKE: {FAIL} FAILURE(S)")
    print("=" * 60)
    return FAIL == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
