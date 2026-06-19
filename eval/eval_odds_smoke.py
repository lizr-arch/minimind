"""
OddsMind P0.1 Smoke Test

Verifies the minimal scaffold end-to-end without training a real model.

Checks:
  1. Fixture JSONL can be read.
  2. Dataset returns correct fields.
  3. Collator can pad variable-length sequences.
  4. OddsMindModel forward produces euro_logits, asian_logits.
  5. Loss can be computed.
  6. Training loop can run 1 epoch on fixture data.
  7. Inference can output JSON probabilities.
  8. Original MiniMind files are untouched (manual check).

Usage:
    python eval/eval_odds_smoke.py
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


def test_fixture_readable():
    """1. Fixture JSONL can be read."""
    print("\n--- Test 1: Fixture JSONL readable ---")
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    check("file exists and has lines", len(lines) > 0, f"got {len(lines)} lines")
    check("first sample has match_id", "match_id" in lines[0])
    check("first sample has odds_timeline", "odds_timeline" in lines[0])
    check("first sample has label", "label" in lines[0])
    return lines


def test_dataset():
    """2. Dataset returns correct fields."""
    print("\n--- Test 2: Dataset fields ---")
    ds = OddsDataset(FIXTURE_PATH, max_seq_len=64)
    check("dataset has samples", len(ds) > 0, f"got {len(ds)}")

    sample = ds[0]
    check("has features key", "features" in sample)
    check("features is 2D tensor", sample["features"].dim() == 2)
    check("features has 7 columns", sample["features"].shape[-1] == 7)
    check("has euro_label", "euro_label" in sample)
    check("euro_label in range", 0 <= sample["euro_label"] <= 2)
    check("has asian_label", "asian_label" in sample)
    check("asian_label in range", 0 <= sample["asian_label"] <= 2)
    check("has match_id", "match_id" in sample)
    return ds


def test_collator():
    """3. Collator can pad variable-length sequences."""
    print("\n--- Test 3: Collator padding ---")
    ds = OddsDataset(FIXTURE_PATH, max_seq_len=64)
    collator = OddsCollator()

    # Pick two samples with different lengths
    samples = [ds[0], ds[1]]
    batch = collator(samples)

    check("features shape is 3D", batch["features"].dim() == 3)
    check("features batch_size=2", batch["features"].shape[0] == 2)
    check("attention_mask shape matches", batch["attention_mask"].shape == batch["features"].shape[:2])
    check("euro_labels shape", batch["euro_labels"].shape == (2,))
    check("asian_labels shape", batch["asian_labels"].shape == (2,))

    # Verify padding
    seq_lens = [s["features"].shape[0] for s in samples]
    max_len = max(seq_lens)
    check("max_len matches max(seq_lens)", batch["features"].shape[1] == max_len)
    for i, sl in enumerate(seq_lens):
        mask_sum = batch["attention_mask"][i].sum().item()
        check(f"sample {i} mask sum = seq_len", mask_sum == sl, f"{mask_sum} != {sl}")

    return batch


def test_model_forward():
    """4. Model forward produces euro_logits, asian_logits.  5. Loss computed."""
    print("\n--- Test 4 & 5: Model forward + loss ---")
    config = OddsMindConfig(hidden_size=256, num_hidden_layers=4, num_attention_heads=8)
    model = OddsMindModel(config).to(DEVICE)

    # Dummy batch
    features = torch.randn(2, 10, 7, device=DEVICE)
    mask = torch.ones(2, 10, dtype=torch.bool, device=DEVICE)
    mask[0, 7:] = False  # first sample shorter
    euro_labels = torch.tensor([0, 2], device=DEVICE)
    asian_labels = torch.tensor([1, 0], device=DEVICE)

    out = model(features, attention_mask=mask, euro_labels=euro_labels, asian_labels=asian_labels)

    check("euro_logits shape", out["euro_logits"].shape == (2, 3), f"got {out['euro_logits'].shape}")
    check("asian_logits shape", out["asian_logits"].shape == (2, 3), f"got {out['asian_logits'].shape}")
    check("loss computed", "loss" in out, "loss missing")
    check("loss is scalar", out["loss"].dim() == 0)
    check("loss > 0", out["loss"].item() > 0)
    check("euro_loss computed", "euro_loss" in out)
    check("asian_loss computed", "asian_loss" in out)
    return model


def test_training_smoke():
    """6. Training loop can run 1 epoch on fixture data."""
    print("\n--- Test 6: Smoke training 1 epoch ---")
    config = OddsMindConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=4)
    model = OddsMindModel(config).to(DEVICE)

    ds = OddsDataset(FIXTURE_PATH, max_seq_len=64)
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
        out = model(features, attention_mask=attention_mask, euro_labels=euro_labels, asian_labels=asian_labels)
        out["loss"].backward()
        optimizer.step()
        losses.append(out["loss"].item())

    check("at least 1 batch processed", len(losses) > 0)
    check("all losses finite", all(math.isfinite(l) for l in losses))
    print(f"  losses: {[f'{l:.4f}' for l in losses]}")
    return model


def test_inference_output():
    """7. Inference can output JSON probabilities."""
    print("\n--- Test 7: Inference output format ---")
    import math
    config = OddsMindConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=4)
    model = OddsMindModel(config).to(DEVICE)
    model.eval()

    ds = OddsDataset(FIXTURE_PATH, max_seq_len=64)
    collator = OddsCollator()
    sample = collator([ds[0]])

    with torch.no_grad():
        out = model(
            sample["features"].to(DEVICE),
            attention_mask=sample["attention_mask"].to(DEVICE),
        )

    euro_probs = torch.softmax(out["euro_logits"], dim=-1)[0].cpu()
    asian_probs = torch.softmax(out["asian_logits"], dim=-1)[0].cpu()

    euro_map_rev = {v: k for k, v in EURO_MAP.items()}
    asian_map_rev = {v: k for k, v in ASIAN_MAP.items()}

    euro_pred = euro_map_rev[int(torch.argmax(euro_probs).item())]
    asian_pred = asian_map_rev[int(torch.argmax(asian_probs).item())]

    result = {
        "euro_probs": {
            "home": round(euro_probs[0].item(), 4),
            "draw": round(euro_probs[1].item(), 4),
            "away": round(euro_probs[2].item(), 4),
        },
        "asian_probs": {
            "upper": round(asian_probs[0].item(), 4),
            "push": round(asian_probs[1].item(), 4),
            "lower": round(asian_probs[2].item(), 4),
        },
        "euro_prediction": euro_pred,
        "asian_prediction": asian_pred,
    }

    check("euro_probs has home/draw/away", set(result["euro_probs"].keys()) == {"home", "draw", "away"})
    check("asian_probs has upper/push/lower", set(result["asian_probs"].keys()) == {"upper", "push", "lower"})
    check("euro_prediction is string", isinstance(result["euro_prediction"], str))
    check("asian_prediction is string", isinstance(result["asian_prediction"], str))

    print("  Inference result:")
    print(f"  {json.dumps(result, indent=2)}")
    return result


# ── Main ────────────────────────────────────────────────────────────────

def main():
    global PASS, FAIL
    PASS = 0
    FAIL = 0

    print("=" * 60)
    print("OddsMind P0.1 Smoke Test")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    try:
        test_fixture_readable()
        test_dataset()
        test_collator()
        test_model_forward()
        test_training_smoke()
        test_inference_output()
    except Exception as e:
        print(f"\n[ERROR] Smoke test raised exception: {e}")
        import traceback
        traceback.print_exc()
        FAIL += 1

    # Boundary check 8: MiniMind files untouched (manual verification)
    print("\n--- Boundary Check 8: MiniMind files untouched ---")
    print("  Manual verification required: check that original MiniMind files")
    print("  (model/model_minimind.py, dataset/lm_dataset.py, trainer/*.py)")
    print("  have NOT been modified.  All new code is in separate files.")

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL == 0:
        print("SMOKE TEST: ALL CHECKS PASSED")
    else:
        print(f"SMOKE TEST: {FAIL} FAILURE(S)")
    print("=" * 60)

    return FAIL == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
