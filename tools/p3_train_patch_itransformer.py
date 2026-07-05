"""
P3: OddsPatch-iTransformer training script.

Trains the channel-independent feature embedding + cross-market Transformer
on v6_event schema data with time-bucketed core euro features.

Target: val_logloss ≤ 0.930 (match RawPooledMLP baseline).

Usage:
    python tools/p3_train_patch_itransformer.py \
        --data data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl \
        --train-ids data/odds_real/splits_v6/train_match_ids.txt \
        --val-ids data/odds_real/splits_v6/val_match_ids.txt \
        --epochs 50 --batch-size 64 --lr 0.001 \
        --d-model 64 --n-layers 2 --n-heads 4 --d-ff 128 --dropout 0.1 \
        --device cuda --out-dir runs/p3_patch_itransformer
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.odds_patch_itransformer import OddsPatchITransformer, bucketize_sample
from dataset.odds_dataset import OddsDataset
from eval.odds_baselines_v2 import Phase1Baseline
from eval.odds_metrics import (
    accuracy_from_probs,
    logloss_from_probs,
    brier_from_probs,
    ece_from_probs,
    class_counts,
    prediction_counts_from_probs,
)

EURO_MAP = {"home": 0, "draw": 1, "away": 2}


# ── Data helpers ──────────────────────────────────────────────────────

def load_split_ids(path: str) -> set:
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            mid = line.strip()
            if mid:
                ids.add(mid)
    return ids


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prepare_output_dir(out_dir: str, allow_overwrite: bool = False) -> None:
    existing_outputs = [
        os.path.join(out_dir, "report.json"),
        os.path.join(out_dir, "best_model.pth"),
    ]
    if not allow_overwrite and any(os.path.exists(path) for path in existing_outputs):
        raise FileExistsError(f"Refusing to overwrite existing run outputs in {out_dir}")
    os.makedirs(out_dir, exist_ok=True)


def build_bucketed_dataset(samples: list, dataset: OddsDataset):
    """Pre-compute bucketed features for all samples."""
    X_list, y_list = [], []
    for i in range(len(dataset)):
        raw_tl = dataset.samples[i].get("raw_timeline", [])
        if not raw_tl:
            raw_tl = dataset.samples[i].get("odds_timeline", [])
        xb = bucketize_sample(raw_tl)
        X_list.append(xb)
        y_list.append(dataset[i]["euro_label"])
    return torch.stack(X_list), torch.tensor(y_list, dtype=torch.long)


def _validation_match_ids_for_baseline(samples: list, val_ids: set) -> list:
    selected = []
    seen = set()
    for sample in samples:
        mid = sample.get("match_id")
        if mid and mid in val_ids and mid not in seen:
            selected.append(mid)
            seen.add(mid)
    return selected if selected else sorted(val_ids)


def _pass_fail_suffix(passed: bool) -> str:
    return " PASS" if passed else " FAIL"


# ── Training ──────────────────────────────────────────────────────────

def train_epoch(model, X_train, y_train, optimizer, batch_size, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0
    perm = torch.randperm(len(X_train))

    for start in range(0, len(X_train), batch_size):
        idx = perm[start : start + batch_size]
        xb = X_train[idx].to(device)
        yb = y_train[idx].to(device)

        logits = model(xb)
        loss = F.cross_entropy(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(-1) == yb).sum().item()
        total += len(yb)
        n_batches += 1

    return total_loss / n_batches, correct / total


@torch.no_grad()
def evaluate(model, X_val, y_val, batch_size, device):
    model.eval()
    all_probs = []

    for start in range(0, len(X_val), batch_size):
        xb = X_val[start : start + batch_size].to(device)
        logits = model(xb)
        probs = F.softmax(logits, dim=-1).clamp(1e-9, 1 - 1e-9)
        all_probs.append(probs.cpu())

    probs = torch.cat(all_probs, dim=0)
    labels = y_val
    C = 3

    return {
        "accuracy": accuracy_from_probs(probs, labels),
        "logloss": logloss_from_probs(probs, labels),
        "brier": brier_from_probs(probs, labels, C),
        "ece": ece_from_probs(probs, labels, C)["ece"],
        "label_counts": class_counts(labels, C),
        "prediction_counts": prediction_counts_from_probs(probs, C),
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="P3 OddsPatch-iTransformer Training")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--train-ids", type=str, required=True)
    parser.add_argument("--val-ids", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=str, default="runs/p3_patch_itransformer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of samples (0=all)")
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    set_deterministic_seed(args.seed)
    prepare_output_dir(args.out_dir, allow_overwrite=args.allow_overwrite)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ──
    train_ids = load_split_ids(args.train_ids)
    val_ids = load_split_ids(args.val_ids)
    print(f"Train IDs: {len(train_ids)}, Val IDs: {len(val_ids)}")

    train_ds = OddsDataset(
        args.data,
        max_seq_len=args.max_seq_len,
        feature_schema_version="v6_event",
        allowed_match_ids=train_ids,
        min_events=1,
    )
    val_ds = OddsDataset(
        args.data,
        max_seq_len=args.max_seq_len,
        feature_schema_version="v6_event",
        allowed_match_ids=val_ids,
        min_events=1,
    )
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    if args.max_samples > 0:
        train_ds.samples = train_ds.samples[:args.max_samples]
        val_ds.samples = val_ds.samples[:min(args.max_samples, len(val_ds.samples))]
        print(f"Limited to {len(train_ds)} train / {len(val_ds)} val samples")

    # ── Pre-compute bucketed features ──
    print("Building bucketed train features...")
    X_train, y_train = build_bucketed_dataset(train_ds.samples, train_ds)
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")

    print("Building bucketed val features...")
    X_val, y_val = build_bucketed_dataset(val_ds.samples, val_ds)
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")

    # ── Model ──
    model = OddsPatchITransformer(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    # ── Baseline ──
    baseline = Phase1Baseline()

    # ── Majority baseline ──
    label_counts_train = torch.bincount(y_train, minlength=3)
    majority_class = label_counts_train.argmax().item()
    majority_acc = (y_val == majority_class).float().mean().item()
    print(f"Majority baseline: class={majority_class}, val_acc={majority_acc:.4f}")

    # ── Training loop ──
    best_val_logloss = float("inf")
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, X_train, y_train, optimizer, args.batch_size, device,
        )
        val_metrics = evaluate(
            model, X_val, y_val, args.batch_size * 2, device,
        )
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_accuracy": val_metrics["accuracy"],
            "val_logloss": val_metrics["logloss"],
            "val_brier": val_metrics["brier"],
            "val_ece": val_metrics["ece"],
            "lr": scheduler.get_last_lr()[0],
        })

        is_best = val_metrics["logloss"] < best_val_logloss
        if is_best:
            best_val_logloss = val_metrics["logloss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "d_model": args.d_model,
                        "n_heads": args.n_heads,
                        "n_layers": args.n_layers,
                        "d_ff": args.d_ff,
                        "dropout": args.dropout,
                    },
                    "best_epoch": best_epoch,
                    "best_val_logloss": best_val_logloss,
                },
                os.path.join(args.out_dir, "best_model.pth"),
            )

        print(
            f"Epoch {epoch:3d} | "
            f"loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val acc={val_metrics['accuracy']:.4f} "
            f"logloss={val_metrics['logloss']:.4f} "
            f"brier={val_metrics['brier']:.4f} "
            f"ece={val_metrics['ece']:.4f}"
            + (" *" if is_best else "")
        )

    # ── Final evaluation with best model ──
    ckpt_path = os.path.join(args.out_dir, "best_model.pth")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    final_val = evaluate(model, X_val, y_val, args.batch_size * 2, device)

    # ── Baseline comparison ──
    baseline_match_ids = set(_validation_match_ids_for_baseline(val_ds.samples, val_ids))
    val_rows = _load_raw_rows(args.data, baseline_match_ids)

    bprobs, blabels = [], []
    for row in val_rows:
        try:
            q = baseline.compute_euro_prior(
                float(row["close_euro_h"]),
                float(row["close_euro_d"]),
                float(row["close_euro_a"]),
            )
            bprobs.append(torch.tensor(q))
        except Exception:
            bprobs.append(torch.full((3,), 1.0 / 3.0))
        blabels.append(EURO_MAP.get(row["label"]["euro_result"], 0))
    bp = torch.stack(bprobs)
    bl = torch.tensor(blabels)
    baseline_val = {
        "accuracy": accuracy_from_probs(bp, bl),
        "logloss": logloss_from_probs(bp, bl),
        "brier": brier_from_probs(bp, bl, 3),
        "ece": ece_from_probs(bp, bl, 3)["ece"],
    }

    # ── Report ──
    report = {
        "config": vars(args),
        "data": {
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "feature_dim": "9_core × 2_agg × 10_buckets",
        },
        "model": {"n_params": n_params},
        "best_epoch": best_epoch,
        "best_val_logloss": best_val_logloss,
        "final_val": final_val,
        "baseline_val": baseline_val,
        "comparison": {
            "delta_accuracy": round(
                final_val["accuracy"] - baseline_val["accuracy"], 6
            ),
            "delta_logloss": round(
                final_val["logloss"] - baseline_val["logloss"], 6
            ),
            "delta_brier": round(
                final_val["brier"] - baseline_val["brier"], 6
            ),
            "delta_ece": round(
                final_val["ece"] - baseline_val["ece"], 6
            ),
        },
        "history": history,
    }
    report_path = os.path.join(args.out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved to {report_path}")

    # ── Summary ──
    print(f"\n=== Final Comparison ===")
    print(
        f"  Model   | acc={final_val['accuracy']:.4f}  "
        f"logloss={final_val['logloss']:.4f}  "
        f"brier={final_val['brier']:.4f}  "
        f"ece={final_val['ece']:.4f}"
    )
    print(
        f"  Baseline| acc={baseline_val['accuracy']:.4f}  "
        f"logloss={baseline_val['logloss']:.4f}  "
        f"brier={baseline_val['brier']:.4f}  "
        f"ece={baseline_val['ece']:.4f}"
    )
    print(
        f"  Delta   | acc={report['comparison']['delta_accuracy']:+.4f}  "
        f"logloss={report['comparison']['delta_logloss']:+.4f}"
    )
    print(f"\n  Target: val_logloss ≤ 0.930")
    print(
        f"  Result: val_logloss = {best_val_logloss:.4f}"
        + _pass_fail_suffix(best_val_logloss <= 0.930)
    )


def _load_raw_rows(jsonl_path: str, match_ids: set) -> list:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            if row["match_id"] in match_ids:
                rows.append(row)
                if len(rows) >= len(match_ids):
                    break
    return rows


if __name__ == "__main__":
    main()
