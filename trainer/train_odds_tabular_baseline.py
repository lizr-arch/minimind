"""
OddsMind Tabular Baseline Training Script (P0.5B)

Trains a logistic regression or tiny MLP on extracted tabular features.

Usage:
    python trainer/train_odds_tabular_baseline.py \
        --data ...5class.jsonl --model-type logistic \
        --cutoffs 90,60,30 --cutoff-mode exhaustive --asian-label-mode 5class \
        --train-match-ids .../train_match_ids.txt \
        --val-match-ids .../val_match_ids.txt --eval-every-epoch \
        --out-dir runs/p0_5b_logistic
"""

import argparse
import math
import os
import random
import sys
import time
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset.odds_tabular_dataset import OddsTabularDataset
from dataset.odds_split import load_match_ids_from_file
from model.odds_tabular_baselines import OddsLogisticRegression, OddsTinyMLP

warnings.filterwarnings("ignore")


# ── Utilities ──────────────────────────────────────────────────────────

def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def Logger(content):
    print(content)


def setup_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def collate_tabular(batch):
    features = torch.stack([item["features"] for item in batch])
    euro_labels = torch.tensor([item["euro_label"] for item in batch])
    asian_labels = torch.tensor([item["asian_label"] for item in batch])
    return {"features": features, "euro_labels": euro_labels, "asian_labels": asian_labels}


# ── Training ──────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, epoch, args):
    model.train()
    total_loss = 0.0
    steps = len(loader)
    for step, batch in enumerate(loader, start=1):
        features = batch["features"].to(args.device)
        euro_labels = batch["euro_labels"].to(args.device)
        asian_labels = batch["asian_labels"].to(args.device)

        lr = get_lr((epoch - 1) * steps + step, args.epochs * steps, args.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        out = model(features)
        euro_loss = F.cross_entropy(out["euro_logits"], euro_labels)
        asian_loss = F.cross_entropy(out["asian_logits"], asian_labels)
        loss = euro_loss + asian_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if step % max(1, steps // 5) == 0 or step == steps:
            Logger(f"Epoch {epoch}/{args.epochs} [{step}/{steps}] loss={loss.item():.4f} lr={lr:.6f}")

    return total_loss / steps


def main():
    parser = argparse.ArgumentParser(description="OddsMind Tabular Baseline Train")
    parser.add_argument("--data", type=str, default="data/odds_fixtures/sample_odds_matches_5class.jsonl")
    parser.add_argument("--model-type", type=str, default="logistic", choices=["logistic", "tiny_mlp"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="out_odds_tabular")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cutoffs", type=str, default="")
    parser.add_argument("--cutoff-mode", type=str, default="none")
    parser.add_argument("--asian-label-mode", type=str, default="3class")
    parser.add_argument("--min-events", type=int, default=1)
    parser.add_argument("--train-match-ids", type=str, default="")
    parser.add_argument("--val-match-ids", type=str, default="")
    parser.add_argument("--eval-every-epoch", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=64)
    args = parser.parse_args()

    setup_seed(args.seed)

    cutoffs = None
    if args.cutoffs:
        cutoffs = [float(c.strip()) for c in args.cutoffs.split(",") if c.strip()]

    train_ids = None
    val_ids = None
    if args.train_match_ids:
        train_ids = load_match_ids_from_file(args.train_match_ids)
    if args.val_match_ids:
        val_ids = load_match_ids_from_file(args.val_match_ids)

    asian_num_classes = 5 if args.asian_label_mode == "5class" else 3

    # Model
    if args.model_type == "logistic":
        model = OddsLogisticRegression(asian_num_classes=asian_num_classes).to(args.device)
    else:
        model = OddsTinyMLP(hidden_size=args.hidden_size, asian_num_classes=asian_num_classes).to(args.device)

    Logger(f"Model: {args.model_type}, params: {sum(p.numel() for p in model.parameters()):,}")

    # Dataset
    ds = OddsTabularDataset(
        args.data, max_seq_len=args.max_seq_len,
        cutoffs=cutoffs, cutoff_mode=args.cutoff_mode,
        min_events=args.min_events, asian_label_mode=args.asian_label_mode,
        allowed_match_ids=train_ids, seed=args.seed,
    )
    Logger(f"Train: {ds.num_raw_matches} raw matches, {len(ds)} samples")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_tabular)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, loader, optimizer, epoch, args)
        Logger(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")

        if args.eval_every_epoch and val_ids:
            model.eval()
            from eval.odds_metrics import accuracy_from_logits
            val_ds = OddsTabularDataset(
                args.data, max_seq_len=args.max_seq_len,
                cutoffs=cutoffs, cutoff_mode=args.cutoff_mode,
                min_events=args.min_events, asian_label_mode=args.asian_label_mode,
                allowed_match_ids=val_ids, seed=args.seed,
            )
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_tabular)
            eu_logits, as_logits = [], []
            eu_labels, as_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    f = batch["features"].to(args.device)
                    out = model(f)
                    eu_logits.append(out["euro_logits"].cpu())
                    as_logits.append(out["asian_logits"].cpu())
                    eu_labels.append(batch["euro_labels"])
                    as_labels.append(batch["asian_labels"])
            eu_l = torch.cat(eu_logits, 0)
            as_l = torch.cat(as_logits, 0)
            eu_lb = torch.cat(eu_labels, 0)
            as_lb = torch.cat(as_labels, 0)
            Logger(f"  Val [{len(val_ds)} samples]: euro_acc={accuracy_from_logits(eu_l, eu_lb):.4f} asian_acc={accuracy_from_logits(as_l, as_lb):.4f}")
            model.train()

    os.makedirs(args.out_dir, exist_ok=True)
    ckp_name = f"odds_tabular_{args.model_type}.pth"
    torch.save(model.state_dict(), os.path.join(args.out_dir, ckp_name))
    Logger(f"Saved to {args.out_dir}/{ckp_name}")


if __name__ == "__main__":
    main()
