"""
OddsMind — supervised smoke training script (P0.2).

This is a minimal training loop for verification purposes.  It uses
fixture data, runs 1 epoch by default, and saves a single checkpoint.

Usage:
    # P0.1 mode (no cutoffs)
    python trainer/train_odds_supervised.py \
        --data data/odds_fixtures/sample_odds_matches.jsonl \
        --epochs 1 --batch-size 4 --device cpu --out-dir out_odds

    # P0.2 exhaustive cutoff mode
    python trainer/train_odds_supervised.py \
        --data data/odds_fixtures/sample_odds_matches.jsonl \
        --epochs 1 --batch-size 2 --hidden-size 64 --num-layers 2 \
        --num-heads 4 --device cpu \
        --cutoffs 90,60,30 --cutoff-mode exhaustive \
        --out-dir runs/oddsmind_p0_2_cutoff_smoke
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import math
import random
import time
import warnings

import torch
from torch import optim
from torch.utils.data import DataLoader

from model.model_oddsmind import OddsMindConfig, OddsMindModel
from dataset.odds_dataset import OddsDataset
from dataset.odds_collator import OddsCollator

warnings.filterwarnings("ignore")


# ── Inlined utilities (avoid pulling trainer_utils + its deps) ─────────

def get_lr(current_step, total_steps, lr):
    """Cosine warmup-decay LR schedule (same formula as MiniMind)."""
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def Logger(content):
    """Print helper."""
    print(content)


def setup_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ── Training loop ──────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, epoch, args):
    """Single epoch training loop."""
    model.train()
    total_loss = 0.0
    total_euro = 0.0
    total_asian = 0.0
    steps = len(loader)
    start_time = time.time()

    for step, batch in enumerate(loader, start=1):
        features = batch["features"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        euro_labels = batch["euro_labels"].to(args.device)
        asian_labels = batch["asian_labels"].to(args.device)

        # LR schedule
        lr = get_lr(
            (epoch - 1) * steps + step,
            args.epochs * steps,
            args.learning_rate,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad()

        out = model(
            features,
            attention_mask=attention_mask,
            euro_labels=euro_labels,
            asian_labels=asian_labels,
        )

        loss = out["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_euro += out["euro_loss"].item()
        total_asian += out["asian_loss"].item()

        if step % max(1, steps // 5) == 0 or step == steps:
            Logger(
                f"Epoch {epoch}/{args.epochs} [{step}/{steps}] "
                f"loss={loss.item():.4f} "
                f"(euro={out['euro_loss'].item():.4f} "
                f"asian={out['asian_loss'].item():.4f}) "
                f"lr={lr:.6f}"
            )

    avg_loss = total_loss / steps
    avg_euro = total_euro / steps
    avg_asian = total_asian / steps
    elapsed = time.time() - start_time
    Logger(
        f"Epoch {epoch} complete: "
        f"avg_loss={avg_loss:.4f} "
        f"avg_euro={avg_euro:.4f} "
        f"avg_asian={avg_asian:.4f} "
        f"time={elapsed:.1f}s"
    )
    return avg_loss


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OddsMind Supervised Smoke Train")
    parser.add_argument("--data", type=str, default="data/odds_fixtures/sample_odds_matches.jsonl")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="out_odds")
    parser.add_argument("--seed", type=int, default=42)
    # P0.2 cutoff args
    parser.add_argument("--cutoffs", type=str, default="",
                        help="Comma-separated cutoff minutes, e.g. '90,60,30'")
    parser.add_argument("--cutoff-mode", type=str, default="none",
                        choices=["none", "exhaustive", "random"],
                        help="Cutoff expansion mode (default: none = P0.1 behaviour)")
    parser.add_argument("--min-events", type=int, default=1,
                        help="Minimum events after cutoff filter to keep a sample")
    # P0.3 asian label args
    parser.add_argument("--asian-label-mode", type=str, default="3class",
                        choices=["3class", "5class"],
                        help="Asian handicap label granularity (default: 3class)")
    # P0.4 split args
    parser.add_argument("--train-match-ids", type=str, default="",
                        help="Path to file with train match_ids (one per line)")
    parser.add_argument("--val-match-ids", type=str, default="",
                        help="Path to file with val match_ids (one per line)")
    parser.add_argument("--eval-every-epoch", action="store_true",
                        help="Run evaluation on val set after each epoch")
    args = parser.parse_args()

    setup_seed(args.seed)

    # Load match ID splits
    from dataset.odds_split import load_match_ids_from_file
    train_ids = None
    val_ids = None
    if args.train_match_ids:
        train_ids = load_match_ids_from_file(args.train_match_ids)
        Logger(f"Train match IDs: {len(train_ids)} from {args.train_match_ids}")
    if args.val_match_ids:
        val_ids = load_match_ids_from_file(args.val_match_ids)
        Logger(f"Val match IDs: {len(val_ids)} from {args.val_match_ids}")

    # Parse cutoffs
    cutoffs = None
    if args.cutoffs:
        cutoffs = [float(c.strip()) for c in args.cutoffs.split(",") if c.strip()]

    # Config
    asian_num_classes = 5 if args.asian_label_mode == "5class" else 3
    config = OddsMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        asian_num_classes=asian_num_classes,
    )

    # Data
    Logger(f"Loading data from: {args.data}")
    dataset = OddsDataset(
        args.data,
        max_seq_len=args.max_seq_len,
        cutoffs=cutoffs,
        cutoff_mode=args.cutoff_mode,
        min_events=args.min_events,
        asian_label_mode=args.asian_label_mode,
        allowed_match_ids=train_ids,
        seed=args.seed,
    )
    Logger(f"  Raw matches: {dataset.num_raw_matches}")
    Logger(f"  Asian label mode: {args.asian_label_mode} ({asian_num_classes} classes)")
    Logger(f"  Cutoff mode: {args.cutoff_mode}")
    if cutoffs:
        Logger(f"  Cutoffs: {cutoffs}")
    Logger(f"  Training samples: {len(dataset)}")
    if dataset.cutoff_counts:
        for k in sorted(dataset.cutoff_counts.keys(), key=int):
            Logger(f"    cutoff={k}: {dataset.cutoff_counts[k]} samples")
    collator = OddsCollator()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    # Model
    Logger(f"Building OddsMindModel: hidden={config.hidden_size}, layers={config.num_hidden_layers}")
    model = OddsMindModel(config).to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    Logger(f"  Params: {total_params:,} ({total_params/1e6:.3f}M)")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Train
    for epoch in range(1, args.epochs + 1):
        train_epoch(model, loader, optimizer, epoch, args)

        # P0.4: optional val evaluation
        if args.eval_every_epoch and val_ids is not None:
            from eval.odds_metrics import accuracy_from_logits, logloss_from_logits
            model.eval()
            val_ds = OddsDataset(
                args.data,
                max_seq_len=args.max_seq_len,
                cutoffs=cutoffs,
                cutoff_mode=args.cutoff_mode,
                min_events=args.min_events,
                asian_label_mode=args.asian_label_mode,
                allowed_match_ids=val_ids,
                seed=args.seed,
            )
            val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                    shuffle=False, collate_fn=collator)
            all_eu_logits, all_as_logits = [], []
            all_eu_labels, all_as_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    f = batch["features"].to(args.device)
                    m = batch["attention_mask"].to(args.device)
                    el = batch["euro_labels"].to(args.device)
                    al = batch["asian_labels"].to(args.device)
                    out = model(f, attention_mask=m)
                    all_eu_logits.append(out["euro_logits"].cpu())
                    all_as_logits.append(out["asian_logits"].cpu())
                    all_eu_labels.append(el.cpu())
                    all_as_labels.append(al.cpu())
            eu_logits = torch.cat(all_eu_logits, dim=0)
            as_logits = torch.cat(all_as_logits, dim=0)
            eu_labels = torch.cat(all_eu_labels, dim=0)
            as_labels = torch.cat(all_as_labels, dim=0)
            Logger(f"  Val [{len(val_ds)} samples]: "
                   f"euro_acc={accuracy_from_logits(eu_logits, eu_labels):.4f} "
                   f"euro_loss={logloss_from_logits(eu_logits, eu_labels):.4f} "
                   f"asian_acc={accuracy_from_logits(as_logits, as_labels):.4f} "
                   f"asian_loss={logloss_from_logits(as_logits, as_labels):.4f}")
            model.train()

    # Save checkpoint
    os.makedirs(args.out_dir, exist_ok=True)
    ckp_path = os.path.join(args.out_dir, "oddsmind_smoke.pth")
    torch.save(model.state_dict(), ckp_path)
    Logger(f"Checkpoint saved to: {ckp_path}")

    Logger("OddsMind smoke training complete.")


if __name__ == "__main__":
    main()
