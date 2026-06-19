"""
OddsMind Pretrain Training Script (P0.7)

Usage:
    python trainer/train_odds_pretrain.py --task masked_reconstruction ...
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

from model.model_oddsmind_pretrain import OddsPretrainConfig, OddsMindPretrainModel
from dataset.odds_pretrain_dataset import OddsPretrainDataset
from dataset.odds_pretrain_collator import OddsPretrainCollator
from dataset.odds_split import load_match_ids_from_file

warnings.filterwarnings("ignore")


def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def Logger(content):
    print(content)


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def prepare_batch(batch, device):
    """Move batch tensors to device."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def train_epoch(model, loader, optimizer, epoch, total_steps, args):
    model.train()
    total_loss = 0.0
    steps = len(loader)
    for step, batch in enumerate(loader, start=1):
        batch = prepare_batch(batch, args.device)
        lr = get_lr((epoch - 1) * steps + step, total_steps * args.epochs, args.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        out = model(
            features=batch["features"],
            attention_mask=batch.get("attention_mask"),
            target_features=batch.get("target_features"),
            target_mask=batch.get("target_mask"),
            target_event=batch.get("target_event"),
        )
        loss = out["loss"]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if step % max(1, steps // 5) == 0 or step == steps:
            Logger(f"Epoch {epoch}/{args.epochs} [{step}/{steps}] loss={loss.item():.4f} lr={lr:.6f}")
    return total_loss / steps


def main():
    parser = argparse.ArgumentParser(description="OddsMind Pretrain")
    parser.add_argument("--data", type=str, default="data/odds_fixtures/sample_odds_matches_5class.jsonl")
    parser.add_argument("--task", type=str, default="masked_reconstruction")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="out_odds_pretrain")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cutoffs", type=str, default="")
    parser.add_argument("--cutoff-mode", type=str, default="none")
    parser.add_argument("--train-match-ids", type=str, default="")
    parser.add_argument("--val-match-ids", type=str, default="")
    parser.add_argument("--eval-every-epoch", action="store_true")
    parser.add_argument("--transformer-backend", type=str, default="odds_native")
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--min-events", type=int, default=2)
    args = parser.parse_args()

    setup_seed(args.seed)

    cutoffs = None
    if args.cutoffs:
        cutoffs = [float(c.strip()) for c in args.cutoffs.split(",") if c.strip()]

    train_ids = load_match_ids_from_file(args.train_match_ids) if args.train_match_ids else None
    val_ids = load_match_ids_from_file(args.val_match_ids) if args.val_match_ids else None

    config = OddsPretrainConfig(
        task=args.task,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        transformer_backend=args.transformer_backend,
        max_seq_len=args.max_seq_len,
    )

    model = OddsMindPretrainModel(config).to(args.device)
    Logger(f"Model: {sum(p.numel() for p in model.parameters()):,} params, task={args.task}, backend={args.transformer_backend}")

    ds = OddsPretrainDataset(args.data, max_seq_len=args.max_seq_len,
                             cutoffs=cutoffs, cutoff_mode=args.cutoff_mode,
                             allowed_match_ids=train_ids, task=args.task,
                             mask_ratio=args.mask_ratio, seed=args.seed,
                             min_events=args.min_events)
    Logger(f"Train: {ds.num_raw_matches} raw, {len(ds)} samples")

    collator = OddsPretrainCollator()
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(loader)

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, loader, optimizer, epoch, total_steps, args)
        Logger(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")

        if args.eval_every_epoch and val_ids:
            model.eval()
            val_ds = OddsPretrainDataset(args.data, max_seq_len=args.max_seq_len,
                                         cutoffs=cutoffs, cutoff_mode=args.cutoff_mode,
                                         allowed_match_ids=val_ids, task=args.task,
                                         mask_ratio=args.mask_ratio, seed=args.seed,
                                         min_events=args.min_events)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = prepare_batch(batch, args.device)
                    out = model(features=batch["features"], attention_mask=batch.get("attention_mask"),
                                target_features=batch.get("target_features"),
                                target_mask=batch.get("target_mask"),
                                target_event=batch.get("target_event"))
                    val_losses.append(out["loss"].item())
            Logger(f"  Val [{len(val_ds)} samples]: loss={sum(val_losses)/len(val_losses):.4f}")
            model.train()

    os.makedirs(args.out_dir, exist_ok=True)
    ckp_name = f"oddsmind_pretrain_{args.task}.pth"
    torch.save(model.state_dict(), os.path.join(args.out_dir, ckp_name))
    Logger(f"Saved to {args.out_dir}/{ckp_name}")


if __name__ == "__main__":
    main()
