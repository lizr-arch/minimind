"""
Phase 1b Event-Driven Transformer Training

Trains OddsMindModel on v6_event schema (33-dim per-event features)
with variable-length event sequences, lead-lag, and euro-asian alignment.

Dual-objective loss:
    loss = CE(euro) + CE(asian) + beta * KL(euro_probs, market_prior)

Usage:
    python trainer/train_phase1b_event.py \
        --data data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl \
        --train-ids data/odds_real/splits_v6/train_match_ids.txt \
        --val-ids data/odds_real/splits_v6/val_match_ids.txt \
        --epochs 50 --batch-size 64 --lr 0.001 --beta 0.1 \
        --max-seq-len 128 --use-leadlag --use-alignment --use-ou \
        --device cuda --out-dir runs/phase1b_event
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model_oddsmind import OddsMindConfig, OddsMindModel
from dataset.odds_dataset import OddsDataset
from dataset.odds_collator_v6_event import OddsEventCollator, build_pinnacle_index
from eval.odds_baselines_v2 import Phase1Baseline
from eval.odds_metrics import (
    accuracy_from_probs,
    logloss_from_probs,
    brier_from_probs,
    ece_from_probs,
    class_counts,
    prediction_counts_from_probs,
)

EPS = 1e-9
EURO_MAP = {"home": 0, "draw": 1, "away": 2}


# ── Data loading ────────────────────────────────────────────────────

def load_split_ids(path: str) -> set:
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            mid = line.strip()
            if mid:
                ids.add(mid)
    return ids


def load_and_dedup(jsonl_path: str, split_ids: set) -> list:
    """Pinnacle > Bet365 priority dedup, return list of raw rows."""
    priority = {"Pinnacle": 0, "Bet365": 1}
    best = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            mid = row["match_id"]
            if mid not in split_ids:
                continue
            rank = priority.get(row.get("bookmaker_id", ""), 99)
            if mid not in best or rank < best[mid][0]:
                best[mid] = (rank, row)
    return [r for _, r in best.values()]


# ── Training helpers ────────────────────────────────────────────────

def extract_market_prior(sample, baseline):
    """Extract pure-euro prior from the sample's last valid euro event."""
    tl = sample.get("raw_timeline", [])
    for e in reversed(tl):
        if e.get("has_euro") and e.get("euro_h"):
            try:
                q = baseline.compute_euro_prior(
                    float(e["euro_h"]), float(e["euro_d"]), float(e["euro_a"])
                )
                return torch.tensor(q, dtype=torch.float32)
            except (TypeError, ValueError, KeyError):
                continue
    return torch.full((3,), 1.0 / 3.0)


# ── Collate with market prior ───────────────────────────────────────

class OddsEventCollatorWithPrior(OddsEventCollator):
    """Extends OddsEventCollator to include market_prior per sample."""

    def __init__(self, pinnacle_index=None, pad_value=0.0,
                 use_leadlag=True, use_alignment=True):
        super().__init__(pinnacle_index, pad_value)
        self.use_leadlag = use_leadlag
        self.use_alignment = use_alignment
        self.baseline = Phase1Baseline()

    def __call__(self, batch):
        result = super().__call__(batch)

        # Extract market priors
        priors = []
        for item in batch:
            priors.append(extract_market_prior(item, self.baseline))
        result["market_prior"] = torch.stack(priors)

        # Ablation: zero out lead-lag / alignment features if disabled
        if not self.use_leadlag:
            result["features"][:, :, 31] = 0.0
        if not self.use_alignment:
            result["features"][:, :, 32] = 0.0

        return result


# ── Training epoch ──────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, beta, device):
    model.train()
    total_loss = total_ce_euro = total_ce_asian = total_kl = 0.0
    n_batches = 0

    for batch in loader:
        feats = batch["features"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        missing_mask = batch.get("missing_mask")
        if missing_mask is not None:
            missing_mask = missing_mask.to(device)
        euro_labels = batch["euro_labels"].to(device)
        asian_labels = batch["asian_labels"].to(device)
        prior = batch["market_prior"].to(device)

        output = model(
            features=feats,
            attention_mask=attn_mask,
            missing_mask=missing_mask,
        )

        euro_logits = output["euro_logits"]
        asian_logits = output["asian_logits_raw"]  # unmasked for loss

        ce_euro = F.cross_entropy(euro_logits, euro_labels)
        ce_asian = F.cross_entropy(asian_logits, asian_labels)
        euro_probs = F.softmax(euro_logits, dim=-1)

        kl = (
            euro_probs
            * (torch.log(euro_probs.clamp(min=EPS))
               - torch.log(prior.clamp(min=EPS)))
        ).sum(dim=-1).mean()

        loss = ce_euro + ce_asian + beta * kl

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_ce_euro += ce_euro.item()
        total_ce_asian += ce_asian.item()
        total_kl += kl.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "ce_euro": total_ce_euro / n_batches,
        "ce_asian": total_ce_asian / n_batches,
        "kl": total_kl / n_batches,
    }


# ── Evaluation ──────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_euro_probs = []
    all_euro_labels = []
    all_priors = []
    all_event_counts = []

    for batch in loader:
        feats = batch["features"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        missing_mask = batch.get("missing_mask")
        if missing_mask is not None:
            missing_mask = missing_mask.to(device)

        output = model(feats, attn_mask, missing_mask=missing_mask)
        euro_probs = F.softmax(output["euro_logits"], dim=-1)

        all_euro_probs.append(euro_probs.cpu())
        all_euro_labels.append(batch["euro_labels"])
        all_priors.append(batch["market_prior"])
        all_event_counts.append(batch["event_counts"])

    probs = torch.cat(all_euro_probs, dim=0)
    labels = torch.cat(all_euro_labels, dim=0)
    priors = torch.cat(all_priors, dim=0)
    counts = torch.cat(all_event_counts, dim=0)
    C = 3

    # Track 2
    t2 = {
        "accuracy": accuracy_from_probs(probs, labels),
        "logloss": logloss_from_probs(probs, labels),
        "brier": brier_from_probs(probs, labels, C),
        "ece": ece_from_probs(probs, labels, C)["ece"],
        "label_counts": class_counts(labels, C),
        "prediction_counts": prediction_counts_from_probs(probs, C),
    }

    # Track 1: KL(model || market_prior)
    kl_vals = (
        probs * (torch.log(probs.clamp(min=EPS))
                 - torch.log(priors.clamp(min=EPS)))
    ).sum(dim=-1)
    t1 = {
        "mean_kl": float(kl_vals.mean().item()),
        "median_kl": float(kl_vals.median().item()),
    }

    # By event-count bucket
    by_events = {}
    buckets = [
        ("1-5", 1, 5), ("6-15", 6, 15),
        ("16-50", 16, 50), ("51+", 51, 9999),
    ]
    for name, lo, hi in buckets:
        idx = (counts >= lo) & (counts <= hi)
        n = idx.sum().item()
        if n < 10:
            continue
        p_b = probs[idx]
        l_b = labels[idx]
        by_events[name] = {
            "count": n,
            "accuracy": accuracy_from_probs(p_b, l_b),
            "logloss": logloss_from_probs(p_b, l_b),
        }

    return {
        "track1": t1,
        "track2": t2,
        "by_event_count": by_events,
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 1b Event-Driven Training")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--train-ids", type=str, required=True)
    parser.add_argument("--val-ids", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--use-leadlag", action="store_true", default=True)
    parser.add_argument("--use-alignment", action="store_true", default=True)
    parser.add_argument("--use-ou", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, default="runs/phase1b_event")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ──
    train_ids = load_split_ids(args.train_ids)
    val_ids = load_split_ids(args.val_ids)
    print(f"Train IDs: {len(train_ids)}, Val IDs: {len(val_ids)}")

    train_rows = load_and_dedup(args.data, train_ids)
    val_rows = load_and_dedup(args.data, val_ids)
    print(f"Train rows: {len(train_rows)}, Val rows: {len(val_rows)}")

    # ── Dataset & Loader ──
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

    pinnacle_idx = build_pinnacle_index(train_ds.samples + val_ds.samples)
    print(f"Pinnacle index: {len(pinnacle_idx)} matches")

    train_collator = OddsEventCollatorWithPrior(
        pinnacle_index=pinnacle_idx,
        use_leadlag=args.use_leadlag,
        use_alignment=args.use_alignment,
    )
    val_collator = OddsEventCollatorWithPrior(
        pinnacle_index=pinnacle_idx,
        use_leadlag=args.use_leadlag,
        use_alignment=args.use_alignment,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=train_collator, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        collate_fn=val_collator,
    )

    # ── Model ──
    config = OddsMindConfig(
        feature_schema_version="v6_event",
        asian_line_feature_index=3,
        max_seq_len=args.max_seq_len,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        dropout=args.dropout,
        pooling_mode="mean",
        asian_num_classes=5,
    )
    model = OddsMindModel(config).to(device)
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

    # ── Training loop ──
    best_val_logloss = float("inf")
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, args.beta, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        t2 = val_metrics["track2"]
        t1 = val_metrics["track1"]
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val_track2": t2,
            "val_track1": t1,
            "lr": scheduler.get_last_lr()[0],
        })

        is_best = t2["logloss"] < best_val_logloss
        if is_best:
            best_val_logloss = t2["logloss"]
            best_epoch = epoch

        print(
            f"Epoch {epoch:3d} | "
            f"loss={train_metrics['loss']:.4f} "
            f"ce_e={train_metrics['ce_euro']:.4f} "
            f"ce_a={train_metrics['ce_asian']:.4f} "
            f"kl={train_metrics['kl']:.4f} | "
            f"val acc={t2['accuracy']:.4f} logloss={t2['logloss']:.4f} "
            f"kl={t1['mean_kl']:.4f}"
            + (" *" if is_best else "")
        )

    # ── Save best model ──
    best_path = os.path.join(args.out_dir, "best_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "feature_schema_version": "v6_event",
            "asian_line_feature_index": 3,
            "max_seq_len": args.max_seq_len,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
        },
        "best_epoch": best_epoch,
        "best_val_logloss": best_val_logloss,
    }, best_path)
    print(f"\nBest model saved to {best_path} (epoch {best_epoch}, logloss={best_val_logloss:.4f})")

    # ── Final evaluation ──
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_val = evaluate(model, val_loader, device)

    # ── Baseline comparison ──
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
            bprobs.append(torch.full((3,), 1.0/3.0))
        blabels.append(EURO_MAP.get(row["label"]["euro_result"], 0))
    bp_t = torch.stack(bprobs)
    bl_t = torch.tensor(blabels)
    baseline_val = {
        "accuracy": accuracy_from_probs(bp_t, bl_t),
        "logloss": logloss_from_probs(bp_t, bl_t),
        "brier": brier_from_probs(bp_t, bl_t, 3),
        "ece": ece_from_probs(bp_t, bl_t, 3)["ece"],
    }

    # ── Report ──
    report = {
        "config": vars(args),
        "data": {
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "feature_dim": 33,
            "schema": "v6_event",
        },
        "model": {"n_params": n_params},
        "best_epoch": best_epoch,
        "best_val_logloss": best_val_logloss,
        "final_val": final_val,
        "baseline_val": baseline_val,
        "comparison": {
            "delta_accuracy": round(final_val["track2"]["accuracy"] - baseline_val["accuracy"], 6),
            "delta_logloss": round(final_val["track2"]["logloss"] - baseline_val["logloss"], 6),
            "delta_brier": round(final_val["track2"]["brier"] - baseline_val["brier"], 6),
            "delta_ece": round(final_val["track2"]["ece"] - baseline_val["ece"], 6),
        },
        "history": history,
    }
    report_path = os.path.join(args.out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved to {report_path}")

    # ── Summary ──
    print(f"\n=== Final Comparison ===")
    print(f"  Model   | acc={final_val['track2']['accuracy']:.4f}  "
          f"logloss={final_val['track2']['logloss']:.4f}  "
          f"kl={final_val['track1']['mean_kl']:.4f}")
    print(f"  Baseline| acc={baseline_val['accuracy']:.4f}  "
          f"logloss={baseline_val['logloss']:.4f}")
    print(f"  Delta   | acc={report['comparison']['delta_accuracy']:.4f}  "
          f"logloss={report['comparison']['delta_logloss']:.4f}")
    for name, bucket in final_val.get("by_event_count", {}).items():
        print(f"  Events {name:6s} | n={bucket['count']:5d}  "
              f"acc={bucket['accuracy']:.4f}  logloss={bucket['logloss']:.4f}")


if __name__ == "__main__":
    main()
