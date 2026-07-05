"""
Phase 1a Static Model Training

Trains a small MLP on flat Phase 1a features (close/open/delta for
euro + asian only, 18-d raw features) to predict 1X2 match outcomes.

Dual-objective loss:
    loss = CE(pred, label) + beta * KL(pred, market_prior)

where market_prior = Pinnacle close_no_vig_euro (Phase1Baseline).

Data: Pinnacle priority + Bet365 fallback for full coverage.

Usage:
    python trainer/train_phase1a_static.py \
        --data data/odds_real/v6_phase1a.jsonl \
        --train-ids data/odds_real/splits_v6/train_match_ids.txt \
        --val-ids data/odds_real/splits_v6/val_match_ids.txt \
        --epochs 50 --batch-size 256 --lr 0.001 \
        --beta 0.1 --hidden-dim 64 \
        --out-dir runs/phase1a_static
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

# ── Feature keys (order matters) ────────────────────────────────────

FEATURE_KEYS = [
    # Close euro
    "close_euro_h", "close_euro_d", "close_euro_a",
    # Open euro
    "open_euro_h", "open_euro_d", "open_euro_a",
    # Delta euro
    "delta_euro_h", "delta_euro_d", "delta_euro_a",
    # Close asian
    "close_asian_line", "close_asian_upper_water", "close_asian_lower_water",
    # Open asian
    "open_asian_line", "open_asian_upper_water", "open_asian_lower_water",
    # Delta asian
    "delta_asian_line", "delta_asian_upper_water", "delta_asian_lower_water",
]
N_FEATURES = len(FEATURE_KEYS)


# ── Data loading ────────────────────────────────────────────────────

def load_split_ids(path: str) -> set:
    """Load match_ids from a split file (one per line)."""
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            mid = line.strip()
            if mid:
                ids.add(mid)
    return ids


def load_and_dedup(
    jsonl_path: str,
    split_ids: set,
    bookmaker_priority: list = ("Pinnacle", "Bet365"),
) -> list:
    """
    Load flat JSONL, filter by split_ids, deduplicate by match_id
    using bookmaker priority (first available wins).
    
    Returns list of dicts with extracted features + labels.
    """
    priority_rank = {bk: i for i, bk in enumerate(bookmaker_priority)}
    best: dict = {}  # match_id → (rank, row)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            mid = row["match_id"]
            if mid not in split_ids:
                continue
            bk = row.get("bookmaker_id", "")
            rank = priority_rank.get(bk, len(bookmaker_priority))
            if mid not in best or rank < best[mid][0]:
                best[mid] = (rank, row)

    return [row for _, row in best.values()]


def build_league_map(rows: list, min_count: int = 5) -> dict:
    """Build league_id → index mapping (rare leagues → 'other')."""
    from collections import Counter
    league_counts = Counter(r.get("league_id", "unknown") for r in rows)
    mapping = {}
    idx = 0
    for lg, cnt in league_counts.most_common():
        if cnt >= min_count:
            mapping[lg] = idx
            idx += 1
    mapping["__other__"] = idx
    return mapping


def build_bookmaker_map() -> dict:
    """Hardcoded for Phase 1a: Pinnacle + Bet365."""
    return {"Pinnacle": 0, "Bet365": 1}


# ── Dataset ─────────────────────────────────────────────────────────

class Phase1aDataset(Dataset):
    """Flat feature → label dataset for Phase 1a static model."""

    def __init__(self, rows: list, league_map: dict, bk_map: dict):
        self.samples = []
        baseline = Phase1Baseline()

        for r in rows:
            # Features: fill nulls with 0.0
            feats = []
            for k in FEATURE_KEYS:
                v = r.get(k)
                feats.append(float(v) if v is not None else 0.0)

            # Label
            label = EURO_MAP.get(r["label"]["euro_result"], 0)

            # Market prior (Pinnacle or Bet365 close_no_vig_euro)
            try:
                q = baseline.compute_euro_prior(
                    float(r["close_euro_h"]),
                    float(r["close_euro_d"]),
                    float(r["close_euro_a"]),
                )
            except (TypeError, ValueError, KeyError):
                q = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

            # Embedding indices
            lg = r.get("league_id", "unknown")
            league_idx = league_map.get(lg, league_map["__other__"])
            bk_idx = bk_map.get(r.get("bookmaker_id", ""), 0)

            self.samples.append({
                "features": torch.tensor(feats, dtype=torch.float32),
                "label": label,
                "market_prior": torch.tensor(q, dtype=torch.float32),
                "league_idx": league_idx,
                "bk_idx": bk_idx,
                "match_id": r["match_id"],
                "league_id": lg,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Model ───────────────────────────────────────────────────────────

class Phase1aModel(nn.Module):
    """Small MLP for static Phase 1a features."""

    def __init__(
        self,
        n_features: int = N_FEATURES,
        hidden_dim: int = 64,
        num_leagues: int = 2,
        num_bookmakers: int = 2,
        league_emb_dim: int = 4,
        bk_emb_dim: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.league_emb = nn.Embedding(num_leagues, league_emb_dim)
        self.bk_emb = nn.Embedding(num_bookmakers, bk_emb_dim)

        input_dim = n_features + league_emb_dim + bk_emb_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, features, league_idx, bk_idx):
        """
        Args:
            features: [B, N_FEATURES]
            league_idx: [B] long
            bk_idx: [B] long
        Returns:
            logits: [B, 3]
        """
        le = self.league_emb(league_idx)
        be = self.bk_emb(bk_idx)
        x = torch.cat([features, le, be], dim=-1)
        return self.net(x)


# ── Training ────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, beta, device):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    n_batches = 0

    for batch in loader:
        feats = batch["features"].to(device)
        labels = batch["label"].to(device)
        prior = batch["market_prior"].to(device)
        lg_idx = batch["league_idx"].to(device)
        bk_idx = batch["bk_idx"].to(device)

        logits = model(feats, lg_idx, bk_idx)
        probs = F.softmax(logits, dim=-1)

        # Cross-entropy
        ce = F.cross_entropy(logits, labels)

        # KL(probs || prior)
        kl = (probs * (torch.log(probs.clamp(min=EPS))
                        - torch.log(prior.clamp(min=EPS)))).sum(dim=-1).mean()

        loss = ce + beta * kl

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_ce += ce.item()
        total_kl += kl.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "ce": total_ce / n_batches,
        "kl": total_kl / n_batches,
    }


@torch.no_grad()
def evaluate(model, loader, device, baseline: Phase1Baseline = None):
    """Track 1 + Track 2 evaluation."""
    model.eval()

    if baseline is None:
        baseline = Phase1Baseline()

    all_probs = []
    all_labels = []
    all_priors = []

    for batch in loader:
        feats = batch["features"].to(device)
        lg_idx = batch["league_idx"].to(device)
        bk_idx = batch["bk_idx"].to(device)

        logits = model(feats, lg_idx, bk_idx)
        probs = F.softmax(logits, dim=-1)

        all_probs.append(probs.cpu())
        all_labels.append(batch["label"])
        all_priors.append(batch["market_prior"])

    probs = torch.cat(all_probs, dim=0)  # [N, 3]
    labels = torch.cat(all_labels, dim=0)  # [N]
    priors = torch.cat(all_priors, dim=0)  # [N, 3]
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
        "p90_kl": float(kl_vals.kthvalue(int(kl_vals.shape[0] * 0.9)).values.item()),
    }

    return {"track1": t1, "track2": t2}


def collate_fn(batch):
    """Custom collate: stack tensors."""
    features = torch.stack([b["features"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    priors = torch.stack([b["market_prior"] for b in batch])
    lg_idx = torch.tensor([b["league_idx"] for b in batch], dtype=torch.long)
    bk_idx = torch.tensor([b["bk_idx"] for b in batch], dtype=torch.long)
    return {
        "features": features,
        "label": labels,
        "market_prior": priors,
        "league_idx": lg_idx,
        "bk_idx": bk_idx,
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 1a Static Model Training")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to v6_phase1a.jsonl")
    parser.add_argument("--train-ids", type=str, required=True)
    parser.add_argument("--val-ids", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Weight on KL(probs || market_prior)")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, default="runs/phase1a_static")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ──
    train_ids = load_split_ids(args.train_ids)
    val_ids = load_split_ids(args.val_ids)
    print(f"Train match IDs: {len(train_ids)}, Val match IDs: {len(val_ids)}")

    train_rows = load_and_dedup(args.data, train_ids)
    val_rows = load_and_dedup(args.data, val_ids)
    print(f"Train rows (deduped): {len(train_rows)}, Val rows: {len(val_rows)}")

    # ── Build mappings ──
    all_rows = train_rows + val_rows
    league_map = build_league_map(all_rows)
    bk_map = build_bookmaker_map()
    print(f"Leagues: {len(league_map)} (incl __other__)")
    print(f"Bookmakers: {bk_map}")

    # ── Datasets ──
    train_ds = Phase1aDataset(train_rows, league_map, bk_map)
    val_ds = Phase1aDataset(val_rows, league_map, bk_map)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        collate_fn=collate_fn,
    )

    # ── Model ──
    model = Phase1aModel(
        n_features=N_FEATURES,
        hidden_dim=args.hidden_dim,
        num_leagues=len(league_map),
        num_bookmakers=len(bk_map),
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

    # ── Training loop ──
    best_val_logloss = float("inf")
    best_epoch = 0
    history = []

    baseline = Phase1Baseline()

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, args.beta, device)
        val_metrics = evaluate(model, val_loader, device, baseline)
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
            f"train loss={train_metrics['loss']:.4f} "
            f"ce={train_metrics['ce']:.4f} kl={train_metrics['kl']:.4f} | "
            f"val acc={t2['accuracy']:.4f} logloss={t2['logloss']:.4f} "
            f"ece={t2['ece']:.4f} kl={t1['mean_kl']:.4f}"
            + (" *" if is_best else "")
        )

    # ── Save best model ──
    best_path = os.path.join(args.out_dir, "best_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "n_features": N_FEATURES,
            "hidden_dim": args.hidden_dim,
            "num_leagues": len(league_map),
            "num_bookmakers": len(bk_map),
            "dropout": args.dropout,
        },
        "league_map": league_map,
        "bk_map": bk_map,
        "feature_keys": FEATURE_KEYS,
        "best_epoch": best_epoch,
        "best_val_logloss": best_val_logloss,
    }, best_path)
    print(f"\nBest model saved to {best_path} (epoch {best_epoch}, logloss={best_val_logloss:.4f})")

    # ── Final val evaluation ──
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_val = evaluate(model, val_loader, device, baseline)

    # ── Train set evaluation (for diagnosis) ──
    train_eval = evaluate(model, train_loader, device, baseline)

    # ── Baseline comparison ──
    # Use pure euro prior (not joint) for baseline — this is Phase 1a's gold standard.
    baseline_probs_list = []
    baseline_labels = []
    for row in val_rows:
        try:
            q = baseline.compute_euro_prior(
                float(row["close_euro_h"]),
                float(row["close_euro_d"]),
                float(row["close_euro_a"]),
            )
            bp = torch.tensor(q)
        except Exception:
            bp = torch.full((3,), 1.0/3.0)
        baseline_probs_list.append(bp)
        baseline_labels.append(EURO_MAP.get(row["label"]["euro_result"], 0))
    bp_t = torch.stack(baseline_probs_list)
    bl_t = torch.tensor(baseline_labels)
    baseline_val_metrics = {
        "accuracy": accuracy_from_probs(bp_t, bl_t),
        "logloss": logloss_from_probs(bp_t, bl_t),
        "brier": brier_from_probs(bp_t, bl_t, 3),
        "ece": ece_from_probs(bp_t, bl_t, 3)["ece"],
    }

    # ── Output report ──
    report = {
        "config": {
            "data": args.data,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "beta": args.beta,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
        },
        "data": {
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "n_features": N_FEATURES,
            "feature_keys": FEATURE_KEYS,
            "n_leagues": len(league_map),
            "n_bookmakers": len(bk_map),
        },
        "model": {
            "n_params": n_params,
        },
        "best_epoch": best_epoch,
        "best_val_logloss": best_val_logloss,
        "final_val": final_val,
        "final_train": train_eval,
        "baseline_val": baseline_val_metrics,
        "comparison": {
            "delta_accuracy": round(
                final_val["track2"]["accuracy"] - baseline_val_metrics["accuracy"], 6
            ),
            "delta_logloss": round(
                final_val["track2"]["logloss"] - baseline_val_metrics["logloss"], 6
            ),
            "delta_brier": round(
                final_val["track2"]["brier"] - baseline_val_metrics["brier"], 6
            ),
            "delta_ece": round(
                final_val["track2"]["ece"] - baseline_val_metrics["ece"], 6
            ),
            "delta_kl": round(
                final_val["track1"]["mean_kl"], 6
            ),
        },
        "history": history,
    }

    report_path = os.path.join(args.out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved to {report_path}")

    # ── Quick summary ──
    print(f"\n=== Final Comparison ===")
    print(f"  Model   | acc={final_val['track2']['accuracy']:.4f}  "
          f"logloss={final_val['track2']['logloss']:.4f}  "
          f"brier={final_val['track2']['brier']:.4f}  "
          f"ece={final_val['track2']['ece']:.4f}  "
          f"kl={final_val['track1']['mean_kl']:.4f}")
    print(f"  Baseline| acc={baseline_val_metrics['accuracy']:.4f}  "
          f"logloss={baseline_val_metrics['logloss']:.4f}  "
          f"brier={baseline_val_metrics['brier']:.4f}  "
          f"ece={baseline_val_metrics['ece']:.4f}")
    print(f"  Delta   | acc={report['comparison']['delta_accuracy']:.4f}  "
          f"logloss={report['comparison']['delta_logloss']:.4f}  "
          f"brier={report['comparison']['delta_brier']:.4f}  "
          f"ece={report['comparison']['delta_ece']:.4f}")


if __name__ == "__main__":
    main()
