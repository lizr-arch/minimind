"""Formal RawPooledMLP baseline for P3 training verification.

Uses train/val split match ids only. Test split is not read.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.odds_metrics import (
    accuracy_from_probs,
    brier_from_probs,
    class_counts,
    ece_from_probs,
    logloss_from_probs,
    prediction_counts_from_probs,
)


EURO_MAP = {"home": 0, "draw": 1, "away": 2}
FEATURE_NAMES = [
    "euro_h",
    "euro_d",
    "euro_a",
    "imp_h",
    "imp_d",
    "imp_a",
    "margin",
    "time_log",
    "has_euro",
]
LOSS_CHOICES = ("ce", "weighted_ce", "label_smoothing", "focal")
CLASS_WEIGHT_CHOICES = ("none", "inverse_freq", "mild_draw")
LOSS_EPS = 1e-6


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


def build_class_weights(
    labels: torch.Tensor,
    class_weights: str,
    draw_weight: float,
    device: torch.device,
) -> torch.Tensor | None:
    if class_weights == "none":
        return None
    if class_weights == "mild_draw":
        return torch.tensor([1.0, float(draw_weight), 1.0], dtype=torch.float32, device=device)
    if class_weights == "inverse_freq":
        counts = torch.bincount(labels.cpu(), minlength=3).float()
        inv = torch.where(counts > 0, labels.numel() / (3.0 * counts.clamp_min(1.0)), torch.ones_like(counts))
        return (inv / inv.mean().clamp_min(LOSS_EPS)).to(device)
    raise ValueError(f"Unknown class_weights: {class_weights}")


def compute_training_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_name: str,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    if loss_name == "ce":
        return F.cross_entropy(logits, labels)
    if loss_name == "weighted_ce":
        return F.cross_entropy(logits, labels, weight=class_weights)
    if loss_name == "label_smoothing":
        return F.cross_entropy(logits, labels, label_smoothing=float(label_smoothing))
    if loss_name == "focal":
        ce = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** float(focal_gamma)) * ce).mean()
    raise ValueError(f"Unknown loss: {loss_name}")


def load_rows_for_ids(jsonl_path: str, match_ids: set) -> list:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("match_id") in match_ids:
                rows.append(row)
    return rows


def build_raw_pooled_features(row: dict) -> torch.Tensor:
    timeline = row.get("raw_timeline", []) or row.get("odds_timeline", [])
    event_features = [_event_features(e) for e in timeline]
    if not event_features:
        return torch.zeros(len(FEATURE_NAMES), dtype=torch.float32)
    return torch.stack(event_features).mean(dim=0)


def _event_features(event: dict) -> torch.Tensor:
    h = _as_float(event.get("euro_h", 0.0))
    d = _as_float(event.get("euro_d", 0.0))
    a = _as_float(event.get("euro_a", 0.0))
    has_euro = bool(event.get("has_euro", True)) and h > 1.0 and d > 1.0 and a > 1.0

    if has_euro:
        rh, rd, ra = 1.0 / h, 1.0 / d, 1.0 / a
        total = rh + rd + ra
        imp_h, imp_d, imp_a = rh / total, rd / total, ra / total
        margin = total - 1.0
    else:
        h, d, a = 0.0, 0.0, 0.0
        imp_h, imp_d, imp_a = 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
        margin = 0.0

    minutes = max(_as_float(event.get("minutes_before_kickoff", 0.0)), 0.0)
    return torch.tensor(
        [
            h,
            d,
            a,
            imp_h,
            imp_d,
            imp_a,
            margin,
            math.log1p(minutes),
            1.0 if has_euro else 0.0,
        ],
        dtype=torch.float32,
    )


def _as_float(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def build_dataset(rows: list) -> tuple[torch.Tensor, torch.Tensor]:
    X, y = [], []
    for row in rows:
        label = row.get("label", {}).get("euro_result")
        if label not in EURO_MAP:
            continue
        X.append(build_raw_pooled_features(row))
        y.append(EURO_MAP[label])
    if not X:
        raise ValueError("No labelled rows available for RawPooledMLP baseline")
    return torch.stack(X), torch.tensor(y, dtype=torch.long)


class RawPooledMLP(nn.Module):
    def __init__(self, input_dim: int = 9, hidden_dim: int = 128, hidden_dim_2: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim_2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_epoch(
    model,
    X_train,
    y_train,
    optimizer,
    batch_size,
    device,
    loss_name: str = "ce",
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
):
    model.train()
    perm = torch.randperm(len(X_train))
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0

    for start in range(0, len(X_train), batch_size):
        idx = perm[start : start + batch_size]
        xb = X_train[idx].to(device)
        yb = y_train[idx].to(device)

        logits = model(xb)
        loss = compute_training_loss(
            logits,
            yb,
            loss_name=loss_name,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(-1) == yb).sum().item()
        total += len(yb)
        n_batches += 1

    return total_loss / n_batches, correct / total


@torch.no_grad()
def predict_probabilities(model, X, batch_size, device) -> torch.Tensor:
    model.eval()
    probs_all = []
    for start in range(0, len(X), batch_size):
        xb = X[start : start + batch_size].to(device)
        probs = F.softmax(model(xb), dim=-1).clamp(1e-9, 1 - 1e-9)
        probs_all.append(probs.cpu())
    return torch.cat(probs_all)


@torch.no_grad()
def evaluate(model, X, y, batch_size, device):
    probs = predict_probabilities(model, X, batch_size, device)
    return {
        "accuracy": accuracy_from_probs(probs, y),
        "logloss": logloss_from_probs(probs, y),
        "brier": brier_from_probs(probs, y, 3),
        "ece": ece_from_probs(probs, y, 3)["ece"],
        "label_counts": class_counts(y, 3),
        "prediction_counts": prediction_counts_from_probs(probs, 3),
    }


def labelled_match_ids(rows: list) -> list[str]:
    match_ids = []
    for idx, row in enumerate(rows):
        if row.get("label", {}).get("euro_result") in EURO_MAP:
            match_ids.append(str(row.get("match_id", idx)))
    return match_ids


def write_val_predictions_csv(path: str, match_ids: list[str], labels: torch.Tensor, probs: torch.Tensor) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["match_id", "y_true", "p_home", "p_draw", "p_away", "pred_class", "correct"],
        )
        writer.writeheader()
        for idx, prob in enumerate(probs):
            pred_class = int(torch.argmax(prob).item())
            y_true = int(labels[idx].item())
            writer.writerow(
                {
                    "match_id": match_ids[idx] if idx < len(match_ids) else str(idx),
                    "y_true": y_true,
                    "p_home": float(prob[0].item()),
                    "p_draw": float(prob[1].item()),
                    "p_away": float(prob[2].item()),
                    "pred_class": pred_class,
                    "correct": int(pred_class == y_true),
                }
            )


def main():
    parser = argparse.ArgumentParser(description="Save formal RawPooledMLP baseline")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--train-ids", type=str, required=True)
    parser.add_argument("--val-ids", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss", type=str, default="ce", choices=LOSS_CHOICES)
    parser.add_argument("--class-weights", type=str, default="mild_draw", choices=CLASS_WEIGHT_CHOICES)
    parser.add_argument("--draw-weight", type=float, default=1.2)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--save-val-predictions", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    started = time.time()
    set_deterministic_seed(args.seed)
    prepare_output_dir(args.out_dir, allow_overwrite=args.allow_overwrite)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ids = load_split_ids(args.train_ids)
    val_ids = load_split_ids(args.val_ids)
    print(f"Train IDs: {len(train_ids)}, Val IDs: {len(val_ids)}")

    print("Loading train rows...")
    train_rows = load_rows_for_ids(args.data, train_ids)
    print("Loading val rows...")
    val_rows = load_rows_for_ids(args.data, val_ids)
    print(f"Train rows: {len(train_rows)}, Val rows: {len(val_rows)}")

    print("Building raw pooled features...")
    X_train, y_train = build_dataset(train_rows)
    X_val, y_val = build_dataset(val_rows)
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")

    model = RawPooledMLP(input_dim=len(FEATURE_NAMES)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    active_class_weights = (
        build_class_weights(y_train, args.class_weights, args.draw_weight, device)
        if args.loss in {"weighted_ce", "focal"}
        else None
    )
    if active_class_weights is not None:
        print(f"Class weights: {[round(float(v), 6) for v in active_class_weights.detach().cpu()]}")

    best_val_logloss = float("inf")
    best_epoch = 0
    history = []
    best_path = os.path.join(args.out_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model,
            X_train,
            y_train,
            optimizer,
            args.batch_size,
            device,
            loss_name=args.loss,
            class_weights=active_class_weights,
            label_smoothing=args.label_smoothing,
            focal_gamma=args.focal_gamma,
        )
        val_metrics = evaluate(model, X_val, y_val, args.batch_size * 2, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_accuracy": val_metrics["accuracy"],
            "val_logloss": val_metrics["logloss"],
            "val_brier": val_metrics["brier"],
            "val_ece": val_metrics["ece"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)

        is_best = val_metrics["logloss"] < best_val_logloss
        if is_best:
            best_val_logloss = val_metrics["logloss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_val_logloss": best_val_logloss,
                    "feature_names": FEATURE_NAMES,
                    "config": vars(args),
                },
                best_path,
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

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    final_train = evaluate(model, X_train, y_train, args.batch_size * 2, device)
    final_val = evaluate(model, X_val, y_val, args.batch_size * 2, device)
    val_prediction_path = None
    if args.save_val_predictions:
        val_prediction_path = os.path.join(args.out_dir, "val_predictions.csv")
        val_probs = predict_probabilities(model, X_val, args.batch_size * 2, device)
        write_val_predictions_csv(val_prediction_path, labelled_match_ids(val_rows), y_val, val_probs)
        print(f"Val predictions saved to {val_prediction_path}")

    report = {
        "model": "RawPooledMLP",
        "feature_set": "9-dim Euro raw pooled mean",
        "feature_names": FEATURE_NAMES,
        "params": n_params,
        "best_epoch": best_epoch,
        "best_val_logloss": best_val_logloss,
        "config": vars(args),
        "loss_config": {
            "loss": args.loss,
            "class_weights": args.class_weights,
            "draw_weight": args.draw_weight,
            "label_smoothing": args.label_smoothing,
            "focal_gamma": args.focal_gamma,
            "active_class_weights": (
                [float(v) for v in active_class_weights.detach().cpu()]
                if active_class_weights is not None
                else None
            ),
        },
        "data": {
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
        },
        "train_metrics": {
            "accuracy": final_train["accuracy"],
            "logloss": final_train["logloss"],
            "brier": final_train["brier"],
            "ece": final_train["ece"],
        },
        "val_metrics": final_val,
        "history": history,
        "val_predictions": val_prediction_path,
        "train_time": time.time() - started,
        "run_path": args.out_dir,
        "status": "SAVED",
    }

    report_path = os.path.join(args.out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved to {report_path}")
    print(
        f"RawPooledMLP | val_acc={final_val['accuracy']:.4f} "
        f"val_logloss={final_val['logloss']:.4f} "
        f"brier={final_val['brier']:.4f} ece={final_val['ece']:.4f} "
        f"best_epoch={best_epoch}"
    )


if __name__ == "__main__":
    main()
