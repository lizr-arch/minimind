"""
P3b: OddsPatch-iTransformer V2 training script.

Trains cross-market Transformer with Euro + Asian + OU features (25 total).
Target: val_logloss ≤ 0.930 (match RawPooledMLP baseline).

Usage:
    python tools/p3b_train_patch_itransformer.py \
        --data data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl \
        --train-ids data/odds_real/splits_v6/train_match_ids.txt \
        --val-ids data/odds_real/splits_v6/val_match_ids.txt \
        --epochs 50 --batch-size 64 --lr 0.001 \
        --device cuda --out-dir runs/p3b_crosstransformer
"""

import argparse
import csv
import math
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.odds_patch_itransformer_v2 import (
    OddsPatchITransformerV2,
    BUCKET_EDGES,
    bucketize_sample_v2,
    extract_cross_market_feature,
    N_FEATURES,
    N_AGG,
    N_BUCKETS,
    FEATURE_DEFS,
)
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
FEATURE_GROUPS = {
    "euro": [i for i, (_, market) in enumerate(FEATURE_DEFS) if market == 0],
    "asian": [i for i, (_, market) in enumerate(FEATURE_DEFS) if market == 1],
    "ou": [i for i, (_, market) in enumerate(FEATURE_DEFS) if market == 2],
}
TIME_QUALITY_FEATURES = [i for i, (_, market) in enumerate(FEATURE_DEFS) if market == 3]
FEATURE_INDEX = {name: i for i, (name, _) in enumerate(FEATURE_DEFS)}
GROUP_HAS_FEATURE = {
    "euro": "has_euro",
    "asian": "has_asian",
    "ou": "has_ou",
}
MARKET_CLEAN_TARGETS = {
    "has_asian": [
        "asian_line",
        "upper_water",
        "lower_water",
        "asian_upper_implied",
        "asian_lower_implied",
        "asian_margin",
    ],
    "has_ou": [
        "over_under_line",
        "over_water",
        "under_water",
        "over_implied",
        "under_implied",
        "ou_margin",
    ],
}
SCALING_CHOICES = ("none", "standard", "robust", "clipped_standard")
SCALER_EXCLUDED_FEATURES = {
    "has_euro",
    "has_asian",
    "has_ou",
    "bucket_valid_count",
    "market_present_count",
}
LOSS_CHOICES = ("ce", "weighted_ce", "label_smoothing", "focal")
CLASS_WEIGHT_CHOICES = ("none", "inverse_freq", "mild_draw")
SCALING_EPS = 1e-6


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


def _parse_feature_groups(feature_groups: str) -> set:
    selected = {part.strip().lower() for part in feature_groups.split(",") if part.strip()}
    allowed = set(FEATURE_GROUPS)
    invalid = selected - allowed
    if invalid:
        raise ValueError(f"Unknown feature group(s): {', '.join(sorted(invalid))}")
    return selected or set(allowed)


def apply_feature_group_mask(x: torch.Tensor, feature_groups: str) -> torch.Tensor:
    selected = _parse_feature_groups(feature_groups)
    zero_indices = []
    for group, indices in FEATURE_GROUPS.items():
        if group not in selected:
            zero_indices.extend(indices)
    if zero_indices:
        x[..., zero_indices, :] = 0
    present_idx = FEATURE_INDEX["market_present_count"]
    selected_has_indices = [FEATURE_INDEX[GROUP_HAS_FEATURE[group]] for group in sorted(selected)]
    x[..., present_idx, :] = x[..., selected_has_indices, :].sum(dim=-2)
    return x


def _zero_when_mask_absent(x: torch.Tensor, has_feature: str, target_features: list[str]) -> None:
    has_idx = FEATURE_INDEX[has_feature]
    absent = x[..., has_idx, :] <= 0
    for name in target_features:
        idx = FEATURE_INDEX[name]
        x[..., idx, :] = torch.where(absent, torch.zeros_like(x[..., idx, :]), x[..., idx, :])


def apply_clean_missing_markets(x: torch.Tensor) -> torch.Tensor:
    _zero_when_mask_absent(
        x,
        "has_asian",
        [
            "asian_line",
            "upper_water",
            "lower_water",
            "asian_upper_implied",
            "asian_lower_implied",
            "asian_margin",
        ],
    )
    _zero_when_mask_absent(
        x,
        "has_ou",
        [
            "over_under_line",
            "over_water",
            "under_water",
            "over_implied",
            "under_implied",
            "ou_margin",
        ],
    )
    return x


def _tensor_float(value: torch.Tensor | float) -> float:
    return float(torch.as_tensor(value).detach().cpu().item())


def _safe_scale(values: torch.Tensor) -> float:
    scale = values.std(unbiased=False)
    scale_value = _tensor_float(scale)
    return scale_value if math.isfinite(scale_value) and scale_value > SCALING_EPS else 1.0


def fit_feature_scaler(x_train: torch.Tensor, scaling: str) -> dict:
    if scaling not in SCALING_CHOICES:
        raise ValueError(f"Unknown scaling: {scaling}")

    excluded = [name for name, _ in FEATURE_DEFS if name in SCALER_EXCLUDED_FEATURES]
    scaler = {
        "scaling": scaling,
        "fit_split": "train",
        "excluded_features": excluded,
        "scaled_features": [],
        "features": {},
    }
    if scaling == "none":
        return scaler

    for idx, (name, _) in enumerate(FEATURE_DEFS):
        if name in SCALER_EXCLUDED_FEATURES:
            continue
        values = x_train[..., idx, :].reshape(-1).float()
        if values.numel() == 0:
            center = 0.0
            scale = 1.0
            stats = {"index": idx, "center": center, "scale": scale}
        elif scaling == "standard":
            center = _tensor_float(values.mean())
            scale = _safe_scale(values)
            stats = {"index": idx, "center": center, "scale": scale}
        elif scaling == "robust":
            center = _tensor_float(torch.quantile(values, 0.5))
            q25 = _tensor_float(torch.quantile(values, 0.25))
            q75 = _tensor_float(torch.quantile(values, 0.75))
            iqr = q75 - q25
            fallback = None
            if not math.isfinite(iqr) or iqr <= SCALING_EPS:
                iqr = _safe_scale(values)
                fallback = "std_or_unit"
            stats = {
                "index": idx,
                "center": center,
                "scale": iqr,
                "q25": q25,
                "q75": q75,
                "fallback": fallback,
            }
        else:
            clip_min = _tensor_float(torch.quantile(values, 0.01))
            clip_max = _tensor_float(torch.quantile(values, 0.99))
            clipped = values.clamp(clip_min, clip_max)
            center = _tensor_float(clipped.mean())
            scale = _safe_scale(clipped)
            stats = {
                "index": idx,
                "center": center,
                "scale": scale,
                "clip_min": clip_min,
                "clip_max": clip_max,
            }
        scaler["scaled_features"].append(name)
        scaler["features"][name] = stats
    return scaler


def apply_feature_scaler(x: torch.Tensor, scaler: dict) -> torch.Tensor:
    if scaler.get("scaling") == "none":
        return x
    for name, stats in scaler.get("features", {}).items():
        idx = int(stats["index"])
        values = x[..., idx, :]
        if scaler.get("scaling") == "clipped_standard":
            values = values.clamp(float(stats["clip_min"]), float(stats["clip_max"]))
        x[..., idx, :] = (values - float(stats["center"])) / float(stats["scale"])
    return x


def save_feature_scaler(scaler: dict, out_dir: str) -> str:
    path = os.path.join(out_dir, "scaler.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scaler, f, indent=2)
    return path


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
        return (inv / inv.mean().clamp_min(SCALING_EPS)).to(device)
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


def _extract_training_feature(event: dict, feature_name: str, clean_missing_markets: bool) -> float:
    if clean_missing_markets:
        for has_feature, targets in MARKET_CLEAN_TARGETS.items():
            if feature_name in targets and extract_cross_market_feature(event, has_feature) <= 0:
                return 0.0
    return extract_cross_market_feature(event, feature_name)


def bucketize_sample_v2_for_training(raw_timeline: list, clean_missing_markets: bool = False) -> torch.Tensor:
    if not clean_missing_markets:
        return bucketize_sample_v2(raw_timeline)

    buckets = defaultdict(list)
    for event in raw_timeline:
        hours = event.get("minutes_before_kickoff", 0) / 60.0
        for idx, (hi, lo, _) in enumerate(BUCKET_EDGES):
            if hours <= hi and hours > lo:
                buckets[idx].append(event)
                break

    result = torch.zeros(N_BUCKETS, N_FEATURES, N_AGG)
    feature_names = [fd[0] for fd in FEATURE_DEFS]

    for bucket_idx in range(N_BUCKETS):
        evts = sorted(
            buckets.get(bucket_idx, []),
            key=lambda event: event.get("minutes_before_kickoff", 0),
            reverse=True,
        )
        if not evts:
            continue

        for fi, feat_name in enumerate(feature_names):
            if feat_name == "bucket_valid_count":
                result[bucket_idx, fi, 0] = float(len(evts))
                result[bucket_idx, fi, 1] = float(len(evts))
                continue
            if feat_name == "market_present_count":
                vals = [
                    _extract_training_feature(evt, "has_euro", clean_missing_markets)
                    + _extract_training_feature(evt, "has_asian", clean_missing_markets)
                    + _extract_training_feature(evt, "has_ou", clean_missing_markets)
                    for evt in evts
                ]
            else:
                vals = [
                    _extract_training_feature(evt, feat_name, clean_missing_markets)
                    for evt in evts
                ]

            vals = [value for value in vals if math.isfinite(value)]
            if not vals:
                continue
            vt = torch.tensor(vals, dtype=torch.float32)
            result[bucket_idx, fi, 0] = vt[-1]
            result[bucket_idx, fi, 1] = vt.mean()

    return result


def build_bucketed_dataset_v2(
    samples: list,
    dataset: OddsDataset,
    feature_groups: str = "euro,asian,ou",
    clean_missing_markets: bool = False,
):
    X_list, y_list = [], []
    for i in range(len(dataset)):
        raw_tl = dataset.samples[i].get("raw_timeline", [])
        if not raw_tl:
            raw_tl = dataset.samples[i].get("odds_timeline", [])
        xb = bucketize_sample_v2_for_training(raw_tl, clean_missing_markets=clean_missing_markets)
        xb = apply_feature_group_mask(xb, feature_groups)
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(-1) == yb).sum().item()
        total += len(yb)
        n_batches += 1

    return total_loss / n_batches, correct / total


@torch.no_grad()
def predict_probabilities(model, X, batch_size, device) -> torch.Tensor:
    model.eval()
    all_probs = []

    for start in range(0, len(X), batch_size):
        xb = X[start : start + batch_size].to(device)
        logits = model(xb)
        probs = F.softmax(logits, dim=-1).clamp(1e-9, 1 - 1e-9)
        all_probs.append(probs.cpu())

    return torch.cat(all_probs, dim=0)


@torch.no_grad()
def evaluate(model, X_val, y_val, batch_size, device):
    probs = predict_probabilities(model, X_val, batch_size, device)
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


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="P3b OddsPatch-iTransformer V2 Training")
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
    parser.add_argument("--out-dir", type=str, default="runs/p3b_crosstransformer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples (0=all)")
    parser.add_argument("--feature-groups", type=str, default="euro,asian,ou")
    parser.add_argument("--clean-missing-markets", action="store_true")
    parser.add_argument("--scaling", type=str, default="none", choices=SCALING_CHOICES)
    parser.add_argument("--loss", type=str, default="ce", choices=LOSS_CHOICES)
    parser.add_argument("--class-weights", type=str, default="mild_draw", choices=CLASS_WEIGHT_CHOICES)
    parser.add_argument("--draw-weight", type=float, default=1.2)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--save-val-predictions", action="store_true")
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
        val_ds.samples = val_ds.samples[: min(args.max_samples, len(val_ds.samples))]
        print(f"Limited to {len(train_ds)} train / {len(val_ds)} val samples")

    # ── Pre-compute bucketed features ──
    print("Building V2 bucketed train features...")
    X_train, y_train = build_bucketed_dataset_v2(
        train_ds.samples,
        train_ds,
        feature_groups=args.feature_groups,
        clean_missing_markets=args.clean_missing_markets,
    )
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")

    print("Building V2 bucketed val features...")
    X_val, y_val = build_bucketed_dataset_v2(
        val_ds.samples,
        val_ds,
        feature_groups=args.feature_groups,
        clean_missing_markets=args.clean_missing_markets,
    )
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")

    scaler = fit_feature_scaler(X_train, args.scaling)
    scaler_path = save_feature_scaler(scaler, args.out_dir)
    X_train = apply_feature_scaler(X_train, scaler)
    X_val = apply_feature_scaler(X_val, scaler)
    print(f"Scaling: {args.scaling} (saved to {scaler_path})")

    # ── Model ──
    model = OddsPatchITransformerV2(
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
    active_class_weights = (
        build_class_weights(y_train, args.class_weights, args.draw_weight, device)
        if args.loss in {"weighted_ce", "focal"}
        else None
    )
    if active_class_weights is not None:
        print(f"Class weights: {[round(float(v), 6) for v in active_class_weights.detach().cpu()]}")

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
                        "feature_groups": args.feature_groups,
                        "clean_missing_markets": args.clean_missing_markets,
                        "scaling": args.scaling,
                        "loss": args.loss,
                        "class_weights": args.class_weights,
                        "draw_weight": args.draw_weight,
                        "label_smoothing": args.label_smoothing,
                        "focal_gamma": args.focal_gamma,
                        "max_seq_len": args.max_seq_len,
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
    val_prediction_path = None
    if args.save_val_predictions:
        val_prediction_path = os.path.join(args.out_dir, "val_predictions.csv")
        val_match_ids = [str(sample.get("match_id", idx)) for idx, sample in enumerate(val_ds.samples)]
        val_probs = predict_probabilities(model, X_val, args.batch_size * 2, device)
        write_val_predictions_csv(val_prediction_path, val_match_ids, y_val, val_probs)
        print(f"Val predictions saved to {val_prediction_path}")

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

    # ── Verdict ──
    warnings = []
    if best_val_logloss > 0.930:
        warnings.append(f"val_logloss {best_val_logloss:.4f} > 0.930 baseline")
    if best_val_logloss > 0.930:
        verdict = "FAIL"
    elif warnings:
        verdict = "PASS_WITH_WARNINGS"
    else:
        verdict = "PASS"

    # ── Report ──
    report = {
        "phase": "P3b OddsPatch-iTransformer",
        "model": "OddsPatchITransformerV2",
        "params": n_params,
        "feature_count": N_FEATURES,
        "agg_count": N_AGG,
        "config": vars(args),
        "scaler": {
            "path": scaler_path,
            "scaling": args.scaling,
            "scaled_features": scaler.get("scaled_features", []),
            "excluded_features": scaler.get("excluded_features", []),
        },
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
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
        },
        "best_epoch": best_epoch,
        "best_val_logloss": best_val_logloss,
        "train_metrics": {
            "final_train_loss": history[-1]["train_loss"],
            "final_train_acc": history[-1]["train_acc"],
        },
        "val_metrics": final_val,
        "majority_baseline": {
            "class": majority_class,
            "val_acc": majority_acc,
        },
        "baseline": {
            "raw_pooled_mlp_logloss": 0.930,
            "euro_prior_logloss": baseline_val["logloss"],
            "euro_prior_accuracy": baseline_val["accuracy"],
        },
        "verdict": verdict,
        "warnings": warnings,
        "history": history,
        "val_predictions": val_prediction_path,
        "run_path": args.out_dir,
    }
    report_path = os.path.join(args.out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved to {report_path}")

    # ── Summary ──
    print(f"\n=== P3b Final Comparison ===")
    print(
        f"  Model   | acc={final_val['accuracy']:.4f}  "
        f"logloss={final_val['logloss']:.4f}  "
        f"brier={final_val['brier']:.4f}  "
        f"ece={final_val['ece']:.4f}"
    )
    print(
        f"  Baseline| acc={baseline_val['accuracy']:.4f}  "
        f"logloss={baseline_val['logloss']:.4f}"
    )
    print(f"  Majority| acc={majority_acc:.4f}")
    print(f"  Target  | logloss ≤ 0.930")
    print(
        f"  Result  | logloss = {best_val_logloss:.4f}"
        + _pass_fail_suffix(best_val_logloss <= 0.930)
    )
    print(f"  Verdict | {verdict}")


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
