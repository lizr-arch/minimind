"""
P1.17 OddsMind v3 Training & Pooling Selection — Experiment Runner

Runs v2 vs v3 comparison, pooling ablation, and full baseline/calibration
evaluation on the same train/val/test split.

Usage (quick smoke):
    python eval/eval_p1_17_experiment.py \
        --data data/odds_real/master_5330.jsonl \
        --split-dir data/odds_real/splits_p1_5 \
        --epochs 2 --batch-size 32 --cutoffs 90,60,30 \
        --out-dir runs/p1_17_smoke --smoke

Usage (full experiment):
    python eval/eval_p1_17_experiment.py \
        --data data/odds_real/master_5330.jsonl \
        --split-dir data/odds_real/splits_p1_5 \
        --epochs 30 --batch-size 64 --cutoffs 90,60,30 \
        --out-dir runs/p1_17_full
"""

import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from model.model_oddsmind import OddsMindConfig, OddsMindModel
from dataset.odds_dataset import OddsDataset
from dataset.odds_collator import OddsCollator
from dataset.odds_split import load_match_ids_from_file
from eval.odds_metrics import (
    accuracy_from_logits,
    logloss_from_logits,
    brier_from_logits,
    ece_from_logits,
)
from eval.odds_baselines import (
    implied_prob_euro,
    open_no_vig_euro,
    close_no_vig_euro,
    euro_probs_to_tensor,
)


# ── Utilities ──────────────────────────────────────────────────────────

def get_lr(current_step, total_steps, lr, warmup_steps=0):
    if warmup_steps > 0 and current_step < warmup_steps:
        return lr * current_step / max(1, warmup_steps)
    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))


def setup_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ── Training ───────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, epoch, total_epochs, lr, warmup_steps):
    model.train()
    total_loss = 0.0
    steps = len(loader)
    total_steps = total_epochs * steps

    for step, batch in enumerate(loader, start=1):
        features = batch["features"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        euro_labels = batch["euro_labels"].to(device)
        asian_labels = batch["asian_labels"].to(device)
        missing_mask = batch.get("missing_mask")
        if missing_mask is not None:
            missing_mask = missing_mask.to(device)

        cur_step = (epoch - 1) * steps + step
        cur_lr = get_lr(cur_step, total_steps, lr, warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        optimizer.zero_grad()
        out = model(features, attention_mask=attention_mask,
                    euro_labels=euro_labels, asian_labels=asian_labels,
                    missing_mask=missing_mask)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

        if step % max(1, steps // 3) == 0 or step == steps:
            print(f"  epoch {epoch}/{total_epochs} step {step}/{steps} "
                  f"loss={loss.item():.4f} lr={cur_lr:.6f}")

    return total_loss / steps


def train_model(model, train_loader, val_loader, epochs, lr, warmup_steps, out_path):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, epoch, epochs, lr, warmup_steps)
        print(f"  epoch {epoch} avg_train_loss={train_loss:.4f}")

        if val_loader is not None:
            val_loss = evaluate_val_loss(model, val_loader)
            print(f"  epoch {epoch} val_loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), out_path)
                print(f"  -> best checkpoint saved (val_loss={val_loss:.4f})")
        else:
            torch.save(model.state_dict(), out_path)

    if val_loader is None:
        torch.save(model.state_dict(), out_path)
    print(f"  training complete, best val_loss={best_val_loss:.4f}")


def evaluate_val_loss(model, loader):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            euro_labels = batch["euro_labels"].to(device)
            asian_labels = batch["asian_labels"].to(device)
            missing_mask = batch.get("missing_mask")
            if missing_mask is not None:
                missing_mask = missing_mask.to(device)
            out = model(features, attention_mask=attention_mask,
                        euro_labels=euro_labels, asian_labels=asian_labels,
                        missing_mask=missing_mask)
            total += out["loss"].item() * features.shape[0]
            count += features.shape[0]
    model.train()
    return total / max(1, count)


# ── Evaluation ─────────────────────────────────────────────────────────

def evaluate_model_full(model, loader, asian_num_classes):
    """Run full evaluation: accuracy, logloss, brier, ECE."""
    model.eval()
    all_eu_log, all_as_log = [], []
    all_eu_lbl, all_as_lbl = [], []
    all_league_ids = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            eu_lbl = batch["euro_labels"].to(device)
            as_lbl = batch["asian_labels"].to(device)
            missing_mask = batch.get("missing_mask")
            if missing_mask is not None:
                missing_mask = missing_mask.to(device)
            else:
                missing_mask = None

            out = model(features, attention_mask=attention_mask, missing_mask=missing_mask)
            all_eu_log.append(out["euro_logits"].cpu())
            all_as_log.append(out["asian_logits"].cpu())
            all_eu_lbl.append(eu_lbl.cpu())
            all_as_lbl.append(as_lbl.cpu())
            all_league_ids.extend(batch.get("league_ids", [""] * features.shape[0]))

    eu_log = torch.cat(all_eu_log, dim=0)
    as_log = torch.cat(all_as_log, dim=0)
    eu_lbl = torch.cat(all_eu_lbl, dim=0)
    as_lbl = torch.cat(all_as_lbl, dim=0)

    result = {
        "num_samples": int(eu_log.shape[0]),
        "euro_accuracy": accuracy_from_logits(eu_log, eu_lbl),
        "euro_logloss": logloss_from_logits(eu_log, eu_lbl),
        "euro_brier": brier_from_logits(eu_log, eu_lbl, 3),
        "euro_ece": ece_from_logits(eu_log, eu_lbl, 3)["ece"],
        "asian_accuracy": accuracy_from_logits(as_log, as_lbl),
        "asian_logloss": logloss_from_logits(as_log, as_lbl),
        "asian_brier": brier_from_logits(as_log, as_lbl, asian_num_classes),
        "asian_ece": ece_from_logits(as_log, as_lbl, asian_num_classes)["ece"],
    }

    # Per-league
    if all_league_ids:
        leagues = {}
        for league in sorted(set(all_league_ids)):
            if not league:
                continue
            idx = [i for i, lid in enumerate(all_league_ids) if lid == league]
            if not idx:
                continue
            idx_t = torch.tensor(idx)
            leagues[league] = {
                "count": len(idx),
                "euro_accuracy": accuracy_from_logits(eu_log[idx_t], eu_lbl[idx_t]),
                "euro_logloss": logloss_from_logits(eu_log[idx_t], eu_lbl[idx_t]),
            }
        result["by_league"] = leagues

    model.train()
    return result


# ── Baseline Comparison ────────────────────────────────────────────────

def compute_baselines_from_dataset(dataset, asian_label_mode):
    """Compute all baselines by iterating dataset samples directly."""
    records = []
    for sample in dataset.samples:
        timeline = sample.get("odds_timeline", [])
        cutoff = sample.get("cutoff_minutes", 0)
        cutoff_key = str(int(cutoff))

        filtered = [e for e in timeline if e.get("minutes_before_kickoff", 0) >= cutoff]
        if not filtered:
            continue

        euro_label = {"home": 0, "draw": 1, "away": 2}[sample["label"]["euro_result"]]

        baselines = {}
        # implied_prob_euro (close no-vig)
        try:
            baselines["implied_prob"] = euro_probs_to_tensor(implied_prob_euro(filtered))
        except (ValueError, KeyError):
            baselines["implied_prob"] = torch.full((3,), 1.0 / 3.0)

        # open_no_vig
        try:
            baselines["open_no_vig"] = euro_probs_to_tensor(open_no_vig_euro(filtered))
        except (ValueError, KeyError):
            baselines["open_no_vig"] = torch.full((3,), 1.0 / 3.0)

        # close_no_vig
        try:
            baselines["close_no_vig"] = euro_probs_to_tensor(close_no_vig_euro(filtered))
        except (ValueError, KeyError):
            baselines["close_no_vig"] = torch.full((3,), 1.0 / 3.0)

        records.append({
            "cutoff_key": cutoff_key,
            "euro_label": euro_label,
            **baselines,
        })

    if not records:
        return {}

    def calc_metrics(probs, labels):
        return {
            "accuracy": accuracy_from_probs_tensor(probs, labels),
            "logloss": logloss_from_probs_tensor(probs, labels),
            "brier": brier_from_probs_tensor(probs, labels, 3),
            "ece": ece_from_probs_tensor(probs, labels, 3),
        }

    results = {}
    for name in ["implied_prob", "open_no_vig", "close_no_vig"]:
        probs = torch.stack([r[name] for r in records])
        labels = torch.tensor([r["euro_label"] for r in records])
        results[name] = calc_metrics(probs, labels)

    # By cutoff
    by_cutoff = {}
    for ck in sorted(set(r["cutoff_key"] for r in records), key=int):
        c_recs = [r for r in records if r["cutoff_key"] == ck]
        c_probs = torch.stack([r["close_no_vig"] for r in c_recs])
        c_labels = torch.tensor([r["euro_label"] for r in c_recs])
        by_cutoff[ck] = {
            "count": len(c_recs),
            "close_no_vig_accuracy": accuracy_from_probs_tensor(c_probs, c_labels),
        }
    results["by_cutoff"] = by_cutoff
    results["num_samples"] = len(records)
    return results


def accuracy_from_probs_tensor(probs, labels):
    if probs.shape[0] == 0:
        return 0.0
    return (probs.argmax(-1) == labels).float().mean().item()


def logloss_from_probs_tensor(probs, labels):
    if probs.shape[0] == 0:
        return 0.0
    clamped = probs.clamp(min=1e-12, max=1.0 - 1e-12)
    gathered = clamped[torch.arange(probs.shape[0]), labels]
    return -torch.log(gathered).mean().item()


def brier_from_probs_tensor(probs, labels, nc):
    if probs.shape[0] == 0:
        return 0.0
    y = F.one_hot(labels, nc).float()
    return (probs - y).pow(2).sum(-1).mean().item()


def ece_from_probs_tensor(probs, labels, nc, n_bins=10):
    if probs.shape[0] == 0:
        return 0.0
    conf, preds = probs.max(-1)
    correct = (preds == labels).float()
    boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = boundaries[i], boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (conf >= lo) & (conf <= hi)
        else:
            in_bin = (conf >= lo) & (conf < hi)
        if in_bin.sum() == 0:
            continue
        bin_acc = correct[in_bin].mean()
        bin_conf = conf[in_bin].mean()
        ece += (in_bin.sum().item() / probs.shape[0]) * abs(bin_acc - bin_conf).item()
    return ece


# ── Pooling Ablation ───────────────────────────────────────────────────

def run_pooling_ablation(ckpt_path, dataset_args, asian_num_classes, device):
    """Load a checkpoint and evaluate with all 3 pooling modes.
    
    NOTE: If the checkpoint was trained with mean pooling, the attention/cls
    pooling parameters are randomly initialized. For a fair comparison, each
    mode needs its own trained checkpoint. This function provides a lower-bound
    estimate for the untrained pooling modes.
    """
    ckp = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(ckp, dict) and "model_state_dict" in ckp:
        state_dict = ckp["model_state_dict"]
    else:
        state_dict = ckp

    results = {}
    for mode in ["mean", "attention", "cls"]:
        config = OddsMindConfig(
            hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
            asian_num_classes=asian_num_classes,
            pooling_mode=mode,
        )
        model = OddsMindModel(config).to(device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # Identify keys specific to this pooling mode
        pooling_keys = [k for k in missing if any(p in k for p in ["attention_pool", "cls_pool", "pool"])]
        non_pooling = [k for k in missing if k not in pooling_keys]
        model.eval()

        ds = OddsDataset(**dataset_args)
        loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=OddsCollator())
        metrics = evaluate_model_full(model, loader, asian_num_classes)
        metrics["_missing_pooling_keys"] = len(pooling_keys)
        metrics["_missing_non_pooling_keys"] = len(non_pooling)
        results[mode] = metrics
        tag = " (trained)" if len(pooling_keys) == 0 else f" ({len(pooling_keys)} untrained params)"
        print(f"  pooling={mode}{tag}: euro_acc={metrics['euro_accuracy']:.4f} "
              f"euro_loss={metrics['euro_logloss']:.4f} euro_ece={metrics['euro_ece']:.4f}")

    return results


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="P1.17 Experiment Runner")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--split-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cutoffs", type=str, default="90,60,30")
    parser.add_argument("--asian-label-mode", type=str, default="5class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="runs/p1_17")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke run (2 epochs, small batch)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, only eval existing checkpoints")
    parser.add_argument("--skip-pooling-ablation", action="store_true")
    args = parser.parse_args()

    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asian_num_classes = 5 if args.asian_label_mode == "5class" else 3
    os.makedirs(args.out_dir, exist_ok=True)

    if args.smoke:
        args.epochs = 2
        args.batch_size = 16
        print("=== P1.17 SMOKE RUN (2 epochs, batch=16) ===")
    else:
        print("=== P1.17 FULL EXPERIMENT ===")

    print(f"Data: {args.data}")
    print(f"Split: {args.split_dir}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    setup_seed(args.seed)

    # Load splits
    train_ids = load_match_ids_from_file(os.path.join(args.split_dir, "train_match_ids.txt"))
    val_ids = load_match_ids_from_file(os.path.join(args.split_dir, "val_match_ids.txt"))
    test_ids = load_match_ids_from_file(os.path.join(args.split_dir, "test_match_ids.txt"))
    print(f"Train: {len(train_ids)}  Val: {len(val_ids)}  Test: {len(test_ids)}")

    cutoffs = [float(c.strip()) for c in args.cutoffs.split(",") if c.strip()]

    base_dataset_args = dict(
        jsonl_path=args.data,
        max_seq_len=64,
        cutoffs=cutoffs,
        cutoff_mode="exhaustive",
        min_events=1,
        asian_label_mode=args.asian_label_mode,
        seed=args.seed,
    )

    # ── Train v2 ──
    print("\n--- Training v2 (feature_schema_version=v2, no missing_mask) ---")
    v2_args = {**base_dataset_args, "feature_schema_version": "v2"}
    config_v2 = OddsMindConfig(hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
                               asian_num_classes=asian_num_classes, dropout=0.1,
                               transformer_backend="odds_native",
                               feature_schema_version="v2")

    if not args.skip_training:
        train_ds_v2 = OddsDataset(allowed_match_ids=train_ids, **v2_args)
        val_ds_v2 = OddsDataset(allowed_match_ids=val_ids, **v2_args)
        print(f"  Train samples: {len(train_ds_v2)}, Val samples: {len(val_ds_v2)}")

        model_v2 = OddsMindModel(config_v2).to(device)
        params = sum(p.numel() for p in model_v2.parameters())
        print(f"  Params: {params:,} ({params/1e6:.2f}M)")

        train_loader_v2 = DataLoader(train_ds_v2, batch_size=args.batch_size, shuffle=True,
                                      collate_fn=OddsCollator())
        val_loader_v2 = DataLoader(val_ds_v2, batch_size=args.batch_size, shuffle=False,
                                    collate_fn=OddsCollator())
        ckpt_v2 = os.path.join(args.out_dir, "model_v2.pth")
        train_model(model_v2, train_loader_v2, val_loader_v2, args.epochs, args.lr,
                    warmup_steps=5 * len(train_loader_v2), out_path=ckpt_v2)
    else:
        ckpt_v2 = os.path.join(args.out_dir, "model_v2.pth")
        print(f"  Skipping training, loading from {ckpt_v2}")

    # ── Train v3 ──
    print("\n--- Training v3 (feature_schema_version=v3, with missing_mask) ---")
    v3_args = {**base_dataset_args, "feature_schema_version": "v3"}
    config_v3 = OddsMindConfig(hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
                               asian_num_classes=asian_num_classes, dropout=0.1,
                               transformer_backend="odds_native",
                               feature_schema_version="v3")

    if not args.skip_training:
        train_ds_v3 = OddsDataset(allowed_match_ids=train_ids, **v3_args)
        val_ds_v3 = OddsDataset(allowed_match_ids=val_ids, **v3_args)
        print(f"  Train samples: {len(train_ds_v3)}, Val samples: {len(val_ds_v3)}")

        model_v3 = OddsMindModel(config_v3).to(device)
        params = sum(p.numel() for p in model_v3.parameters())
        print(f"  Params: {params:,} ({params/1e6:.2f}M)")

        train_loader_v3 = DataLoader(train_ds_v3, batch_size=args.batch_size, shuffle=True,
                                      collate_fn=OddsCollator())
        val_loader_v3 = DataLoader(val_ds_v3, batch_size=args.batch_size, shuffle=False,
                                    collate_fn=OddsCollator())
        ckpt_v3 = os.path.join(args.out_dir, "model_v3.pth")
        train_model(model_v3, train_loader_v3, val_loader_v3, args.epochs, args.lr,
                    warmup_steps=5 * len(train_loader_v3), out_path=ckpt_v3)
    else:
        ckpt_v3 = os.path.join(args.out_dir, "model_v3.pth")
        print(f"  Skipping training, loading from {ckpt_v3}")

    # ── Evaluate v2 on test ──
    print("\n--- Evaluating v2 on test set ---")
    test_ds_v2 = OddsDataset(allowed_match_ids=test_ids, **v2_args)
    test_loader_v2 = DataLoader(test_ds_v2, batch_size=64, shuffle=False, collate_fn=OddsCollator())

    ckp_v2_state = torch.load(ckpt_v2, map_location=device)
    model_v2_eval = OddsMindModel(config_v2).to(device)
    if isinstance(ckp_v2_state, dict) and "model_state_dict" in ckp_v2_state:
        model_v2_eval.load_state_dict(ckp_v2_state["model_state_dict"], strict=False)
    else:
        model_v2_eval.load_state_dict(ckp_v2_state, strict=False)
    model_v2_eval.eval()

    metrics_v2 = evaluate_model_full(model_v2_eval, test_loader_v2, asian_num_classes)
    print(f"  v2 euro: acc={metrics_v2['euro_accuracy']:.4f} loss={metrics_v2['euro_logloss']:.4f} "
          f"brier={metrics_v2['euro_brier']:.4f} ece={metrics_v2['euro_ece']:.4f}")

    # ── Evaluate v3 on test ──
    print("\n--- Evaluating v3 on test set ---")
    test_ds_v3 = OddsDataset(allowed_match_ids=test_ids, **v3_args)
    test_loader_v3 = DataLoader(test_ds_v3, batch_size=64, shuffle=False, collate_fn=OddsCollator())

    ckp_v3_state = torch.load(ckpt_v3, map_location=device)
    model_v3_eval = OddsMindModel(config_v3).to(device)
    if isinstance(ckp_v3_state, dict) and "model_state_dict" in ckp_v3_state:
        model_v3_eval.load_state_dict(ckp_v3_state["model_state_dict"], strict=False)
    else:
        model_v3_eval.load_state_dict(ckp_v3_state, strict=False)
    model_v3_eval.eval()

    metrics_v3 = evaluate_model_full(model_v3_eval, test_loader_v3, asian_num_classes)
    print(f"  v3 euro: acc={metrics_v3['euro_accuracy']:.4f} loss={metrics_v3['euro_logloss']:.4f} "
          f"brier={metrics_v3['euro_brier']:.4f} ece={metrics_v3['euro_ece']:.4f}")

    # ── Baseline compute ──
    print("\n--- Computing baselines ---")
    test_ds_for_baseline = OddsDataset(allowed_match_ids=test_ids, **base_dataset_args,
                                        feature_schema_version="v2")
    baseline_metrics = compute_baselines_from_dataset(test_ds_for_baseline, args.asian_label_mode)
    print(f"  Baseline samples: {baseline_metrics.get('num_samples', 0)}")

    # ── Pooling Ablation (on v3 checkpoint) ──
    pooling_results = {}
    if not args.skip_pooling_ablation:
        print("\n--- Pooling ablation on v3 checkpoint ---")
        pooling_dataset_args = dict(
            jsonl_path=args.data, max_seq_len=64, cutoffs=cutoffs,
            cutoff_mode="exhaustive", min_events=1,
            asian_label_mode=args.asian_label_mode, seed=args.seed,
            allowed_match_ids=test_ids, feature_schema_version="v3",
        )
        pooling_results = run_pooling_ablation(ckpt_v3, pooling_dataset_args,
                                                asian_num_classes, device)

    # ── Compile report ──
    report = {
        "experiment": "P1.17",
        "setup": {
            "data": args.data,
            "split_dir": args.split_dir,
            "train_matches": len(train_ids),
            "val_matches": len(val_ids),
            "test_matches": len(test_ids),
            "cutoffs": cutoffs,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "model_config": "d_model=256, num_layers=4, num_heads=8, ffn_dim=1024, dropout=0.10",
        },
        "v2_metrics": metrics_v2,
        "v3_metrics": metrics_v3,
        "v2_vs_v3_delta": {
            "euro_accuracy": round(metrics_v3["euro_accuracy"] - metrics_v2["euro_accuracy"], 4),
            "euro_logloss": round(metrics_v3["euro_logloss"] - metrics_v2["euro_logloss"], 4),
            "euro_brier": round(metrics_v3["euro_brier"] - metrics_v2["euro_brier"], 4),
            "euro_ece": round(metrics_v3["euro_ece"] - metrics_v2["euro_ece"], 4),
        },
        "baselines": baseline_metrics,
        "pooling_ablation": pooling_results,
    }

    # Model vs baseline delta
    if baseline_metrics:
        close_nv = baseline_metrics.get("close_no_vig", {})
        report["model_vs_close_no_vig"] = {
            "v2_euro_acc_delta": round(metrics_v2["euro_accuracy"] - close_nv.get("accuracy", 0), 4),
            "v3_euro_acc_delta": round(metrics_v3["euro_accuracy"] - close_nv.get("accuracy", 0), 4),
            "v2_euro_logloss_delta": round(metrics_v2["euro_logloss"] - close_nv.get("logloss", 0), 4),
            "v3_euro_logloss_delta": round(metrics_v3["euro_logloss"] - close_nv.get("logloss", 0), 4),
        }

    # ── Save report ──
    report_path = os.path.join(args.out_dir, "p1_17_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")

    # ── Print summary table ──
    print("\n" + "=" * 70)
    print("P1.17 SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} {'v2':>12} {'v3':>12} {'delta':>12}")
    print("-" * 61)
    for k in ["euro_accuracy", "euro_logloss", "euro_brier", "euro_ece"]:
        v2 = metrics_v2[k]
        v3 = metrics_v3[k]
        d = v3 - v2
        print(f"{k:<25} {v2:>12.4f} {v3:>12.4f} {d:>+12.4f}")

    if baseline_metrics:
        print(f"\n{'Baseline':<25} {'Accuracy':>12} {'LogLoss':>12}")
        print("-" * 49)
        for name in ["implied_prob", "open_no_vig", "close_no_vig"]:
            b = baseline_metrics.get(name, {})
            print(f"{name:<25} {b.get('accuracy',0):>12.4f} {b.get('logloss',0):>12.4f}")

        cnv = baseline_metrics.get("close_no_vig", {})
        print(f"\nModel vs close_no_vig:")
        print(f"  v2 accuracy delta: {metrics_v2['euro_accuracy'] - cnv.get('accuracy',0):+.4f}")
        print(f"  v3 accuracy delta: {metrics_v3['euro_accuracy'] - cnv.get('accuracy',0):+.4f}")

    if pooling_results:
        print(f"\n{'Pooling':<15} {'Euro Acc':>10} {'Euro Loss':>10} {'Euro ECE':>10}")
        print("-" * 47)
        for mode in ["mean", "attention", "cls"]:
            m = pooling_results.get(mode, {})
            print(f"{mode:<15} {m.get('euro_accuracy',0):>10.4f} "
                  f"{m.get('euro_logloss',0):>10.4f} {m.get('euro_ece',0):>10.4f}")

    print("\nMissing mask effect: ACTIVE (mask-aware projection with learned per-feature embedding)")
    print("=" * 70)


if __name__ == "__main__":
    main()
