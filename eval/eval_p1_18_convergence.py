"""
P1.18 OddsMind Full Convergence & Baseline Challenge

Trains v3 mean-pooling model to convergence with early stopping,
then runs comprehensive evaluation against no-vig baselines.

Usage:
    python eval/eval_p1_18_convergence.py \
        --data data/odds_real/master_5330.jsonl \
        --split-dir data/odds_real/splits_p1_5 \
        --epochs 50 --patience 10 --batch-size 64 \
        --out-dir runs/p1_18_convergence
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
    accuracy_from_logits, accuracy_from_probs,
    logloss_from_logits, logloss_from_probs,
    brier_from_logits, brier_from_probs,
    ece_from_logits, ece_from_probs,
)
from eval.odds_baselines import (
    implied_prob_euro,
    open_no_vig_euro,
    close_no_vig_euro,
    euro_probs_to_tensor,
)

EURO_MAP = {"home": 0, "draw": 1, "away": 2}


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_lr(step, total, lr, warmup=0):
    if warmup > 0 and step < warmup:
        return lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * p)))


# ── Training with early stopping ───────────────────────────────────────

def train_full(model, train_loader, val_loader, epochs, lr, warmup_steps,
               patience, device, ckpt_path, history_path):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "epoch": []}

    total_steps = epochs * len(train_loader)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        steps = len(train_loader)

        for step, batch in enumerate(train_loader, start=1):
            features = batch["features"].to(device)
            am = batch["attention_mask"].to(device)
            el = batch["euro_labels"].to(device)
            al = batch["asian_labels"].to(device)
            mm = batch.get("missing_mask")
            if mm is not None:
                mm = mm.to(device)

            cur = (epoch - 1) * steps + step
            cur_lr = get_lr(cur, total_steps, lr, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            optimizer.zero_grad()
            out = model(features, attention_mask=am, euro_labels=el,
                        asian_labels=al, missing_mask=mm)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item()

        avg_train = train_loss_sum / steps
        history["train_loss"].append(avg_train)
        history["epoch"].append(epoch)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                am = batch["attention_mask"].to(device)
                el = batch["euro_labels"].to(device)
                al = batch["asian_labels"].to(device)
                mm = batch.get("missing_mask")
                if mm is not None:
                    mm = mm.to(device)
                out = model(features, attention_mask=am, euro_labels=el,
                            asian_labels=al, missing_mask=mm)
                n = features.shape[0]
                val_loss_sum += out["loss"].item() * n
                val_count += n
        avg_val = val_loss_sum / max(1, val_count)
        history["val_loss"].append(avg_val)

        print(f"epoch {epoch:3d}/{epochs}  train_loss={avg_train:.4f}  "
              f"val_loss={avg_val:.4f}  lr={optimizer.param_groups[0]['lr']:.6f}"
              f"  patience={patience_counter}/{patience}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> best checkpoint (val_loss={avg_val:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -> early stopping at epoch {epoch}")
                break

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss
    history["early_stop_epoch"] = epoch if patience_counter >= patience else epochs

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete: best_epoch={best_epoch}, "
          f"best_val_loss={best_val_loss:.4f}, stopped_at={epoch}")
    return history


# ── Full evaluation ────────────────────────────────────────────────────

def evaluate_full(model, loader, device, asian_nc):
    model.eval()
    eu_log, as_log = [], []
    eu_lbl, as_lbl = [], []
    leagues, match_ids = [], []

    with torch.no_grad():
        for batch in loader:
            f = batch["features"].to(device)
            am = batch["attention_mask"].to(device)
            el = batch["euro_labels"].to(device)
            al = batch["asian_labels"].to(device)
            mm = batch.get("missing_mask")
            if mm is not None:
                mm = mm.to(device)
            out = model(f, attention_mask=am, missing_mask=mm)
            eu_log.append(out["euro_logits"].cpu())
            as_log.append(out["asian_logits"].cpu())
            eu_lbl.append(el.cpu())
            as_lbl.append(al.cpu())
            leagues.extend(batch.get("league_ids", [""] * f.shape[0]))
            match_ids.extend(batch.get("match_ids", [""] * f.shape[0]))

    eu_log = torch.cat(eu_log, 0)
    as_log = torch.cat(as_log, 0)
    eu_lbl = torch.cat(eu_lbl, 0)
    as_lbl = torch.cat(as_lbl, 0)

    ece_eu = ece_from_logits(eu_log, eu_lbl, 3)
    ece_as = ece_from_logits(as_log, as_lbl, asian_nc)

    return {
        "num_samples": int(eu_log.shape[0]),
        "euro": {
            "accuracy": accuracy_from_logits(eu_log, eu_lbl),
            "logloss": logloss_from_logits(eu_log, eu_lbl),
            "brier": brier_from_logits(eu_log, eu_lbl, 3),
            "ece": ece_eu["ece"],
            "reliability_bins": ece_eu["bins"],
        },
        "asian": {
            "accuracy": accuracy_from_logits(as_log, as_lbl),
            "logloss": logloss_from_logits(as_log, as_lbl),
            "brier": brier_from_logits(as_log, as_lbl, asian_nc),
            "ece": ece_as["ece"],
            "reliability_bins": ece_as["bins"],
        },
        "euro_logits": eu_log,
        "euro_labels": eu_lbl,
        "leagues": leagues,
        "match_ids": match_ids,
    }


def evaluate_per_league(eu_log, eu_lbl, leagues):
    result = {}
    uniq = sorted(set(leagues))
    for league in uniq:
        if not league:
            continue
        idx = torch.tensor([i for i, l in enumerate(leagues) if l == league])
        if idx.numel() == 0:
            continue
        el = eu_log[idx]
        ll = eu_lbl[idx]
        result[league] = {
            "count": int(idx.numel()),
            "accuracy": accuracy_from_logits(el, ll),
            "logloss": logloss_from_logits(el, ll),
            "brier": brier_from_logits(el, ll, 3),
            "ece": ece_from_logits(el, ll, 3)["ece"],
        }
    return result


# ── Baseline computation ───────────────────────────────────────────────

def compute_all_baselines(dataset):
    """Compute all euro baselines and per-cutoff/per-league breakdown."""
    samples = dataset.samples
    records = []
    for s in samples:
        tl = s.get("odds_timeline", [])
        cutoff = s.get("cutoff_minutes", 0)
        ck = str(int(cutoff))
        filtered = [e for e in tl if e.get("minutes_before_kickoff", 0) >= cutoff]
        if not filtered:
            continue
        league = s.get("league_id", "")
        label = EURO_MAP[s["label"]["euro_result"]]

        baselines = {}
        for name, fn in [("close_no_vig", close_no_vig_euro),
                          ("open_no_vig", open_no_vig_euro),
                          ("implied_prob", implied_prob_euro)]:
            try:
                baselines[name] = euro_probs_to_tensor(fn(filtered))
            except (ValueError, KeyError):
                baselines[name] = torch.full((3,), 1.0 / 3.0)

        # latest_available_no_vig = close_no_vig (same thing with cutoff filtering)
        baselines["latest_available_no_vig"] = baselines["close_no_vig"].clone()

        records.append({
            "cutoff_key": ck,
            "league": league,
            "label": label,
            **baselines,
        })

    if not records:
        return {}

    labels = torch.tensor([r["label"] for r in records])

    def bm(probs, lbls):
        if probs.shape[0] == 0:
            return {}
        ece = ece_from_probs(probs, lbls, 3)
        return {
            "accuracy": accuracy_from_probs(probs, lbls),
            "logloss": logloss_from_probs(probs, lbls),
            "brier": brier_from_probs(probs, lbls, 3),
            "ece": ece["ece"],
            "reliability_bins": ece["bins"],
        }

    result = {"num_samples": len(records)}
    for name in ["latest_available_no_vig", "open_no_vig", "close_no_vig", "implied_prob"]:
        probs = torch.stack([r[name] for r in records])
        result[name] = bm(probs, labels)

    # Per-cutoff
    by_cutoff = {}
    for ck in sorted(set(r["cutoff_key"] for r in records), key=int):
        c_recs = [r for r in records if r["cutoff_key"] == ck]
        c_labels = torch.tensor([r["label"] for r in c_recs])
        entry = {"count": len(c_recs)}
        for name in ["latest_available_no_vig", "close_no_vig"]:
            c_probs = torch.stack([r[name] for r in c_recs])
            entry[name + "_accuracy"] = accuracy_from_probs(c_probs, c_labels)
            entry[name + "_logloss"] = logloss_from_probs(c_probs, c_labels)
        by_cutoff[ck] = entry
    result["by_cutoff"] = by_cutoff

    # Per-league
    by_league = {}
    for league in sorted(set(r["league"] for r in records)):
        if not league:
            continue
        l_recs = [r for r in records if r["league"] == league]
        l_labels = torch.tensor([r["label"] for r in l_recs])
        entry = {"count": len(l_recs)}
        for name in ["latest_available_no_vig", "close_no_vig"]:
            l_probs = torch.stack([r[name] for r in l_recs])
            entry[name + "_accuracy"] = accuracy_from_probs(l_probs, l_labels)
            entry[name + "_logloss"] = logloss_from_probs(l_probs, l_labels)
        by_league[league] = entry
    result["by_league"] = by_league

    return result


# ── Per-cutoff model eval ──────────────────────────────────────────────

def evaluate_per_cutoff(model, dataset, collator, device, asian_nc, cutoffs):
    """Evaluate by cutoff using the exhaustive dataset, grouping by cutoff_minutes."""
    result = {}
    for cutoff in cutoffs:
        ck = str(int(cutoff))
        # Build sub-dataset for this cutoff using "none" mode
        ds = OddsDataset(
            jsonl_path=dataset._jsonl_path,
            max_seq_len=dataset.max_seq_len,
            cutoff_minutes=cutoff,
            cutoff_mode="none",
            min_events=1,
            asian_label_mode=dataset.asian_label_mode,
            allowed_match_ids=dataset._allowed_ids,
            feature_schema_version=dataset.feature_schema_version,
            seed=42,
        )
        if len(ds) == 0:
            result[ck] = {"num_samples": 0}
            continue
        loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collator)
        try:
            m = evaluate_full(model, loader, device, asian_nc)
            result[ck] = {
                "num_samples": m["num_samples"],
                "euro_accuracy": m["euro"]["accuracy"],
                "euro_logloss": m["euro"]["logloss"],
                "euro_brier": m["euro"]["brier"],
                "euro_ece": m["euro"]["ece"],
            }
        except Exception as e:
            result[ck] = {"num_samples": len(ds), "error": str(e)[:120]}
    return result


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="P1.18 Full Convergence Experiment")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--split-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cutoffs", type=str, default="90,60,30")
    parser.add_argument("--asian-label-mode", type=str, default="5class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="runs/p1_18")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asian_nc = 5 if args.asian_label_mode == "5class" else 3
    os.makedirs(args.out_dir, exist_ok=True)
    setup_seed(args.seed)

    cutoffs = [float(c.strip()) for c in args.cutoffs.split(",") if c.strip()]

    # Load splits
    train_ids = load_match_ids_from_file(os.path.join(args.split_dir, "train_match_ids.txt"))
    val_ids = load_match_ids_from_file(os.path.join(args.split_dir, "val_match_ids.txt"))
    test_ids = load_match_ids_from_file(os.path.join(args.split_dir, "test_match_ids.txt"))

    print("=" * 60)
    print("P1.18 OddsMind Full Convergence & Baseline Challenge")
    print(f"Data: {args.data}  Split: {args.split_dir}")
    print(f"Train: {len(train_ids)}  Val: {len(val_ids)}  Test: {len(test_ids)}")
    print(f"Cutoffs: {cutoffs}  Epochs: {args.epochs}  Patience: {args.patience}")
    print(f"Device: {device}  Seed: {args.seed}")
    print("=" * 60)

    base_ds_args = dict(
        jsonl_path=args.data, max_seq_len=64, cutoffs=cutoffs,
        cutoff_mode="exhaustive", min_events=1,
        asian_label_mode=args.asian_label_mode, seed=args.seed,
        feature_schema_version="v3",
    )

    config = OddsMindConfig(
        hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
        asian_num_classes=asian_nc, dropout=0.1,
        transformer_backend="odds_native",
        pooling_mode="mean",
        feature_schema_version="v3",
    )

    ckpt_path = os.path.join(args.out_dir, "model_best.pth")
    history_path = os.path.join(args.out_dir, "training_history.json")

    if not args.skip_training:
        train_ds = OddsDataset(allowed_match_ids=train_ids, **base_ds_args)
        val_ds = OddsDataset(allowed_match_ids=val_ids, **base_ds_args)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                   collate_fn=OddsCollator())
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=OddsCollator())
        print(f"Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")

        model = OddsMindModel(config).to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"Params: {params:,} ({params/1e6:.2f}M)")

        warmup = 5 * len(train_loader)
        history = train_full(model, train_loader, val_loader, args.epochs, args.lr,
                             warmup, args.patience, device, ckpt_path, history_path)
    else:
        print(f"Skipping training, loading {ckpt_path}")
        with open(history_path) as f:
            history = json.load(f)

    # ── Load best model ──
    model_eval = OddsMindModel(config).to(device)
    ckp = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(ckp, dict) and "model_state_dict" in ckp:
        model_eval.load_state_dict(ckp["model_state_dict"], strict=False)
    else:
        model_eval.load_state_dict(ckp, strict=False)
    model_eval.eval()

    # ── Full test eval ──
    print("\n--- Full test evaluation ---")
    test_ds = OddsDataset(allowed_match_ids=test_ids, **base_ds_args)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=OddsCollator())
    model_metrics = evaluate_full(model_eval, test_loader, device, asian_nc)
    euro_m = model_metrics["euro"]
    print(f"Model euro: acc={euro_m['accuracy']:.4f} loss={euro_m['logloss']:.4f} "
          f"brier={euro_m['brier']:.4f} ece={euro_m['ece']:.4f}")

    # ── Baselines ──
    print("\n--- Baseline computation ---")
    bl_ds = OddsDataset(allowed_match_ids=test_ids, **{**base_ds_args,
                        "feature_schema_version": "v2"})
    bl_metrics = compute_all_baselines(bl_ds)
    print(f"Baselines on {bl_metrics.get('num_samples', 0)} samples")

    # ── Deltas ──
    deltas = {}
    for name in ["latest_available_no_vig", "open_no_vig", "close_no_vig", "implied_prob"]:
        if name not in bl_metrics:
            continue
        b = bl_metrics[name]
        deltas[name] = {
            "accuracy_delta": round(euro_m["accuracy"] - b["accuracy"], 4),
            "logloss_delta": round(euro_m["logloss"] - b["logloss"], 4),
            "brier_delta": round(euro_m["brier"] - b["brier"], 4),
            "ece_delta": round(euro_m["ece"] - b["ece"], 4),
        }

    # ── Per-cutoff model eval ──
    print("\n--- Per-cutoff evaluation ---")
    per_cutoff_model = evaluate_per_cutoff(model_eval, test_ds, OddsCollator(), device, asian_nc, cutoffs)

    # ── Per-league ──
    per_league_model = evaluate_per_league(
        model_metrics["euro_logits"], model_metrics["euro_labels"],
        model_metrics["leagues"])

    # ── Per-cutoff baseline deltas ──
    per_cutoff_deltas = {}
    bl_by_cutoff = bl_metrics.get("by_cutoff", {})
    for ck in per_cutoff_model:
        if ck not in bl_by_cutoff:
            continue
        per_cutoff_deltas[ck] = {
            "model_accuracy": per_cutoff_model[ck].get("euro_accuracy", 0),
            "baseline_accuracy": bl_by_cutoff[ck].get("latest_available_no_vig_accuracy", 0),
            "delta": round(
                per_cutoff_model[ck].get("euro_accuracy", 0) -
                bl_by_cutoff[ck].get("latest_available_no_vig_accuracy", 0), 4),
            "model_logloss": per_cutoff_model[ck].get("euro_logloss", 0),
            "baseline_logloss": bl_by_cutoff[ck].get("latest_available_no_vig_logloss", 0),
            "logloss_delta": round(
                per_cutoff_model[ck].get("euro_logloss", 0) -
                bl_by_cutoff[ck].get("latest_available_no_vig_logloss", 0), 4),
        }

    # ── Per-league baseline deltas ──
    per_league_deltas = {}
    bl_by_league = bl_metrics.get("by_league", {})
    for league in per_league_model:
        if league not in bl_by_league:
            continue
        per_league_deltas[league] = {
            "count": per_league_model[league]["count"],
            "model_accuracy": per_league_model[league]["accuracy"],
            "baseline_accuracy": bl_by_league[league].get("latest_available_no_vig_accuracy", 0),
            "acc_delta": round(
                per_league_model[league]["accuracy"] -
                bl_by_league[league].get("latest_available_no_vig_accuracy", 0), 4),
        }

    # ── Overfitting check ──
    overfitting_warning = False
    if "train_loss" in history and "val_loss" in history:
        n = len(history["train_loss"])
        if n >= 5:
            recent_train = history["train_loss"][-5:]
            recent_val = history["val_loss"][-5:]
            if min(recent_train) < min(recent_val) * 0.8:
                overfitting_warning = True

    # ── Compile report ──
    report = {
        "experiment": "P1.18",
        "overall": "PENDING",
        "setup": {
            "data": args.data,
            "split_dir": args.split_dir,
            "train_matches": len(train_ids),
            "val_matches": len(val_ids),
            "test_matches": len(test_ids),
            "cutoffs": cutoffs,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "model": "d_model=256, layers=4, heads=8, ffn=1024, dropout=0.10, pooling=mean",
            "feature_schema": "v3 (missing_mask active)",
        },
        "training": {
            "best_epoch": history.get("best_epoch", 0),
            "best_val_loss": history.get("best_val_loss", 0),
            "early_stop_epoch": history.get("early_stop_epoch", 0),
            "train_loss_curve": history.get("train_loss", []),
            "val_loss_curve": history.get("val_loss", []),
            "overfitting_warning": overfitting_warning,
        },
        "model_metrics": {k: v for k, v in model_metrics.items()
                          if k not in ("euro_logits", "euro_labels", "leagues", "match_ids")},
        "baselines": {k: v for k, v in bl_metrics.items()
                      if k not in ("by_cutoff", "by_league")},
        "deltas_vs_baselines": deltas,
        "per_cutoff": {
            "model": per_cutoff_model,
            "baseline": bl_metrics.get("by_cutoff", {}),
            "deltas": per_cutoff_deltas,
        },
        "per_league": {
            "model": per_league_model,
            "baseline": bl_metrics.get("by_league", {}),
            "deltas": per_league_deltas,
        },
    }

    # ── Decision ──
    close_delta = deltas.get("latest_available_no_vig", {})
    acc_delta = close_delta.get("accuracy_delta", -999)
    loss_delta = close_delta.get("logloss_delta", 999)

    if acc_delta > 0.0:
        decision = "PROCEED_TO_POOLING_FULL_ABLATION"
        reason = f"Model beats baseline by {acc_delta:+.4f} accuracy ({euro_m['accuracy']:.4f} vs {bl_metrics.get('latest_available_no_vig', {}).get('accuracy', 0):.4f})"
    elif acc_delta > -0.01 and loss_delta < 0.05:
        decision = "PROCEED_TO_POOLING_FULL_ABLATION"
        reason = f"Model matches baseline (acc={acc_delta:+.4f}) with competitive logloss"
    elif overfitting_warning:
        decision = "FIX_DATA_OR_FEATURES"
        reason = "Overfitting detected"
    else:
        decision = "FIX_DATA_OR_FEATURES"
        reason = f"Model below baseline (acc={acc_delta:+.4f}); need feature improvements"

    report["overall"] = decision
    report["decision"] = {"choice": decision, "reason": reason}

    report_path = os.path.join(args.out_dir, "p1_18_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")

    # ── Print summary ──
    print("\n" + "=" * 60)
    print("P1.18 RESULTS")
    print("=" * 60)
    print(f"Best epoch: {history.get('best_epoch')}  "
          f"Val loss: {history.get('best_val_loss', 0):.4f}")
    print(f"Overfitting: {'YES ⚠️' if overfitting_warning else 'no'}")

    print(f"\n{'':>25} {'Model':>10} {'Baseline':>10} {'Δ':>10}")
    print("-" * 55)
    for name in ["latest_available_no_vig", "close_no_vig", "open_no_vig"]:
        if name not in deltas:
            continue
        d = deltas[name]
        b = bl_metrics.get(name, {})
        print(f"{name:<25} {euro_m['accuracy']:>10.4f} {b.get('accuracy',0):>10.4f} "
              f"{d['accuracy_delta']:>+10.4f}")

    print(f"\nPer-cutoff model vs baseline:")
    for ck in sorted(per_cutoff_deltas.keys(), key=int):
        d = per_cutoff_deltas[ck]
        print(f"  cutoff={ck:>3}min: model={d['model_accuracy']:.4f} "
              f"baseline={d['baseline_accuracy']:.4f} Δ={d['delta']:+.4f}")

    print(f"\nPer-league model vs baseline:")
    for league in sorted(per_league_deltas.keys()):
        d = per_league_deltas[league]
        print(f"  {league:<15}: n={d['count']:>4} model={d['model_accuracy']:.4f} "
              f"baseline={d['baseline_accuracy']:.4f} Δ={d['acc_delta']:+.4f}")

    print(f"\nDecision: {decision}")
    print(f"Reason: {reason}")
    print("=" * 60)


if __name__ == "__main__":
    main()
