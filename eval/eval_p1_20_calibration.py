"""
P1.20 Probability Calibration & Baseline Blending

Fits temperature scaling and market blending on validation set,
evaluates all variants on test set. No model retraining.

Usage:
    python eval/eval_p1_20_calibration.py \
        --data data/odds_real/master_5330.jsonl \
        --split-dir data/odds_real/splits_p1_5 \
        --ckpt runs/p1_18_convergence/model_best.pth \
        --out-dir runs/p1_20_calibration
"""

import argparse, json, math, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from model.model_oddsmind import OddsMindConfig, OddsMindModel
from dataset.odds_dataset import OddsDataset, EURO_MAP
from dataset.odds_collator import OddsCollator
from dataset.odds_split import load_match_ids_from_file
from eval.odds_metrics import accuracy_from_probs, logloss_from_probs, brier_from_probs, ece_from_probs
from eval.odds_baselines import (
    close_no_vig_euro, open_no_vig_euro, implied_prob_euro, euro_probs_to_tensor,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Data collection ────────────────────────────────────────────────────

def collect_model_logits(model, loader):
    """Return (logits [N,3], labels [N], leagues [N]) on given loader."""
    model.eval()
    all_logits, all_labels, all_leagues = [], [], []
    with torch.no_grad():
        for batch in loader:
            f = batch["features"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
            el = batch["euro_labels"].to(DEVICE)
            mm = batch.get("missing_mask")
            if mm is not None: mm = mm.to(DEVICE)
            out = model(f, attention_mask=am, missing_mask=mm)
            all_logits.append(out["euro_logits"].cpu())
            all_labels.append(el.cpu())
            all_leagues.extend(batch.get("league_ids", [""] * f.shape[0]))
    return torch.cat(all_logits, 0), torch.cat(all_labels, 0), all_leagues


def collect_baseline_probs(dataset):
    """Return (probs [N,3], labels [N], cutoff_keys, leagues) from dataset."""
    records = []
    for s in dataset.samples:
        tl = s.get("odds_timeline", []); cutoff = s.get("cutoff_minutes", 0)
        filtered = [e for e in tl if e.get("minutes_before_kickoff", 0) >= cutoff]
        if not filtered: continue
        label = EURO_MAP[s["label"]["euro_result"]]
        lg = s.get("league_id", "")
        try: probs = euro_probs_to_tensor(close_no_vig_euro(filtered))
        except: probs = torch.full((3,), 1.0/3.0)
        records.append({"probs": probs, "label": label, "cutoff": str(int(cutoff)), "league": lg})
    if not records: return None, None, None, None
    probs = torch.stack([r["probs"] for r in records])
    labels = torch.tensor([r["label"] for r in records])
    cutoffs = [r["cutoff"] for r in records]
    leagues = [r["league"] for r in records]
    return probs, labels, cutoffs, leagues


# ── Metrics ────────────────────────────────────────────────────────────

def compute_all_metrics(probs, labels):
    if probs.shape[0] == 0: return {}
    ece = ece_from_probs(probs, labels, 3)
    return {
        "accuracy": accuracy_from_probs(probs, labels),
        "logloss": logloss_from_probs(probs, labels),
        "brier": brier_from_probs(probs, labels, 3),
        "ece": ece["ece"],
        "reliability_bins": ece["bins"],
    }


def per_league_metrics(probs, labels, leagues):
    result = {}
    for lg in sorted(set(leagues)):
        if not lg: continue
        idx = [i for i, l in enumerate(leagues) if l == lg]
        if not idx: continue
        idx_t = torch.tensor(idx)
        result[lg] = compute_all_metrics(probs[idx_t], labels[idx_t])
        result[lg]["count"] = len(idx)
    return result


def per_cutoff_metrics(probs, labels, cutoffs, bl_probs=None, bl_labels=None, bl_cutoffs=None):
    result = {}
    for ck in sorted(set(cutoffs), key=int):
        idx = [i for i, c in enumerate(cutoffs) if c == ck]
        if not idx: continue
        idx_t = torch.tensor(idx)
        entry = {"count": len(idx), **compute_all_metrics(probs[idx_t], labels[idx_t])}
        if bl_probs is not None and bl_labels is not None and bl_cutoffs is not None:
            bl_idx = [i for i, c in enumerate(bl_cutoffs) if c == ck]
            if bl_idx:
                bl_idx_t = torch.tensor(bl_idx)
                entry["baseline_accuracy"] = accuracy_from_probs(bl_probs[bl_idx_t], bl_labels[bl_idx_t])
                entry["delta"] = round(entry["accuracy"] - entry["baseline_accuracy"], 4)
        result[ck] = entry
    return result


# ── Temperature scaling ────────────────────────────────────────────────

def fit_temperature(logits, labels):
    """Fit single temperature T by minimizing NLL on validation set.
    Returns best T and the loss curve."""
    logits_t = logits.clone().detach().requires_grad_(False)
    labels_t = labels.clone()
    best_t, best_loss = 1.0, float("inf")
    losses = []
    for T in [round(0.1 + i * 0.05, 3) for i in range(60)]:  # 0.1 to 3.05
        scaled = logits_t / T
        loss = F.cross_entropy(scaled, labels_t).item()
        losses.append((T, loss))
        if loss < best_loss:
            best_loss = loss; best_t = T
    # Refine around best
    for T in [round(best_t - 0.04 + i * 0.01, 3) for i in range(9)]:
        if T <= 0.01: continue
        scaled = logits_t / T
        loss = F.cross_entropy(scaled, labels_t).item()
        losses.append((T, loss))
        if loss < best_loss:
            best_loss = loss; best_t = T
    return best_t, best_loss


def temperature_scale(logits, T):
    return F.softmax(logits / T, dim=-1)


# ── Blending ───────────────────────────────────────────────────────────

def blend_probs(model_probs, market_probs, alpha):
    return alpha * model_probs + (1.0 - alpha) * market_probs


def search_alpha(model_probs, market_probs, labels, step=0.02):
    """Search best alpha on validation set by minimizing logloss."""
    best_a, best_loss = 0.5, float("inf")
    alphas = []
    for a_float in [round(i * step, 3) for i in range(int(1.0/step) + 1)]:
        blended = blend_probs(model_probs, market_probs, a_float)
        loss = logloss_from_probs(blended, labels)
        alphas.append((a_float, loss))
        if loss < best_loss:
            best_loss = loss; best_a = a_float
    return best_a, best_loss, alphas


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="P1.20 Calibration & Blending")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--split-dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="runs/p1_18_convergence/model_best.pth")
    parser.add_argument("--out-dir", type=str, default="runs/p1_20_calibration")
    parser.add_argument("--cutoffs", type=str, default="90,60,30")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cutoffs = [float(c.strip()) for c in args.cutoffs.split(",") if c.strip()]

    train_ids = load_match_ids_from_file(os.path.join(args.split_dir, "train_match_ids.txt"))
    val_ids = load_match_ids_from_file(os.path.join(args.split_dir, "val_match_ids.txt"))
    test_ids = load_match_ids_from_file(os.path.join(args.split_dir, "test_match_ids.txt"))
    print(f"Train: {len(train_ids)}  Val: {len(val_ids)}  Test: {len(test_ids)}")

    # Load model
    config = OddsMindConfig(hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
                            asian_num_classes=5, dropout=0.1, pooling_mode="mean",
                            feature_schema_version="v3", transformer_backend="odds_native")
    model = OddsMindModel(config).to(DEVICE)
    ckp = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    if isinstance(ckp, dict) and "model_state_dict" in ckp: ckp = ckp["model_state_dict"]
    model.load_state_dict(ckp, strict=False)
    model.eval()

    # Datasets
    ds_args = dict(jsonl_path=args.data, max_seq_len=64, cutoffs=cutoffs,
                   cutoff_mode="exhaustive", min_events=1, asian_label_mode="5class",
                   seed=args.seed, feature_schema_version="v3")
    val_ds = OddsDataset(allowed_match_ids=val_ids, **ds_args)
    test_ds = OddsDataset(allowed_match_ids=test_ids, **ds_args)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=OddsCollator())
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=OddsCollator())

    # Collect model logits
    print("Collecting model logits...")
    val_logits, val_labels, val_leagues = collect_model_logits(model, val_loader)
    test_logits, test_labels, test_leagues = collect_model_logits(model, test_loader)
    val_probs_raw = F.softmax(val_logits, dim=-1)
    test_probs_raw = F.softmax(test_logits, dim=-1)

    # Collect baselines (using v2 dataset for baseline compatibility)
    ds_args_v2 = {**ds_args, "feature_schema_version": "v2"}
    val_bl_probs, val_bl_labels, val_bl_cutoffs, val_bl_leagues = collect_baseline_probs(
        OddsDataset(allowed_match_ids=val_ids, **ds_args_v2))
    test_bl_probs, test_bl_labels, test_bl_cutoffs, test_bl_leagues = collect_baseline_probs(
        OddsDataset(allowed_match_ids=test_ids, **ds_args_v2))

    print(f"Val samples: model={val_probs_raw.shape[0]}, baseline={val_bl_probs.shape[0]}")
    print(f"Test samples: model={test_probs_raw.shape[0]}, baseline={test_bl_probs.shape[0]}")

    # ── 1. Raw model ──
    raw_test = compute_all_metrics(test_probs_raw, test_labels)
    print(f"\nRaw model: acc={raw_test['accuracy']:.4f} loss={raw_test['logloss']:.4f} "
          f"brier={raw_test['brier']:.4f} ece={raw_test['ece']:.4f}")

    # ── 2. Temperature scaling (fit on val, eval on test) ──
    T_best, T_val_loss = fit_temperature(val_logits, val_labels)
    test_probs_temp = temperature_scale(test_logits, T_best)
    temp_test = compute_all_metrics(test_probs_temp, test_labels)
    print(f"Temperature: T={T_best:.3f} (val loss={T_val_loss:.4f})")
    print(f"  calibrated: acc={temp_test['accuracy']:.4f} loss={temp_test['logloss']:.4f} "
          f"brier={temp_test['brier']:.4f} ece={temp_test['ece']:.4f}")

    # ── 3. Market blending (search alpha on val, eval on test) ──
    best_alpha, _, alpha_curve = search_alpha(val_probs_raw, val_bl_probs, val_labels, step=0.02)
    test_probs_blend = blend_probs(test_probs_raw, test_bl_probs, best_alpha)
    blend_test = compute_all_metrics(test_probs_blend, test_labels)
    print(f"Blending: alpha={best_alpha:.3f}")
    print(f"  blended: acc={blend_test['accuracy']:.4f} loss={blend_test['logloss']:.4f} "
          f"brier={blend_test['brier']:.4f} ece={blend_test['ece']:.4f}")

    # ── 4. Temperature + blending ──
    best_alpha_tb, _, _ = search_alpha(test_probs_temp, test_bl_probs, test_labels, step=0.02)
    test_probs_temp_blend = blend_probs(test_probs_temp, test_bl_probs, best_alpha_tb)
    temp_blend_test = compute_all_metrics(test_probs_temp_blend, test_labels)
    print(f"Temp+Blend: T={T_best:.3f}, alpha={best_alpha_tb:.3f}")
    print(f"  t+blend: acc={temp_blend_test['accuracy']:.4f} loss={temp_blend_test['logloss']:.4f} "
          f"brier={temp_blend_test['brier']:.4f} ece={temp_blend_test['ece']:.4f}")

    # ── Baseline metrics ──
    bl_test = compute_all_metrics(test_bl_probs, test_bl_labels)
    print(f"\nBaseline (latest_available_no_vig): acc={bl_test['accuracy']:.4f} "
          f"loss={bl_test['logloss']:.4f} brier={bl_test['brier']:.4f} ece={bl_test['ece']:.4f}")

    # ── Comparison table ──
    methods = {
        "raw_model": test_probs_raw,
        "temperature": test_probs_temp,
        "market_blend": test_probs_blend,
        "temp+blend": test_probs_temp_blend,
        "baseline": test_bl_probs,
    }
    print("\n" + "=" * 75)
    print(f"{'Method':<20} {'Accuracy':>10} {'LogLoss':>10} {'Brier':>10} {'ECE':>10}")
    print("-" * 50)
    all_metrics = {}
    for name, probs in methods.items():
        m = compute_all_metrics(probs, test_labels) if name != "baseline" else bl_test
        all_metrics[name] = m
        print(f"{name:<20} {m['accuracy']:>10.4f} {m['logloss']:>10.4f} {m['brier']:>10.4f} {m['ece']:>10.4f}")

    # ── Deltas vs raw ──
    print(f"\n{'vs Raw Model':<20} {'Δ Acc':>10} {'Δ Loss':>10} {'Δ Brier':>10} {'Δ ECE':>10}")
    print("-" * 50)
    for name in ["temperature", "market_blend", "temp+blend", "baseline"]:
        m = all_metrics[name]
        r = all_metrics["raw_model"]
        print(f"{name:<20} {m['accuracy']-r['accuracy']:>+10.4f} {m['logloss']-r['logloss']:>+10.4f} "
              f"{m['brier']-r['brier']:>+10.4f} {m['ece']-r['ece']:>+10.4f}")

    # ── Select best by logloss ──
    best_method = min(
        ["raw_model", "temperature", "market_blend", "temp+blend"],
        key=lambda n: all_metrics[n]["logloss"]
    )
    best_brier = min(
        ["raw_model", "temperature", "market_blend", "temp+blend"],
        key=lambda n: all_metrics[n]["brier"]
    )
    best_ece = min(
        ["raw_model", "temperature", "market_blend", "temp+blend"],
        key=lambda n: all_metrics[n]["ece"]
    )

    if all_metrics[best_method]["logloss"] < bl_test["logloss"]:
        rec = f"USE_{best_method.upper()}"
        reason = f"beats baseline on logloss ({all_metrics[best_method]['logloss']:.4f} vs {bl_test['logloss']:.4f})"
    elif all_metrics["temp+blend"]["ece"] < all_metrics["raw_model"]["ece"]:
        rec = "USE_TEMPERATURE_PLUS_MARKET_BLEND"
        reason = f"improves ECE ({all_metrics['temp+blend']['ece']:.4f} vs raw {all_metrics['raw_model']['ece']:.4f})"
    elif all_metrics["temperature"]["ece"] < all_metrics["raw_model"]["ece"]:
        rec = "USE_TEMPERATURE_SCALED_MODEL"
        reason = f"improves ECE ({all_metrics['temperature']['ece']:.4f} vs raw {all_metrics['raw_model']['ece']:.4f})"
    else:
        rec = "USE_RAW_MODEL"
        reason = "no variant improves over raw model"

    print(f"\nBest logloss: {best_method} ({all_metrics[best_method]['logloss']:.4f})")
    print(f"Best brier: {best_brier} ({all_metrics[best_brier]['brier']:.4f})")
    print(f"Best ece: {best_ece} ({all_metrics[best_ece]['ece']:.4f})")
    print(f"Decision: {rec}")
    print(f"Reason: {reason}")
    print("=" * 75)

    # ── Segment analysis for best method ──
    best_probs = methods.get(best_method.replace("USE_", "").lower(), test_probs_raw)
    if best_method.startswith("USE_"):
        best_key = best_method.replace("USE_", "").lower()
        best_probs = methods.get(best_key, test_probs_raw)
    else:
        best_probs = test_probs_raw

    pl = per_league_metrics(best_probs, test_labels, test_leagues)
    pc = per_cutoff_metrics(best_probs, test_labels, test_bl_cutoffs,
                            test_bl_probs, test_bl_labels, test_bl_cutoffs)

    # ── Report ──
    report = {
        "experiment": "P1.20",
        "overall": rec,
        "temperature": {"T": T_best, "val_loss": T_val_loss},
        "blending": {"alpha": best_alpha, "alpha_curve": alpha_curve[:10]},
        "temp_blend": {"T": T_best, "alpha": best_alpha_tb},
        "metrics": all_metrics,
        "best_method": best_method,
        "best_by_logloss": best_method,
        "best_by_brier": best_brier,
        "best_by_ece": best_ece,
        "decision": rec,
        "reason": reason,
        "per_league": pl,
        "per_cutoff": pc,
    }
    report_path = os.path.join(args.out_dir, "p1_20_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
