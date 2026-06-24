"""
P1.19 Pooling Full Ablation — trains attention/cls pooling, compares all 3.

Reuses P1.18 mean-pooling checkpoint. Trains attention and cls from scratch
with identical config/seed/split. Evaluates all 3 with full metrics.

Usage:
    python eval/eval_p1_19_pooling.py \
        --data data/odds_real/master_5330.jsonl \
        --split-dir data/odds_real/splits_p1_5 \
        --epochs 50 --patience 10 --batch-size 64 \
        --mean-ckpt runs/p1_18_convergence/model_best.pth \
        --out-dir runs/p1_19_pooling
"""

import argparse, json, math, os, random, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch, torch.nn.functional as F
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
    implied_prob_euro, open_no_vig_euro, close_no_vig_euro, euro_probs_to_tensor,
)

EURO_MAP = {"home": 0, "draw": 1, "away": 2}
DEVICE = None

def setup_seed(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_lr(step, total, lr, warmup=0):
    if warmup > 0 and step < warmup: return lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * p)))

# ── Training ───────────────────────────────────────────────────────────

def train_one(model, train_loader, val_loader, epochs, lr, warmup_steps,
              patience, ckpt_path, history_path):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    best_val = float("inf"); best_ep = 0; wait = 0
    hist = {"train_loss": [], "val_loss": [], "epoch": []}
    total_steps = epochs * len(train_loader)

    for ep in range(1, epochs + 1):
        model.train(); tloss = 0.0; steps = len(train_loader)
        for step, batch in enumerate(train_loader, start=1):
            f = batch["features"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
            el = batch["euro_labels"].to(DEVICE); al = batch["asian_labels"].to(DEVICE)
            mm = batch.get("missing_mask")
            if mm is not None: mm = mm.to(DEVICE)

            cur = (ep - 1) * steps + step
            for pg in optimizer.param_groups: pg["lr"] = get_lr(cur, total_steps, lr, warmup_steps)
            optimizer.zero_grad()
            out = model(f, attention_mask=am, euro_labels=el, asian_labels=al, missing_mask=mm)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tloss += out["loss"].item()

        at = tloss / steps; hist["train_loss"].append(at); hist["epoch"].append(ep)

        model.eval(); vloss = 0.0; vn = 0
        with torch.no_grad():
            for batch in val_loader:
                f = batch["features"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
                el = batch["euro_labels"].to(DEVICE); al = batch["asian_labels"].to(DEVICE)
                mm = batch.get("missing_mask")
                if mm is not None: mm = mm.to(DEVICE)
                out = model(f, attention_mask=am, euro_labels=el, asian_labels=al, missing_mask=mm)
                n = f.shape[0]; vloss += out["loss"].item() * n; vn += n
        av = vloss / max(1, vn); hist["val_loss"].append(av)

        print(f"  epoch {ep:3d}/{epochs}  train={at:.4f}  val={av:.4f}  patience={wait}/{patience}")

        if av < best_val: best_val = av; best_ep = ep; wait = 0
        else: wait += 1
        torch.save(model.state_dict(), ckpt_path)  # always save latest
        if wait >= patience: print(f"  -> early stop at {ep}"); break

    hist["best_epoch"] = best_ep; hist["best_val_loss"] = best_val
    hist["early_stop_epoch"] = ep
    with open(history_path, "w") as f: json.dump(hist, f, indent=2)
    # Load best
    ckp = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    if isinstance(ckp, dict) and "model_state_dict" in ckp: ckp = ckp["model_state_dict"]
    model.load_state_dict(ckp, strict=False)
    return hist

# ── Full eval ──────────────────────────────────────────────────────────

def evaluate_full(model, loader):
    model.eval(); eu_log, as_log = [], []; eu_lbl, as_lbl = [], []; leagues = []
    with torch.no_grad():
        for batch in loader:
            f = batch["features"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
            el = batch["euro_labels"].to(DEVICE); al = batch["asian_labels"].to(DEVICE)
            mm = batch.get("missing_mask")
            if mm is not None: mm = mm.to(DEVICE)
            out = model(f, attention_mask=am, missing_mask=mm)
            eu_log.append(out["euro_logits"].cpu()); as_log.append(out["asian_logits"].cpu())
            eu_lbl.append(el.cpu()); as_lbl.append(al.cpu())
            leagues.extend(batch.get("league_ids", [""] * f.shape[0]))
    eu_log = torch.cat(eu_log, 0); as_log = torch.cat(as_log, 0)
    eu_lbl = torch.cat(eu_lbl, 0); as_lbl = torch.cat(as_lbl, 0)
    ece_eu = ece_from_logits(eu_log, eu_lbl, 3)
    return {
        "num_samples": int(eu_log.shape[0]),
        "euro": {"accuracy": accuracy_from_logits(eu_log, eu_lbl),
                  "logloss": logloss_from_logits(eu_log, eu_lbl),
                  "brier": brier_from_logits(eu_log, eu_lbl, 3),
                  "ece": ece_eu["ece"], "reliability_bins": ece_eu["bins"]},
        "euro_logits": eu_log, "euro_labels": eu_lbl, "leagues": leagues,
    }

def per_league_eval(eu_log, eu_lbl, leagues):
    result = {}
    for lg in sorted(set(leagues)):
        if not lg: continue
        idx = torch.tensor([i for i, l in enumerate(leagues) if l == lg])
        if idx.numel() == 0: continue
        el = eu_log[idx]; ll = eu_lbl[idx]; ece = ece_from_logits(el, ll, 3)
        result[lg] = {"count": int(idx.numel()),
                       "accuracy": accuracy_from_logits(el, ll),
                       "logloss": logloss_from_logits(el, ll),
                       "brier": brier_from_logits(el, ll, 3),
                       "ece": ece["ece"]}
    return result

def per_cutoff_eval(model, data_path, cutoffs, asian_label_mode, test_ids):
    result = {}
    for cutoff in cutoffs:
        ck = str(int(cutoff))
        ds = OddsDataset(jsonl_path=data_path, max_seq_len=64, cutoff_minutes=cutoff,
                         cutoff_mode="none", min_events=1, asian_label_mode=asian_label_mode,
                         allowed_match_ids=test_ids, feature_schema_version="v3", seed=42)
        if len(ds) == 0: result[ck] = {"num_samples": 0}; continue
        loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=OddsCollator())
        try:
            m = evaluate_full(model, loader)
            result[ck] = {"num_samples": m["num_samples"],
                          "euro_accuracy": m["euro"]["accuracy"],
                          "euro_logloss": m["euro"]["logloss"]}
        except Exception as e:
            result[ck] = {"num_samples": len(ds), "error": str(e)[:100]}
    return result

def compute_baselines(dataset):
    records = []
    for s in dataset.samples:
        tl = s.get("odds_timeline", []); cutoff = s.get("cutoff_minutes", 0)
        filtered = [e for e in tl if e.get("minutes_before_kickoff", 0) >= cutoff]
        if not filtered: continue
        lg = s.get("league_id", ""); label = EURO_MAP[s["label"]["euro_result"]]
        baselines = {}
        for name, fn in [("close_no_vig", close_no_vig_euro),
                          ("open_no_vig", open_no_vig_euro),
                          ("implied_prob", implied_prob_euro)]:
            try: baselines[name] = euro_probs_to_tensor(fn(filtered))
            except: baselines[name] = torch.full((3,), 1.0/3.0)
        baselines["latest_available_no_vig"] = baselines["close_no_vig"].clone()
        records.append({"cutoff_key": str(int(cutoff)), "league": lg, "label": label, **baselines})
    if not records: return {}
    labels = torch.tensor([r["label"] for r in records])
    def bm(probs, lbls):
        if probs.shape[0] == 0: return {}
        e = ece_from_probs(probs, lbls, 3)
        return {"accuracy": accuracy_from_probs(probs, lbls), "logloss": logloss_from_probs(probs, lbls),
                "brier": brier_from_probs(probs, lbls, 3), "ece": e["ece"]}
    result = {"num_samples": len(records)}
    for name in ["latest_available_no_vig", "open_no_vig", "close_no_vig"]:
        probs = torch.stack([r[name] for r in records])
        result[name] = bm(probs, labels)
    # by cutoff
    by_cutoff = {}
    for ck in sorted(set(r["cutoff_key"] for r in records), key=int):
        c_recs = [r for r in records if r["cutoff_key"] == ck]
        c_l = torch.tensor([r["label"] for r in c_recs])
        e = {"count": len(c_recs)}
        for n in ["latest_available_no_vig"]:
            p = torch.stack([r[n] for r in c_recs])
            e[n + "_accuracy"] = accuracy_from_probs(p, c_l); e[n + "_logloss"] = logloss_from_probs(p, c_l)
        by_cutoff[ck] = e
    result["by_cutoff"] = by_cutoff
    return result

# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="P1.19 Pooling Ablation")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--split-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cutoffs", type=str, default="90,60,30")
    parser.add_argument("--asian-label-mode", type=str, default="5class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mean-ckpt", type=str, default="runs/p1_18_convergence/model_best.pth")
    parser.add_argument("--out-dir", type=str, default="runs/p1_19_pooling")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    global DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    asian_nc = 5 if args.asian_label_mode == "5class" else 3
    os.makedirs(args.out_dir, exist_ok=True)
    setup_seed(args.seed)
    cutoffs = [float(c.strip()) for c in args.cutoffs.split(",") if c.strip()]

    train_ids = load_match_ids_from_file(os.path.join(args.split_dir, "train_match_ids.txt"))
    val_ids = load_match_ids_from_file(os.path.join(args.split_dir, "val_match_ids.txt"))
    test_ids = load_match_ids_from_file(os.path.join(args.split_dir, "test_match_ids.txt"))

    print("=" * 60)
    print("P1.19 Pooling Full Ablation")
    print(f"Data: {args.data}  Split: {args.split_dir}")
    print(f"Train: {len(train_ids)}  Val: {len(val_ids)}  Test: {len(test_ids)}")
    print(f"Cutoffs: {cutoffs}  Epochs: {args.epochs}  Patience: {args.patience}")
    print("=" * 60)

    base_ds_args = dict(jsonl_path=args.data, max_seq_len=64, cutoffs=cutoffs,
                        cutoff_mode="exhaustive", min_events=1,
                        asian_label_mode=args.asian_label_mode, seed=args.seed,
                        feature_schema_version="v3")

    train_ds = OddsDataset(allowed_match_ids=train_ids, **base_ds_args)
    val_ds = OddsDataset(allowed_match_ids=val_ids, **base_ds_args)
    test_ds = OddsDataset(allowed_match_ids=test_ids, **base_ds_args)
    print(f"Samples — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=OddsCollator())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=OddsCollator())
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=OddsCollator())
    warmup = 5 * len(train_loader)

    all_results = {}

    for mode in ["mean", "attention", "cls"]:
        print(f"\n{'='*40}\n  Pooling: {mode}\n{'='*40}")
        ckpt_path = os.path.join(args.out_dir, f"model_{mode}.pth")
        hist_path = os.path.join(args.out_dir, f"history_{mode}.json")

        config = OddsMindConfig(
            hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
            asian_num_classes=asian_nc, dropout=0.1, transformer_backend="odds_native",
            pooling_mode=mode, feature_schema_version="v3",
        )

        if mode == "mean" and args.mean_ckpt and os.path.exists(args.mean_ckpt):
            print(f"  Reusing P1.18 mean checkpoint: {args.mean_ckpt}")
            model = OddsMindModel(config).to(DEVICE)
            ckp = torch.load(args.mean_ckpt, map_location=DEVICE, weights_only=True)
            if isinstance(ckp, dict) and "model_state_dict" in ckp: ckp = ckp["model_state_dict"]
            model.load_state_dict(ckp, strict=False)
            model.eval()
            hist = json.load(open(os.path.join(os.path.dirname(args.mean_ckpt), "training_history.json")))
            # Also copy to out_dir
            torch.save(model.state_dict(), ckpt_path)
            with open(hist_path, "w") as f: json.dump(hist, f)
        elif not args.skip_training:
            model = OddsMindModel(config).to(DEVICE)
            params = sum(p.numel() for p in model.parameters())
            print(f"  Params: {params:,} ({params/1e6:.2f}M)")
            train_loader_r = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=OddsCollator())
            val_loader_r = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=OddsCollator())
            hist = train_one(model, train_loader_r, val_loader_r, args.epochs, args.lr, warmup,
                            args.patience, ckpt_path, hist_path)
        else:
            print(f"  Loading: {ckpt_path}")
            model = OddsMindModel(config).to(DEVICE)
            ckp = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            if isinstance(ckp, dict) and "model_state_dict" in ckp: ckp = ckp["model_state_dict"]
            model.load_state_dict(ckp, strict=False)
            model.eval()
            hist = json.load(open(hist_path))

        # Eval
        model.eval()
        metrics = evaluate_full(model, test_loader)
        pl = per_league_eval(metrics["euro_logits"], metrics["euro_labels"], metrics["leagues"])
        pc = per_cutoff_eval(model, args.data, cutoffs, args.asian_label_mode, test_ids)

        all_results[mode] = {
            "training": {"best_epoch": hist.get("best_epoch", 0),
                          "best_val_loss": hist.get("best_val_loss", 0),
                          "early_stop_epoch": hist.get("early_stop_epoch", 0)},
            "metrics": {k: v for k, v in metrics.items() if k not in ("euro_logits", "euro_labels", "leagues")},
            "per_league": pl, "per_cutoff": pc,
        }
        em = metrics["euro"]
        print(f"  euro: acc={em['accuracy']:.4f} loss={em['logloss']:.4f} "
              f"brier={em['brier']:.4f} ece={em['ece']:.4f}")

    # Baseline
    bl_ds = OddsDataset(allowed_match_ids=test_ids, **{**base_ds_args, "feature_schema_version": "v2"})
    bl = compute_baselines(bl_ds)
    print(f"\nBaseline latest_available_no_vig: acc={bl.get('latest_available_no_vig',{}).get('accuracy',0):.4f}")

    # Comparison table
    print("\n" + "=" * 70)
    print("POOLING ABLATION RESULTS")
    print("=" * 70)
    header = f"{'Metric':<25} {'mean':>10} {'attention':>10} {'cls':>10}"
    print(header); print("-" * 55)
    for metric in ["accuracy", "logloss", "brier", "ece"]:
        vals = []
        for m in ["mean", "attention", "cls"]:
            vals.append(all_results[m]["metrics"]["euro"][metric])
        print(f"{'euro_'+metric:<25} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f}")

    # Deltas vs baseline
    print(f"\n{'Delta vs baseline':<25} {'mean':>10} {'attention':>10} {'cls':>10}")
    print("-" * 55)
    bl_acc = bl.get("latest_available_no_vig", {}).get("accuracy", 0)
    bl_loss = bl.get("latest_available_no_vig", {}).get("logloss", 0)
    for m in ["mean", "attention", "cls"]:
        em = all_results[m]["metrics"]["euro"]
        print(f"{m:<25} {em['accuracy']-bl_acc:>+10.4f} {em['logloss']-bl_loss:>+10.4f}")

    # Selection logic — priority: logloss > brier > ece > accuracy
    modes = ["mean", "attention", "cls"]
    def rank(modes, key):
        return sorted(modes, key=lambda m: all_results[m]["metrics"]["euro"][key])
    by_logloss = rank(modes, "logloss")
    by_brier = rank(modes, "brier")
    by_ece = rank(modes, "ece")
    by_acc = rank(modes, "accuracy")

    # Primary: logloss
    best = by_logloss[0]
    reason = f"lowest logloss ({all_results[best]['metrics']['euro']['logloss']:.4f})"

    if len(by_logloss) > 1 and abs(all_results[by_logloss[0]]["metrics"]["euro"]["logloss"] -
                                     all_results[by_logloss[1]]["metrics"]["euro"]["logloss"]) < 0.005:
        # logloss tie → break by brier
        best = by_brier[0]
        reason += f", logloss tied → lowest brier ({all_results[best]['metrics']['euro']['brier']:.4f})"
        if abs(all_results[by_brier[0]]["metrics"]["euro"]["brier"] -
               all_results[by_brier[1]]["metrics"]["euro"]["brier"]) < 0.001:
            # brier tie → break by ece
            best = by_ece[0]
            reason += f", brier tied → lowest ece ({all_results[best]['metrics']['euro']['ece']:.4f})"

    if all_results[best]["metrics"]["euro"]["ece"] > 0.06:
        decision = "NEED_CALIBRATION_BEFORE_POOLING_DECISION"
        reason += " (⚠️ ECE > 0.06 — calibration recommended)"
    elif best == "mean":
        decision = "KEEP_MEAN_POOLING"
    elif best == "attention":
        decision = "SWITCH_TO_ATTENTION_POOLING"
    else:
        decision = "SWITCH_TO_CLS_POOLING"

    print(f"\nRecommended: {best} ({decision})")
    print(f"Reason: {reason}")

    # Save report
    report = {
        "experiment": "P1.19",
        "overall": decision,
        "setup": {"data": args.data, "split_dir": args.split_dir, "cutoffs": cutoffs,
                   "epochs": args.epochs, "seed": args.seed},
        "baselines": {k: v for k, v in bl.items() if k != "by_cutoff"},
        "results": all_results,
        "decision": {"choice": decision, "recommended_pooling": best, "reason": reason},
    }
    report_path = os.path.join(args.out_dir, "p1_19_report.json")
    with open(report_path, "w") as f: json.dump(report, f, indent=2, default=str)
    print(f"\nReport: {report_path}")

if __name__ == "__main__":
    main()
