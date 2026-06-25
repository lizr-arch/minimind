"""
P1.5 Verify: v5 schema + league embedding vs v3 baseline

Reuses P1.18 v3 checkpoint for comparison. Trains v5 from scratch
with identical config/seed/split. Compares accuracy/logloss/brier/ECE.

Usage:
    python eval/eval_p1_5_verify.py \
        --data data/odds_real/master_5330.jsonl \
        --split-dir data/odds_real/splits_p1_5 \
        --v3-ckpt runs/p1_18_convergence/model_best.pth \
        --epochs 50 --patience 10 --batch-size 64 \
        --out-dir runs/p1_5_verify
"""

import argparse, json, math, os, random, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch, torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from model.model_oddsmind import OddsMindConfig, OddsMindModel
from dataset.odds_dataset import OddsDataset, register_leagues, lock_league_registry, get_league_count
from dataset.odds_collator import OddsCollator
from dataset.odds_split import load_match_ids_from_file
from eval.odds_metrics import (
    accuracy_from_logits, logloss_from_logits, brier_from_logits, ece_from_logits,
    accuracy_from_probs, logloss_from_probs, brier_from_probs, ece_from_probs,
)
from eval.odds_baselines import close_no_vig_euro, euro_probs_to_tensor
from dataset.odds_dataset import EURO_MAP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def setup_seed(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_lr(step, total, lr, warmup=0):
    if warmup > 0 and step < warmup: return lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * p)))

# ── Training ───────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, epochs, lr, warmup, patience, ckpt_path):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    best_val = float("inf"); best_ep = 0; wait = 0
    total_steps = epochs * len(train_loader)
    for ep in range(1, epochs + 1):
        model.train(); tloss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            f = batch["features"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
            el = batch["euro_labels"].to(DEVICE); al = batch["asian_labels"].to(DEVICE)
            mm = batch.get("missing_mask")
            if mm is not None: mm = mm.to(DEVICE)
            lid = batch.get("league_id_tensor")
            if lid is not None: lid = lid.to(DEVICE)
            cur = (ep - 1) * len(train_loader) + step
            for pg in optimizer.param_groups: pg["lr"] = get_lr(cur, total_steps, lr, warmup)
            optimizer.zero_grad()
            out = model(f, attention_mask=am, euro_labels=el, asian_labels=al,
                        missing_mask=mm, league_ids=lid)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); tloss += out["loss"].item()

        model.eval(); vloss = 0.0; vn = 0
        with torch.no_grad():
            for batch in val_loader:
                f = batch["features"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
                el = batch["euro_labels"].to(DEVICE); al = batch["asian_labels"].to(DEVICE)
                mm = batch.get("missing_mask")
                if mm is not None: mm = mm.to(DEVICE)
                lid = batch.get("league_id_tensor")
                if lid is not None: lid = lid.to(DEVICE)
                out = model(f, attention_mask=am, euro_labels=el, asian_labels=al,
                            missing_mask=mm, league_ids=lid)
                n = f.shape[0]; vloss += out["loss"].item() * n; vn += n
        av = vloss / max(1, vn)
        at = tloss / len(train_loader)
        print(f"  epoch {ep:3d}/{epochs}  train={at:.4f}  val={av:.4f}  patience={wait}/{patience}")
        if av < best_val: best_val = av; best_ep = ep; wait = 0; torch.save(model.state_dict(), ckpt_path)
        else: wait += 1
        if wait >= patience: print(f"  -> early stop at {ep}"); break

    ckp = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    if isinstance(ckp, dict) and "model_state_dict" in ckp: ckp = ckp["model_state_dict"]
    model.load_state_dict(ckp, strict=False)
    return best_ep, best_val

# ── Eval ───────────────────────────────────────────────────────────────

def evaluate(model, loader):
    model.eval(); eu_log, eu_lbl, leagues = [], [], []
    with torch.no_grad():
        for batch in loader:
            f = batch["features"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
            el = batch["euro_labels"].to(DEVICE)
            mm = batch.get("missing_mask")
            if mm is not None: mm = mm.to(DEVICE)
            lid = batch.get("league_id_tensor")
            if lid is not None: lid = lid.to(DEVICE)
            out = model(f, attention_mask=am, missing_mask=mm, league_ids=lid)
            eu_log.append(out["euro_logits"].cpu()); eu_lbl.append(el.cpu())
            leagues.extend(batch.get("league_ids", [""] * f.shape[0]))
    eu_log = torch.cat(eu_log, 0); eu_lbl = torch.cat(eu_lbl, 0)
    probs = F.softmax(eu_log / 1.09, dim=-1)
    ece = ece_from_probs(probs, eu_lbl, 3)
    return {
        "num_samples": int(eu_log.shape[0]),
        "accuracy": accuracy_from_probs(probs, eu_lbl),
        "logloss": logloss_from_probs(probs, eu_lbl),
        "brier": brier_from_probs(probs, eu_lbl, 3),
        "ece": ece["ece"],
        "logits": eu_log, "labels": eu_lbl, "leagues": leagues,
    }

def per_league_metrics(logits, labels, leagues):
    result = {}
    for lg in sorted(set(leagues)):
        if not lg: continue
        idx = torch.tensor([i for i, l in enumerate(leagues) if l == lg])
        if idx.numel() == 0: continue
        p = F.softmax(logits[idx] / 1.09, dim=-1)
        l = labels[idx]
        result[lg] = {
            "count": int(idx.numel()),
            "accuracy": accuracy_from_probs(p, l),
            "logloss": logloss_from_probs(p, l),
            "brier": brier_from_probs(p, l, 3),
            "ece": ece_from_probs(p, l, 3)["ece"],
        }
    return result

# ── Baseline ───────────────────────────────────────────────────────────

def baseline_metrics(dataset):
    recs = []
    for s in dataset.samples:
        tl = s.get("odds_timeline", []); cutoff = s.get("cutoff_minutes", 0)
        filtered = [e for e in tl if e.get("minutes_before_kickoff", 0) >= cutoff]
        if not filtered: continue
        try: probs = euro_probs_to_tensor(close_no_vig_euro(filtered))
        except: probs = torch.full((3,), 1.0/3.0)
        recs.append({"probs": probs, "label": EURO_MAP[s["label"]["euro_result"]],
                      "league": s.get("league_id", "")})
    if not recs: return {}, {}, {}
    probs = torch.stack([r["probs"] for r in recs])
    labels = torch.tensor([r["label"] for r in recs])
    leagues = [r["league"] for r in recs]
    overall = {"accuracy": accuracy_from_probs(probs, labels),
               "logloss": logloss_from_probs(probs, labels),
               "brier": brier_from_probs(probs, labels, 3),
               "ece": ece_from_probs(probs, labels, 3)["ece"]}
    pl = {}
    for lg in sorted(set(leagues)):
        if not lg: continue
        idx = torch.tensor([i for i, l in enumerate(leagues) if l == lg])
        if idx.numel() == 0: continue
        pl[lg] = {"count": int(idx.numel()), "accuracy": accuracy_from_probs(probs[idx], labels[idx])}
    return overall, pl, leagues

# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="P1.5 Verify v5 vs v3")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--split-dir", type=str, required=True)
    parser.add_argument("--v3-ckpt", type=str, default="runs/p1_18_convergence/model_best.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cutoffs", type=str, default="90,60,30")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="runs/p1_5_verify")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    setup_seed(args.seed)
    cutoffs = [float(c.strip()) for c in args.cutoffs.split(",") if c.strip()]

    train_ids = load_match_ids_from_file(os.path.join(args.split_dir, "train_match_ids.txt"))
    val_ids = load_match_ids_from_file(os.path.join(args.split_dir, "val_match_ids.txt"))
    test_ids = load_match_ids_from_file(os.path.join(args.split_dir, "test_match_ids.txt"))
    print(f"Train:{len(train_ids)} Val:{len(val_ids)} Test:{len(test_ids)}")

    # P0: scan data for all league IDs and register them
    print("Registering leagues...")
    all_leagues = set()
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                m = json.loads(line)
                all_leagues.add(m.get("league_id", ""))
    register_leagues(list(all_leagues - {""}))
    lock_league_registry()
    num_leagues = get_league_count()
    print(f"  {num_leagues} leagues registered")

    ds_args = dict(jsonl_path=args.data, max_seq_len=64, cutoffs=cutoffs,
                   cutoff_mode="exhaustive", min_events=1, asian_label_mode="5class",
                   seed=args.seed)

    # ── Train v5 ──
    ckpt_path = os.path.join(args.out_dir, "model_v5.pth")
    config_v5 = OddsMindConfig(hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
                               asian_num_classes=5, dropout=0.1, pooling_mode="mean",
                               feature_schema_version="v5", num_leagues=num_leagues,
                               transformer_backend="odds_native")

    if not args.skip_training:
        print("Loading train dataset...")
        train_ds = OddsDataset(allowed_match_ids=train_ids, feature_schema_version="v5", **ds_args)
        print(f"  {len(train_ds)} samples")
        print("Loading val dataset...")
        val_ds = OddsDataset(allowed_match_ids=val_ids, feature_schema_version="v5", **ds_args)
        print(f"  {len(val_ds)} samples")
        tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=OddsCollator())
        vl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=OddsCollator())
        warmup = 5 * len(tl)
        model = OddsMindModel(config_v5).to(DEVICE)
        params = sum(p.numel() for p in model.parameters())
        print(f"\nv5 training: {params:,} params, {len(train_ds)} samples")
        best_ep, best_val = train(model, tl, vl, args.epochs, args.lr, warmup, args.patience, ckpt_path)
        print(f"v5 done: best_epoch={best_ep}, best_val_loss={best_val:.4f}")
    else:
        print(f"Loading v5 from {ckpt_path}")

    # ── Eval v5 ──
    print("\n--- v5 evaluation ---")
    model_v5 = OddsMindModel(config_v5).to(DEVICE)
    ckp = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    if isinstance(ckp, dict) and "model_state_dict" in ckp: ckp = ckp["model_state_dict"]
    model_v5.load_state_dict(ckp, strict=False); model_v5.eval()

    test_ds_v5 = OddsDataset(allowed_match_ids=test_ids, feature_schema_version="v5", **ds_args)
    test_loader_v5 = DataLoader(test_ds_v5, batch_size=64, shuffle=False, collate_fn=OddsCollator())
    m5 = evaluate(model_v5, test_loader_v5)
    pl5 = per_league_metrics(m5["logits"], m5["labels"], m5["leagues"])

    # ── Eval v3 ──
    print("--- v3 evaluation ---")
    config_v3 = OddsMindConfig(hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
                               asian_num_classes=5, dropout=0.1, pooling_mode="mean",
                               feature_schema_version="v3", transformer_backend="odds_native")
    model_v3 = OddsMindModel(config_v3).to(DEVICE)
    ckp3 = torch.load(args.v3_ckpt, map_location=DEVICE, weights_only=True)
    if isinstance(ckp3, dict) and "model_state_dict" in ckp3: ckp3 = ckp3["model_state_dict"]
    model_v3.load_state_dict(ckp3, strict=False); model_v3.eval()

    test_ds_v3 = OddsDataset(allowed_match_ids=test_ids, feature_schema_version="v3", **ds_args)
    test_loader_v3 = DataLoader(test_ds_v3, batch_size=64, shuffle=False, collate_fn=OddsCollator())
    m3 = evaluate(model_v3, test_loader_v3)
    pl3 = per_league_metrics(m3["logits"], m3["labels"], m3["leagues"])

    # ── Baseline ──
    print("--- baseline ---")
    bl_ds = OddsDataset(allowed_match_ids=test_ids, feature_schema_version="v2", **ds_args)
    bl_overall, bl_pl, _ = baseline_metrics(bl_ds)

    # ── Print comparison ──
    print("\n" + "=" * 70)
    print("P1.5 VERIFY: v5 vs v3 vs baseline")
    print("=" * 70)
    print(f"{'Metric':<20} {'v3':>12} {'v5':>12} {'Δ':>10} {'baseline':>12}")
    print("-" * 66)
    for k in ["accuracy", "logloss", "brier", "ece"]:
        print(f"{k:<20} {m3[k]:>12.4f} {m5[k]:>12.4f} {m5[k]-m3[k]:>+10.4f} {bl_overall[k]:>12.4f}")

    print(f"\n{'Per-league accuracy':<20} {'v3':>12} {'v5':>12} {'Δ':>10} {'baseline':>12}")
    print("-" * 66)
    for lg in sorted(set(list(pl3.keys()) + list(pl5.keys()))):
        v3a = pl3.get(lg, {}).get("accuracy", 0)
        v5a = pl5.get(lg, {}).get("accuracy", 0)
        bla = bl_pl.get(lg, {}).get("accuracy", 0)
        print(f"{lg:<20} {v3a:>12.4f} {v5a:>12.4f} {v5a-v3a:>+10.4f} {bla:>12.4f}")

    # Decision
    v5_win = sum(1 for lg in pl3 if pl5.get(lg, {}).get("accuracy", 0) > pl3.get(lg, {}).get("accuracy", 0))
    total = len([lg for lg in pl3 if lg in pl5])
    print(f"\nv5 beats v3 on {v5_win}/{total} leagues")
    if m5["accuracy"] > m3["accuracy"]:
        print(f"v5 overall accuracy +{m5['accuracy']-m3['accuracy']:.4f} — P1 effective, proceed to P2")
    else:
        print(f"v5 no overall improvement ({m5['accuracy']-m3['accuracy']:+.4f}) — wait for P0 data")

    # Save
    report = {"v3": {k: v for k, v in m3.items() if k not in ("logits", "labels", "leagues")},
              "v5": {k: v for k, v in m5.items() if k not in ("logits", "labels", "leagues")},
              "baseline": bl_overall, "per_league_v3": pl3, "per_league_v5": pl5,
              "per_league_bl": bl_pl}
    with open(os.path.join(args.out_dir, "p1_5_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport: {os.path.join(args.out_dir, 'p1_5_report.json')}")

if __name__ == "__main__":
    main()
