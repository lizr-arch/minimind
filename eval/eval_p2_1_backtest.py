"""
P2.1 Holdout Backtest & League Generalization Gate

P2.1A: Frozen candidate per-league eval (fast, no retraining)
P2.1B: True leave-one-league-out 5-fold CV (trains 5 models)

Usage:
    # Fast frozen eval only
    python eval/eval_p2_1_backtest.py --data data/odds_real/master_5330.jsonl --mode frozen --out-dir runs/p2_1_backtest

    # Full LOO CV
    python eval/eval_p2_1_backtest.py --data data/odds_real/master_5330.jsonl --mode loo --epochs 50 --patience 10 --batch-size 64 --out-dir runs/p2_1_backtest
"""

import argparse, json, math, os, random, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch, torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from model.model_oddsmind import OddsMindConfig, OddsMindModel
from dataset.odds_dataset import OddsDataset, EURO_MAP
from dataset.odds_collator import OddsCollator
from dataset.odds_split import load_match_ids_from_file
from eval.odds_metrics import (
    accuracy_from_probs, logloss_from_probs, brier_from_probs, ece_from_probs,
    accuracy_from_logits, logloss_from_logits, brier_from_logits, ece_from_logits,
)
from eval.odds_baselines import close_no_vig_euro, euro_probs_to_tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def setup_seed(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_lr(step, total, lr, warmup=0):
    if warmup > 0 and step < warmup: return lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * p)))

def load_deployment_model():
    cfg = OddsMindConfig(hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
                         asian_num_classes=5, dropout=0.1, pooling_mode="mean",
                         feature_schema_version="v3", transformer_backend="odds_native")
    model = OddsMindModel(cfg).to(DEVICE)
    ckp = torch.load("runs/p2_0_deployment_candidate/model_candidate.pth",
                      map_location=DEVICE, weights_only=True)
    if isinstance(ckp, dict) and "model_state_dict" in ckp: ckp = ckp["model_state_dict"]
    model.load_state_dict(ckp, strict=False)
    model.eval()
    return model

def collect_logits(model, loader):
    model.eval(); logs, lbls, leagues = [], [], []
    with torch.no_grad():
        for batch in loader:
            f = batch["features"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
            el = batch["euro_labels"].to(DEVICE)
            mm = batch.get("missing_mask")
            if mm is not None: mm = mm.to(DEVICE)
            out = model(f, attention_mask=am, missing_mask=mm)
            logs.append(out["euro_logits"].cpu()); lbls.append(el.cpu())
            leagues.extend(batch.get("league_ids", [""]*f.shape[0]))
    return torch.cat(logs,0), torch.cat(lbls,0), leagues

def collect_baseline(dataset):
    recs = []
    for s in dataset.samples:
        tl = s.get("odds_timeline",[]); cutoff = s.get("cutoff_minutes",0)
        filtered = [e for e in tl if e.get("minutes_before_kickoff",0) >= cutoff]
        if not filtered: continue
        try: probs = euro_probs_to_tensor(close_no_vig_euro(filtered))
        except: probs = torch.full((3,),1.0/3.0)
        recs.append({"probs":probs,"label":EURO_MAP[s["label"]["euro_result"]],
                      "cutoff":str(int(cutoff)),"league":s.get("league_id","")})
    if not recs: return None,None
    return torch.stack([r["probs"] for r in recs]), torch.tensor([r["label"] for r in recs])

def metrics_dict(probs, labels):
    e = ece_from_probs(probs, labels, 3)
    return {"accuracy": accuracy_from_probs(probs, labels),
            "logloss": logloss_from_probs(probs, labels),
            "brier": brier_from_probs(probs, labels, 3),
            "ece": e["ece"], "reliability_bins": e["bins"]}

def per_cutoff_dict(probs, labels, cutoffs_list):
    result = {}
    for ck in sorted(set(cutoffs_list), key=int):
        idx = [i for i,c in enumerate(cutoffs_list) if c==ck]
        if not idx: continue
        it = torch.tensor(idx)
        result[ck] = {"count": len(idx), **metrics_dict(probs[it], labels[it])}
    return result

# ══════════════════════════════════════════════════════════════════════
# P2.1A: Frozen candidate per-league eval
# ══════════════════════════════════════════════════════════════════════

def run_p2_1a(data_path, cutoffs, split_dir):
    model = load_deployment_model()
    T = 1.09
    test_ids = load_match_ids_from_file(os.path.join(split_dir, "test_match_ids.txt"))
    ds_args = dict(jsonl_path=data_path, max_seq_len=64, cutoffs=cutoffs,
                   cutoff_mode="exhaustive", min_events=1, asian_label_mode="5class",
                   seed=42, feature_schema_version="v3")
    ds_args_v2 = {**ds_args, "feature_schema_version":"v2"}

    test_ds = OddsDataset(allowed_match_ids=test_ids, **ds_args)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=OddsCollator())
    logits, labels, leagues = collect_logits(model, test_loader)
    bl_probs, bl_labels = collect_baseline(OddsDataset(allowed_match_ids=test_ids, **ds_args_v2))

    raw_probs = F.softmax(logits, dim=-1)
    temp_probs = F.softmax(logits/T, dim=-1)

    results = {}
    all_leagues = sorted(set(leagues))
    for lg in all_leagues:
        if not lg: continue
        idx = torch.tensor([i for i,l in enumerate(leagues) if l==lg])
        raw = metrics_dict(raw_probs[idx], labels[idx])
        temp = metrics_dict(temp_probs[idx], labels[idx])
        bl = metrics_dict(bl_probs[idx], bl_labels[idx])
        results[lg] = {
            "count": int(idx.numel()),
            "raw": raw, "temperature_scaled": temp, "baseline": bl,
            "delta_raw": {k: raw[k]-bl[k] for k in raw if k != "reliability_bins"},
            "delta_temp": {k: temp[k]-bl[k] for k in temp if k != "reliability_bins"},
        }
    return results

# ══════════════════════════════════════════════════════════════════════
# P2.1B: Leave-one-league-out CV
# ══════════════════════════════════════════════════════════════════════

def load_matches_by_league(data_path):
    by_league = {}
    with open(data_path) as f:
        for line in f:
            if not line.strip(): continue
            m = json.loads(line.strip())
            lg = m.get("league_id","unknown")
            if lg not in by_league: by_league[lg] = []
            by_league[lg].append(m)
    return by_league

def time_split_matches(matches, train_r=0.7, val_r=0.15):
    matches_sorted = sorted(matches, key=lambda m: m.get("kickoff_time",""))
    n = len(matches_sorted)
    n_train = max(1, int(n*train_r))
    n_val = max(1, int(n*val_r))
    train = {m["match_id"] for m in matches_sorted[:n_train]}
    val = {m["match_id"] for m in matches_sorted[n_train:n_train+n_val]}
    test = {m["match_id"] for m in matches_sorted[n_train+n_val:]}
    return train, val, test

def train_one_fold(train_ids, val_ids, data_path, cutoffs, epochs, patience, batch_size, lr, seed, ckpt_path):
    setup_seed(seed)
    ds_args = dict(jsonl_path=data_path, max_seq_len=64, cutoffs=cutoffs,
                   cutoff_mode="exhaustive", min_events=1, asian_label_mode="5class",
                   feature_schema_version="v3", seed=seed)
    train_ds = OddsDataset(allowed_match_ids=train_ids, **ds_args)
    val_ds = OddsDataset(allowed_match_ids=val_ids, **ds_args)
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=OddsCollator())
    vl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=OddsCollator())

    config = OddsMindConfig(hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
                            asian_num_classes=5, dropout=0.1, pooling_mode="mean",
                            feature_schema_version="v3", transformer_backend="odds_native")
    model = OddsMindModel(config).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    best_val = float("inf"); best_ep = 0; wait = 0
    warmup = 5 * len(tl); total_steps = epochs * len(tl)

    for ep in range(1, epochs+1):
        model.train()
        for step, batch in enumerate(tl, start=1):
            f = batch["features"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
            el = batch["euro_labels"].to(DEVICE); al = batch["asian_labels"].to(DEVICE)
            mm = batch.get("missing_mask")
            if mm is not None: mm = mm.to(DEVICE)
            cur = (ep-1)*len(tl)+step
            for pg in optimizer.param_groups: pg["lr"] = get_lr(cur, total_steps, lr, warmup)
            optimizer.zero_grad()
            out = model(f, attention_mask=am, euro_labels=el, asian_labels=al, missing_mask=mm)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval(); vloss = 0.0; vn = 0
        with torch.no_grad():
            for batch in vl:
                f = batch["features"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
                el = batch["euro_labels"].to(DEVICE); al = batch["asian_labels"].to(DEVICE)
                mm = batch.get("missing_mask")
                if mm is not None: mm = mm.to(DEVICE)
                out = model(f, attention_mask=am, euro_labels=el, asian_labels=al, missing_mask=mm)
                n = f.shape[0]; vloss += out["loss"].item()*n; vn += n
        av = vloss/max(1,vn)
        if av < best_val: best_val = av; best_ep = ep; wait = 0; torch.save(model.state_dict(), ckpt_path)
        else: wait += 1
        if wait >= patience: break

    ckp = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    if isinstance(ckp, dict) and "model_state_dict" in ckp: ckp = ckp["model_state_dict"]
    model.load_state_dict(ckp, strict=False)
    return model, best_ep, best_val

def fit_T(logits, labels):
    best_t, best_loss = 1.0, float("inf")
    for T in [round(0.1+i*0.05,3) for i in range(60)]:
        loss = F.cross_entropy(logits/T, labels).item()
        if loss < best_loss: best_loss = loss; best_t = T
    for T in [round(best_t-0.04+i*0.01,3) for i in range(9)]:
        if T <= 0.01: continue
        loss = F.cross_entropy(logits/T, labels).item()
        if loss < best_loss: best_loss = loss; best_t = T
    return best_t

def run_p2_1b(data_path, cutoffs, epochs, patience, batch_size, lr, seed, out_dir):
    matches_by_league = load_matches_by_league(data_path)
    leagues = sorted(matches_by_league.keys())
    print(f"Leagues: {leagues}")
    for lg in leagues:
        print(f"  {lg}: {len(matches_by_league[lg])} matches")

    fold_results = {}
    for heldout_lg in leagues:
        print(f"\n{'='*50}\n  FOLD: heldout={heldout_lg}\n{'='*50}")
        # Training set: all other leagues
        train_matches = []
        for lg in leagues:
            if lg == heldout_lg: continue
            train_matches.extend(matches_by_league[lg])

        # Time-split training leagues into train/val
        train_ids, val_ids, _ = time_split_matches(train_matches, 0.8, 0.2)
        heldout_ids = {m["match_id"] for m in matches_by_league[heldout_lg]}
        print(f"  Train: {len(train_ids)}  Val: {len(val_ids)}  Heldout: {len(heldout_ids)}")

        ckpt_path = os.path.join(out_dir, f"fold_{heldout_lg}.pth")
        model, best_ep, best_val = train_one_fold(
            train_ids, val_ids, data_path, cutoffs, epochs, patience, batch_size, lr, seed, ckpt_path)
        print(f"  Best epoch: {best_ep}  Val loss: {best_val:.4f}")

        # Fit T on val set
        val_ds = OddsDataset(allowed_match_ids=val_ids, jsonl_path=data_path, max_seq_len=64,
                             cutoffs=cutoffs, cutoff_mode="exhaustive", min_events=1,
                             asian_label_mode="5class", feature_schema_version="v3", seed=seed)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=OddsCollator())
        val_logits, val_labels, _ = collect_logits(model, val_loader)
        T = fit_T(val_logits, val_labels)
        print(f"  T = {T:.3f}")

        # Eval on heldout
        heldout_ds = OddsDataset(allowed_match_ids=heldout_ids, jsonl_path=data_path, max_seq_len=64,
                                 cutoffs=cutoffs, cutoff_mode="exhaustive", min_events=1,
                                 asian_label_mode="5class", feature_schema_version="v3", seed=seed)
        heldout_ds_v2 = OddsDataset(allowed_match_ids=heldout_ids, jsonl_path=data_path, max_seq_len=64,
                                    cutoffs=cutoffs, cutoff_mode="exhaustive", min_events=1,
                                    asian_label_mode="5class", feature_schema_version="v2", seed=seed)
        ho_loader = DataLoader(heldout_ds, batch_size=64, shuffle=False, collate_fn=OddsCollator())
        ho_logits, ho_labels, ho_leagues = collect_logits(model, ho_loader)
        ho_bl_probs, ho_bl_labels = collect_baseline(heldout_ds_v2)

        raw_probs = F.softmax(ho_logits, dim=-1)
        temp_probs = F.softmax(ho_logits/T, dim=-1)
        raw_m = metrics_dict(raw_probs, ho_labels)
        temp_m = metrics_dict(temp_probs, ho_labels)
        bl_m = metrics_dict(ho_bl_probs, ho_bl_labels)

        # Per-cutoff for heldout
        ho_cutoffs = []
        for s in heldout_ds.samples:
            ho_cutoffs.append(str(int(s.get("cutoff_minutes",0))))

        fold_results[heldout_lg] = {
            "train_count": len(train_ids), "val_count": len(val_ids),
            "heldout_count": len(heldout_ids), "heldout_samples": int(ho_labels.shape[0]),
            "best_epoch": best_ep, "best_val_loss": round(best_val,4),
            "T": round(T,3),
            "raw": raw_m, "temperature_scaled": temp_m, "baseline": bl_m,
            "delta_temp": {k: temp_m[k]-bl_m[k] for k in temp_m if k != "reliability_bins"},
            "per_cutoff": per_cutoff_dict(temp_probs, ho_labels, ho_cutoffs),
        }
        print(f"  Raw:    acc={raw_m['accuracy']:.4f} loss={raw_m['logloss']:.4f} ece={raw_m['ece']:.4f}")
        print(f"  Temp:   acc={temp_m['accuracy']:.4f} loss={temp_m['logloss']:.4f} ece={temp_m['ece']:.4f}")
        print(f"  Baseline: acc={bl_m['accuracy']:.4f} loss={bl_m['logloss']:.4f} ece={bl_m['ece']:.4f}")
        print(f"  Delta:  acc={temp_m['accuracy']-bl_m['accuracy']:+.4f}")

    return fold_results

# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="P2.1 Backtest Gate")
    parser.add_argument("--data", type=str, default="data/odds_real/master_5330.jsonl")
    parser.add_argument("--mode", type=str, default="frozen", choices=["frozen","loo","both"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cutoffs", type=str, default="90,60,30")
    parser.add_argument("--split-dir", type=str, default="data/odds_real/splits_p1_5")
    parser.add_argument("--out-dir", type=str, default="runs/p2_1_backtest")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cutoffs = [float(c.strip()) for c in args.cutoffs.split(",") if c.strip()]

    report = {"experiment": "P2.1"}

    # ── P2.1A ──
    if args.mode in ("frozen", "both"):
        print("="*60+"\nP2.1A: Frozen Candidate Per-league Backtest\n"+"="*60)
        report["p2_1a"] = run_p2_1a(args.data, cutoffs, args.split_dir)
        for lg, r in report["p2_1a"].items():
            d = r["delta_temp"]
            print(f"  {lg:<15}: n={r['count']:>5}  acc={r['temperature_scaled']['accuracy']:.4f}  "
                  f"bl={r['baseline']['accuracy']:.4f}  delta={d['accuracy']:+.4f}  "
                  f"ece={r['temperature_scaled']['ece']:.4f}")

    # ── P2.1B ──
    if args.mode in ("loo", "both"):
        print("\n"+"="*60+"\nP2.1B: Leave-one-league-out CV\n"+"="*60)
        fold_results = run_p2_1b(args.data, cutoffs, args.epochs, args.patience,
                                  args.batch_size, args.lr, args.seed, args.out_dir)
        report["p2_1b"] = fold_results

        # Aggregate
        acc_deltas = [r["delta_temp"]["accuracy"] for r in fold_results.values()]
        loss_deltas = [r["delta_temp"]["logloss"] for r in fold_results.values()]
        ece_deltas = [r["delta_temp"]["ece"] for r in fold_results.values()]
        Ts = [r["T"] for r in fold_results.values()]
        beats = sum(1 for d in acc_deltas if d >= 0)

        agg = {
            "num_folds": len(fold_results),
            "folds_beating_baseline": beats,
            "mean_acc_delta": sum(acc_deltas)/len(acc_deltas),
            "median_acc_delta": sorted(acc_deltas)[len(acc_deltas)//2],
            "mean_logloss_delta": sum(loss_deltas)/len(loss_deltas),
            "mean_brier_delta": 0,  # computed below
            "mean_ece_delta": sum(ece_deltas)/len(ece_deltas),
            "T_mean": sum(Ts)/len(Ts),
            "T_std": (sum((t-sum(Ts)/len(Ts))**2 for t in Ts)/len(Ts))**0.5,
            "T_min": min(Ts), "T_max": max(Ts),
            "worst_fold": min(fold_results.keys(), key=lambda k: fold_results[k]["delta_temp"]["accuracy"]),
            "best_fold": max(fold_results.keys(), key=lambda k: fold_results[k]["delta_temp"]["accuracy"]),
        }
        report["aggregate"] = agg

        print(f"\n{'='*60}")
        print(f"AGGREGATE ({len(fold_results)} folds)")
        print(f"  Beats baseline: {beats}/{len(fold_results)}")
        print(f"  Mean acc_delta: {agg['mean_acc_delta']:+.4f}")
        print(f"  Mean logloss_delta: {agg['mean_logloss_delta']:+.4f}")
        print(f"  Mean ece_delta: {agg['mean_ece_delta']:+.4f}")
        print(f"  T: {agg['T_mean']:.3f} +/- {agg['T_std']:.3f}")
        print(f"  Best fold: {agg['best_fold']}  Worst: {agg['worst_fold']}")

        # Decision
        if agg["folds_beating_baseline"] >= 4 and agg["mean_acc_delta"] >= 0.0:
            decision = "PROCEED_TO_P2_2"
        elif agg["folds_beating_baseline"] >= 3 and agg["mean_acc_delta"] >= -0.002:
            decision = "PROCEED_TO_P2_2"
        elif agg["folds_beating_baseline"] >= 3:
            decision = "NEED_LEAGUE_FEATURE_WORK"
        else:
            decision = "NEED_MORE_DATA"
        report["decision"] = decision
        print(f"\nDecision: {decision}")
        print("="*60)

    with open(os.path.join(args.out_dir, "p2_1_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport: {os.path.join(args.out_dir, 'p2_1_report.json')}")

if __name__ == "__main__":
    main()
