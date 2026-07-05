"""
P2.5B Global Base Fine-tuning Probe

For each of 5 leagues: time-split into train/val/test, then:
  A) Evaluate global base (P2.0 candidate) directly — NO training
  B) Fine-tune from global checkpoint (lr=3e-5, early stop)
  C) Train same-league from scratch (lr=1e-3, early stop)
  D) Market baseline (latest_available_no_vig)
  E) KNN retrieval (K=100)

Compares all 5 methods per league. Outputs per-league tables + aggregate decision.

Usage:
    python eval/eval_p2_5b_finetune.py --data data/odds_real/master_5330.jsonl --global-ckpt runs/p2_0_deployment_candidate/model_candidate.pth --epochs 40 --patience 8 --out-dir runs/p2_5b_finetune
"""

import argparse, json, math, os, random, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch, torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from model.model_oddsmind import OddsMindConfig, OddsMindModel
from dataset.odds_dataset import OddsDataset, EURO_MAP
from dataset.odds_collator import OddsCollator
from eval.odds_metrics import (
    accuracy_from_probs, logloss_from_probs, brier_from_probs, ece_from_probs,
)
from eval.odds_baselines import close_no_vig_euro, euro_probs_to_tensor
from infer.knn_retrieval import build_index, retrieve_similar, retrieval_predict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEAGUES = ["EPL", "LaLiga", "Bundesliga", "SerieA", "Ligue1"]

def setup_seed(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_lr(step, total, lr, warmup=0):
    if warmup > 0 and step < warmup: return lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * p)))

def load_matches(path):
    matches = []
    with open(path) as f:
        for line in f:
            if line.strip(): matches.append(json.loads(line.strip()))
    return matches

def time_split(matches, train_r=0.7, val_r=0.15):
    s = sorted(matches, key=lambda m: m.get("kickoff_time",""))
    n = len(s); nt = max(1,int(n*train_r)); nv = max(1,int(n*val_r))
    return s[:nt], s[nt:nt+nv], s[nt+nv:]

def build_dataset_from_matches(matches, cutoffs, seed=42):
    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8')
    for m in matches: tmp.write(json.dumps(m)+'\n')
    tmp.close()
    ds = OddsDataset(jsonl_path=tmp.name, max_seq_len=64, cutoffs=cutoffs,
                     cutoff_mode="exhaustive", min_events=1, asian_label_mode="5class",
                     feature_schema_version="v3", seed=seed)
    os.unlink(tmp.name)
    return ds

def make_model():
    cfg = OddsMindConfig(hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
                         asian_num_classes=5, dropout=0.1, pooling_mode="mean",
                         feature_schema_version="v3", transformer_backend="odds_native")
    return OddsMindModel(cfg).to(DEVICE)

def load_global_ckpt(path):
    model = make_model()
    ckp = torch.load(path, map_location=DEVICE, weights_only=True)
    if isinstance(ckp, dict) and "model_state_dict" in ckp: ckp = ckp["model_state_dict"]
    model.load_state_dict(ckp, strict=False)
    model.eval()
    return model

def collect_probs_and_labels(model, loader, T=1.09):
    model.eval(); logs, lbls = [], []
    with torch.no_grad():
        for batch in loader:
            f = batch["features"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
            el = batch["euro_labels"].to(DEVICE)
            mm = batch.get("missing_mask")
            if mm is not None: mm = mm.to(DEVICE)
            out = model(f, attention_mask=am, missing_mask=mm)
            logs.append(out["euro_logits"].cpu()); lbls.append(el.cpu())
    logits = torch.cat(logs,0); labels = torch.cat(lbls,0)
    return F.softmax(logits/T, dim=-1), labels

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

def metrics(probs, labels):
    if probs.shape[0]==0: return {}
    e = ece_from_probs(probs, labels, 3)
    return {"accuracy": accuracy_from_probs(probs, labels),
            "logloss": logloss_from_probs(probs, labels),
            "brier": brier_from_probs(probs, labels, 3),
            "ece": e["ece"]}

def baseline_on_matches(matches, cutoffs):
    probs, labels = [], []
    for m in matches:
        tl = m.get("odds_timeline",[]); cutoff = m.get("cutoff_minutes",0)
        filtered = [e for e in tl if e.get("minutes_before_kickoff",0) >= cutoff]
        if not filtered: continue
        try: p = euro_probs_to_tensor(close_no_vig_euro(filtered))
        except: p = torch.full((3,),1.0/3.0)
        probs.append(p); labels.append(EURO_MAP[m["label"]["euro_result"]])
    if not probs: return None, None
    return torch.stack(probs), torch.tensor(labels)

def knn_on_matches(test_matches, train_matches, K=100):
    index = build_index(train_matches, cutoff_minutes=0)
    probs, labels = [], []
    for m in test_matches:
        retrieved = retrieve_similar(m, index, K=K)
        if not retrieved: continue
        probs.append(retrieval_predict(retrieved))
        labels.append(EURO_MAP[m["label"]["euro_result"]])
    if not probs: return None, None
    return torch.stack(probs), torch.tensor(labels)

def train_model(model, train_matches, val_matches, cutoffs, epochs, patience, batch_size, lr, seed, ckpt_path):
    setup_seed(seed)
    train_ds = build_dataset_from_matches(train_matches, cutoffs, seed=seed)
    val_ds = build_dataset_from_matches(val_matches, cutoffs, seed=seed)
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=OddsCollator())
    vl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=OddsCollator())
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    best_val = float("inf"); best_ep = 0; wait = 0; warmup = 5*len(tl); total_steps = epochs*len(tl)

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

# ── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="P2.5B Fine-tuning Probe")
    parser.add_argument("--data", type=str, default="data/odds_real/master_5330.jsonl")
    parser.add_argument("--global-ckpt", type=str, default="runs/p2_0_deployment_candidate/model_candidate.pth")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ft-lr", type=float, default=3e-5)
    parser.add_argument("--scratch-lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cutoffs", type=str, default="90,60,30")
    parser.add_argument("--out-dir", type=str, default="runs/p2_5b_finetune")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cutoffs = [float(c.strip()) for c in args.cutoffs.split(",") if c.strip()]
    all_matches = load_matches(args.data)
    global_model = load_global_ckpt(args.global_ckpt)
    global_T = 1.09  # from P2.0 calibration

    results = {}
    for lg in LEAGUES:
        print(f"\n{'='*50}\n  LEAGUE: {lg}\n{'='*50}")
        lg_matches = [m for m in all_matches if m.get("league_id")==lg]
        if len(lg_matches) < 50:
            print(f"  INSUFFICIENT_DATA: {len(lg_matches)} matches")
            results[lg] = {"error": "INSUFFICIENT_DATA", "count": len(lg_matches)}
            continue

        train_m, val_m, test_m = time_split(lg_matches, 0.7, 0.15)
        print(f"  Train: {len(train_m)}  Val: {len(val_m)}  Test: {len(test_m)}")

        # ── A) Global base ──
        test_ds = build_dataset_from_matches(test_m, cutoffs, seed=args.seed)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=OddsCollator())
        gb_probs, gb_labels = collect_probs_and_labels(global_model, test_loader, T=global_T)
        gb_m = metrics(gb_probs, gb_labels)
        print(f"  Global base: acc={gb_m['accuracy']:.4f} loss={gb_m['logloss']:.4f} ece={gb_m['ece']:.4f}")

        # ── B) Fine-tune ──
        ft_ckpt = os.path.join(args.out_dir, f"ft_{lg}.pth")
        if not args.skip_training:
            print(f"  Fine-tuning (lr={args.ft_lr})...")
            ft_model = load_global_ckpt(args.global_ckpt)  # fresh copy
            ft_model, ft_ep, ft_val = train_model(ft_model, train_m, val_m, cutoffs,
                                                   args.epochs, args.patience, args.batch_size,
                                                   args.ft_lr, args.seed, ft_ckpt)
            print(f"    best epoch={ft_ep} val_loss={ft_val:.4f}")
        else:
            ft_model = load_global_ckpt(ft_ckpt)

        # Fit T on val
        val_ds = build_dataset_from_matches(val_m, cutoffs, seed=args.seed)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=OddsCollator())
        ft_val_logits, ft_val_labels = [], []
        ft_model.eval()
        with torch.no_grad():
            for batch in val_loader:
                f = batch["features"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
                el = batch["euro_labels"].to(DEVICE)
                mm = batch.get("missing_mask")
                if mm is not None: mm = mm.to(DEVICE)
                out = ft_model(f, attention_mask=am, missing_mask=mm)
                ft_val_logits.append(out["euro_logits"].cpu()); ft_val_labels.append(el.cpu())
        ft_T = fit_T(torch.cat(ft_val_logits,0), torch.cat(ft_val_labels,0))
        ft_probs, ft_labels = collect_probs_and_labels(ft_model, test_loader, T=ft_T)
        ft_m = metrics(ft_probs, ft_labels)
        print(f"  Fine-tuned (T={ft_T:.3f}): acc={ft_m['accuracy']:.4f} loss={ft_m['logloss']:.4f} ece={ft_m['ece']:.4f}")

        # ── C) Scratch ──
        sc_ckpt = os.path.join(args.out_dir, f"scratch_{lg}.pth")
        if not args.skip_training:
            print(f"  Training from scratch (lr={args.scratch_lr})...")
            sc_model = make_model()
            sc_model, sc_ep, sc_val = train_model(sc_model, train_m, val_m, cutoffs,
                                                    args.epochs, args.patience, args.batch_size,
                                                    args.scratch_lr, args.seed, sc_ckpt)
            print(f"    best epoch={sc_ep} val_loss={sc_val:.4f}")
        else:
            sc_model = make_model()
            ckp = torch.load(sc_ckpt, map_location=DEVICE, weights_only=True)
            if isinstance(ckp, dict) and "model_state_dict" in ckp: ckp = ckp["model_state_dict"]
            sc_model.load_state_dict(ckp, strict=False)

        # Fit T on val for scratch
        sc_val_logits, sc_val_labels = [], []
        sc_model.eval()
        with torch.no_grad():
            for batch in val_loader:
                f = batch["features"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
                el = batch["euro_labels"].to(DEVICE)
                mm = batch.get("missing_mask")
                if mm is not None: mm = mm.to(DEVICE)
                out = sc_model(f, attention_mask=am, missing_mask=mm)
                sc_val_logits.append(out["euro_logits"].cpu()); sc_val_labels.append(el.cpu())
        sc_T = fit_T(torch.cat(sc_val_logits,0), torch.cat(sc_val_labels,0))
        sc_probs, sc_labels = collect_probs_and_labels(sc_model, test_loader, T=sc_T)
        sc_m = metrics(sc_probs, sc_labels)
        print(f"  Scratch (T={sc_T:.3f}): acc={sc_m['accuracy']:.4f} loss={sc_m['logloss']:.4f} ece={sc_m['ece']:.4f}")

        # ── D) Baseline ──
        bl_probs, bl_labels = baseline_on_matches(test_m, cutoffs)
        bl_m = metrics(bl_probs, bl_labels)
        print(f"  Baseline: acc={bl_m['accuracy']:.4f} loss={bl_m['logloss']:.4f}")

        # ── E) KNN ──
        knn_probs, knn_labels = knn_on_matches(test_m, train_m, K=100)
        knn_m = metrics(knn_probs, knn_labels) if knn_probs is not None else {}
        print(f"  KNN: acc={knn_m.get('accuracy',0):.4f} loss={knn_m.get('logloss',0):.4f}")

        results[lg] = {
            "train_count": len(train_m), "val_count": len(val_m), "test_count": len(test_m),
            "global_base": {"T": global_T, **gb_m,
                            "delta": {k: round(gb_m[k]-bl_m[k],4) for k in gb_m}},
            "fine_tuned": {"T": round(ft_T,3), **ft_m,
                            "delta": {k: round(ft_m[k]-bl_m[k],4) for k in ft_m}},
            "scratch": {"T": round(sc_T,3), **sc_m,
                         "delta": {k: round(sc_m[k]-bl_m[k],4) for k in sc_m}},
            "baseline": bl_m,
            "knn": knn_m,
        }

    # ── Summary table ──
    print("\n"+"="*80)
    print(f"{'League':<12} {'Method':<15} {'Acc':>8} {'Loss':>8} {'Brier':>8} {'ECE':>8} {'ΔAcc':>8}")
    print("-"*67)
    for lg in LEAGUES:
        r = results.get(lg, {})
        if "error" in r: continue
        for method, key in [("Global base","global_base"),("Fine-tuned","fine_tuned"),
                             ("Scratch","scratch"),("Baseline","baseline")]:
            m = r.get(key,{})
            if not m: continue
            d = m.get("delta",{}).get("accuracy",0)
            print(f"{lg:<12} {method:<15} {m['accuracy']:>8.4f} {m['logloss']:>8.4f} {m['brier']:>8.4f} {m['ece']:>8.4f} {d:>+8.4f}")

    # ── Decision per league ──
    print(f"\n{'League':<12} {'Best Neural':<15} {'Beats BL?':>10} {'FT>Global?':>12} {'FT>Scratch?':>12} {'Decision':<25}")
    print("-"*86)
    decisions = {}
    for lg in LEAGUES:
        r = results.get(lg, {})
        if "error" in r: continue
        gb = r["global_base"]; ft = r["fine_tuned"]; sc = r["scratch"]; bl = r["baseline"]
        best = min([("global_base",gb),("fine_tuned",ft),("scratch",sc)], key=lambda x: x[1]["logloss"])
        beats_bl = "YES" if best[1]["accuracy"] > bl["accuracy"] else "NO"
        ft_better_global = "YES" if ft["logloss"] <= gb["logloss"] else "NO"
        ft_better_scratch = "YES" if ft["logloss"] <= sc["logloss"] else "NO"

        if ft_better_global == "YES" and ft_better_scratch == "YES":
            dec = "FINE_TUNE_PROMISING"
        elif sc["logloss"] <= ft["logloss"] and sc["logloss"] <= gb["logloss"]:
            dec = "SCRATCH_BETTER"
        elif gb["logloss"] <= ft["logloss"] and gb["logloss"] <= sc["logloss"]:
            dec = "GLOBAL_BASE_ENOUGH"
        else:
            dec = "BASELINE_STRONGER"
        decisions[lg] = dec
        print(f"{lg:<12} {best[0]:<15} {beats_bl:>10} {ft_better_global:>12} {ft_better_scratch:>12} {dec:<25}")

    # ── Overall decision ──
    ft_promising = sum(1 for d in decisions.values() if d == "FINE_TUNE_PROMISING")
    if ft_promising >= 3:
        overall = "FINE_TUNE_PROMISING_WITH_WARNINGS"
    elif ft_promising >= 1:
        overall = "SCRATCH_MODELS_PREFERRED"
    else:
        overall = "GLOBAL_BASE_ONLY"
    print(f"\nOverall: {overall} ({ft_promising}/5 leagues fine-tune promising)")

    # Save report
    report = {"experiment": "P2.5B", "overall": overall, "per_league": results, "decisions": decisions}
    with open(os.path.join(args.out_dir, "p2_5b_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report: {os.path.join(args.out_dir, 'p2_5b_report.json')}")


if __name__ == "__main__":
    main()
