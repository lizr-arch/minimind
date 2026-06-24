"""
P1.21 Robustness & Statistical Confidence Gate — Multi-seed training + bootstrap CI.

Reuses seed=42 from P1.18, trains seeds 123 and 456. Fits temperature on val,
evaluates on test, computes bootstrap confidence intervals.

Usage:
    python eval/eval_p1_21_robustness.py \
        --data data/odds_real/master_5330.jsonl \
        --split-dir data/odds_real/splits_p1_5 \
        --seeds 42,123,456 --epochs 50 --patience 10 --batch-size 64 \
        --out-dir runs/p1_21_robustness
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
from eval.odds_baselines import close_no_vig_euro, open_no_vig_euro, euro_probs_to_tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def setup_seed(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_lr(step, total, lr, warmup=0):
    if warmup > 0 and step < warmup: return lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * p)))

# ── Training ───────────────────────────────────────────────────────────

def train_one_seed(model, train_loader, val_loader, epochs, lr, warmup, patience, ckpt_path):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    best_val = float("inf"); best_ep = 0; wait = 0
    total_steps = epochs * len(train_loader)
    history = {"train_loss": [], "val_loss": [], "epoch": []}
    for ep in range(1, epochs + 1):
        model.train(); tloss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            f = batch["features"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
            el = batch["euro_labels"].to(DEVICE); al = batch["asian_labels"].to(DEVICE)
            mm = batch.get("missing_mask")
            if mm is not None: mm = mm.to(DEVICE)
            cur = (ep-1)*len(train_loader)+step
            for pg in optimizer.param_groups: pg["lr"] = get_lr(cur, total_steps, lr, warmup)
            optimizer.zero_grad()
            out = model(f, attention_mask=am, euro_labels=el, asian_labels=al, missing_mask=mm)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); tloss += out["loss"].item()
        at = tloss/len(train_loader); history["train_loss"].append(at); history["epoch"].append(ep)
        model.eval(); vloss = 0.0; vn = 0
        with torch.no_grad():
            for batch in val_loader:
                f = batch["features"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
                el = batch["euro_labels"].to(DEVICE); al = batch["asian_labels"].to(DEVICE)
                mm = batch.get("missing_mask")
                if mm is not None: mm = mm.to(DEVICE)
                out = model(f, attention_mask=am, euro_labels=el, asian_labels=al, missing_mask=mm)
                n = f.shape[0]; vloss += out["loss"].item()*n; vn += n
        av = vloss/max(1,vn); history["val_loss"].append(av)
        if av < best_val: best_val = av; best_ep = ep; wait = 0; torch.save(model.state_dict(), ckpt_path)
        else: wait += 1
        if wait >= patience: break
    history["best_epoch"] = best_ep; history["best_val_loss"] = best_val
    # Load best
    ckp = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    if isinstance(ckp, dict) and "model_state_dict" in ckp: ckp = ckp["model_state_dict"]
    model.load_state_dict(ckp, strict=False)
    return history

# ── Data collection ────────────────────────────────────────────────────

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
    if not recs: return None,None,None,None
    return (torch.stack([r["probs"] for r in recs]),
            torch.tensor([r["label"] for r in recs]),
            [r["cutoff"] for r in recs],
            [r["league"] for r in recs])

def fit_temperature(logits, labels):
    best_t, best_loss = 1.0, float("inf")
    for T in [round(0.1+i*0.05,3) for i in range(60)]:
        loss = F.cross_entropy(logits/T, labels).item()
        if loss < best_loss: best_loss = loss; best_t = T
    for T in [round(best_t-0.04+i*0.01,3) for i in range(9)]:
        if T <= 0.01: continue
        loss = F.cross_entropy(logits/T, labels).item()
        if loss < best_loss: best_loss = loss; best_t = T
    return best_t

def metrics_dict(probs, labels):
    if probs.shape[0] == 0: return {}
    e = ece_from_probs(probs, labels, 3)
    return {"accuracy": accuracy_from_probs(probs, labels),
            "logloss": logloss_from_probs(probs, labels),
            "brier": brier_from_probs(probs, labels, 3),
            "ece": e["ece"]}

# ── Bootstrap ──────────────────────────────────────────────────────────

def bootstrap_delta(model_logits, model_labels, bl_probs, bl_labels, T, n_boot=1000):
    """Bootstrap CI for model-baseline delta after temperature scaling."""
    N = model_logits.shape[0]
    deltas = {"accuracy": [], "logloss": [], "brier": [], "ece": []}
    for _ in range(n_boot):
        idx = torch.randint(0, N, (N,))
        m_log = model_logits[idx]; m_lbl = model_labels[idx]
        b_pr = bl_probs[idx]; b_lbl = bl_labels[idx]
        m_pr = F.softmax(m_log/T, dim=-1)
        m = metrics_dict(m_pr, m_lbl)
        b = metrics_dict(b_pr, b_lbl)
        for k in deltas:
            deltas[k].append(m[k] - b[k])
    ci = {}
    for k in deltas:
        arr = torch.tensor(deltas[k])
        ci[k] = {"mean": arr.mean().item(), "std": arr.std().item(),
                  "ci_95_low": arr.quantile(0.025).item(),
                  "ci_95_high": arr.quantile(0.975).item()}
    return ci

# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="P1.21 Robustness Gate")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--split-dir", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cutoffs", type=str, default="90,60,30")
    parser.add_argument("--out-dir", type=str, default="runs/p1_21_robustness")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    cutoffs = [float(c.strip()) for c in args.cutoffs.split(",") if c.strip()]

    train_ids = load_match_ids_from_file(os.path.join(args.split_dir,"train_match_ids.txt"))
    val_ids = load_match_ids_from_file(os.path.join(args.split_dir,"val_match_ids.txt"))
    test_ids = load_match_ids_from_file(os.path.join(args.split_dir,"test_match_ids.txt"))
    print(f"Train:{len(train_ids)} Val:{len(val_ids)} Test:{len(test_ids)} Seeds:{seeds}")

    ds_args = dict(jsonl_path=args.data, max_seq_len=64, cutoffs=cutoffs,
                   cutoff_mode="exhaustive", min_events=1, asian_label_mode="5class",
                   feature_schema_version="v3")
    ds_args_v2 = {**ds_args, "feature_schema_version":"v2"}

    # Shared baseline (same for all seeds)
    bl_probs, bl_labels, bl_cutoffs, bl_leagues = collect_baseline(
        OddsDataset(allowed_match_ids=test_ids, **ds_args_v2))
    bl_val_probs, bl_val_labels, _, _ = collect_baseline(
        OddsDataset(allowed_match_ids=val_ids, **ds_args_v2))

    seed_results = {}
    all_test_logits, all_test_labels = [], []

    for seed in seeds:
        print(f"\n{'='*40}\n  SEED {seed}\n{'='*40}")
        setup_seed(seed)
        ckpt_path = os.path.join(args.out_dir, f"model_seed{seed}.pth")

        config = OddsMindConfig(hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
                                asian_num_classes=5, dropout=0.1, pooling_mode="mean",
                                feature_schema_version="v3", transformer_backend="odds_native")

        # Reuse seed=42 from P1.18 if available
        reuse_ckpt = "runs/p1_18_convergence/model_best.pth" if seed == 42 and os.path.exists("runs/p1_18_convergence/model_best.pth") else None

        if reuse_ckpt and not args.skip_training:
            print(f"  Reusing P1.18 seed=42 checkpoint")
            model = OddsMindModel(config).to(DEVICE)
            ckp = torch.load(reuse_ckpt, map_location=DEVICE, weights_only=True)
            if isinstance(ckp, dict) and "model_state_dict" in ckp: ckp = ckp["model_state_dict"]
            model.load_state_dict(ckp, strict=False)
            model.eval()
            torch.save(model.state_dict(), ckpt_path)
        elif not args.skip_training:
            train_ds = OddsDataset(allowed_match_ids=train_ids, seed=seed, **ds_args)
            val_ds = OddsDataset(allowed_match_ids=val_ids, seed=seed, **ds_args)
            model = OddsMindModel(config).to(DEVICE)
            params = sum(p.numel() for p in model.parameters())
            print(f"  Params: {params:,}  Train samples: {len(train_ds)}")
            tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=OddsCollator())
            vl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=OddsCollator())
            warmup = 5 * len(tl)
            hist = train_one_seed(model, tl, vl, args.epochs, args.lr, warmup, args.patience, ckpt_path)
            print(f"  Best epoch: {hist['best_epoch']}  Val loss: {hist['best_val_loss']:.4f}")
        else:
            model = OddsMindModel(config).to(DEVICE)
            ckp = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            if isinstance(ckp, dict) and "model_state_dict" in ckp: ckp = ckp["model_state_dict"]
            model.load_state_dict(ckp, strict=False)
            model.eval()

        # Collect predictions
        val_ds_eval = OddsDataset(allowed_match_ids=val_ids, seed=seed, **ds_args)
        test_ds_eval = OddsDataset(allowed_match_ids=test_ids, seed=seed, **ds_args)
        val_loader = DataLoader(val_ds_eval, batch_size=64, shuffle=False, collate_fn=OddsCollator())
        test_loader = DataLoader(test_ds_eval, batch_size=64, shuffle=False, collate_fn=OddsCollator())

        val_logits, val_labels, _ = collect_logits(model, val_loader)
        test_logits, test_labels, test_leagues = collect_logits(model, test_loader)

        # Fit temperature on val
        T = fit_temperature(val_logits, val_labels)
        test_probs_raw = F.softmax(test_logits, dim=-1)
        test_probs_temp = F.softmax(test_logits/T, dim=-1)

        raw_m = metrics_dict(test_probs_raw, test_labels)
        temp_m = metrics_dict(test_probs_temp, test_labels)
        bl_m = metrics_dict(bl_probs, bl_labels)

        print(f"  Raw:    acc={raw_m['accuracy']:.4f} loss={raw_m['logloss']:.4f} brier={raw_m['brier']:.4f} ece={raw_m['ece']:.4f}")
        print(f"  Temp T={T:.3f}: acc={temp_m['accuracy']:.4f} loss={temp_m['logloss']:.4f} brier={temp_m['brier']:.4f} ece={temp_m['ece']:.4f}")
        print(f"  Baseline:  acc={bl_m['accuracy']:.4f} loss={bl_m['logloss']:.4f}")

        seed_results[str(seed)] = {
            "T": T,
            "raw": raw_m, "temperature": temp_m,
            "delta_raw": {k: raw_m[k]-bl_m[k] for k in raw_m},
            "delta_temp": {k: temp_m[k]-bl_m[k] for k in temp_m},
        }
        all_test_logits.append(test_logits)
        all_test_labels.append(test_labels)

    # ── Aggregate ──
    Ts = [seed_results[str(s)]["T"] for s in seeds]
    acc_deltas = [seed_results[str(s)]["delta_temp"]["accuracy"] for s in seeds]
    loss_deltas = [seed_results[str(s)]["delta_temp"]["logloss"] for s in seeds]
    ece_deltas = [seed_results[str(s)]["delta_temp"]["ece"] for s in seeds]

    print(f"\n{'='*60}")
    print(f"MULTI-SEED SUMMARY ({len(seeds)} seeds)")
    print(f"{'='*60}")
    print(f"Temperature: mean={sum(Ts)/len(Ts):.3f} std={torch.tensor(Ts).std().item():.3f} range=[{min(Ts):.3f}, {max(Ts):.3f}]")
    print(f"Acc delta: mean={sum(acc_deltas)/len(acc_deltas):.4f} std={torch.tensor(acc_deltas).std().item():.4f}")
    print(f"Loss delta: mean={sum(loss_deltas)/len(loss_deltas):.4f} std={torch.tensor(loss_deltas).std().item():.4f}")
    print(f"ECE delta: mean={sum(ece_deltas)/len(ece_deltas):.4f} std={torch.tensor(ece_deltas).std().item():.4f}")

    # ── Bootstrap CI (pool test predictions across seeds) ──
    all_logits = torch.cat(all_test_logits, dim=0)  # [seeds*N, 3]
    all_labels = torch.cat(all_test_labels, dim=0)
    # Repeat baseline to match
    bl_p = bl_probs.repeat(len(seeds), 1)
    bl_l = bl_labels.repeat(len(seeds))
    avg_T = sum(Ts)/len(Ts)

    ci = bootstrap_delta(all_logits, all_labels, bl_p, bl_l, avg_T, n_boot=args.n_bootstrap)
    print(f"\nBootstrap CI ({args.n_bootstrap} samples, pooled across seeds):")
    for k in ["accuracy", "logloss", "brier", "ece"]:
        c = ci[k]
        crosses_zero = c["ci_95_low"] <= 0 <= c["ci_95_high"]
        flag = " X crosses zero" if crosses_zero else ""
        print(f"  {k}_delta: {c['mean']:+.4f} [{c['ci_95_low']:+.4f}, {c['ci_95_high']:+.4f}]{flag}")

    # ── Decision ──
    mean_acc_delta = sum(acc_deltas)/len(acc_deltas)
    acc_ci_low = ci["accuracy"]["ci_95_low"]
    acc_ci_high = ci["accuracy"]["ci_95_high"]
    ece_ci_low = ci["ece"]["ci_95_low"]

    if mean_acc_delta > 0 and acc_ci_low > -0.001:
        decision = "PROCEED_TO_P2_0"
        reason = f"mean acc_delta={mean_acc_delta:+.4f} with 95% CI [{acc_ci_low:+.4f}, {acc_ci_high:+.4f}] — stable advantage"
    elif mean_acc_delta > 0 and ece_ci_low < 0:
        decision = "PROCEED_TO_P2_0"
        reason = f"mean acc_delta={mean_acc_delta:+.4f} with ECE consistently better than baseline"
    elif mean_acc_delta > -0.003:
        decision = "PROCEED_TO_P2_0"
        reason = f"model competitive (acc_delta={mean_acc_delta:+.4f}) with calibration advantage"
    else:
        decision = "NEED_FEATURE_WORK"
        reason = f"model below baseline (acc_delta={mean_acc_delta:+.4f})"

    print(f"\nDecision: {decision}")
    print(f"Reason: {reason}")

    # ── Report ──
    report = {
        "experiment": "P1.21",
        "overall": decision,
        "seeds": seeds,
        "seed_results": seed_results,
        "temperature_stats": {"mean": sum(Ts)/len(Ts), "std": torch.tensor(Ts).std().item(),
                               "min": min(Ts), "max": max(Ts)},
        "delta_stats": {"acc_mean": sum(acc_deltas)/len(acc_deltas),
                         "acc_std": torch.tensor(acc_deltas).std().item(),
                         "loss_mean": sum(loss_deltas)/len(loss_deltas)},
        "bootstrap_ci": ci,
        "decision": decision,
        "reason": reason,
    }
    with open(os.path.join(args.out_dir, "p1_21_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport: {os.path.join(args.out_dir, 'p1_21_report.json')}")


if __name__ == "__main__":
    main()
