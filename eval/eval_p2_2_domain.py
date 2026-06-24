"""
P2.2 Domain Transfer Experiment — train on domain X, test on domain Y,
with market residual model and similar-match retrieval baseline.

Usage:
    # Full transfer matrix (subset for speed)
    python eval/eval_p2_2_domain.py --data data/odds_real/master_5330.jsonl \
        --mode transfer --out-dir runs/p2_2_domain
"""

import argparse, json, math, os, random, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch, torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from model.model_oddsmind import OddsMindConfig, OddsMindModel
from dataset.odds_dataset import OddsDataset, EURO_MAP
from dataset.odds_collator import OddsCollator
from eval.odds_metrics import (
    accuracy_from_probs, logloss_from_probs, brier_from_probs, ece_from_probs,
)
from eval.odds_baselines import close_no_vig_euro, euro_probs_to_tensor
from eval.domain_transfer.domain_taxonomy import get_domain, get_domain_family
from eval.domain_transfer.market_quality import compute_market_quality_features

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-9

def setup_seed(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

def get_lr(step, total, lr, warmup=0):
    if warmup > 0 and step < warmup: return lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * p)))

# ══════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════

def load_matches(data_path):
    matches = []
    with open(data_path) as f:
        for line in f:
            if line.strip(): matches.append(json.loads(line.strip()))
    return matches

def filter_by_domain(matches, domain_key, domain_value):
    """Filter matches by domain taxonomy field. domain_key='league_id' for league."""
    return [m for m in matches if m.get(domain_key, "") == domain_value]

def build_dataset(matches, cutoffs, max_seq_len=64, seed=42):
    """Build OddsDataset from a list of match dicts."""
    # Write to temp JSONL
    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
    for m in matches:
        tmp.write(json.dumps(m) + '\n')
    tmp.close()
    ds = OddsDataset(jsonl_path=tmp.name, max_seq_len=max_seq_len, cutoffs=cutoffs,
                     cutoff_mode="exhaustive", min_events=1, asian_label_mode="5class",
                     feature_schema_version="v3", seed=seed)
    os.unlink(tmp.name)
    return ds

# ══════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════

def train_model(matches_train, matches_val, cutoffs, epochs, patience, batch_size, lr, seed,
                ckpt_path, model_type="direct", lambda_delta=0.01):
    setup_seed(seed)
    train_ds = build_dataset(matches_train, cutoffs, seed=seed)
    val_ds = build_dataset(matches_val, cutoffs, seed=seed)
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
        model.train(); tloss = 0.0
        for batch in tl:
            f = batch["features"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
            el = batch["euro_labels"].to(DEVICE); al = batch["asian_labels"].to(DEVICE)
            mm = batch.get("missing_mask")
            if mm is not None: mm = mm.to(DEVICE)
            step_global = (ep-1)*len(tl) + (tloss/tloss if tloss else 0)  # approximate
            for pg in optimizer.param_groups: pg["lr"] = get_lr(ep*len(tl), total_steps, lr, warmup)
            optimizer.zero_grad()

            out = model(f, attention_mask=am, euro_labels=el, asian_labels=al, missing_mask=mm)

            if model_type == "market_residual":
                # Get baseline probs from batch
                bl_probs = _batch_baseline_probs(batch)
                bl_logits = torch.log(bl_probs.to(DEVICE) + EPS)
                euro_logits = out["euro_logits"]
                delta_logits = euro_logits - bl_logits
                final_logits = bl_logits + delta_logits
                loss = F.cross_entropy(final_logits, el) + lambda_delta * delta_logits.pow(2).mean()
            else:
                loss = out["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tloss += loss.item()

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

def _batch_baseline_probs(batch):
    """Extract baseline probabilities for a batch using timeline data."""
    # We need the raw timeline — access via dataset samples
    probs_list = []
    for i, mid in enumerate(batch.get("match_ids", [])):
        # Use the cutoff-filtered events from features to estimate baseline
        feats = batch["features"][i]
        am = batch["attention_mask"][i]
        valid_feats = feats[am]  # [T, 13]
        if valid_feats.shape[0] == 0:
            probs_list.append(torch.full((3,), 1.0/3.0))
            continue
        # Extract closing euro odds from features (indices 1,2,3)
        euro = valid_feats[-1, 1:4]
        if (euro > 1.0).all():
            rh, rd, ra = 1.0/euro[0], 1.0/euro[1], 1.0/euro[2]
            total = rh + rd + ra
            probs_list.append(torch.tensor([rh/total, rd/total, ra/total]))
        else:
            probs_list.append(torch.full((3,), 1.0/3.0))
    return torch.stack(probs_list)

# ══════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════

def metrics_dict(probs, labels):
    e = ece_from_probs(probs, labels, 3)
    return {"accuracy": accuracy_from_probs(probs, labels),
            "logloss": logloss_from_probs(probs, labels),
            "brier": brier_from_probs(probs, labels, 3),
            "ece": e["ece"]}

def evaluate_model_on_matches(model, matches, cutoffs, T=1.0):
    ds = build_dataset(matches, cutoffs)
    if len(ds) == 0: return None
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=OddsCollator())
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            f = batch["features"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
            el = batch["euro_labels"].to(DEVICE)
            mm = batch.get("missing_mask")
            if mm is not None: mm = mm.to(DEVICE)
            out = model(f, attention_mask=am, missing_mask=mm)
            all_logits.append(out["euro_logits"].cpu()); all_labels.append(el.cpu())
    logits = torch.cat(all_logits, 0); labels = torch.cat(all_labels, 0)
    probs = F.softmax(logits/T, dim=-1)
    return metrics_dict(probs, labels)

def baseline_on_matches(matches, cutoffs):
    """Compute baseline metrics on a list of matches."""
    records = []
    for m in matches:
        tl = m.get("odds_timeline", []); cutoff = m.get("cutoff_minutes", 0)
        filtered = [e for e in tl if e.get("minutes_before_kickoff", 0) >= cutoff]
        if not filtered: continue
        try: probs = euro_probs_to_tensor(close_no_vig_euro(filtered))
        except: probs = torch.full((3,), 1.0/3.0)
        label = EURO_MAP[m["label"]["euro_result"]]
        records.append({"probs": probs, "label": label})
    if not records: return None
    probs = torch.stack([r["probs"] for r in records])
    labels = torch.tensor([r["label"] for r in records])
    return metrics_dict(probs, labels)

# ══════════════════════════════════════════════════════════════════════
# Similar-match retrieval
# ══════════════════════════════════════════════════════════════════════

def build_retrieval_index(train_matches, cutoff_minutes):
    """Build retrieval index from training matches at specified cutoff."""
    index = []
    for m in train_matches:
        tl = m.get("odds_timeline", [])
        filtered = [e for e in tl if e.get("minutes_before_kickoff", 0) >= cutoff_minutes]
        if not filtered: continue
        try: bl = euro_probs_to_tensor(close_no_vig_euro(filtered))
        except: bl = torch.full((3,), 1.0/3.0)
        mq = compute_market_quality_features(tl, cutoff_minutes)
        domain = get_domain(m.get("league_id", ""))
        label = EURO_MAP[m["label"]["euro_result"]]
        index.append({
            "match_id": m["match_id"],
            "baseline_probs": bl,
            "market_quality": mq,
            "domain_family": domain["domain_family"],
            "league_id": m.get("league_id", ""),
            "label": label,
            "cutoff_minutes": cutoff_minutes,
        })
    return index

def retrieve_similar(target_match, index, K=50):
    """Retrieve K most similar matches from index."""
    tl = target_match.get("odds_timeline", [])
    cutoff = target_match.get("cutoff_minutes", 0)
    try: t_bl = euro_probs_to_tensor(close_no_vig_euro(
        [e for e in tl if e.get("minutes_before_kickoff", 0) >= cutoff]))
    except: t_bl = torch.full((3,), 1.0/3.0)
    t_mq = compute_market_quality_features(tl, cutoff)

    scored = []
    for entry in index:
        bl_dist = (t_bl - entry["baseline_probs"]).pow(2).sum().item()
        # Domain match bonus
        t_domain = get_domain(target_match.get("league_id", ""))["domain_family"]
        domain_match = 0.0 if t_domain == entry["domain_family"] else 0.1
        score = bl_dist + domain_match
        scored.append((score, entry))

    scored.sort(key=lambda x: x[0])
    return [e for _, e in scored[:K]]

def retrieval_predict(retrieved, label):
    """Predict from retrieved matches: majority vote distribution."""
    counts = {0: 0, 1: 0, 2: 0}
    for r in retrieved:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    total = max(1, sum(counts.values()))
    probs = torch.tensor([counts[0]/total, counts[1]/total, counts[2]/total])
    return probs

# ══════════════════════════════════════════════════════════════════════
# Main experiment
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="P2.2 Domain Transfer")
    parser.add_argument("--data", type=str, default="data/odds_real/master_5330.jsonl")
    parser.add_argument("--mode", type=str, default="transfer", choices=["transfer","retrieval","full"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cutoffs", type=str, default="90,60,30")
    parser.add_argument("--out-dir", type=str, default="runs/p2_2_domain")
    parser.add_argument("--retrieval-k", type=int, default=100)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cutoffs = [float(c.strip()) for c in args.cutoffs.split(",") if c.strip()]

    all_matches = load_matches(args.data)
    leagues = sorted(set(m.get("league_id","") for m in all_matches))
    print(f"Leagues: {leagues}")
    for lg in leagues:
        print(f"  {lg}: {sum(1 for m in all_matches if m.get('league_id')==lg)} matches")

    report = {"experiment": "P2.2", "leagues": leagues}

    # ══════════════════════════════════════════════════════
    # Domain transfer matrix (select key pairs to keep runtime manageable)
    # ══════════════════════════════════════════════════════
    if args.mode in ("transfer", "full"):
        print("\n" + "="*60 + "\nDOMAIN TRANSFER MATRIX\n" + "="*60)
        transfer_pairs = [
            ("EPL", "LaLiga"), ("LaLiga", "EPL"),
            ("Bundesliga", "SerieA"), ("SerieA", "Bundesliga"),
            ("EPL", "SerieA"), ("LaLiga", "Ligue1"),
        ]
        matrix = {}

        for train_lg, test_lg in transfer_pairs:
            print(f"\n--- train={train_lg} -> test={test_lg} ---")
            train_m = [m for m in all_matches if m.get("league_id") == train_lg]
            test_m = [m for m in all_matches if m.get("league_id") == test_lg]

            # Time-split train into train/val
            train_sorted = sorted(train_m, key=lambda m: m.get("kickoff_time",""))
            n = len(train_sorted); n_train = max(1, int(n*0.8))
            train_set = train_sorted[:n_train]; val_set = train_sorted[n_train:]

            print(f"  Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_m)}")

            ckpt = os.path.join(args.out_dir, f"transfer_{train_lg}_to_{test_lg}.pth")
            model, best_ep, best_val = train_model(
                train_set, val_set, cutoffs, args.epochs, args.patience,
                args.batch_size, args.lr, args.seed, ckpt)

            model_m = evaluate_model_on_matches(model, test_m, cutoffs, T=1.09)
            bl_m = baseline_on_matches(test_m, cutoffs)

            key = f"{train_lg}->{test_lg}"
            matrix[key] = {
                "train_count": len(train_set), "test_count": len(test_m),
                "best_epoch": best_ep, "best_val_loss": round(best_val, 4),
                "model": model_m, "baseline": bl_m,
                "delta": {k: round(model_m[k]-bl_m[k],4) for k in model_m} if model_m and bl_m else {},
            }
            if model_m and bl_m:
                print(f"  Model: acc={model_m['accuracy']:.4f}  BL: acc={bl_m['accuracy']:.4f}  "
                      f"Delta: {matrix[key]['delta']['accuracy']:+.4f}")

        report["transfer_matrix"] = matrix

    # ══════════════════════════════════════════════════════
    # Retrieval baseline
    # ══════════════════════════════════════════════════════
    if args.mode in ("retrieval", "full"):
        print("\n" + "="*60 + "\nSIMILAR-MATCH RETRIEVAL\n" + "="*60)
        retrieval_results = {}
        for test_lg in leagues:
            train_m = [m for m in all_matches if m.get("league_id") != test_lg]
            test_m = [m for m in all_matches if m.get("league_id") == test_lg]
            if not test_m: continue

            index = build_retrieval_index(train_m, 30)  # cutoff=30
            all_probs, all_labels = [], []
            for m in test_m:
                m["cutoff_minutes"] = 30  # standardize
                retrieved = retrieve_similar(m, index, K=args.retrieval_k)
                probs = retrieval_predict(retrieved, EURO_MAP[m["label"]["euro_result"]])
                all_probs.append(probs)
                all_labels.append(EURO_MAP[m["label"]["euro_result"]])

            probs = torch.stack(all_probs); labels = torch.tensor(all_labels)
            ret_m = metrics_dict(probs, labels)
            bl_m = baseline_on_matches(test_m, [30.0])
            retrieval_results[test_lg] = {
                "count": len(test_m),
                "retrieval": ret_m, "baseline": bl_m,
                "delta": {k: round(ret_m[k]-bl_m[k],4) for k in ret_m} if bl_m else {},
            }
            print(f"  Retrieval on {test_lg} (K={args.retrieval_k}, n={len(test_m)}): "
                  f"acc={ret_m['accuracy']:.4f} bl={bl_m['accuracy']:.4f} "
                  f"delta={retrieval_results[test_lg]['delta']['accuracy']:+.4f}")

        report["retrieval"] = retrieval_results

    # ══════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════
    if matrix := report.get("transfer_matrix", {}):
        deltas = [v["delta"].get("accuracy", -999) for v in matrix.values() if v.get("delta")]
        if deltas:
            beats = sum(1 for d in deltas if d >= 0)
            report["transfer_summary"] = {
                "pairs_tested": len(matrix),
                "beats_baseline": beats,
                "mean_delta": round(sum(deltas)/len(deltas), 4),
                "best_pair": max(matrix.keys(), key=lambda k: matrix[k].get("delta",{}).get("accuracy",-999)),
                "worst_pair": min(matrix.keys(), key=lambda k: matrix[k].get("delta",{}).get("accuracy",-999)),
            }
            print(f"\nTransfer summary: {beats}/{len(deltas)} beats baseline, mean delta={sum(deltas)/len(deltas):+.4f}")

    with open(os.path.join(args.out_dir, "p2_2_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport: {os.path.join(args.out_dir, 'p2_2_report.json')}")

if __name__ == "__main__":
    main()
