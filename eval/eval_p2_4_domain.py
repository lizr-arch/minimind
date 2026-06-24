"""
P2.4 Domain-Isolated Training & Transfer Error Matrix

Trains 8 single-league transfer pairs, evaluates direct + market_residual variants,
compares against retrieval baseline and P2.1 results. Outputs full transfer matrix
with negative transfer detection.

Usage:
    python eval/eval_p2_4_domain.py --mode transfer --epochs 30 --patience 8 --batch-size 64 --out-dir runs/p2_4_domain
    python eval/eval_p2_4_domain.py --mode full --epochs 30 --patience 8 --batch-size 64 --out-dir runs/p2_4_domain
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
    accuracy_from_logits, logloss_from_logits, brier_from_logits, ece_from_logits,
)
from eval.odds_baselines import close_no_vig_euro, euro_probs_to_tensor
from infer.knn_retrieval import build_index, retrieve_similar, retrieval_predict, compute_explanation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-9

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

def build_dataset(matches, cutoffs, seed=42):
    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8')
    for m in matches: tmp.write(json.dumps(m) + '\n')
    tmp.close()
    ds = OddsDataset(jsonl_path=tmp.name, max_seq_len=64, cutoffs=cutoffs,
                     cutoff_mode="exhaustive", min_events=1, asian_label_mode="5class",
                     feature_schema_version="v3", seed=seed)
    os.unlink(tmp.name)
    return ds

# ── Training ───────────────────────────────────────────────────────────

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

    for ep in range(1, epochs + 1):
        model.train()
        for step, batch in enumerate(tl, start=1):
            f = batch["features"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
            el = batch["euro_labels"].to(DEVICE); al = batch["asian_labels"].to(DEVICE)
            mm = batch.get("missing_mask")
            if mm is not None: mm = mm.to(DEVICE)
            cur = (ep - 1) * len(tl) + step
            for pg in optimizer.param_groups: pg["lr"] = get_lr(cur, total_steps, lr, warmup)
            optimizer.zero_grad()
            out = model(f, attention_mask=am, euro_labels=el, asian_labels=al, missing_mask=mm)

            if model_type == "market_residual":
                bl_probs = _batch_baseline_probs(batch).to(DEVICE)
                bl_logits = torch.log(bl_probs.clamp(min=EPS))
                euro_logits = out["euro_logits"]
                delta_logits = euro_logits - bl_logits
                final_logits = bl_logits + delta_logits
                loss = F.cross_entropy(final_logits, el) + lambda_delta * delta_logits.pow(2).mean()
            else:
                loss = out["loss"]

            loss.backward()
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
                n = f.shape[0]; vloss += out["loss"].item() * n; vn += n
        av = vloss / max(1, vn)
        if av < best_val: best_val = av; best_ep = ep; wait = 0; torch.save(model.state_dict(), ckpt_path)
        else: wait += 1
        if wait >= patience:
            print(f"    early stop at epoch {ep}")
            break

    ckp = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    if isinstance(ckp, dict) and "model_state_dict" in ckp: ckp = ckp["model_state_dict"]
    model.load_state_dict(ckp, strict=False)
    return model, best_ep, round(best_val, 4)


def _batch_baseline_probs(batch):
    probs_list = []
    for i in range(batch["features"].shape[0]):
        feats = batch["features"][i]
        am = batch["attention_mask"][i]
        valid = feats[am]
        if valid.shape[0] == 0:
            probs_list.append(torch.full((3,), 1.0/3.0))
            continue
        euro = valid[-1, 1:4]
        if (euro > 1.0).all():
            rh, rd, ra = 1.0/euro[0], 1.0/euro[1], 1.0/euro[2]
            total = rh + rd + ra
            probs_list.append(torch.tensor([rh/total, rd/total, ra/total]))
        else:
            probs_list.append(torch.full((3,), 1.0/3.0))
    return torch.stack(probs_list)


# ── Evaluation ─────────────────────────────────────────────────────────

def fit_T(logits, labels):
    best_t, best_loss = 1.0, float("inf")
    for T in [round(0.1 + i*0.05, 3) for i in range(60)]:
        loss = F.cross_entropy(logits/T, labels).item()
        if loss < best_loss: best_loss = loss; best_t = T
    for T in [round(best_t - 0.04 + i*0.01, 3) for i in range(9)]:
        if T <= 0.01: continue
        loss = F.cross_entropy(logits/T, labels).item()
        if loss < best_loss: best_loss = loss; best_t = T
    return best_t


def evaluate_model_on_matches(model, matches, cutoffs, T=1.09, model_type="direct"):
    """Evaluate model on a list of match dicts. Returns metrics dict."""
    if not matches: return None
    test_ds = build_dataset(matches, cutoffs)
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=OddsCollator())
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            f = batch["features"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
            el = batch["euro_labels"].to(DEVICE)
            mm = batch.get("missing_mask")
            if mm is not None: mm = mm.to(DEVICE)
            out = model(f, attention_mask=am, missing_mask=mm)

            if model_type == "market_residual":
                bl_probs = _batch_baseline_probs(batch).to(DEVICE)
                bl_logits = torch.log(bl_probs.clamp(min=EPS))
                euro_logits = out["euro_logits"]
                final_logits = bl_logits + euro_logits - bl_logits  # = euro_logits (same for eval)
                logits = final_logits
            else:
                logits = out["euro_logits"]

            all_logits.append(logits.cpu()); all_labels.append(el.cpu())

    logits = torch.cat(all_logits, 0); labels = torch.cat(all_labels, 0)
    probs_raw = F.softmax(logits, dim=-1)
    probs_temp = F.softmax(logits / T, dim=-1)
    ece = ece_from_probs(probs_temp, labels, 3)
    return {
        "num_samples": int(logits.shape[0]),
        "accuracy": accuracy_from_probs(probs_temp, labels),
        "logloss": logloss_from_probs(probs_temp, labels),
        "brier": brier_from_probs(probs_temp, labels, 3),
        "ece": ece["ece"],
    }


def baseline_on_matches(matches, cutoffs):
    """Compute latest_available_no_vig baseline on match list."""
    all_probs, all_labels = [], []
    for m in matches:
        tl = m.get("odds_timeline", [])
        cutoff = m.get("cutoff_minutes", 0)
        filtered_cutoff = [e for e in tl if e.get("minutes_before_kickoff", 0) >= cutoff]
        if not filtered_cutoff: continue
        try:
            probs = euro_probs_to_tensor(close_no_vig_euro(filtered_cutoff))
        except:
            probs = torch.full((3,), 1.0/3.0)
        all_probs.append(probs)
        all_labels.append(EURO_MAP[m["label"]["euro_result"]])
    if not all_probs: return None
    probs = torch.stack(all_probs); labels = torch.tensor(all_labels)
    ece = ece_from_probs(probs, labels, 3)
    return {
        "num_samples": int(labels.shape[0]),
        "accuracy": accuracy_from_probs(probs, labels),
        "logloss": logloss_from_probs(probs, labels),
        "brier": brier_from_probs(probs, labels, 3),
        "ece": ece["ece"],
    }


def retrieval_on_matches(test_matches, train_matches, cutoffs, K=100):
    """KNN retrieval baseline. train_matches is the index; test_matches is evaluated."""
    if not test_matches: return None
    index = build_index(train_matches, cutoff_minutes=0)
    all_probs, all_labels = [], []
    for m in test_matches:
        m_with_cutoff = dict(m)
        retrieved = retrieve_similar(m_with_cutoff, index, K=K)
        if not retrieved: continue
        probs = retrieval_predict(retrieved, None)
        all_probs.append(probs)
        all_labels.append(EURO_MAP[m["label"]["euro_result"]])
    if not all_probs: return None
    probs = torch.stack(all_probs); labels = torch.tensor(all_labels)
    ece = ece_from_probs(probs, labels, 3)
    return {
        "num_samples": int(labels.shape[0]),
        "accuracy": accuracy_from_probs(probs, labels),
        "logloss": logloss_from_probs(probs, labels),
        "brier": brier_from_probs(probs, labels, 3),
        "ece": ece["ece"],
    }


# ── Matrix building ────────────────────────────────────────────────────

def build_transfer_matrix(all_matches, cutoffs, args):
    leagues = sorted(set(m.get("league_id", "") for m in all_matches))
    # Key transfer pairs — single-league train → single-league test
    pairs = [
        ("EPL", "LaLiga"), ("LaLiga", "EPL"),
        ("Bundesliga", "SerieA"), ("SerieA", "Bundesliga"),
        ("EPL", "SerieA"), ("LaLiga", "Ligue1"),
        ("EPL", "Bundesliga"), ("SerieA", "Ligue1"),
    ]

    matrix = {}  # key = "train→test": {direct: {...}, residual: {...}, baseline: {...}, retrieval: {...}}
    for train_lg, test_lg in pairs:
        key = f"{train_lg}->{test_lg}"
        print(f"\n{'='*50}\n  {key}\n{'='*50}")

        train_m = [m for m in all_matches if m.get("league_id") == train_lg]
        test_m = [m for m in all_matches if m.get("league_id") == test_lg]
        train_sorted = sorted(train_m, key=lambda m: m.get("kickoff_time", ""))
        n = len(train_sorted); n_train = max(1, int(n * 0.8))
        train_set = train_sorted[:n_train]; val_set = train_sorted[n_train:]
        print(f"  Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_m)}")

        # Direct
        ckpt = os.path.join(args.out_dir, f"direct_{train_lg}_to_{test_lg}.pth")
        print(f"  [direct] training...")
        model_d, ep_d, val_d = train_model(train_set, val_set, cutoffs, args.epochs, args.patience,
                                            args.batch_size, args.lr, args.seed, ckpt, "direct")
        model_m = evaluate_model_on_matches(model_d, test_m, cutoffs)

        # Market residual
        ckpt_r = os.path.join(args.out_dir, f"residual_{train_lg}_to_{test_lg}.pth")
        print(f"  [residual] training...")
        try:
            model_r, ep_r, val_r = train_model(train_set, val_set, cutoffs, args.epochs, args.patience,
                                                args.batch_size, args.lr, args.seed, ckpt_r, "market_residual")
            residual_m = evaluate_model_on_matches(model_r, test_m, cutoffs, model_type="market_residual")
        except Exception as e:
            print(f"  [residual] FAILED: {e}")
            residual_m = None

        # Baselines
        bl_m = baseline_on_matches(test_m, cutoffs)
        # Retrieval: index from train domain only (no test leakage)
        ret_m = retrieval_on_matches(test_m, train_set, cutoffs, K=100)

        cell = {
            "train_count": len(train_set), "test_count": len(test_m),
            "direct": {"best_epoch": ep_d, "metrics": model_m},
            "residual": {"best_epoch": ep_r if residual_m else None, "metrics": residual_m},
            "baseline": bl_m,
            "retrieval": ret_m,
        }
        # Deltas
        if model_m and bl_m:
            cell["direct_delta"] = {k: round(model_m[k] - bl_m[k], 4) for k in ["accuracy", "logloss", "brier", "ece"]}
        if residual_m and bl_m:
            cell["residual_delta"] = {k: round(residual_m[k] - bl_m[k], 4) for k in ["accuracy", "logloss", "brier", "ece"]}
        if ret_m and bl_m:
            cell["retrieval_delta"] = {k: round(ret_m[k] - bl_m[k], 4) for k in ["accuracy", "logloss", "brier", "ece"]}

        matrix[key] = cell
        if model_m and bl_m:
            d = cell["direct_delta"]
            print(f"  direct Δ: acc={d['accuracy']:+.4f} loss={d['logloss']:+.4f}")
        if ret_m and bl_m:
            print(f"  retrieval Δ: acc={cell['retrieval_delta']['accuracy']:+.4f}")

    return matrix, leagues


# ── Negative transfer detection ────────────────────────────────────────

def detect_negative_transfer(matrix, leagues):
    findings = []
    for key, cell in matrix.items():
        train_lg, test_lg = key.split("->")
        direct_d = cell.get("direct_delta", {})
        retrieval_d = cell.get("retrieval_delta", {})
        residual_d = cell.get("residual_delta", {})

        d_acc = direct_d.get("accuracy", -999)
        r_acc = retrieval_d.get("accuracy", -999)
        res_acc = residual_d.get("accuracy", -999)

        # Negative transfer: direct < retrieval AND direct < -0.01
        if d_acc < r_acc and d_acc < -0.01:
            severity = "severe" if d_acc < -0.05 else "moderate" if d_acc < -0.02 else "mild"
            findings.append({
                "type": "negative_transfer",
                "pair": key,
                "severity": severity,
                "direct_delta": d_acc,
                "retrieval_delta": r_acc,
                "gap": round(r_acc - d_acc, 4),
                "message": f"Training {train_lg}→{test_lg} hurts prediction. "
                           f"Direct Δ={d_acc:+.4f} vs retrieval Δ={r_acc:+.4f}. "
                           f"KNN alone is {r_acc - d_acc:+.3f}pp better than the trained model.",
            })

        # Residual beats direct?
        if res_acc > d_acc + 0.003:
            findings.append({
                "type": "residual_improves",
                "pair": key,
                "direct_delta": d_acc,
                "residual_delta": res_acc,
                "gap": round(res_acc - d_acc, 4),
                "message": f"Market_residual improves {train_lg}→{test_lg} by {res_acc - d_acc:+.4f}pp.",
            })

    return findings


# ── Policy recommendation ─────────────────────────────────────────────

def recommend_policy(matrix, findings, leagues):
    # Count negative transfer cases
    neg_count = sum(1 for f in findings if f["type"] == "negative_transfer")
    severe_count = sum(1 for f in findings if f["type"] == "negative_transfer" and f["severity"] == "severe")
    total_pairs = len(matrix)

    # Identify best training strategy per target league
    per_league = {}
    for lg in leagues:
        best_delta = -999; best_method = "none"
        for key, cell in matrix.items():
            if key.endswith(f"->{lg}"):
                d = cell.get("direct_delta", {}).get("accuracy", -999)
                r = cell.get("retrieval_delta", {}).get("accuracy", -999)
                if d > best_delta: best_delta = d; best_method = f"train_{key.split('->')[0]}"
                if r > best_delta: best_delta = r; best_method = "retrieval"
        per_league[lg] = {"best_delta": best_delta, "best_method": best_method}

    if severe_count >= 3:
        policy = "ISOLATION_REQUIRED"
        detail = f"{severe_count}/{total_pairs} pairs show severe negative transfer. Train separate models per league."
    elif neg_count >= total_pairs / 2:
        policy = "SELECTIVE_MIXING"
        detail = f"{neg_count}/{total_pairs} pairs show negative transfer. Mix only compatible leagues."
    else:
        policy = "MIXING_ALLOWED"
        detail = f"Only {neg_count}/{total_pairs} negative transfer cases. Multi-league training is safe."

    return {"policy": policy, "detail": detail, "per_league": per_league}


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="P2.4 Domain-Isolated Training")
    parser.add_argument("--data", type=str, default="data/odds_real/master_5330.jsonl")
    parser.add_argument("--p21_report", type=str, default="runs/p2_1_backtest/p2_1_report.json",
                        help="P2.1B LOO CV report for all-but-one results")
    parser.add_argument("--mode", type=str, default="transfer", choices=["transfer", "full"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cutoffs", type=str, default="90,60,30")
    parser.add_argument("--out-dir", type=str, default="runs/p2_4_domain")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cutoffs = [float(c.strip()) for c in args.cutoffs.split(",") if c.strip()]

    all_matches = load_matches(args.data)
    leagues = sorted(set(m.get("league_id", "") for m in all_matches))
    print(f"Leagues: {leagues}")
    for lg in leagues:
        print(f"  {lg}: {sum(1 for m in all_matches if m.get('league_id')==lg)} matches")

    # ── Transfer matrix ──
    matrix, _ = build_transfer_matrix(all_matches, cutoffs, args)

    # ── Negative transfer ──
    findings = detect_negative_transfer(matrix, leagues)

    # ── Policy ──
    policy = recommend_policy(matrix, findings, leagues)

    # ── Integrate P2.1 results if available ──
    p21_data = {}
    if os.path.exists(args.p21_report):
        with open(args.p21_report) as f:
            p21_data = json.load(f)

    # ── Build summary tables ──
    # Accuracy delta matrix
    print("\n" + "=" * 70)
    print("ACCURACY DELTA MATRIX (direct model Δ vs baseline)")
    print("=" * 70)
    header = f"{'train↓/test→':<18}"
    for lg in leagues: header += f" {lg:>10}"
    print(header); print("-" * (18 + 11 * len(leagues)))
    for train_lg in leagues:
        row = f"{train_lg:<18}"
        for test_lg in leagues:
            if train_lg == test_lg:
                row += f" {'---':>10}"
            else:
                key = f"{train_lg}->{test_lg}"
                cell = matrix.get(key, {})
                d = cell.get("direct_delta", {}).get("accuracy", None)
                if d is not None: row += f" {d:>+10.4f}"
                else: row += f" {'N/A':>10}"
        print(row)

    # All-but-one row
    if "p2_1b" in p21_data:
        row = f"{'all_except_test':<18}"
        for lg in leagues:
            fold = p21_data["p2_1b"].get(lg, {})
            d = fold.get("delta_temp", {}).get("accuracy", None)
            if d is not None: row += f" {d:>+10.4f}"
            else: row += f" {'N/A':>10}"
        print(row)

    # Same-league row
    if "p2_1a" in p21_data:
        row = f"{'same_league':<18}"
        for lg in leagues:
            sl = p21_data["p2_1a"].get(lg, {})
            d = sl.get("delta_temp", {}).get("accuracy", None)
            if d is not None: row += f" {d:>+10.4f}"
            else: row += f" {'N/A':>10}"
        print(row)

    print(f"\nNegative transfer: {len(findings)} cases")
    for f in findings:
        print(f"  [{f['severity']}] {f['pair']}: {f['message'][:100]}")

    print(f"\nPolicy: {policy['policy']}")
    print(f"  {policy['detail']}")
    for lg, info in policy["per_league"].items():
        print(f"  {lg}: best={info['best_method']} (Δ={info['best_delta']:+.4f})")

    # ── Save report ──
    report = {
        "experiment": "P2.4",
        "leagues": leagues,
        "transfer_pairs": list(matrix.keys()),
        "matrix": {k: {"direct_delta": v.get("direct_delta"),
                        "residual_delta": v.get("residual_delta"),
                        "retrieval_delta": v.get("retrieval_delta"),
                        "train_count": v["train_count"], "test_count": v["test_count"]}
                   for k, v in matrix.items()},
        "negative_transfer": findings,
        "policy": policy,
        "p21_integrated": bool(p21_data),
    }
    with open(os.path.join(args.out_dir, "p2_4_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport: {os.path.join(args.out_dir, 'p2_4_report.json')}")


if __name__ == "__main__":
    main()
