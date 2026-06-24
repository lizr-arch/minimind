"""
P1.13B: ScoreGrid loss weight tuning — quick search on val subset.

Tests 5 configs around baseline, trains 1 epoch on 500 val samples,
evaluates on remaining val samples.
"""
import sys, os, copy, json, torch, random, argparse
sys.path.insert(0, '.')
from model.model_oddsmind import OddsMindConfig, OddsMindModel
from model.score_grid_utils import compute_score_grid_loss
from dataset.odds_dataset import OddsDataset
from dataset.odds_collator import OddsCollator
from dataset.odds_split import load_match_ids_from_file
from torch.utils.data import DataLoader, Subset


CONFIGS = [
    {"name": "baseline",    "ce": 1.0, "result": 0.3, "total": 0.3, "diff": 0.2, "soft": 0.75},
    {"name": "more_aux",    "ce": 1.0, "result": 0.5, "total": 0.5, "diff": 0.3, "soft": 0.75},
    {"name": "less_aux",    "ce": 1.0, "result": 0.15, "total": 0.15, "diff": 0.1, "soft": 0.75},
    {"name": "sharper",     "ce": 1.0, "result": 0.3, "total": 0.3, "diff": 0.2, "soft": 0.90},
    {"name": "softer",      "ce": 1.0, "result": 0.3, "total": 0.3, "diff": 0.2, "soft": 0.60},
    {"name": "high_ce",     "ce": 1.5, "result": 0.2, "total": 0.2, "diff": 0.1, "soft": 0.75},
    {"name": "even_weights","ce": 1.0, "result": 0.3, "total": 0.3, "diff": 0.3, "soft": 0.75},
]


def train_one_epoch(model, loader, optimizer, config, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        f = batch['features'].to(device)
        m = batch['attention_mask'].to(device)
        sl = batch['score_labels'].to(device)

        optimizer.zero_grad()
        out = model(f, attention_mask=m,
                    euro_labels=batch['euro_labels'].to(device),
                    asian_labels=batch['asian_labels'].to(device),
                    score_labels=sl, score_loss_type='poisson',
                    score_loss_weight=0.0)

        grid_logits = out['score_grid_logits']
        loss, _ = compute_score_grid_loss(
            grid_logits, sl,
            ce_weight=config['ce'],
            result_weight=config['result'],
            total_weight=config['total'],
            diff_weight=config['diff'],
            soft_target=True,
            soft_self_weight=config['soft'],
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_scoregrid(model, loader, device):
    """Quick eval: total_mae, result_from_score, and per-bin MAEs."""
    model.eval()
    all_expected_home = []
    all_expected_away = []
    all_true_home = []
    all_true_away = []

    with torch.no_grad():
        for batch in loader:
            f = batch['features'].to(device)
            m = batch['attention_mask'].to(device)
            sl = batch['score_labels']
            out = model(f, attention_mask=m)
            sp = out['score_preds']  # [B, 2] expected goals
            all_expected_home.append(sp[:, 0].cpu())
            all_expected_away.append(sp[:, 1].cpu())
            all_true_home.append(sl[:, 0].cpu())
            all_true_away.append(sl[:, 1].cpu())

    eh = torch.cat(all_expected_home)
    ea = torch.cat(all_expected_away)
    th = torch.cat(all_true_home)
    ta = torch.cat(all_true_away)

    total_pred = eh + ea
    total_true = th + ta
    total_mae = (total_pred - total_true).abs().mean().item()

    pred_result = torch.where(eh > ea, 0, torch.where(eh < ea, 2, 1))
    true_result = torch.where(th > ta, 0, torch.where(th < ta, 2, 1))
    result_acc = (pred_result == true_result).float().mean().item()

    # Per-bin MAE
    bin_mae = {}
    for lo, hi, label in [(0,1,"0"),(1,2,"1"),(2,3,"2"),(3,4,"3"),(4,5,"4"),(5,99,"5+")]:
        mask = (total_true >= lo) & (total_true < hi)
        if mask.sum() > 0:
            bin_mae[f"bin_{label}"] = (total_pred[mask] - total_true[mask]).abs().mean().item()

    return {"total_mae": total_mae, "result_acc": result_acc, "bin_mae": bin_mae}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/odds_real/v5_b365.jsonl")
    parser.add_argument("--checkpoint", default="runs/v7b_line_fix/oddsmind_smoke.pth")
    parser.add_argument("--val-ids", default="data/odds_real/splits_v5/val_match_ids.txt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-samples", type=int, default=500)
    parser.add_argument("--out-dir", default="data/reports/p1_13b")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device

    # Load model with ScoreGridHead
    cfg = OddsMindConfig(hidden_size=512, num_hidden_layers=16, num_attention_heads=8,
                         asian_num_classes=3, score_head_version='grid')
    base_model = OddsMindModel(cfg).to(device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=True)
    base_model.load_state_dict(ck, strict=False)
    base_model.eval()

    # Get base model's teacher logits for distillation (future P1.14)
    # For now, just use for monitoring

    # Load val data
    all_ids = load_match_ids_from_file(args.val_ids)
    ds = OddsDataset(args.data, cutoffs=[0], cutoff_mode='exhaustive',
                     asian_label_mode='3class', allowed_match_ids=all_ids)
    collator = OddsCollator()

    # Split val into train subset and eval subset
    n = len(ds)
    train_n = min(args.train_samples, n // 2)
    indices = list(range(n))
    random.shuffle(indices)
    train_idx = indices[:train_n]
    eval_idx = indices[train_n:]

    train_ds = Subset(ds, train_idx)
    eval_ds = Subset(ds, eval_idx)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_ds, batch_size=64, shuffle=False, collate_fn=collator)

    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")
    print()

    results = []
    for cfg_dict in CONFIGS:
        # Fresh model for each config
        model = OddsMindModel(cfg).to(device)
        model.load_state_dict(ck, strict=False)

        # Freeze backbone
        for n, p in model.named_parameters():
            if 'score_head' not in n:
                p.requires_grad = False

        optimizer = torch.optim.AdamW(model.score_head.parameters(), lr=1e-3)

        name = cfg_dict['name']
        print(f"Training {name}: ce={cfg_dict['ce']} res={cfg_dict['result']} "
              f"tot={cfg_dict['total']} diff={cfg_dict['diff']} soft={cfg_dict['soft']}")

        train_loss = train_one_epoch(model, train_loader, optimizer, cfg_dict, device)
        metrics = eval_scoregrid(model, eval_loader, device)
        metrics['train_loss'] = train_loss
        metrics['config'] = cfg_dict['name']

        # Composite score: total_mae (lower better) + (1-result_acc) (lower better)
        composite = metrics['total_mae'] + 0.5 * (1.0 - metrics['result_acc'])
        metrics['composite'] = composite

        results.append(metrics)
        print(f"  loss={train_loss:.4f} total_mae={metrics['total_mae']:.4f} "
              f"result_acc={metrics['result_acc']:.4f} composite={composite:.4f}")
        print()

    # Rank by composite
    results.sort(key=lambda x: x['composite'])
    print("=== RANKED ===")
    for i, r in enumerate(results):
        print(f"{i+1}. {r['config']:<15} composite={r['composite']:.4f} "
              f"total_mae={r['total_mae']:.4f} result_acc={r['result_acc']:.4f}")

    best = results[0]
    print(f"\nBest: {best['config']}")
    print(json.dumps(best, indent=2))

    with open(os.path.join(args.out_dir, "loss_weight_search.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save best config
    best_config = next(c for c in CONFIGS if c['name'] == best['config'])
    with open(os.path.join(args.out_dir, "best_loss_config.json"), "w") as f:
        json.dump(best_config, f, indent=2)


if __name__ == "__main__":
    main()
