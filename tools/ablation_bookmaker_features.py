"""Quick tabular train with bookmaker features. P1.6 ablation."""
import json, torch, sys, os, argparse
sys.path.insert(0, '.')
from model.odds_tabular_baselines import OddsTinyMLP
from dataset.odds_split import load_match_ids_from_file
from torch.utils.data import DataLoader
from torch import optim

def collate(batch):
    feats = torch.stack([item[0] for item in batch])
    eu = torch.tensor([item[1] for item in batch])
    ah = torch.tensor([item[2] for item in batch])
    return {"features": feats, "euro_labels": eu, "asian_labels": ah}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--train-ids", required=True)
    parser.add_argument("--val-ids", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-set", default="all", choices=["all","consensus_only","consensus_dispersion","consensus_sharp","no_ah"])
    args = parser.parse_args()

    # Load data
    with open(args.data) as f:
        samples = [json.loads(l) for l in f if l.strip()]

    train_ids = load_match_ids_from_file(args.train_ids)
    val_ids = load_match_ids_from_file(args.val_ids)

    # Get feature names
    from dataset.odds_bookmaker_features import get_feature_names
    ALL_FEATURES = get_feature_names()

    # Feature sets
    CONSENSUS = [n for n in ALL_FEATURES if 'prob_avg' in n or 'prob_median' in n]
    DISPERSION = [n for n in ALL_FEATURES if 'prob_std' in n or 'prob_range' in n]
    SHARP = [n for n in ALL_FEATURES if any(s in n for s in ['pinnacle','betfair','b365','soft_avg','_vs_'])]
    AH = [n for n in ALL_FEATURES if n.startswith('ah_')]

    if args.feature_set == "consensus_only":
        feature_names = CONSENSUS
    elif args.feature_set == "consensus_dispersion":
        feature_names = CONSENSUS + DISPERSION
    elif args.feature_set == "consensus_sharp":
        feature_names = CONSENSUS + DISPERSION + SHARP
    elif args.feature_set == "no_ah":
        feature_names = [n for n in ALL_FEATURES if not n.startswith('ah_')]
    else:
        feature_names = ALL_FEATURES

    print(f"Feature set: {args.feature_set}, dim={len(feature_names)}")

    # Build datasets
    def build_xy(samples, match_ids):
        X, y_eu, y_ah = [], [], []
        for s in samples:
            if s['match_id'] not in match_ids:
                continue
            feats = s.get('bookmaker_features', {})
            vec = [feats.get(n, 0.0) for n in feature_names]
            X.append(vec)
            from dataset.odds_dataset import EURO_MAP, ASIAN_MAP_5CLASS
            y_eu.append(EURO_MAP[s['label']['euro_result']])
            y_ah.append(ASIAN_MAP_5CLASS[s['label']['asian_result']])
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y_eu), torch.tensor(y_ah)

    X_train, eu_train, ah_train = build_xy(samples, train_ids)
    X_val, eu_val, ah_val = build_xy(samples, val_ids)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    train_ds = list(zip(X_train, eu_train, ah_train))
    loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)

    asian_num_classes = 5
    model = OddsTinyMLP(hidden_size=64, asian_num_classes=asian_num_classes).to(args.device)
    # Override feat_dim
    import torch.nn as nn
    model.shared[0] = nn.Linear(len(feature_names), 64)
    model.euro_head = nn.Linear(64, 3)
    model.asian_head = nn.Linear(64, asian_num_classes)
    model.to(args.device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    import torch.nn.functional as F

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            f = batch["features"].to(args.device)
            eu = batch["euro_labels"].to(args.device)
            ah = batch["asian_labels"].to(args.device)
            optimizer.zero_grad()
            out = model(f)
            loss = F.cross_entropy(out["euro_logits"], eu) + F.cross_entropy(out["asian_logits"], ah)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Val
        model.eval()
        with torch.no_grad():
            out = model(X_val.to(args.device))
            eu_acc = (out["euro_logits"].argmax(-1).cpu() == eu_val).float().mean().item()
            ah_acc = (out["asian_logits"].argmax(-1).cpu() == ah_val).float().mean().item()
        print(f"Epoch {epoch+1}: loss={total_loss/len(loader):.4f}, val_euro={eu_acc:.4f}, val_asian={ah_acc:.4f}")
        model.train()

if __name__ == "__main__":
    main()
