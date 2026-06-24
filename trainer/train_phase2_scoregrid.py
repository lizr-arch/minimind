"""P1.13A: Two-phase training — freeze backbone, train ScoreGridHead only.

ScoreGridHead outputs an 8x8 probability grid directly (no Poisson assumption).
Loss = CE(64-class) + 0.3*marginal_1X2 + 0.3*marginal_total + 0.2*marginal_diff."""
import argparse, torch, sys, os, copy, math, random, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, '.')

from model.model_oddsmind import OddsMindConfig, OddsMindModel
from dataset.odds_dataset import OddsDataset
from dataset.odds_collator import OddsCollator
from dataset.odds_split import load_match_ids_from_file
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True, help="Phase 1 checkpoint (euro+asian trained)")
    parser.add_argument("--train-ids", required=True)
    parser.add_argument("--val-ids", default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="runs/p1_13a_scoregrid")
    parser.add_argument("--soft-self-weight", type=float, default=0.75)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()

    random.seed(42); torch.manual_seed(42)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load Phase 1 model
    cfg = OddsMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_layers,
                         num_attention_heads=args.num_heads, asian_num_classes=5,
                         score_head_version="grid")
    model = OddsMindModel(cfg).to(args.device)
    ckp = torch.load(args.checkpoint, map_location=args.device)
    if 'model_state_dict' in ckp: ckp = ckp['model_state_dict']
    model.load_state_dict(ckp, strict=False)

    # FREEZE everything except score_head
    for name, param in model.named_parameters():
        if 'score_head' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    score_params = sum(p.numel() for p in model.score_head.parameters())
    trainable = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"ScoreGridHead params: {score_params:,}  (trainable: {trainable:,} / total: {total:,})")
    print(f"Grid size: {model.score_head.GRID_SIZE}x{model.score_head.GRID_SIZE} = {model.score_head.NUM_CLASSES} classes")

    # Data
    train_ids = load_match_ids_from_file(args.train_ids)
    ds = OddsDataset(args.data, cutoffs=[0], cutoff_mode='exhaustive',
                     asian_label_mode='5class', allowed_match_ids=train_ids,
                     feature_schema_version="v2")  # P1.15B
    collator = OddsCollator()
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    print(f"Train: {len(ds)} samples")

    optimizer = torch.optim.AdamW(model.score_head.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(loader, start=1):
            f = batch['features'].to(args.device)
            m = batch['attention_mask'].to(args.device)
            sl = batch['score_labels'].to(args.device)

            optimizer.zero_grad()
            out = model(f, attention_mask=m, euro_labels=batch['euro_labels'].to(args.device),
                        asian_labels=batch['asian_labels'].to(args.device),
                        score_labels=sl, score_loss_type='poisson',
                        score_loss_weight=0.0)

            loss = out.get('score_loss', out.get('loss', torch.tensor(0.0)))
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item()

            if step % 50 == 0 or step == len(loader):
                aux = out.get('aux_losses', {})
                aux_parts = []
                for k in ['score_grid_ce', 'marginal_result_ce', 'marginal_total_ce', 'marginal_diff_ce']:
                    if k in aux:
                        aux_parts.append(f"{k.split('_')[-1]}={aux[k]:.3f}")
                aux_str = " " + " ".join(aux_parts) if aux_parts else ""
                print(f"Epoch {epoch}/{args.epochs} [{step}/{len(loader)}] "
                      f"loss={loss.item():.4f}{aux_str}")

        avg = total_loss / len(loader)
        print(f"Epoch {epoch} complete: avg_loss={avg:.4f}")

    # Save full model (with score head)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "oddsmind_with_score.pth"))
    print(f"Saved to {args.out_dir}/oddsmind_with_score.pth")

if __name__ == "__main__":
    main()
