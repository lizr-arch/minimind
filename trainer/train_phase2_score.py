"""P1.9/P1.10: Two-phase training — freeze backbone, train score head only.

P1.10: ScoreHeadV2 with bounded log-rate + total_goals/goal_diff auxiliary heads."""
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
    parser.add_argument("--out-dir", default="runs/phase2_score")
    parser.add_argument("--score-loss-weight", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()

    random.seed(42); torch.manual_seed(42)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load Phase 1 model
    cfg = OddsMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_layers,
                         num_attention_heads=args.num_heads, asian_num_classes=3)
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
    print(f"Score head params: {score_params:,}  (trainable: {trainable:,} / total: {total:,})")
    print(f"Bounded log-rate: {getattr(model.score_head, 'bounded_log_rate', 'N/A')}")

    # Data
    train_ids = load_match_ids_from_file(args.train_ids)
    ds = OddsDataset(args.data, cutoffs=[0], cutoff_mode='exhaustive',
                     asian_label_mode='3class', allowed_match_ids=train_ids)
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
                        score_loss_weight=1.0)

            loss = out.get('score_loss', out.get('loss', torch.tensor(0.0)))
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item()

            if step % 50 == 0 or step == len(loader):
                avg_pred = out['score_preds'].mean(0).cpu().tolist()
                raw_rng = (out.get('score_raw_min', 0), out.get('score_raw_max', 0))
                log_rng = (out.get('score_log_rate_min', 0), out.get('score_log_rate_max', 0))
                aux = out.get('aux_losses', {})
                aux_str = f" total_aux={aux['total_aux']:.4f} diff_aux={aux['diff_aux']:.4f}" if aux else ""
                print(f"Epoch {epoch}/{args.epochs} [{step}/{len(loader)}] "
                      f"loss={loss.item():.4f}{aux_str} pred_avg=[{avg_pred[0]:.2f}, {avg_pred[1]:.2f}]"
                      f" raw=[{raw_rng[0]:.1f},{raw_rng[1]:.1f}] log=[{log_rng[0]:.1f},{log_rng[1]:.1f}]")

        avg = total_loss / len(loader)
        print(f"Epoch {epoch} complete: avg_loss={avg:.4f}")

    # Save full model (with score head)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "oddsmind_with_score.pth"))
    print(f"Saved to {args.out_dir}/oddsmind_with_score.pth")

if __name__ == "__main__":
    main()
