"""
P1.14: Score-Aware Backbone Fine-tuning — phased unfreezing + distillation.

Phase 1 (warmup):  freeze backbone, train ScoreGridHead only  (P1.13A — already done)
Phase 2 (partial):  unfreeze last 2 transformer blocks + ScoreGridHead
Phase 3 (deeper):   unfreeze last 4 transformer blocks + ScoreGridHead

Distillation: KL(current_euro || teacher_euro) + KL(current_asian || teacher_asian)
to prevent Euro/Asian regression.

Red lines:
  Euro accuracy drop > 0.5pp → reject
  Asian accuracy drop > 0.5pp → reject

Usage:
  # Phase 2: unfreeze last 2 blocks
  python trainer/train_p1_14_score_aware.py \
      --data data/odds_real/v5_b365.jsonl \
      --checkpoint runs/p1_13a/oddsmind_with_score.pth \
      --train-ids data/odds_real/splits_v5/train_match_ids.txt \
      --val-ids data/odds_real/splits_v5/val_match_ids.txt \
      --hidden-size 512 --num-layers 16 --num-heads 8 --device cuda \
      --epochs 5 --unfreeze-layers 2 --kl-weight 0.1 \
      --out-dir runs/p1_14_phase2
"""
import argparse, torch, sys, os, copy, random, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, '.')

import torch.nn.functional as F
from model.model_oddsmind import OddsMindConfig, OddsMindModel
from model.score_grid_utils import compute_score_grid_loss, score_grid_predictions
from dataset.odds_dataset import OddsDataset
from dataset.odds_collator import OddsCollator
from dataset.odds_split import load_match_ids_from_file
from torch.utils.data import DataLoader


def build_teacher(checkpoint_path, config, device):
    """Build frozen teacher model for distillation."""
    teacher = OddsMindModel(config).to(device)
    ck = torch.load(checkpoint_path, map_location=device, weights_only=True)
    teacher.load_state_dict(ck, strict=False)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    return teacher


def set_requires_grad(model, unfreeze_layers: int):
    """Freeze all except ScoreGridHead + last N transformer blocks."""
    total_layers = len(model.layers)
    freeze_until = total_layers - unfreeze_layers  # e.g., 16-2=14 → freeze layers 0..13

    for name, param in model.named_parameters():
        if 'score_head' in name:
            param.requires_grad = True
        elif 'layers.' in name:
            # Extract layer index from name like 'layers.14.attn_norm.weight'
            parts = name.split('.')
            try:
                layer_idx = int(parts[1])
                param.requires_grad = (layer_idx >= freeze_until)
            except (ValueError, IndexError):
                param.requires_grad = False
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Unfreezing last {unfreeze_layers}/{total_layers} layers")
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def compute_kl_loss(student_logits, teacher_logits):
    """KL divergence: KL(teacher || student) = teacher * log(teacher/student)."""
    teacher_probs = F.softmax(teacher_logits, dim=-1).detach()
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    kl = (teacher_probs * (torch.log(teacher_probs + 1e-10) - student_log_probs)).sum(-1).mean()
    return kl


def quick_eval(model, loader, device):
    """Fast evaluation: Euro/Asian accuracy."""
    model.eval()
    euro_correct = 0
    asian_correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            f = batch['features'].to(device)
            m = batch['attention_mask'].to(device)
            el = batch['euro_labels'].to(device)
            al = batch['asian_labels'].to(device)
            out = model(f, attention_mask=m)
            euro_pred = out['euro_logits'].argmax(-1)
            asian_pred = out['asian_logits'].argmax(-1)
            euro_correct += (euro_pred == el).sum().item()
            asian_correct += (asian_pred == al).sum().item()
            total += el.shape[0]
    model.train()
    return euro_correct / total, asian_correct / total


def compute_representation_drift(model, teacher, loader, device):
    """Compute cosine similarity between student and teacher pooled representations."""
    model.eval()
    teacher.eval()
    cosines = []
    with torch.no_grad():
        for batch in loader:
            f = batch['features'].to(device)
            m = batch['attention_mask'].to(device)
            # We need pooled representation — hack: run forward and capture intermediate
            # For now, compare score_head input (pooled) by running encoder manually
            h = model.encoder(f)
            key_pad = ~m.bool() if m is not None else None
            for layer in model.layers:
                h = layer(h, key_padding_mask=key_pad)
            h_student = model.final_norm(h)
            if m is not None:
                pooled_s = (h_student * m.unsqueeze(-1).float()).sum(1) / m.sum(1, keepdim=True).clamp(min=1).float()
            else:
                pooled_s = h_student.mean(1)

            # Teacher
            h_t = teacher.encoder(f)
            for layer_t in teacher.layers:
                h_t = layer_t(h_t, key_padding_mask=key_pad)
            h_t = teacher.final_norm(h_t)
            if m is not None:
                pooled_t = (h_t * m.unsqueeze(-1).float()).sum(1) / m.sum(1, keepdim=True).clamp(min=1).float()
            else:
                pooled_t = h_t.mean(1)

            cos = F.cosine_similarity(pooled_s, pooled_t, dim=-1).mean().item()
            cosines.append(cos)
    model.train()
    teacher.eval()
    return sum(cosines) / len(cosines) if cosines else 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True, help="P1.13A ScoreGrid checkpoint")
    parser.add_argument("--train-ids", required=True)
    parser.add_argument("--val-ids", default="")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5, help="Lower LR for backbone fine-tuning")
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="runs/p1_14_phase2")
    parser.add_argument("--unfreeze-layers", type=int, default=2, help="Number of top layers to unfreeze")
    parser.add_argument("--kl-weight", type=float, default=0.1, help="Distillation KL loss weight")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every-epoch", action="store_true", default=True)
    args = parser.parse_args()

    random.seed(42); torch.manual_seed(42)
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device

    cfg = OddsMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_layers,
                         num_attention_heads=args.num_heads, asian_num_classes=3,
                         score_head_version='grid')

    # Build student model
    model = OddsMindModel(cfg).to(device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ck, strict=False)

    # Build frozen teacher (same checkpoint)
    teacher = build_teacher(args.checkpoint, cfg, device)

    # Phase 2: unfreeze last N layers + ScoreGridHead
    set_requires_grad(model, args.unfreeze_layers)

    # Data
    train_ids = load_match_ids_from_file(args.train_ids)
    ds = OddsDataset(args.data, cutoffs=[0], cutoff_mode='exhaustive',
                     asian_label_mode='3class', allowed_match_ids=train_ids)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=OddsCollator())
    print(f"Train: {len(ds)} samples")

    # Val loader for monitoring
    val_loader = None
    if args.val_ids:
        val_ids = load_match_ids_from_file(args.val_ids)
        val_ds = OddsDataset(args.data, cutoffs=[0], cutoff_mode='exhaustive',
                             asian_label_mode='3class', allowed_match_ids=val_ids)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=OddsCollator())

    # Optimizer: separate LR for backbone vs head
    backbone_params = [p for n, p in model.named_parameters() if 'score_head' not in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if 'score_head' in n and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': head_params, 'lr': args.lr * 10},       # head trains faster
        {'params': backbone_params, 'lr': args.lr},         # backbone fine-tuned slowly
    ])

    # Baseline eval before training
    if val_loader:
        base_euro, base_asian = quick_eval(model, val_loader, device)
        base_drift = compute_representation_drift(model, teacher, val_loader, device)
        print(f"Baseline val: Euro={base_euro:.4f} Asian={base_asian:.4f} drift_cosine={base_drift:.4f}")

    best_score_loss = float('inf')
    log = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_score = 0.0
        total_kl = 0.0

        for step, batch in enumerate(loader, start=1):
            f = batch['features'].to(device)
            m = batch['attention_mask'].to(device)
            sl = batch['score_labels'].to(device)
            el = batch['euro_labels'].to(device)
            al = batch['asian_labels'].to(device)

            optimizer.zero_grad()

            # Student forward
            out = model(f, attention_mask=m, euro_labels=el, asian_labels=al,
                        score_labels=sl, score_loss_type='poisson', score_loss_weight=0.0)

            # ScoreGrid loss
            score_loss, aux = compute_score_grid_loss(
                out['score_grid_logits'], sl,
                ce_weight=1.0, result_weight=0.3, total_weight=0.3, diff_weight=0.2,
                soft_target=True, soft_self_weight=0.75,
            )

            # Distillation loss
            with torch.no_grad():
                teacher_out = teacher(f, attention_mask=m)

            kl_euro = compute_kl_loss(out['euro_logits'], teacher_out['euro_logits'])
            kl_asian = compute_kl_loss(out['asian_logits'], teacher_out['asian_logits'])
            kl_loss = args.kl_weight * (kl_euro + kl_asian)

            loss = score_loss + kl_loss
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            total_score += score_loss.item()
            total_kl += kl_loss.item()

            if step % 50 == 0 or step == len(loader):
                aux_str = " ".join([f"{k.split('_')[-1]}={v:.3f}" for k, v in aux.items()])
                print(f"E{epoch}/{args.epochs} [{step}/{len(loader)}] "
                      f"loss={loss.item():.3f} score={score_loss.item():.3f} "
                      f"kl_e={kl_euro.item():.4f} kl_a={kl_asian.item():.4f} {aux_str}")

        avg_loss = total_loss / len(loader)
        avg_score = total_score / len(loader)
        avg_kl = total_kl / len(loader)
        print(f"Epoch {epoch} complete: loss={avg_loss:.4f} score={avg_score:.4f} kl={avg_kl:.4f}")

        # Eval on val
        epoch_log = {"epoch": epoch, "loss": avg_loss, "score_loss": avg_score, "kl_loss": avg_kl}
        if val_loader and args.eval_every_epoch:
            euro_acc, asian_acc = quick_eval(model, val_loader, device)
            drift = compute_representation_drift(model, teacher, val_loader, device)
            euro_drop = base_euro - euro_acc
            asian_drop = base_asian - asian_acc
            print(f"  Val: Euro={euro_acc:.4f} (drop={euro_drop:+.4f}) "
                  f"Asian={asian_acc:.4f} (drop={asian_drop:+.4f}) drift={drift:.4f}")
            epoch_log.update({
                "euro_acc": euro_acc, "euro_drop": euro_drop,
                "asian_acc": asian_acc, "asian_drop": asian_drop,
                "drift_cosine": drift,
            })

            # Red line check
            if euro_drop > 0.005:
                print(f"  ⚠ WARNING: Euro drop {euro_drop:.4f} exceeds 0.5pp!")
            if asian_drop > 0.005:
                print(f"  ⚠ WARNING: Asian drop {asian_drop:.4f} exceeds 0.5pp!")

        log.append(epoch_log)

        # Save checkpoint
        if avg_score < best_score_loss:
            best_score_loss = avg_score
            torch.save(model.state_dict(), os.path.join(args.out_dir, "oddsmind_with_score.pth"))
            print(f"  Saved best (score_loss={best_score_loss:.4f})")

    # Save log
    with open(os.path.join(args.out_dir, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"Done. Best score_loss={best_score_loss:.4f}")


if __name__ == "__main__":
    main()
