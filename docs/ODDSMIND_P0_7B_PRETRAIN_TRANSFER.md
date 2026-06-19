# OddsMind P0.7B — Pretrain-to-Supervised Weight Transfer

## 1. Why Transfer

Self-supervised pretraining learns general odds trajectory patterns.
These learned encoder+transformer weights can serve as a better
initialization for supervised classification than random weights.

## 2. What Gets Transferred

| Transferred | NOT Transferred |
|---|---|
| `encoder.*` (OddsEventEncoder) | `recon_head.*` (reconstruction head) |
| `layers.*` / `_minimind_adapter.*` (Transformer) | `event_head.*` (event prediction head) |
| `final_norm.*` | `euro_head.*` (supervised classification) |
| | `asian_head.*` (supervised classification) |

## 3. Usage

```bash
# Dry-run (no file written)
python tools/odds_transfer_pretrain_to_supervised.py --dry-run ...

# Generate supervised init checkpoint
python tools/odds_transfer_pretrain_to_supervised.py --out supervised_init.pth ...

# Train directly with pretrained encoder
python trainer/train_odds_supervised.py \
  --pretrained-encoder-checkpoint pretrain_checkpoint.pth ...
```

## 4. Both Backends Supported

`odds_native` and `minimind` backends are supported. Backend must match
between pretrain checkpoint and supervised model.

## 5. Next Phase

**P0.8 — Real Data Import Adapter**
