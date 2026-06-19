# OddsMind P0.7 — Self-Supervised Pretraining Scaffold

## 1. Why Pretrain on Odds

Supervised training directly predicts match outcomes (win/draw/loss). But
odds timelines contain rich structural patterns — how odds move, how they
interact, typical trajectories — that can be learned without labels.

Self-supervised pretraining lets the model learn the "language" of odds
before being asked to predict match results. This is analogous to how LLMs
pretrain on raw text before SFT.

## 2. Three Tasks

| Task | Input | Target | Loss |
|---|---|---|---|
| `masked_reconstruction` | Partially masked timeline | Original values at masked positions | MSE |
| `next_event` | All events except last | The last event's features | MSE |
| `closing_prediction` | Early events | Final visible event | MSE |

None of these tasks use match result labels (euro_result, asian_result).

## 3. No Tokenizer / LM Head

Pretraining operates directly on continuous feature vectors `[T, 7]`.
No discrete tokenization, no vocabulary lookup, no causal language modeling head.

## 4. Usage

```bash
# Masked reconstruction pretrain
python trainer/train_odds_pretrain.py --task masked_reconstruction --mask-ratio 0.15 ...

# Next-event prediction pretrain
python trainer/train_odds_pretrain.py --task next_event ...

# Evaluate
python eval/eval_odds_pretrain.py --task masked_reconstruction --model .../checkpoint.pth ...
```

Both `odds_native` and `minimind` backends are supported via `--transformer-backend`.

## 5. Current Status — Scaffold Only

All tasks train and produce decreasing loss on fixture data, but no meaningful
representations have been learned. This is pure scaffold — real pretraining
requires much more data and tuning.

## 6. Next Phase

**P0.7B — Pretrain-to-Supervised Weight Transfer** or **P0.8 — Real Data Import Adapter**
