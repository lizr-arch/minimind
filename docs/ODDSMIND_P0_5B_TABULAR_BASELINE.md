# OddsMind P0.5B — Lightweight Trainable Tabular Baselines

## 1. Why Trainable Tabular Baselines

Deterministic baselines (P0.5A) use fixed rules (lowest odds, implied prob,
low water).  Trainable baselines can learn patterns from the data —
they can weight features, capture non-linear interactions, and potentially
outperform the deterministic rules.

A tabular model also serves as a sanity check: if the Transformer can't beat
a simple MLP on the same features, the Transformer architecture is likely
overkill or misconfigured.

## 2. Feature Extraction

`dataset/odds_features.py` extracts 32 fixed-length features per sample:

| Group | Count | Description |
|---|---|---|
| Open event | 7 | Raw features from earliest odds event |
| Last event | 7 | Raw features from event closest to cutoff |
| Implied prob | 3 | 1/odds normalized from last event |
| Open→last delta | 7 | Change in each feature from open to last |
| Timeline stats | 8 | min/max of euro_h, euro_d, euro_a, asian_line |

## 3. Models

| Model | Params | Description |
|---|---|---|
| `OddsLogisticRegression` | ~264 | Linear classifier, one head for euro (3) + one for asian (3 or 5) |
| `OddsTinyMLP` | ~6.8K (hidden=64) | 2-layer MLP with SiLU + dropout, shared body + separate heads |

Both output `{"euro_logits": [B,3], "asian_logits": [B,3 or 5]}`.

## 4. Usage

```bash
# Train
python trainer/train_odds_tabular_baseline.py --model-type logistic ...
python trainer/train_odds_tabular_baseline.py --model-type tiny_mlp --hidden-size 64 ...

# Evaluate
python eval/eval_odds_tabular_baseline.py --model runs/.../odds_tabular_logistic.pth ...

# Compare
python eval/compare_odds_evals.py \
  --eval-json tabular=runs/tabular_eval.json \
  --eval-json deterministic=runs/baseline_eval.json
```

## 5. No External Dependencies

All components use only Python stdlib + PyTorch.  No numpy, sklearn, or LightGBM.

## 6. Next Phase

**P0.6 — MiniMindBlock Reuse Probe**
