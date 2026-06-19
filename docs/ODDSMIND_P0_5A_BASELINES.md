# OddsMind P0.5A — Simple Deterministic Baselines

## 1. Why Baselines

Before claiming a model "works", you need a floor. Baselines define the
minimum a model must beat. If a simple rule-based approach outperforms
the model, the model is not adding value.

## 2. Baselines Implemented

| Baseline | What it does |
|---|---|
| `lowest_odds_euro` | Pick the outcome with the lowest decimal odds (highest implied probability direction) |
| `implied_prob_euro` | Convert 1/odds to probabilities, normalize to sum=1 |
| `low_water_asian_3class` | Pick the side with lower water level; equal water → push |
| `low_water_asian_5class` | Coarse projection: low water → full_win/full_loss; equal → push. Cannot predict half-win/loss |
| `uniform` | Equal probability for all classes (sanity check) |

## 3. Euro Baseline Definitions

### lowest_odds_euro

```
min(euro_h, euro_d, euro_a) → that outcome gets probability 1.0
```

### implied_prob_euro

```
raw_i = 1 / odds_i
prob_i = raw_i / sum(raw)
```

This is the standard "bookmaker margin normalization" approach.
Note: bookmakers build a margin into their odds (~5-8%), so the sum
of raw probabilities is > 1. Normalizing removes the margin evenly.

## 4. Asian Baseline Definitions

### low_water_asian_3class

```
if upper_water < lower_water → upper
elif lower_water < upper_water → lower
else (|diff| <= eps) → push
```

### low_water_asian_5class

Same logic, but maps to:
- upper → upper_full_win
- lower → upper_full_loss
- push → push

**Limitation**: cannot predict upper_half_win or upper_half_loss because
the water level alone doesn't encode score margin. This is a documented
weakness of the deterministic baseline.

## 5. Baselines Are NOT Betting Advice

All baselines are purely for model evaluation comparison. They do not
constitute betting recommendations, value judgments, or gambling advice.

## 6. Usage

```bash
# Run all baselines on a test split
python eval/eval_odds_baselines.py \
  --data ...5class.jsonl \
  --split-match-ids splits/test_match_ids.txt \
  --cutoffs 90,60,30 --cutoff-mode exhaustive \
  --asian-label-mode 5class \
  --out-json runs/baseline_eval.json

# Compare with model eval
python eval/eval_odds_model.py --untrained ...  # model metrics
python eval/eval_odds_baselines.py ...           # baseline metrics
```

## 7. Next Phase

**P0.5B — LightGBM / Logistic Regression Tabular Baseline**
or
**P0.6 — MiniMindBlock Reuse Probe**
