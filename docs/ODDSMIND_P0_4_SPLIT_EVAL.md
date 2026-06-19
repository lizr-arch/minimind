# OddsMind P0.4 — Time-Based Grouped Split + Evaluation Metrics

## 1. Why Not Random Shuffle

Odds prediction is a **time-series task**.  Future matches cannot be used to
train a model that predicts past matches.  Random shuffling would mix
2024-H2 data into training and 2024-H1 into testing — classic data leakage.

**Rule**: split by `kickoff_time`, earliest matches → train, later → val,
latest → test.

## 2. Why Group by match_id

Every `match_id` may generate multiple cutoff samples (T-90, T-60, T-30).
All cutoff variants share the **same final label** (euro_result, asian_result).
If `match_001__cutoff_90` goes to train and `match_001__cutoff_30` goes to test,
the model sees the label during training — leakage.

**Rule**: all cutoff samples from the same match_id MUST be in the same split.

## 3. Split Implementation

`dataset/odds_split.py`:
- `load_match_metadata(jsonl_path)` → list of match dicts sorted by time
- `grouped_time_split(matches, ratios...)` → {train_ids, val_ids, test_ids}
- `load_match_ids_from_file(path)` → set of match IDs

Builder CLI: `tools/odds_build_splits.py`

## 4. Split File Format

```
train_match_ids.txt   — one match_id per line
val_match_ids.txt
test_match_ids.txt
split_summary.json    — counts, time ranges, overlap check
```

## 5. Supported Metrics (Pure PyTorch, No NumPy)

| Metric | Function |
|---|---|
| Accuracy | `accuracy_from_logits(logits, labels)` |
| Log-loss | `logloss_from_logits(logits, labels)` |
| Brier score | `brier_from_logits(logits, labels, num_classes)` |
| Class counts | `class_counts(labels, num_classes)` |
| Prediction distribution | `prediction_counts(logits, num_classes)` |
| By-cutoff breakdown | `evaluate_by_cutoff(model, dataset, ...)` |

## 6. Usage

```bash
# Build splits
python tools/odds_build_splits.py --data ...5class.jsonl --out-dir splits/

# Train with split
python trainer/train_odds_supervised.py \
  --train-match-ids splits/train_match_ids.txt \
  --val-match-ids splits/val_match_ids.txt --eval-every-epoch ...

# Evaluate on test split
python eval/eval_odds_model.py \
  --split-match-ids splits/test_match_ids.txt \
  --cutoffs 90,60,30 --cutoff-mode exhaustive --asian-label-mode 5class ...
```

## 7. No Real Data

All splits and evaluations use hand-crafted fixture data only.

## 8. Next Phase

**P0.5 — LightGBM / Simple Baseline Comparison**
