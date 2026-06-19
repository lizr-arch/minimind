# OddsMind P0.2 — Random Cutoff Sample Generation

## 1. Why Cutoff Samples

In real-world odds prediction, you don't know the full pre-match timeline at
prediction time.  At T-90 minutes before kickoff you only see odds published
up to that point.  At T-30 you have more information (the odds movement
between T-90 and T-30).

Training with a fixed cutoff fails to teach the model about this progressive
information reveal.  **Cutoff sample generation** creates multiple training
examples from a single match, each simulating the information set available
at a different pre-kickoff moment.

## 2. `minutes_before_kickoff` Semantics

- **Larger value** = further from kickoff (earlier in time).
- **Smaller value** = closer to kickoff (later in time).
- `cutoff=90` means: "only use data from 90+ minutes before kickoff".
- Events with `minutes_before_kickoff = 30` MUST NOT appear when `cutoff=60`.

The timeline is sorted descending: `[10080, 4320, 1440, 180, 90, 60, 30, ...]`

## 3. Cutoff Filtering Rules

```
cutoff = 90:  input = all events where minutes_before_kickoff >= 90
cutoff = 60:  input = all events where minutes_before_kickoff >= 60
cutoff = 30:  input = all events where minutes_before_kickoff >= 30
```

The label (euro_result, asian_result) never changes — it is the final match
outcome regardless of cutoff.

## 4. Exhaustive vs Random Cutoff

| Mode | Behavior | Use case |
|---|---|---|
| `none` | One sample per JSONL line (P0.1). Uses per-sample `cutoff_minutes` field. | Backward compatibility |
| `exhaustive` | Expands each match into N samples (one per valid cutoff). All samples live in the dataset at init time. | Fixed-size dataset, reproducible |
| `random` | Each `__getitem__` randomly picks a valid cutoff. Dataset length = number of raw matches. | Infinite variety, each epoch sees different cutoffs |

## 5. Avoiding Future-Information Leakage

**Within a sample:** The `filter_timeline_by_cutoff` function guarantees that
no event with `minutes_before_kickoff < cutoff_minutes` enters the input.

**Across train/test splits:** All cutoff samples derived from the same match
MUST go into the same split. This is NOT enforced in P0.2 — it will be
implemented in P0.4 via `GroupedTimeSeriesSplit`.

For now, when evaluating with exhaustive cutoff mode:
- Ensure the entire dataset is used for smoke testing only (no train/test split).
- In production, split by `source_match_id` before expanding cutoffs.

## 6. Files Added / Modified

### Added

| File | Purpose |
|---|---|
| `dataset/odds_cutoff.py` | `filter_timeline_by_cutoff`, `build_cutoff_samples`, `build_exhaustive_cutoff_dataset` |
| `eval/eval_odds_cutoff_smoke.py` | 9 cutoff-specific smoke tests |
| `docs/ODDSMIND_P0_2_CUTOFF.md` | This file |

### Modified

| File | Change |
|---|---|
| `dataset/odds_dataset.py` | Added `cutoffs`, `cutoff_mode`, `min_events` params; exhaustive/random modes; backward-compatible |
| `trainer/train_odds_supervised.py` | Added `--cutoffs`, `--cutoff-mode`, `--min-events` args; cutoff stats logging |
| `.gitignore` | Added `runs/` to prevent smoke checkpoints from being committed |

**Zero MiniMind original files modified.**

## 7. Smoke Commands

```bash
# P0.1 backward compatibility
python eval/eval_odds_smoke.py

# P0.2 cutoff smoke tests
python eval/eval_odds_cutoff_smoke.py

# P0.2 exhaustive cutoff training
python trainer/train_odds_supervised.py \
  --data data/odds_fixtures/sample_odds_matches.jsonl \
  --epochs 1 --batch-size 2 --hidden-size 64 \
  --num-layers 2 --num-heads 4 --device cpu \
  --cutoffs 90,60,30 --cutoff-mode exhaustive \
  --out-dir runs/oddsmind_p0_2_cutoff_smoke

# Inference (unchanged, still works)
python infer/predict_odds_match.py \
  --input data/odds_fixtures/sample_odds_matches.jsonl \
  --untrained --hidden-size 64 --num-layers 2 --num-heads 4 --device cpu
```

## 8. Next Phase

**P0.3 — Asian Handicap 5-Class Labeling**

Extend `AsianResultHead` from 3 classes (upper/push/lower) to 5 classes:
`upper_full_win / upper_half_win / push / upper_half_loss / upper_full_loss`.
