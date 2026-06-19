# OddsMind P0.3 — Asian Handicap 5-Class Labeling

## 1. Why 5-Class Labels

Standard Asian handicap betting has five possible outcomes per bet:

| Class | Meaning | Example |
|---|---|---|
| `upper_full_win`  | Both half-lines win    | Upper -0.5, win by 1 |
| `upper_half_win`  | One half wins, one pushes | Upper -0.75, win by 1 |
| `push`            | Both halves push (or line=0 draw) | Upper -1.0, win by 1 |
| `upper_half_loss` | One half loses, one pushes | Upper -1.25, win by 1 |
| `upper_full_loss` | Both half-lines lose   | Upper -0.5, draw |

The previous 3-class system collapsed `upper_full_win` + `upper_half_win` →
`upper`, and `upper_half_loss` + `upper_full_loss` → `lower`, losing
important granularity.

## 2. `asian_line` Canonical Semantics

`asian_line` always represents the handicap **applied to the upper side team**.

```
asian_line = upper side handicap line
```

- `asian_line = -0.5`: upper team gives 0.5 goal head start to lower team
- `asian_line = +0.25`: upper team receives 0.25 goal head start
- `asian_line = 0.0`: level ball (no handicap)

## 3. Settlement from Upper Side Perspective

```python
raw_margin = upper_goals - lower_goals
adjusted   = raw_margin + asian_line
```

For quarter lines, the line is split:
```
-0.25 → [0.0, -0.5]
+0.75 → [+1.0, +0.5]
```

Each half-line settles independently (win/push/loss), then combined:

| Combination | 5-Class |
|---|---|
| win + win   | upper_full_win |
| win + push  | upper_half_win |
| push + win  | upper_half_win |
| push + push | push |
| loss + push | upper_half_loss |
| push + loss | upper_half_loss |
| loss + loss | upper_full_loss |

## 4. 5-Class Label Map

| Index | Label |
|---|---|
| 0 | upper_full_win |
| 1 | upper_half_win |
| 2 | push |
| 3 | upper_half_loss |
| 4 | upper_full_loss |

## 5. 3-Class Compatibility

Existing 3-class mode maps 5-class labels as:
```
upper_full_win, upper_half_win  → upper (0)
push                             → push  (1)
upper_half_loss, upper_full_loss → lower (2)
```

`asian_label_mode="3class"` is the default; `"5class"` enables granular output.

## 6. Dataset Usage

```python
# 5class mode
ds = OddsDataset("data/odds_fixtures/sample_odds_matches_5class.jsonl",
                 asian_label_mode="5class")
```

Labels in JSONL must use 5-class strings:
```json
"label": {"euro_result": "home", "asian_result": "upper_half_win"}
```

## 7. Training & Inference

```bash
# Train with 5class labels
python trainer/train_odds_supervised.py \
  --data data/odds_fixtures/sample_odds_matches_5class.jsonl \
  --asian-label-mode 5class --cutoffs 90,60,30 --cutoff-mode exhaustive \
  ...

# Inference
python infer/predict_odds_match.py \
  --input data/odds_fixtures/sample_odds_matches_5class.jsonl \
  --asian-label-mode 5class --untrained ...
```

## 8. No Real Data

All fixture data is hand-crafted. No real match results or odds from any
bookmaker are included. The settlement logic is verified against manually
computed canonical test cases.

## 9. Next Phase

**P0.4 — Time-Based Grouped Split + Evaluation Metrics**
