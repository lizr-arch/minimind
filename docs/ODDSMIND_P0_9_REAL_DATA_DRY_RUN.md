# OddsMind P0.9 — Real Odds Dataset Dry Run

## 1. Purpose

Run the complete OddsMind pipeline on user-provided data to validate
the end-to-end workflow before committing to real training.

## 2. Pipeline

```
local CSV → import → JSONL → quality audit → split →
  deterministic baseline → tabular baseline → OddsMind supervised →
  comparison report
```

## 3. Status

- Fixture-based dry run: **PASS** (15/15 smoke checks)
- Real data dry run: **BLOCKED** — needs user-provided CSV path

## 4. To Run With Real Data

```bash
python tools/odds_real_dataset_dry_run.py \
  --input /path/to/real.csv --is-real \
  --preset football_data_b365 --league-id EPL --bookmaker-id B365 \
  --upper-side home --limit 500 --out-dir runs/p0_9_real
```

## 5. Required User Inputs

1. CSV file path
2. Data source type (preset)
3. league_id, season, bookmaker_id
4. upper_side: home / away / auto_home_handicap
5. Asian line field semantics confirmation

## 6. Next Phase

**P1.0 — First Real Backtest**
