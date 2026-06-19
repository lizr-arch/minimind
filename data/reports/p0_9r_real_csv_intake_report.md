# P0.9R OddsMind Real CSV Intake Run Final Report

## 1. Overall

**PASS_WITH_NOTES** — 380 matches imported, 0 errors, full pipeline (import → audit → split → baseline → tabular → OddsMind → comparison) completed. Zero MiniMind files modified.

## 2. Real CSV

- Path: `data/raw_csv/football_data/E0_2425.csv`
- Exists: Yes (copied from external path)
- Rows: 380
- League: E0 (Premier League)
- Season: 2024-25
- Bookmaker: Bet365

## 3. Header / Field Check

All required fields present. Open + closing euro + asian fields confirmed.

## 4. Asian Handicap Semantic

- AHh/AHCh = home handicap
- B365AHH/B365CAHH = home price
- B365AHA/B365CAHA = away price
- Canonical mapping: upper_side=home, upper_water=home_price, lower_water=away_price

## 5. Import Result

- 380 records written to `data/odds_real/E0_2024-25_bet365.jsonl`
- Timeline type: open+closing (2 events each)
- 0 skipped, 0 errors

## 6. Data Quality

- 380 matches, 0 schema errors, 0 invalid odds
- All dual-timeline (open @ 10080 min, closing @ 0 min)

## 7. Split

- Train: 266, Val: 57, Test: 57
- Time range: 2024-08-16 to 2025-05-25 (281 days)
- Overlap: pass

## 8. Dry Run Results (Test Split, 57 matches)

| Model | Euro Acc | Euro LogLoss | Asian Acc | Asian LogLoss |
|---|---|---|---|---|
| Deterministic (implied prob) | 61.4% | 0.938 | - | - |
| Deterministic (low water) | - | - | 35.1% | 17.9 |
| Tabular TinyMLP (1 epoch) | 43.9% | 7.40 | 45.6% | 15.0 |
| **OddsMind (1 epoch)** | **52.6%** | **1.08** | **49.1%** | **1.32** |

## 9. Commands

See full command log in P0.9R output.

## 10. Known Issues

- football-data CSV has open/close snapshots only — no true timestamped odds movement
- `--cutoffs 0` used because T-90/T-60/T-30 are not meaningful for this CSV
- Metrics are 1-epoch dry-run, not formal prediction quality
- AH fields are home handicap, mapped to canonical upper_side=home
- Model bias toward single class (55/57 predicted as upper_full_loss) — needs more training

## 11. Boundary Check

All constraints satisfied. Zero MiniMind files modified.

## 12. Next Phase

**P1.0 First Real Backtest**
