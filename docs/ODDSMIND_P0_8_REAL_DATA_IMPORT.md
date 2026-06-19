# OddsMind P0.8 — Real Data Import Adapter

## 1. Purpose

Convert user-local CSV match data into OddsMind canonical JSONL format.
No scraping, no networking, no real data bundled.

## 2. Supported Presets

| Preset | Source |
|---|---|
| `football_data_b365` | Football-Data.co.uk style CSV (B365 odds) |
| `generic_single_snapshot` | Flat CSV with euro_h/euro_d/euro_a etc. |

## 3. Usage

```bash
# Dry-run (no files written)
python tools/odds_import_csv.py --input raw.csv --preset football_data_b365 ...

# Write JSONL + report
python tools/odds_import_csv.py --input raw.csv ... --write --out out.jsonl --report report.json
```

## 4. upper_side / asian_line

`--upper-side home` treats home as the upper side. Asian handicap labels
are computed from the imported data using the canonical P0.3 settlement logic.

## 5. Imported JSONL Compatibility

Output JSONL is directly consumable by:
- `OddsDataset` (including cutoff exhaustive)
- `eval_odds_model.py`
- `train_odds_supervised.py`
- `predict_odds_match.py`

## 6. Next Phase

**P0.9 — Real Odds Dataset Dry Run**
