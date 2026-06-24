# Label Mapping Report — P1.0B

> Input: `data/odds_real/titan007_pure_v4_market_semantics.jsonl`
> Total samples: 1491
> Asian label mode: 3class

---

## 1. Euro Label Distribution

| Label | Count | Pct |
|-------|-------|-----|
| home | 639 | 42.9% |
| draw | 309 | 20.7% |
| away | 516 | 34.6% |

## 2. Asian Raw Label Distribution

| Label | Count | Pct |
|-------|-------|-----|
| full_loss | 703 | 47.1% |
| full_win | 504 | 33.8% |
| push | 145 | 9.7% |
| half_loss | 75 | 5.0% |
| half_win | 37 | 2.5% |

## 3. Asian 3-Class Distribution

| Label | Count | Pct |
|-------|-------|-----|
| upper | 541 | 36.3% |
| push | 145 | 9.7% |
| lower | 778 | 52.2% |

## 4. Unknown Labels

- Unknown euro labels: 0
- Unknown asian labels: 0
- Null/empty labels: 27

## 4b. Asian Label Status

| Status | Count | Pct |
|--------|-------|-----|
| ok | 1464 | 98.2% |
| missing_handicap | 27 | 1.8% |

## 5. Mapping Tables

### Euro (3-class)

| Raw | ID |
|-----|-----|
| home | 0 |
| draw | 1 |
| away | 2 |

### Asian 3-Class

| Raw | ID |
|-----|-----|
| upper | 0 |
| push | 1 |
| lower | 2 |

### Asian 5-Class

| Raw | ID |
|-----|-----|
| full_win | 0 |
| half_win | 1 |
| push | 2 |
| half_loss | 3 |
| full_loss | 4 |

### Asian 5→3 Collapse

| 5-Class | 3-Class |
|---------|---------|
| full_win | upper |
| half_win | upper |
| push | push |
| half_loss | lower |
| full_loss | lower |

## 6. Notes

- **Default training mode**: 3class
- **Why 3class**: Sample count is modest (~1500). half_win/half_loss may be sparse.
- **5class available**: `--asian-label-mode 5class` for finer granularity.
- **No silent fallback**: Unknown labels raise ValueError at encode time.
- **P1.0C allowed**: Time-based train/val/test split repair.
