# Label Mapping Report — P1.0B

> Input: `D:/code/git/betmind/minimind/data/odds_real/titan007_pure_v3_time.jsonl`
> Total samples: 1490
> Asian label mode: 3class

---

## 1. Euro Label Distribution

| Label | Count | Pct |
|-------|-------|-----|
| home | 653 | 43.8% |
| draw | 313 | 21.0% |
| away | 524 | 35.2% |

## 2. Asian Raw Label Distribution

| Label | Count | Pct |
|-------|-------|-----|
| full_win | 653 | 43.8% |
| full_loss | 524 | 35.2% |
| push | 313 | 21.0% |

## 3. Asian 3-Class Distribution

| Label | Count | Pct |
|-------|-------|-----|
| upper | 653 | 43.8% |
| push | 313 | 21.0% |
| lower | 524 | 35.2% |

## 4. Unknown Labels

- Unknown euro labels: 0
- Unknown asian labels: 0
- Null/empty labels: 0

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
