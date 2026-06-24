# Split Report — P1.0C

> Generated: 2026-06-24T00:08:35.705810
> Input: `D:/code/git/betmind/minimind/data/odds_real/titan007_pure_v3_time.jsonl`

---

## 1. Summary

| Metric | Value |
|--------|-------|
| Total samples | 1490 |
| Unique match_ids | 1473 |
| Invalid match_ids | 0 |

## 2. Split Counts

| Split | Match IDs | Samples |
|-------|-----------|---------|
| train | 1031 | 1036 |
| val | 221 | 221 |
| test | 221 | 233 |

## 3. Time Ranges

| Split | Min kickoff | Max kickoff |
|-------|-------------|-------------|
| train | 2022-04-10T00:30:00 | 2025-02-23T01:30:00 |
| val | 2025-02-23T02:00:00 | 2025-04-12T21:00:00 |
| test | 2025-04-12T21:30:00 | 2025-06-01T03:00:00 |

## 4. Overlap Check

- train ∩ val: 0 PASS
- train ∩ test: 0 PASS
- val ∩ test: 0 PASS

## 5. Train Distribution

### Euro Labels

| Label | Count | Pct |
|-------|-------|-----|
| home | 468 | 45.2% |
| draw | 214 | 20.7% |
| away | 354 | 34.2% |

### Asian Labels

| Label | Count | Pct |
|-------|-------|-----|
| full_win | 468 | 45.2% |
| full_loss | 354 | 34.2% |
| push | 214 | 20.7% |
| half_win | 0 | 0.0% |
| half_loss | 0 | 0.0% |

### Leagues

| League | Count |
|--------|-------|
| epl | 249 |
| bundesliga | 228 |
| laliga | 201 |
| seriea | 189 |
| ligue1 | 169 |

### Bookmakers

| Bookmaker | Count |
|-----------|-------|
| Sbobet | 810 |
| Macau | 107 |
| Bet365 | 62 |
| Pinnacle | 57 |

### Timeline Length

min=1, p25=21, median=35, p75=48, max=196

## 5. Val Distribution

### Euro Labels

| Label | Count | Pct |
|-------|-------|-----|
| home | 93 | 42.1% |
| draw | 49 | 22.2% |
| away | 79 | 35.7% |

### Asian Labels

| Label | Count | Pct |
|-------|-------|-----|
| full_win | 93 | 42.1% |
| full_loss | 79 | 35.7% |
| push | 49 | 22.2% |
| half_win | 0 | 0.0% |
| half_loss | 0 | 0.0% |

### Leagues

| League | Count |
|--------|-------|
| bundesliga | 68 |
| ligue1 | 43 |
| epl | 42 |
| laliga | 42 |
| seriea | 26 |

### Bookmakers

| Bookmaker | Count |
|-----------|-------|
| Sbobet | 204 |
| Bet365 | 11 |
| Macau | 6 |

### Timeline Length

min=1, p25=28, median=37, p75=47, max=94

## 5. Test Distribution

### Euro Labels

| Label | Count | Pct |
|-------|-------|-----|
| home | 92 | 39.5% |
| draw | 50 | 21.5% |
| away | 91 | 39.1% |

### Asian Labels

| Label | Count | Pct |
|-------|-------|-----|
| full_win | 92 | 39.5% |
| full_loss | 91 | 39.1% |
| push | 50 | 21.5% |
| half_win | 0 | 0.0% |
| half_loss | 0 | 0.0% |

### Leagues

| League | Count |
|--------|-------|
| epl | 54 |
| laliga | 52 |
| bundesliga | 45 |
| ligue1 | 41 |
| seriea | 37 |
| coppa_italia | 2 |
| champions_league | 2 |

### Bookmakers

| Bookmaker | Count |
|-----------|-------|
| Sbobet | 213 |
| Bet365 | 10 |
| Pinnacle | 8 |
| Macau | 2 |

### Timeline Length

min=4, p25=25, median=36, p75=47, max=197

## 6. Multi-Bookmaker Leak Check

- Matches with multiple bookmaker samples: 11
- Matches leaking across splits: 0 PASS
