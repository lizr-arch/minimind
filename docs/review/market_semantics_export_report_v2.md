# Market Semantics Export Report — P1.1A

> Generated: 2026-06-26T21:54:18.566033

---

## 1. Summary

| Metric | Value |
|--------|-------|
| Total matches | 40626 |
| With kickoff | 40478 |
| Without kickoff | 2397 |
| Valid samples | 38047 |
| Invalid samples | 148 |

## 2. Raw Event Counts

| Market | Count |
|--------|-------|
| 1x2 | 3165544 |
| handicap | 4121522 |
| overunder | 2239231 |
| Total | 9526297 |

## 3. Exported Event Source Distribution

| Source | euro | asian | over_under |
|--------|------|-------|------------|
| raw_update | 1252919 | 160337 | 50679 |
| forward_fill | 0 | 1067920 | 879364 |
| missing | 0 | 24662 | 322876 |

## 4. Market Presence in Exported Events

| Market | has=true count | Total events |
|--------|---------------|--------------|
| euro | 1252919 | 1252919 |
| asian | 1228257 | 1252919 |
| over_under | 930043 | 1252919 |

## 5. asian_line=0 Analysis

| Condition | Count |
|-----------|-------|
| asian_line=0 AND has_asian=true (real flat handicap) | 527717 |
| asian_line=0 AND has_asian=false (missing placeholder) | 24662 |

## 6. Asian Label Status

| Status | Count |
|--------|-------|
| ok | 37484 |
| missing_handicap | 563 |

## 7. Event Time Stats

| Metric | Value |
|--------|-------|
| Recovered events | 1252919 |
| Post-kickoff skipped | 0 |
| Bad change_time | 1912625 |

## 8. Verification

- has_asian=true events > 0: PASS
- has_over_under=true events > 0: PASS
- asian_source=raw_update > 0: PASS
- ou_source=raw_update > 0: PASS
- asian_line=0 AND has_asian=true can exist: PASS
- asian_line=0 AND has_asian=false can exist: PASS
