# Market Semantics Export Report — P1.1A

> Generated: 2026-06-24T01:29:24.988227

---

## 1. Summary

| Metric | Value |
|--------|-------|
| Total matches | 5419 |
| With kickoff | 1500 |
| Without kickoff | 3919 |
| Valid samples | 1491 |
| Invalid samples | 3919 |

## 2. Raw Event Counts

| Market | Count |
|--------|-------|
| 1x2 | 144259 |
| handicap | 205359 |
| overunder | 199750 |
| Total | 549368 |

## 3. Exported Event Source Distribution

| Source | euro | asian | over_under |
|--------|------|-------|------------|
| raw_update | 54631 | 5521 | 2927 |
| forward_fill | 0 | 47659 | 50169 |
| missing | 0 | 1451 | 1535 |

## 4. Market Presence in Exported Events

| Market | has=true count | Total events |
|--------|---------------|--------------|
| euro | 54631 | 54631 |
| asian | 53180 | 54631 |
| over_under | 53096 | 54631 |

## 5. asian_line=0 Analysis

| Condition | Count |
|-----------|-------|
| asian_line=0 AND has_asian=true (real flat handicap) | 21644 |
| asian_line=0 AND has_asian=false (missing placeholder) | 1451 |

## 6. Asian Label Status

| Status | Count |
|--------|-------|
| ok | 1464 |
| missing_handicap | 27 |

## 7. Event Time Stats

| Metric | Value |
|--------|-------|
| Recovered events | 54631 |
| Post-kickoff skipped | 0 |
| Bad change_time | 88180 |

## 8. Verification

- has_asian=true events > 0: PASS
- has_over_under=true events > 0: PASS
- asian_source=raw_update > 0: PASS
- ou_source=raw_update > 0: PASS
- asian_line=0 AND has_asian=true can exist: PASS
- asian_line=0 AND has_asian=false can exist: PASS
