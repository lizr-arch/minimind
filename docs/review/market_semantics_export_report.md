# Market Semantics Export Report — P1.1A

> Generated: 2026-06-24T16:38:02.188166

---

## 1. Summary

| Metric | Value |
|--------|-------|
| Total matches | 13058 |
| With kickoff | 5806 |
| Without kickoff | 9505 |
| Valid samples | 3518 |
| Invalid samples | 7252 |

## 2. Raw Event Counts

| Market | Count |
|--------|-------|
| 1x2 | 321436 |
| handicap | 431798 |
| overunder | 332096 |
| Total | 1085330 |

## 3. Exported Event Source Distribution

| Source | euro | asian | over_under |
|--------|------|-------|------------|
| raw_update | 126039 | 13535 | 5836 |
| forward_fill | 0 | 109844 | 102900 |
| missing | 0 | 2660 | 17303 |

## 4. Market Presence in Exported Events

| Market | has=true count | Total events |
|--------|---------------|--------------|
| euro | 126039 | 126039 |
| asian | 123379 | 126039 |
| over_under | 108736 | 126039 |

## 5. asian_line=0 Analysis

| Condition | Count |
|-----------|-------|
| asian_line=0 AND has_asian=true (real flat handicap) | 57521 |
| asian_line=0 AND has_asian=false (missing placeholder) | 2660 |

## 6. Asian Label Status

| Status | Count |
|--------|-------|
| ok | 3459 |
| missing_handicap | 59 |

## 7. Event Time Stats

| Metric | Value |
|--------|-------|
| Recovered events | 126039 |
| Post-kickoff skipped | 0 |
| Bad change_time | 195397 |

## 8. Verification

- has_asian=true events > 0: PASS
- has_over_under=true events > 0: PASS
- asian_source=raw_update > 0: PASS
- ou_source=raw_update > 0: PASS
- asian_line=0 AND has_asian=true can exist: PASS
- asian_line=0 AND has_asian=false can exist: PASS
