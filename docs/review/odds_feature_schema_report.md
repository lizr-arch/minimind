# Odds Feature Schema Report

Generated: 2026-06-24T01:09:00.670833

Input: `data/odds_real/titan007_pure_v3_time.jsonl`
Splits dir: `data/odds_real/splits/v3_time`
Feature schema: `v4`

## 1. Schema Versions
| Schema | Features | Dim | Description |
|--------|----------|-----|-------------|
| v1 | FEATURE_KEYS only | 10 | Raw odds + minutes |
| v2 | + has_euro/has_asian/has_ou | 13 | + market availability flags |
| v3 | same as v2 + missing_mask | 13 + mask | + per-feature missing mask |
| v4 | raw + implied + no-vig + overround | 32 | Full probability features |

## 2. Current Schema: v4
- **Feature dim**: 32
- **Total events analyzed**: 54614

## 3. Feature List
| Index | Name | Source | Missing Handling |
|-------|------|--------|------------------|
| 0 | minutes_before_kickoff | raw | missing_mask |
| 1 | euro_h | raw | missing_mask |
| 2 | euro_d | raw | missing_mask |
| 3 | euro_a | raw | missing_mask |
| 4 | euro_h_implied | computed | safe default (0.0 or 1/3) |
| 5 | euro_d_implied | computed | safe default (0.0 or 1/3) |
| 6 | euro_a_implied | computed | safe default (0.0 or 1/3) |
| 7 | euro_overround | computed | safe default (0.0 or 1/3) |
| 8 | euro_h_novig | computed | safe default (0.0 or 1/3) |
| 9 | euro_d_novig | computed | safe default (0.0 or 1/3) |
| 10 | euro_a_novig | computed | safe default (0.0 or 1/3) |
| 11 | has_euro | detection | missing_mask |
| 12 | asian_line | raw | missing_mask |
| 13 | upper_water | raw | missing_mask |
| 14 | lower_water | raw | missing_mask |
| 15 | asian_upper_implied | computed | safe default (0.0 or 1/3) |
| 16 | asian_lower_implied | computed | safe default (0.0 or 1/3) |
| 17 | asian_overround | computed | safe default (0.0 or 1/3) |
| 18 | asian_upper_novig | computed | safe default (0.0 or 1/3) |
| 19 | asian_lower_novig | computed | safe default (0.0 or 1/3) |
| 20 | asian_water_spread | computed | safe default (0.0 or 1/3) |
| 21 | has_asian | detection | missing_mask |
| 22 | over_under_line | raw | missing_mask |
| 23 | over_water | raw | missing_mask |
| 24 | under_water | raw | missing_mask |
| 25 | ou_over_implied | computed | safe default (0.0 or 1/3) |
| 26 | ou_under_implied | computed | safe default (0.0 or 1/3) |
| 27 | ou_overround | computed | safe default (0.0 or 1/3) |
| 28 | ou_over_novig | computed | safe default (0.0 or 1/3) |
| 29 | ou_under_novig | computed | safe default (0.0 or 1/3) |
| 30 | ou_water_spread | computed | safe default (0.0 or 1/3) |
| 31 | has_over_under | detection | missing_mask |

## 4. Finite Check Results
- Total values checked: 1747648
- Finite: **1747648**
- NaN: **0**
- Inf: **0**
- **PASS**: All features are finite

## 5. Per-Feature Statistics
| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| minutes_before_kickoff | 2686.2205 | 5266.0172 | 0.0000 | 85512.0000 |
| euro_h | 2.6790 | 1.4461 | 1.0100 | 18.0000 |
| euro_d | 3.9519 | 0.9347 | 2.4900 | 15.0000 |
| euro_a | 3.9181 | 2.6305 | 1.0200 | 51.0000 |
| euro_h_implied | 0.4299 | 0.1630 | 0.0493 | 0.9184 |
| euro_d_implied | 0.2507 | 0.0434 | 0.0628 | 0.3798 |
| euro_a_implied | 0.3193 | 0.1500 | 0.0184 | 0.8700 |
| euro_overround | 0.0495 | 0.0142 | 0.0149 | 0.1656 |
| euro_h_novig | 0.4299 | 0.1630 | 0.0493 | 0.9184 |
| euro_d_novig | 0.2507 | 0.0434 | 0.0628 | 0.3798 |
| euro_a_novig | 0.3193 | 0.1500 | 0.0184 | 0.8700 |
| has_euro | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| asian_line | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| upper_water | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| lower_water | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| asian_upper_implied | 0.5000 | 0.0000 | 0.5000 | 0.5000 |
| asian_lower_implied | 0.5000 | 0.0000 | 0.5000 | 0.5000 |
| asian_overround | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| asian_upper_novig | 0.5000 | 0.0000 | 0.5000 | 0.5000 |
| asian_lower_novig | 0.5000 | 0.0000 | 0.5000 | 0.5000 |
| asian_water_spread | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| has_asian | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| over_under_line | 2.5000 | 0.0000 | 2.5000 | 2.5000 |
| over_water | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| under_water | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| ou_over_implied | 0.5000 | 0.0000 | 0.5000 | 0.5000 |
| ou_under_implied | 0.5000 | 0.0000 | 0.5000 | 0.5000 |
| ou_overround | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ou_over_novig | 0.5000 | 0.0000 | 0.5000 | 0.5000 |
| ou_under_novig | 0.5000 | 0.0000 | 0.5000 | 0.5000 |
| ou_water_spread | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| has_over_under | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## 6. Example v4 Features (first event)
| Feature | Value |
|---------|-------|
| minutes_before_kickoff | 5902.000000 |
| euro_h | 2.250000 |
| euro_d | 3.500000 |
| euro_a | 2.750000 |
| euro_h_implied | 0.406332 |
| euro_d_implied | 0.261214 |
| euro_a_implied | 0.332454 |
| euro_overround | 0.093795 |
| euro_h_novig | 0.406332 |
| euro_d_novig | 0.261214 |
| euro_a_novig | 0.332454 |
| has_euro | 1.000000 |
| asian_line | 0.000000 |
| upper_water | 1.000000 |
| lower_water | 1.000000 |
| asian_upper_implied | 0.500000 |
| asian_lower_implied | 0.500000 |
| asian_overround | 0.000000 |
| asian_upper_novig | 0.500000 |
| asian_lower_novig | 0.500000 |
| asian_water_spread | 0.000000 |
| has_asian | 0.000000 |
| over_under_line | 2.500000 |
| over_water | 1.000000 |
| under_water | 1.000000 |
| ou_over_implied | 0.500000 |
| ou_under_implied | 0.500000 |
| ou_overround | 0.000000 |
| ou_over_novig | 0.500000 |
| ou_under_novig | 0.500000 |
| ou_water_spread | 0.000000 |
| has_over_under | 0.000000 |

## 7. Implied / No-vig / Overround Examples
- **euro_h_implied**: 0.406332
- **euro_d_implied**: 0.261214
- **euro_a_implied**: 0.332454
- **euro_overround**: 0.093795
- **euro_h_novig**: 0.406332

## 8. Per-Split Finite Check
| Split | Matches | Events | NaN | Inf | Finite |
|-------|---------|--------|-----|-----|--------|
| train | 1031 | 37271 | 0 | 0 | 1192672 |
| val | 221 | 8291 | 0 | 0 | 265312 |
| test | 221 | 9052 | 0 | 0 | 289664 |

## 9. NaN / Inf Detection
- **No NaN or Inf detected** in any feature

## 10. Recommendation

**v4 schema is safe for training use.**
- All features are finite
- Missing values use safe defaults (0.0 for odds, 1/3 for probabilities)
- Missing mask correctly identifies placeholder values
- Implied/no-vig/overround features add probabilistic interpretation

**Verdict: PASS**

---
*Report generated by `tools/report_odds_features.py`*