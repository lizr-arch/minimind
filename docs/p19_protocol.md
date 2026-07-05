# P19 Rolling Backtest Protocol

P19 validates the P18 AH high-confidence candidate edge before any test-set
audit. It does not add a new model family, widen the search grid, or use test
artifacts.

## Frozen Candidates

- Baseline: `P18.0_goal_diff_directional_baseline`
- Cover signal: `p18_ahw_003`
- Direct signal: `p18d_direct030`
- Locked strategy: `agreement_both_t0.250`
- Nested-selection grid: thresholds `0.15,0.20,0.25,0.30`, mode `both`

The canonical registry is `configs/p19_candidate_registry.json`.

## Preflight Audits

P19 starts with `tools/p19_rolling_backtest.py audit`.

Required checks:

- no path component named `test`, `test_*`, or `test-*`
- feature-name blacklist rejects target/downstream fields
- AH settlement is recomputed from score, line, and upper side
- duplicate fixture counts are reported so row-level metrics cannot hide
  bookmaker replication

If the audit status is not `PASS`, stop and return to label/feature review.

## Rolling Folds

`tools/p19_rolling_backtest.py build-folds` groups rows by `match_id`, sorts
fixtures by `kickoff_time`, and creates expanding chronological folds:

- `train`: older fixtures
- `selection`: immediately after train, used only for threshold/candidate choice
- `forward`: immediately after selection, used only for scoring

All bookmaker rows for a fixture stay in the same window. A fold is invalid if
any fixture appears in more than one window.

## Main Metrics

Reports must include both row-level and fixture-collapsed values:

- selected rows
- selected unique fixtures
- row average payoff
- fixture average payoff
- fixture total units
- row and fixture direction accuracy excluding push rows
- latest-fold payoff
- positive-fold rate

Fixture-collapsed metrics are the promotion gate. Row-level metrics are only
diagnostic.

## Stop Rules

Stop P19 and do not train further if any of these occur:

- audit status is not `PASS`
- fixture overlap exists inside a fold
- AH settlement recomputation mismatches labels
- feature blacklist hits a model input
- latest fold fixture payoff is below `0.05`
- aggregate fixture payoff is non-positive
- positive fold rate is below `0.70`
- a promoted fold has fewer than `80` selected unique fixtures

Only after P19 passes can the frozen candidate proceed to a one-time final test
audit.
