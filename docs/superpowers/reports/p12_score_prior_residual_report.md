# P12 Explicit Score-Prior Residual Report

## Verdict

`P12_REJECT_EXPLICIT_PRIOR_MOVE_TO_DATA_FEATURES`

P12 tested whether score-derived 1X2 probabilities can work as an explicit
prior with a learned residual final head. The score/count structure again
learned useful signal, but explicit prior parameterization made final logloss
materially worse than the P6 control and did not improve final draw behavior.

## Protocol

- Runs: 15/15 complete
- Control: 3 seeds of same-protocol P6 Euro default
- Candidates: 4 fixed P12 variants x 3 seeds
- Seeds: 42, 123, 2025
- Test split: not used
- Validation posthoc fit: not used
- Score label coverage: train 1.0, val 1.0
- Artifacts: 15 reports, 15 logs, 15 best checkpoints, 12 prior prediction CSVs

## Results

| Model | logloss | ECE | draw NLL | draw top2 | p(draw) true draw | draw margin | count NLL | score draw top2 | prior logloss | final-prior logloss | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P12 control P6 | 0.937336 | 0.027492 | 1.644268 | 0.306884 | 0.198023 | 0.397617 | - | - | - | - | control |
| P12 A score prior residual | 0.941805 | 0.021501 | 1.622771 | 0.313406 | 0.200919 | 0.386769 | 1.482011 | 0.423913 | 0.943084 | -0.001280 | NO_ACCEPTED_VARIANT |
| P12 B score prior + residual L2 | 0.941776 | 0.021824 | 1.620914 | 0.314130 | 0.201277 | 0.386147 | 1.482007 | 0.423913 | 0.943076 | -0.001300 | NO_ACCEPTED_VARIANT |
| P12 C anchor blend 0.75 | 0.941630 | 0.022649 | 1.625668 | 0.294928 | 0.199924 | 0.385390 | 1.482617 | 0.435870 | 0.941517 | 0.000113 | NO_ACCEPTED_VARIANT |
| P12 D anchor blend 0.90 | 0.941641 | 0.024380 | 1.620947 | 0.301087 | 0.201102 | 0.383923 | 1.482612 | 0.437681 | 0.941412 | 0.000229 | NO_ACCEPTED_VARIANT |

Baselines:

- Constant train mean-rate count NLL: 1.560327
- Constant class-prior 1X2 logloss: 1.039944

## Interpretation

The score/count head remained sane:

- Count NLL improved from the constant mean-rate baseline by about 0.078.
- Score-derived 1X2 logloss remained far better than class prior.
- Tail mass and lambda saturation gates passed.

The explicit-prior bridge failed:

- All candidates were about 0.0043 to 0.0045 worse than P12 P6 control logloss.
- No candidate reached the draw top2 gate of 0.40.
- Final draw top2 stayed near 0.29 to 0.31.
- Final p(draw) on true draws stayed near 0.20.
- Final residual improved pure score-prior logloss by only about 0.0013, below the 0.003 residual bridge gate.
- Anchor blends improved calibration relative to P6 but undercut draw top2 and still lost too much logloss.

This supports the P11 finding: score structure exists, but using it as a
score-derived final prior does not produce a competitive 1X2 model under the
current features and labels.

## Recommendation

Stop loss/head variants for this branch.

Next work should move to data and feature diagnostics:

1. Audit match/market segments where score-derived draw probability is high but
   final label is not draw, and where true draws have low market draw probability.
2. Analyze league/time/market coverage slices for draw calibration drift.
3. Inspect whether Euro close odds already encode most 1X2 logloss and whether
   Asian/OU features mainly help margin/goal-structure diagnostics rather than
   final 1X2 prediction.
4. Treat future model work as feature/data-driven, not another draw-loss or
   score-prior bridge attempt.

Primary artifact: `runs/p12_score_prior_residual/summary.json`
