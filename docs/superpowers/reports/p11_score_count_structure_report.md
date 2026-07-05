# P11 Score-Count Structure Probe Report

## Verdict

`P11_SCORE_HEAD_LEARNS_BUT_NO_BRIDGE`

The Poisson goal-count head learned useful score structure, but the detached
score-derived 1X2 consistency loss did not move the final 1X2 draw behavior
enough to become a mainline candidate.

## Protocol

- Runs: 15/15 complete
- Control: 3 seeds of unchanged P6 Euro default
- Candidates: 4 fixed P11 variants x 3 seeds
- Seeds: 42, 123, 2025
- Test split: not used
- Validation posthoc fit: not used
- Score label coverage: train 1.0, val 1.0
- Artifacts: 15 reports, 15 logs, 15 best checkpoints, 12 score prediction CSVs

## Results

| Model | logloss | ECE | draw NLL | draw top2 | p(draw) true draw | draw margin | count NLL | score 1X2 logloss | score draw top2 | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P11 control P6 | 0.937336 | 0.027492 | 1.644268 | 0.306884 | 0.198023 | 0.397617 | - | - | - | control |
| P11 A count aux 0.10 | 0.937483 | 0.034169 | 1.647675 | 0.323913 | 0.197568 | 0.401986 | 1.488117 | 0.951239 | 0.455435 | SCORE_HEAD_LEARNS_BUT_NO_BRIDGE |
| P11 B count + KL 0.005 | 0.937477 | 0.034323 | 1.647163 | 0.324275 | 0.197659 | 0.401812 | 1.488157 | 0.951289 | 0.455797 | SCORE_HEAD_LEARNS_BUT_NO_BRIDGE |
| P11 C count + KL 0.020 | 0.937458 | 0.034308 | 1.645650 | 0.326812 | 0.197930 | 0.401298 | 1.488261 | 0.951418 | 0.455797 | SCORE_HEAD_LEARNS_BUT_NO_BRIDGE |
| P11 D close KL 0.010 | 0.937476 | 0.034319 | 1.646779 | 0.324275 | 0.197735 | 0.401702 | 1.488190 | 0.951328 | 0.456159 | SCORE_HEAD_LEARNS_BUT_NO_BRIDGE |

Baselines:

- Constant train mean-rate count NLL: 1.560327
- Constant class-prior 1X2 logloss: 1.039944

All P11 variants beat the constant score/count baselines:

- Count NLL improved by about 0.072
- Score-derived 1X2 logloss improved by about 0.089
- Score-derived draw top2 reached about 0.455

But final 1X2 did not pass the draw gates:

- Final draw top2 stayed around 0.324, far below the 0.40 gate
- Final p(draw) on true draws stayed around 0.198, below the 0.210 gate
- Final draw NLL stayed worse than the 1.609 gate
- ECE worsened from 0.0275 control to about 0.0343

## Interpretation

P11 answered the key question: a small goal-count head can learn meaningful
score structure from the current features. The score-derived probabilities are
substantially better than class prior and identify draw as top2 much more often
than the final 1X2 head.

The failure is in the bridge. Detached KL from score-derived 1X2 into final 1X2
did not transfer enough draw mass, and increasing KL from 0.005 to 0.020 barely
changed the outcome. This suggests the final residual 1X2 objective is still
dominated by the Euro anchor / CE optimum, while the score-count head learns in
parallel.

## Recommended Next Step

Do not promote P11 as-is.

Recommended P12 direction:

1. Keep the score-count head as a diagnostic teacher.
2. Freeze or detach the final 1X2 path in a probe and inspect where score-derived
   draw mass conflicts with Euro anchor residuals.
3. Try a bridge that changes the final parameterization, not just a weak KL:
   score-derived prior plus residual final logits, with strict calibration gates.
4. If that still fails, stop loss-head variants and move to data/feature work:
   market timing, league segmentation, and draw-specific feature audits.

Primary artifact: `runs/p11_score_count_structure/summary.json`
