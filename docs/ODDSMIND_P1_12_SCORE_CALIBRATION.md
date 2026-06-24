# P1.12 Score Distribution Calibration

## Status: IN PROGRESS

## Overview

P1.12 applies post-hoc calibration to the independent Poisson score grid
produced by a frozen ScoreHeadV2. The goal is to improve distribution shape
(total/diff variance, extreme scorelines) without retraining any model weights.

## Why Independent Poisson is Under-Dispersed

The ScoreHeadV2 outputs lambda_home and lambda_away, and the independent Poisson
model assumes:

    P(home=i, away=j) = Poisson(i|lambda_home) * Poisson(j|lambda_away)

This assumes zero correlation between home and away goals. In real football,
goals are correlated (e.g., 0-0 and 1-1 are more common than independent Poisson
predicts). This causes:

- pred_total_std ≈ 0.6 vs true ≈ 1.8 (under-dispersion)
- Extreme scorelines (0 goals, 5+ total) have poor probability estimates
- Score-derived 1X2 can be over-confident

## Why This Is Output Calibration, Not Model Retraining

- Backbone:          FROZEN (no gradient)
- Euro/Asian Heads:  FROZEN
- ScoreHeadV2:       FROZEN
- Only calibrator params are fit on val set

The calibrator modifies the output distribution (score grid) without touching
model weights. This is a post-processing step, analogous to Platt scaling
or temperature scaling in classification.

## Calibration Parameters

| Parameter | Description | Range |
|---|---|---|
| total_temperature | Flattens (>1) or sharpens (<1) total-goal distribution | 0.5 - 3.0 |
| diff_temperature | Flattens goal-difference distribution | 0.5 - 3.0 |
| lambda_scale | Multiplies lambda_home/away (shifts mean) | 0.5 - 2.0 |
| dc_rho | Dixon-Coles style low-score correlation | -0.3 - 0.3 |
| tail_boost | Additive mass boost for extreme scorelines | 0.0 - 0.2 |

## Val-Only Fitting Rule

All calibration parameters are searched on the validation set only.
The test set is evaluated exactly once with the best config.

## Test-Once Rule

The best calibration config from val search is applied to test set.
No iterative tuning on test set.

## Dixon-Coles Style Correction Limitations

This implementation is a simplified Dixon-Coles style correction,
NOT a full fitted Dixon-Coles (1997) bivariate Poisson model.

The full model would:
- Fit rho via MLE
- Model shared covariance between home and away goals
- Produce proper bivariate Poisson probabilities

Our approximation:
- Applies heuristic correction factors to 0-0, 1-0, 0-1, 1-1 cells
- Controlled by a single dc_rho parameter
- Correct for qualitative behavior (rho>0 → more 0-0 and 1-1)

## Disagreement Policy

The Euro-vs-Score disagreement policy categorizes model consistency:

| Level | Condition | Adjustment |
|---|---|---|
| high_agreement | Same winner, JS < 0.10 | keep |
| same_winner_high_js | Same winner, JS >= 0.10 | soften |
| winner_disagreement | Different winner, JS >= 0.10 | flag_review |
| score_extreme_euro_conservative | Score max prob > 0.75, Euro max < 0.55 | soften |

This is model diagnostics only. No betting advice.

## Inference Output Fields

With P1.12 calibration config:
- `score_calibrated_1x2`: calibrated home/draw/away win probabilities
- `calibrated_top_k_scorelines`: re-ranked top-k after calibration
- `disagreement_policy`: agreement level + confidence adjustment

## Known Limitations

1. Independent Poisson remains approximate — no true bivariate structure
2. Dixon-Coles style correction is approximate, not MLE-fitted
3. Calibration can improve distribution shape but cannot add missing match context
4. Extreme-score calibration may remain weak (need model-side features)
5. Temperature scaling can trade off accuracy for better distribution shape
6. Current calibration does not account for match-specific factors (league, strength)

## Next Phase Suggestion

If P1.12 PASS:
- P1.13: Euro + Score Probability Fusion
  - Combine Euro Head probs with calibrated score-derived probs
  - Learn val-selected fusion weight
  - Evaluate consistency-aware confidence adjustment

## Compliance

- No betting advice
- No profit claim
- No internet/crawler/API
- No tokenizer/vocab/LM Head
- No MiniMind original files modified
- No backbone retraining
- No test-set tuning
- This is model calibration and diagnostics only
