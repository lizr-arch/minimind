# P6.3 Draw Decision Probe

## Verdict

`P6_3_ARGMAX_RULE_NOT_PRIMARY_BOTTLENECK`

P6.3 did not train a new model. It evaluated fixed, pre-registered draw decision
policies on validation predictions to test whether draw collapse is mainly an
argmax decision-rule problem.

The answer is mostly no. The final probabilities do not put draw close enough to
rank1 on enough true-draw samples. Small margin policies do almost nothing.
Aggressive top2 draw policies recover draw recall only by sacrificing too much
overall accuracy.

## Inputs

- Accepted baseline: `runs/p6_residual_patch_itransformer`
- Coupled probe model: `runs/p6_2_coupled_draw`
- Seeds: `42,123,2025`
- Test split: not used
- Posthoc fitting: not used
- Policy selection: none; all policies are reported as probes only

## Baseline Decision Curve

| Policy | Accuracy | Delta vs argmax | Draw recall | Draw precision | Draw decisions / seed |
|---|---:|---:|---:|---:|---:|
| argmax | 0.5776 | 0.0000 | 0.0011 | 1.0000 | 1.0 |
| draw top2 margin <= 0.20 | 0.5769 | -0.0007 | 0.0011 | 0.1806 | 7.7 |
| draw top2 margin <= 0.30 | 0.5722 | -0.0054 | 0.0188 | 0.2401 | 72.0 |
| draw top2 margin <= 0.40 | 0.5660 | -0.0116 | 0.0659 | 0.2689 | 222.0 |
| draw top2 margin <= 0.50 | 0.5342 | -0.0433 | 0.1315 | 0.2219 | 542.3 |
| draw top2 any | 0.3772 | -0.2003 | 0.3069 | 0.1595 | 1765.0 |

## P6.2 Coupled Weak Decision Curve

| Policy | Accuracy | Delta vs argmax | Draw recall | Draw precision | Draw decisions / seed |
|---|---:|---:|---:|---:|---:|
| argmax | 0.5776 | 0.0000 | 0.0011 | 1.0000 | 1.0 |
| draw top2 margin <= 0.20 | 0.5768 | -0.0008 | 0.0033 | 0.2929 | 13.0 |
| draw top2 margin <= 0.30 | 0.5717 | -0.0058 | 0.0221 | 0.2247 | 84.0 |
| draw top2 margin <= 0.40 | 0.5640 | -0.0136 | 0.0714 | 0.2662 | 249.0 |
| draw top2 margin <= 0.50 | 0.5318 | -0.0457 | 0.1373 | 0.2209 | 572.7 |
| draw top2 any | 0.3779 | -0.1996 | 0.3080 | 0.1598 | 1766.7 |

## Interpretation

The useful region is narrow and weak:

- Margin <= 0.20 barely increases draw recall.
- Margin <= 0.30 reaches only about 0.02 draw recall and already costs about
  0.5-0.6 accuracy points.
- Margin <= 0.40 reaches about 0.07 draw recall, but costs 1.2-1.4 accuracy
  points.
- Top2-any is not viable: recall improves, but precision is about 0.16 and
  accuracy drops about 20 points.

This suggests the model's probability ranking is the core issue. Draw is not
merely hidden behind a too-strict argmax rule; in most true draws, p_draw is too
far below the top class.

## Next Direction

Do not promote post-hoc draw decision rules as a fix. The next useful phase
should investigate why draw is systematically low in the probability model:

- data segmentation by league / odds regime / favorite strength,
- draw-conditioned feature gaps,
- target formulation that models draw as a first-class state,
- or a separate pre-registered calibration study with out-of-fold calibration,
  not official-val posthoc tuning.
