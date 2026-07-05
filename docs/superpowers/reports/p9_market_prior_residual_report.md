# P9 Market-Prior Anchored Residual Report

## Verdict

- matrix verdict: `P9_NO_MAINLINE_CANDIDATE`
- best by logloss: `p9_c_anchor_kl_001`
- best draw-recovery variant: `p9_d_true_draw_floor_010`

P9 confirms that mild residual anchoring can slightly improve overall logloss,
but it does not recover enough draw behavior. A stronger true-draw floor improves
draw-specific metrics, but still fails the mainline gate because mean true-draw
probability and draw margin remain short of target.

## Protocol

- formal reports: `12/12`
- test split: `not loaded`
- protocol hard_fail_reasons: `[]`
- summary: `runs/p9_market_prior_residual/results_summary.json`

## Results

| Variant | Seeds | Mean Logloss | Std Logloss | Draw NLL | Draw Top2 | Mean P(draw true draw) | Mean Draw Margin | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `p9_a_p6_explicit_market_anchor` | 3 | 0.9373360475 | 0.0001556304 | 1.6442678769 | 0.3068840603 | 0.1980232845 | 0.3976168831 | `P9_FAIL_GATES` |
| `p9_b_draw_delta_l2_001` | 3 | 0.9373086691 | 0.0001959580 | 1.6424448093 | 0.3148550689 | 0.1983828743 | 0.3974951704 | `P9_FAIL_GATES` |
| `p9_c_anchor_kl_001` | 3 | 0.9372987747 | 0.0001686165 | 1.6412734985 | 0.3086956541 | 0.1986071716 | 0.3965498706 | `P9_FAIL_GATES` |
| `p9_d_true_draw_floor_010` | 3 | 0.9374504884 | 0.0003488527 | 1.5990122159 | 0.3818840683 | 0.2074971149 | 0.3839951754 | `P9_FAIL_GATES` |

## Interpretation

- `p9_c_anchor_kl_001` is the best overall logloss result so far, slightly better than the P6 control mean, but draw behavior is still near P6.
- `p9_d_true_draw_floor_010` proves the loss can move draw behavior: draw NLL improves from about `1.6443` to `1.5990`, and draw top2 improves from about `0.3069` to `0.3819`.
- The true-draw floor still underperforms the P8 market prior on draw behavior: P8 market close had draw top2 about `0.6315` and mean true-draw p_draw about `0.2532`.
- Next work should not be a larger backbone. The useful signal is in market-prior draw geometry, but direct residual regularization is too weak or too expensive in logloss/margin.

## Recommended Next Step

Move to a narrow P10 decision:

- either distill market draw ranking into training without using val,
- or model score/goal-difference structure before folding back to 1X2,
- but do not add more blind P9 lambdas without a new diagnostic reason.
