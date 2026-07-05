# P10 Market Draw Pairwise Report

Date: 2026-07-03

## Verdict

`P10_NO_ACCEPTED_VARIANT`

P10 confirms that market draw pairwise ranking can move draw behavior, but not
cleanly enough to become mainline. The best logloss variant improves overall
logloss slightly while draw remains weak. The strongest draw variant improves
draw top2 substantially but pays too much in logloss/ECE and still misses the
mean true-draw probability gate.

## Protocol

- Same-protocol control: `p10_control_p6_euro_default`
- Control seeds: `42, 123, 2025`
- Candidate variants: 4
- Candidate seeds: `42, 123, 2025`
- Total formal runs: 15
- Reports written: 15
- Logs written: 15
- Test split used: no
- Val posthoc fit used: no
- Train teacher coverage: 0.999924
- Val teacher coverage: 1.000000

## Results

| Model | Mean Logloss | ECE | Draw NLL | Draw Top2 | Mean P(draw true draw) | Draw Margin | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| P10 control P6 euro default | 0.937336 | 0.027492 | 1.644268 | 0.306884 | 0.198023 | 0.397617 | control |
| p10_a_pairwise_all_005 | 0.937291 | 0.030049 | 1.633497 | 0.324638 | 0.200177 | 0.394781 | P10_FAIL_GATES |
| p10_b_pairwise_all_020 | 0.937368 | 0.030199 | 1.613327 | 0.360145 | 0.204409 | 0.389303 | P10_FAIL_GATES |
| p10_c_pairwise_market_top2_010 | 0.937204 | 0.030217 | 1.620911 | 0.343478 | 0.202521 | 0.390995 | P10_FAIL_GATES |
| p10_d_pairwise_top2_true_margin | 0.939712 | 0.035617 | 1.572106 | 0.484058 | 0.212800 | 0.385259 | P10_FAIL_GATES |

## Gate Readout

Mainline candidate gates:

- `mean_val_logloss <= control + 0.0003`
- `mean_ece <= max(control_ece + 0.005, 0.033)`
- `draw_top2 >= 0.42`
- `draw_class_nll <= 1.5990`
- `mean_p_draw_true_draw >= 0.215`
- `mean_draw_margin_to_top <= 0.375`

`p10_c_pairwise_market_top2_010` is the best logloss variant at 0.937204, but
draw top2 is only 0.343478 and mean true-draw p_draw is only 0.202521.

`p10_d_pairwise_top2_true_margin` is the best draw variant: draw top2 reaches
0.484058 and draw NLL reaches 1.572106, but logloss regresses to 0.939712, ECE
rises to 0.035617, mean true-draw p_draw remains below 0.215, and draw margin
remains above 0.375.

## Interpretation

Market pairwise ranking is useful as a diagnostic pressure: increasing the
weight moves draw top2, pairwise agreement, draw NLL, and market draw MAE in the
expected direction. But the clean variants do not move draw enough, and the
strong variant moves it by spending too much overall predictive quality.

This means the current draw issue is unlikely to be solved by ranking loss
escalation alone. The next phase should move away from pure 1X2 draw loss
engineering and test score/goal-diff structure more directly.

## Artifacts

- Formal summary: `runs/p10_market_draw_pairwise/summary.json`
- Formal report: `runs/p10_market_draw_pairwise/report.md`
- Per-run logs: `runs/p10_market_draw_pairwise/logs/`
- Per-run reports: `runs/p10_market_draw_pairwise/*/report.json`

## Recommended Next Step

Start P11 as a score/goal-diff structure probe. Keep P10's pairwise result as a
diagnostic reference, not as a mainline candidate. Do not continue by simply
raising pairwise or true-draw margin weights.
