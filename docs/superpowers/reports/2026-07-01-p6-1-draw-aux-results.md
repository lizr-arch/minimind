# P6.1 Draw Auxiliary Formal Results

## Verdict

`P6_1_LOGLOSS_GAIN_BUT_DRAW_NOT_FIXED_UNSTABLE`

P6.1 is a useful negative result. A small draw BCE auxiliary head produced a real
small validation logloss gain and slightly better draw-class NLL, but it did not
change final 1X2 draw behavior. The final probabilities still select draw only
once per seed, so this is not a draw fix and should not be promoted as the main
line.

## Scope

- Runs root: `runs/p6_1_draw_aux`
- Fixed variants before seeing validation results:
  - `euro_default_draw_aux_0005`, `lambda_draw_bce=0.005`
  - `euro_default_draw_aux_001`, `lambda_draw_bce=0.010`
  - `euro_default_draw_aux_002`, `lambda_draw_bce=0.020`
- Seeds: `42,123,2025`
- Same data, split, and non-draw-aux hyperparameters as accepted P6 baseline.
- No test split used.
- No posthoc validation fitting.
- No new lambda candidates were added after seeing validation results.

## Key Metrics

| Variant | mean logloss | std | mean ECE | draw NLL | draw recall | draw top2 | draw ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| P6 baseline | 0.937336 | 0.000156 | 0.027492 | 1.644268 | 0.001087 | 0.306884 | 0.011259 |
| P6.1 0.005 | 0.936885 | 0.000606 | 0.027460 | 1.632966 | 0.001087 | 0.314493 | 0.012035 |
| P6.1 0.010 | 0.936878 | 0.000610 | 0.027256 | 1.633180 | 0.001087 | 0.314130 | 0.012143 |
| P6.1 0.020 | 0.936866 | 0.000619 | 0.027308 | 1.633520 | 0.001087 | 0.312681 | 0.012218 |

## Audits

- Full grouped report: `runs/p6_1_draw_aux/p6_1_draw_aux_report.json`
- Best mean-logloss variant sanity audit versus accepted P6:
  - File: `runs/p6_1_draw_aux/p6_1_sanity_audit_002_vs_p6.json`
  - mean_match_delta_nll: `-0.0004706111`
  - p_candidate_better: `0.998600`
  - verdict: bootstrap pass, top20 not concentrated, no test, no validation fit.
- Leak audits:
  - `runs/p6_1_draw_aux/p6_1_leak_audit_0005.json`: pass
  - `runs/p6_1_draw_aux/p6_1_leak_audit_001.json`: pass
  - `runs/p6_1_draw_aux/p6_1_leak_audit_002.json`: pass

## Interpretation

The draw auxiliary head is not coupled into final logits. It can regularize the
shared representation enough to improve logloss slightly, but it does not force
the final residual head to allocate more argmax mass to draw. This explains why
draw-class NLL improves while draw recall remains unchanged.

Next phase should be P6.2, not more P6.1 lambda tuning. The P6.2 question should
be: can a draw-specific signal safely enter `p_final = softmax(log(anchor) +
delta)` without hurting logloss, ECE, or seed stability?
