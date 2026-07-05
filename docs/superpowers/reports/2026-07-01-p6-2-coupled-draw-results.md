# P6.2 Coupled Final-Draw Results

## Verdict

`P6_2_COUPLED_DRAW_NO_SAFE_DRAW_FIX`

P6.2 verified the mechanism, but the fixed variants did not solve draw collapse.
The direct final-draw BCE variants reproduced the accepted P6 baseline almost
exactly. The weak draw-logit coupling produced a small logloss and draw-NLL
improvement, but final draw recall stayed unchanged and seed stability crossed
the red-light threshold.

## Scope

- Runs root: `runs/p6_2_coupled_draw`
- Fixed variants before seeing validation results:
  - `euro_default_final_draw_bce_005`, `lambda_final_draw_bce=0.005`
  - `euro_default_final_draw_bce_010`, `lambda_final_draw_bce=0.010`
  - `euro_default_draw_logit_coupled_weak`, `lambda_final_draw_bce=0.005`, `draw_coupling_scale=0.25`
- Seeds: `42,123,2025`
- Same data, split, and non-draw hyperparameters as accepted P6 baseline.
- No test split used.
- No posthoc validation fitting.
- No new lambda or scale was added after seeing validation results.

## Diagnostic Before P6.2

P6/P6.1 coupling diagnostic showed why P6.1 failed:

| Variant | true draws | rank1 | rank2 | rank3 | mean margin | mean p_draw true draw |
|---|---:|---:|---:|---:|---:|---:|
| P6 baseline | 2760 | 3 | 844 | 1913 | 0.359968 | 0.198023 |
| P6.1 0.005 | 2760 | 3 | 865 | 1892 | 0.355916 | 0.200214 |
| P6.1 0.010 | 2760 | 3 | 864 | 1893 | 0.355964 | 0.200176 |
| P6.1 0.020 | 2760 | 3 | 860 | 1897 | 0.356027 | 0.200117 |

P6.1 increased true-draw probability slightly but did not move draws near rank1.

## P6.2 Key Metrics

| Variant | mean logloss | std | mean ECE | draw NLL | draw recall | draw top2 | draw ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| P6 baseline | 0.937336 | 0.000156 | 0.027492 | 1.644268 | 0.001087 | 0.306884 | 0.011259 |
| final draw BCE 0.005 | 0.937335 | 0.000154 | 0.027629 | 1.644221 | 0.001087 | 0.306884 | 0.011249 |
| final draw BCE 0.010 | 0.937334 | 0.000153 | 0.027626 | 1.644172 | 0.001087 | 0.306884 | 0.011211 |
| draw-logit coupled weak | 0.936944 | 0.000507 | 0.028071 | 1.638330 | 0.001087 | 0.307971 | 0.013236 |

## Audits

- Full grouped report: `runs/p6_2_coupled_draw/p6_2_coupled_draw_report.json`
- Leak audits:
  - `p6_2_leak_audit_final_draw_bce_005.json`: pass
  - `p6_2_leak_audit_final_draw_bce_010.json`: pass
  - `p6_2_leak_audit_draw_logit_coupled_weak.json`: pass
- Sanity audits versus accepted P6:
  - final BCE 0.005 mean delta: `-0.0000011980`
  - final BCE 0.010 mean delta: `-0.0000022662`
  - coupled weak mean delta: `-0.0003924724`

All three sanity audits pass bootstrap, but only coupled weak has a meaningful
logloss movement. None passes the draw-recall gate.

## Interpretation

P6.2 shows that a bounded direct coupling can affect logloss and draw NLL, but
the weak scale does not overcome the large draw margin observed in the diagnostic.
For most true draws, final p_draw is still rank3 and about 0.35 below the top
class. A weak 0.25 bounded residual cannot reliably flip those examples.

Accepted mainline should remain P6 euro_default. P6.1 and P6.2 should be kept as
diagnostic results, not promoted as draw fixes.
