# OddsMind Large-Model Direction History

Date: 2026-07-01

This note summarizes the local numeric evidence for the large-model / sequence-model direction. It separates legacy P2 test results from the later P3-P6 official validation line.

## Executive Summary

The main lesson is not "make the Transformer bigger." The successful path was:

1. Direct Transformer capacity did not beat the small raw Euro MLP in P3.
2. Feature scaling and market fusion helped diagnostics, but still did not create a reliable Transformer win.
3. P4 changed the objective: Euro anchor + residual 1X2 delta + goal-diff supervision.
4. P6 ported that P4 objective into Patch-iTransformer sequence training and produced the first clear large-model win.

Current best checkpoint:

- Phase: P6 residual Patch-iTransformer objective probe
- Variant: `euro_default`
- Mean validation logloss: `0.9373360872`
- Std: `0.0001555752`
- Delta vs P4 mean: `-0.0017056664`
- Delta vs P3.2 raw label-smoothing baseline: about `-0.0035254757`
- Verdict: `P6_BEATS_P4`, `P6_SEQUENCE_OBJECTIVE_VALIDATED`, `P6_LOGLOSS_STRONG_PASS`
- Limitation: `DRAW_STILL_COLLAPSED`

## Metric Timeline

| Stage | Scope | Main idea | Split / caution | Mean logloss | Std | ECE | Draw recall | Verdict |
|---|---|---|---|---:|---:|---:|---:|---|
| P2 deployment candidate | OddsMind v1.0 | Earlier model candidate with temperature scaling | Legacy test split, not directly comparable with P3-P6 | `0.9745` temp test, baseline `0.9649` | N/A | `0.0191` temp | N/A | Accuracy edge small: `+0.17pp` mean, CI crossed zero |
| P3 RawPooledMLP | MLP baseline | 9-dim Euro raw pooled mean | Official val, 3 seeds | `0.9414160053` | `0.0004444277` | `0.016426` | `0.0` argmax draw in diagnostics | Strong simple baseline |
| P3a | Euro Patch-iTransformer | Direct Euro sequence model | Official val, 3 seeds | `0.9425018430` | `0.0017679937` | `0.019135` | `0.0` | Did not beat RawPooledMLP |
| P3b | Cross-market Patch-iTransformer | Euro + Asian + OU sequence model | Official val, 3 seeds | `0.9430417816` | `0.0008111293` | `0.019741` | `0.0` | Cross-market direct Transformer no gain |
| P3.2 best P3b scaling | P3b robust scaling | Feature scaling / ablation | Official val, 3 seeds | `0.9414794048` | N/A | N/A | Still collapsed | Scaling helped, still not enough |
| P3.2 raw label smoothing | Raw MLP calibration | Label smoothing `0.02` | Official val, 3 seeds | `0.9408615629` | N/A | N/A | Still collapsed | Best P3-era score |
| P3.3 stacking fusion | Market fusion | Separate Euro/Asian/OU models + fusion | Train/cal split + official val | `0.9441548983` | `0.0007907657` | `0.028263` | `0.0` | Fusion improved local variants, not global best |
| P3.4 OOF stacking | Train-only OOF fusion | OOF market stacking | Official val, no val-fit fusion | `0.9416168729` | `0.0003428764` | `0.023754` | `0.0` | Close, but not better than P3.2 raw LS |
| P4 anchor only | Euro anchor | Direct Euro odds anchor | Official val | `0.9431851506` | N/A | `0.041227` | `0.001087` | Anchor alone not enough |
| P4 Euro residual | MLP objective probe | `softmax(log(anchor)+delta)` + goal-diff head | Official val, 3 seeds | `0.9390417536` | `0.0001696418` | `0.030949` | `0.000725` | New best; objective works |
| P5 anchor mix | Draw calibration probe | Train-only anchor/draw mixing | Train-only selection, not official val score | no safe candidate | N/A | N/A | `0.0` selected | `passed_candidates=0`; do not continue anchor mix |
| P6 Euro residual Patch-iTransformer | Sequence objective port | P4 objective inside Patch-iTransformer | Official val, 3 formal seeds | `0.9373360872` | `0.0001555752` | `0.027492` | `0.001087` | Accepted; sequence objective validated |

## Important Deltas

| Comparison | Delta logloss | Meaning |
|---|---:|---|
| P3b direct Transformer vs P3 RawPooledMLP | `+0.0016257763` | Direct cross-market Transformer was worse. |
| P3.2 robust P3b vs P3 RawPooledMLP | `+0.0000633995` | Scaling almost caught up, but still not a win. |
| P3.2 raw label smoothing vs P3 RawPooledMLP | `-0.0005544424` | Calibration-style regularization gave a small baseline improvement. |
| P4 Euro residual vs P3.2 raw label smoothing | `-0.0018198093` | Objective change mattered more than architecture size. |
| P4 Euro residual vs anchor only | `-0.0041433970` | Residual learning improved the Euro anchor. |
| P6 Patch-iTransformer vs P4 Euro residual | `-0.0017056664` | Sequence capacity helped after the objective was fixed. |
| P6 Patch-iTransformer vs P3.2 raw label smoothing | `-0.0035254757` | Current best improvement over the pre-P4 best. |

## What Changed The Direction

P3 showed a negative result: simply feeding richer odds sequences into a Transformer did not beat a compact Euro-only MLP. Asian and OU were not useless as data sources, but direct cross-market modeling was noisy and did not improve final 1X2 logloss.

P4 changed the problem definition. Instead of asking the model to infer 1X2 probabilities from scratch, it anchored on Euro market probabilities and learned a residual delta, while a goal-diff auxiliary task shaped the representation. That made the first real jump.

P6 then answered the large-model question more cleanly: once the objective is right, a Patch-iTransformer sequence model can improve over the compact P4 MLP probe.

## Draw Status

Draw is still the main unsolved issue.

| Stage | Hard draw behavior |
|---|---|
| P3 diagnostics | Raw MLP, P3a, P3b all had `0` argmax draw predictions in validation diagnostics. |
| P4 Euro residual | Mean argmax draw count `0.6667`, mean draw recall `0.000725`. |
| P5 anchor mix | Train-only selected identity `alpha=0.0`; `passed_candidates=0`; draw recall remained `0.0`. |
| P6 Patch-iTransformer | Mean argmax draw count `1.0 / 4747`, mean draw recall `0.001087`. |

So P6 solved logloss progress, not draw collapse.

## Current Recommendation

Do P6.1 draw-specific auxiliary ablation next.

Use P6 `euro_default` as the locked baseline:

- `mean_val_logloss = 0.9373360872`
- `std = 0.0001555752`
- `mean_ece = 0.027492`
- `mean_draw_recall = 0.0010869565`
- `mean_argmax_draw_count = 1.0`

Run only pre-fixed draw auxiliary candidates:

- `euro_default_draw_aux_0005`
- `euro_default_draw_aux_001`
- `euro_default_draw_aux_002`

Seeds:

- `42`
- `123`
- `2025`

Hard constraints:

- No test set.
- No validation fitting.
- No train/val split change.
- No Asian/OU yet.
- No anchor-mix retry.
- No ordinal/Skellam yet.
- No larger Transformer yet.

Success should require both probability quality and draw behavior:

- `mean_val_logloss <= 0.9375360872`
- `seed_std <= 0.00035`
- `mean_ece <= 0.030492`
- `draw_recall >= 0.005`
- `draw_class_nll` improves by at least `0.003`
- `draw_precision` is reported and not degenerate

## Bottom Line

The direction should stay narrow:

1. Keep P6 as the accepted large-model baseline.
2. Solve draw behavior with a small train-time auxiliary objective.
3. Only if P6.1 fails should we consider a larger objective change such as ordinal / Skellam goal-diff modeling.
4. Asian/OU should wait until draw behavior is understood, because earlier Asian/OU experiments did not safely improve final 1X2 logloss.
