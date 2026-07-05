# OddsMind Mainline Status

Verdict: `P16_PROMOTE_P14_MARKET_REPLICATION_MAINLINE`

## Current Mainline
- mainline_id: `p16_b_mktkl_100x_old_balanced`
- model_family: `P6ResidualPatchITransformer`
- train_script: `tools/p14_train_market_replication.py`
- selection_rule: `old_balanced`
- market_loss_weight: `0.3`
- checkpoint_selection: `balanced`
- balanced_logloss_ceiling: `0.9375`

## Promotion Evidence
- control_logloss: `0.9368612766265869`
- mainline_logloss: `0.9370384613672892`
- control_draw_top2: `0.3072463870048523`
- mainline_draw_top2: `0.3851449290911357`
- control_true_draw_p: `0.19887619217236838`
- mainline_true_draw_p: `0.2160190294186274`
- mainline_market_draw_mae: `0.03618671620885531`
- slice_gate_pass: `True`

## Guardrails
- val-fitted calibration is diagnostic only and is not promotable.
- P15 draw-risk auxiliary remains diagnostic-only.
- P17 value detection is design-only until explicitly requested.
