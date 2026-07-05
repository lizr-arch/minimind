# P8 Draw Signal Forensics Runbook

## Purpose

P8 is diagnostic-only. It tests whether draw collapse comes from market priors,
P6 draw shrinkage, split/label issues, or missing score-structure signal.

Accepted mainline remains `P6 euro_default`. P8 probes must not be promoted.

## Formal Command

```powershell
python tools\p8_draw_signal_forensics.py `
  --data data\odds_real\titan007_pure_v4_market_semantics_v2.jsonl `
  --train-ids data\odds_real\splits_v6\train_match_ids.txt `
  --val-ids data\odds_real\splits_v6\val_match_ids.txt `
  --p6-predictions runs\p6_residual_patch_itransformer\p6_euro_default_seed42\val_predictions.csv `
  --out-root runs\p8_draw_signal_forensics `
  --diagnostics-dir diagnostics\p8_draw_signal_forensics
```

## Outputs

- `runs/p8_draw_signal_forensics/p8_final_report.json`
- `runs/p8_draw_signal_forensics/protocol_audit.json`
- `runs/p8_draw_signal_forensics/probe_results_by_seed.csv`
- `runs/p8_draw_signal_forensics/probe_results_summary.json`
- `runs/p8_draw_signal_forensics/probe_predictions/`
- `diagnostics/p8_draw_signal_forensics/p8_market_priors.csv`
- `diagnostics/p8_draw_signal_forensics/p8_p6_vs_market_draw_audit.json`
- `diagnostics/p8_draw_signal_forensics/p8_label_split_audit.json`

## Fixed Probes

- `probe_0_train_class_prior`
- `probe_1_market_close_de_vig`
- `probe_2_market_open_de_vig`
- `probe_3_train_only_multinomial_logistic_market_features`
- `probe_4_train_only_binary_draw_then_conditional_home_away_logistic`
- `probe_5_train_only_tiny_tabular_mlp_market_features`

Trainable probes use seeds `42,123,2025`. Deterministic probes are reported once.

## No-Go Items

- Do not use the test split.
- Do not fit on official validation.
- Do not promote P8 probes to mainline.
- Do not change P6 default behavior.
- Do not add probes after reading validation diagnostics.
- Do not add new Patch-iTransformer architecture, focal loss, posthoc draw threshold, or new draw BCE lambdas in P8.

## Current Verdict

P8 recommends `P9 Market-Prior Anchored Residual Model` because market close
draw ranking is much stronger than P6, while P6 shrinks true-draw probability
below the market prior.
