# P9 Market-Prior Anchored Residual Runbook

## Purpose

P9 tests fixed draw-preservation constraints on top of the existing P6
market-close-prior residual architecture. It does not introduce a larger
backbone or a new Patch-iTransformer family.

## Fixed Matrix

Variants:

- `p9_a_p6_explicit_market_anchor`
- `p9_b_draw_delta_l2_001`
- `p9_c_anchor_kl_001`
- `p9_d_true_draw_floor_010`

Seeds:

- `42`
- `123`
- `2025`

## Formal Command

```powershell
python tools\p9_run_matrix.py `
  --no-test `
  --data data\odds_real\titan007_pure_v4_market_semantics_v2.jsonl `
  --train-ids data\odds_real\splits_v6\train_match_ids.txt `
  --val-ids data\odds_real\splits_v6\val_match_ids.txt `
  --out-root runs\p9_market_prior_residual `
  --epochs 10 `
  --batch-size 256 `
  --device cuda
```

## Outputs

- `runs/p9_market_prior_residual/protocol_audit.json`
- `runs/p9_market_prior_residual/manifest.json`
- `runs/p9_market_prior_residual/results_by_seed.csv`
- `runs/p9_market_prior_residual/results_summary.json`
- per run: `report.json`, `best_model.pth`, `val_predictions.csv`, `val_goal_diff_predictions.csv`, `scaler.json`

## Gates

Mainline candidate requires:

- `mean_val_logloss <= 0.93760`
- `draw_class_nll <= 1.620`
- `draw_top2 >= 0.38`
- `mean_p_draw_true_draw >= 0.215`
- `mean_draw_margin_to_top <= 0.36`
- `std_val_logloss <= 0.00060`
- protocol audit pass

## No-Go

- Do not use the test split.
- Do not change P6 default behavior.
- Do not add unregistered variants after seeing validation results.
- Do not add posthoc draw thresholds or league-specific thresholds.
- Do not promote smoke or single-seed runs.
