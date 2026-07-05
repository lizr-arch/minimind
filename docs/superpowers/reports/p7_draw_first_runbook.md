# P7 Draw-First Factorized 1X2 Runbook

## Purpose

P7 is a structural draw-collapse probe. It replaces the final 3-way softmax
with a draw-first factorized head:

- `p_draw = sigmoid(draw_logit)`
- `p_home = (1 - p_draw) * sigmoid(home_away_logit)`
- `p_away = (1 - p_draw) * (1 - sigmoid(home_away_logit))`

P7 does not add focal loss, goal-difference modeling, total-goal modeling,
posthoc draw thresholds, extra seeds, or extra lambda values.

## Fixed Matrix

Seeds are exactly:

- `42`
- `123`
- `2025`

Variants are exactly:

- `p7_a_factorized_ce_only`
- `p7_b_factorized_ce_draw_bce_005`
- `p7_c_factorized_ce_draw_bce_010`
- `p7_d_factorized_ce_cond_ha_aux_005`

## Formal Command

```powershell
python tools\p7_run_matrix.py `
  --no-test `
  --data data\odds_real\titan007_pure_v4_market_semantics_v2.jsonl `
  --train-ids data\odds_real\splits_v6\train_match_ids.txt `
  --val-ids data\odds_real\splits_v6\val_match_ids.txt `
  --out-root runs\p7_draw_first_matrix `
  --epochs 10 `
  --batch-size 256 `
  --device cuda
```

Use `--device cpu` only for smoke runs.

## Required Outputs

Matrix-level outputs:

- `runs/p7_draw_first_matrix/manifest.json`
- `runs/p7_draw_first_matrix/results_by_seed.csv`
- `runs/p7_draw_first_matrix/results_summary.json`
- `runs/p7_draw_first_matrix/draw_diagnostics_summary.md`
- `runs/p7_draw_first_matrix/protocol_audit.json`

Per-run outputs:

- `val_predictions.csv`
- `val_predictions.parquet` when optional `pyarrow` is installed
- `val_metrics.json`
- `classwise_metrics.json`
- `draw_rank_metrics.json`
- `draw_margin_metrics.json`
- `calibration_metrics.json`
- `training_manifest.json`
- `report.json`

## Frozen Draw Segments

After a run completes:

```powershell
python tools\p7_draw_segments.py `
  --data data\odds_real\titan007_pure_v4_market_semantics_v2.jsonl `
  --predictions runs\p7_draw_first_matrix\<variant_seed>\val_predictions.csv `
  --run-id <variant_seed> `
  --out-dir diagnostics
```

This diagnostic reads frozen prediction artifacts only. It does not create or
select candidates.

## Mainline Gate

P7 mainline requires all of:

- `mean_val_logloss <= 0.9373360872`
- `std_val_logloss <= 0.0003500000`
- `mean_ece <= 0.0300000000`
- `draw_class_nll <= 1.6200000000`
- `draw_top2 >= 0.3400000000`
- `mean_p_draw_true_draw >= 0.2150000000`
- `mean_draw_margin_to_top <= 0.3300000000`
- `draw_recall_argmax >= 0.0200000000`
- `draw_precision_argmax >= 0.1800000000`
- `sanity_audit_vs_p6 == PASS`

Any test split reference, candidate mutation, seed mutation, or P6 default
behavior change blocks acceptance.
