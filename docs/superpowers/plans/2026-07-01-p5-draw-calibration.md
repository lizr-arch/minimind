# P5 Draw Diagnostics and Calibration Plan

> Ship mode: Pro coach attempted first; local Codex remains execution owner.

## Brief

P4 established a new best validation logloss through a Euro-anchor residual objective:
`p_final = softmax(log(p_euro_anchor) + delta)`. The best P4 variant is `euro_default`
with 3-seed mean val logloss `0.9390418`, beating RawMLP label smoothing `0.940862`.
However, draw argmax remains collapsed: mean draw recall is about `0.000725`.

P5 should not start with a larger architecture. It should first diagnose whether the
draw problem is a probability-calibration problem, an argmax decision problem, or a
missing draw-specific signal problem.

## Pro Coach Packet

Sent to ChatGPT Pro thread `https://chatgpt.com/c/6a44953c-a5d0-83ea-8d9e-360d174ccd23`.

The packet included:

- Repo path and branch.
- P4 constraints: no test set, no split changes, no broad architecture churn.
- P4 implementation files and line references.
- P4 formal 3-seed results:
  - Euro anchor only: `0.943185`
  - RawMLP label smoothing: `0.940862`
  - P4 `euro_default`: `0.9390418`
  - P4 `euro_no_consistency`: `0.9391009`
  - P4 `euro_asian_default`: `0.9391681`
  - P4 `euro_asian_no_consistency`: `0.9393257`
- Current verdict:
  - `P4_NEW_BEST`
  - `P4_ASIAN_GOAL_DIFF_NO_GAIN`
  - `DRAW_STILL_COLLAPSED`

Pro reply was insufficient: it only returned high-level framing and did not provide
the requested task split or success criteria. Local execution proceeds with this
plan, and Pro is treated as attempted but not decisive.

## Scope

P5.1 is diagnostic-only:

- Add a P5 draw diagnostics tool.
- Read P4 validation prediction artifacts only.
- Do not fit parameters on validation.
- Do not touch test set.
- Do not alter P4 model architecture.
- Do not start AH settlement loss until diagnostics justify it.

P5.2, if P5.1 finds actionable signal, may run train-only calibration:

- Split official train into base/calibration folds.
- Train P4 on base split.
- Fit calibration only on calibration split.
- Evaluate once on official validation.

## Required Diagnostics

For every P4 run and aggregated variant:

- Argmax draw count and draw recall.
- Draw threshold curve:
  - thresholds such as `0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30`
  - predicted draw count
  - draw precision
  - draw recall
  - accuracy if threshold decision is used
  - note that probability logloss is unchanged by threshold-only decisions
- Draw calibration bins:
  - bin range
  - count
  - draw rate
  - mean predicted p_draw
- Final-vs-diff draw comparison:
  - mean `p_final_draw - p_from_diff_draw`
  - mean absolute gap
  - same gap on true draws
- Draw probability diagnostics:
  - mean p_draw
  - mean p_draw on true draw
  - mean p_draw on non-draw

## Success Criteria

P5.1 diagnostic success:

- Produces `runs/p5_draw_diagnostics/p5_draw_diagnostics_report.json`.
- Produces `runs/p5_draw_diagnostics/p5_draw_diagnostics_report.md`.
- Confirms `test_ids_used == false`.
- Contains all four P4 variants:
  - `euro_default`
  - `euro_no_consistency`
  - `euro_asian_default`
  - `euro_asian_no_consistency`
- Identifies at least one of:
  - draw probabilities are under-calibrated,
  - draw probabilities are calibrated but never top-1,
  - p_from_diff has stronger draw signal than p_final,
  - no useful draw signal exists in current P4 outputs.

P5.2 calibration success, if executed:

- Mean val logloss must remain `<= 0.93935`.
- Draw recall should improve by at least `+0.02` absolute, or report clearly that
  argmax draw improvement is impossible without logloss harm.
- ECE must not worsen by more than `+0.005`.
- All decisions must come from train-only calibration, not validation fitting.

## Red Lights

Stop or do not proceed to model changes if:

- Any tool reads test ids or test artifacts.
- Any calibration is fitted on official validation predictions.
- Threshold tuning is selected by optimizing official validation and then reported
  as a model improvement.
- Draw recall improves only by destroying logloss beyond `0.93935`.
- Asian signal appears only in p_from_diff but cannot improve p_final without
  worsening logloss.

## Subagent Split

Writer:

- Owns `tools/p5_draw_diagnostics.py`.
- Owns `tests/test_p5_draw_diagnostics.py`.
- Implements diagnostic-only CLI and report payload.

Tester:

- Runs P5 unit tests.
- Runs P4 regression tests affected by shared prediction/report assumptions.
- Runs the P5 CLI against `runs/p4_residual_goal_diff`.
- Verifies report schema and no test-set usage.

Reviewer:

- Checks no validation fitting is smuggled in.
- Checks no test-set reads exist.
- Checks threshold curves are labelled diagnostic-only.
- Checks variant aggregation does not average incompatible runs.

## Commands

```powershell
pytest tests/test_p5_draw_diagnostics.py -q
pytest tests/test_p4_goal_diff_utils.py tests/test_p4_training_report.py tests/test_p4_audit_report.py -q
python tools/p5_draw_diagnostics.py `
  --p4-root runs/p4_residual_goal_diff `
  --out-json runs/p5_draw_diagnostics/p5_draw_diagnostics_report.json `
  --out-md runs/p5_draw_diagnostics/p5_draw_diagnostics_report.md
```

## P5.1 Diagnostic Checkpoint Verdict

- `P5_DIAGNOSTIC_CHECKPOINT_SHIPPABLE`
- `NO_TEST_USED`
- `NO_VALIDATION_FIT`
- `DRAW_COLLAPSE_CONFIRMED`
- `DIFF_DRAW_COLLAPSE_CONFIRMED`
- `THRESHOLD_RECALL_IS_DIAGNOSTIC_ONLY`
- `NEXT_STAGE_REQUIRES_TRAIN_ONLY_INTERVENTION`

Threshold curves are diagnostic-only. They are not production decision rules and are not validation-tuned calibration results.
