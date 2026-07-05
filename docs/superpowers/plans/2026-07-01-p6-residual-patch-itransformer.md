# P6 Residual Patch-iTransformer 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将 P4 已验证有效的 Euro-anchor residual + goal-diff objective 移植到现有 Patch-iTransformer 时序模型，验证更强 sequence capacity 是否能稳定超过 P4 MLP。

**架构：** P6 复用 P3b 的 per-feature Patch-iTransformer encoder，不重写数据 split、不引入 test、不做 official val fitting。模型输出 `delta_logits` 和 `goal_diff_logits`，训练损失复用 P4 的 `compute_p4_losses`，最终概率仍为 `softmax(log(p_euro_anchor) + delta)`。第一阶段只做 Euro-only，Asian/draw auxiliary/ordinal/Skellam 作为后续 ablation。

**技术栈：** Python、PyTorch、现有 P3b bucketized features、现有 P4 anchor/goal-diff utilities、pytest。

---

## Pro Coach Verdict

- 可以进入“大模型训练”，但必须定义为 `P6: P4 objective -> Patch-iTransformer port`。
- 不做泛化大架构重写，不直接上 draw loss，不先做 ordinal/Skellam，不先做 Asian settlement/payoff。
- P5.2a anchor mix `passed_candidates=0`，所以不要继续 anchor floor/calibration hack。
- 先验证 sequence Transformer capacity 是否能稳定超过 P4 MLP baseline `0.9390417536`。

## Baselines And Gates

- P4 MLP best mean val_logloss: `0.9390417536`。
- P4 best seed val_logloss: `0.9389190674`。
- P4 seed std: `0.0001696418`。
- RawMLP label smoothing baseline: `0.940862`。
- Anchor-only val_logloss: `0.9431851506`。

Minimum P6 pass:

- mean val_logloss `<= 0.9392418`
- seed std `<= 0.00035`
- no test reads
- no official val fitting

Strong P6 pass:

- mean val_logloss `<= 0.93870`
- seed std `<= 0.00030`
- ECE not worse than P4 by more than `0.003`
- draw_class_nll not worse than P4

Red lights:

- any test read
- any official val fitting
- any split change
- best-seed-only report
- mean val_logloss `> 0.9395418`
- seed std `> 0.00050`
- ECE worse by `> 0.005`
- report claims draw fixed without draw gates passing

## Files

- Create: `model/p6_residual_patch_itransformer.py`
  - Transformer encoder with P4-style residual heads.
- Create: `tools/p6_train_residual_patch_itransformer.py`
  - Train/evaluate one P6 run, train/val only.
- Create: `tools/p6_generate_report.py`
  - Aggregate P6 run reports and compare to P4.
- Create: `tests/test_p6_residual_patch_itransformer_model.py`
  - Model shape and input validation tests.
- Create: `tests/test_p6_training_report.py`
  - Report schema, no-test, no-val-fit, metric gates.
- Create: `tests/test_p6_data_scope.py`
  - Train/val-only guards and feature isolation smoke tests.

## Task 1: P6 Model

**Files:**
- Create: `model/p6_residual_patch_itransformer.py`
- Test: `tests/test_p6_residual_patch_itransformer_model.py`

- [ ] **Step 1: Write failing tests**

```python
def test_p6_model_forward_returns_residual_and_goal_diff_heads():
    model = P6ResidualPatchITransformer(n_features=10, d_model=32, n_heads=4, n_layers=1, d_ff=64, dropout=0.0)
    x = torch.randn(3, 10, 10, 2)
    out = model(x)
    assert set(out) == {"delta_logits", "goal_diff_logits"}
    assert out["delta_logits"].shape == (3, 3)
    assert out["goal_diff_logits"].shape == (3, 7)
```

```python
def test_p6_model_rejects_wrong_feature_count():
    model = P6ResidualPatchITransformer(n_features=10, d_model=32, n_heads=4, n_layers=1, d_ff=64)
    with pytest.raises(ValueError, match="Expected input shape"):
        model(torch.randn(2, 10, 11, 2))
```

- [ ] **Step 2: Implement model**

Implementation requirements:

- Reuse the P3b pattern:
  - per-feature `Linear(n_buckets * n_agg, d_model)`
  - feature id embedding
  - optional market type embedding only when caller supplies market type ids
  - CLS token
  - Transformer blocks
  - final norm
- Output heads:
  - `delta_1x2_head -> 3`
  - `goal_diff_head -> 7`
- Input shape is `[B, n_buckets, n_features, n_agg]`.

- [ ] **Step 3: Run tests**

```powershell
pytest tests/test_p6_residual_patch_itransformer_model.py -q
```

Expected: PASS.

## Task 2: P6 Training Script

**Files:**
- Create: `tools/p6_train_residual_patch_itransformer.py`
- Test: `tests/test_p6_training_report.py`
- Test: `tests/test_p6_data_scope.py`

- [ ] **Step 1: Write failing tests**

Test report payload:

```python
def test_p6_report_contains_required_sections_and_no_test_or_val_fit():
    payload = build_p6_report_payload(
        config={"feature_groups": "euro", "seed": 42},
        data={"train_samples": 10, "val_samples": 4, "test_ids_used": False, "validation_fit_used": False},
        best_epoch=1,
        best_val_logloss=0.94,
        val_metrics={"logloss": 0.94, "ece": 0.03, "draw_recall": 0.0, "argmax_draw_count": 0},
        goal_diff_metrics={"diff_nll": 1.8, "zero_recall": 0.0, "p_from_diff_logloss": 0.95},
        anchor_only_baseline={"anchor_only_val_logloss": 0.943},
        warnings=[],
    )
    assert payload["data"]["test_ids_used"] is False
    assert payload["data"]["validation_fit_used"] is False
    assert payload["phase"] == "P6 residual Patch-iTransformer objective probe"
```

Test split guard:

```python
def test_p6_cli_does_not_accept_test_ids_argument():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--test-ids", "data/odds_real/splits_v6/test_match_ids.txt"])
```

- [ ] **Step 2: Implement train script**

CLI:

```powershell
python tools/p6_train_residual_patch_itransformer.py `
  --data data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl `
  --train-ids data/odds_real/splits_v6/train_match_ids.txt `
  --val-ids data/odds_real/splits_v6/val_match_ids.txt `
  --feature-groups euro `
  --epochs 3 `
  --batch-size 128 `
  --lr 0.0003 `
  --weight-decay 0.01 `
  --d-model 64 `
  --n-heads 4 `
  --n-layers 2 `
  --d-ff 128 `
  --dropout 0.10 `
  --diff-loss-weight 0.30 `
  --consistency-loss-weight 0.10 `
  --delta-l2-weight 0.01 `
  --scaling robust `
  --seed 42 `
  --device cuda `
  --out-dir runs/p6_residual_patch_itransformer/smoke_seed42
```

Implementation requirements:

- Reuse P4 helpers:
  - `load_split_ids`
  - `load_rows_for_ids`
  - `build_p4_dataset`
  - `fit_p4_feature_scaler`
  - `apply_p4_feature_scaler`
  - `compute_p4_losses`
  - `compute_1x2_metrics`
  - `compute_anchor_only_metrics`
  - `fold_goal_diff_probs`
- Use P6 model instead of P4 MLP.
- Write:
  - `best_model.pth`
  - `report.json`
  - `val_predictions.csv`
  - `val_goal_diff_predictions.csv`
  - `scaler.json`
- Refuse overwrite unless `--allow-overwrite`.
- No `--test-ids` argument.
- Report must include `test_ids_used=false` and `validation_fit_used=false`.

- [ ] **Step 3: Run tests**

```powershell
pytest tests/test_p6_training_report.py tests/test_p6_data_scope.py -q
```

Expected: PASS.

## Task 3: P6 Report Aggregator

**Files:**
- Create: `tools/p6_generate_report.py`
- Test: `tests/test_p6_training_report.py`

- [ ] **Step 1: Write failing tests**

```python
def test_p6_aggregate_requires_three_seed_mean_not_best_seed_only(tmp_path):
    root = tmp_path / "p6"
    for seed, loss in [(42, 0.9400), (123, 0.9390), (2025, 0.9410)]:
        run_dir = root / f"p6_euro_default_seed{seed}"
        run_dir.mkdir(parents=True)
        (run_dir / "report.json").write_text(json.dumps({
            "config": {"seed": seed, "feature_groups": "euro"},
            "best_val_logloss": loss,
            "data": {"test_ids_used": False, "validation_fit_used": False},
            "val_metrics": {"logloss": loss, "ece": 0.03, "draw_recall": 0.0},
        }), encoding="utf-8")
    payload = aggregate_p6_reports(root, p4_mean_val_logloss=0.9390417536)
    assert payload["variants"][0]["seed_count"] == 3
    assert payload["variants"][0]["mean_val_logloss"] == pytest.approx(0.94)
```

```python
def test_p6_verdict_compares_against_p4_baseline():
    verdict = p6_verdict(
        mean_val_logloss=0.9386,
        std_val_logloss=0.0002,
        mean_ece=0.030,
        p4_mean_val_logloss=0.9390417536,
        p4_mean_ece=0.0309487,
        test_ids_used=False,
        validation_fit_used=False,
    )
    assert "P6_STRONG_PASS" in verdict
```

- [ ] **Step 2: Implement aggregator**

CLI:

```powershell
python tools/p6_generate_report.py `
  --root runs/p6_residual_patch_itransformer `
  --p4-report runs/p4_residual_goal_diff/p4_report.json `
  --out-json runs/p6_residual_patch_itransformer/p6_report.json `
  --out-md runs/p6_residual_patch_itransformer/p6_report.md
```

Report must include:

- input paths
- `test_ids_used=false`
- per-variant mean/std val_logloss over seeds
- best variant by mean, not by best seed
- comparison to P4 mean `0.9390417536`
- verdict:
  - `P6_MINIMUM_PASS`
  - `P6_STRONG_PASS`
  - `P6_FAIL_REGRESSION`
  - `DRAW_STILL_COLLAPSED`

## Task 4: Smoke And Formal Run Plan

- [ ] **Smoke**

```powershell
python tools/p6_train_residual_patch_itransformer.py `
  --data data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl `
  --train-ids data/odds_real/splits_v6/train_match_ids.txt `
  --val-ids data/odds_real/splits_v6/val_match_ids.txt `
  --feature-groups euro `
  --epochs 1 `
  --batch-size 64 `
  --d-model 32 `
  --n-heads 4 `
  --n-layers 1 `
  --d-ff 64 `
  --max-samples 128 `
  --device cpu `
  --out-dir runs/p6_residual_patch_itransformer/smoke_cpu_128_seed42
```

- [ ] **Formal 3-seed baseline**

Only after smoke and tests pass:

```powershell
python tools/p6_train_residual_patch_itransformer.py `
  --data data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl `
  --train-ids data/odds_real/splits_v6/train_match_ids.txt `
  --val-ids data/odds_real/splits_v6/val_match_ids.txt `
  --feature-groups euro --epochs 10 --batch-size 256 --lr 0.0003 --weight-decay 0.01 `
  --d-model 96 --n-heads 4 --n-layers 2 --d-ff 192 --dropout 0.10 `
  --diff-loss-weight 0.30 --consistency-loss-weight 0.10 --delta-l2-weight 0.01 `
  --scaling robust --seed 42 --device cuda `
  --out-dir runs/p6_residual_patch_itransformer/p6_euro_default_seed42

python tools/p6_train_residual_patch_itransformer.py `
  --data data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl `
  --train-ids data/odds_real/splits_v6/train_match_ids.txt `
  --val-ids data/odds_real/splits_v6/val_match_ids.txt `
  --feature-groups euro --epochs 10 --batch-size 256 --lr 0.0003 --weight-decay 0.01 `
  --d-model 96 --n-heads 4 --n-layers 2 --d-ff 192 --dropout 0.10 `
  --diff-loss-weight 0.30 --consistency-loss-weight 0.10 --delta-l2-weight 0.01 `
  --scaling robust --seed 123 --device cuda `
  --out-dir runs/p6_residual_patch_itransformer/p6_euro_default_seed123

python tools/p6_train_residual_patch_itransformer.py `
  --data data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl `
  --train-ids data/odds_real/splits_v6/train_match_ids.txt `
  --val-ids data/odds_real/splits_v6/val_match_ids.txt `
  --feature-groups euro --epochs 10 --batch-size 256 --lr 0.0003 --weight-decay 0.01 `
  --d-model 96 --n-heads 4 --n-layers 2 --d-ff 192 --dropout 0.10 `
  --diff-loss-weight 0.30 --consistency-loss-weight 0.10 --delta-l2-weight 0.01 `
  --scaling robust --seed 2025 --device cuda `
  --out-dir runs/p6_residual_patch_itransformer/p6_euro_default_seed2025
```

- [ ] **Ablations after baseline only**

Variants:

- `euro_no_goal_diff`: `diff_loss_weight=0, consistency=0, delta_l2=0.01`
- `euro_no_delta_l2`: `diff_loss_weight=0.30, consistency=0.10, delta_l2=0`
- `euro_no_consistency`: `diff_loss_weight=0.30, consistency=0, delta_l2=0.01`

Do not run draw auxiliary or Asian variants until the base P6 port is verified.

## Reviewer Checklist

- [ ] No test reads.
- [ ] No official val fitting.
- [ ] No split change.
- [ ] No best-seed-only verdict.
- [ ] P6 model is a P4 objective port, not a broad architecture rewrite.
- [ ] P5.2a failure is respected; no anchor calibration hack.
- [ ] Reports do not claim draw fixed unless draw gates pass.

## Checkpoint Result

P6 checkpoint is accepted after Pro coach review and local report hygiene patch.

Formal report:

- JSON: `runs/p6_residual_patch_itransformer/p6_matrix_report.json`
- Markdown: `runs/p6_residual_patch_itransformer/p6_matrix_report.md`

Formal metrics:

- `formal_seed_count`: 3
- `formal_report_count`: 3
- `raw_report_count`: 6
- `ignored_report_count`: 3, all `smoke_run`
- `mean_val_logloss`: 0.9373360872268677
- `std_val_logloss`: 0.00015557516412158728
- `delta_vs_P4_mean`: -0.0017056664
- `mean_ece`: 0.027492
- `mean_draw_recall`: 0.0010869565217391304
- `mean_argmax_draw_count`: 1.0
- `test_ids_used`: false
- `validation_fit_used`: false
- `anchor_fallback_count`: 0 for each formal seed
- `negative_time_valid_euro_count`: 0 for each formal seed

Verdicts:

- `P6_BEATS_P4`
- `P6_SEQUENCE_OBJECTIVE_VALIDATED`
- `P6_LOGLOSS_STRONG_PASS`
- `P6_STRONG_PASS`
- `P6_MINIMUM_PASS`
- `DRAW_STILL_COLLAPSED`
- `P6_NO_TEST_USED`
- `P6_NO_VAL_FIT`
- `P6_AUDIT_FLAGS_PASS`

Interpretation:

- P6 validates that the P4 Euro-anchor residual + goal-diff objective transfers to Patch-iTransformer sequence training and materially improves validation logloss.
- P6 does not fix draw collapse. Draw remains a dedicated follow-up problem.
- Do not claim `DRAW_FIXED`, `draw solved`, or `Transformer universally better`.

Next phase:

- Start P6.1 draw-specific auxiliary ablation.
- Do not move to Asian/OU, anchor-mix retry, ordinal/Skellam, or a larger Transformer before P6.1.
- Fixed P6.1 draw candidates: `euro_default_draw_aux_0005`, `euro_default_draw_aux_001`, `euro_default_draw_aux_002`.
- Fixed seeds: `42,123,2025`.
- P6.1 must keep no test usage, no validation fitting, and no train/val split changes.
