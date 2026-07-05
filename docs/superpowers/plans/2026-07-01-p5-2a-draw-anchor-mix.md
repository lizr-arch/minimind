# P5.2a Draw Anchor Mix 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在不使用 test、不在 official val 上拟合参数、不改 P4 模型架构的前提下，验证 train-only segmented draw-anchor mix 是否能缓解 P4 的 draw collapse。

**架构：** P5.1 已证明 draw collapse 同时存在于 final head 和 diff-derived path。P5.2a 先做 train-only draw-prone segmentation，再用低自由度 anchor mix calibrator 只调整 draw probability，并保持 home/away 相对比例。official val 只能用 train-only frozen params 做 locked eval。

**技术栈：** Python、CSV/JSON artifacts、pytest、现有 P4/P5 report/prediction artifacts。

---

## Pro Coach 结论

- P5.1 diagnostic checkpoint 可以 ship，但不能叫 draw fix。
- 下一步不要先做 draw loss，也不要先做 ordinal/Skellam。
- 第一小实验：train-only draw-prone segmentation + segmented draw-anchor mix。
- 参数选择只能来自 official train 内部 deterministic split。
- official val 只能 locked eval，不能调 alpha、gate、threshold。

## 本地事实

- P4 best: `euro_default` mean val logloss `0.9390417536`。
- RawMLP label smoothing baseline: `0.940862`。
- Anchor-only val logloss: `0.9431851506`。
- P5.1 diagnosed 12 P4 runs, `test_ids_used=false`。
- P5.1 final argmax draw count mean: `0.666667 / 4747`。
- P5.1 final draw recall mean: `0.000725`。
- P5.1 diff argmax draw count mean: `0.0`。
- P5.1 diff draw recall mean: `0.0`。
- true-draw mean `p_draw` is only about `0.196-0.200`; non-draw mean `p_draw` is about `0.184-0.189`。

## 文件结构

- 修改：`docs/superpowers/plans/2026-07-01-p5-draw-calibration.md`
  - 补充 P5.1 diagnostic checkpoint verdict。
- 创建：`tools/p5_export_p4_split_predictions.py`
  - 从 P4 `best_model.pth`、`report.json`、`scaler.json` 导出指定 split 的 per-row final、anchor、diff 概率。
- 创建：`tools/p5_draw_segmentation.py`
  - 从 P4/P5 artifacts 生成 train-only 或 diagnostic-only segmentation report。
- 创建：`tools/p5_draw_anchor_mix.py`
  - 实现 deterministic train-internal split、anchor mix candidates、train-only selection、locked official val eval。
- 创建：`tests/test_p5_export_p4_split_predictions.py`
  - 覆盖 split safety、anchor columns、no test export。
- 创建：`tests/test_p5_draw_segmentation.py`
  - 覆盖 segmentation split safety 和 bin metrics。
- 创建：`tests/test_p5_draw_anchor_mix.py`
  - 覆盖 probability transform、deterministic split、no val fit、locked eval safety。

## 不变约束

- 禁止读取 test split 或 test artifacts。
- 禁止在 official val 上 fit alpha、gate、threshold、candidate family。
- 禁止修改 official train/val split。
- 禁止修改 P4 模型结构和默认 feature list。
- 禁止把 threshold curve 当生产决策规则。
- 禁止声称 draw 已修复，除非 locked val 同时满足 logloss、draw NLL、draw recall、ECE 红线。

## 任务 1：P5.1 checkpoint verdict

**文件：**
- 修改：`docs/superpowers/plans/2026-07-01-p5-draw-calibration.md`

- [ ] **步骤 1：补充 P5.1 verdict 文档**

在计划中追加：

```markdown
## P5.1 Diagnostic Checkpoint Verdict

- `P5_DIAGNOSTIC_CHECKPOINT_SHIPPABLE`
- `NO_TEST_USED`
- `NO_VALIDATION_FIT`
- `DRAW_COLLAPSE_CONFIRMED`
- `DIFF_DRAW_COLLAPSE_CONFIRMED`
- `THRESHOLD_RECALL_IS_DIAGNOSTIC_ONLY`
- `NEXT_STAGE_REQUIRES_TRAIN_ONLY_INTERVENTION`

Threshold curves are diagnostic-only. They are not production decision rules and are not validation-tuned calibration results.
```

- [ ] **步骤 2：运行现有 P5 diagnostics 测试**

运行：

```powershell
pytest tests/test_p5_draw_diagnostics.py -q
```

预期：PASS。

## 任务 2：P4 split prediction exporter

**前置现实：** P4 正式 run 目录目前只有 `val_predictions.csv` 和 `val_goal_diff_predictions.csv`，没有 train predictions，也没有 per-row anchor columns。P5.2a 必须先导出 train split predictions，否则会被迫在 val 上 fit，这是红线。

**文件：**
- 创建：`tools/p5_export_p4_split_predictions.py`
- 创建：`tests/test_p5_export_p4_split_predictions.py`

- [ ] **步骤 1：先写失败测试**

测试必须覆盖：

```python
def test_export_refuses_test_split():
    ...

def test_export_requires_existing_checkpoint():
    ...

def test_export_rows_include_final_anchor_and_diff_columns():
    ...

def test_export_uses_report_config_for_feature_groups_and_scaler():
    ...
```

- [ ] **步骤 2：实现 split prediction exporter**

CLI：

```powershell
python tools/p5_export_p4_split_predictions.py `
  --run-dir runs/p4_residual_goal_diff/p4_1_runs/p4_euro_default_seed42 `
  --split train `
  --out-csv runs/p5_draw_anchor_mix_train/source/p4_euro_default_seed42_train_predictions.csv `
  --no-test
```

实现要求：

- 从 `run_dir/report.json` 读取 `config.data`、`config.train_ids`、`config.val_ids`、`config.feature_groups`、`config.d_model`、`config.dropout`、`config.batch_size`。
- 从 `run_dir/best_model.pth` 加载 P4 模型。
- 使用 `tools/p4_train_residual_goal_diff.py` 中已有的 `load_split_ids`、`load_rows_for_ids`、`build_p4_dataset`、`apply_p4_feature_scaler`、`predict_outputs`，避免重复特征工程。
- `--split test` 必须 raise；不得接受 test ids 路径。
- `--split val` 只允许导出 locked eval source，不允许 downstream fit。
- 输出字段至少包含：

```text
match_id,y_true,y_goal_diff,
p_home,p_draw,p_away,
p_anchor_home,p_anchor_draw,p_anchor_away,
p_home_from_diff,p_draw_from_diff,p_away_from_diff,
pred_class,correct,pred_diff_bucket,correct_diff_bucket
```

- [ ] **步骤 3：运行 exporter 测试**

运行：

```powershell
pytest tests/test_p5_export_p4_split_predictions.py -q
```

预期：PASS。

## 任务 3：train-only draw segmentation

**文件：**
- 创建：`tools/p5_draw_segmentation.py`
- 创建：`tests/test_p5_draw_segmentation.py`

- [ ] **步骤 1：先写失败测试**

测试必须覆盖：

```python
def test_segmentation_refuses_test_split():
    ...

def test_segmentation_requires_train_for_param_selection():
    ...

def test_segmentation_bins_cover_all_rows():
    ...

def test_segmentation_outputs_draw_gap_metrics():
    ...
```

- [ ] **步骤 2：实现 CLI**

CLI：

```powershell
python tools/p5_draw_segmentation.py `
  --predictions-csv runs/p5_draw_anchor_mix_train/source/p4_euro_default_seed42_train_predictions.csv `
  --split train `
  --out-json runs/p5_draw_segmentation/p5_draw_segmentation_train.json `
  --out-md runs/p5_draw_segmentation/p5_draw_segmentation_train.md
```

安全规则：

- `--split test` 必须 raise。
- `--split val` 必须 raise，除非显式传 `--allow-val-diagnostic`。
- 所有 bin threshold 必须基于输入 split 本身；参数选择阶段只能传 train。

输出 bins：

- `anchor_draw_prob_decile`
- `anchor_favorite_margin_decile`
- `final_favorite_margin_decile`
- `final_p_draw_decile`
- `draw_suppression_decile`
- `p_diff_draw_decile`
- `final_vs_diff_draw_gap_decile`

每个 bin 输出：

- `n`
- `true_draw_rate`
- `mean_final_p_draw`
- `mean_anchor_p_draw`
- `mean_p_diff_draw`
- `draw_calibration_gap`
- `final_logloss`
- `draw_class_nll`
- `home_away_logloss_on_non_draw`
- `argmax_draw_count`

- [ ] **步骤 3：运行测试**

运行：

```powershell
pytest tests/test_p5_draw_segmentation.py -q
```

预期：PASS。

## 任务 4：draw-anchor mix calibrator

**文件：**
- 创建：`tools/p5_draw_anchor_mix.py`
- 创建：`tests/test_p5_draw_anchor_mix.py`

- [ ] **步骤 1：先写失败测试**

测试必须覆盖：

```python
def test_train_inner_split_is_deterministic():
    ...

def test_train_inner_split_has_no_overlap():
    ...

def test_anchor_mix_alpha_zero_identity():
    ...

def test_anchor_mix_probability_sum_one():
    ...

def test_anchor_mix_preserves_home_away_ratio():
    ...

def test_anchor_mix_never_uses_val_for_fit():
    ...

def test_anchor_mix_refuses_test_rows():
    ...

def test_margin_gate_uses_train_quantiles_only():
    ...

def test_suppression_gate_uses_train_quantiles_only():
    ...

def test_locked_val_eval_cannot_change_params():
    ...
```

- [ ] **步骤 2：实现 deterministic train-internal split**

使用 stable hash：

```python
bucket = stable_hash(match_id_or_row_id) % 100
```

规则：

- `< 70`: `train_base_fit`
- `70 <= bucket < 85`: `train_cal_fit`
- `>= 85`: `train_cal_eval`

- [ ] **步骤 3：实现 anchor mix transform**

Global:

```python
p_new_draw = (1 - alpha) * p_final_draw + alpha * p_anchor_draw
p_new_home = p_final_home * (1 - p_new_draw) / max(1e-8, 1 - p_final_draw)
p_new_away = p_final_away * (1 - p_new_draw) / max(1e-8, 1 - p_final_draw)
```

Margin gated:

```python
gate = anchor_favorite_margin <= margin_threshold
p_new_draw = (1 - alpha * gate) * p_final_draw + alpha * gate * p_anchor_draw
```

Suppression gated:

```python
draw_suppression = logit(p_anchor_draw) - logit(p_final_draw)
gate = draw_suppression >= suppression_threshold
```

- [ ] **步骤 4：实现 train-only selection**

Candidate grid：

- `global_anchor_mix`: `alpha = 0.00,0.03,0.05,0.08,0.10,0.15`
- `margin_gated_anchor_mix`: `alpha = 0.03,0.05,0.08,0.10`; margin threshold from train_cal_fit quantiles `q20,q30,q40,q50`
- `suppression_gated_anchor_mix`: `alpha = 0.03,0.05,0.08,0.10`; suppression threshold from train_cal_fit quantiles `q50,q60,q70,q80`

Selection metrics on `train_cal_eval`：

- `overall_logloss`
- `delta_logloss_vs_uncalibrated`
- `draw_class_nll`
- `delta_draw_class_nll`
- `draw_recall`
- `draw_precision`
- `draw_top2_recall`
- `expected_draw_count`
- `hard_draw_count`
- `top_label_ece`
- `classwise_ece_draw`
- `home_away_logloss_on_non_draw`

Pass gate：

- `overall_logloss <= baseline_logloss + 0.00020`
- `draw_class_nll <= baseline_draw_class_nll - 0.003`
- `draw_recall >= max(0.010, 3x baseline_draw_recall)`
- `classwise_ece_draw <= baseline_classwise_ece_draw + 0.005`
- `top_label_ece <= baseline_top_label_ece + 0.003`
- `home_away_logloss_on_non_draw degradation <= 0.00030`

- [ ] **步骤 5：实现 CLI**

Train-only selection：

```powershell
python tools/p5_draw_anchor_mix.py `
  --predictions-csv runs/p5_draw_anchor_mix_train/source/p4_euro_default_seed42_train_predictions.csv `
  --base-variant euro_default `
  --fit-split train `
  --inner-split stable_hash_70_15_15 `
  --candidates global_anchor_mix,margin_gated_anchor_mix,suppression_gated_anchor_mix `
  --out-dir runs/p5_draw_anchor_mix_train `
  --no-test `
  --forbid-val-fit
```

Locked val eval：

```powershell
python tools/p5_draw_anchor_mix.py `
  --predictions-csv runs/p5_draw_anchor_mix_val_locked/source/p4_euro_default_seed42_val_predictions.csv `
  --base-variant euro_default `
  --locked-params runs/p5_draw_anchor_mix_train/params.json `
  --eval-split val `
  --out-dir runs/p5_draw_anchor_mix_val_locked `
  --no-test `
  --no-refit-on-val
```

- [ ] **步骤 6：运行测试**

运行：

```powershell
pytest tests/test_p5_draw_anchor_mix.py -q
```

预期：PASS。

## 任务 5：整体验证和正式运行

**文件：**
- 生成：`runs/p5_draw_segmentation/p5_draw_segmentation_train.json`
- 生成：`runs/p5_draw_segmentation/p5_draw_segmentation_train.md`
- 生成：`runs/p5_draw_anchor_mix_train/report.json`
- 生成：`runs/p5_draw_anchor_mix_train/report.md`
- 条件生成：`runs/p5_draw_anchor_mix_val_locked/report.json`
- 条件生成：`runs/p5_draw_anchor_mix_val_locked/report.md`

- [ ] **步骤 1：运行 P5/P4 回归**

```powershell
pytest `
  tests/test_p4_goal_diff_utils.py `
  tests/test_p4_training_report.py `
  tests/test_p4_audit_report.py `
  tests/test_p5_draw_diagnostics.py `
  tests/test_p5_export_p4_split_predictions.py `
  tests/test_p5_draw_segmentation.py `
  tests/test_p5_draw_anchor_mix.py -q
```

- [ ] **步骤 2：导出 train source predictions**

```powershell
python tools/p5_export_p4_split_predictions.py `
  --run-dir runs/p4_residual_goal_diff/p4_1_runs/p4_euro_default_seed42 `
  --split train `
  --out-csv runs/p5_draw_anchor_mix_train/source/p4_euro_default_seed42_train_predictions.csv `
  --no-test
```

- [ ] **步骤 3：运行 train-only segmentation**

```powershell
python tools/p5_draw_segmentation.py `
  --predictions-csv runs/p5_draw_anchor_mix_train/source/p4_euro_default_seed42_train_predictions.csv `
  --split train `
  --out-json runs/p5_draw_segmentation/p5_draw_segmentation_train.json `
  --out-md runs/p5_draw_segmentation/p5_draw_segmentation_train.md
```

- [ ] **步骤 4：运行 train-only anchor mix selection**

```powershell
python tools/p5_draw_anchor_mix.py `
  --predictions-csv runs/p5_draw_anchor_mix_train/source/p4_euro_default_seed42_train_predictions.csv `
  --base-variant euro_default `
  --fit-split train `
  --inner-split stable_hash_70_15_15 `
  --candidates global_anchor_mix,margin_gated_anchor_mix,suppression_gated_anchor_mix `
  --out-dir runs/p5_draw_anchor_mix_train `
  --no-test `
  --forbid-val-fit
```

- [ ] **步骤 5：只在 train-only pass 后运行 locked val eval**

如果 `runs/p5_draw_anchor_mix_train/report.json` 中有 candidate 满足 pass gate，再运行：

先导出 locked val source：

```powershell
python tools/p5_export_p4_split_predictions.py `
  --run-dir runs/p4_residual_goal_diff/p4_1_runs/p4_euro_default_seed42 `
  --split val `
  --out-csv runs/p5_draw_anchor_mix_val_locked/source/p4_euro_default_seed42_val_predictions.csv `
  --no-test
```

再运行 locked eval：

```powershell
python tools/p5_draw_anchor_mix.py `
  --predictions-csv runs/p5_draw_anchor_mix_val_locked/source/p4_euro_default_seed42_val_predictions.csv `
  --base-variant euro_default `
  --locked-params runs/p5_draw_anchor_mix_train/params.json `
  --eval-split val `
  --out-dir runs/p5_draw_anchor_mix_val_locked `
  --no-test `
  --no-refit-on-val
```

Minimum locked val pass：

- `val_logloss <= 0.939042 + 0.00020`
- `draw_recall >= 0.010`
- `draw_class_nll` improves versus P4
- `top_label_ece <= P4 + 0.003`
- `classwise_ece_draw <= P4 + 0.005`

## Reviewer Checklist

- [ ] No test reads.
- [ ] No validation fitting.
- [ ] No official train/val split change.
- [ ] No P4 model architecture change.
- [ ] Alpha/gate selection only uses train_cal_fit and train_cal_eval.
- [ ] Official val locked eval uses frozen params only.
- [ ] Reports do not claim draw fixed unless locked val gates pass.
