# P9 Market-Prior Anchored Residual 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在不改变 P6 默认行为和不新增大模型架构的前提下，验证“market close prior + residual + draw-preservation constraint”能否减少 P6 对 true-draw 的概率压缩。

**架构：** 复用 `P6ResidualPatchITransformer` 与 P4/P6 数据管线。P9 显式把 `p_euro_anchor` 标记为 market close prior，并在 P6 loss 外增加固定的 draw-preservation 正则项，例如 draw residual L2、anchor-to-final KL、true-draw draw floor hinge。所有变体固定注册，不允许看结果后追加。

**技术栈：** PyTorch、现有 P4/P6 bucketized dataset、CSV/JSON report、pytest。

---

## 文件职责

- 创建 `tools/p9_train_market_prior_residual.py`：P9 训练脚本，复用 P6 model/data，新增 P9 loss、metrics、report。
- 创建 `tools/p9_run_matrix.py`：固定 P9 variant × seed runner，拒绝 test split 和未注册变体。
- 创建 `tests/test_p9_market_prior_residual.py`：P9 loss、协议、runner、P6 默认不变的窄测。
- 创建 `docs/superpowers/reports/p9_market_prior_residual_runbook.md`：正式运行命令、固定矩阵、gate。

## 固定矩阵

Seeds: `42,123,2025`

Variants:

- `p9_a_p6_explicit_market_anchor`: P6-equivalent explicit market-anchor control。
- `p9_b_draw_delta_l2_001`: `0.01 * mean(delta_draw^2)`，只限制 draw residual。
- `p9_c_anchor_kl_001`: `0.01 * KL(anchor || final)`，限制 final 不要远离 market prior。
- `p9_d_true_draw_floor_010`: `0.10 * mean(max(anchor_draw - final_draw, 0))` on true-draw rows，训练中只用 train labels。

## Gates

Mainline candidate 需要：

- mean val logloss <= `0.93760`
- draw_class_nll <= `1.620`
- draw_top2 >= `0.38`
- mean_p_draw_true_draw >= `0.215`
- mean_draw_margin_to_top <= `0.36`
- std val logloss <= `0.00060`
- protocol audit pass

Reject P9 residual constraints if all non-control variants fail both:

- draw_top2 improvement over P6 < `0.03`
- mean_p_draw_true_draw improvement over P6 < `0.015`

## 任务 1：P9 Loss 窄测

**文件：**
- 创建：`tests/test_p9_market_prior_residual.py`
- 创建：`tools/p9_train_market_prior_residual.py`

- [ ] **步骤 1：编写失败测试**

测试应覆盖：

```python
def test_p9_zero_weights_matches_p6_loss():
    ...

def test_draw_delta_l2_only_penalizes_draw_residual():
    ...

def test_anchor_kl_penalty_increases_when_final_moves_from_anchor():
    ...

def test_true_draw_floor_penalizes_draw_suppression_only_on_true_draws():
    ...
```

- [ ] **步骤 2：运行红灯**

运行：`pytest tests/test_p9_market_prior_residual.py -q`

预期：模块不存在或函数不存在。

- [ ] **步骤 3：实现最小 P9 loss**

实现：

- `compute_p9_losses(...)`
- `draw_delta_l2`
- `anchor_kl`
- `true_draw_floor`

- [ ] **步骤 4：运行绿灯**

运行：`pytest tests/test_p9_market_prior_residual.py -q`

## 任务 2：P9 Runner 与协议

**文件：**
- 修改：`tests/test_p9_market_prior_residual.py`
- 创建：`tools/p9_run_matrix.py`

- [ ] 编写 runner 测试：固定 variants、固定 seeds、拒绝 test split、拒绝未知 variant、写 protocol audit。
- [ ] 实现 runner。
- [ ] 运行：`pytest tests/test_p9_market_prior_residual.py -q`

## 任务 3：Smoke 与报告

**文件：**
- 修改：`tools/p9_train_market_prior_residual.py`
- 创建：`docs/superpowers/reports/p9_market_prior_residual_runbook.md`

- [ ] 跑 128 样本 smoke。
- [ ] 确认写出 `report.json`、`val_predictions.csv`、`val_goal_diff_predictions.csv`。
- [ ] 运行 P6/P8/P9 窄测和 `py_compile`。

## No-Go

- 不用 test split。
- 不改 P6 默认脚本行为。
- 不新增 Patch-iTransformer 架构。
- 不追加未注册 variant。
- 不做 posthoc draw threshold。
- 不把 P9 单 seed 或 smoke 提升为 mainline。
