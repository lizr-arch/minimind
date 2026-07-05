# P4 Residual Goal-Diff Distribution Model 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 构建并验证一个“欧赔 1X2 anchor + 亚盘净胜球结构监督”的 P4.1 模型：模型输出 `delta_1x2_logits` 和 `goal_diff_logits`，用 `final_logits = log(p_euro) + delta_logits` 得到最终胜平负概率，并用真实比分监督净胜球分布。

**架构：** 第一阶段只做 `Euro-only` 与 `Euro+Asian` 两组对照，不加入 OU。复用现有 bucket timeline 与 P3b feature extraction，新增 P4 数据/标签工具、残差净胜球 objective probe、训练脚本和报告脚本。主指标仍是 validation 1X2 logloss；goal-diff loss 是结构辅助目标，不替代最终 1X2 评估。

**技术栈：** Python、PyTorch、pytest、现有 `OddsDataset` / `bucketize_sample_v2_for_training` / `eval.odds_metrics`。

---

## 0. 背景与硬边界

### 当前基线

- P3.2 RawPooledMLP label smoothing 0.02: mean val_logloss `0.940862`
- P3.2 RawPooledMLP CE: mean val_logloss `0.941416`
- P3.4 OOF market fusion: mean val_logloss `0.941617`
- P3.2 robust + euro-only Transformer ablation: mean val_logloss `0.941479`
- P3.2 robust full P3b: mean val_logloss `0.942592`

### P4.1 验证问题

1. `Euro-only residual goal-diff` 是否能接近或超过 RawMLP?
2. `Euro+Asian residual goal-diff` 是否稳定优于 `Euro-only`?
3. 独立 `goal_diff=0` 监督是否改善 draw behavior，而不牺牲 1X2 logloss?

### 禁止项

- 不使用 test set。
- 不修改 train / val split。
- P4.1 不加入 OU。
- P4.1 不加入 AH settlement loss。
- 不上 PLE / SAM / masked pretraining。
- 不用 official val 拟合任何 scaler、权重、fusion 或超参。
- 不覆盖已有 `runs/` 结果；所有新结果写到 `runs/p4_residual_goal_diff/`。
- 代码实现前先写失败测试。

---

## 0.1 Required Fixes Before Execution

P4 计划执行前必须先纳入以下修正；任何实现任务都要以这些修正后的定义为准。

**Plan gate verdict:** `P4_PLAN_APPROVED_WITH_FIXES`

1. Euro-only feature isolation:
   - P4.1 第一版直接移除 `market_present_count`。
   - `P4_FEATURES["time"] = ["time_log", "bucket_valid_count"]`。
   - `feature_groups=euro` 不得包含 Asian / OU presence 信息。

2. Anchor safety:
   - `extract_euro_anchor_probs` 只允许使用 `minutes_before_kickoff >= 0` 的有效欧赔事件。
   - 如果没有有效非负时间欧赔，返回 `[1/3, 1/3, 1/3]`。
   - 记录 `anchor_fallback_count`。
   - 额外记录 `negative_time_valid_euro_count`，用于 audit 判断负时间有效欧赔是否大量存在。
   - 不允许在 P4.1 中使用负时间 odds 作为 anchor fallback。

3. Anchor-only baseline:
   - 在 val 上直接评估 `p_euro_anchor`。
   - 报告 `anchor_only_val_logloss`、`anchor_only_brier`、`anchor_only_ECE`、`anchor_only_acc`。
   - P4 residual 模型必须证明自己没有把 anchor 搞坏。

4. Consistency ablation:
   - 每个 P4.1 variant 都跑两组权重：
     - `default`: `diff=0.30, consistency=0.10, delta_l2=0.01`
     - `no_consistency`: `diff=0.30, consistency=0.00, delta_l2=0.01`
   - 以 3-seed mean val_logloss 判断 consistency loss 是否过早约束 final 1X2。

5. Draw / goal-diff diagnostics:
   - 报告 `argmax_draw_count`、`draw_recall`、`mean_p_draw`、`mean_p_draw_on_true_draw`。
   - 报告 `goal_diff_bucket_acc`、`goal_diff_zero_recall`、`p_from_diff_logloss`、`final_vs_diff_kl`。
   - 比较 final `p_draw` 与 `p_from_diff` 的 draw probability。

6. Naming:
   - 本轮报告必须写成 `P4 residual goal-diff objective probe`。
   - 不得写成 `P4 Transformer 成功/失败`；Transformer 版本只有在 objective probe 过关后才进入 `P4.1T`。

---

## 1. 文件结构

### 新增文件

- `tools/p4_goal_diff_utils.py`
  - goal-diff 分桶、从 `q(goal_diff)` 折叠 1X2、欧赔 anchor 提取、P4 loss 计算。

- `model/p4_residual_goal_diff.py`
  - P4 objective-probe 模型：bucket tensor encoder + `delta_1x2_head` + `goal_diff_head`。
  - 只输出 logits，不读原始 row，不计算 loss。

- `tools/p4_train_residual_goal_diff.py`
  - P4.1 训练入口。
  - 支持 `--feature-groups euro|euro,asian`。
  - 支持 `--diff-loss-weight`、`--consistency-loss-weight`、`--delta-l2-weight`。
  - 保存 `report.json`、`best_model.pth`、`val_predictions.csv`、`val_goal_diff_predictions.csv`。

- `tools/p4_generate_report.py`
  - 汇总 P4.0 audit、P4.1-A、P4.1-B 真实 run 结果。
  - 输出 `runs/p4_residual_goal_diff/p4_report.md` 和 `.json`。

- `tools/p4_audit_goal_diff_labels.py`
  - 只读审计比分 label 覆盖率、goal_diff bucket 分布、asian_line 分布。

- `tests/test_p4_goal_diff_utils.py`
  - 单元测试 goal_diff 分桶、折叠、欧赔 anchor、loss 组合。

- `tests/test_p4_residual_goal_diff_model.py`
  - 模型 shape 与 forward 行为测试。

- `tests/test_p4_training_report.py`
  - report schema、no test ids、输出字段测试。

### 修改文件

- 不修改 `model/odds_patch_itransformer_v2.py`。
- 不修改 P3 训练脚本。
- 如需复用 P3b helper，仅 import：
  - `tools.p3b_train_patch_itransformer.bucketize_sample_v2_for_training`
  - `tools.p3b_train_patch_itransformer.apply_feature_group_mask`
  - `tools.p3b_train_patch_itransformer.fit_feature_scaler`
  - `tools.p3b_train_patch_itransformer.apply_feature_scaler`

---

## 2. 数据与标签定义

### 1X2 标签

沿用：

```python
EURO_MAP = {"home": 0, "draw": 1, "away": 2}
y_1x2 = EURO_MAP[row["label"]["euro_result"]]
```

### goal_diff 标签

```python
goal_diff = int(row["label"]["home_goals"]) - int(row["label"]["away_goals"])
```

7 桶：

```text
0: <= -3
1: -2
2: -1
3: 0
4: +1
5: +2
6: >= +3
```

### q(goal_diff) 折叠为 1X2

```python
p_away = q[:, 0] + q[:, 1] + q[:, 2]
p_draw = q[:, 3]
p_home = q[:, 4] + q[:, 5] + q[:, 6]
p_from_diff = torch.stack([p_home, p_draw, p_away], dim=-1)
```

注意最终 1X2 顺序必须是 `[home, draw, away]`。

### 欧赔 anchor

`p_euro` 从 row timeline 中最后一个有效欧赔事件提取：

```python
valid = euro_h > 1.0 and euro_d > 1.0 and euro_a > 1.0
raw = [1/euro_h, 1/euro_d, 1/euro_a]
p_euro = raw / sum(raw)
```

事件选择规则：

1. 使用 `raw_timeline`，没有则 `odds_timeline`。
2. 过滤有效欧赔事件。
3. 选 `minutes_before_kickoff >= 0` 且最小的事件。
4. 如果没有非负时间有效欧赔，返回 `[1/3, 1/3, 1/3]`，并记录 `anchor_fallback_count`。
5. 如果存在负时间有效欧赔，不使用它们作为 anchor，只记录 `negative_time_valid_euro_count`。

---

## 3. Loss 设计

模型输出：

```python
delta_logits: [B, 3]
goal_diff_logits: [B, 7]
```

最终概率：

```python
anchor_logits = torch.log(p_euro.clamp_min(1e-8))
final_logits = anchor_logits + delta_logits
p_final = softmax(final_logits)
```

辅助概率：

```python
q_diff = softmax(goal_diff_logits)
p_from_diff = fold_goal_diff_probs(q_diff)
```

Loss：

```python
L_1x2 = cross_entropy(final_logits, y_1x2)
L_diff = cross_entropy(goal_diff_logits, y_goal_diff_bucket)
L_consistency = kl_div(
    log_softmax(final_logits),
    p_from_diff.detach(),
    reduction="batchmean",
)
L_delta = delta_logits.pow(2).mean()

L = L_1x2
  + diff_loss_weight * L_diff
  + consistency_loss_weight * L_consistency
  + delta_l2_weight * L_delta
```

默认权重：

```text
diff_loss_weight = 0.30
consistency_loss_weight = 0.10
delta_l2_weight = 0.01
```

`p_from_diff.detach()` 是第一版推荐：让最终 1X2 向净胜球分布保持一致，但不让 1X2 anchor 反向污染已经被真实比分监督的 goal_diff head。

---

## 4. 任务分解

### 任务 1：实现 goal_diff utils 的红灯测试

**文件：**
- 创建：`tests/test_p4_goal_diff_utils.py`
- 创建：`tools/p4_goal_diff_utils.py`

- [ ] **步骤 1：编写失败测试**

```python
import torch

from tools.p4_goal_diff_utils import (
    goal_diff_to_bucket,
    fold_goal_diff_probs_to_1x2,
    extract_euro_anchor_probs,
    compute_p4_loss,
)


def test_goal_diff_to_bucket_boundaries():
    assert [goal_diff_to_bucket(x) for x in [-5, -3, -2, -1, 0, 1, 2, 3, 6]] == [0, 0, 1, 2, 3, 4, 5, 6, 6]


def test_fold_goal_diff_probs_to_1x2_uses_home_draw_away_order():
    q = torch.tensor([[0.10, 0.20, 0.15, 0.25, 0.12, 0.10, 0.08]])
    p = fold_goal_diff_probs_to_1x2(q)
    assert torch.allclose(p, torch.tensor([[0.30, 0.25, 0.45]]))


def test_extract_euro_anchor_probs_uses_latest_valid_euro_event():
    row = {
        "raw_timeline": [
            {"minutes_before_kickoff": 120, "euro_h": 2.0, "euro_d": 4.0, "euro_a": 4.0},
            {"minutes_before_kickoff": 10, "euro_h": 1.5, "euro_d": 3.0, "euro_a": 6.0},
        ]
    }
    p, diag = extract_euro_anchor_probs(row)
    assert diag["used_fallback"] is False
    assert torch.allclose(p, torch.tensor([0.5714286, 0.2857143, 0.1428571]), atol=1e-6)


def test_extract_euro_anchor_probs_does_not_use_negative_time_odds():
    row = {
        "raw_timeline": [
            {"minutes_before_kickoff": -5, "euro_h": 1.2, "euro_d": 6.0, "euro_a": 12.0},
        ]
    }
    p, diag = extract_euro_anchor_probs(row)
    assert diag["used_fallback"] is True
    assert diag["negative_time_valid_euro_count"] == 1
    assert torch.allclose(p, torch.full((3,), 1.0 / 3.0))


def test_compute_p4_loss_returns_named_components():
    delta = torch.zeros(2, 3, requires_grad=True)
    diff_logits = torch.zeros(2, 7, requires_grad=True)
    anchor = torch.tensor([[0.5, 0.25, 0.25], [0.3, 0.3, 0.4]])
    y_1x2 = torch.tensor([0, 2])
    y_diff = torch.tensor([4, 2])
    result = compute_p4_loss(delta, diff_logits, anchor, y_1x2, y_diff)
    assert set(result) >= {"loss", "L_1x2", "L_diff", "L_consistency", "L_delta", "p_final", "p_from_diff"}
    result["loss"].backward()
    assert delta.grad is not None
    assert diff_logits.grad is not None
```

- [ ] **步骤 2：运行测试确认失败**

运行：

```powershell
pytest tests/test_p4_goal_diff_utils.py -q
```

预期：FAIL，`ModuleNotFoundError: No module named 'tools.p4_goal_diff_utils'`。

- [ ] **步骤 3：实现最少 utils**

在 `tools/p4_goal_diff_utils.py` 中实现：

```python
import math
import torch
import torch.nn.functional as F

EPS = 1e-8


def goal_diff_to_bucket(goal_diff: int) -> int:
    if goal_diff <= -3:
        return 0
    if goal_diff == -2:
        return 1
    if goal_diff == -1:
        return 2
    if goal_diff == 0:
        return 3
    if goal_diff == 1:
        return 4
    if goal_diff == 2:
        return 5
    return 6


def fold_goal_diff_probs_to_1x2(q: torch.Tensor) -> torch.Tensor:
    p_away = q[:, 0:3].sum(dim=1)
    p_draw = q[:, 3]
    p_home = q[:, 4:7].sum(dim=1)
    return torch.stack([p_home, p_draw, p_away], dim=1).clamp(EPS, 1.0)


def extract_euro_anchor_probs(row: dict) -> tuple[torch.Tensor, dict]:
    timeline = row.get("raw_timeline", []) or row.get("odds_timeline", [])
    valid = []
    for event in timeline:
        h = _as_float(event.get("euro_h"))
        d = _as_float(event.get("euro_d"))
        a = _as_float(event.get("euro_a"))
        if h > 1.0 and d > 1.0 and a > 1.0:
            valid.append((event, h, d, a))
    non_negative = [item for item in valid if _as_float(item[0].get("minutes_before_kickoff")) >= 0]
    negative_count = len(valid) - len(non_negative)
    if not non_negative:
        return torch.full((3,), 1.0 / 3.0), {"used_fallback": True, "negative_time_valid_euro_count": negative_count}
    event, h, d, a = min(non_negative, key=lambda item: _as_float(item[0].get("minutes_before_kickoff")))
    inv = torch.tensor([1.0 / h, 1.0 / d, 1.0 / a], dtype=torch.float32)
    return inv / inv.sum(), {"used_fallback": False, "negative_time_valid_euro_count": negative_count}


def compute_p4_loss(
    delta_logits: torch.Tensor,
    goal_diff_logits: torch.Tensor,
    p_euro_anchor: torch.Tensor,
    y_1x2: torch.Tensor,
    y_goal_diff: torch.Tensor,
    diff_loss_weight: float = 0.30,
    consistency_loss_weight: float = 0.10,
    delta_l2_weight: float = 0.01,
) -> dict:
    final_logits = torch.log(p_euro_anchor.clamp_min(EPS)) + delta_logits
    q_diff = F.softmax(goal_diff_logits, dim=-1).clamp(EPS, 1.0)
    p_from_diff = fold_goal_diff_probs_to_1x2(q_diff)
    L_1x2 = F.cross_entropy(final_logits, y_1x2)
    L_diff = F.cross_entropy(goal_diff_logits, y_goal_diff)
    L_consistency = F.kl_div(F.log_softmax(final_logits, dim=-1), p_from_diff.detach(), reduction="batchmean")
    L_delta = delta_logits.pow(2).mean()
    loss = L_1x2 + diff_loss_weight * L_diff + consistency_loss_weight * L_consistency + delta_l2_weight * L_delta
    return {
        "loss": loss,
        "L_1x2": L_1x2.detach(),
        "L_diff": L_diff.detach(),
        "L_consistency": L_consistency.detach(),
        "L_delta": L_delta.detach(),
        "p_final": F.softmax(final_logits, dim=-1).detach(),
        "p_from_diff": p_from_diff.detach(),
    }


def _as_float(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
```

- [ ] **步骤 4：运行测试确认通过**

运行：

```powershell
pytest tests/test_p4_goal_diff_utils.py -q
```

预期：`6 passed`。

---

### 任务 2：实现 P4 模型 shape 测试与模型文件

**文件：**
- 创建：`tests/test_p4_residual_goal_diff_model.py`
- 创建：`model/p4_residual_goal_diff.py`

- [ ] **步骤 1：编写失败测试**

```python
import torch

from model.p4_residual_goal_diff import ResidualGoalDiffModel


def test_residual_goal_diff_model_outputs_delta_and_goal_diff_logits():
    model = ResidualGoalDiffModel(n_features=17, n_buckets=10, n_agg=2, d_model=64)
    x = torch.randn(4, 10, 17, 2)
    out = model(x)
    assert out["delta_logits"].shape == (4, 3)
    assert out["goal_diff_logits"].shape == (4, 7)
```

- [ ] **步骤 2：运行测试确认失败**

运行：

```powershell
pytest tests/test_p4_residual_goal_diff_model.py -q
```

预期：FAIL，`ModuleNotFoundError`。

- [ ] **步骤 3：实现最少模型**

在 `model/p4_residual_goal_diff.py` 中创建紧凑模型。第一版不复制完整 P3b Transformer，先用 bucket tensor flatten + MLP 诊断目标是否有效：

```python
import torch
from torch import nn


class ResidualGoalDiffModel(nn.Module):
    def __init__(self, n_features: int, n_buckets: int = 10, n_agg: int = 2, d_model: int = 128, dropout: float = 0.15):
        super().__init__()
        input_dim = n_features * n_buckets * n_agg
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.delta_head = nn.Linear(d_model, 3)
        self.goal_diff_head = nn.Linear(d_model, 7)

    def forward(self, x: torch.Tensor) -> dict:
        h = self.encoder(x.flatten(start_dim=1))
        return {
            "delta_logits": self.delta_head(h),
            "goal_diff_logits": self.goal_diff_head(h),
        }
```

说明：如果这个 MLP 版本有效，再做 P4.1T Transformer 版本；第一版先验证 objective，不引入新复杂度。

- [ ] **步骤 4：运行测试确认通过**

运行：

```powershell
pytest tests/test_p4_residual_goal_diff_model.py -q
```

预期：`1 passed`。

---

### 任务 3：实现 P4 数据集构造测试

**文件：**
- 修改：`tests/test_p4_goal_diff_utils.py`
- 创建：`tools/p4_train_residual_goal_diff.py`

- [ ] **步骤 1：补数据构造失败测试**

追加：

```python
from tools.p4_train_residual_goal_diff import build_p4_dataset


def test_build_p4_dataset_returns_bucket_features_anchors_and_labels():
    rows = [
        {
            "match_id": "m1",
            "raw_timeline": [
                {
                    "minutes_before_kickoff": 10,
                    "euro_h": 2.0,
                    "euro_d": 4.0,
                    "euro_a": 4.0,
                    "asian_source": "raw_update",
                    "asian_line": -1.0,
                    "upper_water": 0.9,
                    "lower_water": 1.0,
                }
            ],
            "label": {"euro_result": "home", "home_goals": 2, "away_goals": 0},
        }
    ]
    data = build_p4_dataset(rows, feature_groups="euro,asian")
    assert data["X"].shape[0] == 1
    assert data["X"].shape[1:] == (10, 17, 2)
    assert data["y_1x2"].tolist() == [0]
    assert data["y_goal_diff"].tolist() == [5]
    assert data["p_euro_anchor"].shape == (1, 3)
    assert data["anchor_fallback_count"] == 0
    assert data["negative_time_valid_euro_count"] == 0
```

- [ ] **步骤 2：运行测试确认失败**

运行：

```powershell
pytest tests/test_p4_goal_diff_utils.py::test_build_p4_dataset_returns_bucket_features_anchors_and_labels -q
```

预期：FAIL，`build_p4_dataset` 不存在。

- [ ] **步骤 3：实现 dataset builder**

在 `tools/p4_train_residual_goal_diff.py` 中先实现可 import 的数据函数：

```python
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.odds_patch_itransformer_v2 import FEATURE_DEFS
from tools.p3b_train_patch_itransformer import bucketize_sample_v2_for_training
from tools.p4_goal_diff_utils import extract_euro_anchor_probs, goal_diff_to_bucket

EURO_MAP = {"home": 0, "draw": 1, "away": 2}
FEATURE_INDEX = {name: idx for idx, (name, _) in enumerate(FEATURE_DEFS)}
P4_FEATURES = {
    "euro": ["euro_h", "euro_d", "euro_a", "imp_h", "imp_d", "imp_a", "euro_margin", "has_euro"],
    "asian": ["asian_line", "upper_water", "lower_water", "asian_upper_implied", "asian_lower_implied", "asian_margin", "has_asian"],
    "time": ["time_log", "bucket_valid_count"],
}


def selected_feature_names(feature_groups: str) -> list[str]:
    selected = {part.strip() for part in feature_groups.split(",") if part.strip()}
    names = list(P4_FEATURES["euro"])
    if "asian" in selected:
        names.extend(P4_FEATURES["asian"])
    names.extend(P4_FEATURES["time"])
    return names


def build_p4_dataset(rows: list[dict], feature_groups: str) -> dict:
    names = selected_feature_names(feature_groups)
    indices = [FEATURE_INDEX[name] for name in names]
    X, y_1x2, y_goal_diff, anchors, match_ids = [], [], [], [], []
    fallback_count = 0
    negative_time_valid_euro_count = 0
    for row in rows:
        label = row.get("label", {})
        result = label.get("euro_result")
        if result not in EURO_MAP or "home_goals" not in label or "away_goals" not in label:
            continue
        timeline = row.get("raw_timeline", []) or row.get("odds_timeline", [])
        xb = bucketize_sample_v2_for_training(timeline, clean_missing_markets=True)
        X.append(xb[:, indices, :])
        y_1x2.append(EURO_MAP[result])
        y_goal_diff.append(goal_diff_to_bucket(int(label["home_goals"]) - int(label["away_goals"])))
        anchor, anchor_diag = extract_euro_anchor_probs(row)
        fallback_count += int(anchor_diag["used_fallback"])
        negative_time_valid_euro_count += int(anchor_diag["negative_time_valid_euro_count"])
        anchors.append(anchor)
        match_ids.append(str(row.get("match_id", len(match_ids))))
    if not X:
        raise ValueError("No P4 labelled rows available")
    return {
        "X": torch.stack(X),
        "y_1x2": torch.tensor(y_1x2, dtype=torch.long),
        "y_goal_diff": torch.tensor(y_goal_diff, dtype=torch.long),
        "p_euro_anchor": torch.stack(anchors),
        "match_ids": match_ids,
        "feature_names": names,
        "anchor_fallback_count": fallback_count,
        "negative_time_valid_euro_count": negative_time_valid_euro_count,
    }
```

- [ ] **步骤 4：运行测试确认通过**

运行：

```powershell
pytest tests/test_p4_goal_diff_utils.py -q
```

预期：`5 passed`。

---

### 任务 4：实现训练脚本 CLI 与 report schema

**文件：**
- 修改：`tools/p4_train_residual_goal_diff.py`
- 创建：`tests/test_p4_training_report.py`

- [ ] **步骤 1：编写 report schema 测试**

```python
from tools.p4_train_residual_goal_diff import build_report_payload


def test_p4_report_payload_contains_required_sections():
    payload = build_report_payload(
        config={"feature_groups": "euro"},
        data={"train_samples": 10, "val_samples": 4, "test_ids_used": False},
        best_epoch=2,
        best_val_logloss=0.95,
        anchor_only_baseline={
            "anchor_only_val_logloss": 0.96,
            "anchor_only_brier": 0.56,
            "anchor_only_ECE": 0.02,
            "anchor_only_acc": 0.57,
            "argmax_draw_count": 0,
            "draw_recall": 0.0,
            "mean_p_draw": 0.25,
            "mean_p_draw_on_true_draw": 0.27,
        },
        train_metrics={"logloss": 0.9},
        val_metrics={
            "logloss": 0.95,
            "brier": 0.55,
            "ece": 0.02,
            "accuracy": 0.58,
            "argmax_draw_count": 0,
            "draw_recall": 0.0,
            "mean_p_draw": 0.25,
            "mean_p_draw_on_true_draw": 0.27,
        },
        goal_diff_metrics={
            "bucket_acc": 0.3,
            "zero_recall": 0.2,
            "p_from_diff_logloss": 0.98,
            "final_vs_diff_kl": 0.01,
        },
        history=[],
        warnings=[],
    )
    assert set(payload) >= {
        "phase",
        "model",
        "config",
        "data",
        "best_epoch",
        "best_val_logloss",
        "train_metrics",
        "val_metrics",
        "goal_diff_metrics",
        "anchor_only_baseline",
        "history",
        "warnings",
    }
    assert payload["data"].get("test_ids_used") is False
    assert set(payload["anchor_only_baseline"]) >= {
        "anchor_only_val_logloss",
        "anchor_only_brier",
        "anchor_only_ECE",
        "anchor_only_acc",
    }
```

- [ ] **步骤 2：运行测试确认失败**

运行：

```powershell
pytest tests/test_p4_training_report.py -q
```

预期：FAIL，`build_report_payload` 不存在。

- [ ] **步骤 3：实现训练 CLI**

在 `tools/p4_train_residual_goal_diff.py` 中补齐：

- `set_deterministic_seed`
- `prepare_output_dir`
- `load_split_ids`
- `load_rows_for_ids`
- `evaluate`
- `write_val_predictions`
- `build_report_payload`
- `main`

CLI 参数：

```text
--data
--train-ids
--val-ids
--feature-groups euro|euro,asian
--epochs 50
--batch-size 128
--lr 0.001
--device cuda
--seed 42
--d-model 128
--dropout 0.15
--diff-loss-weight 0.30
--consistency-loss-weight 0.10
--delta-l2-weight 0.01
--scaling robust|standard|none
--max-samples 0
--out-dir
--allow-overwrite
```

训练循环每 epoch 记录：

```python
{
    "epoch": epoch,
    "train_loss": ...,
    "train_L_1x2": ...,
    "train_L_diff": ...,
    "train_L_consistency": ...,
    "val_logloss": ...,
    "val_brier": ...,
    "val_ece": ...,
    "val_acc": ...,
    "val_diff_nll": ...,
    "val_diff_acc": ...,
    "val_goal_diff_zero_recall": ...,
    "val_p_from_diff_logloss": ...,
    "val_final_vs_diff_kl": ...,
    "val_argmax_draw_count": ...,
    "val_draw_recall": ...,
    "val_mean_p_draw": ...,
    "val_mean_p_draw_on_true_draw": ...,
}
```

best model 选择标准：

```text
primary = val_logloss from p_final
```

不按 diff loss 选 best。

- [ ] **步骤 4：运行 schema 测试通过**

运行：

```powershell
pytest tests/test_p4_training_report.py -q
```

预期：`1 passed`。

---

### 任务 5：P4.0 label / market audit

**文件：**
- 创建：`tools/p4_audit_goal_diff_labels.py`
- 创建输出目录：`runs/p4_residual_goal_diff/audit`

- [ ] **步骤 1：实现 audit 脚本**

脚本输出：

```json
{
  "train": {
    "rows": 0,
    "labelled_rows": 0,
    "missing_score_rows": 0,
    "goal_diff_bucket_counts": {},
    "asian_present_rate": 0.0,
    "asian_line_top_values": [],
    "anchor_fallback_count": 0,
    "negative_time_valid_euro_count": 0
  },
  "val": {
    "rows": 0,
    "labelled_rows": 0,
    "missing_score_rows": 0,
    "goal_diff_bucket_counts": {},
    "asian_present_rate": 0.0,
    "asian_line_top_values": [],
    "anchor_fallback_count": 0,
    "negative_time_valid_euro_count": 0
  },
  "test_ids_used": false
}
```

- [ ] **步骤 2：运行 audit**

运行：

```powershell
New-Item -ItemType Directory -Force runs/p4_residual_goal_diff/audit, runs/p4_residual_goal_diff/logs
python tools/p4_audit_goal_diff_labels.py `
  --data data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl `
  --train-ids data/odds_real/splits_v6/train_match_ids.txt `
  --val-ids data/odds_real/splits_v6/val_match_ids.txt `
  --out-json runs/p4_residual_goal_diff/audit/goal_diff_audit.json `
  --out-md runs/p4_residual_goal_diff/audit/goal_diff_audit.md `
  *> runs/p4_residual_goal_diff/logs/p4_audit.log
```

预期：

- `goal_diff_audit.json` 存在。
- `missing_score_rows == 0` 或报告明确列出缺失比例。
- `test_ids_used == false`。
- `negative_time_valid_euro_count` 明确记录；P4 anchor 不使用这些事件。

如果 score label 缺失严重，停止 P4.1，写 `BLOCKED_SCORE_LABELS`。

---

### 任务 6：Smoke 训练验证

**文件：**
- 使用：`tools/p4_train_residual_goal_diff.py`

- [ ] **步骤 1：Euro-only smoke**

运行：

```powershell
python tools/p4_train_residual_goal_diff.py `
  --data data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl `
  --train-ids data/odds_real/splits_v6/train_match_ids.txt `
  --val-ids data/odds_real/splits_v6/val_match_ids.txt `
  --feature-groups euro `
  --epochs 2 `
  --max-samples 256 `
  --device cuda `
  --seed 42 `
  --out-dir runs/p4_residual_goal_diff/smoke/p4_euro_seed42 `
  *> runs/p4_residual_goal_diff/logs/smoke_p4_euro_seed42.log
```

预期：

- `report.json` 存在。
- `best_model.pth` 存在。
- `val_predictions.csv` 存在。
- report 中 `data.test_ids_used == false`。

- [ ] **步骤 2：Euro+Asian smoke**

运行：

```powershell
python tools/p4_train_residual_goal_diff.py `
  --data data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl `
  --train-ids data/odds_real/splits_v6/train_match_ids.txt `
  --val-ids data/odds_real/splits_v6/val_match_ids.txt `
  --feature-groups euro,asian `
  --epochs 2 `
  --max-samples 256 `
  --device cuda `
  --seed 42 `
  --out-dir runs/p4_residual_goal_diff/smoke/p4_euro_asian_seed42 `
  *> runs/p4_residual_goal_diff/logs/smoke_p4_euro_asian_seed42.log
```

预期同上。

---

### 任务 7：P4.1-A Euro-only 3-seed 正式实验与 consistency ablation

**文件：**
- 输出目录：`runs/p4_residual_goal_diff/p4_1_runs/`
- 日志目录：`runs/p4_residual_goal_diff/logs/`

- [ ] **步骤 1：运行 default 3 seeds**

seeds:

```text
42, 123, 2025
```

命令模板：

```powershell
python tools/p4_train_residual_goal_diff.py `
  --data data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl `
  --train-ids data/odds_real/splits_v6/train_match_ids.txt `
  --val-ids data/odds_real/splits_v6/val_match_ids.txt `
  --feature-groups euro `
  --epochs 50 `
  --device cuda `
  --seed <SEED> `
  --diff-loss-weight 0.30 `
  --consistency-loss-weight 0.10 `
  --delta-l2-weight 0.01 `
  --scaling robust `
  --out-dir runs/p4_residual_goal_diff/p4_1_runs/p4_euro_default_seed<SEED> `
  *> runs/p4_residual_goal_diff/logs/p4_euro_default_seed<SEED>.log
```

- [ ] **步骤 2：运行 no-consistency 3 seeds**

命令模板：

```powershell
python tools/p4_train_residual_goal_diff.py `
  --data data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl `
  --train-ids data/odds_real/splits_v6/train_match_ids.txt `
  --val-ids data/odds_real/splits_v6/val_match_ids.txt `
  --feature-groups euro `
  --epochs 50 `
  --device cuda `
  --seed <SEED> `
  --diff-loss-weight 0.30 `
  --consistency-loss-weight 0.00 `
  --delta-l2-weight 0.01 `
  --scaling robust `
  --out-dir runs/p4_residual_goal_diff/p4_1_runs/p4_euro_no_consistency_seed<SEED> `
  *> runs/p4_residual_goal_diff/logs/p4_euro_no_consistency_seed<SEED>.log
```

- [ ] **步骤 3：检查产物**

每个 seed 必须有：

- `report.json`
- `best_model.pth`
- `val_predictions.csv`
- `val_goal_diff_predictions.csv`

---

### 任务 8：P4.1-B Euro+Asian 3-seed 正式实验与 consistency ablation

**文件：**
- 输出目录：`runs/p4_residual_goal_diff/p4_1_runs/`
- 日志目录：`runs/p4_residual_goal_diff/logs/`

- [ ] **步骤 1：运行 default 3 seeds**

命令模板：

```powershell
python tools/p4_train_residual_goal_diff.py `
  --data data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl `
  --train-ids data/odds_real/splits_v6/train_match_ids.txt `
  --val-ids data/odds_real/splits_v6/val_match_ids.txt `
  --feature-groups euro,asian `
  --epochs 50 `
  --device cuda `
  --seed <SEED> `
  --diff-loss-weight 0.30 `
  --consistency-loss-weight 0.10 `
  --delta-l2-weight 0.01 `
  --scaling robust `
  --out-dir runs/p4_residual_goal_diff/p4_1_runs/p4_euro_asian_default_seed<SEED> `
  *> runs/p4_residual_goal_diff/logs/p4_euro_asian_default_seed<SEED>.log
```

- [ ] **步骤 2：运行 no-consistency 3 seeds**

命令模板：

```powershell
python tools/p4_train_residual_goal_diff.py `
  --data data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl `
  --train-ids data/odds_real/splits_v6/train_match_ids.txt `
  --val-ids data/odds_real/splits_v6/val_match_ids.txt `
  --feature-groups euro,asian `
  --epochs 50 `
  --device cuda `
  --seed <SEED> `
  --diff-loss-weight 0.30 `
  --consistency-loss-weight 0.00 `
  --delta-l2-weight 0.01 `
  --scaling robust `
  --out-dir runs/p4_residual_goal_diff/p4_1_runs/p4_euro_asian_no_consistency_seed<SEED> `
  *> runs/p4_residual_goal_diff/logs/p4_euro_asian_no_consistency_seed<SEED>.log
```

- [ ] **步骤 3：检查产物**

同任务 7。

---

### 任务 9：生成 P4.1 汇总报告

**文件：**
- 创建：`tools/p4_generate_report.py`
- 输出：
  - `runs/p4_residual_goal_diff/p4_report.json`
  - `runs/p4_residual_goal_diff/p4_report.md`

- [ ] **步骤 1：实现报告脚本**

报告 JSON 至少包含：

```json
{
  "inputs": {
    "data": "...",
    "train_ids": "...",
    "val_ids": "...",
    "test_ids_used": false
  },
  "audit": {
    "anchor_fallback_count": 0,
    "negative_time_valid_euro_count": 0
  },
  "anchor_only_baseline": {
    "anchor_only_val_logloss": 0.0,
    "anchor_only_brier": 0.0,
    "anchor_only_ECE": 0.0,
    "anchor_only_acc": 0.0,
    "argmax_draw_count": 0,
    "draw_recall": 0.0,
    "mean_p_draw": 0.0,
    "mean_p_draw_on_true_draw": 0.0
  },
  "p4_1": {
    "variants": [],
    "comparison": {
      "best_euro_variant": "",
      "best_euro_asian_variant": "",
      "asian_delta_vs_best_euro": 0.0,
      "best_p4_delta_vs_anchor_only": 0.0,
      "consistency_ablation": {}
    }
  },
  "best_model": {},
  "verdict": [],
  "next_actions": []
}
```

每个 variant 统计：

```text
name
feature_groups
loss_config
seeds
mean_val_logloss
std_val_logloss
best_val_logloss
mean_brier
mean_ECE
mean_val_acc
mean_diff_nll
mean_diff_acc
mean_argmax_draw_count
mean_draw_recall
mean_p_draw
mean_p_draw_on_true_draw
mean_goal_diff_bucket_acc
mean_goal_diff_zero_recall
mean_p_from_diff_logloss
mean_final_vs_diff_kl
mean_delta_l2
mean_delta_vs_anchor_only_logloss
```

`p4_report.md` 至少输出一张主表：

```text
Model / config                  mean val_logloss
Euro anchor only                ...
RawMLP label smoothing          0.940862
P3.4 OOF fusion                 0.941617
P4 euro default                 ...
P4 euro no_consistency          ...
P4 euro+asian default           ...
P4 euro+asian no_consistency    ...
```

另输出一张 draw / goal-diff 诊断表，比较每个 P4 variant 的：

```text
argmax_draw_count
draw_recall
mean_p_draw
mean_p_draw_on_true_draw
goal_diff_bucket_acc
goal_diff_zero_recall
p_from_diff_logloss
final_vs_diff_kl
```

报告脚本只读取真实 run 产物；如果某个 run 缺失，必须把该 variant 标为 failed / incomplete，不得用默认数值补齐。

- [ ] **步骤 2：运行报告**

```powershell
python tools/p4_generate_report.py `
  --root runs/p4_residual_goal_diff `
  --raw-mlp-label-smoothing-logloss 0.940862 `
  --p3-4-oof-fusion-logloss 0.941617 `
  *> runs/p4_residual_goal_diff/logs/p4_generate_report.log
```

- [ ] **步骤 3：验收报告**

预期：

- `p4_report.json` 存在。
- `p4_report.md` 存在。
- `anchor_only_baseline` 包含 `anchor_only_val_logloss`、`anchor_only_brier`、`anchor_only_ECE`、`anchor_only_acc`。
- `p4_1.variants` 包含 `euro_default`、`euro_no_consistency`、`euro_asian_default`、`euro_asian_no_consistency`。
- `p4_1.comparison.consistency_ablation` 同时包含 Euro-only 和 Euro+Asian 的 default vs no-consistency 对比。
- `test_ids_used == false`。

---

## 5. Verdict 规则

允许 verdict：

```text
P4_EURO_RESIDUAL_BEATS_RAWMLP
P4_EURO_RESIDUAL_CLOSE_TO_RAWMLP
P4_EURO_RESIDUAL_FAILS_BASELINE
P4_ASIAN_GOAL_DIFF_HELPS
P4_ASIAN_GOAL_DIFF_NO_GAIN
P4_ASIAN_GOAL_DIFF_HURTS
P4_RESIDUAL_IMPROVES_ANCHOR
P4_RESIDUAL_NO_ANCHOR_GAIN
P4_RESIDUAL_HURTS_ANCHOR
P4_CONSISTENCY_HELPS
P4_CONSISTENCY_NO_GAIN
P4_CONSISTENCY_HURTS
P4_READY_FOR_AH_SETTLEMENT_LOSS
P4_STOP_BEFORE_AH_SETTLEMENT
DRAW_RECALL_IMPROVED_WITHOUT_LOGLOSS_HURT
DRAW_STILL_COLLAPSED
RAW_MLP_STILL_BEST
P4_NEW_BEST
```

判定：

```text
best_euro = min(euro_default.mean_val_logloss, euro_no_consistency.mean_val_logloss)
best_euro_asian = min(euro_asian_default.mean_val_logloss, euro_asian_no_consistency.mean_val_logloss)
best_p4 = min(best_euro, best_euro_asian)

if best_euro < 0.940862:
  P4_EURO_RESIDUAL_BEATS_RAWMLP
elif best_euro <= 0.941862:
  P4_EURO_RESIDUAL_CLOSE_TO_RAWMLP
else:
  P4_EURO_RESIDUAL_FAILS_BASELINE

if best_euro_asian < best_euro - 0.0003:
  P4_ASIAN_GOAL_DIFF_HELPS
elif best_euro_asian > best_euro + 0.0003:
  P4_ASIAN_GOAL_DIFF_HURTS
else:
  P4_ASIAN_GOAL_DIFF_NO_GAIN

if best_p4 < anchor_only_val_logloss - 0.0003:
  P4_RESIDUAL_IMPROVES_ANCHOR
elif best_p4 > anchor_only_val_logloss + 0.0003:
  P4_RESIDUAL_HURTS_ANCHOR
else:
  P4_RESIDUAL_NO_ANCHOR_GAIN

best_default = min(euro_default.mean_val_logloss, euro_asian_default.mean_val_logloss)
best_no_consistency = min(euro_no_consistency.mean_val_logloss, euro_asian_no_consistency.mean_val_logloss)
if best_default < best_no_consistency - 0.0003:
  P4_CONSISTENCY_HELPS
elif best_default > best_no_consistency + 0.0003:
  P4_CONSISTENCY_HURTS
else:
  P4_CONSISTENCY_NO_GAIN

if P4_ASIAN_GOAL_DIFF_HELPS:
  P4_READY_FOR_AH_SETTLEMENT_LOSS
else:
  P4_STOP_BEFORE_AH_SETTLEMENT

if best_p4 < 0.940862:
  P4_NEW_BEST
else:
  RAW_MLP_STILL_BEST

if best_p4.draw_recall > anchor_only.draw_recall + 0.02 and best_p4.mean_val_logloss <= anchor_only_val_logloss + 0.0003:
  DRAW_RECALL_IMPROVED_WITHOUT_LOGLOSS_HURT
elif best_p4.argmax_draw_count == 0 or best_p4.draw_recall <= anchor_only.draw_recall + 0.005:
  DRAW_STILL_COLLAPSED
```

---

## 6. P4.2 只在 P4.1-B 过关时执行

P4.2 不属于本轮必须实现项。只有满足：

```text
P4_ASIAN_GOAL_DIFF_HELPS
P4_RESIDUAL_IMPROVES_ANCHOR or P4_RESIDUAL_NO_ANCHOR_GAIN
```

才进入 P4.2。

P4.2 增加：

```text
L_ah = MSE(E_payoff_from_q_diff, true_payoff)
```

并考虑把 goal_diff bucket 扩到 9 或 11 桶，避免大盘口在 `>=+3` / `<=-3` 尾桶中丢失结算粒度。

---

## 7. 最终验证命令

完整测试：

```powershell
pytest tests/test_p4_goal_diff_utils.py tests/test_p4_residual_goal_diff_model.py tests/test_p4_training_report.py -q
pytest tests/test_p3_4_oof_market_fusion.py tests/test_p3_3_market_fusion.py tests/test_p3_2_scaling_calibration.py tests/test_p3_diagnostics_tools.py tests/test_p3_raw_pooled_mlp_baseline.py tests/test_p3_training_report_scope.py -q
```

报告完整性检查：

```powershell
python - <<'PY'
import json
from pathlib import Path
p = Path("runs/p4_residual_goal_diff/p4_report.json")
r = json.loads(p.read_text(encoding="utf-8"))
assert r["inputs"]["test_ids_used"] is False
assert set(r["anchor_only_baseline"]) >= {
    "anchor_only_val_logloss",
    "anchor_only_brier",
    "anchor_only_ECE",
    "anchor_only_acc",
}
names = {v["name"] for v in r["p4_1"]["variants"]}
assert {
    "euro_default",
    "euro_no_consistency",
    "euro_asian_default",
    "euro_asian_no_consistency",
} <= names
assert "consistency_ablation" in r["p4_1"]["comparison"]
print("p4_report_ok", r["best_model"], r["verdict"])
PY
```

---

## 8. 最终输出格式

完成后终端只输出 20 行以内：

```text
P4 report markdown: runs/p4_residual_goal_diff/p4_report.md
P4 report json: runs/p4_residual_goal_diff/p4_report.json
Anchor-only val_logloss: ...
P4.1-A best Euro config: ... mean val_logloss: ...
P4.1-B best Euro+Asian config: ... mean val_logloss: ...
RawMLP label smoothing baseline: 0.940862
P3.4 OOF fusion baseline: 0.941617
Best P4 variant: ...
Asian goal-diff delta: ...
Consistency ablation: ...
Draw recall delta: ...
Final-vs-diff KL: ...
Verdict: ...
Next action: ...
```

---

## 9. 自检

- 规格覆盖：
  - 欧赔 anchor 与 anchor-only baseline：任务 1、任务 3、任务 4、任务 9。
  - 净胜球分布：任务 1、任务 2、任务 4。
  - Euro-only / Euro+Asian 对照：任务 7、任务 8。
  - consistency ablation：任务 7、任务 8、任务 9。
  - draw / goal-diff diagnostics：任务 4、任务 9。
  - 不使用 test set：任务 4、任务 5、任务 9、最终验证。
  - 不使用负时间欧赔作为 anchor：任务 1、任务 3、任务 9。
  - P4.2 settlement loss 延后：第 6 节。

- 范围控制：
  - 不加入 OU。
  - Euro-only 不包含 Asian / OU presence；`market_present_count` 已从 P4.1 time features 移除。
  - 不加入 AH settlement loss。
  - 不加入新复杂训练技巧。
  - 第一版用 MLP encoder 验证 objective probe；如果 objective 过关，再复制到 Transformer。

- 主要风险：
  - 如果真实比分 label 缺失，P4.1 必须停止。
  - 如果 Euro+Asian 不优于 Euro-only，不进入 P4.2。
  - 如果 consistency loss 使 logloss 变差，后续默认关闭 consistency，保留 goal-diff auxiliary supervision。
  - 如果 residual 模型劣于 anchor-only，优先诊断 delta head / regularization，而不是继续扩大模型。
