# Phase 1b Step 2 — Event-Driven Collator 规格书

> 此文档即提示词，可直接发给实现者。

---

## 项目归属

| 项 | 值 |
|---|---|
| **项目类型** | 大模型训练项目 |
| **开发路径** | `D:\code\git\betmind\minimind` |
| **验收路径** | `D:\code\git\betmind\minimind`（同一路径，本地跑测试） |
| **依赖** | Step 1 完成后的 `OddsDataset`（v6_event schema） |

---

## 任务

实现事件驱动的 collator，支持变长序列的 batch 处理、attention mask 生成、以及 lead-lag 特征的计算。

## 边界

**在范围内：**
- 变长序列 padding + attention mask
- Lead-lag 特征计算（跨庄家比对 Pinnacle timeline）
- 欧亚一致性特征计算
- 单庄家事件序列的 batch 组装

**不在范围内：**
- 特征提取（Step 1 已做）
- 模型训练逻辑
- 多庄家融合（本 step 只做单庄家序列，lead-lag 是特征不是多输入）

## 输入

### Step 1 的输出

每个 sample 由 OddsDataset 产出：
```python
sample = {
    "features": torch.Tensor,      # [T, 31] 含 padding（已截断到 max_seq_len）
    "missing_mask": torch.Tensor,  # [T, 31]
    "attention_mask": torch.Tensor, # [T] 1=有效事件, 0=padding（由 OddsDataset 预生成）
    "euro_label": int,             # 0/1/2
    "asian_label": int,            # 0-4
    "match_id": str,
    "bookmaker_id": str,
    "league_id": str,
    # 原始 timeline（用于 lead-lag 计算）
    "raw_timeline": list[dict],    # ≤ max_seq_len 个事件 dict
}
```

### Lead-lag 计算需要的数据

```python
# 外部索引：match_id → Pinnacle 的 timeline
# 在 collator 初始化时构建
pinnacle_index: dict[str, list[dict]] = {
    "match_001": [事件序列],  # 按 minutes_before_kickoff 降序
    ...
}
```

## 输出

```python
batch = {
    "features": torch.Tensor,       # [B, T_max, 33]  31 + 2 新特征
    "missing_mask": torch.Tensor,   # [B, T_max, 31]  (mask 不包含新特征)
    "attention_mask": torch.Tensor, # [B, T_max]  1=有效, 0=pad
    "euro_labels": torch.Tensor,    # [B] long
    "asian_labels": torch.Tensor,   # [B] long
    "match_ids": list[str],
    "league_ids": list[str],
    # 以下为 monitoring/debug（不参与梯度）
    "event_counts": torch.Tensor,   # [B] 每个样本的实际事件数
}
```

## 功能点

### F1：变长序列 Padding

```python
class OddsEventCollator:
    def __call__(self, batch: list[dict]) -> dict:
        """
        1. 找到 batch 中最长的序列长度 T_max
        2. 将所有样本的 features [T_i, 31] pad 到 [T_max, 31]（右侧填 0）
        3. attention_mask [T_i] pad 到 [T_max]（右侧填 0）
        4. missing_mask 同理
        """
```

### F2：Lead-Lag 特征（核心新增）

```python
def compute_leadlag_feature(
    event: dict,
    pinnacle_timeline: list[dict] | None,
) -> float:
    """
    判断在此事件时刻，Pinnacle 是否已经朝同一方向变动。
    
    算法：
    1. 如果 bookmaker 本身就是 Pinnacle，返回 0.0
    2. 如果 pinnacle_timeline 为 None（无 Pinnacle 数据），返回 0.0
    3. 找到 Pinnacle timeline 中 minutes_before_kickoff <= event.minutes_before_kickoff
       的事件（即"此刻及之后"的 Pinnacle 事件——更靠近比赛）
    4. 取这些事件中最早的一个（即 Pinnacle 在此刻之后的第一反应）
    5. 取 Pinnacle 在此刻之前的最后一个事件
    6. 计算 Pinnacle 的 euro_h 变化方向：
       - 从"此刻之前"到"此刻之后第一反应"，euro_h 下降 → pinnacle_direction = +1（看好主队）
       - euro_h 上升 → pinnacle_direction = -1
       - 无变化或无足够数据 → 0
    7. 计算当前庄家此事件的 euro_h 变化方向（与上一事件比）：
       - euro_h 下降 → our_direction = +1
       - euro_h 上升 → our_direction = -1
       - 不变 → 0
    8. 返回：
       - +1.0 如果 pinnacle_direction != 0 且 pinnacle_direction == our_direction
       - -1.0 如果 pinnacle_direction != 0 且 pinnacle_direction != our_direction
       - 0.0 其他情况
    """
```

**注意：信息泄露控制。** lead-lag 特征只用"此刻及之前"的 Pinnacle 信息。具体来说：
- 对于时刻 T 的事件，只使用 Pinnacle timeline 中 `minutes_before_kickoff >= T` 的事件（即不晚于 T 发生的事件）
- 不使用 T 之后的 Pinnacle 事件（那是未来信息）

### F3：欧亚一致性特征

```python
def compute_euro_asian_alignment(event: dict, prev_event: dict | None) -> float:
    """
    判断欧赔和亚盘在当前事件中是否朝同一方向。
    
    算法：
    1. 如果 prev_event 为 None，返回 0.0
    2. 计算欧赔主胜方向：
       euro_h 下降 → euro_dir = +1（看好主队）
       euro_h 上升 → euro_dir = -1
       不变 → 0
    3. 计算亚盘主队方向：
       asian_line 减小（如 -0.5→-0.75，让球更多）→ asian_dir = +1（看好主队）
       asian_line 增大（让球减少）→ asian_dir = -1
       若 asian_line 不变但 upper_water 下降 → asian_dir = +1
       若 asian_line 不变但 upper_water 上升 → asian_dir = -1
       其他 → 0
    4. 返回：
       +1.0 如果 euro_dir != 0 且 euro_dir == asian_dir
       -1.0 如果 euro_dir != 0 且 euro_dir != asian_dir
       0.0 其他情况
    """
```

### F4：特征拼接

每个事件在 31 维基础上追加 2 维：
- 第 31 维：lead_lag
- 第 32 维：euro_asian_alignment

最终 `features` 形状为 `[B, T_max, 33]`。

## 测试内容

### T1：Padding 正确性
```python
# 创建 3 个样本：T=5, T=10, T=3
# collate 后验证 features.shape == (3, 10, 33)
# 验证 attention_mask 正确（前 5/10/3 为 1，其余为 0）
# 验证短样本的 pad 区域 features 全为 0
```

### T2：Lead-Lag 特征
```python
# 构造 Pinnacle timeline：在 T=60min 时 euro_h 从 2.0 降到 1.8（看好主队）
# 构造 Bet365 timeline：在 T=30min 时 euro_h 从 2.1 降到 1.9（同向）
# 验证 T=30min 事件的 lead_lag = +1.0

# 再构造：Bet365 在 T=30min 时 euro_h 从 1.9 升到 2.1（反向）
# 验证 lead_lag = -1.0

# 再构造：Bet365 是 Pinnacle 自身
# 验证 lead_lag = 0.0

# 再构造：无 Pinnacle 数据
# 验证 lead_lag = 0.0
```

### T3：信息泄露检查
```python
# Pinnacle 在 T=20min 时才变动
# Bet365 在 T=60min 时的事件
# 验证 T=60min 的 lead_lag 不使用 T=20min 的 Pinnacle 信息（因为那是未来）
# 即：T=60min 时 Pinnacle 还没动，lead_lag 应为 0.0
```

### T4：欧亚一致性
```python
# 构造：euro_h 降（看好主队），asian_line 从 0.0→-0.5（也看好主队）
# 验证 alignment = +1.0

# 构造：euro_h 降（看好主队），asian_line 从 -0.5→0.0（看好客队）
# 验证 alignment = -1.0

# 构造：euro_h 不变，asian_line 变
# 验证 alignment = 0.0（euro_dir=0 时返回 0）
```

### T5：边界情况
```python
# 序列长度为 1（只有开盘事件）：lead_lag 和 alignment 均为 0.0
# 所有事件 OU 缺失：不影响 collator（特征已是 0 + mask=1）
# max_seq_len 为 0（空序列）：应被 OddsDataset 过滤掉，collator 不应收到
```

## 验收标准

- [ ] 5 个测试通过
- [ ] Lead-lag 特征不包含未来信息（T3 通过）
- [ ] 与现有 `OddsCollator` 的接口兼容（可以替换使用）
- [ ] 通过 mypy/pylint 检查（无类型错误）

## 文件位置

新建文件：`dataset/odds_collator_v6_event.py`
- 包含 `OddsEventCollator` 类
- 包含 `compute_leadlag_feature()` 和 `compute_euro_asian_alignment()` 函数
- 包含 `build_pinnacle_index()` 函数（从 dataset.samples 构建 Pinnacle 索引）

测试文件：新建 `tests/test_v6_event_collator.py`
