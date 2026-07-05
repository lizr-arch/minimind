# Phase 1b Step 1 — 数据适配器规格书

> 此文档即提示词，可直接发给实现者。

---

## 项目归属

| 项 | 值 |
|---|---|
| **项目类型** | 大模型训练项目 |
| **开发路径** | `D:\code\git\betmind\minimind` |
| **验收路径** | `D:\code\git\betmind\minimind`（同一路径，本地跑测试） |
| **依赖数据** | 数据项目 `D:\code\git\odds-data-probe` 产出的 `titan007_pure_v4_market_semantics_v2.jsonl`（已就绪，放在 `minimind/data/odds_real/` 下） |

---

## 任务

扩展现有 `OddsDataset`，使其支持 `titan007_pure_v4_market_semantics_v2.jsonl` 格式的事件驱动序列数据。

## 边界

**在范围内：**
- 添加新的 feature schema `v6_event`（~31 维 per-event 特征）
- 添加新的 `_event_to_features_v6()` 函数
- 支持新数据中的 snapshot_type、market_updated、*_source 字段
- 支持大小球（OU）缺失标记（复用现有 missing_mask 机制）
- 向后兼容：不影响现有 v3/v4/v5 schema 的代码路径

**不在范围内：**
- Lead-lag 特征计算（留给 Step 2 collator）
- 模型架构修改
- 训练脚本
- 数据清洗或修复

## 输入

```python
# 数据文件
path = "data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl"

# 每行格式（关键字段）
{
    "match_id": "world_cup_1127595",
    "league_id": "world_cup",
    "bookmaker_id": "Bet365",
    "bookmaker_type": "soft",          # sharp/soft
    "kickoff_time": "2015-06-02T09:00:00",
    "season": 2015,
    "competition_type": "continental", # league/cup/continental
    "is_knockout": true,
    "odds_timeline": [
        {
            "minutes_before_kickoff": 1683.0,
            "snapshot_type": "opening",     # "opening" | "movement"
            "market_updated": "1x2",        # "1x2" | "asian" | "over_under"
            # Euro
            "euro_h": 15.0, "euro_d": 7.0, "euro_a": 1.16,
            "has_euro": true,
            "euro_source": "raw_update",    # "raw_update" | "forward_fill" | "missing"
            # Asian
            "asian_line": 0.0,
            "upper_water": 1.0, "lower_water": 1.0,
            "has_asian": true,
            "asian_source": "forward_fill",
            # Over/Under
            "over_under_line": null,        # null when missing
            "over_water": null,
            "under_water": null,
            "has_over_under": false,
            "over_under_source": "missing",
            # Per-field missing flags
            "euro_h_missing": false, "euro_d_missing": false, "euro_a_missing": false,
            "asian_line_missing": false, "upper_water_missing": false, "lower_water_missing": false,
            "over_under_line_missing": true, "over_water_missing": true, "under_water_missing": true
        },
        ...
    ],
    "label": {
        "euro_result": "away",              # "home" | "draw" | "away"
        "asian_result": "full_loss",        # 5-class asian result
        "home_goals": 0, "away_goals": 6
    }
}
```

## 输出

### 1. Feature Schema `v6_event`

**特征维度：31**

按以下顺序排列（固定索引，从 0 开始）：

| 索引 | 特征名 | 来源 | null 时填 |
|------|--------|------|-----------|
| 0 | `euro_h` | `event["euro_h"]` | 0.0 |
| 1 | `euro_d` | `event["euro_d"]` | 0.0 |
| 2 | `euro_a` | `event["euro_a"]` | 0.0 |
| 3 | `asian_line` | `event["asian_line"]` | 0.0 |
| 4 | `upper_water` | `event["upper_water"]` | 0.0 |
| 5 | `lower_water` | `event["lower_water"]` | 0.0 |
| 6 | `ou_line` | `event["over_under_line"]` | 0.0 |
| 7 | `ou_over_water` | `event["over_water"]` | 0.0 |
| 8 | `ou_under_water` | `event["under_water"]` | 0.0 |
| 9 | `euro_implied_h` | `1/euro_h / sum(1/euro_*)` | 1/3 |
| 10 | `euro_implied_d` | 同上 | 1/3 |
| 11 | `euro_implied_a` | 同上 | 1/3 |
| 12 | `euro_margin` | `sum(1/euro_*) - 1` | 0.0 |
| 13 | `asian_p_upper` | `1/upper / (1/upper + 1/lower)` | 0.5 |
| 14 | `asian_margin` | `1/upper + 1/lower - 1` | 0.0 |
| 15 | `ou_p_over` | `1/over / (1/over + 1/under)` | 0.5 |
| 16 | `ou_margin` | `1/over + 1/under - 1` | 0.0 |
| 17 | `minutes_before_kickoff` | `event["minutes_before_kickoff"]` | 0.0 |
| 18 | `time_delta_prev` | 距上一个事件的分钟差（首个事件填 0） | 0.0 |
| 19 | `snapshot_is_opening` | `1.0 if snapshot_type=="opening" else 0.0` | 0.0 |
| 20 | `market_is_euro` | `1.0 if market_updated=="1x2" else 0.0` | 0.0 |
| 21 | `market_is_asian` | `1.0 if market_updated=="asian" else 0.0` | 0.0 |
| 22 | `market_is_ou` | `1.0 if market_updated=="over_under" else 0.0` | 0.0 |
| 23 | `source_is_raw` | `1.0 if euro_source=="raw_update" else 0.0` | 0.0 |
| 24 | `has_euro` | `1.0 if has_euro else 0.0` | 0.0 |
| 25 | `has_asian` | `1.0 if has_asian else 0.0` | 0.0 |
| 26 | `has_ou` | `1.0 if has_over_under else 0.0` | 0.0 |
| 27 | `euro_h_change` | `euro_h - prev_event.euro_h`（首个填 0） | 0.0 |
| 28 | `asian_line_change` | `asian_line - prev_event.asian_line` | 0.0 |
| 29 | `upper_water_change` | `upper_water - prev_event.upper_water` | 0.0 |
| 30 | `ou_line_change` | `ou_line - prev_event.ou_line` | 0.0 |

### 2. Missing Mask

对于特征维度 0-8（原始赔率值），每个维度对应一个 mask 位：

```
mask[i] = 1.0 if 对应来源数据为 null/missing else 0.0
```

对于特征维度 9-30，mask 填 0.0（派生特征不需要 mask，源数据的缺失已在 0-8 的 mask 中体现）。

### 3. 标签（与现有一致）

```python
euro_label: int  # 0=home, 1=draw, 2=away
asian_label: int # 0-4 (5-class)
home_goals: int
away_goals: int
```

## 功能点

### F1：特征提取函数

```python
def _event_to_features_v6(event: dict, prev_event: dict | None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        event: 当前事件 dict
        prev_event: 前一个事件 dict（None 表示这是序列第一个事件）
    Returns:
        features: [31] float32 tensor
        mask: [31] float32 tensor（0.0=有效, 1.0=缺失）
    """
```

### F2：时间线处理

```python
# 在 _build_features_and_labels() 中：
# 1. 取 timeline（按 minutes_before_kickoff 降序排列——数据已排好）
# 2. 截断到 max_seq_len（保留最新的 max_seq_len 个事件）
# 3. 逐事件调用 _event_to_features_v6(event, prev_event)
# 4. 堆叠为 [T, 31] features 和 [T, 31] mask
```

### F3：Schema 注册

在 `OddsDataset.__init__()` 中，当 `feature_schema_version="v6_event"` 时使用新 schema。保持现有 v3/v4/v5 逻辑不变。

## 测试内容

### T1：特征维度正确性
```python
# 加载一行数据，提取特征
ds = OddsDataset(path, max_seq_len=64, feature_schema_version="v6_event")
item = ds[0]
assert item["features"].shape == (64, 31)  # padded
assert item["missing_mask"].shape == (64, 31)
```

### T2：缺失值处理
```python
# 找一个 OU 缺失的事件，验证：
# - has_ou 特征 = 0.0
# - ou_line/ou_over_water/ou_under_water = 0.0
# - mask 对应位 = 1.0
```

### T3：时间差计算
```python
# 前两个事件的时间差 = event[1].minutes - event[0].minutes（正数）
# 验证 time_delta_prev 特征匹配
```

### T4：向后兼容
```python
# 用 feature_schema_version="v5" 加载 v6 数据应该正常降级
# 用 feature_schema_version="v6_event" 加载旧 v5 数据应报错或降级
```

### T5：序列截断
```python
# max_seq_len=10，加载一个有 50 个事件的比赛
# 验证 features.shape[0] == 10（保留最新 10 个事件）
```

## 验收标准

- [ ] 所有 5 个测试通过
- [ ] 现有 v3/v4/v5 schema 的测试不受影响
- [ ] 31 维特征索引与规格一致（按顺序核对）

## 文件位置

修改文件：`dataset/odds_dataset.py`
- 新增 `V6_FEATURE_NAMES`（31 个名称的列表）
- 新增 `_event_to_features_v6(event, prev_event)`
- 修改 `_build_features_and_labels()` 以支持 `feature_schema_version="v6_event"`

测试文件：新建 `tests/test_v6_event_features.py`
