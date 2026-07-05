# OddsMind 数据坑速查表

> 基于 P1.0 ~ P1.1 阶段实际踩坑整理。遇到数据问题先查这张表。

---

## 速查表

| # | 坑 | 症状 | 原因 | 解决方案 |
|---|-----|------|------|----------|
| 1 | **v3 亚盘/大小球全是默认值** | `asian_line` 全为 0.0，`upper_water/lower_water` 全为 1.0，`over_under_line` 全为 2.5 | v3 exporter 未正确抓取 AH/OU 市场，用默认值填充 | 升级到 v4 exporter (P1.1A)，使用 `asian_source`/`over_under_source` 字段区分 |
| 2 | **exporter 标记 has_asian=true 但数据是假的** | v3 数据所有 event 的 `has_asian=True`，但实际全是默认值 | exporter 标记逻辑有 bug，只检查 key 是否存在而非值是否真实 | Dataset 不信任 exporter 的 `has_*` 标记，用内容检测 (P1.0F) 或 source 字段 (P1.1B) |
| 3 | **asian_line=0 歧义：平手盘 vs 缺失** | `asian_line=0` 可能是真实平手盘，也可能是缺失后填 0 | 导出器用 0 填充缺失值，与真实平手盘冲突 | v3: 检查 water 是否为 1.0/1.0 (默认=缺失)；v4: 用 `asian_source` 字段判断 |
| 4 | **over_under_line=2.5 歧义** | `ou_line=2.5` 是最常见盘口，也可能是默认填充 | 同 #3 | v3: 检查 water 是否为 1.0/1.0；v4: 用 `over_under_source` 字段 |
| 5 | **consensus_feats 泄漏未来信息** | 如果 consensus 用了完整 timeline 或 closing odds | cutoff 样本不应该看到 cutoff 之后的数据 | 默认 `consensus_mode="none"` (全零)；需要时用 `visible_only` |
| 6 | **bookmaker_features 不存在** | consensus_feats 永远是 zeros(6) | v3/v4 JSONL 中没有 `bookmaker_features` 字段 | 这是安全的 (无信息泄漏)，但 consensus 通道无用 |
| 7 | **asian_label_status=missing_handicap** | 27 个样本的 `asian_result=null`，Dataset 会崩溃 | 某些比赛确实没有亚盘数据 | Dataset 用 `asian_label_mask` (0.0=不参与 loss)，label 设为 -100 |
| 8 | **encode_asian_label(null) 崩溃** | 传入 None 时报 ValueError | 没有检查 label 是否存在 | 在 `_build_features_and_labels` 中先检查 `asian_label_status` 和 `asian_result` |
| 9 | **数据量远低于 P0 要求** | P0 要求 ≥15,000 match，v4 只有 2,099 唯一 match | 大量原始数据在 exporter 中被标记为 invalid (原因待查) | 需查 odds-data-probe 的 invalid 文件，扩大数据采集 |
| 10 | **kickoff_time 缺失** | 大量 match 没有 kickoff_time，无法做时间分割 | 原始数据源不完整 | 只用有 kickoff_time 的 match；缺失的排除出 split |
| 11 | **change_time 格式不统一** | "04-05 22:08" vs "2022-04-05T22:08:00" | Titan007 原始数据用短格式 | exporter 做格式恢复，恢复失败的跳过 |
| 12 | **cutoff 样本看到未来数据** | cutoff=60 的样本里出现了 T-30 的 event | cutoff 过滤逻辑错误 | `filter_timeline_by_cutoff` 用 `>=` 判断，`assert_cutoff_integrity` 校验 |
| 13 | **label 5class → 3class 映射不全** | `half_win` 找不到对应的 3class | 映射表缺少条目 | `ASIAN_5_TO_3` 映射表覆盖所有 5class 标签 |
| 14 | **missing_mask 语义反转** | 1.0 表示 missing 还是 present？ | 不同代码用不同约定 | 统一：`1.0 = present (有数据)`，`0.0 = missing (缺失)` |
| 15 | **safe_inverse_odds(0) = inf** | odds <= 1 时 1/odds 产生 inf/NaN | 直接除法不安全 | 用 `safe_inverse_odds()`：odds <= 1 返回 0.0 |
| 16 | **no-vig 概率 overround=0 时除零** | 所有 odds 缺失时 implied prob 除零 | `sum(inverse_odds)` 为 0 | `safe_implied_probs()` 检查 total <= 0 时返回均匀分布 |
| 17 | **forward_fill 的亚盘数据可信度** | v4 中 47659 个 event 的 `asian_source=forward_fill` | 前一次更新的盘口被延续使用 | 技术上是 present 数据，但可能不是最新报价。训练时可考虑加权 |
| 18 | **split 不可跨版本复用** | v3 split 不能直接用于 v4 数据 | match_id 可能不同，数据量不同 | 每个版本重新 `generate_odds_splits.py` |
| 19 | **trainer 不支持 asian_label_mask** | Dataset 输出了 mask 但 trainer 没用 | trainer 需要更新 | 训练前需要在 loss 计算中集成 `asian_label_mask` |

---

## 关键检测函数

| 函数 | 位置 | 作用 |
|------|------|------|
| `_event_has_euro()` | `dataset/odds_dataset.py:120` | 检测 euro odds 是否真实 (h>1, d>1, a>1) |
| `_event_has_asian()` | `dataset/odds_dataset.py:127` | v4: 用 `asian_source`；v3: 检测默认值 |
| `_event_has_over_under()` | `dataset/odds_dataset.py:152` | v4: 用 `over_under_source`；v3: 检测默认值 |
| `safe_inverse_odds()` | `dataset/odds_dataset.py:212` | 安全的 1/odds，odds<=1 返回 0 |
| `safe_implied_probs()` | `dataset/odds_dataset.py:218` | 安全的 3-way 隐含概率 |
| `safe_novig_probs()` | `dataset/odds_dataset.py:228` | 安全的 no-vig 概率 |
| `assert_cutoff_integrity()` | `dataset/odds_cutoff.py:26` | 校验 cutoff 样本无未来数据 |

---

## 数据版本对比

| 字段 | v3 | v4 |
|------|-----|-----|
| `has_asian` | 不可信 (全 true) | 可信 (exporter 标记) |
| `has_over_under` | 不可信 (全 true) | 可信 (exporter 标记) |
| `asian_source` | 不存在 | `raw_update` / `forward_fill` / `missing` |
| `over_under_source` | 不存在 | `raw_update` / `forward_fill` / `missing` |
| `asian_line_missing` | 不存在 | boolean |
| `asian_label_status` | 不存在 | `ok` / `missing_handicap` |
| `bookmaker_features` | 不存在 | 不存在 |
| 真实亚盘数据 | 0% | 97.3% |
| 真实大小球数据 | 0% | 97.2% |

---

## 排查流程

```
数据异常？
├── NaN / Inf 特征？
│   ├── 检查 safe_inverse_odds / safe_implied_probs
│   └── 检查 overround=0 时的除零
├── has_asian=false 但期望 true？
│   ├── v3: 检查是否 water=1.0/1.0 (默认值检测)
│   └── v4: 检查 asian_source 是否为 "missing"
├── asian_line=0 但不确定是否真实？
│   ├── v3: water=1.0/1.0 → 缺失；water≠1.0 → 真实平手盘
│   └── v4: asian_source="missing" → 缺失；其他 → 真实
├── Dataset 加载崩溃？
│   ├── 检查 asian_result 是否为 None
│   ├── 检查 asian_label_status 是否为 missing_handicap
│   └── 检查 encode_asian_label 是否收到无效标签
├── cutoff 样本有未来数据？
│   ├── 运行 assert_cutoff_integrity
│   └── 检查 filter_timeline_by_cutoff 的 >= 判断
├── split 有 overlap？
│   ├── 检查 match_id 是否跨 split
│   └── 检查多庄家 match 是否在同一 split
└── consensus_feats 泄漏？
    ├── 确认 consensus_mode="none"
    └── 检查 bookmaker_features 是否存在
```

---

## 常用诊断命令

```powershell
# 检查数据统计
python -c "import json; f=open('data/odds_real/xxx.jsonl'); lines=f.readlines(); print('matches:', len(lines))"

# 运行全 gate 检查
python tools/inspect_odds_data_gate.py --input data/odds_real/xxx.jsonl --splits-dir data/odds_real/splits/xxx --output docs/review/gate.md --feature-schema v4 --consensus-mode none --cutoff-buckets "1440,720,360,180,120,60,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1"

# 生成 split
python tools/generate_odds_splits.py --input data/odds_real/xxx.jsonl --output-dir data/odds_real/splits/xxx --report docs/review/split.md

# 跑测试
python -m pytest tests -q

# 检查 v4 source 分布
python -c "import json; from collections import Counter; f=open('data/odds_real/xxx.jsonl'); [print(Counter(e.get('asian_source','?') for d in [json.loads(l)] for e in d.get('odds_timeline',[]))) for l in f]"
```

---

*最后更新: 2026-06-24 | 基于 P1.0A ~ P1.1B 阶段*
