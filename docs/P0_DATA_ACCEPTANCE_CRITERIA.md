# P0 数据验收标准

## 一、最小规模门禁（硬性）

| 指标 | 最低要求 | 目标 | 验收方式 |
|---|---|---|---|
| 总比赛数 | ≥ 15,000 | ≥ 20,000 | `wc -l data.jsonl` |
| 联赛数 | ≥ 5 | ≥ 8 | 按 `league_id` 去重统计 |
| 每联赛最小比赛数 | ≥ 500 | ≥ 1,000 | 按 `league_id` 分组统计 |
| 赛季覆盖 | ≥ 3 个赛季 | ≥ 5 个赛季 | 按 `kickoff_time` 年份分组 |
| 每赛季每联赛最小比赛数 | ≥ 100 | ≥ 200 | 按 `(league_id, season)` 分组 |

**不满足任一硬性指标 → 拒绝，不进入训练。**

---

## 二、数据格式门禁（硬性）

每行一个 JSON 对象，必须包含以下字段：

```json
{
  "match_id": "string, 非空, 唯一",
  "kickoff_time": "string, ISO 8601, 可解析为 datetime",
  "league_id": "string, 非空",
  "odds_timeline": "list, 长度 >= 1",
  "label": {
    "euro_result": "string, 'home' | 'draw' | 'away'",
    "asian_result": "string, 合法 5-class 标签",
    "home_goals": "int, >= 0",
    "away_goals": "int, >= 0"
  }
}
```

每个 `odds_timeline` 事件必须包含：

```json
{
  "minutes_before_kickoff": "float, > 0, 非 NaN 非 Inf"
}
```

**验收脚本：**

```bash
python -c "
import json, sys
errors = []
with open('DATA_PATH') as f:
    for i, line in enumerate(f, 1):
        if not line.strip(): continue
        try:
            m = json.loads(line)
            assert isinstance(m['match_id'], str) and m['match_id']
            assert isinstance(m['kickoff_time'], str)
            assert isinstance(m['league_id'], str) and m['league_id']
            assert isinstance(m['odds_timeline'], list) and len(m['odds_timeline']) >= 1
            for j, e in enumerate(m['odds_timeline']):
                mbk = e.get('minutes_before_kickoff', 0)
                assert isinstance(mbk, (int, float)) and mbk > 0
            lbl = m['label']
            assert lbl['euro_result'] in ('home', 'draw', 'away')
            assert isinstance(lbl['home_goals'], int) and lbl['home_goals'] >= 0
            assert isinstance(lbl['away_goals'], int) and lbl['away_goals'] >= 0
        except Exception as e:
            errors.append(f'line {i}: {e}')
print(f'{i} lines checked, {len(errors)} errors')
if errors:
    for e in errors[:10]: print(f'  {e}')
    sys.exit(1)
print('PASS: all format checks passed')
"
```

**不满足 → 拒绝。**

---

## 三、数据质量门禁（软性，超阈值拒绝）

| 指标 | 阈值 | 验收方式 |
|---|---|---|
| `minutes_before_kickoff` 为整数的比例 | > 90% | 统计分布 |
| `euro_h/d/a > 1.01` 的比例（有欧赔的事件） | > 95% | 统计 |
| `asian_line ∈ [-3.0, 3.0]` 的比例（有亚盘的事件） | > 95% | 统计 |
| `match_id` 重复率 | 0% | 去重检查 |
| `kickoff_time` 跨度 | ≥ 2 年 | min/max 检查 |
| 空 timeline 的比赛数 | 0 | 检查 |
| 每场比赛 timeline 事件数中位数 | ≥ 3 | 统计 |

**任何指标超过阈值 → 标记 WARNING，但不拒绝。需人工判断。**

---

## 四、泄漏门禁（硬性）

| 检查项 | 要求 | 验收方式 |
|---|---|---|
| 赔率事件时间 | 所有 `minutes_before_kickoff > 0` | 逐事件检查 |
| label 时间一致性 | `kickoff_time` < 数据采集时间 | 外部交叉验证或信任 |
| 无赛后数据混入 | `odds_timeline` 中无比分/赛果字段 | 检查事件字段白名单 |
| bookmaker 来源 | 如果有 `bookmaker_id`，必须非空 | 统计缺失率 |

**不满足 → 拒绝。**

---

## 五、联赛分布门禁（软性）

| 联赛 | 最少比赛数 | 验收方式 |
|---|---|---|
| EPL | ≥ 1,000 | 按 `league_id` 统计 |
| LaLiga | ≥ 1,000 | 同上 |
| SerieA | ≥ 1,000 | 同上 |
| Bundesliga | ≥ 800 | 同上 |
| Ligue1 | ≥ 800 | 同上 |
| 其他联赛（每个） | ≥ 500 | 同上 |

**不满足 → 对应的联赛标记为 `untrained`，不进入训练集。**
**全部不满足 → 拒绝。**

---

## 六、验收流程

```
数据到达
  ↓
1. 格式门禁（脚本自动化）
  ↓ PASS
2. 规模门禁（统计量检查）
  ↓ PASS
3. 质量门禁（分布检查）
  ↓ PASS / PASS_WITH_WARNINGS
4. 泄漏门禁（逐事件抽查）
  ↓ PASS
5. 联赛分布确认
  ↓
6. 生成 split（time-based grouped, 70/15/15）
  ↓
7. 跑 smoke training（2 epochs, 验证无 NaN/Inf/崩溃）
  ↓ PASS
8. P0 验收通过，数据可进入训练
```

---

## 七、验收后交付物

- `data/odds_real/v6_all.jsonl` — 合并后全量数据
- `data/odds_real/splits_v6/train_match_ids.txt` — 训练集 match_id 列表
- `data/odds_real/splits_v6/val_match_ids.txt`
- `data/odds_real/splits_v6/test_match_ids.txt`
- `data/odds_real/splits_v6/split_summary.json` — split 时间范围 + 各联赛分布
- `data/reports/p0_acceptance_report.json` — 所有门禁检查结果
