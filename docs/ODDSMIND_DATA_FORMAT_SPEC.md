# OddsMind 训练数据格式规范

## 文件位置

```
D:\code\git\betmind\minimind\data\odds_real\<任意名称>.jsonl
```

训练脚本读取绝对路径，不限制文件名。

## 格式

**JSONL**（每行一个完整 JSON 对象，行尾 `\n`，UTF-8 编码）。

## 单行模板

```json
{
  "match_id": "fd-epl-16-08-2024-man-united-fulham",
  "league_id": "EPL",
  "bookmaker_id": "Bet365",
  "kickoff_time": "2024-08-16T20:00:00+00:00",
  "odds_timeline": [
    {
      "minutes_before_kickoff": 85572,
      "euro_h": 1.62,
      "euro_d": 4.0,
      "euro_a": 5.0,
      "asian_line": -1.0,
      "upper_water": 2.05,
      "lower_water": 1.88
    },
    {
      "minutes_before_kickoff": 4320,
      "euro_h": 1.60,
      "euro_d": 4.20,
      "euro_a": 5.25,
      "asian_line": -1.0,
      "upper_water": 2.05,
      "lower_water": 1.88
    },
    {
      "minutes_before_kickoff": 0,
      "euro_h": 1.66,
      "euro_d": 4.10,
      "euro_a": 5.00,
      "asian_line": -0.75,
      "upper_water": 1.86,
      "lower_water": 2.07
    }
  ],
  "label": {
    "euro_result": "home",
    "asian_result": "upper_full_win",
    "home_goals": 1,
    "away_goals": 0,
    "upper_side": "home"
  }
}
```

## 字段说明

### 顶层

| 字段 | 类型 | 必填 | 说明 |
|------|------|:--:|------|
| `match_id` | string | ✅ | 唯一标识，建议格式 `{联赛}_{日期}_{主队}_{客队}` |
| `league_id` | string | ✅ | 联赛名，如 `EPL`, `LaLiga`, `SerieA`, `Bundesliga`, `Ligue1` |
| `bookmaker_id` | string | ✅ | 博彩公司，如 `Bet365`, `Pinnacle` |
| `kickoff_time` | string | ✅ | ISO 8601 UTC，如 `2024-08-16T20:00:00+00:00` |
| `odds_timeline` | array | ✅ | 赔率事件列表，按 `minutes_before_kickoff` 从大到小排序 |
| `label` | object | ✅ | 比赛结果标签 |

### odds_timeline[i]

| 字段 | 类型 | 必填 | 说明 |
|------|------|:--:|------|
| `minutes_before_kickoff` | float | ✅ | 距开球分钟数。开盘≈4320~10080（3-7天），收盘=0。**必须 ≥ 0** |
| `euro_h` | float | ✅ | 欧赔主胜（> 0） |
| `euro_d` | float | ✅ | 欧赔平局（> 0） |
| `euro_a` | float | ✅ | 欧赔客胜（> 0） |
| `asian_line` | float | ✅ | 亚盘让球线，主队视角。如 -1.0 表示主让1球，+0.5 表示主受半球。**必须是 0.25 的整数倍** |
| `upper_water` | float | ✅ | 上盘水位（> 0） |
| `lower_water` | float | ✅ | 下盘水位（> 0） |

### label

| 字段 | 类型 | 必填 | 说明 |
|------|------|:--:|------|
| `euro_result` | string | ✅ | 赛果：`"home"` / `"draw"` / `"away"` |
| `asian_result` | string | ✅ | 亚盘结果（从上盘视角），5 选 1：`"upper_full_win"` / `"upper_half_win"` / `"push"` / `"upper_half_loss"` / `"upper_full_loss"` |
| `home_goals` | int | ✅ | 主队进球（≥ 0） |
| `away_goals` | int | ✅ | 客队进球（≥ 0） |
| `upper_side` | string | ✅ | 上盘方：`"home"` 或 `"away"` |

## Timeline 事件数量

- 最少 **1 个**事件（closing-only snapshot）
- 推荐 **2+ 个**事件（开盘 + 收盘）
- 最佳 **多个 movement 事件**（≥10 个更好，支持时间序列学习）

## 亚盘标签计算规则

从上盘视角：

```
raw_margin = upper_goals - lower_goals
adjusted = raw_margin + asian_line
```

对四分之一盘口（如 -0.25, -0.75），拆成两个半盘分别结算：

| 两个半盘结果 | 五分类标签 |
|---|---|
| win + win | `upper_full_win` |
| win + push 或 push + win | `upper_half_win` |
| push + push | `push` |
| loss + push 或 push + loss | `upper_half_loss` |
| loss + loss | `upper_full_loss` |

## 禁止事项

- ❌ `minutes_before_kickoff` < 0（滚球数据，不要混入赛前训练）
- ❌ `euro_h/d/a` ≤ 0 或缺失
- ❌ `asian_line` 不是 0.25 的整数倍
- ❌ `home_goals` / `away_goals` 负值或缺失
- ❌ `odds_timeline` 未按时间降序排列
- ❌ 同一场比赛多行（必须一行一赛）

## 验证

产出 JSONL 后可用以下命令快速验证：

```bash
python -c "
import json
with open('你的文件.jsonl') as f:
    for i, line in enumerate(f, 1):
        s = json.loads(line)
        assert s['match_id'], f'line {i}: missing match_id'
        assert len(s['odds_timeline']) > 0, f'line {i}: empty timeline'
        for e in s['odds_timeline']:
            assert e['minutes_before_kickoff'] >= 0
            assert e['euro_h'] > 0
        assert s['label']['euro_result'] in ('home','draw','away')
        assert s['label']['asian_result'] in ('upper_full_win','upper_half_win','push','upper_half_loss','upper_full_loss')
print(f'{i} lines valid')
"
```
