# OddsMind 训练数据格式规范 v2

## 文件位置

```
D:\code\git\betmind\minimind\data\odds_real\<任意名称>.jsonl
```

## 格式

**JSONL**（每行一个完整 JSON 对象，行尾 `\n`，UTF-8 编码）。

---

## 单行模板

```json
{
  "match_id": "wc2026-spain-vs-saudi-arabia",
  "league_id": "WorldCup",
  "bookmaker_id": "Bet365",
  "kickoff_time": "2026-06-22T00:00:00+00:00",
  "odds_timeline": [
    {
      "minutes_before_kickoff": 10080,
      "euro_h": 1.11,
      "euro_d": 21.0,
      "euro_a": 9.5,
      "asian_line": -1.50,
      "upper_water": 0.95,
      "lower_water": 0.95,
      "over_under_line": 2.5,
      "over_water": 0.90,
      "under_water": 1.00
    },
    {
      "minutes_before_kickoff": 0,
      "euro_h": 1.10,
      "euro_d": 23.0,
      "euro_a": 10.0,
      "asian_line": -1.50,
      "upper_water": 1.03,
      "lower_water": 0.83,
      "over_under_line": 2.5,
      "over_water": 0.95,
      "under_water": 0.95
    }
  ],
  "asian_handicap_lines": [
    {"line": -1.00, "upper_water": 1.30, "lower_water": 3.50},
    {"line": -1.25, "upper_water": 1.95, "lower_water": 1.95},
    {"line": -1.50, "upper_water": 1.03, "lower_water": 0.83}
  ],
  "over_under_lines": [
    {"line": 2.0,  "over_water": 0.70, "under_water": 1.20},
    {"line": 2.5,  "over_water": 0.95, "under_water": 0.95},
    {"line": 3.0,  "over_water": 1.20, "under_water": 0.70}
  ],
  "label": {
    "euro_result": "home",
    "asian_result": "push",
    "home_goals": 3,
    "away_goals": 1,
    "upper_side": "home"
  }
}
```

---

## 字段说明

### 顶层

| 字段 | 类型 | 必填 | 说明 |
|------|------|:--:|------|
| `match_id` | string | ✅ | 唯一标识 |
| `league_id` | string | ✅ | 联赛/杯赛名，如 `EPL`, `WorldCup`, `UEFA_CL` |
| `bookmaker_id` | string | ✅ | 博彩公司 |
| `kickoff_time` | string | ✅ | ISO 8601 UTC |
| `odds_timeline` | array | ✅ | 赔率变动事件，按 `minutes_before_kickoff` 从大到小 |
| `asian_handicap_lines` | array | ✅ | 相邻亚盘盘口线（含当前盘口） |
| `over_under_lines` | array | ✅ | 相邻大小球盘口线（含当前盘口） |
| `label` | object | ✅ | 比赛结果（赛后补） |

### odds_timeline[i] — 新增 3 个字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|:--:|------|
| `minutes_before_kickoff` | float | ✅ | 距开球分钟数，≥ 0 |
| `euro_h` | float | ✅ | 欧赔主胜 |
| `euro_d` | float | ✅ | 欧赔平局 |
| `euro_a` | float | ✅ | 欧赔客胜 |
| `asian_line` | float | ✅ | 亚盘让球线（主队视角），0.25 整数倍 |
| `upper_water` | float | ✅ | 上盘水位 |
| `lower_water` | float | ✅ | 下盘水位 |
| `over_under_line` | float | ✅ | **新增** 大小球盘口线，如 2.5，0.25 整数倍 |
| `over_water` | float | ✅ | **新增** 大球水位 |
| `under_water` | float | ✅ | **新增** 小球水位 |

### asian_handicap_lines[i]

当前盘口 + 相邻盘口的水位快照（一般是收盘时刻）。

| 字段 | 类型 | 说明 |
|------|------|------|
| `line` | float | 盘口线 |
| `upper_water` | float | 上盘水位 |
| `lower_water` | float | 下盘水位 |

**水位最接近的那条就是当前盘口**。模型自动识别。

### over_under_lines[i]

| 字段 | 类型 | 说明 |
|------|------|------|
| `line` | float | 大小球线 |
| `over_water` | float | 大球水位 |
| `under_water` | float | 小球水位 |

### label

| 字段 | 类型 | 必填 | 说明 |
|------|------|:--:|------|
| `euro_result` | string | ✅ | `"home"` / `"draw"` / `"away"` |
| `asian_result` | string | ✅ | `"full_win"` / `"half_win"` / `"push"` / `"half_loss"` / `"full_loss"`（从上盘视角） |
| `home_goals` | int | ✅ | 主队进球 |
| `away_goals` | int | ✅ | 客队进球 |
| `upper_side` | string | ✅ | `"home"` 或 `"away"` |

---

## 禁止事项

- ❌ `minutes_before_kickoff` < 0（滚球）
- ❌ 赔率 ≤ 0
- ❌ `asian_line` / `over_under_line` 不是 0.25 整数倍
- ❌ `odds_timeline` 未按时间降序
- ❌ 一场多行

## 当前盘口自动识别

```
水位差 = |upper_water - lower_water|
当前盘口 = argmin(水位差)
```

## 绝对路径

```
D:\code\git\betmind\minimind\docs\ODDSMIND_DATA_FORMAT_SPEC.md
```
