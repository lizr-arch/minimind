# OddsMind 数据目录结构规范

## 绝对路径根

```
D:\code\git\betmind\minimind\data\odds_real\
```

## 目录层级

```
data/odds_real/
│
├── bookmaker=Bet365/
│   ├── country=England/
│   │   ├── league=EPL/
│   │   │   ├── EPL_2019-20.jsonl
│   │   │   ├── EPL_2020-21.jsonl
│   │   │   ├── EPL_2021-22.jsonl
│   │   │   ├── EPL_2022-23.jsonl
│   │   │   ├── EPL_2023-24.jsonl
│   │   │   └── EPL_2024-25.jsonl
│   │   ├── league=Championship/
│   │   │   ├── CHA_2022-23.jsonl
│   │   │   └── CHA_2023-24.jsonl
│   │   └── cup=FA_Cup/
│   │       ├── FA_Cup_2022-23.jsonl
│   │       └── FA_Cup_2023-24.jsonl
│   │
│   ├── country=Spain/
│   │   ├── league=LaLiga/
│   │   │   ├── LaLiga_2020-21.jsonl
│   │   │   └── ...
│   │   └── cup=Copa_del_Rey/
│   │       └── ...
│   │
│   ├── country=Italy/
│   │   └── league=SerieA/
│   │       └── ...
│   │
│   ├── country=Germany/
│   │   └── league=Bundesliga/
│   │       └── ...
│   │
│   └── country=France/
│       └── league=Ligue1/
│           └── ...
│
├── bookmaker=Pinnacle/
│   └── country=England/
│       └── league=EPL/
│           └── ...
│
└── bookmaker=Bet365/          ← 已有多赛季合并文件
    └── all_leagues_1920-2425.jsonl
```

## 命名规则

```
{联赛名}_{赛季}.jsonl
```

| 部分 | 规则 | 示例 |
|------|------|------|
| 联赛名 | 完整英文名，空格用 `_` | `EPL`, `LaLiga`, `SerieA`, `Bundesliga`, `Ligue1`, `Championship`, `FA_Cup` |
| 赛季 | `YYYY-YY` | `2022-23`, `2024-25` |
| cup 用 `cup=` 前缀 | 区别于 league | `cup=FA_Cup`, `cup=Copa_del_Rey` |
| 文件名 | `{联赛}_{赛季}.jsonl` | `EPL_2024-25.jsonl` |

## JSONL 格式

每行一个完整比赛对象。格式详见 `docs/ODDSMIND_DATA_FORMAT_SPEC.md`，不因目录结构改变。

## 训练时如何读取

训练脚本支持传入**目录**自动合并，或传入**单个文件**独立训练：

```bash
# 单联赛单赛季
python trainer/train_odds_supervised.py --data data/odds_real/bookmaker=Bet365/country=England/league=EPL/EPL_2024-25.jsonl ...

# 多文件合并（按通配符）
python trainer/train_odds_supervised.py --data data/odds_real/bookmaker=Bet365/country=England/league=EPL/*.jsonl ...

# 多联赛（用逗号分隔路径或先合并）
python tools/odds_merge_jsonl.py --dirs bookmaker=Bet365/country=England/ --out all_england.jsonl
```

## 产出要求（给数据团队）

1. 在 `D:\code\git\betmind\minimind\data\odds_real\` 下按上述结构创建目录和文件
2. 每个 JSONL 文件内格式不变（`docs/ODDSMIND_DATA_FORMAT_SPEC.md`）
3. 文件名 = `{联赛}_{赛季}.jsonl`
4. 放好后告知

## 当前已有数据

```
data/odds_real/
├── master_5330.jsonl              # Bet365 × 5大联赛 × 2022-25（含 Titan007）
├── historical_seasons.jsonl       # Bet365 × 5大联赛 × 2019-22
├── all_10701.jsonl                # 上述两者合并
├── E0_2024-25_bet365.jsonl        # 早期单联赛导入
└── epl_2425_final.jsonl           # 早期单联赛导入
```
