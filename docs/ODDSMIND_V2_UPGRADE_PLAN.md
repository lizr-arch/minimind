# OddsMind v1.0 → v2.0 现状审计、数据规范与升级方案

---

## 一、当前架构审计：已有 vs 缺失

### 1.1 数据层

| 模块 | 状态 | 详情 |
|---|---|---|
| JSONL match-level 数据格式 | ✅ 已有 | `match_id, league_id, bookmaker_id, kickoff_time, odds_timeline, label` |
| 赔率时间序列事件 | ✅ 已有 | `euro_h/d/a, asian_line, upper/lower_water, over_under_line, over/under_water` |
| 时间特征 | ✅ 已有 | `minutes_before_kickoff` |
| 赛果标签 | ✅ 已有 | `euro_result (home/draw/away)`, `asian_result`, `home_goals/away_goals` |
| 单庄家数据 | ✅ 已有 | 每场只记录一个 bookmaker |
| team_id (主队/客队) | ❌ 缺失 | 无法做 team-level analysis 或 GNN |
| competition_type (联赛/杯赛/欧冠) | ⚠️ 部分 | 有 league_id 但无 taxonomy 字段 |
| is_knockout / is_group_stage | ❌ 缺失 | 无法区分联赛(双循环)和杯赛(淘汰制) |
| 多庄家同时刻赔率 | ❌ 缺失 | 无法做 inter-bookmaker agreement |
| 庄家类型标记 (sharp/soft) | ❌ 缺失 | Pinnacle=sharp, Bet365=soft — 无法建模 |
| 赛季标记 | ❌ 缺失 | 无法按赛季分组分析 |
| 球队排名/实力 | ❌ 缺失 | 当前是纯 odds-only |
| 比赛重要性标记 | ❌ 缺失 | 保级战/争冠战 vs 中游无关比赛 |
| 天气/场地条件 | ❌ 缺失 | 外部因素 |

### 1.2 特征工程层

| 模块 | 状态 | 详情 |
|---|---|---|
| Feature schema v3 (13-dim) | ✅ 已有 | `minutes, euro_h/d/a, asian_line, upper/lower, ou_line, ou_over/under, has_euro/asian/ou` |
| Missing mask (per-event, per-feature) | ✅ 已有 | `missing_mask [B,T,13]` |
| Learned missing embedding | ✅ 已有 | `OddsEventEncoder.missing_embed` |
| OddsEventEncoder (MLP投影) | ✅ 已有 | Linear→SiLU→Dropout→Linear |
| Cutoff filtering | ✅ 已有 | `filter_timeline_by_cutoff()` |
| Cutoff integrity assertion | ✅ 已有 | `assert_cutoff_integrity()` |
| Bookmaker embedding | ⚠️ 部分 | 6 个庄家，pooled 后加在表示上 |
| League embedding | ❌ 缺失 | `num_leagues=0` 在 config 中占位但未启用 |
| Team embedding | ❌ 缺失 | 需要 team_id 数据 |
| 时间编码 (sinusoidal) | ❌ 缺失 | `minutes_before_kickoff` 只是标量 |
| Market quality proxy features | ✅ 已有 | P2.2 `market_quality.py` 的 18 维特征 |
| Multi-bookmaker consensus features | ⚠️ 部分 | `consensus_proj` 在模型中预留，但数据不支持 |
| 赔率变化率 (一阶/二阶差分) | ❌ 缺失 | 当前只用原始赔率值 |
| Volatility / trend 特征 | ❌ 缺失 | 赔率在时间维度上的统计量 |

### 1.3 模型架构层

| 模块 | 状态 | 详情 |
|---|---|---|
| Transformer Encoder (bidirectional) | ✅ 已有 | 4 layers, 8 heads, 256-dim, 1024 FFN |
| Mean pooling | ✅ 已有 | 默认，已验证最优 |
| Attention/CLS pooling | ⚠️ 已测试 | P1.19: 在当前规模下不可行 |
| EuroResultHead (3-class) | ✅ 已有 | 主任务 |
| AsianResultHead (5-class) | ✅ 已有 | 辅助任务 |
| ScoreHeadV2 (Poisson regression) | ✅ 已有 | 辅助任务 |
| Legal class mask for Asian | ✅ 已有 | P1.18 已修复 |
| Causal masking | ❌ 已移除 | 正确使用 bidirectional |
| LM Head / next-token loss | ❌ 已移除 | 正确替换为 CE loss |
| Tokenizer | ❌ 已移除 | 正确替换为 OddsEventEncoder |
| Variable Selection Networks | ❌ 缺失 | TFT 的设计 |
| Gated Residual Networks | ❌ 缺失 | TFT 的设计 |
| Interpretable attention | ❌ 缺失 | 当前 attention 不可解释 |
| Quantile regression head | ❌ 缺失 | 只能输出点概率，不能输出分布 |
| Static vs temporal feature separation | ❌ 缺失 | league/bookmaker 应该单独处理 |
| LSTM/GRU temporal backbone | ❌ 缺失 | 纯 attention 处理时间序列 |
| Residual connection to baseline | ⚠️ 已测试 | P2.4: market_residual 不可行 |

### 1.4 训练与评估层

| 模块 | 状态 | 详情 |
|---|---|---|
| CrossEntropy loss (Euro 1X2) | ✅ 已有 | 主 loss |
| Label smoothing | ✅ 已有 | 0.1 |
| Mixup data augmentation | ✅ 已有 | alpha=0.0 (disabled by default) |
| EMA weight averaging | ✅ 已有 | decay=0.999 |
| Cosine warmup-decay LR | ✅ 已有 | |
| Early stopping | ✅ 已有 | patience=10 |
| Gradient clipping | ✅ 已有 | max_norm=1.0 |
| Time-based grouped split | ✅ 已有 | match_id 不跨 split |
| Seed reproducibility | ✅ 已有 | setup_seed() |
| latest_available_no_vig baseline | ✅ 已有 | cutoff-safe |
| Accuracy/LogLoss/Brier/ECE metrics | ✅ 已有 | full suite |
| Reliability bins | ✅ 已有 | 10-bin ECE |
| Per-league/per-cutoff metrics | ✅ 已有 | |
| Bootstrap CI | ✅ 已有 | P1.21 |
| Temperature scaling calibration | ✅ 已有 | T=1.09 on val only |
| KNN retrieval baseline | ✅ 已有 | P2.2 |
| Multi-seed validation | ✅ 已有 | 3 seeds |
| Leave-one-league-out CV | ✅ 已有 | P2.1B |
| Domain transfer matrix | ✅ 已有 | P2.4 |
| Dual-path inference policy | ✅ 已有 | P2.3 |

### 1.5 推理与部署层

| 模块 | 状态 | 详情 |
|---|---|---|
| Single-match prediction CLI | ✅ 已有 | `predict_match.py` |
| JSONL batch prediction | ✅ 已有 | |
| Calibration loading | ✅ 已有 | from `calibration.json` |
| Market baseline side-by-side output | ✅ 已有 | |
| Model-market delta output | ✅ 已有 | |
| Warning generation | ✅ 已有 | missing markets, insufficient timeline |
| Known/unknown league routing | ✅ 已有 | P2.3 dual-path |
| KNN explanation output | ✅ 已有 | similarity + historical distribution |
| API server | ❌ 缺失 | 目前只有 CLI |
| Model registry | ⚠️ 部分 | P2.5C gate registry |
| A/B test framework | ❌ 缺失 | |
| Production monitoring | ❌ 缺失 | |
| Checkpoint versioning | ❌ 缺失 | 文件手动管理 |

---

## 二、数据收集规范

### 2.1 核心字段（每条比赛一条 JSON 记录）

```json
{
  "match_id": "string — 唯一标识，建议格式: {source}_{league}_{season}_{date}_{home}_{away}",
  "kickoff_time": "string — ISO 8601 格式: '2025-05-10T20:00:00Z'",

  "league_id": "string — 联赛简称: EPL, LaLiga, SerieA, Bundesliga, Ligue1, UCL, ...",
  "competition_type": "string — 'domestic_league' | 'domestic_cup' | 'continental_club' | 'international'",
  "is_knockout": "bool — true=淘汰赛, false=联赛/小组赛",
  "season": "string — '2024-25'",

  "home_team_id": "string — 主队标识: 'Arsenal', 'Real_Madrid', ...",
  "away_team_id": "string — 客队标识",

  "bookmaker_id": "string — 庄家名: Bet365, Pinnacle, William_Hill, 1XBet, Bwin, Betfair_Exchange",
  "bookmaker_tier": "string — 'sharp' (Pinnacle/Betfair) | 'soft' (Bet365/1XBet/Bwin/William_Hill)",

  "odds_timeline": [
    {
      "minutes_before_kickoff": 10080,
      "euro_h": 2.30,
      "euro_d": 3.20,
      "euro_a": 3.10,
      "asian_line": 0.0,
      "upper_water": 0.90,
      "lower_water": 1.00,
      "over_under_line": 2.5,
      "over_water": 0.95,
      "under_water": 0.85
    }
  ],

  "label": {
    "euro_result": "home",
    "asian_result": "upper_full_win",
    "home_goals": 2,
    "away_goals": 0,
    "total_goals": 2,
    "goal_difference": 2
  }
}
```

### 2.2 可选扩展字段

```json
{
  "venue": "string — 'home' | 'away' | 'neutral'",
  "attendance": "int — 现场观众数(如有)",
  "referee": "string — 主裁判(如有)",
  "match_importance": "string — 'title_race' | 'relegation_battle' | 'derby' | 'normal' | 'dead_rubber'",
  "home_ranking": "int — 主队赛前联赛排名",
  "away_ranking": "int — 客队赛前联赛排名"
}
```

### 2.3 多庄家格式（如果同一场比赛采集了多个庄家）

```json
{
  "match_id": "...",
  "bookmakers": [
    {
      "bookmaker_id": "Pinnacle",
      "bookmaker_tier": "sharp",
      "odds_timeline": [...]
    },
    {
      "bookmaker_id": "Bet365",
      "bookmaker_tier": "soft",
      "odds_timeline": [...]
    }
  ]
}
```

### 2.4 数据质量要求

| 要求 | 标准 | 失败处理 |
|---|---|---|
| kickoff_time 精确 | ISO 8601, 精确到分钟 | 拒绝整条记录 |
| minutes_before_kickoff 精确 | 不能是估计值; 至少保留到分钟 | 标记 warning |
| 每场比赛 ≥ 3 个赔率事件 | 太少无法建模时间动态 | 降级为 single-snapshot |
| 赔率数值合理 | euro_h/d/a > 1.01; water prices 0.5-1.5 | 拒绝异常值 |
| label 与真实赛果一致 | 交叉验证 (如用另一数据源校验) | 拒绝不匹配 |
| no future leakage | odds 时间戳 < kickoff_time | 拒绝/截断 |
| asian_line 合理 | -3.0 ≤ asian_line ≤ 3.0 | 标记 warning |
| over_under_line 合理 | 1.5 ≤ over_under_line ≤ 5.5 | 标记 warning |
| 联赛/赛季分布均匀 | 每个联赛至少 500 场; 每个赛季至少 200 场 | 标记 insufficient |
| NULL 与 0 区分 | 缺失值用 null/None, 不要用 0 | 数据清洗时处理 |

### 2.5 数据规模目标

| 阶段 | 联赛数 | 赛季数 | 估算比赛数 | 用途 |
|---|---|---|---|---|
| 当前 | 5 | 3 | 5,330 | Stage 1: odds-only base training |
| 扩展 v1 | 5 | 8 (回溯到 2018) | ~20,000 | Stage 2: per-league fine-tune |
| 扩展 v2 | 10-15 | 5 | ~50,000 | Stage 3: cross-league transfer |
| 扩展 v3 | 20+ | 10 | 100,000+ | Stage 4: foundation pretraining |

---

## 三、大模型升级方案（v1.0 → v2.0）

### 3.0 升级路线图

```
v1.0 (当前)         v2.0α               v2.0β              v2.0
3.33M params       10-20M params       30-50M params      50M+ params
5,330 matches      20,000 matches      50,000 matches     100,000 matches
5 leagues          5+ leagues          10+ leagues        20+ leagues
odds-only          odds + team         + domain            full foundation
Transformer Enc    TFT Encoder         TFT-L/XL           pretrain + fine-tune
```

### 3.1 v2.0α：TFT 迁移 + 轻量升级（目标数据: 20,000 场）

**架构变更**：从 generic Transformer Encoder 迁移到 Temporal Fusion Transformer

```
输入层:
  static_features    → Variable Selection → static_context [B, H_static]
    - league_id embedding (新)
    - bookmaker_id embedding (已有)
    - competition_type embedding (新)
    - is_knockout flag (新)

  temporal_features  → Variable Selection → temporal_input [B, T, H_temp]
    - euro_h/d/a raw odds
    - asian_line/upper_water/lower_water
    - over_under_line/over_water/under_water
    - implied_probs (1/odds normalized) ← 新
    - odds_change_rate (一阶差分) ← 新
    - overround (margin) ← 从 market_quality.py 提取
    - missing_mask [B, T, F]

Encoder:
  LSTM Encoder (local processing)
      ↓
  Gated Residual Network (skip non-useful info)
      ↓
  Interpretable Multi-head Attention (long-range dependencies)
      ↓
  Gated Residual Network
      ↓
  [static enrichment: 将 static_context 注入每个时间步]

Decoder/Pooling:
  Attention-weighted sum over time (learnable, 不同于当前的 mean)
      ↓
  Dense → EuroHead [B, 3] / AsianHead [B, 5] / optional ScoreHead
```

**关键新增**：
- `league_id` embedding: 8-16 dim, projected → hidden
- `team_id` embedding (如有数据): 64-128 dim per team
- Implied probs 作为额外输入（不用模型自己从 raw odds 推导）
- Odds change rate：`(odds_t - odds_{t-1}) / odds_{t-1}` 捕获市场动向
- Gated Residual Network：让模型学会"跳过"无信息量的赔率事件
- Interpretable attention：输出"模型最关注哪个时间点的赔率"

**配置**：
- d_model: 256 → 320
- num_layers: 4 → 6
- num_heads: 8 (不变)
- ffn_dim: 1024 (不变)
- LSTM hidden: 160
- static_dim: 64
- 参数量: ~10-15M

**训练策略**：
- Phase 1: 冻结 static embeddings，只训练 temporal encoder (warmup)
- Phase 2: 全参数训练，early stopping patience=10
- Phase 3: 在 val set 上拟合 temperature

### 3.2 v2.0β：规模升级（目标数据: 50,000 场）

在 v2.0α 基础上：
- d_model: 320 → 512
- num_layers: 6 → 8
- 参数量: ~30-40M
- 添加 per-league fine-tune（从 global base 出发）

**新增实验**：
- Cross-league transfer re-evaluation（数据量增大后可能不再 catastrophic fail）
- Multi-task learning with Asian/Score auxiliary losses 的权重搜索
- Ensemble of per-league models vs single global model

### 3.3 实施优先级

| 优先级 | 任务 | 前置条件 | 预计效果 |
|---|---|---|---|
| P0 | 扩充数据到 20,000+ 场 | 数据收集 | 基础提升 |
| P1 | 添加 league/bookmaker embedding | 数据有 league_id/bookmaker_id | 小幅提升 |
| P1 | 添加 implied_probs 作为输入特征 | 无 | 可能提升 logloss |
| P2 | 迁移到 TFT 架构 | v2.0α | 架构适配 |
| P2 | 添加 team_id embedding | 数据有 home_team_id/away_team_id | 跨队泛化 |
| P2 | 添加 odds change rate 特征 | 无 | 捕获市场动态 |
| P3 | 模型规模升级 (d_model/layers) | 数据 ≥ 50,000 场 | 大幅提升 |
| P3 | Cross-league transfer 重新评估 | 多联赛数据 | P2.1/P2.4 结论可能改变 |

### 3.4 不做的事（明确排除）

- 不做 LLM-based 方法（不把赔率转文本让 GPT 预测）
- 不做强化学习 betting agent
- 不做实时 in-play 预测（需要实时数据流）
- 不做 complex ensemble（保持模型简单可解释）
- 手写规则 gate（所有选择基于回测）

---

*本方案基于项目当前状态和文献调研编写。实施节奏取决于数据收集进度。*
