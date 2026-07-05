# Phase 1b — 从分析到实施的决策文档

## 一、分析结论到工程决策的映射

### 结论 1：低事件 = 高准确度 ✅

> 极低事件组（1-5）accuracy 58.5%、logloss 0.924，均优于高事件组。

** → 工程决策：事件数是信号，不是噪声。**

具体做法：
- 在 event-driven Transformer 中，事件数本身作为 match-level 特征（在 pooling 后拼接）
- 模型应该学会"安静的赔率 = 定价更可信"
- 这改变了 Phase 1a 的认知——当时我们认为低事件比赛"信息量少"

### 结论 2：Pinnacle 领先 Bet365，方向一致率 67.6%

> 30,649 个可交易信号，Pinnacle 先动 → Bet365 后续同向的概率 67.6%。

** → 工程决策：添加 lead-lag 特征。**

具体做法：
- 对于非 Pinnacle 庄家的比赛，取 Pinnacle 对同一 match 的 timeline
- 在每个事件时间点，判断 Pinnacle 是否已经朝同一方向变动
- 特征值：`-1`（Pinnacle 反向已动）/ `0`（Pinnacle 未动或缺失）/ `+1`（Pinnacle 同向已动）
- 这是一个 event-level 特征，随序列输入 Transformer

### 结论 3：欧亚同向 = 主队胜率 +22pp

> 欧赔和亚盘同时看好主队时，主队胜率 64.1%（基线 42%）。

** → 工程决策：添加欧亚一致性特征。**

具体做法：
- 每个事件计算：`sign(euro_h_change) == sign(asian_favor_home)`
- 欧赔变化方向：euro_h 降 → 看好主队 (+1)，升 → 不看好 (-1)，不变 → 0
- 亚盘方向：盘口向主队倾斜 → +1，向客队倾斜 → -1
- 特征 = 两者符号是否一致（-1 / 0 / +1）

### 结论 4：临场加速不显著

> "临场爆发"模式仅 0.4%，"先动后静"占 20.7%。

** → 工程决策：不需要特殊的时间分桶策略。**

具体做法：
- 直接用 `minutes_before_kickoff` 和 `time_delta_from_prev_event` 作为连续特征
- 不做 168h/72h/24h/6h/1h 固定快照——事件驱动本身已经保留了真实时间结构
- 模型自己学习"最后 1 小时的变化权重不同"

### 结论 5：大小球覆盖率 74.2%

> 新数据已修复，OU 缺失用 null + has_over_under=false 标记。

** → 工程决策：纳入大小球，缺失用 mask。**

具体做法：
- 事件中有 OU 数据 → 喂入 OU 特征（ou_line, over_water, under_water）
- 事件中无 OU 数据 → OU 位置填 0，missing_mask 对应位为 1
- 这复用现有的 missing_mask 机制，不需要额外架构

---

## 二、Phase 1b 与 Phase 1a 的关键区别

| 维度 | Phase 1a | Phase 1b |
|---|---|---|
| 数据 | 静态快照（开/收） | 完整事件序列 |
| 序列长度 | 1（单帧） | ≤128（变长） |
| 模型 | 4K 参数 MLP | ~13M 参数 Transformer |
| 特征 | 18 维静态 | ~31 维 × 128 步 |
| 大小球 | 跳过了 | 纳入（mask） |
| 时序信号 | 无 | lead-lag、欧亚一致性、事件密度 |
| 目标 | 学习市场定价 | 学习市场定价 + 定价变化规律 |

---

## 三、新数据的增量信息

`titan007_pure_v4_market_semantics_v2.jsonl` 比 v6 多：

| 字段 | Phase 1b 用途 |
|---|---|
| `snapshot_type` (opening/movement) | 区分初始定价 vs 市场修正——初始定价的事件应该被模型不同对待 |
| `market_updated` (1x2/asian/ou) | 知道"这次变动是哪个市场触发的"——欧赔变动和亚盘变动的含义不同 |
| `*_source` (raw_update/forward_fill/missing) | 区分真实变动 vs 插值填充——forward_fill 的事件不是真正的市场信号 |
| `season`, `competition_type`, `is_knockout` | 比赛背景 embedding——杯赛淘汰赛和联赛的赔率行为不同 |
| `home_team`, `away_team` | Phase 2 队级分析——Phase 1b 暂不用 |

**关键发现：`market_updated` 目前全是 "1x2"。** 这意味着原始数据中变化原因标记可能不够细，或者导出脚本未正确填充。如果后续修复，这个字段将区分"欧赔变动事件"vs"亚盘变动事件"vs"大小球变动事件"——对于事件驱动建模非常有价值。当前先用 `snapshot_type` + 各市场的 has_* 变化作为替代。

---

## 四、Phase 1b 执行步骤

### Step 1：数据适配
- 扩展现有 `OddsDataset` 以支持 `titan007_pure_v4_market_semantics_v2.jsonl` 格式
- 新增 event-level 特征提取函数（v6 风格的 31 维特征 + 新字段）
- 复用 `splits_v6`（覆盖率 94.3%），剩余 709 场新比赛归入 train

### Step 2：Event-Driven Collator
- 变长序列 → padding + attention mask
- max_seq_len=128，截断保留最近事件
- 实现 lead-lag 特征计算（跨庄家查找 Pinnacle timeline）

### Step 3：模型训练
- 基于现有 `OddsMindModel`，输入单庄家的事件序列
- 加入新特征维度（snapshot_type embedding, market_updated embedding, lead-lag, euro-asian alignment）
- 训练目标：CE(euro) + CE(asian) + β·KL(euro, market_prior)
- 双轨评估

### Step 4：消融实验
- 事件数特征 on/off
- Lead-lag 特征 on/off
- OU 特征 on/off
- 对比 Phase 1a baseline

---

## 五、风险评估

| 风险 | 概率 | 缓解 |
|---|---|---|
| Lead-lag 特征在训练时信息泄露（用了未来庄家的数据） | 中 | 严格按时间戳对齐，只用"此刻之前"的 Pinnacle 变动 |
| 事件序列过长（>128）的比赛被截断丢失信息 | 低 | P99=170，截断 128 丢 42 个最旧事件——旧事件信息量低 |
| `market_updated` 全为 "1x2" 导致市场区分失效 | 中 | 用各市场 has_* 字段的变化自行推断哪个市场变动了 |
| OU 74% 覆盖率在训练中导致 OU 特征利用不充分 | 低 | missing_mask 机制已验证有效，模型学会忽略缺失 OU |
