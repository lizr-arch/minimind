# Phase 1b — 大小球数据审计提示词

> 复制以下内容发送给另一个大模型执行数据审计。

---

## 项目归属

| 项 | 值 |
|---|---|
| **项目类型** | 数据项目 |
| **开发路径** | `D:\code\git\odds-data-probe` |
| **验收路径** | `D:\code\git\odds-data-probe`（产出报告和修复后的数据） |
| **产出交付到** | 大模型训练项目 `D:\code\git\betmind\minimind\data\odds_real\` |

---

你是 OddsMind 足球预测系统的数据审计员。请检查项目中的大小球（Over/Under）数据，回答以下问题并产出审计报告。

## 背景

OddsMind 是一个从赛前赔率时间序列预测足球比赛结果的 Transformer 模型。当前处于 Phase 1b 规划阶段，需要确定大小球数据是否可以作为训练特征。Phase 1a 已验证欧赔+亚盘的静态快照，Phase 1b 计划加入大小球和赔率时序变化。

核心问题：大小球盘口和水位编码了市场对总进球数的预期。和亚盘（净胜球预期）联立，可以反推主客队各自的进球期望 `λ_h, λ_a`。但数据覆盖率、质量、去水方法需要先审计。

## 数据位置

- **扁平格式**（静态快照）：`data/odds_real/v6_phase1a.jsonl`（35,330 行，每行一场比赛+一个庄家）
- **时间线格式**（多时间点）：`data/odds_real/v6_all.jsonl`（35,330 行，每行含 odds_timeline 数组）
- **数据切分**：`data/odds_real/splits_v6/`（train/val/test match_ids）

字段说明：
- `close_ou_line` / `open_ou_line`：大小球盘口（如 2.5、2.75、3.0）
- `close_ou_over_water` / `close_ou_under_water`：大球/小球水位
- `odds_timeline` 中对应字段：`over_under_line`, `over_water`, `under_water`
- `home_goals` / `away_goals`：实际进球数（label 中）

## 请完成以下检查

### 1. 覆盖率分析

- 按庄家（Pinnacle/Bet365/Sbobet/Macau）统计 close_ou_line 的缺失率
- 按联赛统计缺失率（哪些联赛缺少大小球数据？）
- 按时间统计缺失率（是否早期比赛缺大小球数据更多？用 kickoff_time 字段分组）
- v6_all.jsonl 的 timeline 中 `has_over_under` 字段的覆盖情况（每场比赛的 timeline 中有多少比例的时间点包含大小球数据）
- 结论：哪些子集可以作为 Phase 1b 大小球训练数据？

### 2. 盘口值分布

- ou_line 的值分布（2.0, 2.25, 2.5, 2.75, 3.0, 3.5... 各多少）
- 盘口和联赛的关系（是否某些联赛固定用某个盘口值？如低进球联赛 2.0，高进球联赛 3.0？）
- 盘口和亚盘盘口（asian_line）的关系：两者是否存在相关性？（预期：实力悬殊的比赛 asian_line 大、ou_line 也大）
- 开盘 ou_line 和收盘 ou_line 的变化频率（大小球盘口是否会变动？）

### 3. 水位数据质量

- over_water 和 under_water 的值域范围（是否存在异常值如 0.01 或 9.99？）
- over_water 和 under_water 的隐含 margin 分布——和欧赔 margin 对比（预期：大小球 margin 通常低于欧赔）
- 是否存在 over_water < under_water 但总进球盘口含义不匹配的情况？
- 不同庄家的大小球水位差异（Bet365 vs Pinnacle 的水位差分布）

### 4. 大小球和实际总进球的关系（校准分析）

- 不同 ou_line 下实际总进球分布（如 2.5 盘口的比赛，实际 >2.5 的比例是多少？）
- 去水后大球概率 vs 实际大球率的 ECE 校准分析（和欧赔 ECE 对比）
- 大小球隐含概率是否比欧赔隐含概率更难/更准？（对比 logloss）
- 大小球和亚盘的联立：用 ou_line + asian_line 反推主客队进球期望 λ_h, λ_a，验证：
  - 反推的 λ_h + λ_a ≈ actual_home_goals + actual_away_goals？
  - 反推的 λ_h - λ_a ≈ actual_home_goals - actual_away_goals？

### 5. 缺失模式诊断

- close_ou 缺失的比赛是否同时缺失 open_ou？
- 缺失大小球数据的比赛，其欧赔/亚盘覆盖是否完整？
- 缺失是否有系统性模式：特定联赛、特定时间（如早年间）、特定盘口类型的比赛统一缺失？
- 缺失比赛的实际总进球分布和未缺失比赛是否有显著差异？（排除"缺失 ≠ 随机"的可能性）

## 产出要求

一份 Markdown 报告，包含：

1. 每个检查项的统计数据（表格，含计数、比例、均值/中位数）
2. **是否推荐 Phase 1b 纳入大小球**：
   - 给出推荐等级（强推荐/有条件推荐/暂不推荐）
   - 列明理由（数据质量、覆盖率、预测增量）
   - 评估风险（缺失偏差、去水方法假设）
3. 如果纳入，建议的：
   - 缺失值处理策略（impute / mask / 仅用完整子集）
   - 特征构造方式（原始赔率 / 去水概率 / 变化量）
   - 最小覆盖率要求
4. 如果暂不纳入，说明最小数据要求是什么

## 注意事项

- 使用 Python 脚本分析，但最终输出为可读的 Markdown 报告
- 所有统计数字保留 4 位小数
- 关键结论加粗标注
- 数据路径使用相对于 `minimind/` 的路径
