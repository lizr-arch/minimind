# Phase 1a Baseline v2 — 诊断报告

**日期**: 2026-06-26
**数据**: v6_phase1a.jsonl (35,330 rows, 11,777 unique matches)
**庄家**: Pinnacle, Bet365, Sbobet, Macau
**Test split**: 1,768 matches

---

## 1. 基线版本演进

| 版本 | 方法 | Pinnacle acc | Pinnacle logloss |
|---|---|---|---|
| v1 (旧的) | 纯欧赔 close_no_vig | 0.602 | 0.881 |
| v2 (方案 C) | 欧赔+亚盘联合，仅 line=0.0/-0.5 | 0.602 | 0.881 |
| v2+Poisson | 欧赔+亚盘联合，全部盘口 | 0.506 | 0.955 |

**结论：v2 联合分布在 Pinnacle 上等价于 v1（delta=0），因为 Pinnacle 欧赔和亚盘内部完全自洽。**
**Poisson 建模的 quarter-ball 盘口反而恶化预测——不应使用。**

---

## 2. 为什么联合分布没有改善？

### 2.1 Pinnacle 内部自洽

Pinnacle 的欧赔和亚盘来自同一个定价体系：
- 欧赔去水后的 q_h 和亚盘去水后的 p_upper（line=-0.5）差距 < 0.001（93.3% 的比赛）
- 欧赔去水后的 r_ah = q_h/(q_h+q_a) 和亚盘去水后的 p_upper（line=0.0）差距 < 0.001

这意味着 Pinnacle 的欧赔和亚盘是同一个概率分布的不同表达——没有增量信息。

### 2.2 Bet365 的分歧是噪声

Bet365 的欧赔和亚盘分歧显著（median |diff| = 0.021），但联合分布**恶化**预测：
- acc 从 0.603 → 0.511 (-0.092)
- logloss 从 0.879 → 0.950 (+0.071)

Bet365 的亚盘水位反映的是头寸管理而非真实概率估计。分歧 = 噪声。

### 2.3 Poisson 假设引入偏差

Poisson 建模假设 home_goals ⟂ away_goals | λ_h, λ_a。这个假设在足球中不完全成立（弱队面对强队时进球分布不是独立的）。加上 max_goals=8 的截断误差，导致 quarter-ball 的约束优化引入了额外的偏差。

---

## 3. 多庄家分析摘要

| 假设 | 结论 |
|---|---|
| H1: Margin 差异 | Bet365 最低 (0.067)，Pinnacle (0.082)，Macau 最高 (0.116)。Pinnacle 并非 margin 最低的庄家。 |
| H2: Favorite-Longshot Bias | 详见 `bookmaker_analysis.json`。所有庄家均有轻微 favorite bias。 |
| H3: Home systemic bias | 详见 `bookmaker_analysis.json`。 |
| H4: Lead-Lag | 延期至 Phase 1b（需 timeline 数据）。 |
| H5: 公司间分歧 vs 不确定性 | 分歧不显著预测 home_rate（跨桶 0.45-0.49，差异 < 0.04）。庄家间分歧不是有用的不确定性信号。 |
| H6: 欧亚不一致 vs 预测误差 | **关键发现**：欧亚不一致越大 → home_rate 越低（45%→24%）→ logloss 越低（0.68→0.40）。欧亚分歧 = 主队被高估信号，预测反而更容易。Phase 2 的强信号。 |

---

## 4. Phase 1a 基线最终建议

**Phase 1a baseline = Pinnacle 纯欧赔去水（close_no_vig_euro）。**

理由：
1. Pinnacle 的欧赔和亚盘内部自洽——联合分布不提供增量信息
2. Poisson 建模引入的假设偏差大于约束带来的精度提升
3. 亚盘信息的价值不在静态 1X2 预测，而在时序变化（Phase 1b）和市场分歧信号（Phase 2）

**Track 1（市场复现度）的 baseline 即 Pinnacle close_no_vig_euro。**
**Track 2（赛果预测）的 baseline 即 Pinnacle close_no_vig_euro 对真实赛果的指标。**

---

## 5. Test Set 最终基线指标

| 指标 | Pinnacle close_no_vig_euro |
|---|---|
| accuracy | 0.602 |
| logloss | 0.881 |
| Brier | 0.515 |
| ECE | 0.061 |
| 样本数 | 1,768 |

按联赛拆分详见 `eval/reports/phase1a_baseline_v2.json`。

---

## 6. 对 Phase 1a 模型训练的启示

- Phase 1a 模型的目标不是"超越基线"，而是"忠实复现市场定价"（Track 1）
- 损失函数应包含 KL(pred, market_prior)，market_prior = Pinnacle 纯欧赔去水
- 亚盘特征不应作为约束纳入 baseline，但可以作为模型输入特征（Phase 1a 模型可以看到亚盘信息）
- Phase 2 的信号来自 H6：欧亚不一致大的比赛 = 主队被高估 = 预测更容易（可做 under bet）

---

## 7. Phase 1b 展望

- 亚盘的价值可能在**时序维度**：盘口变化方向、临场加速、欧亚变化背离等
- 大小球的时序变化 + 亚盘的时序变化联立，可能提供欧赔无法表达的净胜球相关信息
- H4（lead-lag）需要在 timeline 格式数据上完成
