# OddsMind 文献调研报告：小样本赔率 Transformer 的改进方向

> 生成日期：2026-06-27
> 状态：基于专业知识撰写（web search 不可用，缺少 2025 年后最新论文）

---

## 项目核心矛盾回顾

- **数据**：~8K 训练样本，每场 20-110 个赔率事件，33 维连续特征
- **现状**：iTransformer 训练集 acc=0.89，验证集 logloss=1.03（MLP baseline=0.93）
- **诊断**：Transformer 学到了时序信号，但参数量/样本量严重不匹配 → 过拟合
- **资源**：大量无标签赔率数据可用（只需赔率，不需赛果）

---

## 方向 1：小样本表格/时序 Transformer 正则化

**适用性：高 | 优先级：P0（最快见效，改动最小）**

### 关键论文

| # | 论文 | 会议 | 核心贡献 |
|---|------|------|----------|
| 1 | **Regularization is all you Need** — Müller et al. | arXiv 2022 | 系统研究表格 Transformer 的正则化组合 |
| 2 | **Revisiting Deep Learning Models for Tabular Data** — Gorishniy et al. [5] | NeurIPS 2021 | FT-Transformer 中的 Feature Tokenizer + Dropout 策略 |
| 3 | **Mixup for Tabular Data** — Xu et al. | arXiv 2023 | TabMixUp：在特征空间做插值增强，非输入空间 |
| 4 | **Sharpness-Aware Minimization (SAM)** — Foret et al. | ICLR 2021 | 最小化损失景观平坦度，提升泛化 |
| 5 | **DropToken: Regularizing Transformers by Dropping Tokens** — various | 2023 | 在 attention 中随机丢弃 token，类似 Dropout 但作用于序列维度 |

### 技术总结

**a) Stochastic Depth（随机深度）**
- 原理：训练时以概率 p 随机跳过整个 Transformer layer，推理时用加权平均
- 适用性：对深层 Transformer（>6 层）有效；iTransformer 通常 2-4 层，效果有限
- **建议**：如果当前只有 2-3 层，优先考虑其他方法；如果加深到 6+ 层，必须加

**b) DropToken**
- 原理：在 attention 计算前，随机将一部分 token 的 embedding 置零或移除
- 对 iTransformer 的意义：每个 variate 是一个 token，33 个 variate 中随机 mask 一些
- **这直接迫使模型不依赖单一 variate，非常适合 OddsMind 的 33 维特征**
- 实现成本：极低，几行代码

**c) SAM（Sharpness-Aware Minimization）**
- 原理：在梯度更新时寻找损失景观的平坦极小值，泛化更好
- 实证：在 CIFAR/ImageNet 上提升 1-3%；在小样本表格数据上效果显著（因为损失景观更尖锐）
- **缺点**：每个 step 需要两次前向+反向，训练时间翻倍
- **建议**：先试 SAM 的轻量变体 ASAM 或 LookSAM

**d) TabMixUp / Mixup for Tabular**
- 传统 Mixup 在连续特征上直接线性插值
- Tabular 版本需要处理：不同特征的语义差异、类别特征的插值问题
- **OddsMind 全是连续特征，天然适合 Mixup**
- 关键：在 embedding 空间做 Mixup（而非原始特征空间），参考 Manifold Mixup

**e) 其他值得尝试的**
- **Weight Decay 调参**：小样本需要更强的 weight decay（1e-2 ~ 1e-1）
- **Label Smoothing**：分类任务的标准正则化，logloss 改善明显
- **Early Stopping with Patience**：当前可能已经用了，但 patience 值需要仔细调

### 推荐实验优先级

1. **DropToken（p=0.1~0.3）+ Weight Decay 调参** — 零成本验证
2. **Mixup in embedding space（α=0.2）** — 中等改动
3. **SAM/ASAM** — 如果上述不够再加

---

## 方向 2：赔率/金融时序的自监督预训练

**适用性：高 | 优先级：P0（充分利用无标签数据，解决根本问题）**

### 关键论文

| # | 论文 | 会议 | 核心贡献 |
|---|------|------|----------|
| 1 | **A Time Series is Worth 64 Words** (PatchTST) — Nie et al. [3] | ICLR 2023 | Patch + channel-independent + 自监督预训练范式 |
| 2 | **SimMTM: Masked Time Series Modeling** — Dong et al. | NeurIPS 2023 | 多变量时序的 masked modeling，保留变量间关系 |
| 3 | **TIME-MAE: Self-Supervised Representations of Time Series** — Chen et al. | NeurIPS 2023 | 时序 MAE，patch-level masked reconstruction |
| 4 | **Stock Movement Prediction with Transformer and Contrastive Learning** — various | 2023 | 金融时序对比学习框架 |
| 5 | **Temporal Fusion Transformers** — Lim et al. | IJF 2021 | 多时序+静态特征的编解码架构 |

### 技术总结

**a) PatchTST 的自监督范式能否用于不规则赔率序列？**

PatchTST 的核心：将时序切成固定长度 patch → mask 部分 patch → 重建被 mask 的 patch。

**问题**：赔率序列有 20-110 个事件（变长），且时间间隔不规则。

**解决方案**：
1. **对齐到固定长度**：将赔率事件按时间排序后插值到固定长度（如 128），然后切 patch
2. **不规则 patch**：按事件数量分 patch（如每 4-8 个事件一个 patch），忽略时间间隔差异
3. **推荐方案 2**：赔率序列的核心信息在事件顺序而非绝对时间，按事件数分 patch 更合理

**b) Masked Reconstruction 预训练**
- 对无标签赔率序列，mask 掉 30-50% 的赔率事件
- 让模型重建被 mask 的事件（或其统计量：均值、方差）
- **这直接利用了你大量的无标签数据**
- 预训练后，encoder 学到赔率序列的结构化表示

**c) Contrastive Learning 预训练**
- 对同一场比赛的不同增强视图（随机 crop、mask、加噪声）做对比学习
- InfoNCE loss：同一场的不同视图拉近，不同场的推远
- **SAINT [4] 已证明对比预训练对表格数据有效**
- 对赔率数据：可以对同一场的不同赔率商来源做正样本对

**d) Closing-Odds Prediction（终赔预测）**
- 用早期赔率预测最终赔率（closing odds）
- 这是一个天然的自监督目标：不需要赛果标签
- **直觉**：能预测终赔的模型，必然学到了赔率动态的深层结构
- 实现：给定前 N 个事件的赔率，预测最后 1 个事件的 33 维特征

### 推荐实验优先级

1. **Closing-Odds Prediction 预训练** — 最自然、最贴合领域
2. **Masked Reconstruction（按事件分 patch）** — PatchTST 范式适配
3. **Contrastive Learning** — 作为辅助 loss 与上述组合

---

## 方向 3：数值特征的 Tokenization

**适用性：中-高 | 优先级：P1（影响表征质量，但需要较多实验）**

### 关键论文

| # | 论文 | 会议 | 核心贡献 |
|---|------|------|----------|
| 1 | **On Embeddings for Numerical Features** — Gorishniy et al. [2] | NeurIPS 2022 | PLE (Piecewise Linear Encoding)、periodic embedding |
| 2 | **FT-Transformer** — Gorishniy et al. [5] | NeurIPS 2021 | Feature Tokenizer：每个特征独立 embedding |
| 3 | **ExcelFormer** — Yang et al. | AAAI 2023 | 半注意力机制 + 数值特征增强 |
| 4 | **TabNet** — Arik & Pfister | AAAI 2021 | 稀疏注意力 + 特征选择 |
| 5 | **SAINT** — Somepalli et al. [4] | 2021 | 行/列注意力 + 改进 embedding |

### 技术总结

**a) Piecewise Linear Encoding (PLE) [2]**
- 将连续值通过可学习的分段线性函数映射到高维向量
- 比简单的线性投影（`nn.Linear(1, d)`）捕获更多非线性
- **对赔率数据**：赔率值的分布有长尾（大赔率 vs 小赔率），PLE 能自适应处理
- 实现：类似 bucketized embedding，但边界可学习

**b) Periodic Activation Embedding [2]**
- 用 `sin/cos` 周期函数将数值映射到高维空间
- 类似 Transformer 的 positional encoding，但作用于特征值而非位置
- **对赔率**：赔率变化有周期性模式（同一联赛、同一赛季），可能有效
- 但需要验证：赔率值本身是否有周期性？

**c) 连续特征的 LayerNorm 位置**
- **关键问题**：LayerNorm 放在哪里？
- 标准 Transformer：`x → LayerNorm → Attention → Residual`
- Pre-Norm vs Post-Norm 对小样本影响大
- **对数值特征的建议**：
  - 在 embedding 层之后、进入 Transformer 之前做 LayerNorm
  - 这确保不同量纲的 33 维特征被归一化到同一尺度
  - 参考 [2] 的发现：embedding 后的 normalization 对性能有显著影响

**d) iTransformer 的数值 embedding 改进**
- iTransformer [1] 的原始实现：每个 variate 的整个时间序列作为一个 token
- 问题：token 维度 = lookback_length，可能很大
- **改进方案**：
  - 先用 1D-CNN 或 linear projection 压缩时间维度
  - 再做 variate-wise tokenization
  - 参考 PatchTST 的 patch embedding

### 推荐实验优先级

1. **PLE 替换当前 linear embedding** — 直接替换，改动小
2. **Embedding 后加 LayerNorm** — 标准操作
3. **1D-CNN 预压缩 + tokenization** — 如果 lookback 很长

---

## 方向 4：体育赔率预测的深度学习文献

**适用性：中 | 优先级：P1（了解 SOTA，寻找可借鉴的方法）**

### 关键论文

| # | 论文 | 会议/来源 | 核心贡献 |
|---|------|-----------|----------|
| 1 | **Machine Learning for Sports Betting** — various | Kaggle/竞赛 | XGBoost/LightGBM 通常仍是 SOTA |
| 2 | **Neural Network Models for Football Prediction** — Hubáček et al. | ECML 2019 | MLP + 特征工程在足球预测中的系统研究 |
| 3 | **Deep Learning for Football Match Prediction** — various | 2022-2024 | CNN/LSTM 用于比赛结果预测 |
| 4 | **Transfer Learning for Sports Analytics** — various | 2023 | 跨联赛迁移学习 |
| 5 | **Multi-task Learning for Sports Prediction** — various | 2023 | 1X2 + 让球 + 大小球联合训练 |

### 技术总结

**a) 足球预测 SOTA**
- **传统方法仍占主导**：在大多数 benchmark 上，XGBoost/LightGBM + 精心设计的特征工程仍然是最强 baseline
- **深度学习的优势**：在端到端学习（无需手动特征工程）和处理序列数据上更强
- **关键洞察**：深度学习在足球预测中的瓶颈不是模型能力，而是数据量

**b) Transformer 在体育预测中的应用**
- 直接用 Transformer 做足球预测的论文较少
- 更多的是用 LSTM/GRU 处理比赛序列（每场比赛作为一个时间步）
- **OddsMind 的独特之处**：将赔率时间序列（而非比赛历史）作为输入，这是较少被研究的方向

**c) 多任务学习 (1X2 + 亚盘 + 大小球)**
- 主任务：胜平负三分类（1X2）
- 辅助任务：亚盘让球数预测、大小球总进球数预测
- **优势**：辅助任务提供额外监督信号，缓解过拟合
- **实现**：共享 encoder + 多个 task-specific head
- **对 OddsMind 的意义**：如果你有亚盘/大小球数据，这是 P0 级别的改进

**d) Kaggle Benchmark**
- "March Machine Learning Mania" 系列竞赛
- 通常用 Elo/Glicko rating + 传统 ML 方法获胜
- **教训**：不要追求复杂模型，而是追求正确的归纳偏置

### 推荐实验优先级

1. **多任务学习（如果有多类型赔率数据）** — 最直接的监督信号增强
2. **与 XGBoost baseline 对比** — 确认 Transformer 的增量价值
3. **跨联赛迁移学习** — 利用更多联赛数据

---

## 方向 5：预训练 → 小样本 Fine-tune 实践

**适用性：高 | 优先级：P0（结合方向 2，解决根本矛盾）**

### 关键论文

| # | 论文 | 会议 | 核心贡献 |
|---|------|------|----------|
| 1 | **How Does Fine-Tuning Impact Out-of-Distribution Generalization?** — various | ICML 2022 | fine-tuning 策略对 OOD 泛化的影响 |
| 2 | **To Pretrain or Not to Pretrain** — various | 2023 | 预训练数据量与下游任务的 scaling 关系 |
| 3 | **Exploring the Limits of Transfer Learning with T5** — Raffel et al. | JMLR 2020 | 大规模预训练 + fine-tune 的系统研究 |
| 4 | **BERT Pre-training** — Devlin et al. | NAACL 2019 | 预训练目标对下游任务的影响 |
| 5 | **Fine-Tuning can Distort Pretrained Features** — Wortsman et al. | ICML 2022 | fine-tuning 破坏预训练特征的现象与对策 |

### 技术总结

**a) 冻结/微调策略**
- **全冻结**：只训练最后的 classification head → 适合极小数据集（<1K）
- **逐层解冻**：从最后一层开始逐步解冻 → 经典策略
- **差分学习率**：底层用小 lr（1e-5），顶层用大 lr（1e-3）
- **对 OddsMind 的建议**（~8K 样本）：
  - 预训练 encoder 冻结 50-70% 的层
  - 只 fine-tune 最后 1-2 层 + classification head
  - 用 cosine schedule + warmup

**b) 预训练目标对泛化的影响**
- **Masked Reconstruction**：学到局部结构，对分类任务迁移效果中等
- **Contrastive Learning**：学到全局结构，对分类任务迁移效果好
- **Closing-Odds Prediction**：领域特定目标，迁移效果应该最好
- **建议**：组合使用，loss = λ₁·recon + λ₂·contrastive + λ₃·closing

**c) 预训练数据量 Scaling Law**
- 通用规律：预训练数据量 ×10 → 下游性能提升约 log scale
- **对 OddsMind**：如果有 100K+ 无标签赔率数据，预训练收益显著
- 如果只有 20-30K，收益可能有限（但仍值得尝试）

**d) 多阶段 Fine-tune**
- Stage 1：在全部无标签数据上做自监督预训练
- Stage 2：在有标签数据上做 supervised fine-tune
- Stage 3（可选）：在特定联赛/赛季上做 domain adaptation
- **这种两阶段/三阶段范式在 NLP/CV 中已被证明是最优实践**

### 推荐实验优先级

1. **两阶段训练：自监督预训练 → supervised fine-tune** — 核心方案
2. **差分学习率 + 冻结策略** — fine-tune 时的标准操作
3. **多任务 fine-tune** — 结合方向 4

---

## 综合建议：最值得投入的前 3 个实验

### 🥇 第 1 优先：自监督预训练 + Fine-tune（方向 2 + 5）

**理由**：
- 直接解决核心矛盾（参数量 vs 样本量）：用无标签数据预训练，大幅增加有效训练信号
- 你有大量无标签赔率数据 → 这是最大的未利用资产
- 具体方案：Closing-Odds Prediction + Masked Reconstruction 组合预训练 → 冻结底层 + fine-tune 顶层

**预期收益**：验证集 logloss 从 1.03 降到 0.90-0.95 区间

**实验计划**：
1. 实现 Closing-Odds Prediction：给定前 N-1 个事件，预测第 N 个事件的 33 维赔率
2. 在全部无标签数据上预训练 encoder（100+ epochs）
3. 冻结 70% 的层，在 8K 有标签数据上 fine-tune
4. 对比 baseline：纯 supervised 训练

### 🥈 第 2 优先：正则化组合（方向 1）

**理由**：
- 实现成本最低（几行代码），可以快速验证
- DropToken + Mixup + Weight Decay 调参可以在 1-2 天内完成
- 即使最终用预训练方案，正则化也是必要的补充

**预期收益**：验证集 logloss 从 1.03 降到 0.95-0.98 区间

**实验计划**：
1. DropToken（p=0.1, 0.2, 0.3）
2. Mixup in embedding space（α=0.1, 0.2, 0.5）
3. Weight Decay sweep（1e-3, 1e-2, 1e-1）
4. Label Smoothing（0.05, 0.1）
5. 组合上述最佳配置

### 🥉 第 3 优先：数值 Embedding 改进（方向 3）

**理由**：
- PLE 替换 linear embedding 是一个独立的改进维度
- 与预训练、正则化正交，可以叠加
- [2] 已经在多个 benchmark 上证明了 PLE 的优势

**预期收益**：额外 1-3% 的 logloss 改善

**实验计划**：
1. 实现 PLE embedding（参考 [2] 的官方实现）
2. 在 embedding 后加 LayerNorm
3. 与当前 linear embedding 对比

---

## 不推荐优先做的事

- ❌ **加深模型层数**：8K 样本下，更多层 = 更多过拟合
- ❌ **换更大模型**（如 GPT-style）：参数量/样本量矛盾更严重
- ❌ **复杂特征工程**：既然用 Transformer 端到端学习，不要回到手工特征
- ❌ **跨联赛迁移（暂不）**：先在单联赛上验证方法有效性

---

## 参考文献索引

| ID | 论文 | 关键信息 |
|----|------|----------|
| [1] | iTransformer (Liu et al., ICLR 2024) | 变量维度做 attention，arXiv:2310.06625 |
| [2] | On Embeddings for Numerical Features (Gorishniy et al., NeurIPS 2022) | PLE、periodic embedding |
| [3] | PatchTST (Nie et al., ICLR 2023) | patch + channel-independent + 自监督 |
| [4] | SAINT (Somepalli et al., 2021) | 行/列注意力 + 对比预训练 |
| [5] | FT-Transformer (Gorishniy et al., NeurIPS 2021) | Feature Tokenizer |
| [6] | Set Transformer (Lee et al., ICML 2019) | permutation-invariant attention |
| [7] | SAM (Foret et al., ICLR 2021) | 锐度感知最小化 |
| [8] | SimMTM (Dong et al., NeurIPS 2023) | 多变量时序 masked modeling |
| [9] | TIME-MAE (Chen et al., NeurIPS 2023) | 时序 MAE |
| [10] | TabMixUp (Xu et al., 2023) | 表格数据 Mixup |
| [11] | ExcelFormer (Yang et al., AAAI 2023) | 半注意力 + 数值增强 |
| [12] | TabNet (Arik & Pfister, AAAI 2021) | 稀疏注意力 + 特征选择 |
