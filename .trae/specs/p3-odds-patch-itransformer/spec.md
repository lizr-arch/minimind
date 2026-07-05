# P3 OddsPatch-iTransformer Spec

## Why

当前最优模型 RawPooledMLP（9维欧赔均值池化 + 3层MLP）达到 val_logloss=0.930，但缺乏对赔率时间序列结构的建模能力。Event-token Transformer 因异质特征混合导致失败（logloss=1.11），iTransformer [B,F,T] 因过拟合而失败（val_logloss=1.03）。核心洞察：需要 channel-independent 的时间编码 + 跨市场 Transformer 注意力。

## What Changes

- 实现 OddsPatch-iTransformer 架构：time bucketing → PLE/RAW embedding → channel-independent temporal encoder → cross-market Transformer → CLS → prediction head
- 基于 P2 PLE 对比结果决定是否使用 PLE embedding（若 PLE ≤ RAW 则使用，否则用 RAW）
- 新增模型文件 `model/odds_patch_itransformer.py`
- 新增训练脚本 `tools/p3_train_patch_itransformer.py`
- 复用现有 time bucketing、数据集、评估指标等基础设施

## Impact

- Affected specs: P2 PLE embedding 结果决定 embedding 方案
- Affected code:
  - `tools/p2_ple_embedding.py` (需先运行获取 PLE 对比结果)
  - `model/odds_patch_itransformer.py` (新增)
  - `tools/p3_train_patch_itransformer.py` (新增)
  - `tools/p1_time_bucketing.py` (复用 BUCKET_EDGES)
  - `eval/odds_metrics.py` (复用评估指标)

## ADDED Requirements

### Requirement: OddsPatch-iTransformer 模型架构

系统 SHALL 提供基于 iTransformer 的赔率预测模型，包含以下组件：

1. **Time Bucketing Layer**: 将 128 padding 事件序列转换为 10 个固定时间桶
2. **Feature Embedding**: 每个特征独立映射到 d_model=64 维空间（支持 PLE 或 RAW）
3. **Channel-Independent Temporal Encoder**: 每个特征独立编码时间变化模式
4. **Cross-Market Transformer**: 特征间 attention（欧赔 ↔ 亚盘 ↔ 大小球）
5. **CLS Token Pooling**: 使用可学习 [CLS] token 汇聚全局信息
6. **Prediction Head**: 3 分类（home/draw/away）

#### Scenario: 模型前向传播
- **WHEN** 输入 [B, 10_buckets, F_core=9, AGG=2] 的桶化特征
- **THEN** 输出 [B, 3] 的 logits，参数量 < 100K

#### Scenario: 训练收敛
- **WHEN** 使用 8243 训练样本训练 50 epochs
- **THEN** val_logloss ≤ 0.930（打平或优于 RawPooledMLP）

### Requirement: 核心特征选择

系统 SHALL 优先使用 9 维核心欧赔特征：
- euro_h, euro_d, euro_a (赔率)
- imp_h, imp_d, imp_a (隐含概率)
- margin (利润率)
- time_log (时间对数)
- has_euro (可用性标记)

每个特征在每个时间桶内聚合为 2 个统计量：last, mean。

#### Scenario: 特征维度验证
- **WHEN** 处理单个样本
- **THEN** 输入张量形状为 [B, 10, 9, 2] = 180 维

### Requirement: 参数量约束

系统 SHALL 将模型参数量控制在 100K 以内，以避免在 8K 训练样本上过拟合。

#### Scenario: 参数量检查
- **WHEN** 模型初始化完成
- **THEN** `sum(p.numel() for p in model.parameters())` < 100,000

## MODIFIED Requirements

### Requirement: Time Bucketing 复用
现有 `tools/p1_time_bucketing.py` 的 BUCKET_EDGES 定义将被复用，但聚合方式简化为 last+mean（去掉 std/delta/count）以降低维度。

## REMOVED Requirements

无

## Architecture Detail

```
输入: [B, T_raw=128, F=33] 原始赔率事件
  ↓
Time Bucketing: 10 个固定时间桶 (opening/7d/3d/1d/12h/6h/3h/1h/30m/closing)
  ↓
核心特征提取: 9 维 × 2 聚合 = 180 维 per sample
  ↓
Reshape: [B, 10, 9, 2] → [B, 9_features, 10_buckets × 2_agg] = [B, 9, 20]
  ↓
Feature Embedding: Linear(20, d_model=64) per feature → [B, 9, 64]
  ↓
[CLS] Token Prepend: [B, 10, 64] (10 = 9 features + CLS)
  ↓
Transformer Encoder (norm_first=True):
  - Layer 1-2: Self-Attention over features (cross-market)
  - d_model=64, n_heads=4, FFN=128
  ↓
Extract CLS: [B, 64]
  ↓
MLP Head: Linear(64, 32) → SiLU → Linear(32, 3)
  ↓
输出: [B, 3] logits
```

## 关键设计决策

1. **Channel-Independent**: 每个特征（欧赔H/D/A、隐含概率等）独立编码时间模式，避免异质特征混淆
2. **Cross-Market Attention**: Transformer 层让不同市场特征（欧赔、亚盘、大小球）相互交互
3. **norm_first=True**: Pre-norm Transformer 更稳定，避免梯度爆炸
4. **d_model=64**: 小维度控制参数量，适合小数据集
5. **仅 1X2 预测**: 简化目标，先验证核心架构有效性

## 已知坑（来自交接文档）

1. 编码器 bias=True 会导致 padding 污染 → 使用 bias=False
2. LayerNorm 让所有事件表示趋同 → 使用 RMSNorm 或去除
3. inline class 定义在 python -c 中会缩进报错 → 使用文件脚本
4. sklearn 未安装 → 使用纯 PyTorch 实现
