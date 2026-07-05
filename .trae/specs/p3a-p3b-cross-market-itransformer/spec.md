# P3a/P3b Cross-Market OddsPatch-iTransformer Spec

## Why

P3a (EuroPatch-iTransformer) 代码已生成但从未运行验证。当前只有 9 维欧赔特征，不是真正的 cross-market Transformer。必须先验证 P3a 能跑通，再实现 P3b 加入亚盘+大小球特征，才能证明 Transformer 路线在赔率预测上的有效性。

## What Changes

- **P3a 验证**: import/params/forward/tiny-overfit smoke tests
- **新增**: `tools/p3_bucket_occupancy_report.py` — 验证 time bucketing 是否真实有效
- **P3b 实现**: `model/odds_patch_itransformer_v2.py` + `tools/p3b_train_patch_itransformer.py`
- P3b 加入 Euro + Asian + OU 共 20+ 特征的 cross-market tokenization
- P3b 增加 market_type embedding (euro/asian/ou/time)
- P3a vs P3b vs RawPooledMLP 公平对比

## Impact

- Affected code:
  - `model/odds_patch_itransformer.py` (P3a, 验证不修改)
  - `tools/p3_train_patch_itransformer.py` (P3a, 可能需修 --max-samples)
  - `model/odds_patch_itransformer_v2.py` (新增, P3b)
  - `tools/p3b_train_patch_itransformer.py` (新增, P3b)
  - `tools/p3_bucket_occupancy_report.py` (新增)

## ADDED Requirements

### Requirement: P3a Smoke Verification
P3a EuroPatch-iTransformer SHALL pass import, params, forward, and tiny-overfit smoke tests.

#### Scenario: Import + Params
- **WHEN** `python -c "from model.odds_patch_itransformer import OddsPatchITransformer; m = OddsPatchITransformer(); print(sum(p.numel() for p in m.parameters()))"`
- **THEN** 模型可导入，参数量 < 100K

#### Scenario: Forward Smoke
- **WHEN** 输入 `torch.randn(4, 10, 9, 2)`
- **THEN** 输出 shape `[4, 3]`，无 NaN/Inf

#### Scenario: Tiny Overfit
- **WHEN** 128 样本训练 50 epochs
- **THEN** train_loss 明显下降，train_acc > 0.45

### Requirement: Bucket Occupancy Report
系统 SHALL 提供 time bucketing 有效性验证脚本。

#### Scenario: 正常数据
- **WHEN** 数据包含有效 `minutes_before_kickoff`
- **THEN** 输出 bucket 分布、avg_non_empty_buckets、closing_event_ratio

#### Scenario: 退化检测
- **WHEN** `closing_event_ratio > 0.8`
- **THEN** 输出 WARN: time bucketing likely collapsed

### Requirement: P3b Cross-Market Features
P3b SHALL 支持 Euro + Asian + OU 共 >= 20 个特征。

#### Scenario: 特征列表
- **WHEN** 构建 P3b 模型
- **THEN** 包含 euro(8) + asian(7) + ou(7) + time_quality(3) = 25 特征

### Requirement: P3b Market Type Embedding
P3b SHALL 为每个特征附加 market_type embedding (euro/asian/ou/time)。

#### Scenario: Embedding 维度
- **WHEN** 特征数 F=25, d_model=64
- **THEN** 加上 market_type embedding(4 types) + feature_id embedding(25 ids)

### Requirement: P3b 参数量约束
P3b 参数量 SHALL < 150K。

### Requirement: P3b vs P3a 对比
P3b SHALL 输出与 P3a 和 RawPooledMLP(0.930) 的完整对比表。

## MODIFIED Requirements

无

## REMOVED Requirements

无

## Architecture Detail — P3b

```
Input: [B, 10, F=25, A=2] (10 buckets, 25 features, 2 agg: last+mean)

Reshape: [B, F, 10*A] = [B, 25, 20]

Per-feature embedding: F × Linear(20, d_model) → [B, 25, 64]
+ market_type_embedding: [25, 64] (euro=0, asian=1, ou=2, time=3)
+ feature_id_embedding: [25, 64]

Prepend CLS: [B, 26, 64]

Cross-Market Transformer (2 layers, d=64, heads=4, RMSNorm, norm_first=True)

Extract CLS → MLP Head → [B, 3]
```

## P3b 特征列表 (基于项目真实字段)

### Euro features (8)
- euro_h, euro_d, euro_a
- imp_h, imp_d, imp_a (computed: 1/odds, normalized)
- euro_margin (computed: sum(1/odds) - 1)
- has_euro

### Asian features (7)
- asian_line
- upper_water, lower_water
- asian_upper_implied (computed: 1/upper_water, normalized)
- asian_lower_implied (computed: 1/lower_water, normalized)
- asian_margin (computed)
- has_asian

### Over/Under features (7)
- over_under_line
- over_water, under_water
- over_implied (computed: 1/over_water, normalized)
- under_implied (computed: 1/under_water, normalized)
- ou_margin (computed)
- has_ou

### Time/Quality features (3)
- time_log (log1p(minutes_before_kickoff))
- bucket_valid_count (events in this bucket)
- market_present_count (how many of euro/asian/ou present)
