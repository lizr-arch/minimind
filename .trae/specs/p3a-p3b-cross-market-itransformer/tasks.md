# Tasks

## Task A: P3a EuroPatch-iTransformer 验证

- [x] A1: 检查文件存在
- [x] A2: Import + params smoke (params=81,731)
- [ ] A3: Forward smoke (BLOCKED: Python hang)
- [x] A4: `--max-samples` + train_acc
- [ ] A5: Tiny overfit (BLOCKED: Python hang)

## Task B: Bucket Occupancy Report

- [x] B1: 创建 `tools/p3_bucket_occupancy_report.py`
- [x] B2: bucket 分布统计
- [x] B3: 退化检测
- [x] B4: JSON 报告 + warnings

## Task C: P3b OddsPatch-iTransformer 模型

- [x] C1: 创建 `model/odds_patch_itransformer_v2.py`
- [x] C2: 25 特征提取
- [x] C3: market_type embedding
- [x] C4: feature_id embedding
- [x] C5: V2 bucketize_sample
- [x] C6: **修复 has_asian/has_ou 逻辑** (添加 asian_source/over_under_source 检查 + 占位符检测)
- [ ] C7: 验证参数量 < 150K (BLOCKED: Python hang)

## Task D: P3b 训练脚本

- [x] D1: 创建 `tools/p3b_train_patch_itransformer.py`
- [x] D2: OddsDataset + splits_v6
- [x] D3: V2 bucketed 特征提取
- [x] D4: 评估指标
- [x] D5: `--max-samples`
- [x] D6: report.json

## Task E: P3b 训练与对比 (BLOCKED: Python hang)

- [ ] E1: P3b import + params smoke
- [ ] E2: P3b forward smoke
- [ ] E3: P3b tiny overfit
- [ ] E4: P3a full train
- [ ] E5: P3b full train
- [ ] E6: 对比表

## Task F: P0.1 Shuffled Label Sanity (可选)

- [ ] F1: shuffled label sanity
- [ ] F2: 结果记录

## 新增

- [x] 创建 `tools/p3b_feature_coverage_report.py` (Phase 4)
- [x] 修复 `has_asian` 逻辑: 添加 asian_source 检查 + asian_line=0/uw=1/lw=1 占位符检测
- [x] 修复 `has_ou` 逻辑: 添加 over_under_source 检查

# Task Dependencies

- Phase 0-2: 静态检查 ✅
- Phase 3: smoke (BLOCKED)
- Phase 4-5: report 脚本已创建，需运行
- Phase 6-7: 训练 (BLOCKED)

# 环境状态

Python 运行时不可用 (hermes-agent venv hang, WindowsApps stub error 9009)。
所有需要 Python 执行的任务 BLOCKED。
