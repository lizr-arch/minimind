# Checklist

## P3a Smoke Verification
- [x] A1: 文件存在 (`model/odds_patch_itransformer.py`, `tools/p3_train_patch_itransformer.py`)
- [x] A2: Import 成功，参数量 < 100K (已验证: 81,731)
- [ ] A3: Forward smoke 通过 (需用户手动运行)
- [x] A4: `--max-samples` 参数已添加 + train_acc 输出
- [ ] A5: Tiny overfit 通过 (需用户手动运行)

## Bucket Occupancy Report
- [x] B1: `tools/p3_bucket_occupancy_report.py` 已创建
- [x] B2: bucket_event_counts 和 bucket_match_coverage 输出正确
- [x] B3: 退化检测逻辑正确 (closing_event_ratio, avg_non_empty_buckets, minutes_zero_ratio)
- [x] B4: JSON 报告格式正确，warnings 逻辑正确

## P3b Model Implementation
- [x] C1: `model/odds_patch_itransformer_v2.py` 已创建
- [x] C2: 25 特征提取正确 (euro 8 + asian 7 + ou 7 + time_quality 3)
- [x] C3: market_type embedding 正确 (4 types)
- [x] C4: feature_id embedding 正确 (25 ids)
- [x] C5: V2 bucketize_sample 输出 [10, 25, 2]
- [ ] C6: 参数量 < 150K (需用户手动运行)

## P3b Training Script
- [x] D1: `tools/p3b_train_patch_itransformer.py` 已创建
- [x] D2: 数据加载正确 (OddsDataset + splits_v6)
- [x] D3: V2 bucketed 特征提取正确
- [x] D4: 评估指标完整 (accuracy, logloss, brier, ECE)
- [x] D5: `--max-samples` 参数支持
- [x] D6: report.json 输出格式正确 (含 phase, params, verdict, warnings)

## P3b Verification (需用户手动运行)
- [ ] E1: P3b import + params smoke 通过
- [ ] E2: P3b forward smoke 通过
- [ ] E3: P3b tiny overfit 通过
- [ ] E4: P3a full train 完成
- [ ] E5: P3b full train 完成
- [ ] E6: 对比表已输出

## P0.1 Shuffled Label (可选)
- [ ] F1: shuffled label sanity 已重跑
- [ ] F2: 结果已记录

## Code Quality
- [x] 无 inline class 定义
- [x] 无 sklearn 依赖
- [x] 无赛后数据泄露
- [x] 字段名与项目一致 (euro_h, asian_line, over_under_line, upper_water, lower_water, over_water, under_water)
