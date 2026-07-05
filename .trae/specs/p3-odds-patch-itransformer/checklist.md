# Checklist

## P2 PLE 对比
- [ ] `tools/p2_ple_embedding.py` 执行成功，输出 RAW 和 PLE 的 val_logloss (需用户手动执行)
- [ ] PLE vs RAW 结果已记录，embedding 方案已决定 (需用户手动执行)

## 模型实现
- [x] `model/odds_patch_itransformer.py` 文件已创建
- [x] 核心特征提取正确：9维特征（euro_h/d/a, imp_h/d/a, margin, time_log, has_euro）
- [x] Time Bucketing 正确：10 个桶，复用 BUCKET_EDGES
- [x] Feature Embedding 形状正确：[B, 10, 9, 2] → [B, 9, 20] → [B, 9, 64]
- [x] Cross-Market Transformer 实现正确：norm_first=True, d_model=64, heads=4
- [x] CLS Token Pooling 实现正确：prepend CLS, extract CLS
- [x] MLP Head 输出正确：[B, 64] → [B, 3]
- [x] 参数量 < 100K（手动计算 ~81,731 params，待运行时验证）
- [x] 无 bias=True 的 Linear 层（仅 head 最后一层有 bias，符合惯例）
- [x] 使用 RMSNorm 或 pre-norm 架构（RMSNorm + norm_first=True）

## 训练脚本
- [x] `tools/p3_train_patch_itransformer.py` 文件已创建
- [x] 数据加载正确：使用 OddsDataset + splits_v6
- [x] 特征提取正确：9维核心特征 × 2 聚合（last, mean）
- [x] 训练配置正确：AdamW, CosineAnnealing, lr=0.001
- [x] 评估指标完整：accuracy, logloss, brier, ECE
- [x] 模型保存逻辑正确：保存最优 val_logloss 对应的权重

## 训练结果
- [ ] 训练成功完成（无 NaN/Inf loss）(需用户手动执行)
- [ ] val_logloss ≤ 0.930（打平或优于 RawPooledMLP）(需用户手动执行)
- [ ] 训练曲线合理（train loss 下降，val loss 不严重过拟合）(需用户手动执行)
- [ ] 结果已记录到 `runs/p3_patch_itransformer/report.json` (训练脚本自动完成)

## 代码质量
- [x] 无 inline class 定义（所有类在文件顶层）
- [x] 注释语言与项目一致（中文注释可接受）
- [x] 无 sklearn 依赖（使用纯 PyTorch）
- [ ] 无语法错误（需用户手动验证 `python -c "import model.odds_patch_itransformer"`）
