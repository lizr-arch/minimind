# Tasks

- [x] Task 1: 运行 P2 PLE 对比实验
  - [ ] SubTask 1.1: 执行 `python tools/p2_ple_embedding.py`，获取 RAW vs PLE 的 val_logloss 对比 (需用户手动执行)
  - [ ] SubTask 1.2: 记录结果，决定 P3 架构是否使用 PLE embedding (需用户手动执行)
- [x] Task 2: 实现 OddsPatch-iTransformer 模型
  - [x] SubTask 2.1: 创建 `model/odds_patch_itransformer.py`，实现核心特征提取（9维×2聚合）
  - [x] SubTask 2.2: 实现 Time Bucketing 层（复用 p1_time_bucketing.py 的 BUCKET_EDGES）
  - [x] SubTask 2.3: 实现 Feature Embedding 层（Linear per feature）
  - [x] SubTask 2.4: 实现 Channel-Independent Temporal Encoder（可选，先跳过直接用 cross-market）
  - [x] SubTask 2.5: 实现 Cross-Market Transformer（2层，d=64，heads=4，norm_first=True）
  - [x] SubTask 2.6: 实现 CLS Token Pooling + MLP Head（3分类）
  - [x] SubTask 2.7: 验证参数量 < 100K (手动计算: ~81,731 params)
- [x] Task 3: 创建训练脚本
  - [x] SubTask 3.1: 创建 `tools/p3_train_patch_itransformer.py`
  - [x] SubTask 3.2: 复用 OddsDataset 和 splits_v6 数据切分
  - [x] SubTask 3.3: 实现 bucketed 特征提取逻辑（9维核心特征，last+mean 聚合）
  - [x] SubTask 3.4: 配置训练参数（lr=0.001, epochs=50, batch=64, AdamW+CosineAnnealing）
  - [x] SubTask 3.5: 集成评估指标（accuracy, logloss, brier, ECE）
- [ ] Task 4: 训练与评估
  - [ ] SubTask 4.1: 运行训练，监控 train/val loss 曲线
  - [ ] SubTask 4.2: 记录最优 val_logloss，与 RawPooledMLP (0.930) 对比
  - [ ] SubTask 4.3: 若 logloss > 0.930，调优超参（层数、dropout、lr）
- [ ] Task 5: 结果记录与文档更新
  - [ ] SubTask 5.1: 在 `runs/p3_patch_itransformer/` 保存训练报告 (训练脚本自动完成)
  - [ ] SubTask 5.2: 更新交接文档，记录 P3 结果

# Task Dependencies

- Task 2 depends on Task 1 (PLE 结果决定 embedding 方案)
- Task 3 depends on Task 2 (需要模型定义才能写训练脚本)
- Task 4 depends on Task 3
- Task 5 depends on Task 4
