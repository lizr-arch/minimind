# Phase 1b Step 3 — 事件驱动模型训练 规格书

> 此文档即提示词，可直接发给实现者。

---

## 项目归属

| 项 | 值 |
|---|---|
| **项目类型** | 大模型训练项目 |
| **开发路径** | `D:\code\git\betmind\minimind` |
| **验收路径** | `D:\code\git\betmind\minimind`（本地跑训练 + 评估） |
| **依赖** | Step 1（v6_event schema）+ Step 2（OddsEventCollator）已验收 |
| **前提** | 需先修复 2 个兼容性问题（见下文"前提修复"） |

---

## 前提修复（训练前必须完成）

### 修复 1：`OddsMindConfig` 添加 v6_event

```python
# model/model_oddsmind.py, __post_init__
schema_dims = {"v1": 10, "v2": 13, "v3": 13, "v4": 32, "v5": 35, "v6_event": 33}
```

### 修复 2：`closing_line` 索引可配置

当前 `features[..., 4:5]` 硬编码 asian_line 在索引 4。v6_event 中 asian_line 在索引 3。

```python
# model/model_oddsmind.py, OddsMindConfig 新增字段
asian_line_feature_index: int = 4  # v1-v5 default, v6_event = 3
```

在 forward 中将 `4:5` 替换为 `self.config.asian_line_feature_index`。

---

## 任务

创建 `trainer/train_phase1b_event.py`，使用事件驱动 Transformer 训练 EuroResult + AsianResult 双任务模型。

## 边界

**在范围内：** 训练脚本、数据加载、训练循环、双轨评估、消融开关、checkpoint、baseline 对比

**不在范围内：** 架构改动、超参搜索、多庄家融合

## 输入

```python
data = "data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl"
train_ids = "data/odds_real/splits_v6/train_match_ids.txt"
val_ids   = "data/odds_real/splits_v6/val_match_ids.txt"
```

## 功能点

### F1：数据加载
- Pinnacle > Bet365 优先级去重
- `OddsDataset(feature_schema_version="v6_event", max_seq_len=128)`
- `build_pinnacle_index(ds.samples)` → `OddsEventCollator(pinnacle_index=idx)`
- 返回 train_loader, val_loader

### F2：模型配置
```python
config = OddsMindConfig(
    feature_schema_version="v6_event",  # → feature_dim=33
    asian_line_feature_index=3,          # v6_event: asian_line at idx 3
    max_seq_len=128,
    hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
    pooling_mode="mean",
    asian_num_classes=5,
)
```

### F3：训练循环
```python
loss = CE(euro_logits, euro_labels) + CE(asian_logits, asian_labels)
       + beta * KL(euro_probs, market_prior)
```
market_prior = 每个样本最后有效欧赔事件的去水概率（从 raw_timeline 提取）。

### F4：双轨评估
- Track 1: KL(model_probs, market_prior)
- Track 2: accuracy, logloss, Brier, ECE
- 按事件数分组、按联赛分组

### F5：消融 CLI flags
```python
--use-leadlag (default True)    # False → collator 中 lead-lag 特征置零
--use-alignment (default True)  # False → alignment 特征置零
--use-ou (default True)         # False → dataset 层 OU 特征置零
```

### F6：Baseline 对比
对 val 每个样本计算 Pinnacle 纯欧赔去水的 accuracy/logloss/Brier/ECE → 输出 delta

## 输出

```
runs/phase1b_event/
  best_model.pth
  report.json
```

## 测试内容

| 测试 | 验证 |
|------|------|
| T1 | DataLoader 产出 `[B, T_max, 33]` features |
| T2 | 单 batch 前向，euro_logits `[B,3]` |
| T3 | 单 epoch 训练 loss < 1.5 |
| T4 | 消融 flags 关闭后对应特征全为 0 |
| T5 | Checkpoint 保存→加载→logloss 一致 |

## 验收标准

- [ ] T1-T5 通过
- [ ] val logloss ≤ Phase 1a baseline (0.943)
- [ ] Track 1 KL ≤ 0.015
- [ ] Lead-lag 开启 ≥ 关闭
- [ ] 50 epochs 内收敛

## 训练命令

```bash
python trainer/train_phase1b_event.py \
    --data data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl \
    --train-ids data/odds_real/splits_v6/train_match_ids.txt \
    --val-ids data/odds_real/splits_v6/val_match_ids.txt \
    --epochs 50 --batch-size 64 --lr 0.001 --beta 0.1 \
    --max-seq-len 128 --hidden-size 256 --num-layers 4 --num-heads 8 \
    --use-leadlag --use-alignment --use-ou \
    --device cuda --out-dir runs/phase1b_event
```
