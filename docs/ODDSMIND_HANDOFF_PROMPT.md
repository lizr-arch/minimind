# OddsMind 项目上下文 — 新对话接续提示词

你正在开发 **OddsMind**，一个基于 Transformer 的足球赔率预测模型，从 MiniMind 项目 fork 而来。

---

## 一、项目是什么（What）

**目标**：输入赛前赔率时间序列，预测三项——胜平负、亚盘上下、比分。

**架构**：
```
赔率时间序列 [T,7] → OddsEventEncoder → Transformer(34M) → pooled
                                                              ├── Euro Head    (3分类)
                                                              ├── Asian Head   (3分类, line-aware+legal mask)
                                                              └── Score Head   (回归, Poisson NLL, 两阶段训练)
```

**当前最佳 checkpoint**：
- Phase 1: `runs/v7b_line_fix/oddsmind_smoke.pth` — Euro 62.7%, Asian 86.8%
- Phase 2: `runs/phase2_score/oddsmind_with_score.pth` — +Score head, Asian 89.9%

---

## 二、为什么这样设计（Why）

### 2.1 为什么用 Transformer 而不是 LightGBM
用户要的是赔率变化过程建模——开盘→中间→临场，Transformer 学时序模式。lightgbm 只能看快照。

### 2.2 为什么 Asian head 要做 line-aware + legal mask
- **line-aware (V6/V7)**：把 `asian_line` 直接作为 head 输入，模型知道"这是半盘还是整数盘"
- **legal mask (V8)**：半盘（-1.50）物理上不可能走水，强制 push 概率为 0
- 否则模型学会"猜 push 最安全"（训练数据 58.8% 是 push）

### 2.3 为什么 Score head 要两阶段训练
Score head 的 Poisson NLL loss 数值远大于 CE loss（几万 vs ~1），一起训时会把 Euro/Asian 的梯度淹没。
**解决**：Phase 1 正常训 Euro+Asian，Phase 2 冻 backbone 只训 Score head。

### 2.4 为什么不用多庄家数据合并
不同庄家赔率差异小（1.58 vs 1.62），合并让模型困惑。共识特征（6 维 avg/median）在 tabular 模型上有用，但 Transformer 内部注入会导致过拟合（P1.6/P1.7 实验结论）。

### 2.5 为什么欧赔转隐含概率再取差值
赔率越小变动越敏感（1.10→1.12 和 5.0→5.5 含义不同）。转成概率后差值可比（P1.6）。

---

## 三、当前完成了什么（How — 进度）

### 模型
- [x] Transformer 34M（512 hidden, 16 layers）— Euro 62.7%
- [x] Asian head line-aware + legal mask — Asian 89.9%
- [x] Score head Poisson NLL, 两阶段训练 — collapse 已修复，MAE ~1.07
- [x] 欧赔/亚盘/比分三项输出

### 数据
- [x] 13,131 Bet365 场次，5 联赛 × 3 赛季含 Titan007 时序
- [x] 多庄家共识特征提取器
- [x] CSV→JSONL 导入器

### 数据格式规范
- [x] `docs/ODDSMIND_DATA_FORMAT_SPEC.md`（v2，含大小球字段）
- [x] `docs/ODDSMIND_DATA_DIR_STRUCTURE.md`

### 训练
- [x] LR warmup + gradient clipping
- [x] Label smoothing + dropout + EMA

---

## 四、正在进行（数据端等待）

| 待完成 | 阻塞原因 |
|---|---|
| 大小球 head（Over/Under） | 等用户提供新的训练数据（含 `over_under_line`/`over_water`/`under_water`） |
| 多线亚盘水位 | 等用户提供 `asian_handicap_lines` 和 `over_under_lines` 数组 |
| League embedding | 数据里已有 `league_id`，等大小球数据到位一起上 |

---

## 五、关键文件路径

```
模型:
  model/model_oddsmind.py        — OddsMindModel (forward, loss, legal mask)
  model/odds_heads.py            — Euro/Asian/Score heads
  model/odds_encoder.py          — OddsEventEncoder

训练:
  trainer/train_odds_supervised.py  — Phase 1 (euro+asian)
  trainer/train_phase2_score.py     — Phase 2 (score head solo)

评估:
  eval/eval_odds_model.py        — 输出 euro/asian/score metrics
  eval/odds_metrics.py           — accuracy/logloss/brier from probs/logits

数据:
  dataset/odds_dataset.py        — OddsDataset, label maps
  dataset/odds_collator.py       — padding collator
  dataset/importers/football_data_csv.py  — CSV importer
  dataset/importers/master_dataset.py     — master_dataset importer
  dataset/odds_bookmaker_features.py      — 多庄家共识特征

Checkpoint:
  runs/v7b_line_fix/oddsmind_smoke.pth        — Phase 1 best (Euro 62.7, Asian 86.8)
  runs/phase2_score/oddsmind_with_score.pth   — Phase 2 (+score head, Asian 89.9)

数据:
  data/odds_real/v5_b365.jsonl            — 13,131 Bet365 matches
  data/odds_real/splits_v5/               — train/val/test split
```

---

## 六、启动命令

```bash
# Phase 1: train euro+asian only
python trainer/train_odds_supervised.py \
  --data data/odds_real/v5_b365.jsonl \
  --epochs 30 --batch-size 64 --hidden-size 512 --num-layers 16 --num-heads 8 \
  --device cuda --cutoffs 0 --cutoff-mode exhaustive --asian-label-mode 3class \
  --train-match-ids data/odds_real/splits_v5/train_match_ids.txt \
  --val-match-ids data/odds_real/splits_v5/val_match_ids.txt \
  --eval-every-epoch --transformer-backend odds_native \
  --label-smoothing 0.1 --dropout 0.1 --ema-decay 0.999 \
  --score-loss-weight 0 --lr-warmup 500 --grad-clip 1.0 \
  --out-dir runs/v10

# Phase 2: freeze backbone, train score head
python trainer/train_phase2_score.py \
  --data data/odds_real/v5_b365.jsonl \
  --checkpoint runs/v10/oddsmind_smoke.pth \
  --train-ids data/odds_real/splits_v5/train_match_ids.txt \
  --hidden-size 512 --num-layers 16 --num-heads 8 --device cuda \
  --epochs 20 --out-dir runs/phase2_v10

# Eval
python eval/eval_odds_model.py \
  --data data/odds_real/v5_b365.jsonl \
  --model runs/phase2_v10/oddsmind_with_score.pth \
  --split-match-ids data/odds_real/splits_v5/test_match_ids.txt \
  --cutoffs 0 --cutoff-mode exhaustive --asian-label-mode 3class \
  --hidden-size 512 --num-layers 16 --num-heads 8 --device cuda

# 单场预测
python infer/predict_odds_match.py \
  --model runs/phase2_v10/oddsmind_with_score.pth \
  --input <json_file> \
  --hidden-size 512 --num-layers 16 --num-heads 8 --device cuda --asian-label-mode 3class
```

---

## 七、不要做的事

- 不要修改 MiniMind 原始文件（`model/model_minimind.py` 等）
- 不要用 tokenizer/vocab/LM Head
- 不要在 Transformer 内部注入共识特征（P1.6 已证明失败）
- 不要多庄家数据直接合并（P1.5 已证明失败）
- Score head 不要和 Euro/Asian 一起训 weight>0

---

## 八、新对话开始时的检查清单

1. `git status` — 确认在 `feature/oddsmind-p0.1-scaffold`
2. `python -c "import torch; print(torch.cuda.is_available())"` — 确认 GPU
3. 确认 `data/odds_real/v5_b365.jsonl` 存在
4. 检查用户是否已提供大小球/多线亚盘的新数据
