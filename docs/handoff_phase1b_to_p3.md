# OddsMind 项目交接文档 — Phase 1b → P3 OddsPatch-iTransformer

**日期：** 2026-06-27
**当前状态：** P0-P2 完成，P3 待执行
**项目路径：** `D:\code\git\betmind\minimind`

---

## 一、我们在做什么（Why）

用 Transformer 从赛前赔率时间序列预测足球比赛 1X2 结果。经历两个阶段后，方向从"放弃 Transformer"反转为"升级为赔率专用 Transformer"。

**核心洞察：** 失败的不是 Transformer，是 event-token 设计。把每个时间点的 33 维异质特征挤进一个 token（欧赔+亚盘+大小球+时间+缺失标记），Transformer attention 无法区分该关注时间变化还是特征交互。

**论文支撑：** iTransformer (ICLR2024) — 变量维度做 attention；PatchTST (ICLR2023) — channel-independent + patching + 自监督预训练优于纯监督。

## 二、已完成的实验（What）

| 阶段 | 内容 | 核心发现 |
|---|---|---|
| Phase 1a | 静态 MLP baseline | 18维静态特征 MLP, val_logloss=0.943 |
| Phase 1b 初期 | Event-token Transformer | 完全不学习 (logloss=1.11) |
| Phase 1b 中期 | 均值池化 MLP | 仅9维欧赔, val_logloss=0.930 — 当前最优 |
| Phase 1b 后期 | iTransformer [B,F,T] | train_acc=0.89 但严重过拟合 val_logloss=1.03 |
| P0 | 防泄漏验证 | match_id 无重叠 ✅, shuffled label略超标(0.506 vs 要求<0.5) |
| P1 | Time bucketing | 128 padding → 10固定时间桶, 270维特征, MLP logloss=0.987 |
| P2 | PLE数值embedding | 脚本已写, 待执行 |

### 当前最优模型

| 模型 | val_logloss | val_acc | 参数 |
|---|---|---|---|
| RawPooledMLP (9维欧赔均值池化) | 0.930 | 0.579 | 12K |
| 纯欧赔 baseline (Pinnacle去水) | 0.943 | 0.573 | — |
| Phase 1a (18维静态MLP) | 0.944 | 0.565 | 4K |

## 三、下一步怎么执行（How）

### OddsPatch-iTransformer 架构

```
[B, T_raw, F=33] 原始赔率事件
  ↓
Time bucketing: 10个固定时间桶
  opening / 7d / 3d / 1d / 12h / 6h / 3h / 1h / 30m / closing
  ↓
[B, 10 buckets, F features, AGG stats]
  ↓
PLE numerical embedding: 核心特征 scalar → embedding vector
  ↓
Channel-independent temporal encoder: 每个特征独立编码时间变化
  ↓
Cross-market Transformer: 特征间 attention (欧赔↔亚盘↔大小球)
  ↓
CLS → prediction heads (1X2 + Asian + OU)
```

### 待执行步骤（按优先级）

**P2（未完）：** 运行 `tools/p2_ple_embedding.py`，对比 PLE vs RAW bucketed MLP。如果 PLE 不优于 RAW，可跳过直接进入 P3。

**P3（主攻）：** 实现 OddsPatch-iTransformer。
- 输入：bucketed features [B, 10, F_core, AGG] → flatten per-bucket → [B, 10, D_bucket]
- Channel-temporal encoder: 每个特征独立过 MLP/LSTM
- Cross-market Transformer: 2层, d=64, heads=4
- 目标：val_logloss ≤ 0.930（打平 RawPooledMLP）

**P4：** SAM optimizer（结构稳定后）
**P5：** Masked pretraining on 38K unlabeled matches

## 四、关键文件

| 文件 | 用途 |
|---|---|
| `tools/p1_time_bucketing.py` | Time bucketing 实现 + 全量1550维测试 |
| `tools/p1_core_bucket_test.py` | 核心9特征×3聚合=270维测试 |
| `tools/p2_ple_embedding.py` | PLE vs RAW bucketed 对比（待执行） |
| `eval/odds_baselines_v2.py` | Phase1Baseline (联合欧亚分布) |
| `trainer/train_phase1b_event.py` | 事件驱动训练脚本 (含iTransformer) |
| `model/odds_encoder.py` | 编码器 (已改为bias=False单Linear) |
| `model/model_oddsmind.py` | 主模型 (含asian_line_feature_index配置) |
| `dataset/odds_collator_v6_event.py` | v6_event collator (含lead-lag特征) |
| `docs/prompt_literature_survey_transformers.md` | 文献调研提示词 |

## 五、新对话启动提示词

```
你是 OddsMind 项目的 AI 开发助手。

【项目背景】
OddsMind 用 Transformer 从赛前赔率时间序列预测足球比赛结果(1X2)。
项目路径: D:\code\git\betmind\minimind
数据: D:\code\git\betmind\minimind\data\odds_real\titan007_pure_v4_market_semantics_v2.jsonl
切分: D:\code\git\betmind\minimind\data\odds_real\splits_v6/

【当前状态】
已完成 P0(防泄漏验证, PASS_WITH_WARNING)、P1(time bucketing)、
P2(PLE script written but not run)。当前最优: RawPooledMLP val_logloss=0.930。

【下一步】
运行 tools/p2_ple_embedding.py 完成 PLE vs RAW 对比。
如果 PLE 不优于 RAW，直接进入 P3: 实现 OddsPatch-iTransformer。
架构: time bucketing → PLE → channel-independent temporal encoder 
→ cross-market Transformer → CLS → prediction heads。
目标: val_logloss ≤ 0.930。

【关键约束】
- 数据约8K训练比赛+30K无标签赔率
- Transformer参数量控制在100K以内(避免过拟合)
- 使用d_model=64, layers=1-2, norm_first=True
- 特征优先使用9维核心欧赔(已验证最优)
- 新代码放在 tools/ 或 model/ 目录下
- 运行测试前确保 sys.path.insert(0, '..') 
```

## 六、已知坑

1. **编码器 bias=True** 导致 padding 污染（已修复为 bias=False）
2. **LayerNorm** 让所有事件表示趋同 cos sim=0.97（已移除）
3. **inline class 定义** 在 `python -c` 中会缩进报错（用文件脚本）
4. **sklearn 未安装**（用纯 PyTorch 实现 logistic regression）
5. **网络不可用**（文献搜索用 ar5iv.org 而非 arxiv.org）

## 七、Python 环境

```
Python: 3.11.15
venv: C:\Users\User\AppData\Local\hermes\hermes-agent\venv
项目: D:\code\git\betmind\minimind (bash中用 /d/code/git/betmind/minimind)
GPU: CUDA available, 1 device

运行脚本:
  cd /d/code/git/betmind/minimind && python tools/xxx.py

多行脚本用 heredoc:
  cd /d/code/git/betmind/minimind && python << 'PYEOF'
  ... code ...
  PYEOF

关键包: torch (CUDA), numpy (no sklearn, no pandas verified)
```

## 八、未解决问题

1. tools/p2_ple_embedding.py 运行 exit=1 无输出 — 新对话需 debug。可能原因：
   - 向量化 PLE encode 逻辑错误导致运行时崩溃
   - 建议先跑简化版：只用1个特征、8个bin、100个样本验证

2. P0.1R (3-seed shuffled robustness) 未执行 — 新对话可并行补做
