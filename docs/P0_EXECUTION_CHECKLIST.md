# P0 执行清单

## 前置条件
- [ ] `data/training/pipeline_export.jsonl` 已就位（35,330 matches, 865 MB, v4 format, 0 errors）
- [ ] 所有 P0 验收门禁已通过（格式/规模/质量/泄漏/联赛分布）

## Step 1: 生成 split

```bash
python tools/generate_p0_split.py \
    --data data/training/pipeline_export.jsonl \
    --out-dir data/training/splits_p0 \
    --train-ratio 0.70 --val-ratio 0.15 --seed 42
```

验收: `data/training/splits_p0/split_summary.json` 生成成功，无 overlap。

## Step 2: Smoke 验证

```bash
python tools/p0_smoke_verify.py \
    --data data/training/pipeline_export.jsonl \
    --split-dir data/training/splits_p0 \
    --epochs 2 --batch-size 64
```

验收: ALL CHECKS PASSED，loss 正常（< 100），无 NaN，无崩溃。

## Step 3: 全量训练 v5 + league embedding

```bash
python eval/eval_p1_5_verify.py \
    --data data/training/pipeline_export.jsonl \
    --split-dir data/training/splits_p0 \
    --epochs 50 --patience 10 --batch-size 64 \
    --out-dir runs/p0_v5_full
```

验收: 训练正常收敛，best checkpoint 保存成功。

## Step 4: 对比 v5 vs P1.18 v3

```bash
python eval/eval_p1_5_verify.py \
    --data data/training/pipeline_export.jsonl \
    --split-dir data/training/splits_p0 \
    --v3-ckpt runs/p1_18_convergence/model_best.pth \
    --skip-training \
    --out-dir runs/p0_compare
```

验收: 对比表输出 v5 vs v3 vs baseline delta。

## 期望结果

| 指标 | v3 (5K data) | v5 (35K data) | 预期 |
|---|---|---|---|
| accuracy vs baseline | +0.18pp | **+0.5~1.5pp** | 显著提升 |
| Bundesliga | tie baseline | **beat baseline** | league emb 生效 |
| SerieA | lose baseline | **at least tie** | league emb 生效 |
| Cross-league (可选) | -11.5pp | **-3~-6pp** | 数据量改善泛化 |

## 风险

- 如果 v5 accuracy 仍然 < baseline：说明 odds-only 信号已到天花板，需加入 team/player 特征
- 如果训练爆 OOM：batch-size 降为 32（RTX 3060 12GB 应该够 35-dim × 64 batch）
- 如果 loss 爆炸（类似 P1.18 的 62M）：检查 LEAGUE_MAP 中 unknown league 被映射到 index 0 是否正确
