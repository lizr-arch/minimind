# 计划：解决 Python Runtime 并跑 P3 验证门禁

## 概述

当前 P3a/P3b 代码已完成，但 Python 运行时不可用（hermes-agent venv hang, WindowsApps stub error 9009）。本计划目标：找到或创建可用的 Python 环境，然后按门禁顺序跑 P3 验证。

## 当前状态分析

### 已完成
- P3a 模型 (`model/odds_patch_itransformer.py`): 81,731 params, 已验证 import
- P3b 模型 (`model/odds_patch_itransformer_v2.py`): ~104K params, 已修复 has_asian/has_ou
- P3a 训练脚本 (`tools/p3_train_patch_itransformer.py`)
- P3b 训练脚本 (`tools/p3b_train_patch_itransformer.py`)
- Bucket occupancy report (`tools/p3_bucket_occupancy_report.py`)
- Feature coverage report (`tools/p3b_feature_coverage_report.py`)

### 未完成（全部 BLOCKED）
- Phase 3: P3a/P3b forward smoke
- Phase 4: Feature coverage report
- Phase 5: Bucket occupancy report
- Phase 6: Tiny overfit

### 项目依赖 (`requirements.txt`)
```
datasets==3.6.0, numpy==1.26.4, scikit_learn==1.5.1, tiktoken==0.10.0,
transformers==4.57.6, einops==0.8.1, ...
# torch==2.6.0  (注释掉，需单独安装)
```

### 本机 Python 情况
- `C:\Users\User\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe` — hang
- `C:\Users\User\AppData\Local\Microsoft\WindowsApps\python.exe` — error 9009
- 无 `.venv`/`venv`/`pyproject.toml`/`environment.yml`

## 实施步骤

### Step 1: 搜索本机可用 Python

在常见目录搜索 `python.exe`，排除 hermes-agent 和 WindowsApps：
- `C:\Users\User` (深度 6)
- `C:\ProgramData` (深度 4)
- `C:\Python*` (深度 2)
- `D:\` (深度 4)

对每个候选运行 `--version` 测试（设置 10s 超时）。

### Step 2: 测试候选 Python

对 Step 1 找到的每个 python.exe：
1. `<path> --version` — 确认版本 ≥ 3.10
2. `<path> -c "import sys; print(sys.executable)"` — 确认不 hang
3. 记录结果

### Step 3: 创建项目 venv（如果需要）

**前提**: 找到一个可用的 Python 3.10/3.11

```powershell
cd D:\code\git\betmind\minimind
<found_python> -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

安装最小 smoke 依赖（CPU 版 torch）：
```powershell
.\.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\pip install numpy==1.26.4
```

验证：
```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Step 4: P3a/P3b Smoke

```powershell
.\.venv\Scripts\python.exe -c "import torch; from model.odds_patch_itransformer import OddsPatchITransformer; m = OddsPatchITransformer(); print(f'P3a params={sum(p.numel() for p in m.parameters()):,}'); y = m(torch.randn(4,10,9,2)); print(y.shape, torch.isnan(y).any().item(), torch.isinf(y).any().item())"

.\.venv\Scripts\python.exe -c "import torch; from model.odds_patch_itransformer_v2 import OddsPatchITransformerV2; m = OddsPatchITransformerV2(); print(f'P3b params={sum(p.numel() for p in m.parameters()):,}'); y = m(torch.randn(4,10,25,2)); print(y.shape, torch.isnan(y).any().item(), torch.isinf(y).any().item())"
```

### Step 5: 运行报告（不跑训练）

```powershell
.\.venv\Scripts\python.exe tools/p3b_feature_coverage_report.py --data data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl --ids data/odds_real/splits_v6/train_match_ids.txt --out runs/p3b_feature_coverage_report.json

.\.venv\Scripts\python.exe tools/p3_bucket_occupancy_report.py --data data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl --ids data/odds_real/splits_v6/train_match_ids.txt --out runs/p3_bucket_occupancy_report.json
```

## 停止条件

| 条件 | Verdict |
|------|---------|
| Python 仍然 hang | `BLOCKED_PYTHON_RUNTIME` |
| torch 无法 import | `BLOCKED_TORCH_MISSING` |
| P3a/P3b smoke 失败 | `BLOCKED_MODEL_SMOKE` |
| Asian/OU 特征大量全 0 | `BLOCKED_FEATURE_EXTRACTION` |
| minutes_before_kickoff 基本全 0 | `BLOCKED_TIME_BUCKETING` |
| 全部通过 | `READY_FOR_TINY_OVERFIT` |

## 不做的事

- 不修改模型代码
- 不跑 full train
- 不跑 tiny overfit（本计划只到报告阶段）
- 不安装不必要的依赖（sklearn, tqdm 等等 smoke 不需要的先不装）
