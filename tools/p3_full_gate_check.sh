#!/bin/bash
set -e
cd /d/code/git/betmind/minimind

echo "=== Phase 0: Python env ==="
python --version
python -c "import torch; print(f'torch={torch.__version__}'); print(f'cuda={torch.cuda.is_available()}')"

echo ""
echo "=== Phase 1: P3a smoke ==="
python -c "
import torch
from model.odds_patch_itransformer import OddsPatchITransformer
m = OddsPatchITransformer()
n = sum(p.numel() for p in m.parameters())
print(f'P3a params={n:,}')
assert n < 100000
x = torch.randn(4,10,9,2)
y = m(x)
assert y.shape == (4,3)
assert not torch.isnan(y).any()
assert not torch.isinf(y).any()
print(f'P3a shape={y.shape} nan=False inf=False')
print('P3a SMOKE PASS')
"

echo ""
echo "=== Phase 2: P3b smoke ==="
python -c "
import torch
from model.odds_patch_itransformer_v2 import OddsPatchITransformerV2
m = OddsPatchITransformerV2()
n = sum(p.numel() for p in m.parameters())
print(f'P3b params={n:,}')
assert n < 150000
x = torch.randn(4,10,25,2)
y = m(x)
assert y.shape == (4,3)
assert not torch.isnan(y).any()
assert not torch.isinf(y).any()
print(f'P3b shape={y.shape} nan=False inf=False')
print('P3b SMOKE PASS')
"

echo ""
echo "=== Phase 3: Feature coverage report ==="
python tools/p3b_feature_coverage_report.py \
  --data data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl \
  --ids data/odds_real/splits_v6/train_match_ids.txt \
  --out runs/p3b_feature_coverage_report.json

echo ""
echo "=== Phase 4: Bucket occupancy report ==="
python tools/p3_bucket_occupancy_report.py \
  --data data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl \
  --ids data/odds_real/splits_v6/train_match_ids.txt \
  --out runs/p3_bucket_occupancy_report.json

echo ""
echo "=== ALL GATES COMPLETE ==="
echo "Check output above for any WARN/BLOCKER lines."
