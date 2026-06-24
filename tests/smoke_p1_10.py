"""P1.10 smoke test: ScoreHeadV2 forward, loss, and model integration."""
import torch, sys
sys.path.insert(0, '.')

from model.odds_heads import ScoreHeadV2
from model.model_oddsmind import OddsMindConfig, OddsMindModel

# Test 1: ScoreHeadV2 shapes
print('=== ScoreHeadV2 shape test ===')
head = ScoreHeadV2(hidden_size=512, dropout=0.1, bounded_log_rate=True)
x = torch.randn(4, 512)
out = head(x)
for k, v in out.items():
    print(f'  {k}: {v.shape}  range=[{v.min().item():.3f}, {v.max().item():.3f}]')

# Test 2: Model forward with ScoreHeadV2
print()
print('=== Model forward test (v2) ===')
cfg = OddsMindConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                     score_head_version='v2', score_bounded_log_rate=True)
model = OddsMindModel(cfg)
model.eval()
features = torch.randn(2, 10, 7)
mask = torch.ones(2, 10, dtype=torch.bool)
euro_lbl = torch.tensor([0, 1])
asian_lbl = torch.tensor([1, 2])
score_lbl = torch.tensor([[1.0, 2.0], [0.0, 3.0]])

with torch.no_grad():
    out = model(features, attention_mask=mask,
                euro_labels=euro_lbl, asian_labels=asian_lbl,
                score_labels=score_lbl, score_loss_type='poisson',
                score_loss_weight=0.0)
for k, v in out.items():
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            print(f'  {k}: {v.shape} val={v.item():.4f}')
        else:
            print(f'  {k}: {v.shape}')
    elif isinstance(v, dict):
        print(f'  {k}: {v}')
    else:
        print(f'  {k}: {v}')

# Test 3: Score loss standalone (simulate Phase 2)
print()
print('=== Score loss standalone (Phase 2 mode) ===')
out2 = model(features, attention_mask=mask,
             euro_labels=euro_lbl, asian_labels=asian_lbl,
             score_labels=score_lbl, score_loss_type='poisson',
             score_loss_weight=1.0)
print(f'  score_loss: {out2["score_loss"].item():.4f}')
print(f'  aux_losses: {out2.get("aux_losses", "N/A")}')
print(f'  score_raw range: [{out2["score_raw_min"]:.3f}, {out2["score_raw_max"]:.3f}]')
print(f'  score_log_rate range: [{out2["score_log_rate_min"]:.3f}, {out2["score_log_rate_max"]:.3f}]')
print(f'  score_preds mean: {out2["score_preds"].mean(0).tolist()}')

# Test 4: Check params
v2_params = sum(p.numel() for p in model.score_head.parameters())
total = sum(p.numel() for p in model.parameters())
print(f'  ScoreHeadV2 params: {v2_params:,} / total: {total:,}')

# Test 5: Verify tanh bounded property
print()
print('=== Bounded log-rate property ===')
big_head = ScoreHeadV2(hidden_size=512, bounded_log_rate=True, log_rate_bound=5.0)
big_x = torch.randn(100, 512) * 3.0  # large inputs
big_out = big_head(big_x)
raw = big_out['score_raw']
log_rate = big_out['score_log_rate']
print(f'  score_raw range: [{raw.min().item():.3f}, {raw.max().item():.3f}]')
print(f'  score_log_rate range: [{log_rate.min().item():.3f}, {log_rate.max().item():.3f}]')
assert log_rate.min() > -5.1 and log_rate.max() < 5.1, 'Bounded log-rate violated!'
print('  PASS: log_rate within [-5, 5]')

# Test 6: Hard clamp variant
print()
print('=== Hard clamp variant ===')
hard_head = ScoreHeadV2(hidden_size=512, bounded_log_rate=False, log_rate_bound=5.0)
hard_out = hard_head(big_x)
hard_log = hard_out['score_log_rate']
print(f'  hard_clamp log_rate range: [{hard_log.min().item():.3f}, {hard_log.max().item():.3f}]')
print(f'  num at boundary: {(hard_log.abs() > 4.99).sum().item()} / {hard_log.numel()}')

print()
print('ALL SMOKE TESTS PASSED')
