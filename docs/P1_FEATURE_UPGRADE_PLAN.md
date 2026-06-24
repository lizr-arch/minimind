# P1: Add League/Bookmaker Embedding + Enhanced Input Features

**Target**: OddsMind v1.0 → v1.1 feature upgrade
**Executor**: LLM agent (read this document, implement step by step)
**Reviewer**: Human (verify each acceptance gate)

---

## Context

OddsMind is a Transformer Encoder that predicts football match outcomes (1X2: home/draw/away) from pre-match odds timelines. The current v1.0 model (3.33M params, d256/l4/h8) uses 13 raw odds features and beats the market baseline by +0.18pp accuracy across 3 seeds.

The model has a known weakness: it cannot distinguish between leagues. P2.1 B showed that leave-one-league-out CV fails on all 5 leagues — the model learns league-specific patterns but has no explicit league awareness. Adding league/bookmaker embeddings and richer input features should improve both same-league accuracy and cross-league generalization.

## Current State (what you start with)

- `model/model_oddsmind.py` — `OddsMindConfig`, `OddsMindModel`
- `model/odds_encoder.py` — `OddsEventEncoder` (13-dim → 256-dim MLP with missing_embed)
- `dataset/odds_dataset.py` — `_event_to_features()` returns 13 features, `_event_to_missing_mask()` returns 13-dim mask
- `dataset/odds_collator.py` — `OddsCollator` pads features/missing_mask
- `dataset/odds_labels.py` — label encoding/decoding
- `eval/domain_transfer/domain_taxonomy.py` — league taxonomy (5 trained leagues)
- Feature schema v3: `[minutes, euro_h/d/a, asian_line, upper/lower_water, ou_line, ou_over/under, has_euro/asian/ou]`

## What You Must NOT Change

- Do NOT modify the Transformer backbone (d_model, layers, heads, FFN)
- Do NOT change pooling_mode (keep "mean")
- Do NOT introduce tokenizer or LM Head
- Do NOT change the training loss (CrossEntropy on Euro 1X2)
- Do NOT change the data split logic
- Do NOT delete any existing code paths — all new features must be optional/config-gated

---

## Task 1: Add League Embedding

### 1.1 Define league mapping

In `dataset/odds_dataset.py`, add after `BOOKMAKER_MAP`:

```python
# P1 league embedding
LEAGUE_MAP = {k: i for i, k in enumerate(sorted([
    "EPL", "LaLiga", "Bundesliga", "Ligue1", "SerieA"
]))}
LEAGUE_COUNT = len(LEAGUE_MAP)
```

Export `LEAGUE_MAP` and `LEAGUE_COUNT` from the module.

### 1.2 Add league_id to dataset output

In `_build_features_and_labels()`, add `league_id` encoding:

```python
league_id = LEAGUE_MAP.get(sample.get("league_id", ""), 0)
```

Add `"league_id": league_id` to the return dict.

### 1.3 Add league_id to collator

In `OddsCollator.__call__()`, add:

```python
league_ids = torch.zeros(batch_size, dtype=torch.long)
# in the loop:
league_ids[i] = item["league_id"]
```

Add `"league_ids": league_ids` to the return dict.

### 1.4 Add league embedding to model

In `OddsMindConfig`:
```python
num_leagues: int = 5  # change from 0 to 5
```

In `OddsMindModel.__init__()`:
```python
if self.config.num_leagues > 0:
    self.league_embed = nn.Sequential(
        nn.Embedding(self.config.num_leagues, self.config.hidden_size // 4),
        nn.Linear(self.config.hidden_size // 4, self.config.hidden_size, bias=False),
    )
```

In `OddsMindModel.forward()`, add `league_ids` parameter and inject after pooling:
```python
league_ids: Optional[torch.Tensor] = None,  # [B] P1
# ... after bookmaker embedding ...
if league_ids is not None and self.config.num_leagues > 0:
    lg_emb = self.league_embed(league_ids)
    pooled = pooled + lg_emb
```

### 1.5 Pass league_ids through training

In `trainer/train_odds_supervised.py`, add:
```python
league_ids = batch["league_ids"].to(args.device)
```
And pass to `model(..., league_ids=league_ids)`.

### Acceptance Gate 1
```
python -c "
import torch
from model.model_oddsmind import OddsMindConfig, OddsMindModel
cfg = OddsMindConfig(num_leagues=5)
model = OddsMindModel(cfg)
x = torch.randn(2, 10, 13)
am = torch.ones(2, 10, dtype=torch.bool)
lid = torch.tensor([0, 2])
out = model(x, attention_mask=am, league_ids=lid)
assert out['euro_logits'].shape == (2, 3), f'Bad shape: {out[\"euro_logits\"].shape}'
assert torch.isfinite(out['euro_logits']).all()
print('PASS: League embedding forward pass works')
"
```

---

## Task 2: Add Implied Probabilities as Input Features

The market-implied probabilities (1/odds normalized) are a strong signal. Adding them as explicit input features lets the model focus on learning the RESIDUAL between market consensus and true outcome probability.

### 2.1 Add implied probs computation

In `dataset/odds_dataset.py`, add a helper:

```python
def _event_implied_probs(event: dict) -> tuple:
    """Compute 1/odds normalized implied probabilities from an event.
    Returns (p_home, p_draw, p_away) or (0,0,0) if odds missing.
    """
    h = event.get("euro_h", 0)
    d = event.get("euro_d", 0)
    a = event.get("euro_a", 0)
    if h <= 1.0 or d <= 1.0 or a <= 1.0:
        return (0.0, 0.0, 0.0)
    ih, id_, ia = 1.0/h, 1.0/d, 1.0/a
    total = ih + id_ + ia
    if total <= 0:
        return (0.0, 0.0, 0.0)
    return (ih/total, id_/total, ia/total)
```

### 2.2 Extend feature vector

Add 3 new features at the END of the feature vector (after has_over_under):

```
index 13: implied_home_prob
index 14: implied_draw_prob
index 15: implied_away_prob
```

This changes `feature_dim` from 13 to **16**.

### 2.3 Update _event_to_features v4

Add a new schema version "v4" that includes implied probs:

```python
if schema_version in ("v2", "v3", "v4"):
    has_euro = ...  # existing
    has_asian = ...
    has_ou = ...
    base_features = base + [has_euro, has_asian, has_ou]
    
if schema_version == "v4":
    imp_h, imp_d, imp_a = _event_implied_probs(event)
    return base_features + [imp_h, imp_d, imp_a]

return base_features
```

### 2.4 Update _event_to_missing_mask for v4

Implied probs are derived (always present if euro odds present):
```python
if feature_schema_version == "v4":
    has_imp = (1.0 if has_euro else 0.0)
    mask = mask + [has_imp, has_imp, has_imp]
```

### 2.5 Update config default

In `OddsMindConfig`:
```python
feature_dim: int = 16  # was 13
feature_schema_version: str = "v4"
```

Wait — do NOT change the default. Keep "v3" as default and add v4 as opt-in. The config `feature_dim` field is auto-set from the schema, so it should stay dynamic. Only add the v4 code paths.

### Acceptance Gate 2
```
python -c "
from dataset.odds_dataset import _event_to_features, _event_to_missing_mask
event = {'minutes_before_kickoff': 90, 'euro_h': 2.0, 'euro_d': 3.2, 'euro_a': 4.0}
feats = _event_to_features(event, schema_version='v4')
mask = _event_to_missing_mask(event)
print(f'v4 features ({len(feats)}): {feats}')
print(f'mask ({len(mask)}): {mask}')
# Verify implied probs sum to ~1
imp_sum = feats[13] + feats[14] + feats[15]
assert abs(imp_sum - 1.0) < 0.01, f'Implied probs do not sum to 1: {imp_sum}'
# Verify mask for implied probs (derived, so always present if euro present)
# euro present -> has_euro=1 -> implied mask=1
assert mask[13] == 1.0 and mask[14] == 1.0 and mask[15] == 1.0
print('PASS: v4 schema works correctly')
"
```

---

## Task 3: Add Odds Change Rate Features

The rate at which odds change is a strong signal for market conviction. A sudden movement in the last hour before kickoff means something different from a gradual drift over a week.

### 3.1 Add change rate computation

In `dataset/odds_dataset.py`, add:

```python
def _compute_odds_change_rate(timeline: list, current_idx: int, key: str, lookback: int = 1) -> float:
    """Compute rate of change for an odds feature.
    
    Returns (current_value - previous_value) / previous_value,
    or 0.0 if previous value is missing or invalid.
    """
    if current_idx < lookback:
        return 0.0
    prev = timeline[current_idx - lookback].get(key, 0)
    curr = timeline[current_idx].get(key, 0)
    if prev <= 0 or curr <= 0:
        return 0.0
    return (curr - prev) / prev
```

### 3.2 Extend to v4 (continued from Task 2)

Add 3 more features after implied probs:

```
index 16: euro_h_change_rate  
index 17: euro_d_change_rate
index 18: euro_a_change_rate
```

Total v4 features: 19.

### 3.3 Update _build_features_and_labels

When schema is v4, compute change rates for each event in the sorted timeline. Since `_event_to_features` only sees one event at a time, the change rate must be computed at the `_build_features_and_labels` level:

```python
# After building the sorted_timeline and features list:
if feature_schema_version == "v4":
    for i, e in enumerate(sorted_timeline):
        cr_h = _compute_odds_change_rate(sorted_timeline, i, "euro_h")
        cr_d = _compute_odds_change_rate(sorted_timeline, i, "euro_d")
        cr_a = _compute_odds_change_rate(sorted_timeline, i, "euro_a")
        # Append to features[i]
        features[i] = torch.cat([features[i], torch.tensor([cr_h, cr_d, cr_a])])
```

Wait — this is messy because `features` is already a tensor at this point. A cleaner approach: compute all features in a list first, then convert to tensor.

Refactor `_build_features_and_labels` to build features as a list of lists, then convert:

```python
feature_rows = []
missing_rows = []
for i, e in enumerate(sorted_timeline):
    row = _event_to_features(e, schema_version=feature_schema_version)
    mrow = _event_to_missing_mask(e) if feature_schema_version == "v3" or "v4" else None
    
    if feature_schema_version == "v4":
        cr_h = _compute_odds_change_rate(sorted_timeline, i, "euro_h")
        cr_d = _compute_odds_change_rate(sorted_timeline, i, "euro_d")
        cr_a = _compute_odds_change_rate(sorted_timeline, i, "euro_a")
        row = row + [cr_h, cr_d, cr_a]
        if mrow is not None:
            has_imp = (1.0 if _event_has_euro(e) else 0.0)
            mrow = mrow + [has_imp, has_imp, has_imp, has_imp, has_imp, has_imp]
    
    feature_rows.append(row)
    if mrow is not None:
        missing_rows.append(mrow)

features = torch.tensor(feature_rows, dtype=torch.float32)
```

Actually, this is getting complex. Let me simplify: for Task 3, compute the change rates at the `_build_features_and_labels` level and append them after the v4 implied probs. Keep the existing flow intact for v1/v2/v3.

### Acceptance Gate 3
```
python -c "
from dataset.odds_dataset import _compute_odds_change_rate
timeline = [
    {'minutes_before_kickoff': 120, 'euro_h': 2.0},
    {'minutes_before_kickoff': 60,  'euro_h': 2.2},
]
# First event: no previous, should be 0
assert _compute_odds_change_rate(timeline, 0, 'euro_h') == 0.0
# Second event: (2.2 - 2.0) / 2.0 = 0.1
assert abs(_compute_odds_change_rate(timeline, 1, 'euro_h') - 0.1) < 0.001
print('PASS: Change rate computation correct')
"
```

---

## Task 4: Integration Test

After completing all 3 tasks, run the existing smoke test to verify backward compatibility:

```bash
python tests/smoke_p1_16.py
# Expected: 54 passed, 0 failed
```

Then run a forward pass with the full v4 schema to verify all components integrate:

```python
import torch
from model.model_oddsmind import OddsMindConfig, OddsMindModel
from dataset.odds_dataset import OddsDataset
from dataset.odds_collator import OddsCollator
from torch.utils.data import DataLoader

# Use v4 schema
ds = OddsDataset(
    "data/odds_fixtures/sample_odds_matches_5class.jsonl",
    max_seq_len=64,
    feature_schema_version="v4",
    asian_label_mode="5class",
)
collator = OddsCollator()
loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collator)
batch = next(iter(loader))

cfg = OddsMindConfig(num_leagues=5, feature_dim=19)
model = OddsMindModel(cfg)
model.eval()

with torch.no_grad():
    out = model(
        batch["features"],
        attention_mask=batch["attention_mask"],
        league_ids=batch.get("league_ids"),
        missing_mask=batch.get("missing_mask"),
    )

print(f"Features shape: {batch['features'].shape}")  # expected: [B, T, 19]
print(f"Missing mask shape: {batch.get('missing_mask', torch.tensor([])).shape}")  # [B, T, 19]
print(f"Euro logits: {out['euro_logits'].shape}")  # [B, 3]
print(f"League IDs: {batch.get('league_ids')}")
assert out['euro_logits'].shape == (4, 3)
assert torch.isfinite(out['euro_logits']).all()
print("PASS: Full v4 + league embedding integration works")
```

---

## Summary of Changes

| File | Change |
|---|---|
| `dataset/odds_dataset.py` | Add LEAGUE_MAP, _event_implied_probs, _compute_odds_change_rate, v4 schema support, league_id to output |
| `dataset/odds_collator.py` | Add league_ids to batch |
| `model/model_oddsmind.py` | Add num_leagues config, league_embed module, league_ids param in forward |
| `trainer/train_odds_supervised.py` | Pass league_ids from batch to model |

## Key Constraints

- v4 schema is **opt-in** — `feature_schema_version="v4"`. v1/v2/v3 unchanged.
- League embedding is **opt-in** — `num_leagues=5`. Default was 0 (disabled).
- All existing smoke tests must pass unchanged.
- No architecture changes to Transformer backbone.
- No changes to pooling, loss, or evaluation metrics.

---

*This document is designed for LLM execution. Each task has an acceptance gate that must pass before moving to the next task. The reviewer will verify each gate against the actual code.*
