# OddsMind P0.1 — Minimal Scaffold

## 1. What Was Added

| File | Purpose |
|---|---|
| `data/odds_fixtures/sample_odds_matches.jsonl` | 8 fixture matches with varied odds timelines, labels, leagues, bookmakers |
| `model/odds_encoder.py` | `OddsEventEncoder` — MLP projection from raw odds features (7-dim) to hidden_size |
| `model/odds_heads.py` | `EuroResultHead` (3-class) and `AsianResultHead` (3-class) |
| `model/model_oddsmind.py` | `OddsMindConfig`, `RMSNorm`, `OddsTransformerBlock`, `OddsMindModel` — full forward pipeline |
| `dataset/odds_dataset.py` | `OddsDataset` — JSONL loader with cutoff filtering and label mapping |
| `dataset/odds_collator.py` | `OddsCollator` — zero-padding collator producing features, attention_mask, labels |
| `trainer/train_odds_supervised.py` | Smoke training script — 1 epoch on fixture data, checkpoint save |
| `eval/eval_odds_smoke.py` | Smoke test — 7 automated checks + boundary confirmation |
| `infer/predict_odds_match.py` | Single-match inference — JSON in, JSON probability output |
| `docs/ODDSMIND_P0_1_SCAFFOLD.md` | This file |

## 2. MiniMind Modules Reused

None directly. All OddsMind code is in new files. However:

- The `OddsTransformerBlock` in `model/model_oddsmind.py` follows the same **pre-norm + MHA + FFN** pattern as `MiniMindBlock`. In P0.6 it can be swapped to directly import `MiniMindBlock` from `model.model_minimind`.
- `trainer/train_odds_supervised.py` imports `get_lr`, `Logger`, `setup_seed` from `trainer.trainer_utils` — these are pure utility functions and their reuse does not modify the original file.

## 3. MiniMind Modules NOT Reused (and why)

| Module | Reason |
|---|---|
| `model_minimind.py` | Not imported — OddsMind uses its own self-contained Transformer block to avoid coupling with the LM pipeline. P0.6 will add an optional swap. |
| `model_lora.py` | LoRA is irrelevant for classification training. May be relevant later. |
| `lm_dataset.py` | Text-based datasets with tokenization are entirely different from odds time-series data. |
| `train_pretrain.py` through `train_ppo.py` | All are text-generation training loops. `train_odds_supervised.py` is a separate script. |
| `eval_llm.py` | Conversational evaluation — irrelevant. |
| `scripts/*` | Chat/web UI — irrelevant. |
| `model/tokenizer.json` & `tokenizer_config.json` | OddsMind uses continuous features, not discrete token IDs. |

## 4. Why No Tokenizer / LM Head

- **Tokenizer**: OddsMind input is a continuous float time series `[T, 7]`, not text. There is no vocabulary to look up. The `OddsEventEncoder` plays the role of "embedding" — it projects raw features to hidden_size via learned linear transformations.
- **LM Head**: OddsMind is a **classifier**, not a generative language model. The output is a small fixed set of classes (3 for euro, 3 for asian), not a vocabulary of 6400+ tokens. Two small MLP heads replace the single `lm_head`.
- **Next-token prediction loss**: Replaced by `cross_entropy` on the pooled representation against ground-truth labels.

## 5. Current Input / Output Format

**Input** (per sample):
```json
{
  "match_id": "fixture_001",
  "odds_timeline": [
    {"minutes_before_kickoff": 10080, "euro_h": 2.30, "euro_d": 3.20, "euro_a": 3.10,
     "asian_line": 0.0, "upper_water": 0.90, "lower_water": 1.00}
  ],
  "label": {"euro_result": "home", "asian_result": "upper"}
}
```

Feature vector (7 dims): `[minutes_before_kickoff, euro_h, euro_d, euro_a, asian_line, upper_water, lower_water]`

**Model tensor shapes**:
- Input: `features [B, T, 7]`, `attention_mask [B, T]`
- Output: `euro_logits [B, 3]`, `asian_logits [B, 3]`
- Loss: `CE(euro_logits, euro_labels) + CE(asian_logits, asian_labels)`

## 6. Scaffold Status — NOT a Trained Model

This P0.1 release is a **functional skeleton**.  All code runs and produces plausible shapes, but:

- The model weights are randomly initialized (no real training has been done).
- Predictions have no statistical meaning yet.
- The default config (4 layers, hidden=256, ~1.9M params) is a smoke-test size, smaller than what a real OddsMind v0.1 would use.
- No hyperparameter tuning has been performed.

## 7. Next Phases (Recommended Order)

| Phase | Description |
|---|---|
| **P0.2** | Add random cutoff sample generation — generate multiple training samples per match by sampling cutoff ∈ {90, 60, 30} |
| **P0.3** | Add 5-class asian labels (upper_full_win / upper_half_win / push / upper_half_loss / upper_full_loss) |
| **P0.4** | Add time-split evaluation — train/val/test by kickoff_time, grouped by match_id |
| **P0.5** | Add LightGBM baseline for comparison |
| **P0.6** | Swap `OddsTransformerBlock` for `MiniMindBlock` from `model.model_minimind`, reuse RoPE |
| **P0.7** | Add self-supervised pre-training tasks (masked odds modeling, next-odds prediction) |
