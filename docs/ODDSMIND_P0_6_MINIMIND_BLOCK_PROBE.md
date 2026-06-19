# OddsMind P0.6 — MiniMindBlock Reuse Probe

## 1. Audit Results

### MiniMind Block Location

| Component | File | Class |
|---|---|---|
| Config | `model/model_minimind.py:10` | `MiniMindConfig(PretrainedConfig)` |
| Attention | `model/model_minimind.py:91` | `Attention` (GQA, QK-norm, flash) |
| FeedForward | `model/model_minimind.py:136` | `FeedForward` (SwiGLU) |
| MoE FF | `model/model_minimind.py:148` | `MOEFeedForward` |
| RMSNorm | `model/model_minimind.py:50` | `RMSNorm` |
| RoPE | `model/model_minimind.py:62-84` | `precompute_freqs_cis`, `apply_rotary_pos_emb` |
| Transformer Block | `model/model_minimind.py:178` | `MiniMindBlock` |

### Block.forward() signature

```python
def forward(self, hidden_states, position_embeddings,
            past_key_value=None, use_cache=False, attention_mask=None):
```

- Input: `hidden_states [B, T, H]`
- **position_embeddings REQUIRED** — (cos, sin) tuple for RoPE
- attention_mask: `[B, T]` float where 0=mask, 1=keep
- **is_causal = True** hardcoded in Attention
- Returns: `(hidden_states, present_key_value)` tuple

### Reuse Assessment

| Aspect | Verdict |
|---|---|
| No token dependency | ✅ Block does not use token IDs |
| Input `[B,T,H]` | ✅ Compatible |
| RoPE position encoding | ⚠️ Adapter must generate embeddings |
| Causal mask | ⚠️ Must be patched to False for encoder mode |
| KV cache return | ⚠️ Must be discarded |
| Config dependency | ⚠️ Dummy MiniMindConfig needed |
| transformers dependency | ❌ Requires `transformers` package |

**Conclusion: ADAPTER REUSE IS FEASIBLE** but requires `transformers` installed.

## 2. Adapter Implementation

`model/oddsmind_minimind_adapter.py` — `MiniMindBlockAdapter`:

- Creates dummy `MiniMindConfig` matching OddsMind dimensions
- Builds `MiniMindBlock` instances
- **Patches `is_causal = False`** on all Attention modules
- Generates RoPE embeddings internally
- Discards KV cache
- Presents clean interface: `forward(x, key_padding_mask) -> x`

## 3. Backend Selection

```bash
--transformer-backend odds_native   # default, self-contained
--transformer-backend minimind      # requires transformers package
```

`odds_native` is always the default and works without any extra dependencies.

## 4. Current Status

- **Native backend**: ✅ Fully functional, all tests pass
- **MiniMind backend**: ⚠️ Code ready, but requires `transformers` package
  (not installed in current environment). Gracefully skipped in smoke tests.

## 5. Why No Tokenizer / Embedding / LM Head

The MiniMindBlock adapter only uses the Transformer blocks — no tokenizer,
no token embeddings, no LM head. These are entirely replaced by OddsMind's
`OddsEventEncoder`, pooling, and classification heads.

## 6. Zero MiniMind Files Modified

All MiniMind source files are imported read-only. Patching (`is_causal=False`)
happens on the adapter's own block instances, not on the original classes.

## 7. P0.6R Runtime Activation Result

| Check | Result |
|---|---|
| `transformers` installed | ✅ 5.12.1 |
| Adapter import | ✅ `MiniMindBlockAdapter` imports cleanly |
| minimind forward | ✅ euro_logits [2,3], asian_logits [2,5] |
| minimind loss | ✅ computed, > 0 |
| minimind masked forward | ✅ attention_mask working |
| minimind smoke train | ✅ 1 epoch, avg_loss=2.67, val acc reported |
| minimind inference | ✅ JSON probabilities, sums ≈ 1 |
| native regression | ✅ unchanged |
| tokenizer / LM Head used | ❌ Not used |
| MiniMind files modified | ❌ Zero modifications |

**Recommendation**: Keep `minimind` backend as an optional, experimental path.
Default stays `odds_native`.  Both backends are now runtime-verified.

## 8. Next Phase

**P0.7 — Odds Self-Supervised Pretraining Scaffold**
