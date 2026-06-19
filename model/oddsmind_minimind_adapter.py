"""
OddsMind → MiniMind Block Adapter (P0.6)

Wraps MiniMind's native Transformer blocks (MiniMindBlock) for use
inside the OddsMind model.  This adapter:

  - Constructs a dummy MiniMindConfig with matching dimensions.
  - Patches `is_causal = False` on Attention modules (encoder mode).
  - Generates RoPE position embeddings internally.
  - Discards KV cache outputs.
  - Presents a clean encoder-only interface: (x, key_padding_mask) -> x.

No MiniMind source files are modified.  All patching is done on the
instantiated adapter's own copies of the blocks.
"""

import math
from typing import Optional

import torch
from torch import nn

# Import MiniMind internals (read-only, no modifications)
from model.model_minimind import (
    MiniMindConfig,
    MiniMindBlock,
    precompute_freqs_cis,
)


class MiniMindBlockAdapter(nn.Module):
    """
    Adapter that wraps a stack of MiniMindBlock instances for OddsMind.

    Interface:
        forward(x, key_padding_mask=None) -> x
        where x: [B, T, H], key_padding_mask: [B, T] bool (True = PAD)
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        num_kv_heads: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        use_moe: bool = False,
    ):
        super().__init__()

        head_dim = hidden_size // num_heads
        intermediate_size = int(math.ceil(hidden_size * math.pi / 64) * 64)

        # Build a MiniMindConfig that matches OddsMind dimensions.
        # vocab_size is a dummy — never used by the blocks.
        self.config = MiniMindConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_seq_len,
            dropout=dropout,
            use_moe=use_moe,
            vocab_size=100,       # dummy, not used by blocks
            flash_attn=True,
        )

        # Build blocks
        self.layers = nn.ModuleList([
            MiniMindBlock(layer_id=i, config=self.config)
            for i in range(num_layers)
        ])

        # Patch Attention modules to disable causal masking (encoder mode)
        for block in self.layers:
            block.self_attn.is_causal = False

        # Precompute RoPE frequencies (same formula as MiniMind)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=head_dim,
            end=max_seq_len,
            rope_base=1e6,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.max_seq_len = max_seq_len

    def forward(
        self,
        x: torch.Tensor,                          # [B, T, H]
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, T] bool, True=PAD
    ) -> torch.Tensor:
        """
        Args:
            x: input hidden states.
            key_padding_mask: True where position is PADDED (inverted from
                              OddsMind attention_mask which is True=valid).

        Returns:
            x: [B, T, H]
        """
        B, T, H = x.shape
        device = x.device

        # Generate RoPE position embeddings for this sequence length
        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds adapter max_seq_len {self.max_seq_len}"
            )
        cos = self.freqs_cos[:T].to(device)
        sin = self.freqs_sin[:T].to(device)
        position_embeddings = (cos, sin)

        # Convert key_padding_mask (True=PAD) to MiniMind attention_mask format.
        # MiniMind attention module expects mask as [B, T] or [B, 1, 1, T]
        # where 0=mask_out, 1=keep. The flash path checks for all-1 or None.
        # Our key_padding_mask is True=PAD, so we invert to get True=valid.
        if key_padding_mask is not None:
            # Invert: True=valid, False=PAD
            attention_mask = (~key_padding_mask).float()
        else:
            attention_mask = None

        for layer in self.layers:
            x, _ = layer(
                x,
                position_embeddings=position_embeddings,
                past_key_value=None,
                use_cache=False,
                attention_mask=attention_mask,
            )

        return x
