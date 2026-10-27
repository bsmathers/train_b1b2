"""Portable sliding-window attention for checkpoint evaluation.

The CUDA training configs use FlashAttention 2 with ``window_size=(32, 0)``.
FlashAttention cannot be installed on macOS, but the attention implementation
has no learned parameters.  This module preserves the same causal receptive
field while using PyTorch scaled-dot-product attention for local inference.
"""

from __future__ import annotations

import gin
import torch
from torch import nn
from torch.nn import functional as F

from amago.nets.transformer import SelfAttention


@gin.configurable
class PortableSlidingWindowAttention(SelfAttention):
    """Causal attention over the current token and ``window_size`` predecessors."""

    def __init__(
        self,
        causal: bool,
        dropout: float,
        window_size: int = gin.REQUIRED,
    ):
        super().__init__(causal=causal, dropout=dropout)
        if window_size < 0:
            raise ValueError("window_size must be non-negative")
        self.window_size = int(window_size)
        self.dropout_layer = nn.Dropout(dropout)

    @staticmethod
    def _quantize_like_flash(tensor: torch.Tensor) -> torch.Tensor:
        # FlashAttention receives BF16 q/k/v in the source implementation.
        # Quantize identically, then return FP32 for portable CPU kernels and
        # the following FP32 projection layer.
        if tensor.dtype == torch.float32:
            return tensor.to(torch.bfloat16).to(torch.float32)
        return tensor

    def _attention(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        allowed: torch.Tensor,
    ) -> torch.Tensor:
        original_dtype = queries.dtype
        queries = self._quantize_like_flash(queries).transpose(1, 2)
        keys = self._quantize_like_flash(keys).transpose(1, 2)
        values = self._quantize_like_flash(values).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=allowed,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        return out.transpose(1, 2).to(original_dtype)

    def forward(self, qkv, key_cache=None, val_cache=None, cache_seqlens=None):
        queries, keys, values = torch.unbind(qkv, dim=2)
        batch, query_len, _heads, _head_dim = queries.shape

        if key_cache is None or val_cache is None or cache_seqlens is None:
            query_pos = torch.arange(query_len, device=qkv.device)[:, None]
            key_pos = torch.arange(query_len, device=qkv.device)[None, :]
            allowed = query_pos - key_pos <= self.window_size
            if self.causal:
                allowed &= key_pos <= query_pos
            allowed = allowed[None, None, :, :].expand(batch, 1, -1, -1)
            return self._attention(queries, keys, values, allowed)

        if self.training:
            raise RuntimeError("KV-cache attention is inference-only")
        if query_len != 1:
            raise ValueError(f"KV-cache inference requires query length 1, got {query_len}")

        cache_rows = torch.arange(batch, device=qkv.device)
        cache_positions = cache_seqlens.to(torch.long)
        key_cache[cache_rows, cache_positions] = keys[:, 0].to(key_cache.dtype)
        val_cache[cache_rows, cache_positions] = values[:, 0].to(val_cache.dtype)
        end = cache_positions + 1
        max_len = int(end.max().item())
        cached_keys = torch.nan_to_num(key_cache[:, :max_len])
        cached_values = torch.nan_to_num(val_cache[:, :max_len])
        key_pos = torch.arange(max_len, device=qkv.device)[None, :]
        allowed = key_pos < end[:, None]
        if self.causal:
            allowed &= key_pos >= (end - self.window_size - 1)[:, None]
        allowed = allowed[:, None, None, :]
        return self._attention(queries, cached_keys, cached_values, allowed)
