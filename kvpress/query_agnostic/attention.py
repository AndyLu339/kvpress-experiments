# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact attention over variable-length KV heads, without padding KV storage."""

from functools import lru_cache
from typing import Sequence

import torch
from packaging.version import Version
from torch.nn import functional as F

from kvpress.query_agnostic.cache import HeadCache


@lru_cache(maxsize=1)
def _flash_kernel():
    from flash_attn import __version__, flash_attn_varlen_func

    if Version(__version__) < Version("2.1"):
        raise ValueError("flash_varlen requires FlashAttention >= 2.1 for bottom-right causal masking.")
    return flash_attn_varlen_func


def grouped_attention(
    queries: torch.Tensor, heads: Sequence[HeadCache], query_positions: torch.Tensor, backend: str
) -> torch.Tensor:
    """Q: [query_heads, query_length, dim]; output: [query_length, query_heads * dim]."""
    hq, q_len, dim = queries.shape
    if hq % len(heads):
        raise ValueError("Query heads must divide into complete GQA groups.")
    groups = hq // len(heads)
    if backend == "flash_varlen":
        if not queries.is_cuda or queries.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("flash_varlen requires CUDA and FP16/BF16 queries.")
        # FlashAttention >= 2.1 aligns the causal mask at the bottom right. Each head's
        # chronological cache ends in exactly the current query block, so this implements
        # original-position causality even when earlier retained positions contain gaps.
        q = queries.reshape(len(heads), groups, q_len, dim).permute(0, 2, 1, 3).reshape(-1, groups, dim)
        k = torch.cat([h.keys for h in heads]).unsqueeze(1)
        v = torch.cat([h.values for h in heads]).unsqueeze(1)
        cu_q = torch.arange(len(heads) + 1, device=q.device, dtype=torch.int32) * q_len
        cu_k = torch.tensor([0] + [h.length for h in heads], device=q.device, dtype=torch.int32).cumsum(0)
        out = _flash_kernel()(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            cu_q,
            cu_k.to(torch.int32),
            q_len,
            max(h.length for h in heads),
            dropout_p=0.0,
            causal=True,
        )
        return out.reshape(len(heads), q_len, groups, dim).permute(1, 0, 2, 3).reshape(q_len, hq * dim)
    if backend != "sdpa":
        raise ValueError(f"Unknown attention backend: {backend}")
    outputs = []
    for head, cache in enumerate(heads):
        allowed = cache.positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        out = F.scaled_dot_product_attention(
            queries[head * groups : (head + 1) * groups].unsqueeze(0),
            cache.keys[None, None].expand(1, groups, -1, -1),
            cache.values[None, None].expand(1, groups, -1, -1),
            attn_mask=allowed,
            dropout_p=0.0,
        )
        outputs.append(out.squeeze(0))
    return torch.cat(outputs).transpose(0, 1).reshape(q_len, hq * dim)
