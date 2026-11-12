# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physically compact, position-aware storage for one KV head (batch size one)."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class HeadCache:
    keys: torch.Tensor
    values: torch.Tensor
    positions: torch.Tensor
    source_chunks: torch.Tensor
    scores: Optional[torch.Tensor] = None

    @property
    def length(self) -> int:
        return self.keys.shape[0]

    @property
    def kv_bytes(self) -> int:
        return self.keys.numel() * self.keys.element_size() + self.values.numel() * self.values.element_size()

    @property
    def auxiliary_bytes(self) -> int:
        tensors = [self.positions, self.source_chunks]
        if self.scores is not None:
            tensors.append(self.scores)
        return sum(t.numel() * t.element_size() for t in tensors)

    def append(self, other: "HeadCache") -> "HeadCache":
        if (self.scores is None) != (other.scores is None):
            raise ValueError("Persistent scores must accompany every cached entry.")
        scores = None if self.scores is None else torch.cat([self.scores, other.scores])
        return HeadCache(
            torch.cat([self.keys, other.keys]),
            torch.cat([self.values, other.values]),
            torch.cat([self.positions, other.positions]),
            torch.cat([self.source_chunks, other.source_chunks]),
            scores,
        )

    def select(self, indices: torch.Tensor) -> "HeadCache":
        # Advanced indexing creates independent storage: a short view of a long tensor would
        # retain its entire allocation, defeating physical KV eviction.
        return HeadCache(
            self.keys[indices],
            self.values[indices],
            self.positions[indices],
            self.source_chunks[indices],
            None if self.scores is None else self.scores[indices],
        )


def select_topk(cache: HeadCache, scores: torch.Tensor, budget: int, protected: torch.Tensor) -> HeadCache:
    """Stable score ranking, followed by chronological storage for causal attention."""
    if scores.shape != cache.positions.shape or protected.shape != scores.shape:
        raise ValueError("Scores and protection mask must have one entry per KV pair.")
    if budget < int(protected.sum()) or budget > cache.length:
        raise ValueError("Head budget must cover protected entries and cannot exceed available entries.")
    if not torch.isfinite(scores).all():
        raise ValueError("Non-finite KV scores.")
    if budget == cache.length:
        return cache
    rank_scores = scores.clone()
    rank_scores[protected] = float("inf")
    # Entries already follow original positions. Stable sorting gives the specified tie break.
    indices = torch.argsort(rank_scores, descending=True, stable=True)[:budget].sort().values
    return cache.select(indices)
