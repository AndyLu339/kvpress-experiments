# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Online scoring and query-only state for the query-agnostic experiment."""

import hashlib
from typing import Optional

import numpy as np
import torch
from torch.nn import functional as F
from transformers.models.qwen2.modeling_qwen2 import rotate_half

from kvpress.query_agnostic.cache import HeadCache
from kvpress.query_agnostic.probes import QueryHistory, tiled_probe_scores

METHODS = ("full", "ea_once", "keydiff", "leverage", "qfilters", "fastkvzip", "probe_mean", "probe_max", "fixed_random")


def stable_seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "little") % (2**63 - 1)


def rotate(vectors: torch.Tensor, positions: torch.Tensor, rotary, inverse: bool = False) -> torch.Tensor:
    cos, sin = rotary(vectors.float().unsqueeze(0), positions.unsqueeze(0))
    sign = -1 if inverse else 1
    return vectors.float() * cos[0] + sign * rotate_half(vectors.float()) * sin[0]


def fixed_priorities(positions: torch.Tensor, seed: int) -> torch.Tensor:
    """Stateless SplitMix64 priorities; no RNG state depends on chunk boundaries."""
    values = positions.detach().cpu().numpy().astype(np.uint64)
    with np.errstate(over="ignore"):
        values = values + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        values = values ^ (values >> np.uint64(31))
    # Exactly representable 53-bit uniform priorities. Stable ranking resolves collisions.
    priorities = (values >> np.uint64(11)).astype(np.float64) / float(2**53)
    return torch.from_numpy(priorities).to(positions.device)


class QueryMoments:
    """Parallel Welford updates over the complete body, excluding template queries."""

    def __init__(self):
        self.count = 0
        self.mean: Optional[torch.Tensor] = None
        self.m2: Optional[torch.Tensor] = None

    def update(self, queries: torch.Tensor):
        queries = queries.float()
        n = queries.shape[-2]
        if n == 0:
            return
        mean = queries.mean(-2)
        centered = queries - mean[:, None]
        m2 = centered.transpose(-1, -2) @ centered
        if self.mean is None:
            self.mean, self.m2, self.count = mean, m2, n
            return
        delta = mean - self.mean
        total = self.count + n
        self.m2 = self.m2 + m2 + delta[:, :, None] * delta[:, None, :] * (self.count * n / total)
        self.mean = self.mean + delta * (n / total)
        self.count = total


class OnlineScorer:
    def __init__(
        self,
        method: str,
        layers: int,
        query_heads: int,
        kv_heads: int,
        rotary,
        seed: int = 0,
        filters: Optional[torch.Tensor] = None,
        gates=None,
        history_capacity: int = 256,
        probes: int = 64,
        sketch_dimension: int = 48,
        ridge: float = 1e-2,
    ):
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}.")
        if sketch_dimension < 1 or ridge <= 0:
            raise ValueError("Leverage sketch dimension and ridge must be positive.")
        self.method, self.seed, self.rotary = method, seed, rotary
        self.layers, self.query_heads, self.kv_heads = layers, query_heads, kv_heads
        self.filters = filters if method == "qfilters" else None
        self.gates = gates if method == "fastkvzip" else None
        self.body_start, self.body_sink = 0, 4
        self.history_capacity, self.probe_count = history_capacity, probes
        self.sketch_dimension, self.ridge = sketch_dimension, ridge
        if method == "qfilters" and filters is None:
            raise ValueError("qfilters requires checkpoint-specific calibrated filters.")
        if method == "fastkvzip" and (gates is None or len(gates) != layers):
            raise ValueError("fastkvzip requires one trained gate per target layer.")
        self.reset("")

    def reset(self, document_id: str):
        self.document_id = document_id
        self.histories: dict[tuple[int, int], QueryHistory] = {}
        self.moments: dict[int, QueryMoments] = {}
        self.sketches: dict[tuple[int, int], torch.Tensor] = {}
        self.current_probes: dict[int, list] = {}

    @property
    def persistent_scores(self) -> bool:
        return self.method in ("fixed_random", "qfilters", "fastkvzip")

    def new_scores(self, layer: int, keys: torch.Tensor, hidden: torch.Tensor, positions: torch.Tensor):
        if self.method == "fixed_random":
            return [
                fixed_priorities(positions, stable_seed(self.seed, self.document_id, layer, head))
                for head in range(self.kv_heads)
            ]
        if self.method == "qfilters":
            filters = self.filters[layer].to(device=keys.device, dtype=torch.float32)
            return list((-(keys.float() * filters[:, None]).sum(-1)).unbind())
        if self.method == "fastkvzip":
            scores = self.gates[layer](hidden)[0].float()
            if scores.shape != keys.shape[:2]:
                raise ValueError("Gate output does not match the target KV-head geometry.")
            return list(scores.unbind())
        return [None] * self.kv_heads

    def observe(self, layer: int, queries: torch.Tensor, positions: torch.Tensor, chunk: int):
        if self.method == "ea_once":
            self.moments.setdefault(layer, QueryMoments()).update(queries)
        elif self.method.startswith("probe_"):
            probes = []
            for head, q in enumerate(queries):
                history = self.histories.setdefault(
                    (layer, head), QueryHistory(self.history_capacity, self.probe_count)
                )
                seed = stable_seed(self.seed, self.document_id, layer, head, chunk, "local")
                history.update(q, positions, seed)
                probes.append(history.scoring_probes(stable_seed(seed, "scoring")))
            self.current_probes[layer] = probes

    def score(self, layer: int, head: int, cache: HeadCache, next_position: int) -> torch.Tensor:
        keys = cache.keys.float()
        if self.persistent_scores:
            return cache.scores
        if self.method == "keydiff":
            anchor = F.normalize(keys, dim=-1).mean(0, keepdim=True)
            return -F.cosine_similarity(keys, anchor, dim=-1)
        if self.method == "leverage":
            raw = rotate(keys, cache.positions, self.rotary, inverse=True)
            identity = (layer, head)
            if identity not in self.sketches:
                generator = torch.Generator(device=keys.device).manual_seed(stable_seed(self.seed, layer, head))
                self.sketches[identity] = (
                    torch.randn(keys.shape[-1], self.sketch_dimension, generator=generator, device=keys.device)
                    / self.sketch_dimension**0.5
                )
            x = (raw - raw.mean(0)) @ self.sketches[identity]
            gram = x.T @ x + self.ridge * torch.eye(self.sketch_dimension, device=x.device)
            factor = torch.linalg.cholesky((gram + gram.T) / 2)
            solved = torch.cholesky_solve(x.T, factor).T
            return (x * solved).sum(-1).clamp_min(0)
        if self.method.startswith("probe_"):
            group = self.query_heads // self.kv_heads
            scores = []
            for probe in self.current_probes[layer][head * group : (head + 1) * group]:
                if probe is None:
                    scores.append(torch.zeros(cache.length, device=keys.device))
                    continue
                virtual_positions = torch.full_like(probe.positions, next_position)
                queries = rotate(probe.representatives, virtual_positions, self.rotary)
                scores.append(tiled_probe_scores(queries, keys, probe.counts, self.method[6:]))
            return torch.stack(scores).mean(0)
        if self.method == "ea_once":
            return self._ea_score(layer, head, cache, next_position)
        raise ValueError(f"{self.method} does not need an eviction score.")

    def _ea_score(self, layer: int, head: int, cache: HeadCache, next_position: int) -> torch.Tensor:
        stats = self.moments.get(layer)
        if stats is None or stats.count == 0:
            return torch.zeros(cache.length, device=cache.keys.device)
        dim = cache.keys.shape[-1]
        positions = torch.arange(next_position, next_position + 512, device=cache.keys.device)
        cos, sin = self.rotary(cache.keys.float().unsqueeze(0), positions.unsqueeze(0))
        eye = torch.eye(dim, device=cache.keys.device)
        rotation = cos[0].mean(0)[:, None] * eye + sin[0].mean(0)[:, None] * rotate_half(eye).T
        group = self.query_heads // self.kv_heads
        start = head * group
        mean = stats.mean[start : start + group] @ rotation.T
        cov = rotation @ (stats.m2[start : start + group] / stats.count) @ rotation.T
        keys = cache.keys.float()
        logits = torch.stack(
            [keys @ mean[g] / dim**0.5 + ((keys @ cov[g]) * keys).sum(-1) / (2 * dim) for g in range(group)]
        )
        # Match ExpectedAttentionPress: protected prefix/sinks are outside the softmax domain.
        valid = cache.positions >= self.body_start + self.body_sink
        logits[:, ~valid] = -float("inf")
        scores = logits.softmax(-1).mean(0) * cache.values.float().norm(dim=-1)
        return scores.masked_fill(~valid, 0)

    @property
    def auxiliary_bytes(self) -> int:
        total = sum(h.clusters.nbytes for h in self.histories.values() if h.clusters is not None)
        total += sum(t.numel() * t.element_size() for t in self.sketches.values())
        for stats in self.moments.values():
            total += sum(t.numel() * t.element_size() for t in (stats.mean, stats.m2) if t is not None)
        if self.filters is not None:
            total += self.filters.numel() * self.filters.element_size()
        if self.gates is not None:
            total += sum(p.numel() * p.element_size() for gate in self.gates for p in gate.parameters())
        return total
