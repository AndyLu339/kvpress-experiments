# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, weighted summaries of body queries, independent of KV survival."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class QueryClusters:
    means: torch.Tensor
    counts: torch.Tensor
    sse: torch.Tensor
    representatives: torch.Tensor
    positions: torch.Tensor

    @property
    def size(self) -> int:
        return self.counts.numel()

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in vars(self).values())


def weighted_kmeans(
    points: torch.Tensor,
    weights: torch.Tensor,
    k: int,
    seed: int,
    initial: Optional[torch.Tensor] = None,
    iterations: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted k-means++; growing local fits retain their preceding fitted centers."""
    points, weights = points.float(), weights.float()
    k = min(k, points.shape[0])
    if k < 1:
        raise ValueError("Cannot cluster an empty query set.")
    generator = torch.Generator(device=points.device).manual_seed(seed)
    centers = [] if initial is None else list(initial.unbind())[:k]
    nearest = torch.full_like(weights, float("inf"))
    for center in centers:
        nearest = torch.minimum(nearest, (points - center).square().sum(-1))
    while len(centers) < k:
        mass = weights if not centers else weights * nearest
        if mass.sum() == 0:
            # Coincident points: extra centers are harmless; empty memberships are repaired below.
            index = len(centers) % len(points)
        else:
            index = int(torch.multinomial(mass, 1, generator=generator))
        centers.append(points[index])
        nearest = torch.minimum(nearest, (points - points[index]).square().sum(-1))
    center_tensor = torch.stack(centers)
    assignment = torch.zeros(len(points), device=points.device, dtype=torch.long)
    for _ in range(iterations):
        # cdist returns only [points, centers], avoiding an N x K x head_dim workspace.
        distances = torch.cdist(points, center_tensor, compute_mode="donot_use_mm_for_euclid_dist").square()
        assignment = distances.argmin(-1)
        members = torch.bincount(assignment, minlength=k)
        # Splitting a non-singleton cluster guarantees every returned cluster has members,
        # including the degenerate case of repeated identical queries.
        residual = weights * distances.gather(1, assignment[:, None]).squeeze(1)
        for empty in torch.where(members == 0)[0].tolist():
            eligible = members[assignment] > 1
            index = int(residual.masked_fill(~eligible, -1).argmax())
            members[assignment[index]] -= 1
            assignment[index] = empty
            members[empty] += 1
        totals = torch.zeros(k, device=points.device).index_add_(0, assignment, weights)
        updated = torch.zeros(k, points.shape[-1], device=points.device)
        updated.index_add_(0, assignment, weights[:, None] * points)
        updated /= totals[:, None]
        converged = torch.equal(center_tensor, updated)
        center_tensor = updated
        if converged:
            break
    return assignment, center_tensor


def merge_clusters(clusters: QueryClusters, assignment: torch.Tensor, means: torch.Tensor) -> QueryClusters:
    """Exact count/mean/SSE merge; representatives remain actual observed queries."""
    k = means.shape[0]
    counts = torch.zeros(k, device=means.device, dtype=torch.long)
    counts.index_add_(0, assignment, clusters.counts)
    terms = clusters.sse + clusters.counts * (clusters.means - means[assignment]).square().sum(-1)
    sse = torch.zeros(k, device=means.device).index_add_(0, assignment, terms)
    representatives, positions = [], []
    for index in range(k):
        members = torch.where(assignment == index)[0]
        distance = (clusters.representatives[members].float() - means[index]).square().sum(-1)
        tied = members[distance == distance.min()]
        closest = tied[clusters.positions[tied].argmin()]
        representatives.append(clusters.representatives[closest])
        positions.append(clusters.positions[closest])
    return QueryClusters(means, counts, sse, torch.stack(representatives), torch.stack(positions))


class QueryHistory:
    def __init__(self, capacity: int = 256, probes: int = 64, iterations: int = 10):
        if not 1 <= probes <= capacity or iterations < 1:
            raise ValueError("Require 1 <= probes <= history capacity and positive Lloyd iterations.")
        self.capacity, self.probes, self.iterations = capacity, probes, iterations
        self.clusters: Optional[QueryClusters] = None
        self.last_local_size = 0
        self.last_local_sse = 0.0

    def update(self, queries: torch.Tensor, positions: torch.Tensor, seed: int):
        if queries.shape[0] == 0:
            return
        n = queries.shape[0]
        raw = QueryClusters(
            queries.float(),
            torch.ones(n, device=queries.device, dtype=torch.long),
            torch.zeros(n, device=queries.device),
            queries.detach(),
            positions,
        )
        baseline = (queries.float() - queries.float().mean(0)).square().sum()
        initial = None
        for size in (1, 8, 16, 32, 64, 128):
            assignment, means = weighted_kmeans(raw.means, raw.counts, min(size, n), seed, initial, self.iterations)
            current = merge_clusters(raw, assignment, means)
            initial = means
            if current.sse.sum() <= 0.10 * baseline or size >= n:
                break
        self.last_local_size = current.size
        self.last_local_sse = float(current.sse.sum())
        if self.clusters is not None:
            current = QueryClusters(
                **{name: torch.cat([getattr(self.clusters, name), value]) for name, value in vars(current).items()}
            )
        if current.size > self.capacity:
            assignment, means = weighted_kmeans(
                current.means, current.counts, self.capacity, seed + 1, iterations=self.iterations
            )
            current = merge_clusters(current, assignment, means)
        self.clusters = current

    def scoring_probes(self, seed: int) -> Optional[QueryClusters]:
        if self.clusters is None or self.clusters.size <= self.probes:
            return self.clusters
        assignment, means = weighted_kmeans(
            self.clusters.means, self.clusters.counts, self.probes, seed, iterations=self.iterations
        )
        return merge_clusters(self.clusters, assignment, means)


def tiled_probe_scores(
    queries: torch.Tensor, keys: torch.Tensor, weights: torch.Tensor, reduction: str, tile: int = 1024
) -> torch.Tensor:
    """Two-pass exact softmax, with bounded query/key tiles and FP32 reductions."""
    queries, keys, weights = queries.float(), keys.float(), weights.float()
    if reduction not in ("mean", "max") or tile < 1:
        raise ValueError("Expected mean/max reduction and a positive tile size.")
    score = torch.zeros(keys.shape[0], device=keys.device)
    scale = keys.shape[-1] ** -0.5
    for q_start in range(0, queries.shape[0], 32):
        q = queries[q_start : q_start + 32]
        w = weights[q_start : q_start + 32]
        log_z = torch.full((len(q),), -float("inf"), device=keys.device)
        for k_start in range(0, keys.shape[0], tile):
            logits = q @ keys[k_start : k_start + tile].T * scale
            log_z = torch.logaddexp(log_z, torch.logsumexp(logits, dim=-1))
        for k_start in range(0, keys.shape[0], tile):
            logits = q @ keys[k_start : k_start + tile].T * scale
            probabilities = (logits - log_z[:, None]).exp()
            target = score[k_start : k_start + tile]
            if reduction == "mean":
                target.add_((probabilities * w[:, None]).sum(0) / weights.sum())
            else:
                target.copy_(torch.maximum(target, probabilities.max(0).values))
    return score
