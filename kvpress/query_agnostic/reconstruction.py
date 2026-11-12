# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streamed HM/KVzip reconstruction labels with distinct softmax domains.

HM follows marginal-attention features.label_accumulator/hooked_passes (commit 63277ce).
Fast KVzip follows its prefilling KVScore._get_score (commit e04afaa).
See SOURCE_AUDIT.md for protocol differences and numerical adaptations.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from kvpress.query_agnostic import ExperimentConfig, QueryAgnosticRuntime

REPEAT_PROMPT = "\n\nRepeat the previous context exactly."
PART_PROMPT = "\n\nRepeat the part of the previous context exactly, starting with "


def reconstruction_peak(
    queries: torch.Tensor,
    keys: torch.Tensor,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    target_start: int,
    target_end: int,
    hidden_norms: Optional[torch.Tensor] = None,
    value_output_norms: Optional[torch.Tensor] = None,
    query_tile: int = 32,
    key_tile: int = 2048,
) -> torch.Tensor:
    """GQA/query-row max over a chosen key interval; denominator covers all supplied keys.

    The original BF16 attention probabilities are reproduced by casting the streamed
    FP32 softmax back to Q's dtype before normalization/max. No N x N tensor is built.
    """
    if query_tile < 1 or key_tile < 1 or not 0 <= target_start <= target_end <= len(keys):
        raise ValueError("Invalid score tiles or target key interval.")
    groups, q_length, dimension = queries.shape
    result = torch.zeros(target_end - target_start, device=keys.device)
    scale = dimension**-0.5

    def logits(q, start, end, q_positions):
        score = (q @ keys[start:end].T) * scale
        return score.float().masked_fill(
            key_positions[None, None, start:end] > q_positions[None, :, None], -float("inf")
        )

    for qs in range(0, q_length, query_tile):
        q = queries[:, qs : qs + query_tile]
        positions = query_positions[qs : qs + query_tile]
        log_z = torch.full((groups, q.shape[1]), -float("inf"), device=keys.device)
        for ks in range(0, len(keys), key_tile):
            tile = logits(q, ks, min(ks + key_tile, len(keys)), positions)
            log_z = torch.logaddexp(log_z, tile.logsumexp(-1))
        for ks in range(target_start, target_end, key_tile):
            end = min(ks + key_tile, target_end)
            attention = (logits(q, ks, end, positions) - log_z[:, :, None]).exp().to(queries.dtype).float()
            if hidden_norms is not None:
                attention = attention / hidden_norms[None, qs : qs + q.shape[1], None].clamp_min(1e-6)
            peak = attention.amax(1)
            if value_output_norms is not None:
                peak *= value_output_norms[:, ks - target_start : end - target_start]
            target = result[ks - target_start : end - target_start]
            target.copy_(torch.maximum(target, peak.amax(0)))
    return result


def value_output_norms(module, values: torch.Tensor, query_head_start: int, groups: int, tile: int = 512):
    """||W_O(head) v||, NOT ||v||; stream the wide residual dimension in token tiles."""
    dim = values.shape[-1]
    result = torch.empty(groups, values.shape[0], device=values.device)
    for group in range(groups):
        start = (query_head_start + group) * dim
        weight = module.o_proj.weight[:, start : start + dim]
        for offset in range(0, len(values), tile):
            projected = values[offset : offset + tile] @ weight.T
            result[group, offset : offset + tile] = projected.norm(dim=-1).float()
    return result


class FeatureWriter:
    """One memory-mapped file per layer, BF16 preserved as uint16 bits."""

    def __init__(self, directory: Path, model, tokens: int):
        directory.mkdir(parents=True, exist_ok=True)
        self.encoding = "bfloat16_bits" if model.dtype == torch.bfloat16 else "float32"
        if model.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("Calibration features require FP32 or BF16.")
        dtype = np.uint16 if self.encoding == "bfloat16_bits" else np.float32
        self.maps = [
            np.lib.format.open_memmap(
                directory / f"layer_{layer:02d}.npy", mode="w+", dtype=dtype, shape=(tokens, model.config.hidden_size)
            )
            for layer in range(model.config.num_hidden_layers)
        ]

    def write(self, layer: int, offset: int, hidden: torch.Tensor):
        data = hidden.detach().contiguous().cpu()
        if self.encoding == "bfloat16_bits":
            data = data.view(torch.uint16)
        self.maps[layer][offset : offset + len(data)] = data.numpy()

    def close(self):
        for array in self.maps:
            array.flush()
            array._mmap.close()
        self.maps.clear()


class ReconstructionCollector:
    def __init__(
        self, model, body_length: int, prefix_length: int, mode: str, writer=None, query_tile=32, key_tile=2048
    ):
        if mode not in ("hm", "gate"):
            raise ValueError("Choose hm or gate labels explicitly.")
        self.model, self.mode, self.writer = model, mode, writer
        self.prefix_length, self.n_context = prefix_length, prefix_length + body_length
        self.query_tile, self.key_tile = query_tile, key_tile
        width = self.n_context if mode == "hm" else body_length
        self.labels = torch.zeros(
            model.config.num_hidden_layers, model.config.num_key_value_heads, width, device=model.device
        )
        self.norms: dict[tuple[int, int], torch.Tensor] = {}
        self.target_start = self.target_end = 0

    def __call__(self, module, hidden, queries, heads, positions, phase):
        layer = module.layer_idx
        if phase == "body" and self.writer is not None:
            self.writer.write(layer, int(positions[0]) - self.prefix_length, hidden[0])
        if phase != "question":
            return
        groups = self.model.config.num_attention_heads // len(heads)
        h_norm = hidden[0].norm(dim=-1).float() if self.mode == "hm" else None
        for head, cache in enumerate(heads):
            q = queries[head * groups : (head + 1) * groups]
            if self.mode == "hm":
                identity = (layer, head)
                if identity not in self.norms:
                    self.norms[identity] = value_output_norms(
                        module, cache.values[: self.n_context], head * groups, groups
                    )
                score = reconstruction_peak(
                    q,
                    cache.keys,
                    positions,
                    cache.positions,
                    0,
                    self.n_context,
                    h_norm,
                    self.norms[identity],
                    self.query_tile,
                    self.key_tile,
                )
                self.labels[layer, head].copy_(torch.maximum(self.labels[layer, head], score))
            else:
                # Native Fast KVzip retains the entire template prefix as attention sinks.
                sink = self.prefix_length
                keys = torch.cat(
                    [cache.keys[:sink], cache.keys[self.target_start : self.target_end], cache.keys[-q.shape[1] :]]
                )
                kp = torch.cat(
                    [
                        cache.positions[:sink],
                        cache.positions[self.target_start : self.target_end],
                        cache.positions[-q.shape[1] :],
                    ]
                )
                score = reconstruction_peak(
                    q,
                    keys,
                    positions,
                    kp,
                    sink,
                    sink + self.target_end - self.target_start,
                    query_tile=self.query_tile,
                    key_tile=self.key_tile,
                )
                start, end = self.target_start - sink, self.target_end - sink
                self.labels[layer, head, start:end] = score


@torch.inference_mode()
def collect_reconstruction(
    model,
    tokenizer,
    body_ids: torch.Tensor,
    prefix_ids: torch.Tensor,
    suffix_ids: torch.Tensor,
    mode: str,
    backend: str = "sdpa",
    prefill_chunk: int = 2048,
    repeat_chunk: Optional[int] = None,
    feature_directory: Optional[Path] = None,
    query_tile: int = 32,
    key_tile: int = 2048,
):
    """Full-context prefill followed by isolated, teacher-forced reconstruction branches."""
    repeat_chunk = repeat_chunk or (512 if mode == "hm" else 2000)
    if repeat_chunk < 1:
        raise ValueError("Reconstruction chunk size must be positive.")
    body_ids, prefix_ids, suffix_ids = [x.flatten().to(model.device) for x in (body_ids, prefix_ids, suffix_ids)]
    writer = None if feature_directory is None else FeatureWriter(feature_directory, model, len(body_ids))
    collector = ReconstructionCollector(model, len(body_ids), len(prefix_ids), mode, writer, query_tile, key_tile)
    config = ExperimentConfig(method="full", chunk_size=prefill_chunk, backend=backend)
    try:
        with QueryAgnosticRuntime(model, config, observer=collector) as runtime:
            runtime.prefill(body_ids, prefix_ids)
            repeated = torch.cat([prefix_ids, body_ids]) if mode == "hm" else body_ids
            for start in range(0, len(repeated), repeat_chunk):
                runtime.branch = [[None] * runtime.kv_heads for _ in range(runtime.layers)]
                runtime.next_position = collector.n_context
                runtime.phase = "question"
                if mode == "hm":
                    prompt = tokenizer.encode(REPEAT_PROMPT, add_special_tokens=False)
                    runtime._forward_tokens(torch.tensor([prompt], device=model.device, dtype=torch.long))
                    runtime._forward_tokens(repeated[None, start : start + repeat_chunk])
                else:
                    end = min(start + repeat_chunk, len(repeated))
                    collector.target_start, collector.target_end = len(prefix_ids) + start, len(prefix_ids) + end
                    prompt = tokenizer.encode(REPEAT_PROMPT if start == 0 else PART_PROMPT, add_special_tokens=False)
                    prompt_ids = torch.tensor(prompt, device=model.device, dtype=torch.long)
                    if start:
                        prompt_ids = torch.cat([prompt_ids, repeated[max(0, start - 8) : start]])
                    request = torch.cat([prompt_ids, suffix_ids, repeated[start:end]])
                    runtime._forward_tokens(request.unsqueeze(0))
            runtime.branch = [[None] * runtime.kv_heads for _ in range(runtime.layers)]
        labels = collector.labels
        if mode == "hm":
            labels = labels.clamp_min(1e-12).log()
        if not torch.isfinite(labels).all():
            raise ValueError("Non-finite reconstruction labels.")
        return labels.cpu().numpy(), None if writer is None else writer.encoding
    finally:
        if writer is not None:
            writer.close()
