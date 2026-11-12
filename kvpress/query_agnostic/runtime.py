# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""True chunked Qwen2 prefill with fixed-budget, physically compact KV heads."""

import time
from dataclasses import dataclass
from functools import partial
from typing import Optional

import numpy as np
import torch
from transformers import Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from kvpress.query_agnostic.attention import grouped_attention
from kvpress.query_agnostic.budget import allocate_budget
from kvpress.query_agnostic.cache import HeadCache, select_topk
from kvpress.query_agnostic.scorers import OnlineScorer


@dataclass
class ExperimentConfig:
    method: str = "keydiff"
    budget: int = 8192
    chunk_size: int = 2048
    backend: str = "sdpa"
    seed: int = 0
    body_sink: int = 4
    recent_window: int = 0
    history_capacity: int = 256
    scoring_probes: int = 64
    sketch_dimension: int = 48
    ridge: float = 1e-2
    diagnostics: bool = False

    def __post_init__(self):
        if self.budget < 1 or self.chunk_size < 1 or self.body_sink < 0 or self.recent_window < 0:
            raise ValueError("Budget/chunk size must be positive; protected window sizes cannot be negative.")
        if self.backend not in ("sdpa", "flash_varlen"):
            raise ValueError("Choose sdpa or flash_varlen explicitly; no silent attention fallback is used.")


class QueryAgnosticRuntime:
    """Use as a context manager; only that model instance's attention forwards are replaced.

    Input tensors are explicit token segments: template prefix, neutral body, then each
    task/question/assistant suffix. The runtime never tokenizes or infers task boundaries.
    Static shares must belong to this checkpoint; uniform shares are an explicit ablation.
    """

    def __init__(
        self,
        model: Qwen2ForCausalLM,
        config: ExperimentConfig,
        shares: Optional[np.ndarray] = None,
        filters: Optional[torch.Tensor] = None,
        gates=None,
    ):
        if not isinstance(model, Qwen2ForCausalLM):
            raise ValueError("This experimental runtime currently supports Qwen2ForCausalLM only.")
        if model.config.use_sliding_window or model.config.rope_scaling is not None:
            raise ValueError("The experiment requires full attention and the target's unscaled default RoPE.")
        devices = {p.device for p in model.parameters()}
        if len(devices) != 1 or next(iter(devices)).type == "meta":
            raise ValueError("Use one fully loaded model on one device; CPU/disk offload is not supported.")
        self.model, self.config = model, config
        self.device = next(iter(devices))
        self.layers = model.config.num_hidden_layers
        self.kv_heads = model.config.num_key_value_heads
        self.query_heads = model.config.num_attention_heads
        if shares is None and config.method != "full":
            raise ValueError("Supply calibrated HM shares, or explicit uniform shares for development/ablation.")
        self.shares = np.ones((self.layers, self.kv_heads)) if shares is None else np.asarray(shares)
        if self.shares.shape != (self.layers, self.kv_heads):
            raise ValueError("HM profile shape does not match this model's layers and KV heads.")
        # Validate shares before allocating model-dependent state.
        allocate_budget(self.shares, np.ones_like(self.shares, dtype=int), np.zeros_like(self.shares, dtype=int), 1)
        head_dim = model.config.hidden_size // self.query_heads
        if filters is not None and tuple(filters.shape) != (self.layers, self.kv_heads, head_dim):
            raise ValueError("Q-filter geometry does not match this model.")
        self.scorer = OnlineScorer(
            config.method,
            self.layers,
            self.query_heads,
            self.kv_heads,
            model.model.rotary_emb,
            seed=config.seed,
            filters=None if filters is None else filters.detach().to(self.device),
            gates=gates,
            history_capacity=config.history_capacity,
            probes=config.scoring_probes,
            sketch_dimension=config.sketch_dimension,
            ridge=config.ridge,
        )
        self._original_forwards: list = []
        self._active = False
        self.reset("")

    def reset(self, document_id: str):
        self.cache: list[list[Optional[HeadCache]]] = [[None] * self.kv_heads for _ in range(self.layers)]
        self.branch: list[list[Optional[HeadCache]]] = [[None] * self.kv_heads for _ in range(self.layers)]
        self.scorer.reset(document_id)
        self.next_position = 0
        self.prefix_length = 0
        self.chunk_index = -1
        self.phase = "idle"
        self.frozen = False
        self.chunk_records: list[dict] = []
        self.retention_records: list[dict] = []
        self.prefill_seconds = 0.0

    def __enter__(self):
        if self._active or getattr(self.model, "_query_agnostic_owner", None) is not None:
            raise RuntimeError("A query-agnostic runtime is already active on this model.")
        self.model.eval()
        for layer in self.model.model.layers:
            module = layer.self_attn
            self._original_forwards.append(module.__dict__.get("forward"))
            module.forward = partial(self._attention_forward, module)
        setattr(self.model, "_query_agnostic_owner", self)
        self._active = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for layer, original in zip(self.model.model.layers, self._original_forwards):
            if original is None:
                del layer.self_attn.forward
            else:
                layer.self_attn.forward = original
        self._original_forwards.clear()
        del self.model._query_agnostic_owner
        self._active = False
        self.phase = "idle"

    def _ids(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype != torch.long:
            raise ValueError("Input token IDs must be int64 with shape [tokens] or [1, tokens].")
        return ids.to(self.device)

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _forward_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        if not self._active:
            raise RuntimeError("Use the runtime inside a with block.")
        length = ids.shape[1]
        self.positions = torch.arange(self.next_position, self.next_position + length, device=self.device)
        # A dict bypasses HF's dense-cache mask construction. Our per-head attention
        # applies original-position causality, and owns its KV rather than using Cache.update.
        output = self.model.model(
            input_ids=ids,
            position_ids=self.positions.unsqueeze(0),
            cache_position=self.positions,
            attention_mask={"full_attention": None},
            use_cache=False,
        )
        self.next_position += length
        return output.last_hidden_state

    def _allocate(self, new_length: int) -> np.ndarray:
        available = np.array([[new_length + (0 if h is None else h.length) for h in row] for row in self.cache])
        seen = self.next_position + new_length
        fixed = min(seen, self.prefix_length + self.config.body_sink)
        protected_count = fixed + min(self.config.recent_window, seen - fixed)
        protected = np.full_like(available, protected_count)
        return allocate_budget(self.shares, available, protected, self.layers * self.kv_heads * self.config.budget)

    @torch.inference_mode()
    def prefill(
        self, body_ids: torch.Tensor, prefix_ids: Optional[torch.Tensor] = None, document_id: str = "document"
    ) -> torch.Tensor:
        body = self._ids(body_ids)
        if body.shape[1] == 0:
            raise ValueError("The body cannot be empty.")
        self.reset(document_id)
        prefix = None if prefix_ids is None else self._ids(prefix_ids)
        self.prefix_length = 0 if prefix is None else prefix.shape[1]
        self.scorer.body_start, self.scorer.body_sink = self.prefix_length, self.config.body_sink
        self._synchronize()
        start = time.perf_counter()
        self.phase = "prefix"
        if prefix is not None and prefix.shape[1]:
            self._forward_tokens(prefix)
        self.phase = "body"
        last = None
        for chunk, offset in enumerate(range(0, body.shape[1], self.config.chunk_size)):
            self.chunk_index = chunk
            block = body[:, offset : offset + self.config.chunk_size]
            self.quotas = None if self.config.method in ("full", "ea_once") else self._allocate(block.shape[1])
            last = self._forward_tokens(block)[:, -1:].clone()
            self.chunk_records.append({"chunk": chunk, "seen_tokens": self.next_position, "kv_bytes": self.kv_bytes})
        if self.config.method == "ea_once":
            self.quotas = self._allocate(0)
            for layer, heads in enumerate(self.cache):
                for head, cache in enumerate(heads):
                    heads[head] = self._prune(layer, head, cache, self.next_position)
        self.frozen = True
        self.phase = "idle"
        self._synchronize()
        self.prefill_seconds = time.perf_counter() - start
        # This is the last body hidden state, not an answer logit: every question is
        # forwarded independently against the frozen body before choosing its first token.
        return last

    def _protected(self, cache: HeadCache, next_position: int) -> torch.Tensor:
        fixed = cache.positions < self.prefix_length + self.config.body_sink
        if self.config.recent_window:
            fixed = fixed | (cache.positions >= next_position - self.config.recent_window)
        return fixed

    def _prune(self, layer: int, head: int, cache: HeadCache, next_position: int) -> HeadCache:
        budget = int(self.quotas[layer, head])
        kept = cache
        if budget < cache.length:
            scores = self.scorer.score(layer, head, cache, next_position)
            kept = select_topk(cache, scores, budget, self._protected(cache, next_position))
        if self.config.diagnostics:
            sources = torch.unique(cache.source_chunks).tolist()
            for source in sources:
                self.retention_records.append(
                    {
                        "layer": layer,
                        "head": head,
                        "chunk": self.chunk_index,
                        "source_chunk": source,
                        "candidates": int((cache.source_chunks == source).sum()),
                        "kept": int((kept.source_chunks == source).sum()),
                    }
                )
        return kept

    def _attention_forward(self, module, hidden_states, position_embeddings, attention_mask=None, **kwargs):
        if self.phase not in ("prefix", "body", "question") or hidden_states.shape[0] != 1:
            raise RuntimeError("Model forwards must be issued through the active runtime (batch size one).")
        layer = module.layer_idx
        length = hidden_states.shape[1]
        shape = (1, length, -1, module.head_dim)
        raw_queries = module.q_proj(hidden_states).view(shape).transpose(1, 2)
        keys = module.k_proj(hidden_states).view(shape).transpose(1, 2)
        values = module.v_proj(hidden_states).view(shape).transpose(1, 2)[0]
        queries, keys = apply_rotary_pos_emb(raw_queries, keys, *position_embeddings)
        queries, keys = queries[0], keys[0]
        is_question = self.phase == "question"
        if self.phase == "body":
            excluded_sink = 0 if self.config.method == "ea_once" else self.config.body_sink
            valid = self.positions >= self.prefix_length + excluded_sink
            self.scorer.observe(layer, raw_queries[0, :, valid], self.positions[valid], self.chunk_index)
        new_scores = [None] * self.kv_heads
        if not is_question:
            new_scores = self.scorer.new_scores(layer, keys, hidden_states, self.positions)
        source = -2 if is_question else self.chunk_index
        candidates, tails = [], []
        for head in range(self.kv_heads):
            new = HeadCache(
                keys[head].contiguous(),
                values[head].contiguous(),
                self.positions.clone(),
                torch.full_like(self.positions, source),
                new_scores[head],
            )
            old = self.cache[layer][head]
            if is_question:
                tail = self.branch[layer][head]
                tail = new if tail is None else tail.append(new)
                tails.append(tail)
                # Only the small per-question tail persists. Full-body K/V concatenation
                # is temporary for this layer, rather than a document-sized branch copy.
                base = HeadCache(old.keys, old.values, old.positions, old.source_chunks)
                candidate = base.append(tail)
            else:
                candidate = new if old is None else old.append(new)
            candidates.append(candidate)
        output = grouped_attention(queries, candidates, self.positions, self.config.backend)
        output = module.o_proj(output.unsqueeze(0))
        if is_question:
            self.branch[layer] = tails
        else:
            end_position = self.next_position + length
            self.cache[layer] = [
                (
                    self._prune(layer, head, cache, end_position)
                    if self.phase == "body" and self.quotas is not None
                    else cache
                )
                for head, cache in enumerate(candidates)
            ]
        self.scorer.current_probes.pop(layer, None)
        return output, None

    @torch.inference_mode()
    def answer(self, question_ids: torch.Tensor, max_new_tokens: int = 96, eos_token_ids=None) -> torch.Tensor:
        if not self.frozen:
            raise RuntimeError("Complete body prefill before answering a question.")
        question = self._ids(question_ids)
        if question.shape[1] == 0 or max_new_tokens < 1:
            raise ValueError("A question suffix and a positive output limit are required.")
        eos = self.model.generation_config.eos_token_id if eos_token_ids is None else eos_token_ids
        eos_set = set(eos if isinstance(eos, (tuple, list, set)) else [eos])
        frozen_position = self.next_position
        generated = []
        self.phase = "question"
        try:
            hidden = self._forward_tokens(question)
            for step in range(max_new_tokens):
                token = int(self.model.lm_head(hidden[:, -1:]).float()[0, 0].argmax())
                generated.append(token)
                if token in eos_set or step + 1 == max_new_tokens:
                    break
                hidden = self._forward_tokens(torch.tensor([[token]], device=self.device))
        finally:
            self.next_position = frozen_position
            self.branch = [[None] * self.kv_heads for _ in range(self.layers)]
            self.phase = "idle"
        return torch.tensor(generated, device=self.device, dtype=torch.long)

    @property
    def kv_bytes(self) -> int:
        return sum(head.kv_bytes for row in self.cache for head in row if head is not None)

    @property
    def auxiliary_bytes(self) -> int:
        return self.scorer.auxiliary_bytes + sum(
            head.auxiliary_bytes for row in self.cache for head in row if head is not None
        )
