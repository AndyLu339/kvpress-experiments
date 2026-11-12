# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No checkpoint downloads: numerical and lifecycle checks on a tiny random Qwen2."""

import importlib.util

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from kvpress.presses.fastkvzip_press import FastKVzipGate
from kvpress.query_agnostic import ExperimentConfig, QueryAgnosticRuntime
from kvpress.query_agnostic.attention import grouped_attention
from kvpress.query_agnostic.budget import allocate_budget
from kvpress.query_agnostic.cache import HeadCache, select_topk
from kvpress.query_agnostic.probes import QueryClusters, QueryHistory, merge_clusters, tiled_probe_scores
from kvpress.query_agnostic.scorers import QueryMoments, fixed_priorities, rotate

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(scope="module", autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(params=DEVICES)
def model(request):
    torch.manual_seed(11)
    config = Qwen2Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rope_theta=10000000,
        attention_dropout=0,
        eos_token_id=96,
    )
    config._attn_implementation = "eager"
    return Qwen2ForCausalLM(config).eval().to(request.param)


def tokens(model, n=35):
    return (torch.arange(n, device=model.device) % 80 + 1).unsqueeze(0)


@torch.inference_mode()
def test_chunked_full_matches_single_forward_and_greedy(model):
    ids = tokens(model)
    expected = model.model(ids).last_hidden_state[:, -1:]
    expected_answer = model.generate(ids, max_new_tokens=5, do_sample=False, pad_token_id=96)[0, ids.shape[1] :]
    original_forward = model.model.layers[0].self_attn.forward
    with QueryAgnosticRuntime(model, ExperimentConfig(method="full", chunk_size=7)) as runtime:
        actual = runtime.prefill(ids[:, 3:-4], ids[:, :3])
        full_body = model.lm_head(actual)
        assert full_body.shape == (1, 1, 97)
        answer = runtime.answer(ids[:, -4:], max_new_tokens=5)
        torch.testing.assert_close(answer, expected_answer, rtol=0, atol=0)
        # Query forward against the frozen body must match the single full input.
        runtime.phase = "question"
        hidden = runtime._forward_tokens(ids[:, -4:])[:, -1:]
        torch.testing.assert_close(hidden, expected, rtol=2e-5, atol=2e-6)
    assert model.model.layers[0].self_attn.forward == original_forward
    torch.testing.assert_close(model.model(ids).last_hidden_state[:, -1:], expected)


@pytest.mark.parametrize(
    "method", ["keydiff", "leverage", "qfilters", "fastkvzip", "probe_mean", "probe_max", "ea_once"]
)
@torch.inference_mode()
def test_no_eviction_matches_full(model, method):
    ids = tokens(model, 21)
    expected = model.model(ids).last_hidden_state[:, -1:]
    gates = [FastKVzipGate(i, 32, 2, 2, model.dtype).to(model.device).eval() for i in range(2)]
    config = ExperimentConfig(method=method, budget=64, chunk_size=7, body_sink=1, history_capacity=4, scoring_probes=2)
    with QueryAgnosticRuntime(
        model, config, shares=np.ones((2, 2)), filters=torch.randn(2, 2, 8), gates=gates
    ) as runtime:
        actual = runtime.prefill(ids[:, 2:], ids[:, :2])
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize(
    "method", ["keydiff", "leverage", "qfilters", "fastkvzip", "probe_mean", "probe_max", "ea_once", "fixed_random"]
)
@torch.inference_mode()
def test_eviction_budget_positions_and_question_rollback(model, method):
    ids = tokens(model, 34)
    config = ExperimentConfig(
        method=method,
        budget=6,
        chunk_size=7,
        body_sink=1,
        history_capacity=4,
        scoring_probes=2,
        diagnostics=True,
    )
    gates = [FastKVzipGate(i, 32, 2, 2, model.dtype).to(model.device).eval() for i in range(2)]
    with QueryAgnosticRuntime(
        model, config, shares=np.array([[9, 3], [2, 1]]), filters=torch.randn(2, 2, 8), gates=gates
    ) as runtime:
        runtime.prefill(ids[:, 2:30], ids[:, :2], document_id="example")
        assert sum(head.length for row in runtime.cache for head in row) == 4 * config.budget
        assert len({head.length for row in runtime.cache for head in row}) > 1
        originals = []
        for row in runtime.cache:
            for head in row:
                assert torch.all(head.positions[1:] > head.positions[:-1])
                assert head.positions[:3].tolist() == [0, 1, 2]
                assert head.keys.untyped_storage().nbytes() == head.keys.numel() * head.keys.element_size()
                originals.append((head.keys.data_ptr(), head.positions.clone(), head.keys.clone()))
        assert runtime.kv_bytes == 4 * config.budget * 8 * 2 * model.dtype.itemsize
        first = runtime.answer(ids[:, 30:], max_new_tokens=4)
        second = runtime.answer(ids[:, 30:], max_new_tokens=4)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        assert runtime.next_position == 30
        assert all(head is None for row in runtime.branch for head in row)
        for cache, (pointer, positions, keys) in zip([h for row in runtime.cache for h in row], originals):
            assert cache.keys.data_ptr() == pointer
            torch.testing.assert_close(cache.positions, positions, rtol=0, atol=0)
            torch.testing.assert_close(cache.keys, keys, rtol=0, atol=0)
        assert runtime.retention_records
        if method.startswith("probe_"):
            for history in runtime.scorer.histories.values():
                assert history.clusters.counts.sum() == 27
                assert history.clusters.size <= 4
                assert torch.all(history.clusters.positions >= 3)
        elif method == "ea_once":
            assert all(stats.count == 28 for stats in runtime.scorer.moments.values())


def test_budget_rounding_floors_caps_and_exact_total():
    shares = np.array([[100, 4], [2, 1]])
    available = np.array([[4, 30], [30, 30]])
    protected = np.full((2, 2), 3)
    quota = allocate_budget(shares, available, protected, 40)
    assert quota.sum() == 40
    assert quota[0, 0] == 4
    assert np.all(quota >= protected)
    assert np.all(quota <= available)
    np.testing.assert_array_equal(allocate_budget(shares, available, protected, 1000), available)
    with pytest.raises(ValueError, match="protected"):
        allocate_budget(shares, available, protected, 11)


def make_cache(n=19, d=8):
    return HeadCache(torch.randn(n, d), torch.randn(n, d), torch.arange(n), torch.arange(n) // 4)


def test_frozen_random_topk_is_chunk_invariant_and_ties_are_chronological():
    full = make_cache()
    priorities = fixed_priorities(full.positions, 12)
    once = select_topk(full, priorities, 7, full.positions < 2)
    old = None
    for start in range(0, full.length, 3):
        new = full.select(torch.arange(start, min(start + 3, full.length)))
        old = new if old is None else old.append(new)
        old = select_topk(old, fixed_priorities(old.positions, 12), min(7, old.length), old.positions < 2)
    torch.testing.assert_close(old.positions, once.positions)
    tied = select_topk(full, torch.ones(full.length), 7, full.positions < 2)
    assert tied.positions.tolist() == list(range(7))


def test_ragged_gqa_matches_dense_original_position_mask():
    torch.manual_seed(3)
    heads = [make_cache(7), make_cache(5)]
    heads[0] = HeadCache(heads[0].keys, heads[0].values, torch.tensor([0, 2, 5, 8, 10, 11, 12]), heads[0].source_chunks)
    heads[1] = HeadCache(heads[1].keys, heads[1].values, torch.tensor([1, 7, 10, 11, 12]), heads[1].source_chunks)
    q = torch.randn(4, 3, 8)
    query_positions = torch.tensor([10, 11, 12])
    actual = grouped_attention(q, heads, query_positions, "sdpa")
    expected = []
    for h, cache in enumerate(heads):
        logits = q[h * 2 : h * 2 + 2] @ cache.keys.T / 8**0.5
        logits.masked_fill_(cache.positions[None] > query_positions[:, None], -float("inf"))
        expected.append(logits.softmax(-1) @ cache.values)
    expected = torch.cat(expected).transpose(0, 1).reshape(3, -1)
    torch.testing.assert_close(actual, expected)


def test_probe_tiling_matches_dense_mean_and_max():
    torch.manual_seed(4)
    q, k, w = torch.randn(37, 8), torch.randn(23, 8), torch.arange(1, 38)
    attention = (q @ k.T / 8**0.5).softmax(-1)
    for reduction, expected in (
        ("mean", (attention * w[:, None]).sum(0) / w.sum()),
        ("max", attention.max(0).values),
    ):
        torch.testing.assert_close(tiled_probe_scores(q, k, w, reduction, tile=5), expected)


def test_weighted_merge_moments_and_real_representatives():
    q = torch.tensor([[1.0, 3.0], [3.0, 5.0], [10.0, 4.0], [12.0, 6.0]])
    raw = QueryClusters(q, torch.ones(4, dtype=torch.long), torch.zeros(4), q, torch.arange(4))
    merged = merge_clusters(raw, torch.tensor([0, 0, 1, 1]), torch.stack([q[:2].mean(0), q[2:].mean(0)]))
    combined = merge_clusters(merged, torch.zeros(2, dtype=torch.long), q.mean(0, keepdim=True))
    assert combined.counts.item() == 4
    torch.testing.assert_close(combined.sse.sum(), (q - q.mean(0)).square().sum())
    assert any(torch.equal(combined.representatives[0], v) for v in q)
    moments = QueryMoments()
    moments.update(q[None, :2])
    moments.update(q[None, 2:])
    torch.testing.assert_close(moments.mean[0], q.mean(0))
    torch.testing.assert_close(moments.m2[0], (q - q.mean(0)).T @ (q - q.mean(0)))


def test_query_history_conserves_mass_and_handles_identical_queries():
    history = QueryHistory(capacity=3, probes=2)
    for chunk in range(4):
        q = torch.ones(9, 8) * (chunk % 2)
        history.update(q, torch.arange(chunk * 9, (chunk + 1) * 9), seed=chunk)
        assert history.clusters.counts.sum() == (chunk + 1) * 9
        assert history.clusters.size <= 3
        probes = history.scoring_probes(seed=99)
        assert probes.counts.sum() == (chunk + 1) * 9
        assert probes.size <= 2


@torch.inference_mode()
def test_rotary_inverse_uses_original_positions(model):
    vectors = torch.randn(9, 8, device=model.device)
    positions = torch.tensor([0, 2, 5, 100, 1000, 3000, 10000, 50000, 120000], device=model.device)
    rotated = rotate(vectors, positions, model.model.rotary_emb)
    restored = rotate(rotated, positions, model.model.rotary_emb, inverse=True)
    torch.testing.assert_close(restored, vectors, rtol=1e-5, atol=1e-6)


@torch.inference_mode()
def test_exception_restores_model_and_question_state(model, monkeypatch):
    ids = tokens(model, 18)
    original = model.model.layers[0].self_attn.forward
    with pytest.raises(RuntimeError, match="deliberate"):
        with QueryAgnosticRuntime(model, ExperimentConfig(method="full", chunk_size=7)) as runtime:
            runtime.prefill(ids)

            def fail(*args, **kwargs):
                raise RuntimeError("deliberate")

            monkeypatch.setattr(model.lm_head, "forward", fail)
            runtime.answer(ids[:, :3])
    assert model.model.layers[0].self_attn.forward == original
    assert runtime.next_position == 18
    assert all(head is None for row in runtime.branch for head in row)


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("flash_attn") is None,
    reason="FlashAttention CUDA extension is not installed/visible.",
)
@pytest.mark.parametrize("q_len", [1, 6])
@pytest.mark.parametrize("groups,head_dim", [(2, 16), (7, 128)])
@torch.inference_mode()
def test_flash_varlen_matches_sdpa_with_gapped_ragged_history(q_len, groups, head_dim):
    torch.manual_seed(17)
    heads = []
    positions = torch.arange(100, 100 + q_len, device="cuda")
    for past in (11, 3, 7, 5):
        n = past + q_len
        heads.append(
            HeadCache(
                torch.randn(n, head_dim, device="cuda", dtype=torch.bfloat16),
                torch.randn(n, head_dim, device="cuda", dtype=torch.bfloat16),
                torch.cat([torch.arange(past, device="cuda") * 3, positions]),
                torch.zeros(n, device="cuda", dtype=torch.long),
            )
        )
    queries = torch.randn(len(heads) * groups, q_len, head_dim, device="cuda", dtype=torch.bfloat16)
    expected = grouped_attention(queries, heads, positions, "sdpa")
    actual = grouped_attention(queries, heads, positions, "flash_varlen")
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
