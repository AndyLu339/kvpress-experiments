# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differential checks against tracked HM author functions, plus offline stage integration."""

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen2Config

from kvpress import ExpectedAttentionPress
from kvpress.presses.fastkvzip_press import FastKVzipGate
from kvpress.query_agnostic import ExperimentConfig, QueryAgnosticRuntime
from kvpress.query_agnostic.artifacts import artifact_metadata
from kvpress.query_agnostic.calibrate import collect_labels, collect_qfilter_samples
from kvpress.query_agnostic.calibration_ops import (
    build_hm_profiles,
    build_qfilters,
    load_hm_primitives,
    qfilters_from_samples,
)
from kvpress.query_agnostic.calibration_smoke import SyntheticTokenizer
from kvpress.query_agnostic.corpus import CorpusRecipe, assert_source_disjoint, prepare_corpus
from kvpress.query_agnostic.data import file_hash
from kvpress.query_agnostic.reconstruction import ReconstructionCollector, collect_reconstruction, reconstruction_peak
from kvpress.query_agnostic.scorers import OnlineScorer, QueryMoments
from kvpress.query_agnostic.train_gates import (
    GateTrainingConfig,
    exported_state,
    gate_loss,
    learning_rate_multiplier,
    train_gates,
    training_prediction,
)

WORKSPACE = Path(__file__).resolve().parents[2]
AUTHOR = WORKSPACE / "marginal-attention"


@pytest.fixture(scope="module", autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def model():
    torch.manual_seed(5)
    config = Qwen2Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        rope_theta=10000000,
    )
    return AutoModelForCausalLM.from_config(config).eval()


def author_label_accumulator():
    if not AUTHOR.exists():
        pytest.skip("HM author checkout is not available.")
    # Execute the exact tracked function bodies without importing the paper's unrelated
    # extraction dependencies or its Transformers 5 environment.
    tree = ast.parse((AUTHOR / "src/aspace/kv/features.py").read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ("label_accumulator", "kv_group_log")
    ]
    namespace = {"torch": torch, "np": np, "Callable": Callable}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(AUTHOR / "src/aspace/kv/features.py"), "exec"),
        namespace,
    )
    return namespace["label_accumulator"]


@torch.inference_mode()
def test_hm_streamed_labels_match_actual_author_accumulator(model, monkeypatch):
    factory = author_label_accumulator()
    modules = [layer.self_attn for layer in model.model.layers]
    prefix, body = torch.tensor([1, 2]), torch.arange(3, 15)
    n_context = len(prefix) + len(body)
    callback, take = factory(modules=modules, n_heads=4, n_kv=2, n_context=n_context, device="cpu")
    original = ReconstructionCollector.__call__

    def compare(self, module, hidden, queries, heads, positions, phase):
        if phase == "question":
            keys = torch.stack([h.keys for h in heads]).repeat_interleave(2, dim=0)
            logits = (queries @ keys.transpose(-1, -2)) * queries.shape[-1] ** -0.5
            logits.masked_fill_(heads[0].positions[None, None] > positions[None, :, None], -float("inf"))
            weights = logits.softmax(-1).unsqueeze(0)
            layers = [None] * 2
            layers[module.layer_idx] = SimpleNamespace(values=torch.stack([h.values for h in heads]).unsqueeze(0))
            callback("repeat", module.layer_idx, weights, {"hidden_states": hidden}, SimpleNamespace(layers=layers))
        original(self, module, hidden, queries, heads, positions, phase)

    monkeypatch.setattr(ReconstructionCollector, "__call__", compare)
    actual, _ = collect_reconstruction(
        model,
        SyntheticTokenizer(),
        body,
        prefix,
        torch.empty(0, dtype=torch.long),
        "hm",
        prefill_chunk=5,
        repeat_chunk=5,
        query_tile=16,
        key_tile=7,
    )
    np.testing.assert_allclose(actual, take(), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("normalized", [False, True])
def test_tiled_reconstruction_peak_matches_dense_softmax(normalized):
    torch.manual_seed(7)
    q, k = torch.randn(7, 11, 8), torch.randn(23, 8)
    qp, kp = torch.arange(12, 23), torch.arange(23)
    hidden = torch.rand(11) + 0.1 if normalized else None
    value = torch.rand(7, 12) if normalized else None
    logits = q @ k.T / 8**0.5
    logits.masked_fill_(kp[None, None] > qp[None, :, None], -float("inf"))
    attention = logits.softmax(-1)[..., :12]
    if normalized:
        attention = attention / hidden[None, :, None] * value[:, None]
    expected = attention.amax((0, 1))
    actual = reconstruction_peak(q, k, qp, kp, 0, 12, hidden, value, query_tile=3, key_tile=5)
    torch.testing.assert_close(actual, expected)


@torch.inference_mode()
def test_ea_matches_registry_epsilon_and_original_score(model):
    n = 17
    keys, values = torch.randn(1, 2, n, 8), torch.randn(1, 2, n, 8)
    from kvpress.query_agnostic.cache import HeadCache

    moments = QueryMoments()
    moments.update(torch.randn(4, 19, 8))
    scorer = OnlineScorer("ea_once", 2, 4, 2, model.model.rotary_emb)
    scorer.body_start, scorer.body_sink = 2, 1
    scorer.moments[0] = moments
    module = model.model.layers[0].self_attn
    module.rotary_emb = model.model.rotary_emb
    press = ExpectedAttentionPress(n_sink=3, epsilon=1e-2)
    press.get_query_statistics = lambda module, hidden: press.apply_avg_rope(
        module, moments.mean.unsqueeze(0), (moments.m2 / moments.count).unsqueeze(0), n
    )
    expected = press.score(module, torch.empty(1, n, 32), keys, values, None, {})[0]
    for head in range(2):
        cache = HeadCache(keys[0, head], values[0, head], torch.arange(n), torch.zeros(n, dtype=torch.long))
        actual = scorer.score(0, head, cache, n)
        torch.testing.assert_close(actual[3:], expected[head, 3:], rtol=2e-5, atol=2e-6)
    del module.rotary_emb


def test_qfilter_svd_matches_author_majority_sign_and_group_mean():
    torch.manual_seed(1)
    q = torch.randn(4, 25, 8) + torch.arange(8).float()
    u, _, vh = torch.linalg.svd(q.float())
    sign = ((u[..., 0] > 0).float().mean(-1) > 0.5).float() * 2 - 1
    expected = (-sign[:, None] * vh[..., 0, :]).reshape(2, 2, 8).mean(-2)
    torch.testing.assert_close(qfilters_from_samples(q, 2), expected)


def test_gate_export_folds_double_bias_and_preserves_gradient():
    torch.manual_seed(3)
    gate = FastKVzipGate(0, 12, 2, 3, torch.float32, output_dim=4, sink=2)
    gate.b.data.normal_()
    x = torch.randn(19, 12)
    prediction = training_prediction(gate, x)
    prediction.sum().backward()
    assert gate.b.grad.abs().sum() > 0
    deployed = FastKVzipGate(0, 12, 2, 3, torch.float32, output_dim=4, sink=2)
    deployed.load_state_dict(exported_state(gate))
    torch.testing.assert_close(deployed(x.unsqueeze(0))[0], prediction)
    expected = (
        torch.nn.functional.binary_cross_entropy(prediction, torch.ones_like(prediction), reduction="none")
        .mean(-1)
        .sum()
    )
    torch.testing.assert_close(gate_loss(prediction, torch.ones_like(prediction)), expected)


def test_author_learning_rate_schedule_boundaries():
    assert [learning_rate_multiplier(i, 5000) for i in (0, 49, 50, 2499, 2500, 3750)] == [0, 0.49, 1, 1, 0.5, 0.25]


def small_corpus(tmp_path):
    recipe = CorpusRecipe(
        hm_documents=2,
        hm_length=12,
        qfilter_documents=2,
        qfilter_length=8,
        gate_short_tokens=32,
        gate_long_tokens=32,
        validation_short_tokens=16,
        validation_long_tokens=16,
        short_min=8,
        short_max=12,
        long_length=16,
    )
    rows = ({"id": str(i), "text": f"Source {i}: " + "abc def " * 12} for i in range(100))
    return prepare_corpus(
        rows,
        SyntheticTokenizer(),
        tmp_path,
        recipe,
        {"kind": "synthetic"},
        {"model_id": "synthetic/tiny-Qwen2", "model_revision": "seed-5"},
    )


def test_corpus_source_split_and_exact_token_totals(tmp_path):
    manifest = small_corpus(tmp_path)
    documents = manifest["documents"]
    assert_source_disjoint(documents)
    assert sum(d["body_tokens"] for d in documents if d["split"] == "train") == 64
    assert sum(d["body_tokens"] for d in documents if d["split"] == "validation") == 32
    documents.append(dict(documents[0], document_id="leaked-source", split="validation"))
    with pytest.raises(ValueError, match="source document"):
        assert_source_disjoint(documents)


def test_offline_stages_and_resume_with_synthetic_model(model, tmp_path):
    if not AUTHOR.exists():
        pytest.skip("HM author checkout is unavailable.")
    manifest = small_corpus(tmp_path)
    manifest_path = tmp_path / "calibration_manifest.json"
    hm = collect_labels(model, SyntheticTokenizer(), tmp_path, "hm", prefill_chunk=8, repeat_chunk=8)
    gate = collect_labels(model, SyntheticTokenizer(), tmp_path, "gate", prefill_chunk=8, repeat_chunk=8)
    qfilter = collect_qfilter_samples(model, tmp_path, samples=8)
    output = tmp_path / "artifacts"
    output.mkdir()
    build_hm_profiles(hm, manifest_path, AUTHOR, output / "hm_profiles.npz")
    build_qfilters(qfilter, manifest_path, output / "qfilters.safetensors")
    primitive = load_hm_primitives(AUTHOR)
    store = json.loads(hm.read_text())
    for budget in (4096, 8192, 16384, 32768):
        raw = []
        for record in store["documents"]:
            labels = np.load(hm.parent / record["labels_file"])
            raw.append(primitive.document_shares(labels.reshape(4, -1).T, compression=1 - budget / 131072))
        expected = primitive.solve_scale(np.mean(raw, axis=0), mean_keep=budget / 131072)["keep"].reshape(2, 2)
        with np.load(output / "hm_profiles.npz") as profile:
            np.testing.assert_allclose(profile[f"shares_{budget}"], expected, rtol=0, atol=0)
    config = GateTrainingConfig(steps=4, batch_size=16, validation_interval=2, dimension=4, sinks=2)
    trained = train_gates(gate, manifest_path, output / "gates/gates.pt", config, "cpu")
    first = file_hash(output / "gates/gates.pt")
    train_gates(gate, manifest_path, output / "gates/gates.pt", config, "cpu")
    assert len(trained["layers"]) == 2
    assert first == file_hash(output / "gates/gates.pt")
    states = torch.load(output / "gates/gates.pt", weights_only=True)["layers"]
    gates = [FastKVzipGate(i, 32, 2, 2, torch.float32, output_dim=4, sink=2) for i in range(2)]
    for gate_module, state in zip(gates, states):
        gate_module.load_state_dict(state)
    with np.load(output / "hm_profiles.npz") as profiles:
        shares = profiles["shares_4096"].copy()
    with QueryAgnosticRuntime(
        model, ExperimentConfig(method="fastkvzip", budget=6, body_sink=1, chunk_size=8), shares=shares, gates=gates
    ) as runtime:
        runtime.prefill(torch.arange(3, 27), torch.tensor([1, 2]))
        answer = runtime.answer(torch.tensor([80, 81]), max_new_tokens=3)
        assert len(answer) == 3
        assert sum(h.length for row in runtime.cache for h in row) == 24
    # Development artifacts identify the random target and cannot pass production provenance checks.
    with pytest.raises(ValueError, match="model_id"):
        artifact_metadata(output / "hm_profiles.npz", "hm")
    # Completed label documents are reused without executing model forwards again.
    old_forward = model.model.forward
    model.model.forward = lambda *a, **k: (_ for _ in ()).throw(AssertionError("unexpected recomputation"))
    try:
        collect_labels(model, SyntheticTokenizer(), tmp_path, "hm", prefill_chunk=8, repeat_chunk=8)
    finally:
        model.model.forward = old_forward
    assert manifest["model_id"].startswith("synthetic/")


def test_gate_local_labels_match_official_kvzip_score():
    path = WORKSPACE / "data/kv_query_agnostic_2026-11-10/repos/KVzip/attention/score.py"
    if not path.exists():
        pytest.skip("Original KVzip score source is unavailable.")
    spec = importlib.util.spec_from_file_location("offline_test_official_kvzip", path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    torch.manual_seed(2)
    n_context, q_length, sink, start, end = 19, 7, 3, 8, 14
    q, k = torch.randn(1, 4, q_length, 8), torch.randn(1, 2, n_context + q_length, 8)
    scorer = original.KVScore()
    scorer.n_layers, scorer.n_heads_kv, scorer.dtype, scorer.device = 1, 2, torch.float32, "cpu"
    scorer.sink, scorer.start_idx, scorer.end_idx = sink, start, end
    scorer.init_score()
    scorer._get_score(q, k, 0)
    for head in range(2):
        indices = torch.cat(
            [torch.arange(sink), torch.arange(start, end), torch.arange(n_context, n_context + q_length)]
        )
        actual = reconstruction_peak(
            q[0, head * 2 : head * 2 + 2],
            k[0, head, indices],
            torch.arange(n_context, n_context + q_length),
            indices,
            sink,
            sink + end - start,
            query_tile=3,
            key_tile=5,
        )
        torch.testing.assert_close(actual, scorer.score[0][0, head])


def test_training_resumes_optimizer_rng_and_batch_order_exactly(model, tmp_path, monkeypatch):
    import kvpress.query_agnostic.train_gates as trainer

    small_corpus(tmp_path)
    store = collect_labels(model, SyntheticTokenizer(), tmp_path, "gate", prefill_chunk=8, repeat_chunk=8)
    manifest = tmp_path / "calibration_manifest.json"
    config = GateTrainingConfig(steps=6, batch_size=16, validation_interval=3, dimension=4, sinks=2)
    original_save = trainer.atomic_torch_save
    interrupted = False

    def stop_after_checkpoint(path, value):
        nonlocal interrupted
        original_save(path, value)
        if path.name.endswith("checkpoint.pt") and not interrupted:
            interrupted = True
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(trainer, "atomic_torch_save", stop_after_checkpoint)
    output = tmp_path / "resumed/gates.pt"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_gates(store, manifest, output, config, "cpu")
    monkeypatch.setattr(trainer, "atomic_torch_save", original_save)
    resumed = train_gates(store, manifest, output, config, "cpu")
    direct_output = tmp_path / "uninterrupted/gates.pt"
    direct = train_gates(store, manifest, direct_output, config, "cpu")
    a = torch.load(output, weights_only=True)["layers"]
    b = torch.load(direct_output, weights_only=True)["layers"]
    for left, right in zip(a, b):
        for name in left:
            torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
    assert [layer["history"] for layer in resumed["layers"]] == [layer["history"] for layer in direct["layers"]]


def test_long_calibration_windows_concatenate_without_source_reuse(tmp_path):
    recipe = CorpusRecipe(
        hm_documents=0,
        qfilter_documents=0,
        gate_short_tokens=0,
        gate_long_tokens=32,
        validation_short_tokens=0,
        validation_long_tokens=16,
        short_min=8,
        short_max=12,
        long_length=16,
    )
    rows = ({"id": str(i), "text": f"x{i:02d}"} for i in range(50))
    manifest = prepare_corpus(rows, SyntheticTokenizer(), tmp_path, recipe, {"kind": "synthetic"})
    assert len(manifest["documents"]) == 3
    assert all(document["construction"] == "concatenated_sources" for document in manifest["documents"])
    assert all(document["body_tokens"] == 16 for document in manifest["documents"])
    assert_source_disjoint(manifest["documents"])
