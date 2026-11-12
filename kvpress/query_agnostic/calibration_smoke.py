# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline end-to-end check with a RANDOM tiny Qwen2; produces development artifacts only."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, Qwen2Config

from kvpress.query_agnostic.calibrate import collect_labels, collect_qfilter_samples
from kvpress.query_agnostic.calibration_ops import author_sources, build_hm_profiles, build_qfilters
from kvpress.query_agnostic.corpus import CorpusRecipe, prepare_corpus
from kvpress.query_agnostic.train_gates import GateTrainingConfig, train_gates


class SyntheticTokenizer:
    def encode(self, text, **kwargs):
        return [ord(character) % 90 + 1 for character in text]

    def apply_chat_template(self, messages, **kwargs):
        return "USR:" + messages[-1]["content"] + "\nASSISTANT:"


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--work-dir", type=Path, required=True)
    cli.add_argument("--author-repo", type=Path, default=Path(__file__).resolve().parents[3] / "marginal-attention")
    cli.add_argument("--device", default="cpu")
    cli.add_argument("--backend", choices=("sdpa", "flash_varlen"), default="sdpa")
    args = cli.parse_args(argv)
    torch.set_num_threads(2)
    torch.manual_seed(0)
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
    model = AutoModelForCausalLM.from_config(config)
    dtype = torch.bfloat16 if str(args.device).startswith("cuda") else torch.float32
    model = model.to(device=args.device, dtype=dtype).eval()
    tokenizer = SyntheticTokenizer()
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
    root = args.work_dir
    manifest_path = root / "calibration_manifest.json"
    if not manifest_path.exists():
        rows = ({"id": str(i), "text": f"Synthetic source {i}: " + "a b c d e f " * 5} for i in range(100))
        prepare_corpus(
            rows,
            tokenizer,
            root,
            recipe,
            {"kind": "synthetic_random_model"},
            {"model_id": "synthetic/tiny-Qwen2", "model_revision": "seed-0"},
        )
    hm_store = collect_labels(
        model, tokenizer, root, "hm", args.backend, 8, 8, sources=author_sources(args.author_repo)
    )
    q_store = collect_qfilter_samples(model, root, args.backend, samples=8)
    gate_store = collect_labels(model, tokenizer, root, "gate", args.backend, 8, 8)
    del model
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    build_hm_profiles(hm_store, manifest_path, args.author_repo, artifacts / "hm_profiles.npz")
    build_qfilters(q_store, manifest_path, artifacts / "qfilters.safetensors", args.device)
    result = train_gates(
        gate_store,
        manifest_path,
        artifacts / "gates/gates.pt",
        GateTrainingConfig(steps=20, batch_size=16, validation_interval=10, dimension=4, sinks=2),
        args.device,
    )
    print(
        json.dumps(
            {"synthetic_random_model": True, "artifacts": str(artifacts), "layers": len(result["layers"])}, indent=2
        )
    )


if __name__ == "__main__":
    main()
