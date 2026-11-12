# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise all runtime paths using a tiny RANDOM model; no pretrained weights are downloaded."""

import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, Qwen2Config

from kvpress.presses.fastkvzip_press import FastKVzipGate
from kvpress.query_agnostic import ExperimentConfig, QueryAgnosticRuntime
from kvpress.query_agnostic.scorers import METHODS


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--device", default="cpu")
    cli.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    cli.add_argument("--backend", choices=("sdpa", "flash_varlen"), default="sdpa")
    args = cli.parse_args(argv)
    torch.manual_seed(0)
    torch.set_num_threads(2)
    config = Qwen2Config(
        vocab_size=101,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        rope_theta=10000000,
        eos_token_id=100,
    )
    model = AutoModelForCausalLM.from_config(config)
    model = model.to(device=args.device, dtype=getattr(torch, args.dtype)).eval()
    ids = torch.arange(80, device=model.device, dtype=torch.long) % 90 + 1
    rows = []
    for method in METHODS:
        gates = [FastKVzipGate(i, 64, 2, 2, model.dtype).to(model.device).eval() for i in range(2)]
        experiment = ExperimentConfig(
            method=method,
            budget=12,
            chunk_size=16,
            backend=args.backend,
            body_sink=2,
            history_capacity=8,
            scoring_probes=2,
            diagnostics=True,
        )
        with QueryAgnosticRuntime(
            model,
            experiment,
            shares=np.array([[4, 2], [1, 1]]),
            filters=torch.randn(2, 2, 16),
            gates=gates,
        ) as runtime:
            runtime.prefill(ids[3:75], ids[:3], document_id="synthetic-smoke-only")
            first = runtime.answer(ids[75:], max_new_tokens=5)
            second = runtime.answer(ids[75:], max_new_tokens=5)
            if not torch.equal(first, second):
                raise RuntimeError(f"Question rollback failed for {method}.")
            rows.append(
                {
                    "method": method,
                    "kv_entries": sum(head.length for row in runtime.cache for head in row),
                    "kv_bytes": runtime.kv_bytes,
                    "auxiliary_bytes": runtime.auxiliary_bytes,
                    "prefill_seconds": runtime.prefill_seconds,
                    "generated_ids": first.tolist(),
                }
            )
    print(
        json.dumps(
            {"synthetic_random_model": True, "device": str(model.device), "dtype": args.dtype, "methods": rows},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
