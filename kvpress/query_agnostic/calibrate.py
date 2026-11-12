# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline pipeline: frozen corpus -> reconstruction/features -> HM, filters and gates."""

import argparse
import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvpress.query_agnostic import ExperimentConfig, QueryAgnosticRuntime
from kvpress.query_agnostic.calibration_ops import author_sources, build_hm_profiles, build_qfilters
from kvpress.query_agnostic.corpus import CorpusRecipe, assert_source_disjoint, prepare_corpus
from kvpress.query_agnostic.data import MODEL_ID, MODEL_REVISION, file_hash, sha256_bytes
from kvpress.query_agnostic.download import verify_snapshot
from kvpress.query_agnostic.reconstruction import collect_reconstruction
from kvpress.query_agnostic.run import atomic_json, code_hash, execution_environment
from kvpress.query_agnostic.train_gates import GateTrainingConfig, train_gates


def frozen_json(path: Path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"Configuration changed: {path}. Use a new work directory.")
    atomic_json(path, value)


def calibration_segments(tokenizer, role: str):
    marker = "__OFFLINE_BODY_BOUNDARY_0CF917__"
    if role == "qfilters":
        return [], []
    if role == "hm":
        # Author features.prefill_text uses user-only chat templating; the tokenizer may
        # insert its own default system message. Keep exactly the part preceding the body.
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": marker}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prefix = rendered.split(marker)[0]
        return tokenizer.encode(prefix, add_special_tokens=False), []
    # The official FastKVzip Qwen template, including its offline generic instruction.
    prefix = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    prefix += "Given the context, answer to the following question or request without explanation.\n\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n"
    return [tokenizer.encode(text, add_special_tokens=False) for text in (prefix, suffix)]


def read_body(root: Path, record: dict) -> torch.Tensor:
    path = root / record["token_file"]
    if file_hash(path) != record["token_file_sha256"]:
        raise ValueError(f"Frozen calibration tokens changed: {path}")
    ids = np.load(path, allow_pickle=False)
    if ids.ndim != 1 or len(ids) != record["body_tokens"]:
        raise ValueError("Calibration token shape does not match its manifest.")
    return torch.from_numpy(ids.astype(np.int64))


def collect_labels(
    model,
    tokenizer,
    root: Path,
    mode: str,
    backend="sdpa",
    prefill_chunk=2048,
    repeat_chunk=None,
    query_tile=128,
    key_tile=4096,
    sources=None,
):
    manifest_path = root / "calibration_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert_source_disjoint(manifest["documents"])
    if mode not in ("hm", "gate"):
        raise ValueError("Choose hm or gate labels explicitly.")
    records = [
        d for d in manifest["documents"] if (d["role"] == "hm" if mode == "hm" else d["role"].startswith("gate_"))
    ]
    if not records:
        raise ValueError("No documents for this calibration stage.")
    directory = root / ("hm_labels" if mode == "hm" else "gate_features")
    directory.mkdir(exist_ok=True)
    configuration = {
        "calibration_manifest_sha256": file_hash(manifest_path),
        "mode": mode,
        "backend": backend,
        "prefill_chunk": prefill_chunk,
        "repeat_chunk": repeat_chunk or (512 if mode == "hm" else 2000),
        "query_tile": query_tile,
        "key_tile": key_tile,
        "source_sha256": code_hash(),
        "environment": execution_environment(str(model.device), backend),
        "author_sources": sources,
    }
    frozen_json(directory / "config.json", configuration)
    config_hash = sha256_bytes(json.dumps(configuration, sort_keys=True).encode())
    prefix, suffix = calibration_segments(tokenizer, mode)
    completed = []
    for record in records:
        location = directory / record["document_id"]
        location.mkdir(exist_ok=True)
        done = location / "complete.json"
        if done.exists():
            result = json.loads(done.read_text())
            if result["config_hash"] != config_hash:
                raise ValueError("Completed calibration document has a different configuration.")
            items = result.get("features", []) + [{"file": result["labels_file"], "sha256": result["labels_sha256"]}]
            if any(file_hash(directory / item["file"]) != item["sha256"] for item in items):
                raise ValueError("A completed calibration file has changed.")
            completed.append(result)
            continue
        started = time.perf_counter()
        try:
            if model.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(model.device)
            labels, encoding = collect_reconstruction(
                model,
                tokenizer,
                read_body(root, record),
                torch.tensor(prefix, dtype=torch.long),
                torch.tensor(suffix, dtype=torch.long),
                mode,
                backend,
                prefill_chunk,
                repeat_chunk,
                location / "features" if mode == "gate" else None,
                query_tile,
                key_tile,
            )
            labels_path = location / "labels.npy"
            np.save(labels_path, labels, allow_pickle=False)
            result = {
                **record,
                "config_hash": config_hash,
                "labels_file": str(labels_path.relative_to(directory)),
                "labels_sha256": file_hash(labels_path),
                "prefix_tokens": len(prefix),
                "feature_encoding": encoding,
                "seconds": time.perf_counter() - started,
                "peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated(model.device) if model.device.type == "cuda" else None
                ),
            }
            if mode == "gate":
                result["features"] = [
                    {"file": str(p.relative_to(directory)), "sha256": file_hash(p)}
                    for p in sorted((location / "features").glob("layer_*.npy"))
                ]
            atomic_json(done, result)
            location.joinpath("error.json").unlink(missing_ok=True)
            completed.append(result)
            print(f"Collected {mode} {record['document_id']}: {record['body_tokens']} body tokens", flush=True)
        except Exception:
            atomic_json(location / "error.json", {"config_hash": config_hash, "error": traceback.format_exc()})
            raise
    store = {
        "kind": "hm_labels" if mode == "hm" else "gate_features",
        "model_id": manifest["model_id"],
        "model_revision": manifest["model_revision"],
        "calibration_manifest_sha256": file_hash(manifest_path),
        "documents": completed,
        "layers": model.config.num_hidden_layers,
        "kv_heads": model.config.num_key_value_heads,
        "query_heads": model.config.num_attention_heads,
        "hidden_size": model.config.hidden_size,
        "configuration": configuration,
    }
    atomic_json(directory / "store.json", store)
    return directory / "store.json"


@torch.inference_mode()
def collect_qfilter_samples(model, root: Path, backend="sdpa", samples=3000, seed=0):
    manifest_path = root / "calibration_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records = [d for d in manifest["documents"] if d["role"] == "qfilters"]
    total = sum(d["body_tokens"] for d in records)
    if samples < 1 or samples > total:
        raise ValueError("Q-filter sampling needs the full requested number of query vectors.")
    directory = root / "qfilter_samples"
    directory.mkdir(exist_ok=True)
    configuration = {
        "calibration_manifest_sha256": file_hash(manifest_path),
        "samples": samples,
        "seed": seed,
        "source_sha256": code_hash(),
        "environment": execution_environment(str(model.device), backend),
    }
    frozen_json(directory / "config.json", configuration)
    store_path = directory / "store.json"
    if store_path.exists():
        existing = json.loads(store_path.read_text())
        if existing["configuration"] != configuration:
            raise ValueError("Q-filter collection configuration changed.")
        if any(file_hash(directory / item["file"]) != item["sha256"] for item in existing["layers"]):
            raise ValueError("Completed Q-filter sample file changed.")
        return store_path
    layers, q_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    dimension = model.config.hidden_size // q_heads
    selected, data = [], []
    for layer in range(layers):
        generator = torch.Generator().manual_seed(seed + layer)
        selected.append(torch.randperm(total, generator=generator)[:samples].sort().values)
        data.append(torch.empty(q_heads, samples, dimension, dtype=model.dtype))
    offset = 0

    def observe(module, hidden, queries, heads, positions, phase):
        if phase != "body":
            return
        indices = selected[module.layer_idx]
        start = offset + int(positions[0])
        mask = (indices >= start) & (indices < start + len(positions))
        local = (indices[mask] - start).to(queries.device)
        data[module.layer_idx][:, mask] = queries[:, local].cpu()

    config = ExperimentConfig(method="full", backend=backend, chunk_size=2048)
    with QueryAgnosticRuntime(model, config, observer=observe) as runtime:
        for record in records:
            runtime.prefill(read_body(root, record), document_id=record["document_id"])
            offset += record["body_tokens"]
    files = []
    for layer, queries in enumerate(data):
        path = directory / f"layer_{layer:02d}.pt"
        torch.save(queries, path)
        files.append({"file": path.name, "sha256": file_hash(path)})
    store = {
        "kind": "qfilter_samples",
        "model_id": manifest["model_id"],
        "model_revision": manifest["model_revision"],
        "calibration_manifest_sha256": file_hash(manifest_path),
        "layers": files,
        "samples_per_head": samples,
        "kv_heads": model.config.num_key_value_heads,
        "sampled_token_indices": [indices.tolist() for indices in selected],
        "configuration": configuration,
    }
    atomic_json(directory / "store.json", store)
    return directory / "store.json"


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "stage",
        choices=(
            "prepare",
            "hm-labels",
            "hm-build",
            "qfilters-collect",
            "qfilters-build",
            "gate-collect",
            "gate-train",
            "all",
        ),
    )
    cli.add_argument("--work-dir", type=Path, required=True)
    cli.add_argument("--model", default=MODEL_ID)
    cli.add_argument("--author-repo", type=Path, default=Path(__file__).resolve().parents[3] / "marginal-attention")
    cli.add_argument("--source-jsonl", type=Path)
    cli.add_argument(
        "--dataset-revision", help="Optional pinned FineWeb-Edu commit; otherwise resolve once and freeze."
    )
    cli.add_argument("--device", default="cuda")
    cli.add_argument("--backend", choices=("sdpa", "flash_varlen"), default="flash_varlen")
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--prefill-chunk", type=int, default=2048)
    cli.add_argument("--score-query-tile", type=int, default=128)
    cli.add_argument("--score-key-tile", type=int, default=4096)
    args = cli.parse_args(argv)
    root = args.work_dir
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "calibration_manifest.json"
    target = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION}
    if Path(args.model).is_dir():
        verify_snapshot(
            Path(args.model), require_weights=args.stage in ("all", "hm-labels", "qfilters-collect", "gate-collect")
        )
    elif args.model != MODEL_ID:
        cli.error("Use the frozen target checkpoint or its verified local snapshot.")
    if args.stage in ("prepare", "all") and not manifest_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=MODEL_REVISION, use_fast=True)
        if args.source_jsonl is not None:

            def rows():
                with args.source_jsonl.open() as stream:
                    for line in stream:
                        yield json.loads(line)

            source = {"kind": "local_jsonl", "sha256": file_hash(args.source_jsonl)}
            iterator = rows()
        else:
            from datasets import load_dataset
            from huggingface_hub import HfApi

            revision = HfApi().dataset_info("HuggingFaceFW/fineweb-edu", revision=args.dataset_revision).sha
            source = {
                "kind": "huggingface",
                "dataset": "HuggingFaceFW/fineweb-edu",
                "config": "sample-10BT",
                "revision": revision,
            }
            iterator = iter(
                load_dataset(source["dataset"], name=source["config"], revision=revision, split="train", streaming=True)
            )
        prepare_corpus(iterator, tokenizer, root, CorpusRecipe(seed=args.seed), source, target)
    if not manifest_path.exists():
        cli.error("Run prepare before this stage.")
    manifest = json.loads(manifest_path.read_text())
    if any(manifest.get(key) != value for key, value in target.items()):
        raise ValueError("Calibration manifest belongs to another checkpoint.")
    if manifest["recipe"] != asdict(CorpusRecipe(seed=args.seed)):
        raise ValueError("Production CLI requires the frozen main-experiment corpus recipe.")
    if args.source_jsonl and manifest["source"].get("sha256") != file_hash(args.source_jsonl):
        raise ValueError("Source JSONL differs from the frozen calibration corpus.")
    if args.dataset_revision and manifest["source"].get("revision") != args.dataset_revision:
        raise ValueError("Use the exact dataset commit recorded in the frozen calibration manifest.")
    if args.stage == "prepare":
        return
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    model_stages = ("hm-labels", "qfilters-collect", "gate-collect", "all")
    if args.stage in model_stages:
        sources = author_sources(args.author_repo) if args.stage in ("hm-labels", "all") else None
        dtype = torch.bfloat16 if str(args.device).startswith("cuda") else torch.float32
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model, revision=MODEL_REVISION, torch_dtype=dtype, attn_implementation="sdpa"
            )
            .to(args.device)
            .eval()
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=MODEL_REVISION, use_fast=True)
        if (
            model.config.num_hidden_layers,
            model.config.num_key_value_heads,
            model.config.num_attention_heads,
            model.config.rope_theta,
        ) != (28, 4, 28, 10000000):
            raise ValueError("Calibration requires the frozen target model geometry and RoPE.")
        for stage in ("hm-labels", "qfilters-collect", "gate-collect") if args.stage == "all" else (args.stage,):
            if stage == "qfilters-collect":
                collect_qfilter_samples(model, root, args.backend, seed=args.seed)
            else:
                collect_labels(
                    model,
                    tokenizer,
                    root,
                    "hm" if stage == "hm-labels" else "gate",
                    args.backend,
                    args.prefill_chunk,
                    query_tile=args.score_query_tile,
                    key_tile=args.score_key_tile,
                    sources=sources if stage == "hm-labels" else None,
                )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if args.stage in ("hm-build", "all"):
        build_hm_profiles(root / "hm_labels/store.json", manifest_path, args.author_repo, artifacts / "hm_profiles.npz")
    if args.stage in ("qfilters-build", "all"):
        build_qfilters(
            root / "qfilter_samples/store.json", manifest_path, artifacts / "qfilters.safetensors", args.device
        )
    if args.stage in ("gate-train", "all"):
        train_gates(
            root / "gate_features/store.json",
            manifest_path,
            artifacts / "gates/gates.pt",
            GateTrainingConfig(seed=args.seed),
            args.device,
        )


if __name__ == "__main__":
    main()
