# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SCBench runner. Example: python -m kvpress.query_agnostic.run --help."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvpress.query_agnostic import ExperimentConfig, QueryAgnosticRuntime
from kvpress.query_agnostic.artifacts import artifact_metadata, load_filters, load_gates, load_hm
from kvpress.query_agnostic.data import (
    LIMITS,
    MODEL_ID,
    MODEL_REVISION,
    TASKS,
    file_hash,
    load_documents,
    sha256_bytes,
    tokenize_document,
)
from kvpress.query_agnostic.download import verify_snapshot
from kvpress.query_agnostic.scorers import METHODS


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def code_hash() -> str:
    sources = sorted(Path(__file__).parent.glob("*.py"))
    sources.append(Path(__file__).parent.parent / "presses/fastkvzip_press.py")
    return sha256_bytes("".join(file_hash(p) for p in sources).encode())


def execution_environment(device: str, backend: str) -> dict:
    target = torch.device(device)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(target) if target.type == "cuda" else None,
        "backend": backend,
        "flash_attn": importlib.metadata.version("flash-attn") if backend == "flash_varlen" else None,
        "worker_concurrency": 1,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    if target.type == "cuda":
        environment["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(target).total_memory
        try:
            environment["nvidia_driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True, timeout=10
            ).strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            environment["nvidia_driver"] = None
    return environment


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", default=MODEL_ID, help="Target repo ID or a local snapshot of the frozen revision.")
    cli.add_argument("--dataset-dir", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True, help="One directory per complete method/budget/config.")
    cli.add_argument("--method", choices=METHODS, default="keydiff")
    cli.add_argument("--budget", type=int, choices=(4096, 8192, 16384), default=8192)
    cli.add_argument("--allocation", choices=("hm", "uniform"), default="hm")
    cli.add_argument("--hm-profile", type=Path)
    cli.add_argument("--filters", type=Path)
    cli.add_argument("--gates", type=Path)
    cli.add_argument("--backend", choices=("sdpa", "flash_varlen"), default="flash_varlen")
    cli.add_argument("--device", default="cuda")
    cli.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    cli.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    cli.add_argument("--limit", type=int, default=0, help="Per-task document limit for validation; 0 means all.")
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--diagnostics", action="store_true")
    return cli


def main(argv=None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    if args.limit < 0:
        cli.error("--limit cannot be negative.")
    if args.method != "full" and args.allocation == "hm" and args.hm_profile is None:
        cli.error("HM allocation requires --hm-profile. Use --allocation uniform explicitly for development.")
    if args.method == "qfilters" and args.filters is None:
        cli.error("qfilters requires --filters.")
    if args.method == "fastkvzip" and args.gates is None:
        cli.error("fastkvzip requires --gates.")
    for path, kind in ((args.hm_profile, "hm"), (args.filters, "qfilters"), (args.gates, "fastkvzip")):
        if path is not None:
            artifact_metadata(path, kind)
    local_snapshot = None
    if Path(args.model).is_dir():
        local_snapshot = verify_snapshot(Path(args.model))
    elif args.model != MODEL_ID:
        cli.error("--model must name the frozen target or a verified local snapshot created by the download module.")
    if args.backend == "flash_varlen":
        if not str(args.device).startswith("cuda") or args.dtype != "bfloat16":
            cli.error("flash_varlen requires CUDA and BF16 in this experiment.")
        importlib.metadata.version("flash-attn")  # Fail before loading weights if the backend is missing.
    torch.manual_seed(args.seed)
    config = ExperimentConfig(
        method=args.method, budget=args.budget, backend=args.backend, seed=args.seed, diagnostics=args.diagnostics
    )
    environment = execution_environment(args.device, args.backend)
    signature = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "local_snapshot": local_snapshot,
        "execution_environment": environment.copy(),
        "runtime": asdict(config),
        "allocation": args.allocation,
        "dtype": args.dtype,
        "tasks": args.tasks,
        "limit": args.limit,
        "source_sha256": code_hash(),
        "datasets": {task: file_hash(args.dataset_dir / f"{task}.parquet") for task in args.tasks},
        "dataset_counts": {
            task: pq.read_metadata(args.dataset_dir / f"{task}.parquet").num_rows for task in args.tasks
        },
        "artifacts": {
            kind: None if path is None else {"file": file_hash(path), "metadata": file_hash(path.with_suffix(".json"))}
            for kind, path in (("hm", args.hm_profile), ("qfilters", args.filters), ("fastkvzip", args.gates))
        },
    }
    signature["config_hash"] = sha256_bytes(json.dumps(signature, sort_keys=True).encode())
    args.output.mkdir(parents=True, exist_ok=True)
    config_file = args.output / "config.json"
    if config_file.exists() and json.loads(config_file.read_text()) != signature:
        raise ValueError("Output configuration differs; choose a separate output directory.")
    atomic_json(config_file, signature)
    for directory in ("documents", "tokens", "final_survivors"):
        (args.output / directory).mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=MODEL_REVISION, use_fast=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, revision=MODEL_REVISION, torch_dtype=getattr(torch, args.dtype), attn_implementation="sdpa"
        )
        .to(args.device)
        .eval()
    )
    if (model.config.num_hidden_layers, model.config.num_attention_heads, model.config.num_key_value_heads) != (
        28,
        28,
        4,
    ):
        raise ValueError("The production runner requires the frozen Qwen2.5-7B-Instruct-1M geometry.")
    if model.config.rope_theta != 10000000:
        raise ValueError("The target checkpoint must retain rope_theta=10000000.")
    shares = None
    if args.method != "full":
        shares = load_hm(args.hm_profile, args.budget) if args.allocation == "hm" else np.ones((28, 4))
    filters = None if args.filters is None else load_filters(args.filters)
    gates = None if args.gates is None else load_gates(args.gates, model)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    environment["commit"] = commit
    atomic_json(args.output / "environment.json", environment)
    manifests, failures = [], []
    with QueryAgnosticRuntime(model, config, shares=shares, filters=filters, gates=gates) as runtime:
        for document in load_documents(args.dataset_dir, args.tasks, args.limit):
            output_file = args.output / "documents" / f"{document.document_id}.json"
            if output_file.exists():
                previous = json.loads(output_file.read_text())
                if previous["config_hash"] == signature["config_hash"] and previous["status"] == "complete":
                    manifests.append(previous["manifest"])
                    continue
            try:
                encoded, manifest = tokenize_document(document, tokenizer)
                manifests.append(manifest)
                np.savez_compressed(
                    args.output / "tokens" / f"{document.document_id}.npz",
                    prefix=np.array(encoded["prefix"], dtype=np.int32),
                    body=np.array(encoded["body"], dtype=np.int32),
                    offsets=np.array(encoded["body_offsets"], dtype=np.int64),
                    **{f"question_{i}": np.array(q, dtype=np.int32) for i, q in enumerate(encoded["questions"])},
                )
                tensor = lambda ids: torch.tensor(ids, dtype=torch.long, device=model.device)  # noqa: E731
                runtime.reset(document.document_id)
                if model.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(model.device)
                runtime.prefill(tensor(encoded["body"]), tensor(encoded["prefix"]), document.document_id)
                result = {
                    "config_hash": signature["config_hash"],
                    "status": "partial",
                    "manifest": manifest,
                    "prefill_seconds": runtime.prefill_seconds,
                    "kv_bytes": runtime.kv_bytes,
                    "auxiliary_bytes": runtime.auxiliary_bytes,
                    "predictions": [],
                    "chunks": runtime.chunk_records,
                    "retention": runtime.retention_records,
                }
                partial_file = output_file.with_suffix(".partial.json")
                if partial_file.exists():
                    partial_result = json.loads(partial_file.read_text())
                    if partial_result["config_hash"] == signature["config_hash"]:
                        result["predictions"] = partial_result["predictions"]
                for index, question in enumerate(encoded["questions"]):
                    if index < len(result["predictions"]):
                        continue
                    runtime._synchronize()
                    start = time.perf_counter()
                    output = runtime.answer(tensor(question), max_new_tokens=LIMITS[document.task])
                    runtime._synchronize()
                    result["predictions"].append(
                        {
                            "question_index": index,
                            "question": document.questions[index],
                            "reference": document.references[index],
                            "prediction": tokenizer.decode(output.tolist(), skip_special_tokens=True),
                            "generated_ids": output.tolist(),
                            "seconds": time.perf_counter() - start,
                        }
                    )
                    atomic_json(partial_file, result)
                np.savez_compressed(
                    args.output / "final_survivors" / f"{document.document_id}.npz",
                    **{
                        f"layer_{layer}_head_{head}": cache.positions.cpu().numpy()
                        for layer, row in enumerate(runtime.cache)
                        for head, cache in enumerate(row)
                    },
                )
                result["peak_allocated_bytes"] = (
                    torch.cuda.max_memory_allocated(model.device) if model.device.type == "cuda" else None
                )
                result["peak_reserved_bytes"] = (
                    torch.cuda.max_memory_reserved(model.device) if model.device.type == "cuda" else None
                )
                result["status"] = "complete"
                atomic_json(output_file, result)
                partial_file.unlink(missing_ok=True)
                output_file.with_suffix(".error.json").unlink(missing_ok=True)
                print(f"Completed {document.document_id}: {manifest['body_tokens']} body tokens", flush=True)
            except Exception:
                failure = {"document_id": document.document_id, "error": traceback.format_exc()}
                failures.append(failure)
                atomic_json(output_file.with_suffix(".error.json"), failure)
                runtime.reset(document.document_id)
                if model.device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"Failed {document.document_id}: see {output_file.with_suffix('.error.json')}", flush=True)
    (args.output / "data_manifest.jsonl").write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifests)
    )
    atomic_json(args.output / "failures.json", failures)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
