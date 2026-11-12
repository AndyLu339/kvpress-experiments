# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Author HM primitives, Q-filter SVD, and provenance for generated offline artifacts."""

import importlib.util
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from kvpress.query_agnostic.data import KV_BUDGETS, REFERENCE_CONTEXT_LENGTH, file_hash
from kvpress.query_agnostic.run import atomic_json, code_hash, execution_environment

UPSTREAM_REFERENCES = {
    "qfilters": {
        "repository": "https://github.com/NathanGodey/qfilters",
        "commit": "cda5c38d3aa3f11815cf991661be80cb4f345815",
    },
    "fastkvzip": {
        "repository": "https://github.com/Janghyun1230/FastKVzip",
        "commit": "e04afaa6007c58d18500d0698d2d8ae592ff22c2",
    },
}


def author_sources(repository: Path) -> dict:
    relative_paths = (
        "src/aspace/kv/features.py",
        "exp/kv/budget/kvzip_shares.py",
        "exp/kv/budget/nemotron_shares.py",
        "exp/kv/evaluation/evaluate.py",
        "conf/exp/kv/budget/kv_nemotron_shares.yaml",
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    return {
        "repository": "https://github.com/Flegyas/marginal-attention",
        "commit": commit,
        "files": {name: file_hash(repository / name) for name in relative_paths},
    }


def load_hm_primitives(repository: Path):
    path = repository / "exp/kv/budget/kvzip_shares.py"
    spec = importlib.util.spec_from_file_location("query_agnostic_author_hm_" + file_hash(path), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_artifact_metadata(path: Path, kind: str, store: dict, manifest_path: Path, **details):
    metadata = {
        "kind": kind,
        "model_id": store["model_id"],
        "model_revision": store["model_revision"],
        "calibration_manifest_sha256": file_hash(manifest_path),
        "artifact_sha256": file_hash(path),
        "builder_source_sha256": code_hash(),
        "upstream_reference": UPSTREAM_REFERENCES.get(kind, details.get("author_sources")),
        **details,
    }
    atomic_json(path.with_suffix(".json"), metadata)
    return metadata


def existing_artifact(path: Path, kind: str, store_path: Path, manifest_path: Path, **expected):
    if not path.exists() or not path.with_suffix(".json").exists():
        return None
    metadata = json.loads(path.with_suffix(".json").read_text())
    identity = {
        "kind": kind,
        "label_store_sha256": file_hash(store_path),
        "calibration_manifest_sha256": file_hash(manifest_path),
        "artifact_sha256": file_hash(path),
        "builder_source_sha256": code_hash(),
        **expected,
    }
    if any(metadata.get(key) != value for key, value in identity.items()):
        raise ValueError("Completed calibration artifact or its inputs changed; use a new output directory.")
    return metadata


def build_hm_profiles(store_path: Path, manifest_path: Path, author_repository: Path, output: Path) -> dict:
    started = time.perf_counter()
    store = json.loads(store_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if store["kind"] != "hm_labels" or store["calibration_manifest_sha256"] != file_hash(manifest_path):
        raise ValueError("HM label store does not belong to this calibration manifest.")
    if len(store["documents"]) != manifest["recipe"]["hm_documents"]:
        raise ValueError("HM label store is incomplete.")
    if not store["documents"]:
        raise ValueError("HM needs at least one calibration document.")
    provenance = author_sources(author_repository)
    environment = {**execution_environment("cpu", "hm_build"), "numpy": np.__version__}
    existing = existing_artifact(
        output, "hm", store_path, manifest_path, author_sources=provenance, builder_environment=environment
    )
    if existing is not None:
        return existing
    primitive = load_hm_primitives(author_repository)
    measured: dict[int, list[np.ndarray]] = {budget: [] for budget in KV_BUDGETS}
    shape = None
    for record in sorted(store["documents"], key=lambda r: r["document_id"]):
        path = store_path.parent / record["labels_file"]
        if file_hash(path) != record["labels_sha256"]:
            raise ValueError(f"HM labels changed: {path}")
        labels = np.load(path, allow_pickle=False)
        if labels.ndim != 3 or not np.isfinite(labels).all():
            raise ValueError("HM labels must be finite log s+ with shape [layers, kv_heads, tokens].")
        if shape is not None and shape != labels.shape[:2]:
            raise ValueError("HM label documents have inconsistent model geometry.")
        shape = labels.shape[:2]
        block = labels.reshape(-1, labels.shape[-1]).T
        for budget in measured:
            measured[budget].append(primitive.document_shares(block, compression=1 - budget / REFERENCE_CONTEXT_LENGTH))
    arrays, diagnostics = {}, {}
    for budget, documents in measured.items():
        raw = np.mean(documents, axis=0)
        solved = primitive.solve_scale(raw, mean_keep=budget / REFERENCE_CONTEXT_LENGTH, floor=0.01, cap=1.0)
        arrays[f"shares_{budget}"] = solved["keep"].reshape(shape)
        arrays[f"raw_shares_{budget}"] = raw.reshape(shape)
        diagnostics[str(budget)] = {key: value for key, value in solved.items() if key != "keep"}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    return write_artifact_metadata(
        output,
        "hm",
        store,
        manifest_path,
        author_sources=provenance,
        builder_environment=environment,
        reference_length=REFERENCE_CONTEXT_LENGTH,
        floor=0.01,
        cap=1.0,
        documents=len(store["documents"]),
        diagnostics=diagnostics,
        label_store_sha256=file_hash(store_path),
        seconds=time.perf_counter() - started,
        recipe="Author document_shares (quantile >=, document mean) and solve_scale; log KVzip+ labels.",
    )


def qfilters_from_samples(queries: torch.Tensor, kv_heads: int) -> torch.Tensor:
    """Uncentered SVD and majority sign of U, exactly as the author's make_filters.py."""
    if queries.ndim != 3 or queries.shape[0] % kv_heads or queries.shape[1] < 1:
        raise ValueError("Q samples must be [query_heads, samples, head_dim] with complete GQA groups.")
    u, singular, vh = torch.linalg.svd(queries.float(), full_matrices=False)
    if not torch.isfinite(singular).all() or (singular[:, 0] <= 0).any():
        raise ValueError("Q-filter samples have no finite nonzero principal direction.")
    sign = ((u[..., 0] > 0).float().mean(-1) > 0.5).float() * 2 - 1
    filters = -sign[:, None] * vh[:, 0, :]
    return filters.reshape(kv_heads, -1, queries.shape[-1]).mean(1)


def build_qfilters(store_path: Path, manifest_path: Path, output: Path, device="cpu") -> dict:
    started = time.perf_counter()
    store = json.loads(store_path.read_text())
    if store["kind"] != "qfilter_samples" or store["calibration_manifest_sha256"] != file_hash(manifest_path):
        raise ValueError("Q-filter samples do not belong to this calibration manifest.")
    environment = {**execution_environment(str(device), "qfilters_build"), "numpy": np.__version__}
    existing = existing_artifact(output, "qfilters", store_path, manifest_path, builder_environment=environment)
    if existing is not None:
        return existing
    filters = []
    for record in store["layers"]:
        path = store_path.parent / record["file"]
        if file_hash(path) != record["sha256"]:
            raise ValueError(f"Q-filter sample file changed: {path}")
        queries = torch.load(path, map_location="cpu", weights_only=True).to(device)
        filters.append(qfilters_from_samples(queries, store["kv_heads"]).cpu())
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file({"q_filters": torch.stack(filters).contiguous()}, str(output))
    return write_artifact_metadata(
        output,
        "qfilters",
        store,
        manifest_path,
        samples_per_head=store["samples_per_head"],
        builder_environment=environment,
        label_store_sha256=file_hash(store_path),
        seconds=time.perf_counter() - started,
        recipe="Post-RoPE Q including bias; uncentered SVD, majority sign of U[:,0], negative filter, GQA mean.",
    )
