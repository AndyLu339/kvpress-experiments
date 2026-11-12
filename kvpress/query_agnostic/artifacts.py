# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit offline-artifact loading; never substitute a different checkpoint's prior."""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from kvpress.presses.fastkvzip_press import FastKVzipGate
from kvpress.query_agnostic.data import MODEL_ID, MODEL_REVISION, file_hash


def artifact_metadata(path: Path, kind: str) -> dict:
    metadata = json.loads(path.with_suffix(".json").read_text())
    for key, expected in (("model_id", MODEL_ID), ("model_revision", MODEL_REVISION), ("kind", kind)):
        if metadata.get(key) != expected:
            raise ValueError(f"{path}: {key} must be {expected!r}, got {metadata.get(key)!r}.")
    if not metadata.get("calibration_manifest_sha256"):
        raise ValueError(f"{path}: missing calibration manifest hash.")
    expected_hash = metadata.get("artifact_sha256")
    if expected_hash is not None and expected_hash != file_hash(path):
        raise ValueError(f"{path}: artifact content does not match its registered SHA256.")
    return metadata


def load_hm(path: Path, budget: int) -> np.ndarray:
    artifact_metadata(path, "hm")
    with np.load(path, allow_pickle=False) as profiles:
        return profiles[f"shares_{budget}"].copy()


def load_filters(path: Path) -> torch.Tensor:
    artifact_metadata(path, "qfilters")
    return load_file(str(path), device="cpu")["q_filters"]


def load_gates(path: Path, model) -> list:
    artifact_metadata(path, "fastkvzip")
    states = torch.load(path, map_location="cpu", weights_only=True)["layers"]
    if len(states) != model.config.num_hidden_layers:
        raise ValueError("Gate layer count differs from the target model.")
    gates = []
    heads = model.config.num_key_value_heads
    groups = model.config.num_attention_heads // heads
    for index, state in enumerate(states):
        dimension = state["q_norm.weight"].numel()
        sink = state["k_base"].shape[-2]
        gate = FastKVzipGate(index, model.config.hidden_size, heads, groups, model.dtype, dimension, sink)
        gate.load_state_dict(state, strict=True)
        gates.append(gate.to(model.device).eval().requires_grad_(False))
    return gates
