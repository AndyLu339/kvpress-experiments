# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layer-at-a-time Fast KVzip training from disk-backed, source-disjoint features."""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from kvpress.presses.fastkvzip_press import FastKVzipGate
from kvpress.query_agnostic.calibration_ops import write_artifact_metadata
from kvpress.query_agnostic.corpus import assert_source_disjoint
from kvpress.query_agnostic.data import file_hash, sha256_bytes
from kvpress.query_agnostic.run import atomic_json, code_hash, execution_environment


@dataclass
class GateTrainingConfig:
    steps: int = 5000
    batch_size: int = 1024
    learning_rate: float = 0.2
    momentum: float = 0.9
    dimension: int = 16
    sinks: int = 16
    validation_interval: int = 500
    seed: int = 0

    def __post_init__(self):
        if min(self.steps, self.batch_size, self.dimension, self.sinks, self.validation_interval) < 1:
            raise ValueError("Gate dimensions, training steps and intervals must be positive.")
        if self.learning_rate <= 0 or not 0 <= self.momentum < 1:
            raise ValueError("Invalid SGD hyperparameters.")


def learning_rate_multiplier(step: int, steps: int) -> float:
    # Exactly utils.lr_schedule from FastKVzip e04afaa, including its warm-up discontinuity.
    if step < 0.01 * steps:
        return 0.01 * step
    if step >= 0.75 * steps:
        return 0.25
    if step >= 0.5 * steps:
        return 0.5
    return 1.0


def training_prediction(gate, inputs: torch.Tensor) -> torch.Tensor:
    # Official optim.Weight adds b twice. Train that parameterization, then export 2*b
    # for the single-bias deployed FastKVzipGate. functional_call preserves the 2x gradient.
    return torch.func.functional_call(gate, {"b": 2 * gate.b}, (inputs.unsqueeze(0),))[0]


def gate_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Author's cnt/BCE reduction: mean over tokens, SUM over KV heads.
    return F.binary_cross_entropy(prediction.float(), target.float(), reduction="none").mean(-1).sum()


def exported_state(gate) -> dict:
    state = {key: tensor.detach().cpu().clone() for key, tensor in gate.state_dict().items()}
    state["b"] *= 2
    return state


def atomic_torch_save(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


class FeatureDataset:
    def __init__(self, root: Path, records: list[dict], layer: int, split: str):
        self.records = [r for r in records if r["split"] == split]
        if not self.records:
            raise ValueError(f"No source-disjoint {split} feature documents.")
        self.features, self.labels = [], []
        self.bounds = np.cumsum([0] + [r["body_tokens"] for r in self.records])
        self.encoding = self.records[0]["feature_encoding"]
        for record in self.records:
            if record["feature_encoding"] != self.encoding:
                raise ValueError("Feature encodings differ across documents.")
            self.features.append(np.load(root / record["features"][layer]["file"], mmap_mode="r", allow_pickle=False))
            self.labels.append(np.load(root / record["labels_file"], mmap_mode="r", allow_pickle=False))
        self.layer = layer
        self.dtype = torch.bfloat16 if self.encoding == "bfloat16_bits" else torch.float32

    def __len__(self):
        return int(self.bounds[-1])

    def batch(self, indices: torch.Tensor):
        indices = indices.cpu().numpy()
        inputs = torch.empty(len(indices), self.features[0].shape[1], dtype=self.dtype)
        targets = torch.empty(self.labels[0].shape[1], len(indices), dtype=torch.float32)
        for i, (features, labels) in enumerate(zip(self.features, self.labels)):
            where = np.flatnonzero((indices >= self.bounds[i]) & (indices < self.bounds[i + 1]))
            if not len(where):
                continue
            local = indices[where] - self.bounds[i]
            values = torch.from_numpy(np.array(features[local], copy=True))
            if self.encoding == "bfloat16_bits":
                values = values.view(torch.bfloat16)
            rows = torch.from_numpy(where)
            inputs[rows] = values
            targets[:, rows] = torch.from_numpy(np.array(labels[self.layer][:, local], copy=True))
        return inputs, targets

    def close(self):
        for array in self.features + self.labels:
            array._mmap.close()


@torch.inference_mode()
def validation_loss(gate, dataset: FeatureDataset, batch_size: int, device) -> float:
    total, count = 0.0, 0
    for start in range(0, len(dataset), batch_size):
        x, y = dataset.batch(torch.arange(start, min(start + batch_size, len(dataset))))
        loss = gate_loss(training_prediction(gate, x.to(device)), y.to(device))
        total += float(loss) * len(x)
        count += len(x)
    return total / count


def train_gates(store_path: Path, manifest_path: Path, output: Path, config: GateTrainingConfig, device="cuda"):
    started = time.perf_counter()
    store = json.loads(store_path.read_text())
    if store["kind"] != "gate_features" or store["calibration_manifest_sha256"] != file_hash(manifest_path):
        raise ValueError("Gate feature store does not belong to this calibration manifest.")
    assert_source_disjoint(store["documents"])
    for record in store["documents"]:
        for item in record["features"] + [{"file": record["labels_file"], "sha256": record["labels_sha256"]}]:
            if file_hash(store_path.parent / item["file"]) != item["sha256"]:
                raise ValueError(f"Training input changed: {item['file']}")
    signature = {
        "store_sha256": file_hash(store_path),
        "config": asdict(config),
        "source_sha256": code_hash(),
        "environment": execution_environment(str(device), "gate_training"),
    }
    config_hash = sha256_bytes(json.dumps(signature, sort_keys=True).encode())
    directory = output.parent
    directory.mkdir(parents=True, exist_ok=True)
    config_file = directory / "training_config.json"
    if config_file.exists() and json.loads(config_file.read_text()) != signature:
        raise ValueError("Gate training configuration changed; use a new output directory.")
    atomic_json(config_file, signature)
    if output.exists() and output.with_suffix(".json").exists():
        existing = json.loads(output.with_suffix(".json").read_text())
        if existing.get("training_config_hash") != config_hash or existing["artifact_sha256"] != file_hash(output):
            raise ValueError("Completed gate artifact changed or belongs to another training configuration.")
        return existing
    layers, reports = [], []
    for layer in range(store["layers"]):
        weight_path = directory / f"layer_{layer:02d}.pt"
        done_path = directory / f"layer_{layer:02d}.json"
        if done_path.exists():
            report = json.loads(done_path.read_text())
            if report["config_hash"] != config_hash or report["weights_sha256"] != file_hash(weight_path):
                raise ValueError("Completed gate layer has incompatible configuration or changed weights.")
            layers.append(torch.load(weight_path, map_location="cpu", weights_only=True))
            reports.append(report)
            continue
        train = FeatureDataset(store_path.parent, store["documents"], layer, "train")
        validation = FeatureDataset(store_path.parent, store["documents"], layer, "validation")
        try:
            torch.manual_seed(config.seed + layer)
            gate = FastKVzipGate(
                layer,
                store["hidden_size"],
                store["kv_heads"],
                store["query_heads"] // store["kv_heads"],
                train.dtype,
                config.dimension,
                config.sinks,
            ).to(device)
            optimizer = torch.optim.SGD(gate.parameters(), lr=config.learning_rate, momentum=config.momentum)
            generator = torch.Generator().manual_seed(config.seed)
            batch_size = min(config.batch_size, len(train))
            batches = max(1, len(train) // batch_size)
            checkpoint_path = directory / f"layer_{layer:02d}.checkpoint.pt"
            start_step, history, permutation = 0, [], torch.empty(0, dtype=torch.long)
            if checkpoint_path.exists():
                saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                if saved["config_hash"] != config_hash:
                    raise ValueError("Gate training checkpoint configuration differs.")
                gate.load_state_dict(saved["gate"])
                optimizer.load_state_dict(saved["optimizer"])
                generator.set_state(saved["rng"])
                start_step, permutation, history = saved["step"], saved["permutation"], saved["history"]
                initial_validation = saved["initial_validation_loss"]
            else:
                initial_validation = validation_loss(gate, validation, config.batch_size, device)
            for step in range(start_step, config.steps):
                split = step % batches
                if split == 0:
                    permutation = torch.randperm(len(train), generator=generator)
                x, y = train.batch(permutation[split * batch_size : (split + 1) * batch_size])
                optimizer.zero_grad(set_to_none=True)
                loss = gate_loss(training_prediction(gate, x.to(device)), y.to(device))
                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite gate loss at layer {layer}, step {step}.")
                loss.backward()
                optimizer.param_groups[0]["lr"] = config.learning_rate * learning_rate_multiplier(step, config.steps)
                optimizer.step()
                if (step + 1) % config.validation_interval == 0 or step + 1 == config.steps:
                    measured = validation_loss(gate, validation, config.batch_size, device)
                    history.append(
                        {"step": step + 1, "training_loss": float(loss.detach()), "validation_loss": measured}
                    )
                    atomic_torch_save(
                        checkpoint_path,
                        {
                            "config_hash": config_hash,
                            "step": step + 1,
                            "gate": gate.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "rng": generator.get_state(),
                            "permutation": permutation,
                            "history": history,
                            "initial_validation_loss": initial_validation,
                        },
                    )
                    print(f"Gate layer {layer}, step {step + 1}: validation BCE {measured:.6f}", flush=True)
            state = exported_state(gate)
            atomic_torch_save(weight_path, state)
            report = {
                "layer": layer,
                "config_hash": config_hash,
                "weights_sha256": file_hash(weight_path),
                "initial_validation_loss": initial_validation,
                "history": history,
                "train_tokens": len(train),
                "validation_tokens": len(validation),
            }
            atomic_json(done_path, report)
            checkpoint_path.unlink(missing_ok=True)
            layers.append(state)
            reports.append(report)
            del gate, optimizer
        finally:
            train.close()
            validation.close()
    atomic_torch_save(output, {"layers": layers})
    return write_artifact_metadata(
        output,
        "fastkvzip",
        store,
        manifest_path,
        training=asdict(config),
        training_config_hash=config_hash,
        layers=reports,
        bias_convention="official training 2*b folded into deployed b",
        recipe=(
            "Soft KVzip targets; BCE mean(tokens), sum(heads); SGD momentum; "
            "author LR schedule; source-disjoint validation."
        ),
        feature_store_sha256=file_hash(store_path),
        seconds=time.perf_counter() - started,
    )
