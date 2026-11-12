# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen, source-disjoint calibration windows; token IDs are the authoritative input."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from kvpress.query_agnostic.data import MODEL_ID, MODEL_REVISION, file_hash, sha256_bytes
from kvpress.query_agnostic.run import atomic_json


@dataclass
class CorpusRecipe:
    seed: int = 0
    hm_documents: int = 200
    hm_length: int = 16384
    qfilter_documents: int = 20
    qfilter_length: int = 2048
    gate_short_tokens: int = 500000
    gate_long_tokens: int = 500000
    validation_short_tokens: int = 50000
    validation_long_tokens: int = 100000
    short_min: int = 10000
    short_max: int = 30000
    long_length: int = 100000

    def __post_init__(self):
        if min(asdict(self).values()) < 0 or self.short_min < 1 or self.short_max < self.short_min:
            raise ValueError("Invalid calibration corpus recipe.")
        if min(self.hm_length, self.qfilter_length, self.long_length) < 1:
            raise ValueError("Window lengths must be positive.")
        for total in (self.gate_long_tokens, self.validation_long_tokens):
            if total % self.long_length:
                raise ValueError("Long-token totals must be multiples of the long-window length.")


def short_lengths(total: int, minimum: int, maximum: int, rng) -> list[int]:
    if total and total < minimum:
        raise ValueError("Short-token total cannot be smaller than one valid short window.")
    sizes = []
    while total:
        size = min(int(rng.integers(minimum, maximum + 1)), total)
        if 0 < total - size < minimum:
            if total <= maximum:
                size = total
            else:
                size = total - minimum
        sizes.append(size)
        total -= size
    return sizes


def prepare_corpus(rows, tokenizer, output: Path, recipe: CorpusRecipe, source: dict, target=None) -> dict:
    """Natural HM/Q-filter/short windows; long gate windows concatenate disjoint sources.

    A source is consumed once, even when only its first window is used. Train and validation
    therefore cannot receive different windows from the same source. Duplicate text is skipped.
    """
    target = target or {"model_id": MODEL_ID, "model_revision": MODEL_REVISION}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "calibration_manifest.json"
    if manifest_path.exists():
        import json

        existing = json.loads(manifest_path.read_text())
        if existing["recipe"] != asdict(recipe) or existing["source"] != source:
            raise ValueError("Calibration corpus recipe/source changed; use a new output directory.")
        if any(existing[key] != value for key, value in target.items()):
            raise ValueError("Calibration corpus belongs to another target checkpoint.")
        for document in existing["documents"]:
            if file_hash(output / document["token_file"]) != document["token_file_sha256"]:
                raise ValueError("Frozen corpus token file changed.")
        assert_source_disjoint(existing["documents"])
        return existing
    token_dir = output / "tokens"
    token_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(recipe.seed)
    natural = [("hm", "calibration", recipe.hm_length)] * recipe.hm_documents
    natural += [("qfilters", "calibration", recipe.qfilter_length)] * recipe.qfilter_documents
    for split, total in (("train", recipe.gate_short_tokens), ("validation", recipe.validation_short_tokens)):
        natural += [("gate_short", split, n) for n in short_lengths(total, recipe.short_min, recipe.short_max, rng)]
    long_jobs = []
    for split, total in (("train", recipe.gate_long_tokens), ("validation", recipe.validation_long_tokens)):
        long_jobs += [("gate_long", split, recipe.long_length)] * (total // recipe.long_length)
    separator = tokenizer.encode("\n\n", add_special_tokens=False)
    documents: list[dict] = []
    pieces: list[int] = []
    seen, contributions = set(), []
    seen_ids = set()
    long_size, scanned = 0, 0

    def save(role, split, ids, origins):
        index = len(documents)
        path = token_dir / f"{index:05d}.npy"
        np.save(path, np.asarray(ids, dtype=np.int32), allow_pickle=False)
        documents.append(
            {
                "document_id": f"{role}-{index:05d}",
                "role": role,
                "split": split,
                "token_file": str(path.relative_to(output)),
                "token_file_sha256": file_hash(path),
                "body_tokens": len(ids),
                "sources": origins,
                "construction": "concatenated_sources" if len(origins) > 1 else "source_prefix",
            }
        )

    for index, row in enumerate(rows):
        if not natural and not long_jobs:
            break
        scanned += 1
        text = row["text"]
        digest = sha256_bytes(text.encode())
        source_id = str(row["id"] if row.get("id") is not None else index)
        if digest in seen or source_id in seen_ids:
            continue
        seen.add(digest)
        seen_ids.add(source_id)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        origin = {"source_id": source_id, "text_sha256": digest}
        job = next((i for i, (_, _, n) in enumerate(natural) if len(ids) >= n), None)
        if job is not None:
            role, split, length = natural.pop(job)
            save(role, split, ids[:length], [dict(origin, source_token_span=[0, length])])
        elif long_jobs:
            role, split, length = long_jobs[0]
            joiner = separator if pieces else []
            remaining = length - long_size
            # A separator never fills the final window without any tokens of its next source.
            if len(joiner) >= remaining:
                joiner = []
            take = min(len(ids), remaining - len(joiner))
            pieces.extend(joiner + ids[:take])
            contributions.append(dict(origin, source_token_span=[0, take], window_token_start=long_size + len(joiner)))
            long_size = len(pieces)
            if long_size == length:
                save(role, split, pieces, contributions)
                long_jobs.pop(0)
                pieces, contributions, long_size = [], [], 0
    if natural or long_jobs:
        raise ValueError(
            f"Calibration corpus exhausted after {scanned} sources: {len(natural)} natural windows and "
            f"{len(long_jobs)} long windows remain. Supply a larger frozen source pool."
        )
    manifest = {
        **target,
        "recipe": asdict(recipe),
        "source": source,
        "scanned_sources": scanned,
        "documents": documents,
    }
    assert_source_disjoint(documents)
    atomic_json(output / "calibration_manifest.json", manifest)
    return manifest


def assert_source_disjoint(documents: list[dict]):
    owners: dict[tuple[str, str], str] = {}
    for document in documents:
        for source in document["sources"]:
            for identity in (("text", source["text_sha256"]), ("id", source["source_id"])):
                previous = owners.setdefault(identity, document["document_id"])
                if previous != document["document_id"]:
                    raise ValueError("A source document occurs in more than one calibration/training window.")
