# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SCBench's shared context protocol, with task instructions moved after the body."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-1M"
MODEL_REVISION = "e28526f7bb80e2a9c8af03b831a9af3812f18fba"
DATA_REVISION = "a079be919d3131822c202180ffd4dc322968de45"
REFERENCE_CONTEXT_LENGTH = 131072
KV_BUDGETS = (4096, 8192, 16384, 32768)
TASKS = ("scbench_vt", "scbench_repoqa", "scbench_qa_eng", "scbench_choice_eng", "scbench_summary")
LIMITS = dict(zip(TASKS, (96, 512, 96, 96, 256)))
INSTRUCTIONS = {
    "scbench_vt": "Memorize and track the chain(s) of variable assignment hidden in the following text.",
    "scbench_qa_eng": "Read the book and answer the question. Be very concise in your answer.",
    "scbench_choice_eng": "Read the book and answer the question.",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Document:
    task: str
    row: int
    raw_context: str
    body: str
    moved_instruction: str
    questions: list[str]
    references: list
    metadata: dict

    @property
    def document_id(self) -> str:
        return f"{self.task}-{self.row:04d}"


def parse_document(task: str, row_index: int, row: dict) -> Document:
    if task not in TASKS:
        raise ValueError(f"Unsupported SCBench task: {task}")
    raw = row["prompts"][0]
    questions, references = row["prompts"][1:], row["ground_truth"]
    if not questions or len(questions) != len(references):
        raise ValueError(f"{task}/{row_index}: question/reference counts do not match.")
    instruction = INSTRUCTIONS.get(task, "")
    if instruction and not raw.startswith(instruction + "\n\n"):
        raise ValueError(f"{task}/{row_index}: unregistered body instruction prefix; inspect rather than guess.")
    body = raw[len(instruction) :]
    metadata = {key: value for key, value in row.items() if key not in ("prompts", "ground_truth")}
    if task == "scbench_repoqa" and len(metadata["func_name"]) != len(questions):
        raise ValueError("RepoQA function names must match the question count.")
    # Original question text includes the choices. The parquet 'options' field can describe
    # only the first question, so never reconstruct subsequent choices from that field.
    return Document(task, row_index, raw, body, instruction, questions, references, metadata)


def load_documents(directory: Path, tasks=TASKS, limit: int = 0):
    import pyarrow.parquet as pq

    if limit < 0:
        raise ValueError("Document limit cannot be negative.")
    for task in tasks:
        table = pq.read_table(directory / f"{task}.parquet")
        if limit:
            table = table.slice(0, limit)
        for index, row in enumerate(table.to_pylist()):
            yield parse_document(task, index, row)


def tokenize_document(document: Document, tokenizer) -> tuple[dict, dict]:
    marker = "__KV_QUERY_AGNOSTIC_BODY_BOUNDARY_7C091D__"
    template = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": marker},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    if template.count(marker) != 1:
        raise ValueError("Chat template must preserve exactly one user content boundary.")
    prefix, suffix = template.split(marker)
    body = tokenizer(document.body, add_special_tokens=False, return_offsets_mapping=True)
    question_texts = [
        "\n\n" + (document.moved_instruction + "\n\n" if document.moved_instruction else "") + q + suffix
        for q in document.questions
    ]
    encoded = {
        "prefix": tokenizer.encode(prefix, add_special_tokens=False),
        "body": body["input_ids"],
        "questions": [tokenizer.encode(q, add_special_tokens=False) for q in question_texts],
        "body_offsets": body["offset_mapping"],
    }
    manifest = {
        "document_id": document.document_id,
        "task": document.task,
        "row": document.row,
        "dataset_revision": DATA_REVISION,
        "raw_context_sha256": sha256_bytes(document.raw_context.encode()),
        "normalized_context_sha256": sha256_bytes(document.body.encode()),
        "moved_instruction": document.moved_instruction,
        "removed_character_span": [0, len(document.moved_instruction)],
        "prefix_tokens": len(encoded["prefix"]),
        "body_tokens": len(encoded["body"]),
        "body_ids_sha256": sha256_bytes(np.asarray(encoded["body"], dtype="<i8").tobytes()),
        "question_count": len(encoded["questions"]),
        "question_tokens": [len(q) for q in encoded["questions"]],
        "question_sha256": [sha256_bytes(q.encode()) for q in question_texts],
        "metadata": document.metadata,
    }
    return encoded, manifest
