# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import numpy as np
import pytest

from kvpress.query_agnostic.artifacts import load_hm
from kvpress.query_agnostic.data import (
    INSTRUCTIONS,
    MODEL_ID,
    MODEL_REVISION,
    file_hash,
    load_documents,
    parse_document,
    tokenize_document,
)
from kvpress.query_agnostic.run import main
from kvpress.query_agnostic.score import load_metrics, score_document


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text), "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def apply_chat_template(self, messages, **kwargs):
        return "SYSTEM:" + messages[0]["content"] + "\nUSER:" + messages[1]["content"] + "\nEND\nASSISTANT:"


def test_shared_body_and_task_question_are_strictly_separated():
    instruction = INSTRUCTIONS["scbench_qa_eng"]
    raw = instruction + "\n\nAn ordinary body with a title and named entities."
    row = {"prompts": [raw, "SECRET_QUESTION_A", "SECRET_QUESTION_B"], "ground_truth": ["a", "b"]}
    document = parse_document("scbench_qa_eng", 0, row)
    encoded, manifest = tokenize_document(document, CharacterTokenizer())
    assert document.moved_instruction + document.body == raw
    assert manifest["removed_character_span"] == [0, len(instruction)]
    prefix_body = "".join(map(chr, encoded["prefix"] + encoded["body"]))
    assert instruction not in prefix_body
    assert "SECRET_QUESTION" not in prefix_body
    for index, question in enumerate(encoded["questions"]):
        suffix = "".join(map(chr, question))
        assert instruction in suffix
        assert row["prompts"][index + 1] in suffix
        assert suffix.endswith("ASSISTANT:")
    assert len(encoded["body_offsets"]) == len(encoded["body"])


def test_choice_questions_retain_their_own_options():
    row = {
        "prompts": [INSTRUCTIONS["scbench_choice_eng"] + "\n\nBook", "A. first\nB. second", "A. third\nB. fourth"],
        "ground_truth": ["first", "fourth"],
        "options": ["first", "second"],
    }
    document = parse_document("scbench_choice_eng", 1, row)
    encoded, _ = tokenize_document(document, CharacterTokenizer())
    assert "third" in "".join(map(chr, encoded["questions"][1]))
    assert "first" not in "".join(map(chr, encoded["questions"][1]))


def test_unknown_prefix_and_missing_hm_fail_before_loading_weights(tmp_path):
    with pytest.raises(ValueError, match="unregistered"):
        parse_document("scbench_qa_eng", 0, {"prompts": ["Unexpected instruction", "Question"], "ground_truth": ["a"]})
    with pytest.raises(SystemExit) as error:
        main(["--dataset-dir", str(tmp_path), "--output", str(tmp_path / "out")])
    assert error.value.code == 2


def test_artifacts_reject_wrong_checkpoint_and_tampered_bytes(tmp_path):
    path = tmp_path / "hm_profiles.npz"
    np.savez(path, shares_8192=np.ones((28, 4)))
    metadata = {
        "kind": "hm",
        "model_id": MODEL_ID,
        "model_revision": "wrong",
        "calibration_manifest_sha256": "example",
        "artifact_sha256": file_hash(path),
    }
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="model_revision"):
        load_hm(path, 8192)
    metadata["model_revision"] = MODEL_REVISION
    sidecar.write_text(json.dumps(metadata))
    assert load_hm(path, 8192).shape == (28, 4)
    np.savez(path, shares_8192=np.ones((28, 4)) * 2)
    with pytest.raises(ValueError, match="SHA256"):
        load_hm(path, 8192)


def test_all_local_scbench_rows_follow_the_registered_protocol():
    directory = Path(__file__).resolve().parents[2] / "data/kv_query_agnostic_2026-11-10/dataset_samples"
    if not directory.exists():
        pytest.skip("Workspace SCBench parquet files are not installed.")
    documents = list(load_documents(directory))
    assert len(documents) == 286
    assert sum(len(d.questions) for d in documents) == 1413
    for document in documents:
        assert document.moved_instruction + document.body == document.raw_context


@pytest.fixture
def original_metrics():
    pytest.importorskip("rouge")
    pytest.importorskip("tree_sitter_languages")
    root = Path(__file__).resolve().parents[2] / "data/kv_query_agnostic_2026-11-10/repos/KVzip"
    if not root.exists():
        pytest.skip("Original workspace KVzip scoring sources are unavailable.")
    return load_metrics(root)


def test_original_scoring_keeps_empty_answers_and_vt_case(original_metrics):
    metric, repoqa = original_metrics
    document = {
        "manifest": {"task": "scbench_vt", "document_id": "vt", "metadata": {}},
        "predictions": [
            {"question_index": 0, "prediction": "VAR_A, VAR_B", "reference": ["VAR_A", "VAR_B"]},
            {"question_index": 1, "prediction": "", "reference": ["VAR_C"]},
            {"question_index": 2, "prediction": "var_d", "reference": ["VAR_D"]},
        ],
    }
    result = score_document(document, metric, repoqa)
    assert [q["score"] for q in result["questions"]] == [1, 0, 0]
    assert result["score"] == pytest.approx(1 / 3)


def test_original_repoqa_scoring_does_not_drop_empty_predictions(original_metrics):
    metric, repoqa = original_metrics
    function_a = "def alpha(a, b, c):\n    return a + b + c\n"
    function_b = "def beta(a, b, c):\n    return a * b * c\n"
    document = {
        "manifest": {
            "task": "scbench_repoqa",
            "document_id": "repo",
            "metadata": {"lang": "python", "repo": "example", "func_name": ["alpha", "beta"]},
        },
        "predictions": [
            {"question_index": 0, "prediction": function_a, "reference": function_a},
            {"question_index": 1, "prediction": "", "reference": function_b},
        ],
    }
    result = score_document(document, metric, repoqa)
    assert [q["score"] for q in result["questions"]] == [1, 0]
    assert result["score"] == 0.5
