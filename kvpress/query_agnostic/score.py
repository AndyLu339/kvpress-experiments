# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score saved predictions with the experiment's original local KVzip/RepoQA sources."""

import argparse
import contextlib
import importlib
import io
import json
import sys
from pathlib import Path

import numpy as np

from kvpress.query_agnostic.data import file_hash
from kvpress.query_agnostic.run import atomic_json


def load_metrics(root: Path):
    root = root.resolve()
    sys.path.insert(0, str(root))
    metric = importlib.import_module("results.metric")
    repoqa = importlib.import_module("results.repo_qa_utils")
    if Path(metric.__file__).resolve() != root / "results/metric.py":
        raise ValueError("A different results.metric module is already imported; run scoring in a fresh process.")
    return metric, repoqa


def score_document(document: dict, metric, repoqa) -> dict:
    task = document["manifest"]["task"]
    predictions = document["predictions"]
    metadata = document["manifest"]["metadata"]
    scores = []
    for row in predictions:
        prediction = row["prediction"].removesuffix("</s>")
        detail = {"question_index": row["question_index"], "score": 0.0}
        if prediction.strip():
            if task == "scbench_repoqa":
                needles = [
                    {"needle": p["reference"], "name": metadata["func_name"][p["question_index"]]} for p in predictions
                ]
                target = metadata["func_name"][row["question_index"]]
                verdict, best, similarity = repoqa.needle_evaluator(
                    prediction,
                    {"func_name": target, "ground_truth": row["reference"]},
                    needles,
                    metadata["lang"],
                    False,
                )
                detail.update(
                    score=float(verdict == repoqa.Result.BEST_MATCH and similarity >= 0.8),
                    best_target=best,
                    similarity=similarity,
                )
            else:
                reference = row["reference"]
                if task == "scbench_vt":
                    reference = ", ".join(reference)
                with contextlib.redirect_stdout(io.StringIO()):
                    detail["score"] = float(metric.evaluate_answer([prediction], [reference], task, "qa")[0])
        # Empty predictions remain zero-valued questions in the denominator.
        scores.append(detail)
    return {
        "document_id": document["manifest"]["document_id"],
        "task": task,
        "questions": scores,
        "score": float(np.mean([q["score"] for q in scores])),
    }


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", type=Path, required=True)
    cli.add_argument(
        "--metric-root", type=Path, required=True, help="Original KVzip root containing results/metric.py."
    )
    args = cli.parse_args(argv)
    metric, repoqa = load_metrics(args.metric_root)
    config = json.loads((args.run / "config.json").read_text())
    documents = []
    for path in sorted((args.run / "documents").glob("*.json")):
        if path.name.endswith((".partial.json", ".error.json")):
            continue
        document = json.loads(path.read_text())
        if document["status"] != "complete" or document["config_hash"] != config["config_hash"]:
            raise ValueError(f"Incomplete or incompatible document result: {path}")
        documents.append(score_document(document, metric, repoqa))
    expected = sum(min(n, config["limit"]) if config["limit"] else n for n in config["dataset_counts"].values())
    complete = len(documents) == expected
    task_scores = {
        task: float(np.mean([d["score"] for d in documents if d["task"] == task]))
        for task in config["tasks"]
        if any(d["task"] == task for d in documents)
    }
    report = {
        "config_hash": config["config_hash"],
        "metric_sha256": file_hash(Path(metric.__file__)),
        "repoqa_sha256": file_hash(Path(repoqa.__file__)),
        "expected_documents": expected,
        "completed_documents": len(documents),
        "complete": complete,
        "available_document_task_scores": task_scores,
        "task_equal_mean": float(np.mean(list(task_scores.values()))) if complete else None,
        "documents": documents,
    }
    atomic_json(args.run / "scores.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "documents"}, ensure_ascii=False, indent=2))
    return int(not complete)


if __name__ == "__main__":
    raise SystemExit(main())
