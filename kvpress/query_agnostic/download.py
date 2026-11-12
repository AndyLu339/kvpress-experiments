# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download the frozen checkpoint to persistent storage; a GPU is not required."""

import argparse
import json
from pathlib import Path

from kvpress.query_agnostic.data import MODEL_ID, MODEL_REVISION, file_hash

SNAPSHOT_MANIFEST = "experiment_snapshot.json"


def verify_snapshot(directory: Path) -> dict:
    manifest = json.loads((directory / SNAPSHOT_MANIFEST).read_text())
    if manifest.get("model_id") != MODEL_ID or manifest.get("model_revision") != MODEL_REVISION:
        raise ValueError("Local snapshot does not identify the frozen target checkpoint.")
    if not manifest.get("complete_weights"):
        raise ValueError("Local snapshot contains only tokenizer/config files; download the weights first.")
    for name, expected in manifest["files"].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f"Local snapshot file hash mismatch: {name}")
    if not any(name.endswith(".safetensors") for name in manifest["files"]):
        raise ValueError("Snapshot manifest contains no model weights.")
    return manifest


def main(argv=None):
    from huggingface_hub import snapshot_download

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True, help="Model directory on the server's persistent disk.")
    cli.add_argument(
        "--tokenizer-only", action="store_true", help="Prepare small config/tokenizer files without weights."
    )
    args = cli.parse_args(argv)
    patterns = ["*.json", "*.txt"]
    if not args.tokenizer_only:
        patterns.append("*.safetensors")
    snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=args.output, allow_patterns=patterns)
    files = sorted(p for p in args.output.iterdir() if p.is_file() and p.suffix in (".json", ".txt", ".safetensors"))
    manifest = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "complete_weights": not args.tokenizer_only,
        "files": {p.name: file_hash(p) for p in files if p.name != SNAPSHOT_MANIFEST},
    }
    (args.output / SNAPSHOT_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {MODEL_ID}@{MODEL_REVISION} in {args.output}")


if __name__ == "__main__":
    main()
