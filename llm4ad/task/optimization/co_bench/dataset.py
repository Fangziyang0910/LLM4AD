from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_DATASET_ID = "co_bench_official_v1"
MANIFEST_PATH = Path(__file__).resolve().parent / "dataset_manifest.json"

DEFAULT_MANIFEST = {
    "dataset_id": DEFAULT_DATASET_ID,
    "task": "co_bench",
    "version": 1,
    "description": "Official CO-Bench dataset protocol. Data is loaded from Hugging Face or an external local cache; synthetic generation is intentionally not provided.",
    "source": {
        "type": "huggingface_dataset",
        "repo_id": "CO-Bench/CO-Bench",
    },
    "splits": {
        "development": {
            "role": "train",
            "description": "CO-Bench development data visible during algorithm search.",
        },
        "test": {
            "role": "test",
            "description": "CO-Bench held-out test data for final reporting.",
        },
    },
    "storage_policy": {
        "git": "do_not_commit_raw_data",
        "local_cache": "llm4ad/task/optimization/co_bench/data is ignored by git",
        "recommended_transfer": "external archive, Hugging Face cache, DVC, or Git LFS outside the source tree",
    },
}


def write_default_dataset() -> dict[str, Any]:
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(DEFAULT_MANIFEST, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return DEFAULT_MANIFEST


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return DEFAULT_MANIFEST
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)
