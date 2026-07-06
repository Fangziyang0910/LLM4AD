from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT
from llm4ad.task.optimization.online_bin_packing import dataset as obp_dataset

DEFAULT_DATASET_ID = "online_bin_packing_2O_v1"
MANIFEST_PATH = Path(__file__).resolve().parent / "dataset_manifest.json"

DEFAULT_MANIFEST = {
    "dataset_id": DEFAULT_DATASET_ID,
    "task": "online_bin_packing_2O",
    "version": 1,
    "description": "Two-objective online bin-packing uses the same fixed instances as online_bin_packing for direct comparability.",
    "source_dataset_id": obp_dataset.DEFAULT_DATASET_ID,
    "source_task": "online_bin_packing",
    "splits": obp_dataset.DEFAULT_SPLIT_SPECS,
}


def write_default_dataset() -> dict[str, Any]:
    obp_dataset.write_default_dataset()
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(DEFAULT_MANIFEST, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return DEFAULT_MANIFEST


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return DEFAULT_MANIFEST
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_split_instances(split: str = DEFAULT_SPLIT):
    return obp_dataset.load_split_instances(split=split)
