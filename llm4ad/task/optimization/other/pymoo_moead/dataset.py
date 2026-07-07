from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT, write_manifest

DEFAULT_DATASET_ID = "pymoo_moead_v1"
DATA_DIR = Path(__file__).resolve().parent / "data"

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "problem_name": "DTLZ4",
        "n_var": 10,
        "n_obj": 3,
        "n_partitions": 12,
        "n_gen": 100,
        "seeds": [0, 1, 2],
        "hv_ref": [1.1, 1.1, 1.1],
    },
    "test_id": {
        "role": "test",
        "problem_name": "DTLZ4",
        "n_var": 10,
        "n_obj": 3,
        "n_partitions": 12,
        "n_gen": 100,
        "seeds": [100, 101, 102, 103, 104],
        "hv_ref": [1.1, 1.1, 1.1],
    },
    "test_ood_var20": {
        "role": "test",
        "problem_name": "DTLZ4",
        "n_var": 20,
        "n_obj": 3,
        "n_partitions": 12,
        "n_gen": 100,
        "seeds": [200, 201, 202, 203, 204],
        "hv_ref": [1.1, 1.1, 1.1],
    },
    "test_ood_obj5": {
        "role": "test",
        "problem_name": "DTLZ4",
        "n_var": 20,
        "n_obj": 5,
        "n_partitions": 6,
        "n_gen": 100,
        "seeds": [300, 301, 302],
        "hv_ref": [1.1, 1.1, 1.1, 1.1, 1.1],
    },
}


def write_default_dataset() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_id": DEFAULT_DATASET_ID,
        "task": "pymoo_moead",
        "version": 1,
        "description": "Fixed pymoo MOEA/D benchmark scenarios and seeds for train/test evaluation.",
        "generator": "llm4ad.task.optimization.other.pymoo_moead.dataset.write_default_dataset",
        "splits": DEFAULT_SPLIT_SPECS,
    }
    return write_manifest(DATA_DIR, manifest)


def load_manifest() -> dict[str, Any]:
    path = DATA_DIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"pymoo MOEA/D manifest not found: {path}. "
            "Run `uv run python -m llm4ad.task.optimization.other.pymoo_moead.generate_dataset`."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_split_case(split: str = DEFAULT_SPLIT) -> dict[str, Any]:
    manifest = load_manifest()
    splits = manifest.get("splits", {})
    if split not in splits:
        available = ", ".join(sorted(splits))
        raise ValueError(f"Unknown pymoo MOEA/D split `{split}`. Available splits: {available}")
    return {
        "dataset_id": manifest["dataset_id"],
        "task": manifest["task"],
        "split": split,
        **splits[split],
    }
