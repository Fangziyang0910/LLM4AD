from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import (
    DEFAULT_SPLIT,
    file_sha256,
    load_pickle_split,
    task_data_dir,
)

DEFAULT_DATASET_ID = "tsp_aco_v1"
DATA_DIR = task_data_dir(__file__)

SOURCE_SEED = 1234
SPLIT_ORDER = [
    "train",
    "val_20",
    "val_50",
    "val_100",
    "test_20",
    "test_50",
    "test_100",
]

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "filename": "train.pkl.gz",
        "n_instances": 5,
        "problem_size": 50,
        "source_file": "train50_dataset.npy",
    },
    "val_20": {
        "role": "validation",
        "filename": "val_20.pkl.gz",
        "n_instances": 64,
        "problem_size": 20,
        "source_file": "val20_dataset.npy",
    },
    "val_50": {
        "role": "validation",
        "filename": "val_50.pkl.gz",
        "n_instances": 64,
        "problem_size": 50,
        "source_file": "val50_dataset.npy",
    },
    "val_100": {
        "role": "validation",
        "filename": "val_100.pkl.gz",
        "n_instances": 64,
        "problem_size": 100,
        "source_file": "val100_dataset.npy",
    },
    "test_20": {
        "role": "test",
        "filename": "test_20.pkl.gz",
        "n_instances": 64,
        "problem_size": 20,
        "source_file": "test20_dataset.npy",
    },
    "test_50": {
        "role": "test",
        "filename": "test_50.pkl.gz",
        "n_instances": 64,
        "problem_size": 50,
        "source_file": "test50_dataset.npy",
    },
    "test_100": {
        "role": "test",
        "filename": "test_100.pkl.gz",
        "n_instances": 64,
        "problem_size": 100,
        "source_file": "test100_dataset.npy",
    },
}


def _generate_all_splits() -> dict[str, np.ndarray]:
    rng = np.random.RandomState(SOURCE_SEED)
    splits: dict[str, np.ndarray] = {}

    splits["train"] = rng.rand(5, 50, 2).astype(np.float64)
    for problem_size in [20, 50, 100]:
        splits[f"val_{problem_size}"] = rng.rand(64, problem_size, 2).astype(np.float64)
    for problem_size in [20, 50, 100]:
        splits[f"test_{problem_size}"] = rng.rand(64, problem_size, 2).astype(np.float64)

    return splits


def write_default_dataset() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    generated = _generate_all_splits()

    splits: dict[str, Any] = {}
    for split in SPLIT_ORDER:
        spec = DEFAULT_SPLIT_SPECS[split]
        path = DATA_DIR / spec["filename"]
        with gzip.open(path, "wb") as handle:
            pickle.dump(generated[split], handle, protocol=pickle.HIGHEST_PROTOCOL)

        splits[split] = {
            **{k: v for k, v in spec.items() if k != "filename"},
            "filename": spec["filename"],
            "format": "pickle_gzip",
            "seed": SOURCE_SEED,
            "sha256": file_sha256(path),
        }

    manifest = {
        "dataset_id": DEFAULT_DATASET_ID,
        "task": "tsp_aco",
        "version": 1,
        "description": "Fixed TSP-ACO splits generated from the ReEvo TSP-ACO instance protocol.",
        "generator": "llm4ad.task.optimization.tsp_aco.dataset.write_default_dataset",
        "source": "reference_code/ReEvo/problems/tsp_aco/gen_inst.py",
        "splits": splits,
    }
    with (DATA_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def load_manifest() -> dict[str, Any]:
    from llm4ad.task.optimization.dataset_io import load_manifest as _load_manifest

    return _load_manifest(DATA_DIR)


def load_split_instances(split: str = DEFAULT_SPLIT):
    return load_pickle_split(data_dir=DATA_DIR, split=split)
