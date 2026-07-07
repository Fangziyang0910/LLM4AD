from __future__ import annotations

from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import (
    DEFAULT_SPLIT,
    load_pickle_split,
    task_data_dir,
    write_pickle_splits,
)

DEFAULT_DATASET_ID = "bp_1d_construct_v1"
DATA_DIR = task_data_dir(__file__)

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "n_instances": 16,
        "n_items": 500,
        "bin_capacity": 150,
        "seed": 4444,
    },
    "test_id": {
        "role": "test",
        "n_instances": 100,
        "n_items": 500,
        "bin_capacity": 150,
        "seed": 4445,
    },
    "test_ood_1000": {
        "role": "test",
        "n_instances": 100,
        "n_items": 1000,
        "bin_capacity": 150,
        "seed": 4446,
    },
}


def _generate_split(split: str, spec: dict[str, Any]) -> list[tuple[list[int], int]]:
    rng = np.random.RandomState(int(spec["seed"]))
    weights = rng.randint(20, 101, size=(int(spec["n_instances"]), int(spec["n_items"])))
    capacity = int(spec["bin_capacity"])
    return [(weights[i].astype(int).tolist(), capacity) for i in range(weights.shape[0])]


def write_default_dataset() -> dict[str, Any]:
    return write_pickle_splits(
        data_dir=DATA_DIR,
        dataset_id=DEFAULT_DATASET_ID,
        task="bp_1d_construct",
        version=1,
        description="Fixed offline 1D bin-packing constructive splits with C=150 and item sizes U[20,100].",
        split_specs=DEFAULT_SPLIT_SPECS,
        generate_split=_generate_split,
        generator="llm4ad.task.optimization.other.bp_1d_construct.dataset.write_default_dataset",
    )


def load_manifest() -> dict[str, Any]:
    from llm4ad.task.optimization.dataset_io import load_manifest as _load_manifest

    return _load_manifest(DATA_DIR)


def load_split_instances(split: str = DEFAULT_SPLIT) -> tuple[list[tuple[list[int], int]], dict[str, Any]]:
    return load_pickle_split(data_dir=DATA_DIR, split=split)
