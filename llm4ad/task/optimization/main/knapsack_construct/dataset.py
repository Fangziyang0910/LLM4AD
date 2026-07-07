from __future__ import annotations

from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import (
    DEFAULT_SPLIT,
    load_pickle_split,
    task_data_dir,
    write_pickle_splits,
)

DEFAULT_DATASET_ID = "knapsack_construct_v1"
DATA_DIR = task_data_dir(__file__)

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "n_instances": 64,
        "n_items": 100,
        "capacity": 25.0,
        "seed": 3333,
    },
    "test_id": {
        "role": "test",
        "n_instances": 250,
        "n_items": 100,
        "capacity": 25.0,
        "seed": 3334,
    },
    "test_ood_50": {
        "role": "test",
        "n_instances": 250,
        "n_items": 50,
        "capacity": 12.5,
        "seed": 3335,
    },
    "test_ood_200": {
        "role": "test",
        "n_instances": 250,
        "n_items": 200,
        "capacity": 25.0,
        "seed": 3336,
    },
    "test_ood_500": {
        "role": "test",
        "n_instances": 250,
        "n_items": 500,
        "capacity": 25.0,
        "seed": 3337,
    },
}


def _generate_split(split: str, spec: dict[str, Any]) -> list[tuple[list[float], list[float], float]]:
    rng = np.random.RandomState(int(spec["seed"]))
    node_positions = rng.rand(int(spec["n_instances"]), int(spec["n_items"]), 2)
    capacity = float(spec["capacity"])
    return [
        (
            node_positions[i, :, 0].astype(float).tolist(),
            node_positions[i, :, 1].astype(float).tolist(),
            capacity,
        )
        for i in range(node_positions.shape[0])
    ]


def write_default_dataset() -> dict[str, Any]:
    return write_pickle_splits(
        data_dir=DATA_DIR,
        dataset_id=DEFAULT_DATASET_ID,
        task="knapsack_construct",
        version=1,
        description="Fixed KP constructive train/test splits following MCTS-AHD/PathWise settings.",
        split_specs=DEFAULT_SPLIT_SPECS,
        generate_split=_generate_split,
        generator="llm4ad.task.optimization.main.knapsack_construct.dataset.write_default_dataset",
    )


def load_manifest() -> dict[str, Any]:
    from llm4ad.task.optimization.dataset_io import load_manifest as _load_manifest

    return _load_manifest(DATA_DIR)


def load_split_instances(
        split: str = DEFAULT_SPLIT,
) -> tuple[list[tuple[list[float], list[float], float]], dict[str, Any]]:
    return load_pickle_split(data_dir=DATA_DIR, split=split)
