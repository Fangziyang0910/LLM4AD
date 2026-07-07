from __future__ import annotations

from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import (
    DEFAULT_SPLIT,
    load_pickle_split,
    task_data_dir,
    write_pickle_splits,
)

DEFAULT_DATASET_ID = "ovrp_construct_v1"
DATA_DIR = task_data_dir(__file__)

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "n_instances": 16,
        "problem_size": 50,
        "capacity": 40,
        "seed": 6101,
    },
    "test_id": {
        "role": "test",
        "n_instances": 100,
        "problem_size": 50,
        "capacity": 40,
        "seed": 6102,
    },
    "test_ood_100": {
        "role": "test",
        "n_instances": 100,
        "problem_size": 100,
        "capacity": 40,
        "seed": 6103,
    },
}


def _generate_split(split: str, spec: dict[str, Any]):
    rng = np.random.RandomState(int(spec["seed"]))
    n_instances = int(spec["n_instances"])
    n_customers = int(spec["problem_size"])
    n_nodes = n_customers + 1
    capacity = int(spec["capacity"])
    instances = []
    for _ in range(n_instances):
        coordinates = rng.rand(n_nodes, 2)
        coordinates[0] = np.array([0.5, 0.5])
        demands = np.append(np.array([0]), rng.randint(1, 10, size=n_customers))
        distances = np.linalg.norm(coordinates[:, np.newaxis] - coordinates, axis=2)
        instances.append((coordinates, distances, demands, capacity))
    return instances


def write_default_dataset() -> dict[str, Any]:
    return write_pickle_splits(
        data_dir=DATA_DIR,
        dataset_id=DEFAULT_DATASET_ID,
        task="ovrp_construct",
        version=1,
        description="Fixed OVRP constructive train/test splits using the platform VRP distribution.",
        split_specs=DEFAULT_SPLIT_SPECS,
        generate_split=_generate_split,
        generator="llm4ad.task.optimization.other.ovrp_construct.dataset.write_default_dataset",
    )


def load_manifest() -> dict[str, Any]:
    from llm4ad.task.optimization.dataset_io import load_manifest as _load_manifest

    return _load_manifest(DATA_DIR)


def load_split_instances(split: str = DEFAULT_SPLIT):
    return load_pickle_split(data_dir=DATA_DIR, split=split)
