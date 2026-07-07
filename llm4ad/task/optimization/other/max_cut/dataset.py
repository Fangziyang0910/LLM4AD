from __future__ import annotations

from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import (
    DEFAULT_SPLIT,
    load_pickle_split,
    task_data_dir,
    write_pickle_splits,
)

DEFAULT_DATASET_ID = "max_cut_v1"
DATA_DIR = task_data_dir(__file__)

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "n_instances": 32,
        "n_nodes": 100,
        "edge_probability": 0.5,
        "seed": 6401,
    },
    "test_id": {
        "role": "test",
        "n_instances": 100,
        "n_nodes": 100,
        "edge_probability": 0.5,
        "seed": 6402,
    },
    "test_ood_200": {
        "role": "test",
        "n_instances": 100,
        "n_nodes": 200,
        "edge_probability": 0.5,
        "seed": 6403,
    },
    "test_ood_sparse": {
        "role": "test",
        "n_instances": 100,
        "n_nodes": 100,
        "edge_probability": 0.2,
        "seed": 6404,
    },
}


def _generate_split(split: str, spec: dict[str, Any]):
    rng = np.random.RandomState(int(spec["seed"]))
    n_instances = int(spec["n_instances"])
    n_nodes = int(spec["n_nodes"])
    edge_probability = float(spec["edge_probability"])
    instances = []
    for _ in range(n_instances):
        adjacency = {i: {} for i in range(n_nodes)}
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if rng.rand() < edge_probability:
                    weight = float(rng.uniform(0.5, 2.0))
                    adjacency[i][j] = weight
                    adjacency[j][i] = weight
        initial_order = list(rng.permutation(n_nodes).astype(int))
        instances.append((adjacency, initial_order))
    return instances


def write_default_dataset() -> dict[str, Any]:
    return write_pickle_splits(
        data_dir=DATA_DIR,
        dataset_id=DEFAULT_DATASET_ID,
        task="max_cut",
        version=1,
        description="Fixed MaxCut graph splits with stored local-search initial orders.",
        split_specs=DEFAULT_SPLIT_SPECS,
        generate_split=_generate_split,
        generator="llm4ad.task.optimization.other.max_cut.dataset.write_default_dataset",
    )


def load_manifest() -> dict[str, Any]:
    from llm4ad.task.optimization.dataset_io import load_manifest as _load_manifest

    return _load_manifest(DATA_DIR)


def load_split_instances(split: str = DEFAULT_SPLIT):
    return load_pickle_split(data_dir=DATA_DIR, split=split)
