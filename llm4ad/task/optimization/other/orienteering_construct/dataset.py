from __future__ import annotations

from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import (
    DEFAULT_SPLIT,
    load_pickle_split,
    task_data_dir,
    write_pickle_splits,
)

DEFAULT_DATASET_ID = "orienteering_construct_v1"
DATA_DIR = task_data_dir(__file__)

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "n_instances": 10,
        "problem_size": 50,
        "max_length": 3.0,
        "seed": 6301,
    },
    "test_id": {
        "role": "test",
        "n_instances": 100,
        "problem_size": 50,
        "max_length": 3.0,
        "seed": 6302,
    },
    "test_ood_100": {
        "role": "test",
        "n_instances": 100,
        "problem_size": 100,
        "max_length": 4.0,
        "seed": 6303,
    },
    "test_ood_200": {
        "role": "test",
        "n_instances": 100,
        "problem_size": 200,
        "max_length": 5.0,
        "seed": 6304,
    },
    "test_ood_500": {
        "role": "test",
        "n_instances": 100,
        "problem_size": 500,
        "max_length": 8.0,
        "seed": 6305,
    },
}


def _generate_split(split: str, spec: dict[str, Any]):
    rng = np.random.default_rng(int(spec["seed"]))
    instances = []
    n_instances = int(spec["n_instances"])
    problem_size = int(spec["problem_size"])
    max_length = float(spec["max_length"])
    for _ in range(n_instances):
        coordinates = rng.random((problem_size, 2))
        coordinates[0] = np.array([0.5, 0.5])
        d0i = np.linalg.norm(coordinates[0] - coordinates, axis=1)
        denominator = np.max(d0i[1:]) if problem_size > 1 else 1.0
        prizes = (1 + np.floor(99 * d0i / denominator)) / 100
        prizes[0] = 0.0
        instances.append({
            "coordinates": coordinates,
            "prizes": prizes,
            "start_node": 0,
            "end_node": 0,
            "max_length": max_length,
        })
    return instances


def write_default_dataset() -> dict[str, Any]:
    return write_pickle_splits(
        data_dir=DATA_DIR,
        dataset_id=DEFAULT_DATASET_ID,
        task="orienteering_construct",
        version=1,
        description="Fixed OP constructive splits following DeepACO/ReEvo prize and length settings.",
        split_specs=DEFAULT_SPLIT_SPECS,
        generate_split=_generate_split,
        generator="llm4ad.task.optimization.other.orienteering_construct.dataset.write_default_dataset",
    )


def load_manifest() -> dict[str, Any]:
    from llm4ad.task.optimization.dataset_io import load_manifest as _load_manifest

    return _load_manifest(DATA_DIR)


def load_split_instances(split: str = DEFAULT_SPLIT):
    instances, metadata = load_pickle_split(data_dir=DATA_DIR, split=split)
    for instance in instances:
        coordinates = instance["coordinates"]
        instance["distance_matrix"] = np.linalg.norm(
            coordinates[:, np.newaxis] - coordinates,
            axis=2,
        )
    return instances, metadata
