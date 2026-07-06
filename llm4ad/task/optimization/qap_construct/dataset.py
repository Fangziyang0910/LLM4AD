from __future__ import annotations

from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import (
    DEFAULT_SPLIT,
    load_pickle_split,
    task_data_dir,
    write_pickle_splits,
)

DEFAULT_DATASET_ID = "qap_construct_v1"
DATA_DIR = task_data_dir(__file__)

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "n_instances": 16,
        "n_facilities": 20,
        "seed": 7301,
    },
    "test_id": {
        "role": "test",
        "n_instances": 100,
        "n_facilities": 20,
        "seed": 7302,
    },
    "test_ood_30": {
        "role": "test",
        "n_instances": 64,
        "n_facilities": 30,
        "seed": 7303,
    },
    "test_ood_50": {
        "role": "test",
        "n_instances": 32,
        "n_facilities": 50,
        "seed": 7304,
    },
}


def _symmetric_zero_diag(rng: np.random.RandomState, n: int) -> np.ndarray:
    matrix = rng.randint(1, 101, size=(n, n))
    matrix = (matrix + matrix.T) // 2
    np.fill_diagonal(matrix, 0)
    return matrix.astype(int)


def _generate_split(split: str, spec: dict[str, Any]):
    rng = np.random.RandomState(int(spec["seed"]))
    instances = []
    for _ in range(int(spec["n_instances"])):
        n = int(spec["n_facilities"])
        instances.append((_symmetric_zero_diag(rng, n), _symmetric_zero_diag(rng, n)))
    return instances


def write_default_dataset() -> dict[str, Any]:
    return write_pickle_splits(
        data_dir=DATA_DIR,
        dataset_id=DEFAULT_DATASET_ID,
        task="qap_construct",
        version=1,
        description="Fixed QAP constructive splits using symmetric flow and distance matrices.",
        split_specs=DEFAULT_SPLIT_SPECS,
        generate_split=_generate_split,
        generator="llm4ad.task.optimization.qap_construct.dataset.write_default_dataset",
    )


def load_manifest() -> dict[str, Any]:
    from llm4ad.task.optimization.dataset_io import load_manifest as _load_manifest

    return _load_manifest(DATA_DIR)


def load_split_instances(split: str = DEFAULT_SPLIT):
    return load_pickle_split(data_dir=DATA_DIR, split=split)
