from __future__ import annotations

from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import (
    DEFAULT_SPLIT,
    load_pickle_split,
    task_data_dir,
    write_pickle_splits,
)

DEFAULT_DATASET_ID = "online_bin_packing_v1"
DATA_DIR = task_data_dir(__file__)

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "seed": 5555,
        "scenarios": [
            {"n_items": 1000, "capacity": 100, "n_instances": 1},
            {"n_items": 1000, "capacity": 500, "n_instances": 1},
            {"n_items": 5000, "capacity": 100, "n_instances": 1},
            {"n_items": 5000, "capacity": 500, "n_instances": 1},
        ],
    },
    "test_id": {
        "role": "test",
        "seed": 5556,
        "scenarios": [
            {"n_items": 1000, "capacity": 100, "n_instances": 10},
            {"n_items": 1000, "capacity": 500, "n_instances": 10},
            {"n_items": 5000, "capacity": 100, "n_instances": 10},
            {"n_items": 5000, "capacity": 500, "n_instances": 10},
        ],
    },
    "test_ood_10000": {
        "role": "test",
        "seed": 5557,
        "scenarios": [
            {"n_items": 10000, "capacity": 100, "n_instances": 10},
            {"n_items": 10000, "capacity": 500, "n_instances": 10},
        ],
    },
}


def _generate_items(rng: np.random.RandomState, n_items: int, capacity: int) -> np.ndarray:
    samples = rng.weibull(3, n_items) * 45
    samples = np.clip(samples, 1, capacity)
    return np.round(samples).astype(int)


def _generate_split(split: str, spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rng = np.random.RandomState(int(spec["seed"]))
    dataset: dict[str, dict[str, Any]] = {}
    for scenario in spec["scenarios"]:
        n_items = int(scenario["n_items"])
        capacity = int(scenario["capacity"])
        n_instances = int(scenario["n_instances"])
        for idx in range(n_instances):
            key = f"{n_items}_c{capacity}_{idx}"
            dataset[key] = {
                "capacity": capacity,
                "num_items": n_items,
                "items": _generate_items(rng, n_items, capacity),
            }
    return dataset


def write_default_dataset() -> dict[str, Any]:
    return write_pickle_splits(
        data_dir=DATA_DIR,
        dataset_id=DEFAULT_DATASET_ID,
        task="online_bin_packing",
        version=1,
        description="Fixed Weibull online bin-packing splits following EoH/MCTS-AHD/PathWise scale settings.",
        split_specs=DEFAULT_SPLIT_SPECS,
        generate_split=_generate_split,
        generator="llm4ad.task.optimization.online_bin_packing.dataset.write_default_dataset",
    )


def load_manifest() -> dict[str, Any]:
    from llm4ad.task.optimization.dataset_io import load_manifest as _load_manifest

    return _load_manifest(DATA_DIR)


def load_split_instances(split: str = DEFAULT_SPLIT) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    return load_pickle_split(data_dir=DATA_DIR, split=split)
