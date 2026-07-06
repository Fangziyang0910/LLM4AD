from __future__ import annotations

from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import (
    DEFAULT_SPLIT,
    load_pickle_split,
    task_data_dir,
    write_pickle_splits,
)

DEFAULT_DATASET_ID = "set_cover_construct_v1"
DATA_DIR = task_data_dir(__file__)

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "n_instances": 32,
        "n_elements": 50,
        "n_subsets": 50,
        "max_subset_size": 8,
        "seed": 7401,
    },
    "test_id": {
        "role": "test",
        "n_instances": 100,
        "n_elements": 50,
        "n_subsets": 50,
        "max_subset_size": 8,
        "seed": 7402,
    },
    "test_ood_100": {
        "role": "test",
        "n_instances": 100,
        "n_elements": 100,
        "n_subsets": 100,
        "max_subset_size": 12,
        "seed": 7403,
    },
}


def _generate_coverable_instance(rng: np.random.RandomState, n_elements: int, n_subsets: int, max_subset_size: int):
    universal_set = list(range(1, n_elements + 1))
    subsets: list[list[int]] = [[element] for element in universal_set]
    for _ in range(max(0, n_subsets - n_elements)):
        subset_size = rng.randint(1, max_subset_size + 1)
        subset = rng.choice(universal_set, size=subset_size, replace=False).astype(int).tolist()
        subsets.append(subset)
    rng.shuffle(subsets)
    return universal_set, subsets


def _generate_split(split: str, spec: dict[str, Any]):
    rng = np.random.RandomState(int(spec["seed"]))
    return [
        _generate_coverable_instance(
            rng,
            int(spec["n_elements"]),
            int(spec["n_subsets"]),
            int(spec["max_subset_size"]),
        )
        for _ in range(int(spec["n_instances"]))
    ]


def write_default_dataset() -> dict[str, Any]:
    return write_pickle_splits(
        data_dir=DATA_DIR,
        dataset_id=DEFAULT_DATASET_ID,
        task="set_cover_construct",
        version=1,
        description="Fixed coverable SCP constructive splits using the platform synthetic distribution.",
        split_specs=DEFAULT_SPLIT_SPECS,
        generate_split=_generate_split,
        generator="llm4ad.task.optimization.set_cover_construct.dataset.write_default_dataset",
    )


def load_manifest() -> dict[str, Any]:
    from llm4ad.task.optimization.dataset_io import load_manifest as _load_manifest

    return _load_manifest(DATA_DIR)


def load_split_instances(split: str = DEFAULT_SPLIT):
    return load_pickle_split(data_dir=DATA_DIR, split=split)
