from __future__ import annotations

from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import (
    DEFAULT_SPLIT,
    load_pickle_split,
    task_data_dir,
    write_pickle_splits,
)

DEFAULT_DATASET_ID = "jssp_construct_v1"
DATA_DIR = task_data_dir(__file__)

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "n_instances": 16,
        "n_jobs": 50,
        "n_machines": 10,
        "seed": 7201,
    },
    "test_id": {
        "role": "test",
        "n_instances": 100,
        "n_jobs": 50,
        "n_machines": 10,
        "seed": 7202,
    },
    "test_ood_m20": {
        "role": "test",
        "n_instances": 50,
        "n_jobs": 50,
        "n_machines": 20,
        "seed": 7203,
    },
    "test_ood_100x20": {
        "role": "test",
        "n_instances": 50,
        "n_jobs": 100,
        "n_machines": 20,
        "seed": 7204,
    },
}


def _generate_split(split: str, spec: dict[str, Any]):
    rng = np.random.RandomState(int(spec["seed"]))
    instances = []
    for _ in range(int(spec["n_instances"])):
        processing_times = rng.randint(
            10,
            100,
            size=(int(spec["n_jobs"]), int(spec["n_machines"])),
        )
        instances.append((
            processing_times.astype(int).tolist(),
            int(spec["n_jobs"]),
            int(spec["n_machines"]),
        ))
    return instances


def write_default_dataset() -> dict[str, Any]:
    return write_pickle_splits(
        data_dir=DATA_DIR,
        dataset_id=DEFAULT_DATASET_ID,
        task="jssp_construct",
        version=1,
        description="Fixed JSSP constructive splits using the platform processing-time distribution.",
        split_specs=DEFAULT_SPLIT_SPECS,
        generate_split=_generate_split,
        generator="llm4ad.task.optimization.jssp_construct.dataset.write_default_dataset",
    )


def load_manifest() -> dict[str, Any]:
    from llm4ad.task.optimization.dataset_io import load_manifest as _load_manifest

    return _load_manifest(DATA_DIR)


def load_split_instances(split: str = DEFAULT_SPLIT):
    return load_pickle_split(data_dir=DATA_DIR, split=split)
