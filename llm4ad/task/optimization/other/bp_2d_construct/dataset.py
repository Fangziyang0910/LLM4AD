from __future__ import annotations

from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import (
    DEFAULT_SPLIT,
    load_pickle_split,
    task_data_dir,
    write_pickle_splits,
)

DEFAULT_DATASET_ID = "bp_2d_construct_v1"
DATA_DIR = task_data_dir(__file__)

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "n_instances": 16,
        "n_items": 100,
        "bin_width": 100,
        "bin_height": 100,
        "seed": 7001,
    },
    "test_id": {
        "role": "test",
        "n_instances": 100,
        "n_items": 100,
        "bin_width": 100,
        "bin_height": 100,
        "seed": 7002,
    },
    "test_ood_200": {
        "role": "test",
        "n_instances": 100,
        "n_items": 200,
        "bin_width": 100,
        "bin_height": 100,
        "seed": 7003,
    },
}


def _generate_split(split: str, spec: dict[str, Any]) -> list[tuple[list[tuple[int, int]], tuple[int, int]]]:
    rng = np.random.RandomState(int(spec["seed"]))
    n_instances = int(spec["n_instances"])
    n_items = int(spec["n_items"])
    bin_width = int(spec["bin_width"])
    bin_height = int(spec["bin_height"])
    instances = []
    for _ in range(n_instances):
        item_widths = rng.randint(10, bin_width - 10, size=n_items)
        item_heights = rng.randint(10, bin_height - 10, size=n_items)
        item_dimensions = list(zip(item_widths.astype(int).tolist(), item_heights.astype(int).tolist()))
        instances.append((item_dimensions, (bin_width, bin_height)))
    return instances


def write_default_dataset() -> dict[str, Any]:
    return write_pickle_splits(
        data_dir=DATA_DIR,
        dataset_id=DEFAULT_DATASET_ID,
        task="bp_2d_construct",
        version=1,
        description="Fixed 2D bin-packing constructive splits using the platform item-size distribution.",
        split_specs=DEFAULT_SPLIT_SPECS,
        generate_split=_generate_split,
        generator="llm4ad.task.optimization.other.bp_2d_construct.dataset.write_default_dataset",
    )


def load_manifest() -> dict[str, Any]:
    from llm4ad.task.optimization.dataset_io import load_manifest as _load_manifest

    return _load_manifest(DATA_DIR)


def load_split_instances(split: str = DEFAULT_SPLIT):
    return load_pickle_split(data_dir=DATA_DIR, split=split)
