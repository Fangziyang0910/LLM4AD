from __future__ import annotations

from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import (
    DEFAULT_SPLIT,
    load_pickle_split,
    task_data_dir,
    write_pickle_splits,
)

DEFAULT_DATASET_ID = "cflp_construct_v1"
DATA_DIR = task_data_dir(__file__)

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "n_instances": 16,
        "n_facilities": 50,
        "n_customers": 50,
        "max_capacity": 100,
        "max_demand": 20,
        "max_cost": 50,
        "seed": 7101,
    },
    "test_id": {
        "role": "test",
        "n_instances": 100,
        "n_facilities": 50,
        "n_customers": 50,
        "max_capacity": 100,
        "max_demand": 20,
        "max_cost": 50,
        "seed": 7102,
    },
    "test_ood_100": {
        "role": "test",
        "n_instances": 100,
        "n_facilities": 100,
        "n_customers": 100,
        "max_capacity": 100,
        "max_demand": 20,
        "max_cost": 50,
        "seed": 7103,
    },
}


def _generate_split(split: str, spec: dict[str, Any]):
    rng = np.random.RandomState(int(spec["seed"]))
    instances = []
    for _ in range(int(spec["n_instances"])):
        n_facilities = int(spec["n_facilities"])
        n_customers = int(spec["n_customers"])
        customer_demands = rng.randint(5, int(spec["max_demand"]) + 1, size=n_customers)
        facility_capacities = rng.randint(5, int(spec["max_capacity"]) + 1, size=n_facilities)
        while facility_capacities.sum() < customer_demands.sum():
            deficit = int(customer_demands.sum() - facility_capacities.sum())
            facility_capacities[rng.randint(0, n_facilities)] += min(deficit, int(spec["max_capacity"]))
        assignment_costs = rng.randint(
            5,
            int(spec["max_cost"]) + 1,
            size=(n_facilities, n_customers),
        )
        instances.append({
            "facility_capacities": facility_capacities.astype(int).tolist(),
            "customer_demands": customer_demands.astype(int).tolist(),
            "assignment_costs": assignment_costs.astype(int).tolist(),
        })
    return instances


def write_default_dataset() -> dict[str, Any]:
    return write_pickle_splits(
        data_dir=DATA_DIR,
        dataset_id=DEFAULT_DATASET_ID,
        task="cflp_construct",
        version=1,
        description="Fixed CFLP constructive train/test splits using the platform synthetic distribution.",
        split_specs=DEFAULT_SPLIT_SPECS,
        generate_split=_generate_split,
        generator="llm4ad.task.optimization.cflp_construct.dataset.write_default_dataset",
    )


def load_manifest() -> dict[str, Any]:
    from llm4ad.task.optimization.dataset_io import load_manifest as _load_manifest

    return _load_manifest(DATA_DIR)


def load_split_instances(split: str = DEFAULT_SPLIT):
    return load_pickle_split(data_dir=DATA_DIR, split=split)
