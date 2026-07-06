from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import (
    DEFAULT_SPLIT,
    load_pickle_split,
    task_data_dir,
    write_pickle_splits,
)

DEFAULT_DATASET_ID = "cvrp_aco_v1"
DATA_DIR = task_data_dir(__file__)

CAPACITY = 50
SOURCE_FILES = {
    "train": "train50_dataset.npy",
    "val_20": "val20_dataset.npy",
    "val_50": "val50_dataset.npy",
    "val_100": "val100_dataset.npy",
    "test_20": "test20_dataset.npy",
    "test_50": "test50_dataset.npy",
    "test_100": "test100_dataset.npy",
}

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "source": "mcts_ahd_cvrp_aco_dataset",
        "source_file": SOURCE_FILES["train"],
        "n_instances": 10,
        "problem_size": 50,
        "capacity": CAPACITY,
        "filename": "train.pkl.gz",
    },
    "val_20": {
        "role": "validation",
        "source": "mcts_ahd_cvrp_aco_dataset",
        "source_file": SOURCE_FILES["val_20"],
        "n_instances": 64,
        "problem_size": 20,
        "capacity": CAPACITY,
        "filename": "val_20.pkl.gz",
    },
    "val_50": {
        "role": "validation",
        "source": "mcts_ahd_cvrp_aco_dataset",
        "source_file": SOURCE_FILES["val_50"],
        "n_instances": 64,
        "problem_size": 50,
        "capacity": CAPACITY,
        "filename": "val_50.pkl.gz",
    },
    "val_100": {
        "role": "validation",
        "source": "mcts_ahd_cvrp_aco_dataset",
        "source_file": SOURCE_FILES["val_100"],
        "n_instances": 64,
        "problem_size": 100,
        "capacity": CAPACITY,
        "filename": "val_100.pkl.gz",
    },
    "test_20": {
        "role": "test",
        "source": "mcts_ahd_cvrp_aco_dataset",
        "source_file": SOURCE_FILES["test_20"],
        "n_instances": 64,
        "problem_size": 20,
        "capacity": CAPACITY,
        "filename": "test_20.pkl.gz",
    },
    "test_50": {
        "role": "test",
        "source": "mcts_ahd_cvrp_aco_dataset",
        "source_file": SOURCE_FILES["test_50"],
        "n_instances": 64,
        "problem_size": 50,
        "capacity": CAPACITY,
        "filename": "test_50.pkl.gz",
    },
    "test_100": {
        "role": "test",
        "source": "mcts_ahd_cvrp_aco_dataset",
        "source_file": SOURCE_FILES["test_100"],
        "n_instances": 64,
        "problem_size": 100,
        "capacity": CAPACITY,
        "filename": "test_100.pkl.gz",
    },
}


def default_source_dir() -> Path:
    return (
        Path(__file__).resolve().parents[5]
        / "reference_code"
        / "MCTS-AHD-master"
        / "problems"
        / "cvrp_aco"
        / "dataset"
    )


def _generate_split_from_source(source_dir: Path):
    def generate_split(split: str, spec: dict[str, Any]):
        path = source_dir / spec["source_file"]
        if not path.exists():
            raise FileNotFoundError(f"CVRP-ACO source split not found: {path}")
        dataset = np.load(path).astype(np.float64)
        return dataset

    return generate_split


def write_default_dataset(source_dir: str | Path | None = None) -> dict[str, Any]:
    source_path = Path(source_dir) if source_dir is not None else default_source_dir()
    if not source_path.exists():
        raise FileNotFoundError(
            f"CVRP-ACO source data not found: {source_path}. "
            "Set source_dir to MCTS-AHD-master/problems/cvrp_aco/dataset."
        )

    return write_pickle_splits(
        data_dir=DATA_DIR,
        dataset_id=DEFAULT_DATASET_ID,
        task="cvrp_aco",
        version=1,
        description="Fixed CVRP-ACO splits from the MCTS-AHD CVRP-ACO benchmark data.",
        split_specs=DEFAULT_SPLIT_SPECS,
        generate_split=_generate_split_from_source(source_path),
        generator="llm4ad.task.optimization.cvrp_aco.dataset.write_default_dataset",
    )


def load_manifest() -> dict[str, Any]:
    from llm4ad.task.optimization.dataset_io import load_manifest as _load_manifest

    return _load_manifest(DATA_DIR)


def load_split_instances(split: str = DEFAULT_SPLIT):
    return load_pickle_split(data_dir=DATA_DIR, split=split)
