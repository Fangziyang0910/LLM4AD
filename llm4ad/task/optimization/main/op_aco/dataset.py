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

DEFAULT_DATASET_ID = "op_aco_v1"
DATA_DIR = task_data_dir(__file__)

SOURCE_FILES = {
    "train": "train50_dataset.npz",
    "val_50": "val50_dataset.npz",
    "val_100": "val100_dataset.npz",
    "val_200": "val200_dataset.npz",
    "test_50": "test50_dataset.npz",
    "test_100": "test100_dataset.npz",
    "test_200": "test200_dataset.npz",
}

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "source": "hsevo_op_aco_dataset",
        "source_file": SOURCE_FILES["train"],
        "n_instances": 5,
        "problem_size": 50,
        "filename": "train.pkl.gz",
    },
    "val_50": {
        "role": "validation",
        "source": "hsevo_op_aco_dataset",
        "source_file": SOURCE_FILES["val_50"],
        "n_instances": 64,
        "problem_size": 50,
        "filename": "val_50.pkl.gz",
    },
    "val_100": {
        "role": "validation",
        "source": "hsevo_op_aco_dataset",
        "source_file": SOURCE_FILES["val_100"],
        "n_instances": 64,
        "problem_size": 100,
        "filename": "val_100.pkl.gz",
    },
    "val_200": {
        "role": "validation",
        "source": "hsevo_op_aco_dataset",
        "source_file": SOURCE_FILES["val_200"],
        "n_instances": 64,
        "problem_size": 200,
        "filename": "val_200.pkl.gz",
    },
    "test_50": {
        "role": "test",
        "source": "hsevo_op_aco_dataset",
        "source_file": SOURCE_FILES["test_50"],
        "n_instances": 64,
        "problem_size": 50,
        "filename": "test_50.pkl.gz",
    },
    "test_100": {
        "role": "test",
        "source": "hsevo_op_aco_dataset",
        "source_file": SOURCE_FILES["test_100"],
        "n_instances": 64,
        "problem_size": 100,
        "filename": "test_100.pkl.gz",
    },
    "test_200": {
        "role": "test",
        "source": "hsevo_op_aco_dataset",
        "source_file": SOURCE_FILES["test_200"],
        "n_instances": 64,
        "problem_size": 200,
        "filename": "test_200.pkl.gz",
    },
}


def default_source_dir() -> Path:
    return (
        Path(__file__).resolve().parents[5]
        / "reference_code"
        / "HSEvo"
        / "problems"
        / "op_aco"
        / "dataset"
    )


def _generate_split_from_source(source_dir: Path):
    def generate_split(split: str, spec: dict[str, Any]):
        path = source_dir / spec["source_file"]
        if not path.exists():
            raise FileNotFoundError(f"OP-ACO source split not found: {path}")
        with np.load(path) as data:
            return data["coordinates"].astype(np.float64)

    return generate_split


def write_default_dataset(source_dir: str | Path | None = None) -> dict[str, Any]:
    source_path = Path(source_dir) if source_dir is not None else default_source_dir()
    if not source_path.exists():
        raise FileNotFoundError(
            f"OP-ACO source data not found: {source_path}. "
            "Set source_dir to HSEvo/problems/op_aco/dataset."
        )

    return write_pickle_splits(
        data_dir=DATA_DIR,
        dataset_id=DEFAULT_DATASET_ID,
        task="op_aco",
        version=1,
        description="Fixed OP-ACO splits from the HSEvo OP-ACO benchmark data.",
        split_specs=DEFAULT_SPLIT_SPECS,
        generate_split=_generate_split_from_source(source_path),
        generator="llm4ad.task.optimization.main.op_aco.dataset.write_default_dataset",
    )


def load_manifest() -> dict[str, Any]:
    from llm4ad.task.optimization.dataset_io import load_manifest as _load_manifest

    return _load_manifest(DATA_DIR)


def load_split_instances(split: str = DEFAULT_SPLIT):
    return load_pickle_split(data_dir=DATA_DIR, split=split)
