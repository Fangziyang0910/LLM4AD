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

DEFAULT_DATASET_ID = "bpp_offline_aco_v1"
DATA_DIR = task_data_dir(__file__)

CAPACITY = 150
SOURCE_FILES = {
    "train": "train500_dataset.npz",
    "val_120": "val120_dataset.npz",
    "val_500": "val500_dataset.npz",
    "val_1000": "val1000_dataset.npz",
    "test_500": "test500_dataset.npz",
    "test_1000": "test1000_dataset.npz",
}

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "source": "mcts_ahd_bpp_offline_aco_dataset",
        "source_file": SOURCE_FILES["train"],
        "n_instances": 5,
        "n_items": 500,
        "capacity": CAPACITY,
        "filename": "train.pkl.gz",
    },
    "val_120": {
        "role": "validation",
        "source": "mcts_ahd_bpp_offline_aco_dataset",
        "source_file": SOURCE_FILES["val_120"],
        "n_instances": 64,
        "n_items": 120,
        "capacity": CAPACITY,
        "filename": "val_120.pkl.gz",
    },
    "val_500": {
        "role": "validation",
        "source": "mcts_ahd_bpp_offline_aco_dataset",
        "source_file": SOURCE_FILES["val_500"],
        "n_instances": 64,
        "n_items": 500,
        "capacity": CAPACITY,
        "filename": "val_500.pkl.gz",
    },
    "val_1000": {
        "role": "validation",
        "source": "mcts_ahd_bpp_offline_aco_dataset",
        "source_file": SOURCE_FILES["val_1000"],
        "n_instances": 64,
        "n_items": 1000,
        "capacity": CAPACITY,
        "filename": "val_1000.pkl.gz",
    },
    "test_500": {
        "role": "test",
        "source": "mcts_ahd_bpp_offline_aco_dataset",
        "source_file": SOURCE_FILES["test_500"],
        "n_instances": 64,
        "n_items": 500,
        "capacity": CAPACITY,
        "filename": "test_500.pkl.gz",
    },
    "test_1000": {
        "role": "test",
        "source": "mcts_ahd_bpp_offline_aco_dataset",
        "source_file": SOURCE_FILES["test_1000"],
        "n_instances": 64,
        "n_items": 1000,
        "capacity": CAPACITY,
        "filename": "test_1000.pkl.gz",
    },
}


def default_source_dir() -> Path:
    return (
        Path(__file__).resolve().parents[5]
        / "reference_code"
        / "MCTS-AHD-master"
        / "problems"
        / "bpp_offline_aco"
        / "dataset"
    )


def _generate_split_from_source(source_dir: Path):
    def generate_split(split: str, spec: dict[str, Any]):
        path = source_dir / spec["source_file"]
        if not path.exists():
            raise FileNotFoundError(f"BPP offline ACO source split not found: {path}")
        with np.load(path) as data:
            return data["demands"].astype(int)

    return generate_split


def write_default_dataset(source_dir: str | Path | None = None) -> dict[str, Any]:
    source_path = Path(source_dir) if source_dir is not None else default_source_dir()
    if not source_path.exists():
        raise FileNotFoundError(
            f"BPP offline ACO source data not found: {source_path}. "
            "Set source_dir to MCTS-AHD-master/problems/bpp_offline_aco/dataset."
        )

    return write_pickle_splits(
        data_dir=DATA_DIR,
        dataset_id=DEFAULT_DATASET_ID,
        task="bpp_offline_aco",
        version=1,
        description="Fixed offline BPP ACO splits from the MCTS-AHD benchmark data.",
        split_specs=DEFAULT_SPLIT_SPECS,
        generate_split=_generate_split_from_source(source_path),
        generator="llm4ad.task.optimization.main.bpp_offline_aco.dataset.write_default_dataset",
    )


def load_manifest() -> dict[str, Any]:
    from llm4ad.task.optimization.dataset_io import load_manifest as _load_manifest

    return _load_manifest(DATA_DIR)


def load_split_instances(split: str = DEFAULT_SPLIT):
    return load_pickle_split(data_dir=DATA_DIR, split=split)
