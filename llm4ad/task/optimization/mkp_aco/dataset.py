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

DEFAULT_DATASET_ID = "mkp_aco_v1"
DATA_DIR = task_data_dir(__file__)

N_DIMENSIONS = 5
SOURCE_FILES = {
    "train": "train100_dataset.npz",
    "val_100": "val100_dataset.npz",
    "val_300": "val300_dataset.npz",
    "val_500": "val500_dataset.npz",
    "test_100": "test100_dataset.npz",
    "test_200": "test200_dataset.npz",
    "test_300": "test300_dataset.npz",
    "test_500": "test500_dataset.npz",
    "test_1000": "test1000_dataset.npz",
}


def _spec(role: str, source_file: str, n_instances: int, n_items: int, filename: str) -> dict[str, Any]:
    return {
        "role": role,
        "source": "mcts_ahd_mkp_aco_dataset",
        "source_file": source_file,
        "n_instances": n_instances,
        "n_items": n_items,
        "n_dimensions": N_DIMENSIONS,
        "filename": filename,
    }


DEFAULT_SPLIT_SPECS = {
    "train": _spec("train", SOURCE_FILES["train"], 5, 100, "train.pkl.gz"),
    "val_100": _spec("validation", SOURCE_FILES["val_100"], 5, 100, "val_100.pkl.gz"),
    "val_300": _spec("validation", SOURCE_FILES["val_300"], 5, 300, "val_300.pkl.gz"),
    "val_500": _spec("validation", SOURCE_FILES["val_500"], 5, 500, "val_500.pkl.gz"),
    "test_100": _spec("test", SOURCE_FILES["test_100"], 64, 100, "test_100.pkl.gz"),
    "test_200": _spec("test", SOURCE_FILES["test_200"], 64, 200, "test_200.pkl.gz"),
    "test_300": _spec("test", SOURCE_FILES["test_300"], 64, 300, "test_300.pkl.gz"),
    "test_500": _spec("test", SOURCE_FILES["test_500"], 64, 500, "test_500.pkl.gz"),
    "test_1000": _spec("test", SOURCE_FILES["test_1000"], 64, 1000, "test_1000.pkl.gz"),
}


def default_source_dir() -> Path:
    return (
        Path(__file__).resolve().parents[5]
        / "reference_code"
        / "MCTS-AHD-master"
        / "problems"
        / "mkp_aco"
        / "dataset"
    )


def _generate_split_from_source(source_dir: Path):
    def generate_split(split: str, spec: dict[str, Any]):
        path = source_dir / spec["source_file"]
        if not path.exists():
            raise FileNotFoundError(f"MKP-ACO source split not found: {path}")
        with np.load(path) as data:
            return {
                "prizes": data["prizes"].astype(np.float64),
                "weights": data["weights"].astype(np.float64),
            }

    return generate_split


def write_default_dataset(source_dir: str | Path | None = None) -> dict[str, Any]:
    source_path = Path(source_dir) if source_dir is not None else default_source_dir()
    if not source_path.exists():
        raise FileNotFoundError(
            f"MKP-ACO source data not found: {source_path}. "
            "Set source_dir to MCTS-AHD-master/problems/mkp_aco/dataset."
        )

    return write_pickle_splits(
        data_dir=DATA_DIR,
        dataset_id=DEFAULT_DATASET_ID,
        task="mkp_aco",
        version=1,
        description="Fixed MKP-ACO splits from the MCTS-AHD benchmark data.",
        split_specs=DEFAULT_SPLIT_SPECS,
        generate_split=_generate_split_from_source(source_path),
        generator="llm4ad.task.optimization.mkp_aco.dataset.write_default_dataset",
    )


def load_manifest() -> dict[str, Any]:
    from llm4ad.task.optimization.dataset_io import load_manifest as _load_manifest

    return _load_manifest(DATA_DIR)


def load_split_instances(split: str = DEFAULT_SPLIT):
    return load_pickle_split(data_dir=DATA_DIR, split=split)
