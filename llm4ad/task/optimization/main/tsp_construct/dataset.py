from __future__ import annotations

from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import (
    DEFAULT_SPLIT,
    load_manifest as load_dataset_manifest,
    load_npz_split,
    task_data_dir,
    write_npz_splits,
)

DEFAULT_DATASET_ID = "tsp_construct_v1"
DATA_DIR = task_data_dir(__file__)

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "filename": "train.npz",
        "n_instances": 64,
        "problem_size": 50,
        "seed": 2024,
    },
    "test_id": {
        "role": "test",
        "filename": "test_id.npz",
        "n_instances": 250,
        "problem_size": 50,
        "seed": 2025,
    },
    "test_ood_100": {
        "role": "test",
        "filename": "test_ood_100.npz",
        "n_instances": 250,
        "problem_size": 100,
        "seed": 2026,
    },
    "test_ood_200": {
        "role": "test",
        "filename": "test_ood_200.npz",
        "n_instances": 250,
        "problem_size": 200,
        "seed": 2027,
    },
}


def _generate_coordinates(n_instances: int, problem_size: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.rand(n_instances, problem_size, 2).astype(np.float64)


def _coordinates_to_instances(coordinates: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    instances = []
    for coords in coordinates:
        distances = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)
        instances.append((coords, distances))
    return instances


def write_default_dataset() -> dict[str, Any]:
    def generate_split(split: str, spec: dict[str, Any]):
        coordinates = _generate_coordinates(
            n_instances=spec["n_instances"],
            problem_size=spec["problem_size"],
            seed=spec["seed"],
        )
        return {"coordinates": coordinates}, {
            "distribution": "uniform_unit_square",
            "coordinate_shape": list(coordinates.shape),
        }

    return write_npz_splits(
        data_dir=DATA_DIR,
        dataset_id=DEFAULT_DATASET_ID,
        task="tsp_construct",
        version=1,
        description="Fixed TSP constructive train/test splits for LLM4AD heuristic search.",
        split_specs=DEFAULT_SPLIT_SPECS,
        generate_split=generate_split,
        generator="llm4ad.task.optimization.main.tsp_construct.dataset.write_default_dataset",
    )


def load_manifest() -> dict[str, Any]:
    return load_dataset_manifest(DATA_DIR)


def _load_split_coordinates(
    split: str = DEFAULT_SPLIT,
) -> tuple[np.ndarray, dict[str, Any]]:
    arrays, metadata = load_npz_split(data_dir=DATA_DIR, split=split)
    return arrays["coordinates"].astype(np.float64, copy=False), metadata


def load_split_instances(
    split: str = DEFAULT_SPLIT,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    coordinates, metadata = _load_split_coordinates(split=split)
    return _coordinates_to_instances(coordinates), metadata
