from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT, write_manifest, file_sha256

DEFAULT_DATASET_ID = "tsp_gls_v1"
DATA_DIR = Path(__file__).resolve().parent / "data"
SOURCE_TASK_DIR = Path(__file__).resolve().parents[5] / "reference_code" / "EoH" / "examples" / "tsp_gls"

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "filename": "train.npz",
        "source_file": "TrainingData/TSPAEL64.pkl",
        "n_instances": 3,
        "problem_size": 100,
        "time_limit": 10.0,
        "ite_max": 1000,
        "perturbation_moves": 1,
        "note": "Original EoH search evaluator: first 3 instances from TSPAEL64.pkl.",
    },
    "test_full": {
        "role": "test",
        "filename": "test_full.npz",
        "source_file": "TestingData/TSP20.pkl",
        "n_instances": 16,
        "problem_size": 20,
        "time_limit": 10.0,
        "ite_max": 1000,
        "perturbation_moves": 1,
        "note": "Original EoH post-hoc evaluator: first 16 instances from TSP20.pkl.",
    },
}


def _load_source_dataset(source_file: str, n_instances: int) -> dict[str, np.ndarray]:
    path = SOURCE_TASK_DIR / source_file
    if not path.exists():
        raise FileNotFoundError(
            f"EoH TSP-GLS source data not found: {path}. "
            "The default committed LLM4AD data can still be loaded without regenerating it."
        )
    with path.open("rb") as handle:
        data = pickle.load(handle)
    return {
        "coordinates": np.asarray(data["coordinate"][:n_instances], dtype=float),
        "optimal_tours": np.asarray(data["optimal_tour"][:n_instances], dtype=int),
        "distance_matrices": np.asarray(data["distance_matrix"][:n_instances], dtype=float),
        "costs": np.asarray(data["cost"][:n_instances], dtype=float),
    }


def write_default_dataset() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    splits: dict[str, Any] = {}
    for split, spec in DEFAULT_SPLIT_SPECS.items():
        arrays = _load_source_dataset(
            source_file=spec["source_file"],
            n_instances=int(spec["n_instances"]),
        )
        path = DATA_DIR / spec["filename"]
        np.savez_compressed(path, **arrays)

        splits[split] = {
            **spec,
            "format": "npz",
            "coordinates_shape": list(arrays["coordinates"].shape),
            "optimal_tours_shape": list(arrays["optimal_tours"].shape),
            "distance_matrices_shape": list(arrays["distance_matrices"].shape),
            "costs_shape": list(arrays["costs"].shape),
            "sha256": file_sha256(path),
        }

    manifest = {
        "dataset_id": DEFAULT_DATASET_ID,
        "task": "tsp_gls",
        "version": 1,
        "description": (
            "Fixed EoH TSP Guided Local Search benchmark. The searched function "
            "updates edge distances to guide local search away from local optima."
        ),
        "generator": "llm4ad.task.optimization.main.tsp_gls.dataset.write_default_dataset",
        "paper": "papers/EoH",
        "source": "reference_code/EoH/examples/tsp_gls",
        "splits": splits,
    }
    return write_manifest(DATA_DIR, manifest)


def load_manifest() -> dict[str, Any]:
    path = DATA_DIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"TSP-GLS manifest not found: {path}. "
            "Run `uv run python -m llm4ad.task.optimization.main.tsp_gls.generate_dataset`."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_split_instances(split: str = DEFAULT_SPLIT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_manifest()
    splits = manifest.get("splits", {})
    if split not in splits:
        available = ", ".join(sorted(splits))
        raise ValueError(f"Unknown TSP-GLS split `{split}`. Available splits: {available}")

    split_info = splits[split]
    path = DATA_DIR / split_info["filename"]
    if not path.exists():
        raise FileNotFoundError(f"TSP-GLS dataset split not found: {path}")
    if file_sha256(path) != split_info["sha256"]:
        raise ValueError(f"TSP-GLS dataset checksum mismatch: {path}")

    with np.load(path) as data:
        coordinates = np.asarray(data["coordinates"], dtype=float)
        optimal_tours = np.asarray(data["optimal_tours"], dtype=int)
        distance_matrices = np.asarray(data["distance_matrices"], dtype=float)
        costs = np.asarray(data["costs"], dtype=float)

    instances = [
        {
            "instance_id": idx,
            "coordinates": coordinates[idx],
            "optimal_tour": optimal_tours[idx],
            "distance_matrix": distance_matrices[idx],
            "optimal_cost": float(costs[idx]),
        }
        for idx in range(distance_matrices.shape[0])
    ]
    metadata = {
        "dataset_id": manifest["dataset_id"],
        "task": manifest["task"],
        "split": split,
        "path": str(path),
        **split_info,
    }
    return instances, metadata
