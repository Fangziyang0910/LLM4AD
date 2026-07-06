from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_DATASET_ID = "tsp_construct_v1"
DEFAULT_SPLIT = "train"
DATA_DIR = Path(__file__).resolve().parent / "data"

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


def _manifest_path() -> Path:
    return DATA_DIR / "manifest.json"


def _split_path(filename: str) -> Path:
    return DATA_DIR / filename


def _generate_coordinates(n_instances: int, problem_size: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.rand(n_instances, problem_size, 2).astype(np.float64)


def _coordinates_to_instances(coordinates: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    instances = []
    for coords in coordinates:
        distances = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)
        instances.append((coords, distances))
    return instances


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_default_dataset() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    splits: dict[str, Any] = {}
    for split, spec in DEFAULT_SPLIT_SPECS.items():
        coordinates = _generate_coordinates(
            n_instances=spec["n_instances"],
            problem_size=spec["problem_size"],
            seed=spec["seed"],
        )
        split_path = _split_path(spec["filename"])
        np.savez_compressed(split_path, coordinates=coordinates)
        splits[split] = {
            **spec,
            "distribution": "uniform_unit_square",
            "coordinate_shape": list(coordinates.shape),
            "sha256": _file_sha256(split_path),
        }

    manifest = {
        "dataset_id": DEFAULT_DATASET_ID,
        "task": "tsp_construct",
        "version": 1,
        "description": "Fixed TSP constructive train/test splits for LLM4AD heuristic search.",
        "generator": "llm4ad.task.optimization.tsp_construct.dataset.write_default_dataset",
        "splits": splits,
    }

    with _manifest_path().open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return manifest


def load_manifest() -> dict[str, Any]:
    path = _manifest_path()
    if not path.exists():
        raise FileNotFoundError(
            f"TSP dataset manifest not found: {path}. "
            "Run `uv run python -m llm4ad.task.optimization.tsp_construct.generate_dataset`."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_split_coordinates(
    split: str = DEFAULT_SPLIT,
) -> tuple[np.ndarray, dict[str, Any]]:
    manifest = load_manifest()
    splits = manifest.get("splits", {})
    if split not in splits:
        available = ", ".join(sorted(splits))
        raise ValueError(f"Unknown TSP dataset split `{split}`. Available splits: {available}")

    split_info = splits[split]
    path = _split_path(split_info["filename"])
    if not path.exists():
        raise FileNotFoundError(f"TSP dataset split file not found: {path}")
    if _file_sha256(path) != split_info["sha256"]:
        raise ValueError(f"TSP dataset split checksum mismatch: {path}")

    with np.load(path) as data:
        coordinates = data["coordinates"].astype(np.float64, copy=False)

    metadata = {
        "dataset_id": manifest["dataset_id"],
        "task": manifest["task"],
        "split": split,
        "role": split_info["role"],
        "problem_size": split_info["problem_size"],
        "n_instances": split_info["n_instances"],
        "seed": split_info["seed"],
        "distribution": split_info["distribution"],
        "path": str(path),
        "sha256": split_info["sha256"],
    }
    return coordinates, metadata


def load_split_instances(
    split: str = DEFAULT_SPLIT,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    coordinates, metadata = _load_split_coordinates(split=split)
    return _coordinates_to_instances(coordinates), metadata
