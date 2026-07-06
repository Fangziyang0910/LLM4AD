from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT
from llm4ad.task.optimization.tsp_gls_2O.get_instance import TSPInstance

DEFAULT_DATASET_ID = "tsp_gls_2O_v1"
DATA_DIR = Path(__file__).resolve().parent / "data"

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "filename": "train.npz",
        "n_instances": 10,
        "problem_size": 200,
        "seed": 8001,
    },
    "test_id": {
        "role": "test",
        "filename": "test_id.npz",
        "n_instances": 250,
        "problem_size": 200,
        "seed": 8002,
    },
    "test_ood_100": {
        "role": "test",
        "filename": "test_ood_100.npz",
        "n_instances": 250,
        "problem_size": 100,
        "seed": 8003,
    },
    "test_ood_500": {
        "role": "test",
        "filename": "test_ood_500.npz",
        "n_instances": 64,
        "problem_size": 500,
        "seed": 8004,
    },
    "test_ood_1000": {
        "role": "test",
        "filename": "test_ood_1000.npz",
        "n_instances": 64,
        "problem_size": 1000,
        "seed": 8005,
    },
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generate_coordinates(n_instances: int, problem_size: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.rand(n_instances, problem_size, 2).astype(np.float64)


def write_default_dataset() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    splits: dict[str, Any] = {}
    for split, spec in DEFAULT_SPLIT_SPECS.items():
        coordinates = _generate_coordinates(
            n_instances=int(spec["n_instances"]),
            problem_size=int(spec["problem_size"]),
            seed=int(spec["seed"]),
        )
        path = DATA_DIR / spec["filename"]
        np.savez_compressed(path, coordinates=coordinates)
        splits[split] = {
            **spec,
            "distribution": "uniform_unit_square",
            "coordinate_shape": list(coordinates.shape),
            "sha256": _file_sha256(path),
        }

    manifest = {
        "dataset_id": DEFAULT_DATASET_ID,
        "task": "tsp_gls_2O",
        "version": 1,
        "description": "Fixed TSP GLS train/test splits following MCTS-AHD/PathWise GLS settings.",
        "generator": "llm4ad.task.optimization.tsp_gls_2O.dataset.write_default_dataset",
        "splits": splits,
    }
    with (DATA_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def load_manifest() -> dict[str, Any]:
    path = DATA_DIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"TSP GLS dataset manifest not found: {path}. "
            "Run `uv run python -m llm4ad.task.optimization.tsp_gls_2O.generate_dataset`."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_split_instances(split: str = DEFAULT_SPLIT) -> tuple[list[TSPInstance], dict[str, Any]]:
    manifest = load_manifest()
    splits = manifest.get("splits", {})
    if split not in splits:
        available = ", ".join(sorted(splits))
        raise ValueError(f"Unknown TSP GLS dataset split `{split}`. Available splits: {available}")

    split_info = splits[split]
    path = DATA_DIR / split_info["filename"]
    if not path.exists():
        raise FileNotFoundError(f"TSP GLS dataset split file not found: {path}")
    if _file_sha256(path) != split_info["sha256"]:
        raise ValueError(f"TSP GLS dataset split checksum mismatch: {path}")

    with np.load(path) as data:
        coordinates = data["coordinates"].astype(np.float64, copy=False)

    instances = [TSPInstance(coords) for coords in coordinates]
    metadata = {
        "dataset_id": manifest["dataset_id"],
        "task": manifest["task"],
        "split": split,
        "path": str(path),
        **split_info,
    }
    return instances, metadata
