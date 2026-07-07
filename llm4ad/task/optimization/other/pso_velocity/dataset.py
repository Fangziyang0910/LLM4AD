from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT, write_manifest

DEFAULT_DATASET_ID = "pso_velocity_v1"
DATA_DIR = Path(__file__).resolve().parent / "data"


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def rastrigin(x: np.ndarray) -> float:
    n = len(x)
    return float(10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    n = len(x)
    return float(
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n))
        - np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
        + 20
        + np.e
    )


def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2))


def griewank(x: np.ndarray) -> float:
    n = len(x)
    i = np.arange(1, n + 1, dtype=float)
    return float(np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(i))) + 1)


FUNCTIONS: dict[str, Callable[[np.ndarray], float]] = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "ackley": ackley,
    "rosenbrock": rosenbrock,
    "griewank": griewank,
}

BASE_CONFIGS = [
    {"name": "sphere", "bounds": [-5.12, 5.12]},
    {"name": "rastrigin", "bounds": [-5.12, 5.12]},
    {"name": "ackley", "bounds": [-32.768, 32.768]},
    {"name": "rosenbrock", "bounds": [-2.048, 2.048]},
    {"name": "griewank", "bounds": [-600.0, 600.0]},
]

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "dimensions": [10],
        "function_names": [config["name"] for config in BASE_CONFIGS],
        "pop_size": 30,
        "max_iterations": 200,
        "n_runs": 3,
        "seed_start": 0,
        "w": 0.729,
        "c1": 1.494,
        "c2": 1.494,
        "v_max_ratio": 0.2,
        "n_instances": 5,
        "note": "Original EoH search evaluator: five 10D continuous benchmark functions.",
    },
    "test_full": {
        "role": "test",
        "dimensions": [10, 20],
        "function_names": [config["name"] for config in BASE_CONFIGS],
        "pop_size": 50,
        "max_iterations": 500,
        "n_runs": 10,
        "seed_start": 0,
        "w": 0.729,
        "c1": 1.494,
        "c2": 1.494,
        "v_max_ratio": 0.2,
        "n_instances": 10,
        "note": "Post-hoc evaluator from the EoH example: 10D and 20D variants.",
    },
    "test_ood_20d": {
        "role": "test",
        "dimensions": [20],
        "function_names": [config["name"] for config in BASE_CONFIGS],
        "pop_size": 50,
        "max_iterations": 500,
        "n_runs": 10,
        "seed_start": 0,
        "w": 0.729,
        "c1": 1.494,
        "c2": 1.494,
        "v_max_ratio": 0.2,
        "n_instances": 5,
        "note": "Dimension-generalization subset using only 20D benchmark variants.",
    },
}


def write_default_dataset() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_id": DEFAULT_DATASET_ID,
        "task": "pso_velocity",
        "version": 1,
        "description": (
            "Fixed EoH PSO velocity-update benchmark. The searched function updates "
            "particle velocities inside a standard PSO loop on continuous benchmarks."
        ),
        "generator": "llm4ad.task.optimization.other.pso_velocity.dataset.write_default_dataset",
        "paper": "papers/EoH",
        "source": "reference_code/EoH/examples/pso_velocity",
        "splits": DEFAULT_SPLIT_SPECS,
    }
    return write_manifest(DATA_DIR, manifest)


def load_manifest() -> dict[str, Any]:
    path = DATA_DIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"PSO velocity manifest not found: {path}. "
            "Run `uv run python -m llm4ad.task.optimization.other.pso_velocity.generate_dataset`."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_split_instances(split: str = DEFAULT_SPLIT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_manifest()
    splits = manifest.get("splits", {})
    if split not in splits:
        available = ", ".join(sorted(splits))
        raise ValueError(f"Unknown PSO velocity split `{split}`. Available splits: {available}")

    split_info = splits[split]
    instances: list[dict[str, Any]] = []
    for dim in split_info["dimensions"]:
        for config in BASE_CONFIGS:
            if config["name"] not in split_info["function_names"]:
                continue
            instances.append({
                "name": config["name"],
                "func": FUNCTIONS[config["name"]],
                "dim": int(dim),
                "bounds": tuple(float(v) for v in config["bounds"]),
            })

    metadata = {
        "dataset_id": manifest["dataset_id"],
        "task": manifest["task"],
        "split": split,
        **split_info,
    }
    return instances, metadata
