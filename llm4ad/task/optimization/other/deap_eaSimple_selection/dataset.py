from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT, write_manifest

DEFAULT_DATASET_ID = "deap_eaSimple_selection_v1"
DATA_DIR = Path(__file__).resolve().parent / "data"

TRAIN_CONFIGS = [
    {"name": "sphere", "dim": 10, "bounds": [-5.12, 5.12]},
    {"name": "rastrigin", "dim": 10, "bounds": [-5.12, 5.12]},
    {"name": "ackley", "dim": 10, "bounds": [-32.768, 32.768]},
    {"name": "rosenbrock", "dim": 10, "bounds": [-2.048, 2.048]},
    {"name": "griewank", "dim": 10, "bounds": [-600.0, 600.0]},
]

TEST_CONFIGS = [
    {"name": "sphere", "dim": 10, "bounds": [-5.12, 5.12]},
    {"name": "sphere", "dim": 20, "bounds": [-5.12, 5.12]},
    {"name": "rastrigin", "dim": 10, "bounds": [-5.12, 5.12]},
    {"name": "rastrigin", "dim": 20, "bounds": [-5.12, 5.12]},
    {"name": "ackley", "dim": 10, "bounds": [-32.768, 32.768]},
    {"name": "ackley", "dim": 20, "bounds": [-32.768, 32.768]},
    {"name": "rosenbrock", "dim": 10, "bounds": [-2.048, 2.048]},
    {"name": "rosenbrock", "dim": 20, "bounds": [-2.048, 2.048]},
    {"name": "griewank", "dim": 10, "bounds": [-600.0, 600.0]},
    {"name": "griewank", "dim": 20, "bounds": [-600.0, 600.0]},
]

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "pop_size": 50,
        "n_gen": 100,
        "tournament_size": 3,
        "cxpb": 0.9,
        "mutpb": 0.1,
        "eta_c": 15.0,
        "eta_m": 20.0,
        "n_runs": 3,
        "configs": TRAIN_CONFIGS,
        "n_instances": len(TRAIN_CONFIGS),
        "note": "Original EoH search evaluator: five 10-D continuous benchmarks.",
    },
    "test_full": {
        "role": "test",
        "pop_size": 100,
        "n_gen": 200,
        "tournament_size": 3,
        "cxpb": 0.9,
        "mutpb": 0.1,
        "eta_c": 15.0,
        "eta_m": 20.0,
        "n_runs": 10,
        "configs": TEST_CONFIGS,
        "n_instances": len(TEST_CONFIGS),
        "note": "Original EoH post-hoc evaluator: 10-D and 20-D variants of each benchmark.",
    },
}


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
        + 20 + np.e
    )


def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2))


def griewank(x: np.ndarray) -> float:
    n = len(x)
    i = np.arange(1, n + 1, dtype=float)
    return float(np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(i))) + 1)


FUNCTIONS = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "ackley": ackley,
    "rosenbrock": rosenbrock,
    "griewank": griewank,
}


def write_default_dataset() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_id": DEFAULT_DATASET_ID,
        "task": "deap_eaSimple_selection",
        "version": 1,
        "description": (
            "Fixed EoH DEAP eaSimple parent-selection benchmark. The searched "
            "function selects parent indices from current fitness values."
        ),
        "generator": "llm4ad.task.optimization.other.deap_eaSimple_selection.dataset.write_default_dataset",
        "paper": "papers/EoH",
        "source": "reference_code/EoH/examples/deap_eaSimple_selection",
        "splits": DEFAULT_SPLIT_SPECS,
    }
    return write_manifest(DATA_DIR, manifest)


def load_manifest() -> dict[str, Any]:
    path = DATA_DIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"DEAP eaSimple selection manifest not found: {path}. "
            "Run `uv run python -m llm4ad.task.optimization.other.deap_eaSimple_selection.generate_dataset`."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_split_instances(split: str = DEFAULT_SPLIT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_manifest()
    splits = manifest.get("splits", {})
    if split not in splits:
        available = ", ".join(sorted(splits))
        raise ValueError(f"Unknown DEAP eaSimple selection split `{split}`. Available splits: {available}")

    split_info = splits[split]
    instances = []
    for config in split_info["configs"]:
        name = config["name"]
        instances.append({
            "name": name,
            "func": FUNCTIONS[name],
            "dim": int(config["dim"]),
            "bounds": tuple(float(value) for value in config["bounds"]),
        })

    metadata = {
        "dataset_id": manifest["dataset_id"],
        "task": manifest["task"],
        "split": split,
        **split_info,
    }
    return instances, metadata
