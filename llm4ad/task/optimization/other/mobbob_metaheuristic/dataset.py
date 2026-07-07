from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT, write_manifest

DEFAULT_DATASET_ID = "mobbob_metaheuristic_v1"
DATA_DIR = Path(__file__).resolve().parent / "data"

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "dim": 10,
        "budget": 10000,
        "n_runs": 3,
        "seed_start": 0,
        "function_names": ["zdt1", "zdt2", "zdt3", "zdt4"],
        "n_instances": 4,
        "note": "Original EoH search evaluator: ZDT1-4, dim=10, budget=10000, three seeds.",
    },
    "test_full": {
        "role": "test",
        "dim": 10,
        "budget": 5000,
        "n_runs": 5,
        "seed_start": 0,
        "function_names": ["zdt1", "zdt2", "zdt3", "zdt4"],
        "n_instances": 4,
        "note": "Post-hoc evaluator from the EoH example: ZDT1-4, dim=10, five seeds.",
    },
}


def zdt1(x: np.ndarray) -> np.ndarray:
    f1 = float(x[0])
    g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
    f2 = float(g * (1.0 - np.sqrt(f1 / g)))
    return np.array([f1, f2])


def zdt2(x: np.ndarray) -> np.ndarray:
    f1 = float(x[0])
    g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
    f2 = float(g * (1.0 - (f1 / g) ** 2))
    return np.array([f1, f2])


def zdt3(x: np.ndarray) -> np.ndarray:
    f1 = float(x[0])
    g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
    f2 = float(g * (1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1)))
    return np.array([f1, f2])


def zdt4(x: np.ndarray) -> np.ndarray:
    f1 = float(x[0])
    n = len(x)
    g = 1.0 + 10.0 * (n - 1) + float(
        np.sum(x[1:] ** 2 - 10.0 * np.cos(4.0 * np.pi * x[1:]))
    )
    f2 = float(g * (1.0 - np.sqrt(f1 / g)))
    return np.array([f1, f2])


FUNCTIONS = {
    "zdt1": zdt1,
    "zdt2": zdt2,
    "zdt3": zdt3,
    "zdt4": zdt4,
}


def _bounds_for(name: str, dim: int) -> tuple[np.ndarray, np.ndarray]:
    if name == "zdt4":
        lower = np.concatenate([[0.0], np.full(dim - 1, -5.0)])
        upper = np.concatenate([[1.0], np.full(dim - 1, 5.0)])
    else:
        lower = np.zeros(dim)
        upper = np.ones(dim)
    return lower, upper


def write_default_dataset() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_id": DEFAULT_DATASET_ID,
        "task": "mobbob_metaheuristic",
        "version": 1,
        "description": (
            "Fixed EoH multi-objective black-box metaheuristic design benchmark. "
            "The searched program returns a decision-space Pareto-front approximation."
        ),
        "generator": "llm4ad.task.optimization.other.mobbob_metaheuristic.dataset.write_default_dataset",
        "paper": "papers/EoH",
        "source": "reference_code/EoH/examples/mobbob_metaheuristic",
        "splits": DEFAULT_SPLIT_SPECS,
    }
    return write_manifest(DATA_DIR, manifest)


def load_manifest() -> dict[str, Any]:
    path = DATA_DIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"MoBBOB metaheuristic manifest not found: {path}. "
            "Run `uv run python -m llm4ad.task.optimization.other.mobbob_metaheuristic.generate_dataset`."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_split_instances(split: str = DEFAULT_SPLIT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_manifest()
    splits = manifest.get("splits", {})
    if split not in splits:
        available = ", ".join(sorted(splits))
        raise ValueError(f"Unknown MoBBOB metaheuristic split `{split}`. Available splits: {available}")

    split_info = splits[split]
    dim = int(split_info["dim"])
    instances = []
    for name in split_info["function_names"]:
        lower, upper = _bounds_for(name, dim)
        instances.append({
            "name": name,
            "func": FUNCTIONS[name],
            "dim": dim,
            "bounds": (lower, upper),
            "ref_point": np.array([1.1, 1.1]),
            "n_obj": 2,
        })

    metadata = {
        "dataset_id": manifest["dataset_id"],
        "task": manifest["task"],
        "split": split,
        **split_info,
    }
    return instances, metadata
