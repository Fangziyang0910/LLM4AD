from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT, write_manifest

DEFAULT_DATASET_ID = "bo_acquisition_v1"
DATA_DIR = Path(__file__).resolve().parent / "data"


class Branin:
    """Branin function scaled to [0, 1]^2. Global minimum ~= 0.397887."""

    f_opt: float = 0.397887

    def __call__(self, x: np.ndarray) -> float:
        x1 = -5.0 + 15.0 * x[0]
        x2 = 15.0 * x[1]
        b = 5.1 / (4.0 * np.pi ** 2)
        c = 5.0 / np.pi
        r, s, t = 6.0, 10.0, 1.0 / (8.0 * np.pi)
        return float((x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s)


class Hartmann3:
    """Hartmann-3 function on [0, 1]^3. Global minimum ~= -3.86278."""

    f_opt: float = -3.86278

    _A = np.array([
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
    ])
    _P = 1e-4 * np.array([
        [3689.0, 1170.0, 2673.0],
        [4699.0, 4387.0, 7470.0],
        [1091.0, 8732.0, 5547.0],
        [381.0, 5743.0, 8828.0],
    ])
    _alpha = np.array([1.0, 1.2, 3.0, 3.2])

    def __call__(self, x: np.ndarray) -> float:
        return float(-np.sum(self._alpha * np.exp(
            -np.sum(self._A * (x - self._P) ** 2, axis=1)
        )))


class Hartmann6:
    """Hartmann-6 function on [0, 1]^6. Global minimum ~= -3.32237."""

    f_opt: float = -3.32237

    _A = np.array([
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
    ])
    _P = 1e-4 * np.array([
        [1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
        [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
        [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
        [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0],
    ])
    _alpha = np.array([1.0, 1.2, 3.0, 3.2])

    def __call__(self, x: np.ndarray) -> float:
        return float(-np.sum(self._alpha * np.exp(
            -np.sum(self._A * (x - self._P) ** 2, axis=1)
        )))


INSTANCE_REGISTRY = {
    "Branin": {"class": Branin, "n_var": 2, "f_opt": Branin.f_opt},
    "Hartmann3": {"class": Hartmann3, "n_var": 3, "f_opt": Hartmann3.f_opt},
    "Hartmann6": {"class": Hartmann6, "n_var": 6, "f_opt": Hartmann6.f_opt},
}

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "instances": ["Branin", "Hartmann3"],
        "n_instances": 2,
    },
    "test_full": {
        "role": "test",
        "instances": ["Branin", "Hartmann3", "Hartmann6"],
        "n_instances": 3,
    },
    "test_ood_hartmann6": {
        "role": "test",
        "instances": ["Hartmann6"],
        "n_instances": 1,
    },
}


def write_default_dataset() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_id": DEFAULT_DATASET_ID,
        "task": "bo_acquisition",
        "version": 1,
        "description": "Fixed analytic BO acquisition benchmark instances from the EoH BO acquisition example.",
        "generator": "llm4ad.task.optimization.main.bo_acquisition.dataset.write_default_dataset",
        "source": "reference_code/EoH/examples/bo_acquisition",
        "splits": DEFAULT_SPLIT_SPECS,
    }
    return write_manifest(DATA_DIR, manifest)


def load_manifest() -> dict[str, Any]:
    path = DATA_DIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"BO acquisition manifest not found: {path}. "
            "Run `uv run python -m llm4ad.task.optimization.main.bo_acquisition.generate_dataset`."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_split_instances(split: str = DEFAULT_SPLIT):
    manifest = load_manifest()
    splits = manifest.get("splits", {})
    if split not in splits:
        available = ", ".join(sorted(splits))
        raise ValueError(f"Unknown BO acquisition split `{split}`. Available splits: {available}")

    split_info = splits[split]
    instances = []
    for name in split_info["instances"]:
        spec = INSTANCE_REGISTRY[name]
        func = spec["class"]()
        instances.append({
            "name": name,
            "func": func,
            "n_var": int(spec["n_var"]),
            "f_opt": float(spec["f_opt"]),
        })

    metadata = {
        "dataset_id": manifest["dataset_id"],
        "task": manifest["task"],
        "split": split,
        **split_info,
    }
    return instances, metadata
