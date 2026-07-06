from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT, file_sha256

DEFAULT_DATASET_ID = "portfolio_construct_v1"
DATA_DIR = Path(__file__).resolve().parent / "data"

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "filename": "train.npz",
        "n_instances": 5,
        "n_assets": 20,
        "n_select": 5,
        "n_periods": 252,
        "seed": 2024,
        "note": "Original EoH search evaluator: five synthetic one-factor return instances.",
    },
    "test_full": {
        "role": "test",
        "filename": "test_full.npz",
        "n_instances": 16,
        "n_assets": 20,
        "n_select": 5,
        "n_periods": 252,
        "seed": 2024,
        "note": "Post-hoc evaluator from the EoH example: sixteen synthetic one-factor return instances.",
    },
}


def _generate_returns(n_instances: int, n_assets: int, n_periods: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    instances = []
    for _ in range(n_instances):
        market = rng.normal(0.0003, 0.012, n_periods)
        betas = rng.uniform(0.4, 1.6, n_assets)
        mu_idio = rng.uniform(-0.0002, 0.0008, n_assets)
        sigma_idio = rng.uniform(0.005, 0.025, n_assets)

        idiosyncratic = np.array([
            rng.normal(mu_idio[i], sigma_idio[i], n_periods)
            for i in range(n_assets)
        ])
        returns = betas[:, np.newaxis] * market[np.newaxis, :] + idiosyncratic
        instances.append(returns)
    return np.asarray(instances, dtype=float)


def write_default_dataset() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    splits: dict[str, Any] = {}
    for split, spec in DEFAULT_SPLIT_SPECS.items():
        filename = spec["filename"]
        returns = _generate_returns(
            n_instances=int(spec["n_instances"]),
            n_assets=int(spec["n_assets"]),
            n_periods=int(spec["n_periods"]),
            seed=int(spec["seed"]),
        )
        path = DATA_DIR / filename
        np.savez_compressed(path, returns=returns)

        splits[split] = {
            **spec,
            "format": "npz",
            "returns_shape": list(returns.shape),
            "sha256": file_sha256(path),
        }

    manifest = {
        "dataset_id": DEFAULT_DATASET_ID,
        "task": "portfolio_construct",
        "version": 1,
        "description": (
            "Fixed EoH greedy portfolio-construction benchmark. The searched function "
            "scores candidate assets for equal-weighted portfolio inclusion."
        ),
        "generator": "llm4ad.task.optimization.portfolio_construct.dataset.write_default_dataset",
        "paper": "papers/EoH",
        "source": "reference_code/EoH/examples/portfolio_construct",
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
            f"Portfolio construct manifest not found: {path}. "
            "Run `uv run python -m llm4ad.task.optimization.portfolio_construct.generate_dataset`."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_split_instances(split: str = DEFAULT_SPLIT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_manifest()
    splits = manifest.get("splits", {})
    if split not in splits:
        available = ", ".join(sorted(splits))
        raise ValueError(f"Unknown portfolio construct split `{split}`. Available splits: {available}")

    split_info = splits[split]
    path = DATA_DIR / split_info["filename"]
    if not path.exists():
        raise FileNotFoundError(f"Portfolio construct dataset split not found: {path}")
    if file_sha256(path) != split_info["sha256"]:
        raise ValueError(f"Portfolio construct dataset checksum mismatch: {path}")

    with np.load(path) as data:
        returns = np.asarray(data["returns"], dtype=float)

    instances = [
        {
            "instance_id": idx,
            "asset_returns": returns[idx],
            "n_assets": int(split_info["n_assets"]),
            "n_select": int(split_info["n_select"]),
            "n_periods": int(split_info["n_periods"]),
        }
        for idx in range(returns.shape[0])
    ]
    metadata = {
        "dataset_id": manifest["dataset_id"],
        "task": manifest["task"],
        "split": split,
        "path": str(path),
        **split_info,
    }
    return instances, metadata
