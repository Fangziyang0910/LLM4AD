from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT, file_sha256

DEFAULT_DATASET_ID = "dpp_ga_v1"
DATA_DIR = Path(__file__).resolve().parent / "data"

DATA_FILES = [
    "DPP_data/01nF_decap.npy",
    "DPP_data/10x10_pkg_chip.npy",
    "DPP_data/freq_201.npy",
    "test_problems/test_100_keepout.npy",
    "test_problems/test_100_keepout_num.npy",
    "test_problems/test_100_probe.npy",
]

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "start": 0,
        "stop": 3,
        "n_instances": 3,
        "n_iter": 5,
    },
    "val": {
        "role": "validation",
        "start": 5,
        "stop": 10,
        "n_instances": 5,
        "n_iter": 10,
    },
    "test": {
        "role": "test",
        "start": -64,
        "stop": None,
        "n_instances": 64,
        "n_iter": 10,
    },
}


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _reference_dir() -> Path:
    return _workspace_root() / "reference_code" / "ReEvo" / "problems" / "dpp_ga"


def _copy_source_data(source_dir: Path) -> dict[str, dict[str, Any]]:
    file_info = {}
    for relative_name in DATA_FILES:
        src = source_dir / relative_name
        if not src.exists():
            raise FileNotFoundError(f"DPP-GA source data file not found: {src}")
        dst = DATA_DIR / relative_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        file_info[relative_name] = {
            "bytes": dst.stat().st_size,
            "sha256": file_sha256(dst),
        }
    return file_info


def write_default_dataset(source_dir: str | Path | None = None) -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    source_path = Path(source_dir) if source_dir is not None else _reference_dir()
    file_info = _copy_source_data(source_path)

    manifest = {
        "dataset_id": DEFAULT_DATASET_ID,
        "task": "dpp_ga",
        "version": 1,
        "description": (
            "Fixed ReEvo Decap Placement Problem data for evolving GA crossover "
            "operators on a 10x10 power distribution network."
        ),
        "generator": "llm4ad.task.optimization.dpp_ga.dataset.write_default_dataset",
        "paper": [
            "papers/ReEvo/sections/05_applications.tex",
            "papers/ReEvo/appendix/01_prompts.tex",
            "papers/ReEvo/appendix/02_experimental_setup.tex",
            "papers/ReEvo/appendix/03_benchmark_problems.tex",
        ],
        "source": "reference_code/ReEvo/problems/dpp_ga",
        "prompt_source": "reference_code/ReEvo/prompts/dpp_ga",
        "parameters": {
            "grid_shape": [10, 10],
            "model_number": 5,
            "freq_pts": 201,
            "n_decap": 20,
            "n_pop": 20,
            "elite_rate": 0.2,
        },
        "files": file_info,
        "splits": DEFAULT_SPLIT_SPECS,
    }
    with (DATA_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def load_manifest() -> dict[str, Any]:
    path = DATA_DIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"DPP-GA manifest not found: {path}. "
            "Run `uv run python -m llm4ad.task.optimization.dpp_ga.generate_dataset`."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _verify_files(manifest: dict[str, Any]) -> None:
    for relative_name, info in manifest["files"].items():
        path = DATA_DIR / relative_name
        if not path.exists():
            raise FileNotFoundError(f"DPP-GA data file not found: {path}")
        if path.stat().st_size != int(info["bytes"]):
            raise ValueError(f"DPP-GA data file size mismatch: {path}")
        if file_sha256(path) != info["sha256"]:
            raise ValueError(f"DPP-GA data file checksum mismatch: {path}")


def load_problem_arrays() -> dict[str, np.ndarray]:
    manifest = load_manifest()
    _verify_files(manifest)
    return {
        "decap": np.load(DATA_DIR / "DPP_data/01nF_decap.npy").reshape(-1),
        "raw_pdn": np.load(DATA_DIR / "DPP_data/10x10_pkg_chip.npy"),
        "freq": np.load(DATA_DIR / "DPP_data/freq_201.npy"),
        "keepout": np.load(DATA_DIR / "test_problems/test_100_keepout.npy"),
        "keepout_num": np.load(DATA_DIR / "test_problems/test_100_keepout_num.npy"),
        "probe": np.load(DATA_DIR / "test_problems/test_100_probe.npy"),
    }


def _load_context_arrays(manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    for relative_name in [
        "test_problems/test_100_keepout.npy",
        "test_problems/test_100_keepout_num.npy",
        "test_problems/test_100_probe.npy",
    ]:
        path = DATA_DIR / relative_name
        info = manifest["files"][relative_name]
        if not path.exists():
            raise FileNotFoundError(f"DPP-GA data file not found: {path}")
        if path.stat().st_size != int(info["bytes"]):
            raise ValueError(f"DPP-GA data file size mismatch: {path}")
        if file_sha256(path) != info["sha256"]:
            raise ValueError(f"DPP-GA data file checksum mismatch: {path}")

    return {
        "keepout": np.load(DATA_DIR / "test_problems/test_100_keepout.npy"),
        "keepout_num": np.load(DATA_DIR / "test_problems/test_100_keepout_num.npy"),
        "probe": np.load(DATA_DIR / "test_problems/test_100_probe.npy"),
    }


def load_split_instances(split: str = DEFAULT_SPLIT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_manifest()
    splits = manifest.get("splits", {})
    if split not in splits:
        available = ", ".join(sorted(splits))
        raise ValueError(f"Unknown DPP-GA split `{split}`. Available splits: {available}")

    split_info = splits[split]
    arrays = _load_context_arrays(manifest)
    start = split_info["start"]
    stop = split_info["stop"]
    instances = []
    for probe, keepout, keepout_num in zip(
            arrays["probe"][start:stop],
            arrays["keepout"][start:stop],
            arrays["keepout_num"][start:stop],
    ):
        instances.append({
            "probe": int(probe),
            "keepout": np.asarray(keepout, dtype=int),
            "keepout_num": int(keepout_num),
        })

    metadata = {
        "dataset_id": manifest["dataset_id"],
        "task": manifest["task"],
        "split": split,
        **split_info,
        "parameters": manifest["parameters"],
    }
    return instances, metadata
