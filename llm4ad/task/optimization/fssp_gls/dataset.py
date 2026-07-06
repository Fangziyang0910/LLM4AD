from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import (
    DEFAULT_SPLIT,
    load_pickle_split,
    task_data_dir,
    write_pickle_splits,
)

DEFAULT_DATASET_ID = "fssp_gls_v1"
DATA_DIR = task_data_dir(__file__)

TAILLARD_FILES = [
    "t_j20_m5.txt",
    "t_j20_m10.txt",
    "t_j20_m20.txt",
    "t_j50_m5.txt",
    "t_j50_m10.txt",
    "t_j50_m20.txt",
    "t_j100_m5.txt",
    "t_j100_m10.txt",
    "t_j100_m20.txt",
    "t_j200_m10.txt",
    "t_j200_m20.txt",
]

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "source": "eoh_training_data",
        "n_instances": 3,
        "filename": "train.pkl.gz",
    },
    "train_full": {
        "role": "train",
        "source": "eoh_training_data",
        "n_instances": 64,
        "filename": "train_full.pkl.gz",
    },
    **{
        f"test_taillard_{name[3:-4]}": {
            "role": "test",
            "source": "taillard",
            "source_file": name,
            "n_instances": 10,
            "filename": f"test_taillard_{name[3:-4]}.pkl.gz",
        }
        for name in TAILLARD_FILES
    },
}


def default_source_dir() -> Path:
    return (
        Path(__file__).resolve().parents[5]
        / "reference_code"
        / "EoH"
        / "examples"
        / "fssp_gls"
    )


def _training_instance(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        n_jobs, n_machines = (int(x) for x in handle.readline().split())
        processing_times = np.zeros((n_jobs, n_machines), dtype=int)
        for job_idx in range(n_jobs):
            parts = handle.readline().split()
            for machine_idx in range(n_machines):
                processing_times[job_idx, machine_idx] = int(float(parts[machine_idx * 2 + 1]))

    return {
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "processing_times": processing_times.tolist(),
        "source": path.name,
    }


def _taillard_instances(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    instances = []
    idx = 0
    while idx < len(lines):
        if "number of jobs" not in lines[idx]:
            idx += 1
            continue

        n_jobs, n_machines, seed, upper_bound, lower_bound = (
            int(x) for x in lines[idx + 1].split()[:5]
        )
        rows = []
        for machine_idx in range(n_machines):
            rows.append([int(v) for v in lines[idx + 3 + machine_idx].split()])
        processing_times = np.array(rows, dtype=int).T
        instances.append({
            "n_jobs": n_jobs,
            "n_machines": n_machines,
            "seed": seed,
            "upper_bound": upper_bound,
            "lower_bound": lower_bound,
            "processing_times": processing_times.tolist(),
            "source": path.name,
        })
        idx += 3 + n_machines

    return instances


def _generate_split_from_source(source_dir: Path):
    def generate_split(split: str, spec: dict[str, Any]):
        if spec["source"] == "eoh_training_data":
            data_dir = source_dir / "TrainingData"
            return [
                _training_instance(data_dir / f"{idx}.txt")
                for idx in range(1, int(spec["n_instances"]) + 1)
            ]

        if spec["source"] == "taillard":
            path = source_dir / "TestingData" / "Taillard" / spec["source_file"]
            instances = _taillard_instances(path)
            return instances[:int(spec["n_instances"])]

        raise ValueError(f"Unknown FSSP data source: {spec['source']}")

    return generate_split


def write_default_dataset(source_dir: str | Path | None = None) -> dict[str, Any]:
    source_path = Path(source_dir) if source_dir is not None else default_source_dir()
    if not source_path.exists():
        raise FileNotFoundError(
            f"FSSP source data not found: {source_path}. "
            "Set source_dir to the EoH examples/fssp_gls directory."
        )

    return write_pickle_splits(
        data_dir=DATA_DIR,
        dataset_id=DEFAULT_DATASET_ID,
        task="fssp_gls",
        version=1,
        description="Fixed FSSP guided local search splits from the EoH FSSP benchmark.",
        split_specs=DEFAULT_SPLIT_SPECS,
        generate_split=_generate_split_from_source(source_path),
        generator="llm4ad.task.optimization.fssp_gls.dataset.write_default_dataset",
    )


def load_manifest() -> dict[str, Any]:
    from llm4ad.task.optimization.dataset_io import load_manifest as _load_manifest

    return _load_manifest(DATA_DIR)


def load_split_instances(split: str = DEFAULT_SPLIT):
    return load_pickle_split(data_dir=DATA_DIR, split=split)
