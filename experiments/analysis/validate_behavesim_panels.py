"""Compare corrected BehaveSim distance matrices across two probe panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from experiments.analysis.behavesim_profiler import TASKS, summarize_distance_matrix


def _load_profile_keys(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        profiles = json.load(handle)
    return [
        profile["candidate"]["key"]
        for profile in profiles
        if "distribution" in profile["roles"]
    ]


def compare_task(task_dir: Path) -> dict:
    panel_a = task_dir / "panel_a"
    panel_b = task_dir / "panel_b"
    keys_a = _load_profile_keys(panel_a / "profiles.json")
    keys_b = _load_profile_keys(panel_b / "profiles.json")
    if keys_a != keys_b:
        raise ValueError(f"Panel candidate mappings differ under {task_dir}")
    matrix_a = np.load(panel_a / "distance_matrix.npy")
    matrix_b = np.load(panel_b / "distance_matrix.npy")
    upper = np.triu_indices(len(keys_a), k=1)
    correlation = spearmanr(matrix_a[upper], matrix_b[upper]).statistic
    metrics_a = summarize_distance_matrix(matrix_a)
    metrics_b = summarize_distance_matrix(matrix_b)
    return {
        "n": len(keys_a),
        "pairwise_spearman": float(correlation),
        "mean_pairwise_a": metrics_a["mean_pairwise_distance"],
        "mean_pairwise_b": metrics_b["mean_pairwise_distance"],
        "median_nearest_a": metrics_a["median_nearest_neighbor_distance"],
        "median_nearest_b": metrics_b["median_nearest_neighbor_distance"],
        "cluster_curve_auc_a": metrics_a["cluster_curve_auc"],
        "cluster_curve_auc_b": metrics_b["cluster_curve_auc"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("experiments/_logs/behavesim_v3/validation24"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = {task: compare_task(args.root / task) for task in TASKS}
    output = args.output or args.root / "panel_stability.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
