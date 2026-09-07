"""Calibrate BehaveSim probes and test its population-level geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr

from experiments.traceaad_refine_e1 import profile_core as core
from experiments.traceaad_refine_e1.prepare import dump

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.json"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def path_from_config(value: str) -> Path:
    return ROOT / value


def code_key(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def _valid_legacy_profile(path: Path) -> bool:
    if not path.exists():
        return False
    profile = json.loads(path.read_text())
    return all(profile["panels"][panel]["ok"] for panel in ("A", "B"))


def _evenly_spaced(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count >= len(rows):
        return rows
    indices = np.rint(np.linspace(0, len(rows) - 1, count)).astype(int)
    return [rows[index] for index in dict.fromkeys(indices.tolist())]


def select_nodes(
    nodes: Iterable[dict[str, Any]], task: str, sample_size: int, legacy: Path
) -> list[dict[str, Any]]:
    eligible = []
    for node in nodes:
        key = code_key(node["code"])
        profile_path = legacy / task / f"{key}.json"
        if np.isfinite(node["fitness"]) and _valid_legacy_profile(profile_path):
            eligible.append({**node, "key": key})
    if len(eligible) < sample_size:
        raise RuntimeError(f"{task}: only {len(eligible)} eligible nodes for n={sample_size}")

    by_fitness = sorted(eligible, key=lambda row: (row["fitness"], row["evaluation_id"]))
    strata = [list(group) for group in np.array_split(np.asarray(by_fitness, dtype=object), 4)]
    quotas = [sample_size // 4] * 4
    for index in range(sample_size % 4):
        quotas[index] += 1
    selected = []
    for stratum, quota in zip(strata, quotas):
        ordered = sorted(stratum, key=lambda row: row["evaluation_id"])
        selected.extend(_evenly_spaced(ordered, quota))
    if len(selected) != sample_size:
        raise AssertionError(f"selection returned {len(selected)} instead of {sample_size}")
    return sorted(selected, key=lambda row: row["evaluation_id"])


def prepare() -> dict[str, Any]:
    config = load_config()
    out = path_from_config(config["output"])
    existing = out / "selection.json"
    if existing.exists():
        return json.loads(existing.read_text())
    source = path_from_config(config["source_snapshot"])
    legacy = path_from_config(config["legacy_profiles"])
    manifest = json.loads((source / "snapshot.json").read_text())
    rows = []
    counts = {}
    for run in manifest["runs"]:
        state = json.loads(
            (source / "snapshot" / run["run_name"] / "tree_state.json").read_text()
        )
        size = config["sample_size_per_run"][run["task"]]
        selected = select_nodes(state["nodes"], run["task"], size, legacy)
        counts[run["run_name"]] = len(selected)
        for node in selected:
            rows.append(
                {
                    "task": run["task"],
                    "repeat": run["repeat"],
                    "run_name": run["run_name"],
                    "node_id": node["id"],
                    "evaluation_id": node["evaluation_id"],
                    "fitness": node["fitness"],
                    "key": node["key"],
                    "code": node["code"],
                }
            )
    result = {
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "runs": counts,
        "rows": rows,
    }
    dump(existing, result)
    dump(out / "config.json", config)
    return result


def _candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {"id": row["node_id"], "key": row["key"], "code": row["code"]}


def _profile_worker(job: tuple[str, dict[str, Any], str, dict[str, Any]]) -> dict[str, Any]:
    task, row, protocol, config = job
    candidate = _candidate(row)
    result = {"hash": row["key"], "protocol": protocol, "panels": {}}
    for panel in ("A", "B"):
        kwargs: dict[str, Any] = {}
        timeout = core.DEFAULT_TIMEOUT_SECONDS[task]
        if protocol == "obp_matched":
            kwargs["obp_scale"] = core.CALIBRATED_OBP_SCALE
            timeout = 120.0
        elif protocol == "aco_multiseed":
            if config["aco"]["seed_offsets"] != list(
                core.CALIBRATED_ACO_SEED_OFFSETS
            ):
                raise ValueError("config and calibrated ACO streams differ")
            kwargs["aco_seed_offsets"] = config["aco"]["seed_offsets"]
            timeout = 900.0
        else:
            raise ValueError(protocol)
        core._init_worker(
            task,
            panel,
            core.DEFAULT_TRAJECTORY_POINTS[task],
            timeout,
            **kwargs,
        )
        result["panels"][panel] = core._profile_candidate(candidate)
        result["panels"][panel]["probe_metadata"] = core._GLOBAL_DATA["probe_metadata"]
    return result


def profile(workers: int) -> dict[str, int]:
    config = load_config()
    out = path_from_config(config["output"])
    selection = prepare()
    jobs = {}
    for row in selection["rows"]:
        if row["task"] == "online_bin_packing":
            protocol = "obp_matched"
        elif row["task"] in {"op_aco", "cvrp_aco"}:
            protocol = "aco_multiseed"
        else:
            continue
        destination = out / "profiles" / protocol / row["task"] / f'{row["key"]}.json'
        if not destination.exists():
            jobs[(protocol, row["task"], row["key"])] = (
                row["task"], row, protocol, config
            )
    started = time.time()
    completed = 0
    failures = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_profile_worker, job): key for key, job in jobs.items()}
        for future in as_completed(futures):
            protocol, task, key = futures[future]
            result = future.result()
            destination = out / "profiles" / protocol / task / f"{key}.json"
            dump(destination, result)
            completed += 1
            failures += int(not all(result["panels"][p]["ok"] for p in ("A", "B")))
            if completed == 1 or completed % 10 == 0:
                print(
                    f"profiles {completed}/{len(jobs)} failures={failures} "
                    f"seconds={time.time() - started:.1f}",
                    flush=True,
                )
    artifacts = list((out / "profiles").glob("*/*/*.json"))
    artifact_failures = 0
    for path in artifacts:
        value = json.loads(path.read_text())
        artifact_failures += int(
            not all(value["panels"][panel]["ok"] for panel in ("A", "B"))
        )
    summary = {
        "scheduled_this_invocation": len(jobs),
        "completed_this_invocation": completed,
        "failures_this_invocation": failures,
        "artifacts_available": len(artifacts),
        "artifact_failures": artifact_failures,
    }
    dump(out / "profile_summary.json", summary)
    return summary


def _load_bundle(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not all(value["panels"][panel]["ok"] for panel in ("A", "B")):
        raise RuntimeError(f"invalid profile: {path}")
    return value


def _subset(profile: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    return {"trajectories": [profile["trajectories"][index] for index in indices]}


def _matrix(bundles: list[dict[str, Any]], task: str, selector=None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    panels = {}
    for panel in ("A", "B"):
        profiles = [bundle["panels"][panel] for bundle in bundles]
        if selector is not None:
            profiles = [_subset(value, selector(len(value["trajectories"]))) for value in profiles]
        panels[panel] = core.compute_distance_matrix(
            profiles, prefix_mode=task in core.PREFIX_TASKS
        )
    return (panels["A"] + panels["B"]) / 2, panels


def _upper(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(len(matrix), 1)]


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return None
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


def knn_overlap(left: np.ndarray, right: np.ndarray, k: int) -> float:
    penalty = np.eye(len(left)) * 1e9
    left_nn = np.argsort(left + penalty, axis=1)[:, :k]
    right_nn = np.argsort(right + penalty, axis=1)[:, :k]
    return float(
        np.mean([len(set(a.tolist()) & set(b.tolist())) / k for a, b in zip(left_nn, right_nn)])
    )


def compare_matrices(left: np.ndarray, right: np.ndarray, k: int) -> dict[str, Any]:
    return {
        "distance_spearman": safe_spearman(_upper(left), _upper(right)),
        "knn_overlap": knn_overlap(left, right, k),
    }


def landscape(matrix: np.ndarray, fitness: np.ndarray) -> dict[str, Any]:
    n = len(matrix)
    k = max(2, math.ceil(0.05 * (n - 1)))
    scale = float(np.std(fitness))
    normalized = (fitness - np.mean(fitness)) / scale if scale > 0 else np.zeros(n)
    gap_matrix = np.abs(normalized[:, None] - normalized[None, :])
    upper = np.triu_indices(n, 1)
    distances = matrix[upper]
    gaps = gap_matrix[upper]
    order = np.argsort(distances, kind="stable")
    bands = np.array_split(order, 3)
    penalty = np.eye(n) * 1e9
    nearest = np.argsort(matrix + penalty, axis=1)[:, :k]
    neighbor_gaps = [gap_matrix[i, j] for i in range(n) for j in nearest[i]]
    random_gaps = gap_matrix[~np.eye(n, dtype=bool)]
    tertiles = [float(np.mean(gaps[band])) for band in bands]
    return {
        "n": n,
        "pairs": len(distances),
        "k": k,
        "distance_fitness_gap_spearman": safe_spearman(distances, gaps),
        "knn_mean_fitness_gap": float(np.mean(neighbor_gaps)),
        "random_pair_mean_fitness_gap": float(np.mean(random_gaps)),
        "random_minus_knn_gap": float(np.mean(random_gaps) - np.mean(neighbor_gaps)),
        "tertile_mean_fitness_gap": {
            "near": tertiles[0],
            "middle": tertiles[1],
            "far": tertiles[2],
        },
        "far_minus_near_gap": tertiles[2] - tertiles[0],
    }


def analyze() -> dict[str, Any]:
    config = load_config()
    out = path_from_config(config["output"])
    legacy = path_from_config(config["legacy_profiles"])
    selection = prepare()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in selection["rows"]:
        grouped.setdefault(row["run_name"], []).append(row)

    per_run = []
    matrices_dir = out / "matrices"
    for run_name, rows in grouped.items():
        task = rows[0]["task"]
        fitness = np.asarray([row["fitness"] for row in rows], dtype=float)
        legacy_bundles = [
            _load_bundle(legacy / task / f'{row["key"]}.json') for row in rows
        ]
        legacy_matrix, legacy_panels = _matrix(legacy_bundles, task)
        k = max(2, math.ceil(0.05 * (len(rows) - 1)))
        calibration: dict[str, Any] = {
            "legacy_panel": compare_matrices(
                legacy_panels["A"], legacy_panels["B"], k
            )
        }
        selected_matrix = legacy_matrix
        selected_protocol = "legacy_two_panel"

        if task == "online_bin_packing":
            matched = [
                _load_bundle(
                    out / "profiles" / "obp_matched" / task / f'{row["key"]}.json'
                )
                for row in rows
            ]
            matched_matrix, matched_panels = _matrix(matched, task)
            calibration["compact_vs_matched"] = compare_matrices(
                legacy_matrix, matched_matrix, k
            )
            calibration["matched_panel"] = compare_matrices(
                matched_panels["A"], matched_panels["B"], k
            )
            selected_matrix = matched_matrix
            selected_protocol = "obp_matched_pending_global_decision"

        elif task in {"op_aco", "cvrp_aco"}:
            multi = [
                _load_bundle(
                    out / "profiles" / "aco_multiseed" / task / f'{row["key"]}.json'
                )
                for row in rows
            ]
            multi_matrix, multi_panels = _matrix(multi, task)
            half_one, _ = _matrix(
                multi, task, lambda count: [i for i in range(count) if i % 4 in (0, 1)]
            )
            half_two, _ = _matrix(
                multi, task, lambda count: [i for i in range(count) if i % 4 in (2, 3)]
            )
            calibration["single_vs_multiseed"] = compare_matrices(
                legacy_matrix, multi_matrix, k
            )
            calibration["multiseed_panel"] = compare_matrices(
                multi_panels["A"], multi_panels["B"], k
            )
            calibration["disjoint_seed_halves"] = compare_matrices(
                half_one, half_two, k
            )
            selected_matrix = multi_matrix
            selected_protocol = "aco_four_stream"

        folder = matrices_dir / run_name
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "selected.npy", selected_matrix)
        dump(folder / "ids.json", [row["node_id"] for row in rows])
        per_run.append(
            {
                "run_name": run_name,
                "task": task,
                "repeat": rows[0]["repeat"],
                "selected_protocol": selected_protocol,
                "calibration": calibration,
                "landscape": landscape(selected_matrix, fitness),
            }
        )

    obp_rows = [row for row in per_run if row["task"] == "online_bin_packing"]
    obp_gate = config["obp"]["retain_compact_if_every_run"]
    retain_compact = all(
        row["calibration"]["compact_vs_matched"]["distance_spearman"]
        >= obp_gate["distance_matrix_spearman_at_least"]
        and row["calibration"]["compact_vs_matched"]["knn_overlap"]
        >= obp_gate["knn_overlap_at_least"]
        for row in obp_rows
    )
    if retain_compact:
        for row in per_run:
            if row["task"] == "online_bin_packing":
                run_rows = grouped[row["run_name"]]
                bundles = [
                    _load_bundle(legacy / row["task"] / f'{item["key"]}.json')
                    for item in run_rows
                ]
                chosen, _ = _matrix(bundles, row["task"])
                row["selected_protocol"] = "obp_compact_256"
                row["landscape"] = landscape(
                    chosen, np.asarray([item["fitness"] for item in run_rows])
                )
                np.save(matrices_dir / row["run_name"] / "selected.npy", chosen)
    else:
        for row in per_run:
            if row["task"] == "online_bin_packing":
                row["selected_protocol"] = "obp_matched_1000_5000"

    aco_gate = config["aco"]["stable_if_every_run"]
    aco_stability = {}
    for task in ("op_aco", "cvrp_aco"):
        task_rows = [row for row in per_run if row["task"] == task]
        aco_stability[task] = all(
            row["calibration"]["multiseed_panel"]["distance_spearman"]
            >= aco_gate["panel_spearman_at_least"]
            and row["calibration"]["disjoint_seed_halves"]["distance_spearman"]
            >= aco_gate["disjoint_seed_half_spearman_at_least"]
            for row in task_rows
        )

    population_support = {}
    for task in core.TASKS:
        task_rows = [row for row in per_run if row["task"] == task]
        population_support[task] = all(
            row["landscape"]["distance_fitness_gap_spearman"] is not None
            and row["landscape"]["distance_fitness_gap_spearman"] > 0
            and row["landscape"]["random_minus_knn_gap"] > 0
            and row["landscape"]["far_minus_near_gap"] > 0
            for row in task_rows
        )

    result = {
        "obp_retain_compact": retain_compact,
        "aco_stable": aco_stability,
        "population_claim_supported": population_support,
        "runs": per_run,
    }
    dump(out / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "profile", "analyze"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.stage == "prepare":
        result = prepare()
        print(json.dumps({"runs": len(result["runs"]), "rows": len(result["rows"])}))
    elif args.stage == "profile":
        print(json.dumps(profile(args.workers)))
    else:
        print(json.dumps(analyze(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
