"""Validate calibrated BehaveSim geometry on time-representative archive samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from experiments.behavesim_population_calibration.run import landscape
from experiments.traceaad_refine_e1 import profile_core as core
from experiments.traceaad_refine_e1.prepare import dump

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("config.json")


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def configured_path(config: dict[str, Any], key: str) -> Path:
    return ROOT / config[key]


def code_key(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def uniform_time_sample(nodes: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ordered = sorted(nodes, key=lambda node: node["evaluation_id"])
    if count >= len(ordered):
        return ordered
    indices = np.rint(np.linspace(0, len(ordered) - 1, count)).astype(int).tolist()
    if len(set(indices)) != count:
        raise AssertionError("time sampling produced duplicate nodes")
    return [ordered[index] for index in indices]


def prepare() -> dict[str, Any]:
    config = load_config()
    out = configured_path(config, "output")
    destination = out / "selection.json"
    if destination.exists():
        return json.loads(destination.read_text())
    source = configured_path(config, "source_snapshot")
    manifest = json.loads((source / "snapshot.json").read_text())
    rows = []
    coverage = []
    for run in manifest["runs"]:
        state_path = source / "snapshot" / run["run_name"] / "tree_state.json"
        nodes = json.loads(state_path.read_text())["nodes"]
        selected = uniform_time_sample(
            nodes, config["sample_size_per_run"][run["task"]]
        )
        coverage.append(
            {
                "run_name": run["run_name"],
                "task": run["task"],
                "archive_nodes": len(nodes),
                "selected_nodes": len(selected),
                "first_evaluation_id": selected[0]["evaluation_id"],
                "last_evaluation_id": selected[-1]["evaluation_id"],
            }
        )
        for node in selected:
            if not np.isfinite(node["fitness"]):
                raise AssertionError("archive contains a non-finite valid-node fitness")
            rows.append(
                {
                    "task": run["task"],
                    "repeat": run["repeat"],
                    "run_name": run["run_name"],
                    "node_id": node["id"],
                    "evaluation_id": node["evaluation_id"],
                    "fitness": node["fitness"],
                    "key": code_key(node["code"]),
                    "code": node["code"],
                }
            )
    result = {
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "coverage": coverage,
        "rows": rows,
    }
    dump(destination, result)
    dump(out / "config.json", config)
    return result


def reusable_profile(config: dict[str, Any], task: str, key: str) -> Path | None:
    if task in {"tsp_construct", "vrptw_construct"}:
        candidate = configured_path(config, "legacy_profiles") / task / f"{key}.json"
    elif task == "online_bin_packing":
        candidate = (
            configured_path(config, "calibration_output")
            / "profiles"
            / "obp_matched"
            / task
            / f"{key}.json"
        )
    else:
        candidate = (
            configured_path(config, "calibration_output")
            / "profiles"
            / "aco_multiseed"
            / task
            / f"{key}.json"
        )
    if not candidate.exists():
        return None
    bundle = json.loads(candidate.read_text())
    if all(bundle["panels"][panel]["ok"] for panel in ("A", "B")):
        return candidate
    return None


def _profile_worker(job: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    task, row = job
    result = {"hash": row["key"], "panels": {}}
    for panel in ("A", "B"):
        kwargs: dict[str, Any] = {}
        timeout = core.DEFAULT_TIMEOUT_SECONDS[task]
        if task == "online_bin_packing":
            kwargs["obp_scale"] = core.CALIBRATED_OBP_SCALE
            timeout = 120.0
        elif task in {"op_aco", "cvrp_aco"}:
            kwargs["aco_seed_offsets"] = core.CALIBRATED_ACO_SEED_OFFSETS
            timeout = 900.0
        core._init_worker(
            task,
            panel,
            core.DEFAULT_TRAJECTORY_POINTS[task],
            timeout,
            **kwargs,
        )
        candidate = {"id": row["node_id"], "key": row["key"], "code": row["code"]}
        result["panels"][panel] = core._profile_candidate(candidate)
        result["panels"][panel]["probe_metadata"] = core._GLOBAL_DATA["probe_metadata"]
    return result


def profile(workers: int) -> dict[str, int]:
    config = load_config()
    out = configured_path(config, "output")
    selection = prepare()
    jobs = {}
    reused = set()
    for row in selection["rows"]:
        identity = (row["task"], row["key"])
        if reusable_profile(config, *identity) is not None:
            reused.add(identity)
            continue
        destination = out / "profiles" / row["task"] / f'{row["key"]}.json'
        if not destination.exists():
            jobs[identity] = (row["task"], row)

    started = time.time()
    completed = 0
    failures = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_profile_worker, job): key for key, job in jobs.items()}
        for future in as_completed(futures):
            task, key = futures[future]
            result = future.result()
            dump(out / "profiles" / task / f"{key}.json", result)
            completed += 1
            failures += int(
                not all(result["panels"][panel]["ok"] for panel in ("A", "B"))
            )
            if completed == 1 or completed % 10 == 0:
                print(
                    f"profiles {completed}/{len(jobs)} failures={failures} "
                    f"seconds={time.time() - started:.1f}",
                    flush=True,
                )
    summary = {
        "unique_selected": len(
            {(row["task"], row["key"]) for row in selection["rows"]}
        ),
        "reused": len(reused),
        "scheduled": len(jobs),
        "completed": completed,
        "failures": failures,
    }
    dump(out / "profile_summary.json", summary)
    return summary


def bundle_path(config: dict[str, Any], task: str, key: str) -> Path:
    local = configured_path(config, "output") / "profiles" / task / f"{key}.json"
    if local.exists():
        return local
    reused = reusable_profile(config, task, key)
    if reused is None:
        raise FileNotFoundError(f"missing profile for {task}/{key}")
    return reused


def distance_matrix(bundles: list[dict[str, Any]], task: str) -> np.ndarray:
    matrices = []
    for panel in ("A", "B"):
        profiles = []
        for bundle in bundles:
            profile = bundle["panels"][panel]
            if not profile["ok"]:
                raise RuntimeError("selected archive node has an invalid profile")
            profiles.append(profile)
        matrices.append(
            core.compute_distance_matrix(
                profiles, prefix_mode=task in core.PREFIX_TASKS
            )
        )
    return (matrices[0] + matrices[1]) / 2


def analyze() -> dict[str, Any]:
    config = load_config()
    out = configured_path(config, "output")
    selection = prepare()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in selection["rows"]:
        grouped.setdefault(row["run_name"], []).append(row)
    runs = []
    for run_name, rows in grouped.items():
        task = rows[0]["task"]
        valid_rows = []
        bundles = []
        profile_failures = []
        for row in rows:
            bundle = json.loads(bundle_path(config, task, row["key"]).read_text())
            failed_panels = [
                panel for panel in ("A", "B") if not bundle["panels"][panel]["ok"]
            ]
            if failed_panels:
                profile_failures.append(
                    {
                        "node_id": row["node_id"],
                        "evaluation_id": row["evaluation_id"],
                        "panels": failed_panels,
                        "errors": {
                            panel: bundle["panels"][panel]["error"]
                            for panel in failed_panels
                        },
                    }
                )
                continue
            valid_rows.append(row)
            bundles.append(bundle)
        matrix = distance_matrix(bundles, task)
        folder = out / "matrices" / run_name
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "behavior.npy", matrix)
        dump(folder / "ids.json", [row["node_id"] for row in valid_rows])
        runs.append(
            {
                "run_name": run_name,
                "task": task,
                "repeat": rows[0]["repeat"],
                "selected": len(rows),
                "profiled": len(valid_rows),
                "profile_failures": profile_failures,
                "landscape": landscape(
                    matrix,
                    np.asarray([row["fitness"] for row in valid_rows], dtype=float),
                ),
            }
        )
        print(f"matrix {len(runs)}/{len(grouped)} {task} rep{rows[0]['repeat']}", flush=True)

    support = {}
    for task in core.TASKS:
        task_runs = [row for row in runs if row["task"] == task]
        support[task] = all(
            row["landscape"]["distance_fitness_gap_spearman"] is not None
            and row["landscape"]["distance_fitness_gap_spearman"] > 0
            and row["landscape"]["random_minus_knn_gap"] > 0
            and row["landscape"]["far_minus_near_gap"] > 0
            for row in task_runs
        )
    primary_supported = all(support[task] for task in config["primary_claim_tasks"])
    result = {
        "archive_geometry_primary_supported": primary_supported,
        "task_support": support,
        "sample_is_fitness_stratified": False,
        "sample_requires_prior_profile": False,
        "runs": runs,
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
        print(json.dumps({"runs": len(result["coverage"]), "rows": len(result["rows"])}))
    elif args.stage == "profile":
        print(json.dumps(profile(args.workers)))
    else:
        print(json.dumps(analyze(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
