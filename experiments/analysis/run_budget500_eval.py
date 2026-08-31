"""Evaluate held-out test sets for best-so-far at search budget 500.

Reads the same run directories as the main-table ``eval_best_*`` artifacts,
picks the best scored sample with ``sample_order <= 500``, and re-evaluates on
held-out splits via ``experiments/evaluate_best.py``.

Typical workflow (sync → remote CPU eval → fetch → summary)::

    uv run python experiments/analysis/run_budget500_eval.py sync
    uv run python experiments/analysis/run_budget500_eval.py run-remote --jobs 8
    uv run python experiments/analysis/run_budget500_eval.py fetch
    uv run python experiments/analysis/run_budget500_eval.py summarize

Remote host defaults to ``B3-server1`` with repo at ``~/code/LLM4AD/LLM4AD``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
REMOTE_HOST = "B3-server1"
REMOTE_REPO = Path("/home/fzy/code/LLM4AD/LLM4AD")
BUDGET = 500
OUTPUT_DIRNAME = "eval_best_budget500_20260830"

BASELINE_METHODS = frozenset({"eoh", "reevo", "mcts_ahd", "pathwise", "calm"})
TRACEAAD_LABELS = {
    "traceaad_v9_16": "V9.16",
    "traceaad_v9_17": "V9.17",
    "traceaad_v9_19": "V9.19",
    "traceaad_v9_20": "V9.20",
    "traceaad_v9_21": "V9.21",
}
DISPLAY = {
    "eoh": "EoH",
    "reevo": "ReEvo",
    "mcts_ahd": "MCTS-AHD",
    "pathwise": "PathWise",
    "calm": "CALM",
    **{k: f"TraceAAD {v}" for k, v in TRACEAAD_LABELS.items()},
}

EVAL_ARTIFACTS: list[tuple[str, Path]] = [
    # baselines (20260824 rerun, four tasks)
    ("tsp_construct", REPO / "experiments/其他实验/基线重跑-20260824/tsp_construct/eoh/eval_best_20260825_rerun/results.json"),
    ("tsp_construct", REPO / "experiments/其他实验/基线重跑-20260824/tsp_construct/reevo/eval_best_20260825_rerun/results.json"),
    ("tsp_construct", REPO / "experiments/其他实验/基线重跑-20260824/tsp_construct/mcts_ahd/eval_best_20260825_rerun/results.json"),
    ("tsp_construct", REPO / "experiments/其他实验/基线重跑-20260824/tsp_construct/pathwise/eval_best_20260825_rerun/results.json"),
    ("tsp_construct", REPO / "experiments/其他实验/基线重跑-20260824/tsp_construct/calm/eval_best_20260825_rerun/results.json"),
    ("cvrp_aco", REPO / "experiments/其他实验/基线重跑-20260824/cvrp_aco/eoh/eval_best_20260825_rerun/results.json"),
    ("cvrp_aco", REPO / "experiments/其他实验/基线重跑-20260824/cvrp_aco/reevo/eval_best_20260825_rerun/results.json"),
    ("cvrp_aco", REPO / "experiments/其他实验/基线重跑-20260824/cvrp_aco/mcts_ahd/eval_best_20260825_rerun/results.json"),
    ("cvrp_aco", REPO / "experiments/其他实验/基线重跑-20260824/cvrp_aco/pathwise/eval_best_20260825_rerun/results.json"),
    ("cvrp_aco", REPO / "experiments/其他实验/基线重跑-20260824/cvrp_aco/calm/eval_best_20260825_rerun/results.json"),
    ("op_aco", REPO / "experiments/其他实验/基线重跑-20260824/op_aco/eoh/eval_best_20260825_rerun/results.json"),
    ("op_aco", REPO / "experiments/其他实验/基线重跑-20260824/op_aco/reevo/eval_best_20260825_rerun/results.json"),
    ("op_aco", REPO / "experiments/其他实验/基线重跑-20260824/op_aco/mcts_ahd/eval_best_20260825_rerun/results.json"),
    ("op_aco", REPO / "experiments/其他实验/基线重跑-20260824/op_aco/pathwise/eval_best_20260825_rerun/results.json"),
    ("op_aco", REPO / "experiments/其他实验/基线重跑-20260824/op_aco/calm/eval_best_20260825_rerun/results.json"),
    ("online_bin_packing", REPO / "experiments/其他实验/基线重跑-20260824/online_bin_packing/eoh/eval_best_20260825_rerun/results.json"),
    ("online_bin_packing", REPO / "experiments/其他实验/基线重跑-20260824/online_bin_packing/reevo/eval_best_20260825_rerun/results.json"),
    ("online_bin_packing", REPO / "experiments/其他实验/基线重跑-20260824/online_bin_packing/mcts_ahd/eval_best_20260825_rerun/results.json"),
    ("online_bin_packing", REPO / "experiments/其他实验/基线重跑-20260824/online_bin_packing/pathwise/eval_best_20260825_rerun/results.json"),
    ("online_bin_packing", REPO / "experiments/其他实验/基线重跑-20260824/online_bin_packing/calm/eval_best_20260825_rerun/results.json"),
    # TraceAAD main-table versions
    ("tsp_construct", REPO / "experiments/tsp_construct/traceaad_v9_16/eval_best_20260823_v916_complete/results.json"),
    ("cvrp_aco", REPO / "experiments/cvrp_aco/traceaad_v9_16/eval_best_20260823_v916_complete/results.json"),
    ("op_aco", REPO / "experiments/op_aco/traceaad_v9_16/eval_best_20260823_v916_complete/results.json"),
    ("online_bin_packing", REPO / "experiments/online_bin_packing/traceaad_v9_16/eval_best_20260823_v916_complete/results.json"),
    ("tsp_construct", REPO / "experiments/tsp_construct/traceaad_v9_17/eval_best_20260824_v917_adaptive_complete/results.json"),
    ("cvrp_aco", REPO / "experiments/cvrp_aco/traceaad_v9_17/eval_best_20260824_v917_adaptive_complete/results.json"),
    ("op_aco", REPO / "experiments/op_aco/traceaad_v9_17/eval_best_20260824_v917_adaptive_complete/results.json"),
    ("online_bin_packing", REPO / "experiments/online_bin_packing/traceaad_v9_17/eval_best_20260824_v917_adaptive_complete/results.json"),
    ("tsp_construct", REPO / "experiments/tsp_construct/traceaad_v9_19/eval_best_v919_fixed_tsp_construct_incremental/results.json"),
    ("cvrp_aco", REPO / "experiments/cvrp_aco/traceaad_v9_19/eval_best_v919_fixed_cvrp_aco_incremental/results.json"),
    ("op_aco", REPO / "experiments/op_aco/traceaad_v9_19/eval_best_v919_fixed_op_aco_incremental/results.json"),
    ("online_bin_packing", REPO / "experiments/online_bin_packing/traceaad_v9_19/eval_best_v919_fixed_online_bin_packing_incremental/results.json"),
    ("tsp_construct", REPO / "experiments/tsp_construct/traceaad_v9_20/eval_best_v920_tsp_construct_incremental/results.json"),
    ("cvrp_aco", REPO / "experiments/cvrp_aco/traceaad_v9_20/eval_best_v920_cvrp_aco_incremental/results.json"),
    ("op_aco", REPO / "experiments/op_aco/traceaad_v9_20/eval_best_v920_op_aco_incremental/results.json"),
    ("online_bin_packing", REPO / "experiments/online_bin_packing/traceaad_v9_20/eval_best_v920_online_bin_packing_incremental/results.json"),
    # TraceAAD V9.21 (OBP full-budget batch; other tasks are ~674-eval snapshots)
    ("online_bin_packing", REPO / "experiments/online_bin_packing/traceaad_v9_21/eval_best_v921_online_bin_packing_incremental/results.json"),
    # VRPTW (original 20260822 baseline batch)
    ("vrptw_construct", REPO / "experiments/vrptw_construct/eoh/eval_best_20260824_vrptw_multiscale/results.json"),
    ("vrptw_construct", REPO / "experiments/vrptw_construct/reevo/eval_best_20260824_vrptw_multiscale/results.json"),
    ("vrptw_construct", REPO / "experiments/vrptw_construct/mcts_ahd/eval_best_20260824_vrptw_multiscale/results.json"),
    ("vrptw_construct", REPO / "experiments/vrptw_construct/pathwise/eval_best_20260824_vrptw_multiscale/results.json"),
    ("vrptw_construct", REPO / "experiments/vrptw_construct/calm/eval_best_20260824_vrptw_multiscale/results.json"),
    ("vrptw_construct", REPO / "experiments/vrptw_construct/traceaad_v9_16/eval_best_v916_vrptw_construct_complete/results.json"),
    ("vrptw_construct", REPO / "experiments/vrptw_construct/traceaad_v9_17/eval_best_20260824_v917_adaptive_complete/results.json"),
    ("vrptw_construct", REPO / "experiments/vrptw_construct/traceaad_v9_19/eval_best_v919_fixed_vrptw_construct_incremental/results.json"),
    ("vrptw_construct", REPO / "experiments/vrptw_construct/traceaad_v9_20/eval_best_v920_vrptw_construct_incremental/results.json"),
]

TASK_TIMEOUT = {
    "tsp_construct": 3000,
    "cvrp_aco": 120,
    "op_aco": 60,
    "online_bin_packing": 30,
    "vrptw_construct": 1000,
}


@dataclass(frozen=True, slots=True)
class EvalGroup:
    task: str
    method_key: str
    run_dirs: tuple[Path, ...]
    output_dir: Path
    source_eval: Path

    @property
    def label(self) -> str:
        return DISPLAY.get(self.method_key, self.method_key)


def _resolve_run_dir(run_dir_value: str, source_eval: Path) -> Path:
    direct = (REPO / run_dir_value).resolve()
    if direct.exists():
        return direct
    run_name = Path(run_dir_value).name
    method = Path(run_dir_value).parent.name
    task = Path(run_dir_value).parent.parent.name
    candidates = [
        REPO / "experiments" / "其他实验" / "基线重跑-20260824" / task / method / run_name,
        source_eval.parent.parent / run_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"cannot resolve run dir {run_dir_value!r}")


def _discover_run_dirs(method_dir: Path, pattern: str = "*rep*") -> tuple[Path, ...] | None:
    if not method_dir.is_dir():
        return None
    runs = sorted(
        path
        for path in method_dir.glob(pattern)
        if path.is_dir() and (path / "logs" / "run_summary.json").exists()
    )
    if len(runs) != 3:
        return None
    return tuple(run.resolve() for run in runs)


def load_groups() -> list[EvalGroup]:
    groups: list[EvalGroup] = []
    for task, source_path in EVAL_ARTIFACTS:
        method_dir = source_path.parent.parent
        if source_path.exists():
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            method_key = str(payload.get("method", method_dir.name))
            run_records = payload.get("run_records", [])
            if len(run_records) != 3:
                raise ValueError(
                    f"expected 3 runs in {source_path}, got {len(run_records)}"
                )
            run_dirs = tuple(
                _resolve_run_dir(str(row["run_dir"]), source_path) for row in run_records
            )
        else:
            method_key = method_dir.name
            run_dirs = _discover_run_dirs(method_dir)
            if run_dirs is None:
                raise FileNotFoundError(
                    f"missing eval artifact and cannot discover 3 runs: {source_path}"
                )
        output_dir = method_dir / OUTPUT_DIRNAME
        groups.append(
            EvalGroup(
                task=task,
                method_key=method_key,
                run_dirs=run_dirs,
                output_dir=output_dir,
                source_eval=source_path,
            )
        )
    return groups


def _ssh(command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", REMOTE_HOST, command],
        check=check,
        capture_output=True,
        text=True,
    )


def _rsync(paths: list[Path], *, dry_run: bool = False, from_remote: bool = False) -> None:
    for path in paths:
        args = ["rsync", "-az"]
        if dry_run:
            args.append("--dry-run")
        rel = path.relative_to(REPO)
        if from_remote:
            remote_src = f"{REMOTE_HOST}:{REMOTE_REPO / rel}/"
            local_dest = REPO / rel
            local_dest.mkdir(parents=True, exist_ok=True)
            args.extend([remote_src, f"{local_dest}/"])
        else:
            remote_dest = f"{REMOTE_HOST}:{REMOTE_REPO / rel.parent}/"
            args.extend([str(path), remote_dest])
        subprocess.run(args, check=True)


def _sync_roots(groups: list[EvalGroup]) -> list[Path]:
    roots: set[Path] = {
        REPO / "experiments/evaluate_best.py",
        REPO / "experiments/eval_artifacts.py",
        REPO / "llm4ad",
        REPO / "experiments/其他实验/基线重跑-20260824",
    }
    for group in groups:
        roots.add(group.run_dirs[0].parent)
    return sorted(roots)


def cmd_sync(groups: list[EvalGroup], *, dry_run: bool = False) -> None:
    roots = _sync_roots(groups)
    print(f"sync {len(roots)} roots to {REMOTE_HOST}", flush=True)
    _rsync(roots, dry_run=dry_run)


def _remote_eval_command(group: EvalGroup, workers: int) -> str:
    timeout = TASK_TIMEOUT[group.task]
    run_args = " ".join(f'"{REMOTE_REPO / run_dir.relative_to(REPO)}"' for run_dir in group.run_dirs)
    output = REMOTE_REPO / group.output_dir.relative_to(REPO)
    return " ".join(
        [
            f"cd {REMOTE_REPO}",
            "&&",
            f"{REMOTE_REPO / '.venv/bin/python'}",
            "experiments/evaluate_best.py",
            run_args,
            "--output-dir",
            f'"{output}"',
            "--max-sample-order",
            str(BUDGET),
            "--timeout",
            str(timeout),
            "--workers",
            str(workers),
        ]
    )


def _group_complete(group: EvalGroup) -> bool:
    path = group.output_dir / "results.json"
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    # evaluate_best only writes a top-level max_sample_order for OBP; the ACO
    # and TSP payloads carry it per run record instead.
    records = payload.get("run_records", [])
    orders = [payload.get("max_sample_order")] + [
        row.get("max_sample_order") for row in records
    ]
    if BUDGET not in orders:
        return False
    names = {row.get("run_name") for row in records}
    return all(run_dir.name in names for run_dir in group.run_dirs)


def cmd_run_remote(groups: list[EvalGroup], *, jobs: int, workers: int, dry_run: bool) -> None:
    pending = [group for group in groups if not _group_complete(group)]
    print(f"remote eval: {len(pending)} pending / {len(groups)} total", flush=True)
    if dry_run:
        for group in pending:
            print(_remote_eval_command(group, workers))
        return

    import concurrent.futures

    def _run(group: EvalGroup) -> tuple[str, int, str]:
        command = _remote_eval_command(group, workers)
        result = _ssh(command, check=False)
        tag = f"{group.task}/{group.method_key}"
        return tag, result.returncode, (result.stdout + result.stderr)[-2000:]

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_run, group) for group in pending]
        for future in concurrent.futures.as_completed(futures):
            tag, code, tail = future.result()
            status = "ok" if code == 0 else f"fail({code})"
            print(f"{status} {tag}", flush=True)
            if code != 0:
                print(tail, file=sys.stderr)


def cmd_fetch(groups: list[EvalGroup], *, dry_run: bool = False) -> None:
    targets: list[Path] = []
    for group in groups:
        remote = REMOTE_REPO / group.output_dir.relative_to(REPO) / "results.json"
        probe = _ssh(f'test -f "{remote}" && echo yes || echo no', check=True)
        if probe.stdout.strip() == "yes":
            targets.append(group.output_dir)
    print(f"fetch {len(targets)} output dirs from {REMOTE_HOST}", flush=True)
    _rsync(targets, dry_run=dry_run, from_remote=True)


def _container_key(payload: dict[str, Any]) -> str:
    for key in (
        "eval_results_by_size",
        "results_by_split",
        "eval_results_by_scale",
        "results_by_size",
    ):
        if key in payload:
            return key
    raise KeyError("no results container in payload")


def _scale_keys(task: str, container: dict[str, Any]) -> list[str]:
    if task == "tsp_construct":
        return ["tsp50", "tsp100", "tsp200"]
    if task == "vrptw_construct":
        return ["vrptw50", "vrptw100", "vrptw200"]
    if task in {"cvrp_aco", "op_aco"}:
        return ["test_50", "test_100", "test_200"]
    if task == "online_bin_packing":
        return [
            "1k_100",
            "5k_100",
            "10k_100",
            "1k_500",
            "5k_500",
            "10k_500",
        ]
    return list(container)


def _objective(row: dict[str, Any]) -> float:
    for key in ("eval_objective", "bins_used_mean", "objective"):
        value = row.get(key)
        if value is not None:
            return float(value)
    raise KeyError(row)


def _mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), (
        statistics.stdev(values) if len(values) >= 2 else 0.0
    )


def cmd_summarize(groups: list[EvalGroup]) -> None:
    summary_path = REPO / "docs/experiments/其他实验/budget500-heldout-summary.json"
    rows: list[dict[str, Any]] = []
    for group in sorted(groups, key=lambda item: (item.task, item.label)):
        result_path = group.output_dir / "results.json"
        if not result_path.exists():
            print(f"skip missing {result_path}", file=sys.stderr)
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        container_key = _container_key(payload)
        container = payload[container_key]
        entry: dict[str, Any] = {
            "task": group.task,
            "method": group.method_key,
            "label": group.label,
            "output_dir": str(group.output_dir.relative_to(REPO)),
            "source_eval": str(group.source_eval.relative_to(REPO)),
            "scales": {},
        }
        for scale in _scale_keys(group.task, container):
            block = container[scale]
            values = [_objective(row) for row in block["results"]]
            mean, std = _mean_std(values)
            entry["scales"][scale] = {
                "mean": mean,
                "sample_std": std,
                "sample_orders": [
                    row.get("best_sample_order") for row in payload.get("run_records", [])
                ],
            }
        rows.append(entry)
        cells = " | ".join(
            f"{entry['scales'][scale]['mean']:.6f} ± {entry['scales'][scale]['sample_std']:.6f}"
            for scale in entry["scales"]
        )
        print(f"{group.task:20s} {group.label:18s} {cells}")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {"budget": BUDGET, "groups": rows},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwritten {summary_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["sync", "run-remote", "fetch", "summarize", "all"],
    )
    parser.add_argument("--jobs", type=int, default=8, help="parallel remote eval jobs")
    parser.add_argument("--workers", type=int, default=12, help="eval workers per job")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    groups = load_groups()
    if args.command == "sync":
        cmd_sync(groups, dry_run=args.dry_run)
    elif args.command == "run-remote":
        cmd_run_remote(groups, jobs=args.jobs, workers=args.workers, dry_run=args.dry_run)
    elif args.command == "fetch":
        cmd_fetch(groups, dry_run=args.dry_run)
    elif args.command == "summarize":
        cmd_summarize(groups)
    else:
        cmd_sync(groups, dry_run=args.dry_run)
        cmd_run_remote(groups, jobs=args.jobs, workers=args.workers, dry_run=args.dry_run)
        cmd_fetch(groups, dry_run=args.dry_run)
        cmd_summarize(groups)


if __name__ == "__main__":
    main()
