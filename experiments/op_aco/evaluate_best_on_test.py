"""Evaluate completed OP-ACO runs on fixed held-out splits (OP50/100/200)."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.eval_artifacts import load_run_summary, load_scored_samples  # noqa: E402
from llm4ad.task.optimization.op_aco import OPACOEvaluation, load_split_instances  # noqa: E402


DEFAULT_SPLITS = ("test_50", "test_100", "test_200")
DEFAULT_WORKERS = 16
N_ANTS = 20
N_ITERATIONS = 50
ACO_SEED = 1234

_WORKER_EVALUATOR: OPACOEvaluation | None = None
_WORKER_HEURISTIC = None


def _resolve_method(run_dir: Path) -> str:
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        method = config.get("method")
        if isinstance(method, str) and method.strip():
            return method.strip()
    return run_dir.parent.name


def _pick_best(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    load_run_summary(run_dir)
    records = load_scored_samples(run_dir)
    if not records:
        raise RuntimeError(f"no valid samples found under {run_dir}")
    return max(records, key=lambda record: float(record["score"])), records


def _init_worker(program: str, split: str) -> None:
    global _WORKER_EVALUATOR, _WORKER_HEURISTIC
    namespace: dict[str, Any] = {}
    exec(program, namespace)
    _WORKER_HEURISTIC = namespace["heuristics"]
    _WORKER_EVALUATOR = OPACOEvaluation(
        split=split,
        n_ants=N_ANTS,
        n_iterations=N_ITERATIONS,
        aco_seed=ACO_SEED,
        timeout_seconds=None,
    )


def _evaluate_instance(args: tuple[int, Any]) -> tuple[int, float]:
    index, instance = args
    assert _WORKER_EVALUATOR is not None
    assert _WORKER_HEURISTIC is not None
    prize = _WORKER_EVALUATOR._solve_instance(instance, _WORKER_HEURISTIC, index)
    return index, float(prize)


def _evaluate_program(
    program: str,
    split: str,
    workers: int,
) -> tuple[float, list[float], float]:
    instances, _ = load_split_instances(split)
    started_at = time.time()
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        processes=min(workers, len(instances)),
        initializer=_init_worker,
        initargs=(program, split),
    ) as pool:
        indexed = pool.map(_evaluate_instance, list(enumerate(instances)))
    prizes = [prize for _, prize in sorted(indexed)]
    return statistics.fmean(prizes), prizes, time.time() - started_at


def _mean_std(values: list[float]) -> dict[str, float | None]:
    finite = [value for value in values if math.isfinite(value)]
    return {
        "mean": statistics.fmean(finite) if finite else None,
        "sample_std": statistics.stdev(finite) if len(finite) >= 2 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate best OP-ACO heuristics on held-out OP50/100/200."
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    splits = tuple(split.strip() for split in args.splits.split(",") if split.strip())
    if not splits:
        raise ValueError("--splits must contain at least one split")

    run_dirs = [run_dir.resolve() for run_dir in args.run_dirs]
    methods = {_resolve_method(run_dir) for run_dir in run_dirs}
    if len(methods) != 1:
        raise ValueError(f"all run directories must belong to one method: {sorted(methods)}")
    method = next(iter(methods))
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_records: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        best, all_samples = _pick_best(run_dir)
        program_path = output_dir / f"{run_dir.name}_sample_{best['sample_order']}_program.py"
        program_path.write_text(str(best["program"]).rstrip() + "\n", encoding="utf-8")
        config_path = run_dir / "run_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        run_records.append(
            {
                "run_dir": str(run_dir),
                "run_name": run_dir.name,
                "status": "finished",
                "num_valid_samples": len(all_samples),
                "best_sample_order": best["sample_order"],
                "best_operator": best.get("operator"),
                "train_best_score": float(best["score"]),
                "program_path": str(program_path),
                "run_config": config,
                "program": best["program"],
            }
        )

    results_by_split: dict[str, Any] = {}
    for split in splits:
        split_rows: list[dict[str, Any]] = []
        for row in run_records:
            score, prizes, seconds = _evaluate_program(row["program"], split, args.workers)
            split_rows.append(
                {
                    "run_name": row["run_name"],
                    "best_sample_order": row["best_sample_order"],
                    "best_operator": row["best_operator"],
                    "score": score,
                    "objective": score,
                    "eval_seconds": seconds,
                    "instance_prizes": prizes,
                    "program_path": row["program_path"],
                }
            )
        objectives = [row["objective"] for row in split_rows]
        results_by_split[split] = {
            "split": split,
            "metadata": load_split_instances(split)[1],
            "config": {
                "n_ants": N_ANTS,
                "n_iterations": N_ITERATIONS,
                "aco_seed": ACO_SEED,
                "workers": args.workers,
            },
            "results": split_rows,
            "summary": _mean_std(objectives),
        }

    first_config = run_records[0]["run_config"]
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": "op_aco",
        "method": method,
        "model": first_config.get("llm", {}).get("model", "unknown"),
        "score_semantics": (
            "score is mean collected prize across instances; higher is better"
        ),
        "run_records": [
            {key: value for key, value in row.items() if key not in {"program", "run_config"}}
            for row in run_records
        ],
        "results_by_split": results_by_split,
    }
    output_path = output_dir / "results.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
