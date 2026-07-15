"""Evaluate the best heuristic from completed OP search runs."""

from __future__ import annotations

import json
import math
import multiprocessing
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm4ad.task.optimization.generated_data_config import get_generated_task_kwargs
from llm4ad.task.optimization.orienteering_construct import OrienteeringEvaluation


TASK = "orienteering_construct"
METHOD = "mcts_ahd"
MODEL = "qwen3.6-27b-awq"
PROBLEM_SIZES = [50, 100, 200]
EVAL_TIMEOUT_SECONDS = None
EVAL_WORKERS = 3
EVAL_INSTANCE_WORKERS = 16
RUN_DIRS = [
    Path("experiments/orienteering_construct/mcts_ahd/20260713_125413"),
    Path("experiments/orienteering_construct/mcts_ahd/20260713_125707"),
    Path("experiments/orienteering_construct/mcts_ahd/20260713_125712"),
]
OUTPUT_DIR = Path(__file__).resolve().parent / "mcts_ahd" / "eval_best_qwen36_27b_20260714"

_WORKER_EVALUATOR: OrienteeringEvaluation | None = None
_WORKER_HEURISTIC = None


def _load_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "logs" / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "finished" or summary.get("search_aborted"):
        raise RuntimeError(f"run is not a completed search: {run_dir}")
    return summary


def _load_best_sample(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _load_summary(run_dir)
    samples_dir = run_dir / "logs" / "samples"
    records: list[dict[str, Any]] = []
    for path in sorted(samples_dir.glob("samples_*.json")):
        if path.name == "samples_best.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        records.extend(
            record
            for record in data
            if isinstance(record, dict) and isinstance(record.get("score"), (int, float))
        )
    if not records:
        raise RuntimeError(f"no valid samples found under {samples_dir}")
    return max(records, key=lambda record: float(record["score"])), records


def _evaluate_program(program: str, task_kwargs: dict[str, Any]) -> tuple[float | None, float]:
    evaluator_kwargs = {**task_kwargs, "timeout_seconds": EVAL_TIMEOUT_SECONDS}
    evaluator = OrienteeringEvaluation(**evaluator_kwargs)
    instances = evaluator._datasets
    problem_size = int(evaluator.problem_size)
    started_at = time.time()
    with ProcessPoolExecutor(
        max_workers=min(EVAL_INSTANCE_WORKERS, len(instances)),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_instance_worker,
        initargs=(program, evaluator_kwargs, problem_size),
    ) as executor:
        scores = list(executor.map(_evaluate_instance, instances))
    return statistics.fmean(scores), time.time() - started_at


def _init_instance_worker(
    program: str,
    evaluator_kwargs: dict[str, Any],
    problem_size: int,
) -> None:
    global _WORKER_EVALUATOR, _WORKER_HEURISTIC
    namespace: dict[str, Any] = {}
    exec(program, namespace)
    _WORKER_HEURISTIC = namespace["select_next_node"]
    _WORKER_EVALUATOR = OrienteeringEvaluation(
        **{**evaluator_kwargs, "n_instance": 1, "problem_size": problem_size}
    )


def _evaluate_instance(instance: dict[str, Any]) -> float:
    if _WORKER_EVALUATOR is None or _WORKER_HEURISTIC is None:
        raise RuntimeError("instance worker was not initialized")
    solution = _WORKER_EVALUATOR.construct_solution(instance, _WORKER_HEURISTIC)
    if solution is None:
        raise RuntimeError("heuristic returned an invalid OP solution")
    return float(solution[1])


def _mean_std(values: list[float]) -> dict[str, float | None]:
    finite = [value for value in values if math.isfinite(value)]
    return {
        "mean": statistics.fmean(finite) if finite else None,
        "sample_std": statistics.stdev(finite) if len(finite) >= 2 else None,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_kwargs = {
        **get_generated_task_kwargs(TASK, "train"),
        "timeout_seconds": EVAL_TIMEOUT_SECONDS,
    }
    eval_base_kwargs = {
        **get_generated_task_kwargs(TASK, "eval"),
        "timeout_seconds": EVAL_TIMEOUT_SECONDS,
    }

    run_records: list[dict[str, Any]] = []
    for relative_run_dir in RUN_DIRS:
        run_dir = PROJECT_ROOT / relative_run_dir
        best, all_samples = _load_best_sample(run_dir)
        program = str(best["program"])
        sample_order = int(best["sample_order"])
        program_path = OUTPUT_DIR / f"{run_dir.name}_sample_{sample_order}_program.py"
        program_path.write_text(program.rstrip() + "\n", encoding="utf-8")
        train_score, train_seconds = _evaluate_program(program, train_kwargs)
        run_records.append(
            {
                "run_dir": str(relative_run_dir),
                "run_name": run_dir.name,
                "num_valid_samples": len(all_samples),
                "best_sample_order": sample_order,
                "best_operator": best.get("operator"),
                "train_artifact_score": float(best["score"]),
                "train_recomputed_score": train_score,
                "train_eval_seconds": train_seconds,
                "program_path": str(program_path.relative_to(PROJECT_ROOT)),
                "program": program,
            }
        )

    eval_results_by_size: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=EVAL_WORKERS) as executor:
        for problem_size in PROBLEM_SIZES:
            eval_kwargs = {
                **eval_base_kwargs,
                "problem_size": problem_size,
                "timeout_seconds": EVAL_TIMEOUT_SECONDS,
            }
            futures = [
                executor.submit(_evaluate_program, row["program"], eval_kwargs)
                for row in run_records
            ]
            rows: list[dict[str, Any]] = []
            for row, future in zip(run_records, futures):
                score, eval_seconds = future.result()
                if score is None:
                    raise RuntimeError(
                        f"evaluation returned no score for {row['run_name']} OP{problem_size}; "
                        "the no-timeout evaluation must complete with a valid score"
                    )
                rows.append(
                    {
                        "run_name": row["run_name"],
                        "best_sample_order": row["best_sample_order"],
                        "best_operator": row["best_operator"],
                        "eval_score": score,
                        "eval_seconds": eval_seconds,
                        "program_path": row["program_path"],
                    }
                )
            scores = [float(row["eval_score"]) for row in rows]
            eval_results_by_size[f"op{problem_size}"] = {
                "problem_size": problem_size,
                "eval_config": eval_kwargs,
                "results": rows,
                "summary": {
                    "num_runs": len(rows),
                    "num_successful_eval_runs": len(scores),
                    "mean_eval_score": _mean_std(scores)["mean"],
                    "sample_std_eval_score": _mean_std(scores)["sample_std"],
                },
            }

    train_scores = [
        float(row["train_recomputed_score"])
        for row in run_records
        if isinstance(row["train_recomputed_score"], (int, float))
        and math.isfinite(row["train_recomputed_score"])
    ]
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": TASK,
        "method": METHOD,
        "model": MODEL,
        "source": f"{len(run_records)} completed {METHOD} orienteering_construct repeat(s)",
        "problem_sizes": PROBLEM_SIZES,
        "eval_timeout_seconds": EVAL_TIMEOUT_SECONDS,
        "eval_workers": EVAL_WORKERS,
        "eval_instance_workers": EVAL_INSTANCE_WORKERS,
        "split_configs": {"train": train_kwargs, "eval_base": eval_base_kwargs},
        "score_semantics": "OrienteeringEvaluation returns mean collected prize; higher score is better.",
        "run_records": [
            {key: value for key, value in row.items() if key != "program"}
            for row in run_records
        ],
        "eval_results_by_size": eval_results_by_size,
        "summary": {
            "num_runs": len(run_records),
            "mean_train_recomputed_score": _mean_std(train_scores)["mean"],
            "sample_std_train_recomputed_score": _mean_std(train_scores)["sample_std"],
        },
    }
    output_path = OUTPUT_DIR / "results.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
