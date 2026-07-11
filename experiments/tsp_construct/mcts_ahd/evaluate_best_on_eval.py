from __future__ import annotations

import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm4ad.base import SecureEvaluator
from llm4ad.task.optimization.generated_data_config import get_generated_task_kwargs
from llm4ad.task.optimization.tsp_construct import TSPEvaluation


TASK = "tsp_construct"
MODEL = "qwen3.6-27b-awq"
METHOD = "mcts_ahd"
PROBLEM_SIZES = [50, 100, 200]
EVAL_TIMEOUT_SECONDS = 60
RUN_DIRS = [
    Path("experiments/tsp_construct/mcts_ahd/20260709_213505"),
    Path("experiments/tsp_construct/mcts_ahd/20260709_213507"),
    Path("experiments/tsp_construct/mcts_ahd/20260709_213510"),
]
OUTPUT_DIR = Path(__file__).resolve().parent / "eval_best_qwen36_27b_20260710"


def _load_best_sample(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "logs" / "samples" / "samples_best.json"
    samples = json.loads(path.read_text(encoding="utf-8"))
    if not samples:
        raise ValueError(f"No best samples found in {path}")
    return samples[-1]


def _evaluate_program(program: str, kwargs: dict[str, Any]) -> tuple[float | None, float]:
    evaluator = TSPEvaluation(**kwargs)
    secure_evaluator = SecureEvaluator(evaluator)
    started = time.time()
    score = secure_evaluator.evaluate_program(program)
    return score, time.time() - started


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.stdev(values)


def _objective(score: float | None) -> float | None:
    if score is None:
        return None
    return -score


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_records: list[dict[str, Any]] = []
    for relative_run_dir in RUN_DIRS:
        run_dir = PROJECT_ROOT / relative_run_dir
        sample = _load_best_sample(run_dir)
        program = sample["program"]
        sample_order = sample["sample_order"]

        train_kwargs = get_generated_task_kwargs(TASK, "train")
        train_score, train_eval_seconds = _evaluate_program(program, train_kwargs)

        code_path = OUTPUT_DIR / f"{run_dir.name}_sample_{sample_order}_program.py"
        code_path.write_text(program.rstrip() + "\n", encoding="utf-8")

        run_records.append(
            {
                "run_dir": str(run_dir),
                "run_name": run_dir.name,
                "best_sample_order": sample_order,
                "best_operator": sample.get("operator"),
                "train_artifact_score": sample.get("score"),
                "train_recomputed_score": train_score,
                "train_recomputed_objective": _objective(train_score),
                "train_eval_seconds": train_eval_seconds,
                "program_path": str(code_path),
                "program": program,
            }
        )

    eval_results_by_size: dict[str, Any] = {}
    eval_base_kwargs = get_generated_task_kwargs(TASK, "eval")
    for problem_size in PROBLEM_SIZES:
        eval_kwargs = {
            **eval_base_kwargs,
            "problem_size": problem_size,
            "timeout_seconds": EVAL_TIMEOUT_SECONDS,
        }
        size_label = f"tsp{problem_size}"
        size_results: list[dict[str, Any]] = []

        for row in run_records:
            score, eval_seconds = _evaluate_program(row["program"], eval_kwargs)
            size_results.append(
                {
                    "run_name": row["run_name"],
                    "best_sample_order": row["best_sample_order"],
                    "best_operator": row["best_operator"],
                    "eval_score": score,
                    "eval_objective": _objective(score),
                    "eval_seconds": eval_seconds,
                    "program_path": row["program_path"],
                }
            )

        successful_eval_scores = [
            row["eval_score"]
            for row in size_results
            if isinstance(row["eval_score"], (int, float)) and math.isfinite(row["eval_score"])
        ]
        successful_eval_objectives = [-score for score in successful_eval_scores]
        eval_results_by_size[size_label] = {
            "problem_size": problem_size,
            "eval_config": eval_kwargs,
            "results": size_results,
            "summary": {
                "num_runs": len(size_results),
                "num_successful_eval_runs": len(successful_eval_scores),
                "mean_eval_score": _mean(successful_eval_scores) if successful_eval_scores else None,
                "sample_std_eval_score": _sample_std(successful_eval_scores),
                "mean_eval_objective": _mean(successful_eval_objectives) if successful_eval_objectives else None,
                "sample_std_eval_objective": _sample_std(successful_eval_objectives),
            },
        }

    successful_train_scores = [
        row["train_recomputed_score"]
        for row in run_records
        if isinstance(row["train_recomputed_score"], (int, float)) and math.isfinite(row["train_recomputed_score"])
    ]

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": TASK,
        "method": METHOD,
        "model": MODEL,
        "source": "latest three completed MCTS-AHD tsp_construct repeats",
        "problem_sizes": PROBLEM_SIZES,
        "eval_timeout_seconds": EVAL_TIMEOUT_SECONDS,
        "eval_timeout_source": "MCTS-AHD paper/reference config limit each heuristic evaluation on dataset D to 60 seconds.",
        "split_configs": {
            "train": get_generated_task_kwargs(TASK, "train"),
            "eval_base": eval_base_kwargs,
        },
        "score_semantics": "TSPEvaluation returns negative average tour length; higher score is better, lower objective is better.",
        "run_records": [
            {key: value for key, value in row.items() if key != "program"}
            for row in run_records
        ],
        "eval_results_by_size": eval_results_by_size,
        "summary": {
            "num_runs": len(run_records),
            "mean_train_recomputed_score": _mean(successful_train_scores) if successful_train_scores else None,
            "sample_std_train_recomputed_score": _sample_std(successful_train_scores),
        },
    }

    output_path = OUTPUT_DIR / "results.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
