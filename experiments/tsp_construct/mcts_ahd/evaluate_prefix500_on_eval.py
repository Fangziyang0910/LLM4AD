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
MAX_SAMPLE_ORDER = 500
PROBLEM_SIZES = [50, 100, 200]
EVAL_TIMEOUT_SECONDS = 60
RUN_DIRS = [
    Path("experiments/tsp_construct/mcts_ahd/20260709_213505"),
    Path("experiments/tsp_construct/mcts_ahd/20260709_213507"),
    Path("experiments/tsp_construct/mcts_ahd/20260709_213510"),
]
OUTPUT_DIR = Path(__file__).resolve().parent / "eval_prefix500_qwen36_27b_20260710"


def _load_best_prefix_sample(run_dir: Path) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for path in (run_dir / "logs" / "samples").glob("samples_*.json"):
        if path.name == "samples_best.json":
            continue
        samples.extend(json.loads(path.read_text(encoding="utf-8")))

    valid_samples = [
        sample
        for sample in samples
        if isinstance(sample.get("score"), (int, float))
        and math.isfinite(sample["score"])
        and int(sample.get("sample_order", MAX_SAMPLE_ORDER + 1)) <= MAX_SAMPLE_ORDER
    ]
    if not valid_samples:
        raise ValueError(f"No valid samples through {MAX_SAMPLE_ORDER} in {run_dir}")
    return max(valid_samples, key=lambda sample: float(sample["score"]))


def _evaluate_program(program: str, kwargs: dict[str, Any]) -> tuple[float | None, float]:
    started = time.time()
    score = SecureEvaluator(TSPEvaluation(**kwargs)).evaluate_program(program)
    return score, time.time() - started


def _objective(score: float | None) -> float | None:
    return -score if score is not None else None


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_records: list[dict[str, Any]] = []
    for relative_run_dir in RUN_DIRS:
        run_dir = PROJECT_ROOT / relative_run_dir
        sample = _load_best_prefix_sample(run_dir)
        program = sample["program"]
        train_score, train_eval_seconds = _evaluate_program(
            program, get_generated_task_kwargs(TASK, "train")
        )
        code_path = OUTPUT_DIR / f"{run_dir.name}_sample_{sample['sample_order']}_program.py"
        code_path.write_text(program.rstrip() + "\n", encoding="utf-8")
        run_records.append(
            {
                "run_dir": str(run_dir),
                "run_name": run_dir.name,
                "best_prefix_sample_order": sample["sample_order"],
                "best_operator": sample.get("operator"),
                "train_artifact_score": sample["score"],
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
        size_results: list[dict[str, Any]] = []
        for row in run_records:
            score, eval_seconds = _evaluate_program(row["program"], eval_kwargs)
            size_results.append(
                {
                    "run_name": row["run_name"],
                    "best_prefix_sample_order": row["best_prefix_sample_order"],
                    "best_operator": row["best_operator"],
                    "eval_score": score,
                    "eval_objective": _objective(score),
                    "eval_seconds": eval_seconds,
                    "program_path": row["program_path"],
                }
            )
        scores = [row["eval_score"] for row in size_results if isinstance(row["eval_score"], (int, float)) and math.isfinite(row["eval_score"])]
        objectives = [-score for score in scores]
        eval_results_by_size[f"tsp{problem_size}"] = {
            "problem_size": problem_size,
            "eval_config": eval_kwargs,
            "results": size_results,
            "summary": {
                "num_runs": len(size_results),
                "num_successful_eval_runs": len(scores),
                "mean_eval_score": statistics.fmean(scores) if scores else None,
                "sample_std_eval_score": statistics.stdev(scores) if len(scores) > 1 else None,
                "mean_eval_objective": statistics.fmean(objectives) if objectives else None,
                "sample_std_eval_objective": statistics.stdev(objectives) if len(objectives) > 1 else None,
            },
        }

    train_scores = [row["train_recomputed_score"] for row in run_records if isinstance(row["train_recomputed_score"], (int, float)) and math.isfinite(row["train_recomputed_score"])]
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": TASK,
        "method": METHOD,
        "model": MODEL,
        "source": "best valid sample from each completed MCTS-AHD tsp_construct repeat through evaluation 500",
        "max_sample_order": MAX_SAMPLE_ORDER,
        "problem_sizes": PROBLEM_SIZES,
        "eval_timeout_seconds": EVAL_TIMEOUT_SECONDS,
        "split_configs": {"train": get_generated_task_kwargs(TASK, "train"), "eval_base": eval_base_kwargs},
        "score_semantics": "TSPEvaluation returns negative average tour length; higher score is better, lower objective is better.",
        "run_records": [{key: value for key, value in row.items() if key != "program"} for row in run_records],
        "eval_results_by_size": eval_results_by_size,
        "summary": {
            "num_runs": len(run_records),
            "mean_train_recomputed_score": statistics.fmean(train_scores) if train_scores else None,
            "sample_std_train_recomputed_score": statistics.stdev(train_scores) if len(train_scores) > 1 else None,
        },
    }
    output_path = OUTPUT_DIR / "results.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
