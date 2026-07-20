"""Evaluate the best heuristic from finished Knapsack Construct search runs.

    uv run python experiments/knapsack_construct/evaluate_best_on_test.py \\
      experiments/knapsack_construct/mcts_ahd/20260719_223427_kp_rep1 \\
      experiments/knapsack_construct/mcts_ahd/20260719_223427_kp_rep2 \\
      experiments/knapsack_construct/mcts_ahd/20260719_223427_kp_rep3 \\
      --output-dir experiments/knapsack_construct/mcts_ahd/eval_best_20260719_223427
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm4ad.task.optimization.generated_data_config import get_generated_task_kwargs
from llm4ad.task.optimization.knapsack_construct import KnapsackEvaluation


TASK = "knapsack_construct"


def _resolve_method(run_dir: Path) -> str:
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        method = config.get("method")
        if isinstance(method, str) and method.strip():
            return method.strip()
    parts = run_dir.parts
    if "version2" in parts or "version1" in parts:
        return "traceaad"
    return run_dir.parent.name


def _load_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "logs" / "run_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"run is not finished: missing {summary_path}")
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


def _evaluate_program(program: str, task_kwargs: dict[str, Any]) -> tuple[float, float]:
    namespace: dict[str, Any] = {"List": List, "Tuple": Tuple}
    exec(program, namespace)
    if "select_next_item" not in namespace or not callable(namespace["select_next_item"]):
        raise RuntimeError("program does not define callable select_next_item(...)")
    evaluator = KnapsackEvaluation(**task_kwargs)
    started_at = time.time()
    score = evaluator.evaluate(namespace["select_next_item"])
    if score is None or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise RuntimeError(f"evaluation returned invalid score: {score!r}")
    return float(score), time.time() - started_at


def _mean_std(values: list[float]) -> dict[str, float | None]:
    finite = [value for value in values if math.isfinite(value)]
    return {
        "mean": statistics.fmean(finite) if finite else None,
        "sample_std": statistics.stdev(finite) if len(finite) >= 2 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate best KP heuristics from finished search runs on held-out seed=2025."
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="per-program timeout; default None disables timeout",
    )
    args = parser.parse_args()

    run_dirs = [run_dir.resolve() for run_dir in args.run_dirs]
    methods = {_resolve_method(run_dir) for run_dir in run_dirs}
    if len(methods) != 1:
        raise ValueError(f"all run directories must belong to one method: {sorted(methods)}")
    method = next(iter(methods))
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_kwargs = {
        **get_generated_task_kwargs(TASK, "train"),
        "timeout_seconds": args.timeout_seconds,
    }
    eval_kwargs = {
        **get_generated_task_kwargs(TASK, "eval"),
        "timeout_seconds": args.timeout_seconds,
    }

    run_records: list[dict[str, Any]] = []
    model = "unknown"
    for run_dir in run_dirs:
        best, all_samples = _load_best_sample(run_dir)
        program = str(best["program"])
        sample_order = int(best["sample_order"])
        program_path = output_dir / f"{run_dir.name}_sample_{sample_order}_program.py"
        program_path.write_text(program.rstrip() + "\n", encoding="utf-8")
        config_path = run_dir / "run_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            model = config.get("llm", {}).get("model", model)
        train_score, train_seconds = _evaluate_program(program, train_kwargs)
        eval_score, eval_seconds = _evaluate_program(program, eval_kwargs)
        try:
            relative_run_dir = str(run_dir.relative_to(PROJECT_ROOT))
        except ValueError:
            relative_run_dir = str(run_dir)
        try:
            relative_program_path = str(program_path.relative_to(PROJECT_ROOT))
        except ValueError:
            relative_program_path = str(program_path)
        run_records.append(
            {
                "run_dir": relative_run_dir,
                "run_name": run_dir.name,
                "num_valid_samples": len(all_samples),
                "best_sample_order": sample_order,
                "best_operator": best.get("operator"),
                "train_artifact_score": float(best["score"]),
                "train_recomputed_score": train_score,
                "train_eval_seconds": train_seconds,
                "eval_score": eval_score,
                "eval_seconds": eval_seconds,
                "program_path": relative_program_path,
            }
        )

    train_scores = [float(row["train_recomputed_score"]) for row in run_records]
    eval_scores = [float(row["eval_score"]) for row in run_records]
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": TASK,
        "method": method,
        "model": model,
        "source": f"{len(run_records)} completed {method} knapsack_construct repeat(s)",
        "eval_timeout_seconds": args.timeout_seconds,
        "split_configs": {"train": train_kwargs, "eval": eval_kwargs},
        "score_semantics": (
            "KnapsackEvaluation returns mean total value across instances; "
            "higher score is better."
        ),
        "run_records": run_records,
        "summary": {
            "num_runs": len(run_records),
            "mean_train_recomputed_score": _mean_std(train_scores)["mean"],
            "sample_std_train_recomputed_score": _mean_std(train_scores)["sample_std"],
            "mean_eval_score": _mean_std(eval_scores)["mean"],
            "sample_std_eval_score": _mean_std(eval_scores)["sample_std"],
        },
    }
    output_path = output_dir / "results.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
