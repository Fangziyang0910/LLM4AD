"""Evaluate the best heuristic from finished TSP-GLS search runs.

Train in-domain: TSP200. Held-out test: TSP100/200/500/1000 (seed=2025).

    uv run python experiments/tsp_gls/evaluate_best_on_test.py \\
      experiments/tsp_gls/pathwise/<run_rep1> \\
      experiments/tsp_gls/pathwise/<run_rep2> \\
      experiments/tsp_gls/pathwise/<run_rep3> \\
      --output-dir experiments/tsp_gls/pathwise/eval_best_<tag>
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
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm4ad.task.optimization.generated_data_config import get_generated_task_kwargs
from llm4ad.task.optimization.tsp_gls_2O import TSPGLSEvaluation


TASK = "tsp_gls_2O"
DEFAULT_SIZES = (100, 200, 500, 1000)


def _resolve_method(run_dir: Path) -> str:
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        method = config.get("method")
        if isinstance(method, str) and method.strip():
            return method.strip()
    parts = run_dir.parts
    if "version3" in parts or "version2" in parts:
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
    namespace: dict[str, Any] = {"np": np, "numpy": np}
    exec(program, namespace)
    if "update_edge_distance" not in namespace or not callable(namespace["update_edge_distance"]):
        raise RuntimeError("program does not define callable update_edge_distance(...)")
    evaluator = TSPGLSEvaluation(**task_kwargs)
    started_at = time.time()
    score = evaluator.evaluate_program(program, namespace["update_edge_distance"])
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
        description="Evaluate best TSP-GLS heuristics on held-out TSP100/200/500/1000."
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sizes",
        default=",".join(str(size) for size in DEFAULT_SIZES),
        help="comma-separated problem sizes (default: 100,200,500,1000)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="per-program timeout; default None disables timeout for multi-scale eval",
    )
    args = parser.parse_args()
    sizes = [int(item.strip()) for item in args.sizes.split(",") if item.strip()]
    if not sizes:
        raise ValueError("--sizes must contain at least one size")

    run_dirs = [run_dir.resolve() for run_dir in args.run_dirs]
    methods = {_resolve_method(run_dir) for run_dir in run_dirs}
    if len(methods) != 1:
        raise ValueError(f"all run directories must belong to one method: {sorted(methods)}")
    method = next(iter(methods))
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_kwargs = {**get_generated_task_kwargs(TASK, "train")}
    eval_base_kwargs = {**get_generated_task_kwargs(TASK, "eval")}
    if args.timeout_seconds is not None:
        train_kwargs["timeout_seconds"] = args.timeout_seconds
        eval_base_kwargs["timeout_seconds"] = args.timeout_seconds
    else:
        # Multi-scale eval (especially 500/1000) needs unlimited wall time.
        train_kwargs["timeout_seconds"] = None
        eval_base_kwargs["timeout_seconds"] = None

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
                "train_mean_tour_cost": -train_score,
                "train_eval_seconds": train_seconds,
                "program_path": relative_program_path,
                "program": program,
            }
        )

    eval_results_by_size: dict[str, Any] = {}
    for problem_size in sizes:
        eval_kwargs = {**eval_base_kwargs, "problem_size": problem_size}
        rows: list[dict[str, Any]] = []
        for row in run_records:
            score, eval_seconds = _evaluate_program(row["program"], eval_kwargs)
            rows.append(
                {
                    "run_name": row["run_name"],
                    "best_sample_order": row["best_sample_order"],
                    "best_operator": row["best_operator"],
                    "eval_score": score,
                    "eval_mean_tour_cost": -score,
                    "eval_seconds": eval_seconds,
                    "program_path": row["program_path"],
                }
            )
        scores = [float(item["eval_score"]) for item in rows]
        costs = [float(item["eval_mean_tour_cost"]) for item in rows]
        eval_results_by_size[f"tsp{problem_size}"] = {
            "problem_size": problem_size,
            "eval_config": eval_kwargs,
            "results": rows,
            "summary": {
                "num_runs": len(rows),
                "num_successful_eval_runs": len(scores),
                "mean_eval_score": _mean_std(scores)["mean"],
                "sample_std_eval_score": _mean_std(scores)["sample_std"],
                "mean_eval_tour_cost": _mean_std(costs)["mean"],
                "sample_std_eval_tour_cost": _mean_std(costs)["sample_std"],
            },
        }

    train_scores = [float(row["train_recomputed_score"]) for row in run_records]
    train_costs = [float(row["train_mean_tour_cost"]) for row in run_records]
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": TASK,
        "method": method,
        "model": model,
        "source": f"{len(run_records)} completed {method} tsp_gls repeat(s)",
        "problem_sizes": sizes,
        "eval_timeout_seconds": train_kwargs.get("timeout_seconds"),
        "split_configs": {"train": train_kwargs, "eval_base": eval_base_kwargs},
        "score_semantics": (
            "TSPGLSEvaluation returns -mean_tour_cost across instances; "
            "higher score is better. mean_tour_cost = -score."
        ),
        "run_records": [
            {key: value for key, value in row.items() if key != "program"} for row in run_records
        ],
        "eval_results_by_size": eval_results_by_size,
        "summary": {
            "num_runs": len(run_records),
            "mean_train_recomputed_score": _mean_std(train_scores)["mean"],
            "sample_std_train_recomputed_score": _mean_std(train_scores)["sample_std"],
            "mean_train_tour_cost": _mean_std(train_costs)["mean"],
            "sample_std_train_tour_cost": _mean_std(train_costs)["sample_std"],
        },
    }
    output_path = output_dir / "results.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
