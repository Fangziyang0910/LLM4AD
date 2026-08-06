"""Evaluate the best heuristic from finished Online Bin Packing search runs.

Train in-domain: Weibull 1k/5k × C∈{100,500}, one fixed instance
per configuration. Held-out test: different fixed instances for
1k/5k/10k × C∈{100,500}; only the 10k configurations are OOD.

    uv run python experiments/online_bin_packing/evaluate_best_on_test.py \\
      experiments/online_bin_packing/mcts_ahd/<run_rep1> \\
      experiments/online_bin_packing/mcts_ahd/<run_rep2> \\
      experiments/online_bin_packing/mcts_ahd/<run_rep3> \\
      --output-dir experiments/online_bin_packing/mcts_ahd/eval_best_<tag>
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm4ad.task.optimization.generated_data_config import get_generated_task_kwargs
from llm4ad.task.optimization.online_bin_packing import OBPEvaluation


TASK = "online_bin_packing"
# Paper-aligned test scales (EoH / ReEvo / MCTS-AHD / PathWise).
DEFAULT_TEST_SCALES = (
    (1000, 100),
    (5000, 100),
    (10000, 100),
    (1000, 500),
    (5000, 500),
    (10000, 500),
)


def _scale_key(n_items: int, capacity: int) -> str:
    if n_items % 1000 == 0 and n_items >= 1000:
        return f"{n_items // 1000}k_{capacity}"
    return f"{n_items}_{capacity}"


def _parse_scales(text: str) -> list[tuple[int, int]]:
    scales: list[tuple[int, int]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "_" not in part:
            raise ValueError(f"invalid scale {part!r}; expected like 5k_100 or 5000_100")
        left, right = part.rsplit("_", 1)
        capacity = int(right)
        if left.endswith("k") or left.endswith("K"):
            n_items = int(left[:-1]) * 1000
        else:
            n_items = int(left)
        scales.append((n_items, capacity))
    return scales


def task_kwargs_for_scale(
    base_kwargs: dict[str, Any],
    n_items: int,
    capacity: int,
) -> dict[str, Any]:
    """Select one fixed held-out scale from the task's evaluation protocol."""
    kwargs = dict(base_kwargs)
    dataset_specs = kwargs.get("dataset_specs")
    if dataset_specs is None:
        kwargs.update(n_items=n_items, capacity=capacity)
        return kwargs

    for spec in dataset_specs:
        if int(spec["n_items"]) == n_items and capacity in spec["capacities"]:
            kwargs["dataset_specs"] = [
                {
                    "n_instances": int(spec["n_instances"]),
                    "n_items": n_items,
                    "capacities": [capacity],
                }
            ]
            return kwargs
    raise ValueError(
        f"scale {n_items}_{capacity} is not part of the fixed OBP test protocol"
    )


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


def _load_best_sample(
    run_dir: Path,
    max_sample_order: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from experiments.eval_artifacts import pick_best_sample

    return pick_best_sample(run_dir, max_sample_order=max_sample_order)


def _evaluate_program(program: str, task_kwargs: dict[str, Any]) -> tuple[float, float]:
    namespace: dict[str, Any] = {}
    exec(program, namespace)
    if "priority" not in namespace or not callable(namespace["priority"]):
        raise RuntimeError("program does not define callable priority(...)")
    evaluator = OBPEvaluation(**task_kwargs)
    started_at = time.time()
    score = evaluator.evaluate(namespace["priority"])
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
        description="Evaluate best OBP heuristics on held-out Weibull multi-scale tests."
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scales",
        default=",".join(_scale_key(n, c) for n, c in DEFAULT_TEST_SCALES),
        help="comma-separated scales like 1k_100,5k_100,...,10k_500",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="per-program timeout; default None disables timeout",
    )
    parser.add_argument(
        "--max-sample-order",
        type=int,
        default=None,
        help="only consider candidates up to this search evaluation",
    )
    args = parser.parse_args()
    scales = _parse_scales(args.scales)
    if not scales:
        raise ValueError("--scales must contain at least one scale")
    if args.max_sample_order is not None and args.max_sample_order <= 0:
        raise ValueError("--max-sample-order must be positive")

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
    eval_base_kwargs = {
        **get_generated_task_kwargs(TASK, "eval"),
        "timeout_seconds": args.timeout_seconds,
    }

    run_records: list[dict[str, Any]] = []
    model = "unknown"
    for run_dir in run_dirs:
        best, all_samples = _load_best_sample(
            run_dir,
            max_sample_order=args.max_sample_order,
        )
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
                "max_sample_order": args.max_sample_order,
                "best_sample_order": sample_order,
                "best_operator": best.get("operator"),
                "train_artifact_score": float(best["score"]),
                "train_recomputed_score": train_score,
                "train_eval_seconds": train_seconds,
                "program_path": relative_program_path,
                "program": program,
            }
        )

    eval_results_by_scale: dict[str, Any] = {}
    for n_items, capacity in scales:
        key = _scale_key(n_items, capacity)
        eval_kwargs = task_kwargs_for_scale(eval_base_kwargs, n_items, capacity)
        rows: list[dict[str, Any]] = []
        for row in run_records:
            score, eval_seconds = _evaluate_program(row["program"], eval_kwargs)
            rows.append(
                {
                    "run_name": row["run_name"],
                    "best_sample_order": row["best_sample_order"],
                    "best_operator": row["best_operator"],
                    "eval_score": score,
                    "bins_used_mean": -score,
                    "eval_seconds": eval_seconds,
                    "program_path": row["program_path"],
                }
            )
        scores = [float(item["eval_score"]) for item in rows]
        bins_used = [float(item["bins_used_mean"]) for item in rows]
        eval_results_by_scale[key] = {
            "n_items": n_items,
            "capacity": capacity,
            "eval_config": eval_kwargs,
            "results": rows,
            "summary": {
                "num_runs": len(rows),
                "num_successful_eval_runs": len(scores),
                "mean_eval_score": _mean_std(scores)["mean"],
                "sample_std_eval_score": _mean_std(scores)["sample_std"],
                "mean_bins_used": _mean_std(bins_used)["mean"],
                "sample_std_bins_used": _mean_std(bins_used)["sample_std"],
            },
        }

    train_scores = [float(row["train_recomputed_score"]) for row in run_records]
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": TASK,
        "method": method,
        "model": model,
        "source": f"{len(run_records)} completed {method} online_bin_packing repeat(s)",
        "test_scales": [_scale_key(n, c) for n, c in scales],
        "eval_timeout_seconds": args.timeout_seconds,
        "max_sample_order": args.max_sample_order,
        "split_configs": {"train": train_kwargs, "eval_base": eval_base_kwargs},
        "score_semantics": (
            "OBPEvaluation returns negative mean bins used across instances; "
            "higher score is better. bins_used_mean = -score."
        ),
        "run_records": [
            {key: value for key, value in row.items() if key != "program"} for row in run_records
        ],
        "eval_results_by_scale": eval_results_by_scale,
        "summary": {
            "num_runs": len(run_records),
            "mean_train_recomputed_score": _mean_std(train_scores)["mean"],
            "sample_std_train_recomputed_score": _mean_std(train_scores)["sample_std"],
        },
    }
    output_path = output_dir / "results.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
