"""Evaluate finished TSP Construct runs on held-out sizes with fixed seed=2025.

用法:
    uv run python experiments/tsp_construct/evaluate_best_on_test.py <run_dir> [...] \\
      [--output-dir DIR] [--sizes 50,100,200] [--timeout 1000] [--workers 16]

不指定 --sample-order 时自动取该 run 里 score 最高的样本（=best）。
指定 --output-dir 时批量评估多个 run 并写出 results.json。
"""
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

from experiments.eval_artifacts import load_run_summary, load_scored_samples
from llm4ad.base.evaluate import SecureEvaluator
from llm4ad.task.optimization.tsp_construct import TSPEvaluation

TASK = "tsp_construct"
TRAIN_SEED = 2024
EVAL_SEED = 2025
DEFAULT_SIZES = (50, 100, 200)
DEFAULT_TIMEOUT = 1000
DEFAULT_WORKERS = 16


def _resolve_method(run_dir: Path) -> str:
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        method = config.get("method")
        if isinstance(method, str) and method.strip():
            return method.strip()
    return run_dir.parent.name


def pick_sample(run_dir: str | Path, sample_order: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    load_run_summary(Path(run_dir))
    samples = load_scored_samples(Path(run_dir))
    if not samples:
        raise RuntimeError(f"no valid samples under {run_dir}")
    if sample_order is None:
        return max(samples, key=lambda x: float(x["score"])), samples
    for x in samples:
        if x.get("sample_order") == sample_order:
            return x, samples
    raise RuntimeError(f"sample_order={sample_order} not found among {len(samples)} valid samples")


def _eval_one_instance(args: tuple[str, int, Any]) -> float | None:
    program, size, dataset = args
    namespace: dict[str, Any] = {}
    exec(program, namespace)
    task = TSPEvaluation(timeout_seconds=None, n_instance=1, problem_size=size, seed=0)
    task._datasets = [dataset]
    return task.evaluate(namespace["select_next_node"])


def eval_size(
    program: str,
    size: int,
    seed: int,
    n_instance: int = 16,
    timeout: int | None = 120,
    workers: int = 1,
) -> tuple[float | None, float]:
    if workers <= 1:
        task = TSPEvaluation(timeout_seconds=timeout, n_instance=n_instance, problem_size=size, seed=seed)
        evaluator = SecureEvaluator(task)
        return evaluator.evaluate_program_record_time(program)

    task = TSPEvaluation(timeout_seconds=None, n_instance=n_instance, problem_size=size, seed=seed)
    worker_args = [(program, size, dataset) for dataset in task._datasets]
    started_at = time.time()
    pool = multiprocessing.get_context("spawn").Pool(processes=min(workers, n_instance))
    try:
        if timeout is None:
            scores = pool.map(_eval_one_instance, worker_args)
        else:
            scores = pool.map_async(_eval_one_instance, worker_args).get(timeout=timeout)
    except multiprocessing.TimeoutError:
        pool.terminate()
        pool.join()
        return None, time.time() - started_at
    else:
        pool.close()
        pool.join()
        if any(score is None for score in scores):
            return None, time.time() - started_at
        return float(sum(scores) / len(scores)), time.time() - started_at


def _mean_std(values: list[float]) -> dict[str, float | None]:
    finite = [value for value in values if math.isfinite(value)]
    return {
        "mean": statistics.fmean(finite) if finite else None,
        "sample_std": statistics.stdev(finite) if len(finite) >= 2 else None,
    }


def _run_batch(run_dirs: list[Path], output_dir: Path, sizes: list[int], timeout: int, workers: int) -> None:
    methods = {_resolve_method(run_dir) for run_dir in run_dirs}
    if len(methods) != 1:
        raise ValueError(f"all run directories must belong to one method: {sorted(methods)}")
    method = next(iter(methods))
    output_dir.mkdir(parents=True, exist_ok=True)

    model = "unknown"
    run_records: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        best, all_samples = pick_sample(run_dir, None)
        program = str(best["program"])
        sample_order = int(best["sample_order"])
        program_path = output_dir / f"{run_dir.name}_sample_{sample_order}_program.py"
        program_path.write_text(program.rstrip() + "\n", encoding="utf-8")
        config_path = run_dir / "run_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            model = config.get("llm", {}).get("model", model)
        train_score, train_seconds = eval_size(program, 50, TRAIN_SEED, timeout=timeout, workers=workers)
        if train_score is None:
            raise RuntimeError(f"train sanity eval timed out for {run_dir.name}")
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
                "train_recomputed_score": float(train_score),
                "train_eval_seconds": train_seconds,
                "program_path": relative_program_path,
                "program": program,
            }
        )

    eval_results_by_size: dict[str, Any] = {}
    for size in sizes:
        rows: list[dict[str, Any]] = []
        for row in run_records:
            score, eval_seconds = eval_size(
                row["program"], size, EVAL_SEED, timeout=timeout, workers=workers
            )
            if score is None:
                raise RuntimeError(
                    f"evaluation returned no score for {row['run_name']} TSP{size}; "
                    "the evaluation must complete with a valid score"
                )
            rows.append(
                {
                    "run_name": row["run_name"],
                    "best_sample_order": row["best_sample_order"],
                    "best_operator": row["best_operator"],
                    "eval_score": score,
                    "eval_objective": -score,
                    "eval_seconds": eval_seconds,
                    "program_path": row["program_path"],
                }
            )
        objectives = [float(item["eval_objective"]) for item in rows]
        scores = [float(item["eval_score"]) for item in rows]
        eval_results_by_size[f"tsp{size}"] = {
            "problem_size": size,
            "eval_config": {
                "n_instance": 16,
                "problem_size": size,
                "seed": EVAL_SEED,
                "timeout_seconds": timeout,
                "workers": workers,
                "evaluation_mode": "complete_run",
            },
            "results": rows,
            "summary": {
                "num_runs": len(rows),
                "num_successful_eval_runs": len(rows),
                "mean_eval_score": _mean_std(scores)["mean"],
                "sample_std_eval_score": _mean_std(scores)["sample_std"],
                "mean_eval_objective": _mean_std(objectives)["mean"],
                "sample_std_eval_objective": _mean_std(objectives)["sample_std"],
            },
        }

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": TASK,
        "method": method,
        "model": model,
        "source": f"{len(run_records)} completed {method} tsp_construct repeat(s)",
        "problem_sizes": sizes,
        "eval_timeout_seconds": timeout,
        "eval_workers": workers,
        "score_semantics": "score is negative mean tour length; higher score is better and lower objective is better",
        "run_records": [
            {key: value for key, value in row.items() if key != "program"} for row in run_records
        ],
        "eval_results_by_size": eval_results_by_size,
    }
    output_path = output_dir / "results.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="finished run directories")
    ap.add_argument("--sample-order", type=int, default=None, help="single-run mode only")
    ap.add_argument("--sizes", default=",".join(str(x) for x in DEFAULT_SIZES))
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="timeout per size in seconds")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    if not sizes:
        raise ValueError("--sizes must contain at least one size")

    run_dirs = [Path(p).resolve() for p in args.run_dirs]
    if args.output_dir is not None:
        if args.sample_order is not None:
            raise ValueError("--sample-order is only supported in single-run print mode")
        output_dir = args.output_dir
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        _run_batch(run_dirs, output_dir, sizes, args.timeout, args.workers)
        return

    if len(run_dirs) != 1:
        raise ValueError("print mode accepts exactly one run_dir; use --output-dir for batch")
    run_dir = run_dirs[0]
    picked, samples = pick_sample(run_dir, args.sample_order)
    program = picked["program"]
    print(f"run_dir     : {run_dir}")
    print(
        f"valid samples: {len(samples)}; using sample_order={picked['sample_order']} "
        f"(logged score={picked['score']:.6f}, op={picked.get('operator')})"
    )
    tr, _ = eval_size(program, 50, TRAIN_SEED, timeout=args.timeout, workers=args.workers)
    if tr is None:
        raise RuntimeError("train sanity eval timed out")
    print(
        f"  [sanity train  | size=50 seed=2024] score={tr:.6f}  "
        f"(logged {picked['score']:.6f}, diff {abs(tr - picked['score']):.2e})"
    )
    for sz in sizes:
        sc, t = eval_size(program, sz, EVAL_SEED, timeout=args.timeout, workers=args.workers)
        if sc is None:
            raise RuntimeError(f"test tsp{sz} timed out after {t:.2f}s")
        print(f"  [test tsp{sz:<3}| seed=2025] score={sc:.6f}  (eval_time {t:.2f}s)")


if __name__ == "__main__":
    main()
