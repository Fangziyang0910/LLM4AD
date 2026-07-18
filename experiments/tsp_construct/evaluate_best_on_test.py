"""Evaluate a run's best evolved heuristic at multiple TSP sizes (tsp50/100/200) on held-out instances.

tsp_construct 的 split 只有 train(seed=2024)/eval(seed=2025)，均 problem_size=50。
本脚本在 eval 种子(2025, held-out) 下，把 best 启发式评估到 problem_size = 50/100/200，
用以测尺度泛化。score = -平均巡回长度（越高越好）。不同 size 的 score 量级不同，不可横比 size，
只能跨方法在同一个 size 上比。

用法:
    uv run python experiments/tsp_construct/evaluate_best_on_test.py <run_dir> [--sample-order N] [--sizes 50,100,200] [--timeout 1000] [--workers 16]
不指定 --sample-order 时自动取该 run 里 score 最高的样本（=best）。
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm4ad.base.evaluate import SecureEvaluator
from llm4ad.task.optimization.tsp_construct import TSPEvaluation

TASK = "tsp_construct"
TRAIN_SEED = 2024   # 搜索时用的
EVAL_SEED = 2025    # held-out 测试种子


def _load_samples(run_dir: str):
    files = [f for f in sorted(glob.glob(f"{run_dir}/logs/samples/samples_*.json")) if "best" not in f]
    out = []
    for f in files:
        try:
            with Path(f).open(encoding="utf-8") as sample_file:
                data = json.load(sample_file)
        except (OSError, json.JSONDecodeError, TypeError) as error:
            print(f"warning: skipping unreadable sample artifact {f}: {error}", file=sys.stderr)
            continue
        for x in data:
            if isinstance(x.get("score"), (int, float)):
                out.append(x)
    return out


def pick_sample(run_dir: str, sample_order: int | None):
    s = _load_samples(run_dir)
    if not s:
        raise RuntimeError(f"no valid samples under {run_dir}/logs/samples/")
    if sample_order is None:
        return max(s, key=lambda x: x["score"]), s
    for x in s:
        if x.get("sample_order") == sample_order:
            return x, s
    raise RuntimeError(f"sample_order={sample_order} not found among {len(s)} valid samples")


def _eval_one_instance(args):
    program, size, dataset = args
    namespace = {}
    exec(program, namespace)
    task = TSPEvaluation(timeout_seconds=None, n_instance=1, problem_size=size, seed=0)
    task._datasets = [dataset]
    return task.evaluate(namespace["select_next_node"])


def eval_size(
    program: str,
    size: int,
    seed: int,
    n_instance: int = 16,
    timeout: int = 120,
    workers: int = 1,
):
    if workers <= 1:
        task = TSPEvaluation(timeout_seconds=timeout, n_instance=n_instance, problem_size=size, seed=seed)
        evaluator = SecureEvaluator(task)
        return evaluator.evaluate_program_record_time(program)

    task = TSPEvaluation(timeout_seconds=None, n_instance=n_instance, problem_size=size, seed=seed)
    worker_args = [(program, size, dataset) for dataset in task._datasets]
    started_at = time.time()
    pool = multiprocessing.get_context("spawn").Pool(processes=min(workers, n_instance))
    try:
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
        return sum(scores) / len(scores), time.time() - started_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="e.g. experiments/tsp_construct/pathwise/20260710_123450")
    ap.add_argument("--sample-order", type=int, default=None, help="use this sample; default=best (max score)")
    ap.add_argument("--sizes", default="50,100,200", help="test problem sizes")
    ap.add_argument("--timeout", type=int, default=120, help="timeout per problem size in seconds")
    ap.add_argument("--workers", type=int, default=1, help="parallel held-out instances; 1 keeps serial evaluation")
    args = ap.parse_args()

    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    picked, samples = pick_sample(args.run_dir, args.sample_order)
    program = picked["program"]
    print(f"run_dir     : {args.run_dir}")
    print(f"valid samples: {len(samples)}; using sample_order={picked['sample_order']} "
          f"(logged score={picked['score']:.6f}, op={picked.get('operator')})")

    # sanity: train (size 50, seed 2024) —— 应与 logged score 完全一致
    tr, _ = eval_size(program, 50, TRAIN_SEED)
    print(f"  [sanity train  | size=50 seed=2024] score={tr:.6f}  "
          f"(logged {picked['score']:.6f}, diff {abs(tr - picked['score']):.2e})")

    for sz in sizes:
        sc, t = eval_size(program, sz, EVAL_SEED, timeout=args.timeout, workers=args.workers)
        if sc is None:
            print(f"  [test tsp{sz:<3}| seed=2025] timeout  (eval_time {t:.2f}s)")
        else:
            print(f"  [test tsp{sz:<3}| seed=2025] score={sc:.6f}  (eval_time {t:.2f}s)")


if __name__ == "__main__":
    main()
