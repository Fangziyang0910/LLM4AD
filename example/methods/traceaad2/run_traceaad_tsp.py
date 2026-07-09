"""TraceAAD2 on TSP Construct with a vLLM (OpenAI-compatible) endpoint.

入口示例：uv run python example/methods/traceaad2/run_traceaad_tsp.py
对比基线用 method/traceaad（v1）与 method/mcts_trace_ahd，各自独立目录、互不影响。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm4ad.method.traceaad2 import TraceAAD2, TraceAAD2Profiler
from llm4ad.task.optimization.main.tsp_construct import TSPEvaluation
from llm4ad.tools.llm.vllm_openai_api import VLLMOpenAIAPI

EXPERIMENT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run TraceAAD2 on TSP Construct with a vLLM endpoint.")
    p.add_argument("--base-url", default="http://222.201.145.8:8080/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--model", default="qwen3.6-27b-awq")
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-sample-nums", type=int, default=1000)
    p.add_argument("--n-init", type=int, default=4)
    p.add_argument("--actions-per-iteration", type=int, default=2)
    p.add_argument("--max-trajectory-length", type=int, default=8)
    p.add_argument("--n-islands", type=int, default=4)
    p.add_argument("--max-active-trajectories", type=int, default=1000)
    p.add_argument("--novelty-threshold", type=float, default=0.92)
    p.add_argument("--num-evaluators", type=int, default=4)
    p.add_argument("--eval-workers", type=int, default=32)
    p.add_argument("--eval-backend", choices=["sequential", "thread", "process"], default="process")
    p.add_argument("--eval-executor", choices=["thread", "process"], default="thread")
    p.add_argument("--task-timeout-seconds", type=float, default=20)
    p.add_argument("--split", default="train")
    p.add_argument("--max-consecutive-sample-failures", type=int, default=20)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    llm = VLLMOpenAIAPI(
        base_url=args.base_url, api_key=args.api_key, model=args.model,
        timeout=args.timeout, max_tokens=args.max_tokens,
        temperature=args.temperature, enable_thinking=False,
    )
    task = TSPEvaluation(
        timeout_seconds=args.task_timeout_seconds, split=args.split,
        eval_workers=args.eval_workers, eval_backend=args.eval_backend,
    )
    method = TraceAAD2(
        llm=llm, evaluation=task,
        profiler=TraceAAD2Profiler(log_dir=str(EXPERIMENT_DIR / "logs"), log_style="complex"),
        max_sample_nums=args.max_sample_nums,
        n_init=args.n_init,
        actions_per_iteration=args.actions_per_iteration,
        max_trajectory_length=args.max_trajectory_length,
        n_islands=args.n_islands,
        max_active_trajectories=args.max_active_trajectories,
        novelty_threshold=args.novelty_threshold,
        num_evaluators=args.num_evaluators,
        multi_thread_or_process_eval=args.eval_executor,
        max_consecutive_sample_failures=args.max_consecutive_sample_failures,
        debug_mode=args.debug,
    )
    method.run()


if __name__ == "__main__":
    main()
