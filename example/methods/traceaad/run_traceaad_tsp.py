from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm4ad.method.traceaad import TraceAAD, TraceAADProfiler
from llm4ad.task.optimization.tsp_construct import TSPEvaluation
from llm4ad.tools.llm.vllm_openai_api import VLLMOpenAIAPI


EXPERIMENT_DIR = Path(__file__).resolve().parent
LOG_DIR = EXPERIMENT_DIR / "logs"


def main():
    llm = VLLMOpenAIAPI(
        base_url="http://127.0.0.1:8001/v1",
        api_key="EMPTY",
        model="Qwen3.6-27B",
        timeout=120,
        max_tokens=16384,
        temperature=1.0,
        enable_thinking=False,
    )

    task = TSPEvaluation(
        timeout_seconds=20,
        split="train",
        eval_workers=16,
        eval_backend="process",
    )

    method = TraceAAD(
        llm=llm,
        evaluation=task,
        profiler=TraceAADProfiler(log_dir=str(LOG_DIR), log_style="complex", create_random_path=False),
        max_sample_nums=1000,
        n_init=4,
        actions_per_iteration=2,
        max_trajectory_length=8,
        max_active_trajectories=1000,
        sampling_strategy="trajectory_ucb",
        top_k=5,
        temperature=0.8,
        num_evaluators=4,
        multi_thread_or_process_eval="thread",
    )
    method.run()


if __name__ == "__main__":
    main()
