import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from llm4ad.method.pathwise import PathWise, PathWiseProfiler
from llm4ad.task.optimization.main.tsp_construct import TSPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi


def main():
    llm = HttpsApi(
        host="xxx",       # e.g. "api.openai.com"
        key="sk-xxx",     # e.g. "sk-..."
        model="xxx",      # e.g. "gpt-4o-mini"
        timeout=60,
    )

    task = TSPEvaluation(timeout_seconds=30, n_instance=4, problem_size=20)
    method = PathWise(
        llm=llm,
        evaluation=task,
        profiler=PathWiseProfiler(log_dir="logs", log_style="complex"),
        max_sample_nums=30,
        pop_size=4,
        init_pop_size=8,
        num_actions=2,
        num_rollouts=2,
        max_inner_steps=2,
        num_evaluators=1,
    )
    method.run()


if __name__ == "__main__":
    main()
