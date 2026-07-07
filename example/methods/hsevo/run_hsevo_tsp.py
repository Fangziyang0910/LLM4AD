import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from llm4ad.method.hsevo import HSEvo, HSEvoProfiler
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
    method = HSEvo(
        llm=llm,
        evaluation=task,
        profiler=HSEvoProfiler(log_dir="logs", log_style="complex"),
        max_sample_nums=20,
        pop_size=2,
        init_pop_size=4,
        mutation_rate=0.5,
        hm_size=3,
        max_iter=2,
        num_evaluators=1,
    )
    method.run()


if __name__ == "__main__":
    main()

