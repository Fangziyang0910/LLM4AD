"""TraceAAD2 smoke test: mock LLM + FakeEvaluation 跑通 init + iterations。

验证三层记忆/三回路/6 算子/portfolio/泛化信用/因果 context 全链路无异常。
不触网：FakeLLM 按提示类型返回程序/actions，FakeEvaluation 返回递增 fitness。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad2 import TraceAAD2

TEMPLATE = '''def select_next_node(current_node: int, unvisited_nodes: set, distance_matrix) -> int:
    """
    Select the next node to visit.
    """
    return min(unvisited_nodes, key=lambda n: distance_matrix[current_node][n])
'''
TASK_DESC = "Design a constructive heuristic to select the next node in TSP."


class FakeLLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.n = 0

    def draw_sample(self, prompt, *args, **kwargs):
        self.n += 1
        if "Generate a complete implementation" in prompt:
            return self._program(self.n, "init")
        if "next-step modifications" in prompt:
            return ("1. Add local density scoring to the candidate selection.\n"
                    "2. Replace greedy pick with a nearest-neighbor ranking rule.\n")
        if "Requested Modification" in prompt:
            return self._program(self.n, "refine")
        return self._program(self.n, "fallback")

    def _program(self, n: int, prefix: str) -> str:
        idea = f"{prefix} heuristic #{n}"
        code = (
            "def select_next_node(current_node, unvisited_nodes, distance_matrix):\n"
            f"    score = {n}\n"
            "    best = None\n"
            "    for node in unvisited_nodes:\n"
            "        d = distance_matrix[current_node][node]\n"
            "        if best is None or d < best[1]:\n"
            "            best = (node, d)\n"
            "    return best[0]\n"
        )
        return f"Idea: {idea}\nCode:\n```python\n{code}```"


class FakeEvaluation(Evaluation):
    def __init__(self) -> None:
        super().__init__(
            template_program=TEMPLATE, task_description=TASK_DESC,
            use_numba_accelerate=False, safe_evaluate=False, timeout_seconds=10,
        )
        self.calls = 0

    def evaluate_program(self, program_str, callable_func, **kwargs):
        self.calls += 1
        return -10.0 + 0.05 * self.calls  # maximize: 递增 → 持续改进


def run_smoke():
    llm = FakeLLM()
    ev = FakeEvaluation()
    method = TraceAAD2(
        llm=llm, evaluation=ev, profiler=None,
        max_sample_nums=14, n_init=3, actions_per_iteration=2,
        n_islands=2, max_per_island=10, maximize=True,
        num_evaluators=1, multi_thread_or_process_eval="thread",
    )
    result = method.run()
    print("=== TraceAAD2 smoke result ===")
    print(f"n_samples={result.n_samples} n_nodes={result.n_total_nodes} "
          f"n_valid={result.n_valid_nodes} n_traj={result.n_trajectories} n_edges={result.n_edges}")
    print(f"best_fitness={result.best_node.fitness if result.best_node else None}")
    print(f"portfolio={method._portfolio.snapshot()}")
    assert result.best_node is not None, "best_node should exist"
    assert result.n_samples > 0
    assert result.n_trajectories > 0
    print("SMOKE OK")


def test_traceaad2_smoke():
    run_smoke()


if __name__ == "__main__":
    run_smoke()
