"""Mock LLM 和评估器跑通 TraceAADV4 核心搜索流程。"""
from __future__ import annotations

from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad_v4 import TraceAADV4

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
    method = TraceAADV4(
        llm=llm, evaluation=ev,
        max_sample_nums=14, n_init=3, actions_per_iteration=2,
        max_active_trajectories=10, maximize=True,
    )
    result = method.run()
    print("=== TraceAADV4 smoke result ===")
    print(f"n_samples={result.n_samples} n_nodes={result.n_total_nodes} "
          f"n_valid={result.n_valid_nodes} n_traj={result.n_trajectories} n_edges={result.n_edges}")
    print(f"best_fitness={result.best_node.fitness if result.best_node else None}")
    assert result.best_node is not None, "best_node should exist"
    assert result.n_samples == 14
    assert result.n_trajectories > 0
    assert result.n_edges > 0
    print("SMOKE OK")


def test_traceaad_smoke():
    run_smoke()


if __name__ == "__main__":
    run_smoke()
