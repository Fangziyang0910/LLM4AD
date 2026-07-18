"""TraceAAD checkpoint round-trip: dump → load → continue search."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad import (
    TraceAAD,
    TraceAADProfiler,
    find_latest_checkpoint,
    resume_traceaad,
    save_checkpoint,
)
from llm4ad.method.traceaad.checkpoint import dump_traceaad_state, load_traceaad_state

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
            return (
                "1. Add local density scoring to the candidate selection.\n"
                "2. Replace greedy pick with a nearest-neighbor ranking rule.\n"
            )
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
            template_program=TEMPLATE,
            task_description=TASK_DESC,
            use_numba_accelerate=False,
            safe_evaluate=False,
            timeout_seconds=10,
        )
        self.calls = 0

    def evaluate_program(self, program_str, callable_func, **kwargs):
        self.calls += 1
        return -10.0 + 0.05 * self.calls


def _build_method(log_dir: Path | None, *, max_sample_nums: int, seed: int = 7) -> TraceAAD:
    profiler = None
    if log_dir is not None:
        profiler = TraceAADProfiler(
            log_dir=str(log_dir),
            log_style="complex",
            create_random_path=False,
        )
    return TraceAAD(
        llm=FakeLLM(),
        evaluation=FakeEvaluation(),
        profiler=profiler,
        max_sample_nums=max_sample_nums,
        n_init=3,
        actions_per_iteration=2,
        n_islands=2,
        max_per_island=10,
        maximize=True,
        num_evaluators=1,
        multi_thread_or_process_eval="thread",
        random_seed=seed,
    )


def test_checkpoint_round_trip_preserves_core_state(tmp_path: Path):
    log_dir = tmp_path / "logs"
    method = _build_method(log_dir, max_sample_nums=10)
    first = method.run()
    assert first.best_node is not None
    assert first.n_samples >= 3

    ckpt = find_latest_checkpoint(log_dir)
    assert ckpt.is_file()
    payload = json.loads(ckpt.read_text(encoding="utf-8"))
    assert payload["tot_sample_nums"] == method._tot_sample_nums
    assert payload["best_node_id"] == method._best_node.id

    restored = _build_method(log_dir, max_sample_nums=10, seed=99)
    load_traceaad_state(restored, payload)

    assert restored._tot_sample_nums == method._tot_sample_nums
    assert restored._stagnation == method._stagnation
    assert restored._best_node is not None
    assert restored._best_node.id == method._best_node.id
    assert restored._best_node.fitness == method._best_node.fitness
    assert restored._best_node.code == method._best_node.code
    assert len(restored._graph.nodes()) == len(method._graph.nodes())
    assert len(restored._graph.edges()) == len(method._graph.edges())
    assert len(restored._memory.active()) == len(method._memory.active())
    assert restored._memory._next_id == method._memory._next_id
    assert restored._graph._next_node_id == method._graph._next_node_id
    assert restored._rng.getstate() == method._rng.getstate()
    assert restored._portfolio.stats["endpoint_refine"].attempt_count == (
        method._portfolio.stats["endpoint_refine"].attempt_count
    )
    assert len(restored._curriculum.champion_events()) == len(
        method._curriculum.champion_events()
    )


def test_resume_continues_sample_budget(tmp_path: Path):
    log_dir = tmp_path / "logs"
    first = _build_method(log_dir, max_sample_nums=8)
    first_result = first.run()
    samples_after_first = first_result.n_samples
    best_after_first = first_result.best_node.fitness
    assert samples_after_first >= 3
    assert samples_after_first <= 8

    second = _build_method(log_dir, max_sample_nums=14, seed=123)
    resume_traceaad(second, log_dir)
    assert second._resume_mode is True
    assert second._tot_sample_nums == samples_after_first
    assert len(second._memory.active()) > 0

    second_result = second.run()
    assert second_result.n_samples > samples_after_first
    assert second_result.n_samples <= 14
    assert second_result.best_node is not None
    assert second_result.best_node.fitness >= best_after_first

    # New samples continue numbering in profiler / samples artifacts.
    sample_files = sorted((log_dir / "samples").glob("samples_*.json"))
    assert sample_files
    orders = []
    for path in sample_files:
        if path.name == "samples_best.json":
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            if isinstance(row, dict) and "sample_order" in row:
                orders.append(int(row["sample_order"]))
    assert max(orders) == second_result.n_samples


def test_save_checkpoint_without_profiler_is_noop():
    method = _build_method(None, max_sample_nums=6)
    method.run()
    assert save_checkpoint(method) is None
