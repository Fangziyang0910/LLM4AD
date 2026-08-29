"""Integration smoke: V9.19 starts and runs on all five tasks with a scripted LLM.

The tracked evaluation (benchmark fitness plus behavior trajectories on the
training instances) is real; only LLM requests are scripted, so these tests
exercise the full mechanism (initialization, landscape statistics, action
selection, checkpointing) without network access.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.runners._common import build_task
from llm4ad.base import LLM
from llm4ad.method.traceaad_v9_19 import (
    BEHAVESIM_PROTOCOL_ID,
    TRACKED_EVALUATIONS,
    RunArtifacts,
    TraceAADV919,
)

N_ROOTS = 8
ORDINARY_SLOTS = {"tsp_construct": 4, "vrptw_construct": 4, "online_bin_packing": 4, "op_aco": 2, "cvrp_aco": 2}

TSP_CODE = """import numpy as np

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    d = distance_matrix[current_node][unvisited_nodes]
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    scores = -({a} * d + {b} * d_dest)
    return int(unvisited_nodes[int(np.argmax(scores))])
"""

VRPTW_CODE = """import numpy as np

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: float, current_time: float,
                     demands: np.ndarray, distance_matrix: np.ndarray, time_windows: np.ndarray) -> int:
    d = distance_matrix[current_node][unvisited_nodes]
    d_depot = distance_matrix[unvisited_nodes, depot]
    scores = {a} * d + {b} * d_depot
    return int(unvisited_nodes[int(np.argmin(scores))])
"""

OBP_CODE = """import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    return {a} * float(item) - {b} * bins.astype(np.float64)
"""

OP_CODE = """import numpy as np

def heuristics(prize: np.ndarray, distance: np.ndarray, maxlen: float) -> np.ndarray:
    return {a} * prize[:, None] / np.power(distance, {p})
"""

CVRP_CODE = """import numpy as np

def heuristics(distance_matrix: np.ndarray, coordinates: np.ndarray, demands: np.ndarray, capacity: int) -> np.ndarray:
    return {a} / np.power(distance_matrix, {p}) + {b} * demands[None, :].astype(np.float64)
"""

TASK_CODE = {
    "tsp_construct": TSP_CODE,
    "vrptw_construct": VRPTW_CODE,
    "online_bin_packing": OBP_CODE,
    "op_aco": OP_CODE,
    "cvrp_aco": CVRP_CODE,
}


class ScriptedLLM(LLM):
    """One unique, valid, task-compatible program per call."""

    def __init__(self, task: str) -> None:
        super().__init__()
        self.task = task
        self.calls = 0
        self.prompts: list[str] = []

    def draw_sample(self, prompt: str, *args, **kwargs) -> str:
        self.prompts.append(prompt)
        self.calls += 1
        step = self.calls
        fields = {
            "a": 1.0 + 0.031 * step,
            "b": 0.017 * step,
            "p": 0.8 + 0.03 * (step % 7),
        }
        code = TASK_CODE[self.task].format(**fields)
        return (
            f"Idea: heuristic mix a={fields['a']:.3f} b={fields['b']:.3f}\n"
            "Code:\n```python\n" + code + "```"
        )


def _tracked_evaluation(task: str):
    _, task_kwargs = build_task(task, eval_workers=1)
    return TRACKED_EVALUATIONS[task](**task_kwargs)


def _run_task(task: str, tmp_path: Path) -> tuple[TraceAADV919, Path, ScriptedLLM]:
    evaluation = _tracked_evaluation(task)
    llm = ScriptedLLM(task)
    budget = N_ROOTS + ORDINARY_SLOTS[task]
    run_dir = tmp_path / "run"
    method = TraceAADV919(
        llm=llm,
        evaluation=evaluation,
        artifacts=RunArtifacts(run_dir, console_output=False),
        budget=budget,
        n_roots=N_ROOTS,
        seed=0,
        checkpoint_dir=run_dir / "checkpoints",
    )
    method.run()
    return method, run_dir, llm


def _assert_task_run(method: TraceAADV919, run_dir: Path, task: str) -> None:
    budget = N_ROOTS + ORDINARY_SLOTS[task]
    n_algorithms = len(method._tree.valid_algorithms())
    assert method._n_eval == budget
    assert n_algorithms >= N_ROOTS
    matrix = method._landscape.matrix
    assert matrix.shape == (n_algorithms, n_algorithms)
    assert np.allclose(matrix, matrix.T)

    summary = json.loads((run_dir / "logs" / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "finished"
    assert summary["budget_slots"] == budget
    assert summary["n_roots"] == N_ROOTS
    assert summary["behave_protocol_id"] == BEHAVESIM_PROTOCOL_ID

    with (run_dir / "evaluations.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == method._n_calls
    assert [int(row["slot"]) for row in rows] == list(range(1, budget + 1))

    events = [
        json.loads(line)
        for line in (run_dir / "mechanism_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    pre = [event for event in events if event["event"] == "pre_decision"]
    actions = [event for event in events if event["event"] == "action_decision"]
    assert len(pre) == ORDINARY_SLOTS[task]
    assert len(actions) == ORDINARY_SLOTS[task]
    assert pre[0]["pool_size"] == N_ROOTS
    assert pre[0]["neighborhood_size"] == 2
    snapshot = pre[0]["snapshot"]
    assert len(snapshot) == N_ROOTS
    assert {"id", "q", "P", "U", "B", "c_t", "T", "S"} == set(snapshot[0])

    assert (run_dir / "best_program.py").is_file()
    history = [
        json.loads(line)
        for line in (run_dir / "best_history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert history
    assert history[-1]["child_id"] == summary["best_algorithm_id"]
    assert history[-1]["program"] in (run_dir / "best_program.py").read_text(encoding="utf-8")

    checkpoint = run_dir / "checkpoints" / "latest.json"
    behave_state = run_dir / "checkpoints" / "behave.npz"
    view_state = json.loads((run_dir / "checkpoints" / "view.json").read_text(encoding="utf-8"))
    assert checkpoint.is_file() and behave_state.is_file()
    assert view_state["n_eval"] == budget
    assert "code" not in view_state["nodes"][0]
    assert view_state["best_id"] == summary["best_algorithm_id"]


@pytest.mark.parametrize("task", ["tsp_construct", "online_bin_packing"])
def test_v919_runs_task(task: str, tmp_path: Path) -> None:
    method, run_dir, _ = _run_task(task, tmp_path)
    _assert_task_run(method, run_dir, task)


@pytest.mark.parametrize("task", ["vrptw_construct", "op_aco", "cvrp_aco"])
def test_v919_runs_slow_task(task: str, tmp_path: Path) -> None:
    method, run_dir, _ = _run_task(task, tmp_path)
    _assert_task_run(method, run_dir, task)


def test_v919_resume_restores_full_state(tmp_path: Path) -> None:
    task = "tsp_construct"
    method, run_dir, _ = _run_task(task, tmp_path)
    resumed = TraceAADV919(
        llm=ScriptedLLM(task),
        evaluation=_tracked_evaluation(task),
        budget=N_ROOTS + ORDINARY_SLOTS[task],
        n_roots=N_ROOTS,
        seed=0,
        checkpoint_dir=tmp_path / "resumed",
        resume_from=run_dir / "checkpoints" / "latest.json",
    )
    assert resumed._n_eval == method._n_eval
    assert np.allclose(resumed._landscape.matrix, method._landscape.matrix)
    assert resumed._landscape.distance(
        *resumed._landscape.node_ids[:2]
    ) == method._landscape.distance(*method._landscape.node_ids[:2])
    assert dict(resumed._outcome_counts) == dict(method._outcome_counts)
    assert dict(resumed._action_counts) == dict(method._action_counts)
    # a resumed run can continue making decisions from the restored landscape
    stats_pool = len(resumed._landscape.node_ids)
    assert stats_pool == len(method._tree.valid_algorithms())


def test_v919_action_prompt_matches_selected_parent(tmp_path: Path) -> None:
    task = "online_bin_packing"
    _, run_dir, llm = _run_task(task, tmp_path)
    assert llm.prompts
    for prompt in llm.prompts:
        assert "[Task]" in prompt
        assert "Output one concise Idea and one complete Python program" in prompt
        assert "Do not write a module or function docstring, and do not write comments." in prompt
        assert "def priority" in prompt
    ordinary = [prompt for prompt in llm.prompts if "[Current Algorithm]" in prompt]
    assert len(ordinary) == ORDINARY_SLOTS[task]
    assert any("Recent Algorithm Improvement History" in prompt for prompt in ordinary)
    events = [
        json.loads(line)
        for line in (run_dir / "mechanism_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    chosen = [
        event["action"] for event in events if event["event"] == "action_decision"
    ]
    assert set(chosen) <= {"develop", "explore", "crossover"}
    decisions = [
        json.loads(line)
        for line in (run_dir / "decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(decisions) == ORDINARY_SLOTS[task]
    assert all(record["action"] in {"develop", "explore", "crossover"} for record in decisions)
