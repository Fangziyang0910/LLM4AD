from __future__ import annotations

import json
from pathlib import Path

from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad_artifacts import TraceAADArtifacts
from llm4ad.method.traceaad_v9_3 import PROTOCOL_ID, TraceAADV93
from llm4ad.method.traceaad_v9_3.checkpoint import (
    CHECKPOINT_VERSION,
    save_checkpoint,
)
from llm4ad.method.traceaad_v9_3.context import canonical_window
from llm4ad.method.traceaad_v9_3.tree import FactGraph

TEMPLATE = """def choose(value: int) -> int:
    return value
"""


class TwoStageLLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.candidate_calls = 0
        self.decision_calls = 0
        self.prompts: list[str] = []

    def draw_sample(self, prompt, *args, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        if "Propose exactly" in prompt:
            count = int(prompt.split("Propose exactly ", 1)[1].split()[0])
            return "\n".join(
                f"Strategy {index}: distinct mechanism {index}"
                for index in range(1, count + 1)
            )
        if "[Decision]" in prompt:
            self.decision_calls += 1
            return f"Idea: trajectory decision {self.decision_calls}"
        self.candidate_calls += 1
        if "[Approved Next Idea]" in prompt:
            return (
                "Code:\n```python\n"
                "# representation noise\n"
                "def choose(value: int) -> int:\n"
                f"    return value + {self.candidate_calls}\n"
                "```"
            )
        return (
            f"Idea: root candidate {self.candidate_calls}\n"
            "```python\n"
            "def choose(value: int) -> int:\n"
            f"    return value + {self.candidate_calls}\n"
            "```"
        )


class SequenceEvaluation(Evaluation):
    def __init__(self, scores: list[float] | None = None) -> None:
        super().__init__(
            template_program=TEMPLATE,
            task_description="Improve choose.",
            use_numba_accelerate=False,
            safe_evaluate=False,
            timeout_seconds=10,
        )
        self.scores = scores
        self.calls = 0

    def evaluate_program(self, program_str, callable_func, **kwargs):
        self.calls += 1
        if self.scores is None:
            return float(self.calls)
        return self.scores[self.calls - 1]


def make_method(
    *,
    scores: list[float] | None = None,
    budget: int = 4,
    routes: int = 1,
    anchors: int = 1,
    run_dir: Path | None = None,
) -> tuple[TraceAADV93, TwoStageLLM]:
    llm = TwoStageLLM()
    method = TraceAADV93(
        llm=llm,
        evaluation=SequenceEvaluation(scores),
        profiler=None if run_dir is None else TraceAADArtifacts(run_dir=run_dir),
        max_sample_nums=budget,
        initial_route_pool_size=routes,
        initial_anchor_count=anchors,
        context_token_limit=24576,
        checkpoint_dir=None if run_dir is None else run_dir / "checkpoints",
    )
    return method, llm


def test_rollout_continues_through_regression_and_credits_best_once() -> None:
    method, _ = make_method(scores=[10.0, 8.0, 20.0, 15.0])
    result = method.run()
    root = method._graph.get_node(method._graph.root.child_ids[0])
    events = method._graph.events()

    assert result.n_samples == 4
    assert result.n_events == 3
    assert [event.outcome for event in events] == ["regress", "improve", "regress"]
    assert events[1].anchor_id == events[0].child_id
    assert events[2].anchor_id == events[1].child_id
    assert root.budget_event_count == 1
    assert root.budget_value == 15.0
    assert method._eligible_node_ids == {events[1].child_id}
    assert events[0].child_id not in method._eligible_node_ids
    assert events[2].child_id not in method._eligible_node_ids


def test_fact_events_do_not_update_anchor_until_rollout_settlement() -> None:
    graph = FactGraph()
    root = graph.add_root(
        code=TEMPLATE,
        idea="root",
        fitness=10.0,
        maximize=True,
        creation_order=1,
    )
    child, event = graph.add_valid_event(
        anchor_id=root.id,
        code="def choose(value: int) -> int:\n    return value + 1\n",
        idea="step",
        fitness=20.0,
        maximize=True,
        stage="search",
        iteration=0,
        budget_order=2,
        rollout_id=0,
        rollout_step=1,
        rollout_start_anchor_id=root.id,
        new_global_best=True,
        global_best_update_reason="strict_fitness",
    )

    assert child.fitness == 20.0
    assert event.credit_value == 20.0
    assert root.budget_event_count == 0
    assert root.budget_value == 10.0

    graph.record_rollout_outcome(root.id, 20.0, 2)
    assert root.budget_event_count == 1
    assert root.budget_value == 15.0


def test_two_stage_prompts_separate_trajectory_reasoning_from_code() -> None:
    method, llm = make_method()
    method.run()
    decision_prompts = [prompt for prompt in llm.prompts if "[Decision]" in prompt]
    code_prompts = [
        prompt for prompt in llm.prompts if "[Approved Next Idea]" in prompt
    ]

    assert len(decision_prompts) == 3
    assert len(code_prompts) == 3
    assert all("[Local Trajectory Evidence]" in prompt for prompt in decision_prompts)
    assert all("Output only:\nIdea:" in prompt for prompt in decision_prompts)
    assert all("[Approved Next Idea]" not in prompt for prompt in decision_prompts)
    assert all("[Local Trajectory Evidence]" not in prompt for prompt in code_prompts)
    assert all("Output only:\nCode:" in prompt for prompt in code_prompts)
    assert all("trajectory decision" in prompt for prompt in code_prompts)


def test_each_rollout_event_carries_stable_rollout_identity() -> None:
    method, _ = make_method()
    method.run()
    root_id = method._graph.root.child_ids[0]
    events = method._graph.events()

    assert {event.rollout_id for event in events} == {0}
    assert [event.rollout_step for event in events] == [1, 2, 3]
    assert {event.rollout_start_anchor_id for event in events} == {root_id}


def test_initial_routes_are_selected_by_short_rollout_representative() -> None:
    method, _ = make_method(
        scores=[100.0, 70.0, 0.0, 120.0, 50.0, 90.0, 95.0, 92.0],
        budget=8,
        routes=2,
        anchors=1,
    )
    method.run()
    roots = [
        method._graph.get_node(node_id) for node_id in method._graph.root.child_ids
    ]
    route_one_events = [
        event
        for event in method._graph.events()
        if event.rollout_start_anchor_id == roots[0].id
    ]

    assert roots[0].budget_value == 110.0
    assert roots[1].budget_value == 82.5
    assert method._eligible_node_ids == {route_one_events[1].child_id}


def test_formal_search_allocates_one_complete_three_step_rollout() -> None:
    method, llm = make_method(budget=7)
    result = method.run()
    formal = [event for event in method._graph.events() if event.stage == "search"]

    assert result.n_iterations == 1
    assert len(formal) == 3
    assert result.n_samples == 7
    assert result.n_evaluations == 7
    assert llm.decision_calls == 6
    assert len({event.rollout_id for event in formal}) == 1


def test_generated_code_is_canonicalized_before_reentering_context() -> None:
    method, llm = make_method()
    method.run()

    assert all(
        "# representation noise" not in node.code for node in method._graph.nodes()
    )
    assert "# representation noise" not in llm.prompts[-2]


def test_artifacts_record_both_llm_stages_and_rollout_fields(tmp_path: Path) -> None:
    method, llm = make_method(run_dir=tmp_path)
    method.run()
    llm_rows = [
        json.loads(line)
        for line in (tmp_path / "artifacts" / "llm_calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    edge_rows = [
        json.loads(line)
        for line in (tmp_path / "artifacts" / "edges.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert [row["prompt"] for row in llm_rows] == llm.prompts
    assert {row["operator"] for row in llm_rows} == {
        "initial_strategy_planning",
        "trajectory_decision",
        "trajectory_rollout_step",
    }
    assert [row["rollout_step"] for row in edge_rows] == [1, 2, 3]
    assert len({row["rollout_id"] for row in edge_rows}) == 1


def test_checkpoint_round_trip_preserves_rollout_state(tmp_path: Path) -> None:
    method, _ = make_method(run_dir=tmp_path)
    method.run()
    method._active_rollout = {
        "rollout_id": 9,
        "start_anchor_id": 0,
        "current_anchor_id": 1,
        "representative_node_id": 2,
        "completed_steps": 2,
        "stage": "search",
        "iteration": 4,
        "initial_strategy": None,
        "event_ids": [3, 4],
        "last_budget_order": 6,
        "budget_value_before": 10.0,
    }
    checkpoint = save_checkpoint(method)
    assert checkpoint is not None

    restored = TraceAADV93(
        llm=TwoStageLLM(),
        evaluation=SequenceEvaluation(),
        max_sample_nums=4,
        initial_route_pool_size=1,
        initial_anchor_count=1,
        context_token_limit=24576,
        resume_from=checkpoint,
    )

    assert restored._active_rollout == method._active_rollout
    assert restored._next_rollout_id == method._next_rollout_id
    assert restored._graph.events() == method._graph.events()
    assert restored.search_configuration()["protocol_id"] == PROTOCOL_ID
    assert (
        restored.search_configuration()["checkpoint_schema_version"]
        == CHECKPOINT_VERSION
    )


def test_window_exposes_ideas_and_results_without_descendant_code() -> None:
    method, _ = make_method()
    method.run()
    root_id = method._graph.root.child_ids[0]
    window = canonical_window(method._graph, root_id)

    assert "Idea implemented: trajectory decision" in window.text
    assert "result fitness" in window.text
    assert "def choose" not in window.text
