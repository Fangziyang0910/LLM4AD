from __future__ import annotations

import random

import pytest

from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad import TraceAAD, ValueWeights
from llm4ad.method.traceaad.traceaad import _prompt_section_chars
from llm4ad.method.traceaad.derivation_graph import DerivationGraph
from llm4ad.method.traceaad.feedback import RankingModel
from llm4ad.method.traceaad.operators import EndpointRefineOp, NoveltyJumpOp
from llm4ad.method.traceaad.schema import OperatorName
from llm4ad.method.traceaad.trajectory_memory import TrajectoryMemory


TEMPLATE = """
def heuristic(x):
    return 0
"""


def _program(value: float) -> str:
    return f"""Idea: constant {value}
Code:
```python
def heuristic(x):
    return {value}
```
"""


class ConstantEvaluation(Evaluation):
    def __init__(self) -> None:
        super().__init__(
            template_program=TEMPLATE,
            task_description="Return a larger constant.",
            safe_evaluate=False,
        )
        self.calls = 0

    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs):
        self.calls += 1
        return None if callable_func is None else float(callable_func(0))


class ScriptedTraceAADLLM(LLM):
    def __init__(
        self,
        *,
        fail_first_init_parse: bool = False,
        fail_first_refine_parse: bool = False,
        program_values: list[float] | None = None,
    ) -> None:
        super().__init__(do_auto_trim=False)
        self.fail_first_init_parse = fail_first_init_parse
        self.fail_first_refine_parse = fail_first_refine_parse
        self.program_values = list(program_values or [])
        self.init_draws = 0
        self.program_index = 0
        self.refine_draws = 0

    def draw_sample(self, prompt: str, *args, **kwargs) -> str:
        if "Generate a complete implementation" in prompt:
            self.init_draws += 1
            if self.fail_first_init_parse and self.init_draws == 1:
                return "This response contains no Python program."
            return _program(self._next_program_value())
        if "next-step modifications" in prompt:
            return "1. Increase the returned constant.\n2. Increase it again."
        if "Requested Modification" in prompt:
            self.refine_draws += 1
            if self.fail_first_refine_parse and self.refine_draws == 1:
                return "This response contains no Python program."
            return _program(self._next_program_value())
        raise AssertionError(f"unexpected prompt: {prompt[:120]}")

    def _next_program_value(self) -> float:
        self.program_index += 1
        if self.program_values:
            return self.program_values.pop(0)
        return self.program_index


class RecordingEndpointRefineOp(EndpointRefineOp):
    seen_iterations: list[int] = []

    def build_constraint(self, ctx, base_node_id):
        self.seen_iterations.append(ctx.iteration)
        return super().build_constraint(ctx, base_node_id)


class AlwaysNoveltyOp(NoveltyJumpOp):
    def trigger(self, ctx) -> bool:
        return True


def _method(*, llm: LLM, evaluation: Evaluation, max_samples: int, actions: int, **kwargs) -> TraceAAD:
    novelty_threshold = kwargs.pop("novelty_threshold", 1.1)
    max_per_island = kwargs.pop("max_per_island", 20)
    max_active_trajectories = kwargs.pop("max_active_trajectories", None)
    operators = kwargs.pop("operators", (EndpointRefineOp,))
    return TraceAAD(
        llm=llm,
        evaluation=evaluation,
        max_sample_nums=max_samples,
        n_init=1,
        actions_per_iteration=actions,
        operators=operators,
        n_islands=1,
        max_per_island=max_per_island,
        max_active_trajectories=max_active_trajectories,
        num_evaluators=1,
        novelty_threshold=novelty_threshold,
        **kwargs,
    )


def test_removed_pattern_parameters_are_rejected() -> None:
    with pytest.raises(TypeError):
        TraceAAD(
            llm=ScriptedTraceAADLLM(),
            evaluation=ConstantEvaluation(),
            k_distill=20,
        )


def test_experience_section_size_can_be_logged_without_tokenizer() -> None:
    prompt = "before\n[Past Action Evidence]\nabc\ndef\n[Contrast Feedback]\nafter"

    assert _prompt_section_chars(
        prompt, "[Past Action Evidence]", "[Contrast Feedback]"
    ) == len("abc\ndef")


def test_run_replaces_parse_failures_until_evaluation_budget_is_spent():
    evaluation = ConstantEvaluation()
    method = _method(
        llm=ScriptedTraceAADLLM(fail_first_refine_parse=True),
        evaluation=evaluation,
        max_samples=3,
        actions=1,
    )

    result = method.run()

    assert evaluation.calls == 3
    assert result.n_samples == 3
    assert result.best_node is not None
    assert result.best_node.fitness == 3.0


def test_initialization_retries_parse_failures_before_entering_search():
    evaluation = ConstantEvaluation()
    method = _method(
        llm=ScriptedTraceAADLLM(fail_first_init_parse=True),
        evaluation=evaluation,
        max_samples=2,
        actions=1,
    )

    result = method.run()

    assert evaluation.calls == 2
    assert result.n_samples == 2
    assert result.best_node is not None


def test_parse_failure_does_not_advance_search_phase_iteration():
    RecordingEndpointRefineOp.seen_iterations = []
    method = _method(
        llm=ScriptedTraceAADLLM(fail_first_refine_parse=True),
        evaluation=ConstantEvaluation(),
        max_samples=2,
        actions=1,
        operators=(RecordingEndpointRefineOp,),
    )

    method.run()

    assert RecordingEndpointRefineOp.seen_iterations[:2] == [0, 0]


def test_portfolio_learns_once_from_the_best_of_each_action_batch():
    method = _method(
        llm=ScriptedTraceAADLLM(),
        evaluation=ConstantEvaluation(),
        max_samples=3,
        actions=2,
    )

    method.run()

    endpoint_stats = method.operator_portfolio_snapshot()["endpoint_refine"]
    assert endpoint_stats["n_calls"] == 1
    assert endpoint_stats["global_best_rate"] == 1.0


def test_operator_selection_event_matches_final_execution_context(monkeypatch):
    events = []

    def record_event(target, event=None, **payload):
        if event == "operator_selection":
            events.append(payload)

    monkeypatch.setattr(
        "llm4ad.method.traceaad.traceaad.log_event",
        record_event,
    )
    method = _method(
        llm=ScriptedTraceAADLLM(),
        evaluation=ConstantEvaluation(),
        max_samples=2,
        actions=1,
        operators=(EndpointRefineOp,),
    )

    method.run()

    assert len(events) == 1
    event = events[0]
    assert event["selected_operator"] == OperatorName.ENDPOINT
    assert event["target_trajectory_id"] == event["selected_trajectory_id"]
    target = next(
        trajectory
        for trajectory in method.active_trajectories()
        if trajectory.id == event["target_trajectory_id"]
    )
    assert event["base_node_id"] == target.endpoint_id
    assert sum(event["probabilities"].values()) == pytest.approx(1.0)
    assert event["temperature"] > 0.0
    assert 0.0 <= event["epsilon"] <= 1.0
    assert event["eligible"] == [OperatorName.ENDPOINT]


def test_global_record_bypasses_similarity_gate_and_remains_active():
    method = _method(
        llm=ScriptedTraceAADLLM(),
        evaluation=ConstantEvaluation(),
        max_samples=2,
        actions=1,
        novelty_threshold=0.0,
    )

    result = method.run()

    assert result.best_node is not None
    assert result.best_node.fitness == 2.0
    assert any(t.endpoint_id == result.best_node.id for t in method.active_trajectories())


def test_only_the_live_global_record_bypasses_similarity_gate_within_a_batch():
    method = _method(
        llm=ScriptedTraceAADLLM(program_values=[1, 3, 2]),
        evaluation=ConstantEvaluation(),
        max_samples=3,
        actions=2,
        novelty_threshold=0.0,
    )

    result = method.run()

    assert result.best_node is not None
    assert result.best_node.fitness == 3.0
    active_scores = {
        method._graph.get_node(trajectory.endpoint_id).fitness
        for trajectory in method.active_trajectories()
    }
    assert 3.0 in active_scores
    assert 2.0 not in active_scores


def test_traceaad_seed_does_not_mutate_the_process_global_random_stream():
    random.seed(12345)
    expected = random.random()
    random.seed(12345)
    method = _method(
        llm=ScriptedTraceAADLLM(),
        evaluation=ConstantEvaluation(),
        max_samples=0,
        actions=1,
        random_seed=999,
    )

    method.run()

    assert random.random() == expected


def test_near_record_fresh_starts_receive_credit_without_bypassing_the_gate():
    method = _method(
        llm=ScriptedTraceAADLLM(program_values=[100.0, 101.0, 100.95]),
        evaluation=ConstantEvaluation(),
        max_samples=3,
        actions=2,
        operators=(AlwaysNoveltyOp,),
    )

    method.run()

    novelty_stats = method.operator_portfolio_snapshot()[OperatorName.NOVELTY]
    assert novelty_stats["near_record_rate"] == 1.0
    assert len(method._graph.nodes()) == 3  # 1 init + 2 novelty fresh starts
    assert len(method._graph.edges()) == 0


def test_survival_cap_keeps_the_global_best_trajectory_active():
    method = _method(
        llm=ScriptedTraceAADLLM(),
        evaluation=ConstantEvaluation(),
        max_samples=3,
        actions=1,
        max_per_island=1,
        max_active_trajectories=1,
    )

    result = method.run()

    assert len(method.active_trajectories()) == 1
    assert method.active_trajectories()[0].endpoint_id == result.best_node.id


def test_duplicate_elite_paths_do_not_expand_the_survival_cap():
    method = _method(
        llm=ScriptedTraceAADLLM(),
        evaluation=ConstantEvaluation(),
        max_samples=1,
        actions=1,
        max_per_island=1,
        max_active_trajectories=1,
    )
    result = method.run()
    elite = method.active_trajectories()[0]
    method._memory.create_initial(node_id=elite.endpoint_id, island_id=elite.island_id)

    method._survive()

    assert len(method.active_trajectories()) == 1
    assert method.active_trajectories()[0].endpoint_id == result.best_node.id


def test_initialization_spreads_across_islands_without_preset_mechanism_hints():
    class RecordingLLM(ScriptedTraceAADLLM):
        def __init__(self) -> None:
            super().__init__()
            self.prompts: list[str] = []

        def draw_sample(self, prompt: str, *args, **kwargs) -> str:
            self.prompts.append(prompt)
            return super().draw_sample(prompt, *args, **kwargs)

    llm = RecordingLLM()
    method = TraceAAD(
        llm=llm,
        evaluation=ConstantEvaluation(),
        max_sample_nums=4,
        n_init=4,
        actions_per_iteration=1,
        operators=(EndpointRefineOp,),
        n_islands=4,
        max_per_island=10,
        num_evaluators=1,
    )

    method.run()

    islands = {trajectory.island_id for trajectory in method.active_trajectories()}
    assert islands == {0, 1, 2, 3}
    joined = "\n".join(llm.prompts)
    assert "nearest neighbor" not in joined.lower()
    assert "local density" not in joined.lower()
    assert "row-wise" not in joined.lower()
    assert "sparsified candidate" not in joined.lower()
    assert "already generated ideas" in joined.lower() or "simple, complete" in joined.lower()


def test_contrast_uses_learned_pairwise_ranking_instead_of_raw_fitness_only():
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    raw_best = graph.add_node(code="a", idea="raw", fitness=10.0)
    ranked_best = graph.add_node(code="b", idea="ranked", fitness=9.0)
    memory.create_initial(node_id=raw_best.id)
    memory.create_initial(node_id=ranked_best.id)
    ranking = RankingModel()
    for _ in range(20):
        ranking.update_pair(ranked_best.id, raw_best.id, 1.0)

    contrast = ranking.contrast(graph=graph, memory=memory, maximize=True)

    assert contrast is not None
    assert contrast["best"]["node_id"] == ranked_best.id


def test_contrast_does_not_compare_elo_scores_across_disconnected_components():
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    weak_parent = graph.add_node(code="wp", idea="wp", fitness=0.0)
    weak_winner = graph.add_node(code="ww", idea="ww", fitness=1.0)
    strong_parent = graph.add_node(code="sp", idea="sp", fitness=101.0)
    strong_loser = graph.add_node(code="sl", idea="sl", fitness=100.0)
    memory.create_initial(node_id=weak_winner.id)
    memory.create_initial(node_id=strong_loser.id)
    ranking = RankingModel()
    ranking.update_pair(weak_winner.id, weak_parent.id, 1.0)
    ranking.update_pair(strong_loser.id, strong_parent.id, 0.0)

    contrast = ranking.contrast(graph=graph, memory=memory, maximize=True)

    assert contrast is not None
    assert contrast["best"]["node_id"] == strong_loser.id
    assert contrast["worst"]["node_id"] == weak_winner.id
