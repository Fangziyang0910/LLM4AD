from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

from llm4ad.base import Evaluation, EvaluationOutcome, LLM, TextFunctionProgramConverter
from llm4ad.method.traceaad_v9_5.checkpoint import load_state
from llm4ad.method.traceaad_v9_5.evidence import render_evidence, select_evidence
from llm4ad.method.traceaad_v9_5.forest import SearchForest
from llm4ad.method.traceaad_v9_5.prompt import parse_program_response
from llm4ad.method.traceaad_v9_5.schema import (
    AnchorState,
    AttemptKind,
    AttemptRecord,
    DiffStatistics,
    DirectOutcome,
)
from llm4ad.method.traceaad_v9_5.selection import select_anchor
from llm4ad.method.traceaad_v9_5.traceaad import (
    EvaluatorInfrastructureFailure,
    TraceAADV95,
    evaluation_contract_hash,
)


class QueueLLM(LLM):
    def __init__(self, responses: Iterable[str | BaseException]) -> None:
        super().__init__(do_auto_trim=False)
        self.responses = iter(responses)
        self.calls = 0
        self.request_kwargs: list[dict[str, object]] = []
        self.temperature = 1.0
        self.top_p = None

    def draw_sample(self, prompt: str, **kwargs) -> str:
        del prompt
        self.calls += 1
        self.request_kwargs.append(kwargs)
        item = next(self.responses)
        if isinstance(item, BaseException):
            raise item
        return item

    @staticmethod
    def count_tokens(text: str) -> int:
        return len(text.split())


class ScalarEvaluation(Evaluation):
    def __init__(self) -> None:
        super().__init__(
            "def score() -> float:\n    return 0.0\n",
            task_description="Return a high scalar score.",
            safe_evaluate=False,
        )
        self.executed_inputs: list[str] = []

    def evaluate_program(self, program_str: str, callable_func, **kwargs):
        del kwargs
        self.executed_inputs.append(program_str)
        return callable_func()


def response(value: float, *, idea: str | None = "change value") -> str:
    prefix = "" if idea is None else f"Idea: {idea}\n"
    return (
        prefix
        + "Code:\n```python\n"
        + "def score() -> float:\n"
        + f"    return {value}\n"
        + "```"
    )


def attempt(
    attempt_id: int,
    *,
    anchor_state_id: int,
    child_state_id: int | None = None,
    artifact_id: int | None = None,
    evaluator_input_hash: str | None = None,
    raw_code_hash: str | None = None,
    outcome: DirectOutcome = DirectOutcome.IMPROVE,
    kind: AttemptKind = AttemptKind.NEW_ARTIFACT,
    parent_fitness: float = 1.0,
    child_fitness: float | None = 2.0,
    failure_category: str | None = None,
    failure_feedback: str | None = None,
) -> AttemptRecord:
    return AttemptRecord(
        attempt_id=attempt_id,
        status="finalized",
        anchor_state_id=anchor_state_id,
        child_state_id=child_state_id,
        artifact_id=artifact_id,
        declared_idea=f"idea {attempt_id}",
        raw_code_hash=raw_code_hash,
        evaluator_input_hash=evaluator_input_hash,
        actual_diff=f"diff {attempt_id}",
        diff_statistics=DiffStatistics(1, 1, 2),
        parent_fitness=parent_fitness,
        child_fitness=child_fitness,
        directed_delta=(
            None if child_fitness is None else child_fitness - parent_fitness
        ),
        direct_outcome=outcome,
        attempt_kind=kind,
        failure_category=failure_category,
        failure_feedback=failure_feedback,
        evaluator_called=kind is AttemptKind.NEW_ARTIFACT,
        candidate_order=attempt_id + 1,
        creation_time="2026-01-01T00:00:00+00:00",
        stage="search",
        iteration=attempt_id,
    )


def make_forest() -> tuple[SearchForest, AnchorState, AnchorState]:
    forest = SearchForest("contract", maximize=True)
    root_artifact = forest.add_artifact(
        evaluator_input_code="def f():\n    return 1\n",
        fitness=1.0,
        discovery_order=1,
    )
    other_artifact = forest.add_artifact(
        evaluator_input_code="def f():\n    return 2\n",
        fitness=2.0,
        discovery_order=2,
    )
    first = forest.add_root_state(
        artifact_id=root_artifact.artifact_id, creation_order=1
    )
    second = forest.add_root_state(
        artifact_id=other_artifact.artifact_id, creation_order=2
    )
    return forest, first, second


def test_optional_idea_does_not_control_code_validity() -> None:
    template = TextFunctionProgramConverter.text_to_program(
        "def score() -> float:\n    return 0.0\n"
    )
    assert template is not None

    parsed = parse_program_response(response(3.0, idea=None), template, "score")

    assert parsed.declared_idea is None
    assert "return 3.0" in str(parsed.program)


def test_evaluator_contract_identity_includes_dataset_contents() -> None:
    first = ScalarEvaluation()
    second = ScalarEvaluation()
    first._datasets = {"instance": [1.0]}  # type: ignore[attr-defined]
    second._datasets = {"instance": [2.0]}  # type: ignore[attr-defined]

    assert evaluation_contract_hash(first) != evaluation_contract_hash(second)


def test_exact_state_direct_evidence_is_deduplicated_before_outcome_coverage() -> None:
    forest, first, second = make_forest()
    forest.add_attempt(
        attempt(
            0,
            anchor_state_id=first.state_id,
            evaluator_input_hash="same-improvement",
        )
    )
    forest.add_attempt(
        attempt(
            1,
            anchor_state_id=first.state_id,
            evaluator_input_hash="same-improvement",
        )
    )
    forest.add_attempt(
        attempt(
            2,
            anchor_state_id=first.state_id,
            evaluator_input_hash="regression",
            outcome=DirectOutcome.REGRESS,
            child_fitness=0.0,
        )
    )
    forest.add_attempt(
        attempt(
            3,
            anchor_state_id=first.state_id,
            raw_code_hash="invalid-code",
            outcome=DirectOutcome.INVALID,
            kind=AttemptKind.INVALID,
            child_fitness=None,
            failure_category="runtime_error",
            failure_feedback="division by zero",
        )
    )
    forest.add_attempt(
        attempt(
            4,
            anchor_state_id=second.state_id,
            evaluator_input_hash="other-state",
        )
    )

    selection = select_evidence(forest, first.state_id, max_items=3)
    rendered = render_evidence(forest, selection, diff_excerpt_chars=100)

    assert selection.direct_attempt_ids == (1, 2, 3)
    assert selection.folded_attempt_ids[1] == (0,)
    assert 4 not in selection.direct_pool_ids
    assert "division by zero" in rendered.text


def test_formation_is_recent_and_only_fills_space_left_by_direct_attempts() -> None:
    forest, root, _ = make_forest()
    first_child_artifact = forest.add_artifact(
        evaluator_input_code="def f():\n    return 3\n",
        fitness=3.0,
        discovery_order=3,
    )
    incoming = attempt(
        0,
        anchor_state_id=root.state_id,
        child_state_id=2,
        artifact_id=first_child_artifact.artifact_id,
        evaluator_input_hash=first_child_artifact.evaluator_input_hash,
    )
    forest.add_attempt(incoming)
    child = forest.add_child_state(
        parent_state_id=root.state_id,
        artifact_id=first_child_artifact.artifact_id,
        attempt_id=incoming.attempt_id,
        creation_order=3,
    )
    forest.add_attempt(
        attempt(
            1,
            anchor_state_id=child.state_id,
            evaluator_input_hash="direct",
        )
    )

    selection = select_evidence(forest, child.state_id, max_items=2)

    assert selection.formation_attempt_ids == (0,)
    assert selection.direct_attempt_ids == (1,)


def test_state_identity_allows_branch_convergence_but_rejects_route_cycles() -> None:
    forest, first, second = make_forest()
    shared = forest.add_artifact(
        evaluator_input_code="def f():\n    return 4\n",
        fitness=4.0,
        discovery_order=3,
    )
    left = forest.add_child_state(
        parent_state_id=first.state_id,
        artifact_id=shared.artifact_id,
        attempt_id=0,
        creation_order=3,
    )
    right = forest.add_child_state(
        parent_state_id=second.state_id,
        artifact_id=shared.artifact_id,
        attempt_id=1,
        creation_order=4,
    )

    assert left.artifact_id == right.artifact_id
    with pytest.raises(ValueError, match="ancestral artifact"):
        forest.add_child_state(
            parent_state_id=left.state_id,
            artifact_id=first.artifact_id,
            attempt_id=2,
            creation_order=5,
        )


def test_allocation_uses_only_quality_count_and_fixed_scale() -> None:
    forest, first, second = make_forest()
    first.generation_count_n = 0
    second.generation_count_n = 8

    selected, scores = select_anchor(forest, optimism_scale=2.0)

    assert selected == first.state_id
    assert {item.state_id for item in scores} == {first.state_id, second.state_id}
    assert next(item for item in scores if item.state_id == first.state_id).score == 3.0


def test_full_lifecycle_uses_exact_evaluator_inputs_cache_and_bootstrap_scale() -> None:
    evaluation = ScalarEvaluation()
    method = TraceAADV95(
        QueueLLM(
            [
                response(1.0),
                response(2.0, idea=None),
                response(1.0),
                response(3.0),
                response(4.0),
            ]
        ),
        evaluation,
        candidate_search_budget=5,
        initial_root_count=2,
        code_max_tokens=100,
        context_token_limit=5000,
        generation_seed=10,
    )

    result = method.run()

    assert result.initialization_complete
    assert result.n_candidates == 5
    assert result.n_evaluations == 4
    assert result.n_artifacts == 4
    assert result.n_states == 4
    assert result.optimism_scale == 1.0
    assert result.best_artifact is not None
    assert result.best_artifact.fitness == 4.0
    assert [item.attempt_kind for item in method._forest.attempts()] == [
        AttemptKind.ROOT_NEW,
        AttemptKind.ROOT_NEW,
        AttemptKind.NO_OP,
        AttemptKind.NEW_ARTIFACT,
        AttemptKind.NEW_ARTIFACT,
    ]
    evaluated_codes = {
        artifact.evaluator_input_code for artifact in method._forest.artifacts()
    }
    assert set(evaluation.executed_inputs) == evaluated_codes


def test_root_duplicate_consumes_candidate_budget_without_evaluator_or_state() -> None:
    method = TraceAADV95(
        QueueLLM([response(1.0), response(1.0), response(2.0)]),
        ScalarEvaluation(),
        candidate_search_budget=3,
        initial_root_count=2,
        code_max_tokens=100,
        context_token_limit=5000,
    )

    result = method.run()

    assert not result.initialization_complete
    assert result.n_candidates == 3
    assert result.n_evaluations == 2
    assert result.n_root_states == 2
    assert [item.attempt_kind for item in method._forest.attempts()] == [
        AttemptKind.ROOT_NEW,
        AttemptKind.ROOT_DUPLICATE,
        AttemptKind.ROOT_NEW,
    ]


def test_cached_branch_convergence_reuses_fitness_and_route_duplicates_create_no_state() -> (
    None
):
    method = TraceAADV95(
        QueueLLM(
            [
                response(1.0),
                response(2.0),
                response(3.0),
                response(3.0),
                response(1.0),
                response(3.0),
            ]
        ),
        ScalarEvaluation(),
        candidate_search_budget=6,
        initial_root_count=2,
        code_max_tokens=100,
        context_token_limit=5000,
    )
    method._initialize()
    root_a = method._forest.get_state(method._forest.root_state_ids[0])
    state_x_from_a = next(
        state
        for state in method._forest.states()
        if state.parent_state_id == root_a.state_id
    )
    state_count = len(method._forest.states())

    method._request_candidate(
        method._build_anchor_prompt(state_x_from_a.state_id),
        anchor_state_id=state_x_from_a.state_id,
        stage="search",
        iteration=0,
    )
    method._request_candidate(
        method._build_anchor_prompt(root_a.state_id),
        anchor_state_id=root_a.state_id,
        stage="search",
        iteration=1,
    )

    assert method._evaluation_count == 3
    assert len(method._forest.states()) == state_count
    assert [item.attempt_kind for item in method._forest.attempts()] == [
        AttemptKind.ROOT_NEW,
        AttemptKind.ROOT_NEW,
        AttemptKind.NEW_ARTIFACT,
        AttemptKind.CACHED_ARTIFACT,
        AttemptKind.ANCESTRAL_RETURN,
        AttemptKind.REPEATED_DUPLICATE,
    ]


def test_transport_retry_does_not_consume_candidate_or_change_generation_seed() -> None:
    llm = QueueLLM([ConnectionError("temporary"), response(1.0), response(1.0)])
    method = TraceAADV95(
        llm,
        ScalarEvaluation(),
        candidate_search_budget=2,
        initial_root_count=1,
        code_max_tokens=100,
        context_token_limit=5000,
        transport_retry_limit=1,
        generation_seed=7,
    )

    result = method.run()

    assert result.n_candidates == 2
    assert result.n_llm_requests == 3
    assert result.n_evaluations == 1
    assert llm.request_kwargs[0]["seed"] == llm.request_kwargs[1]["seed"] == 8
    assert llm.request_kwargs[2]["seed"] == 9


def test_completed_response_without_code_is_finalized_invalid_and_counts_n() -> None:
    method = TraceAADV95(
        QueueLLM([response(1.0), "Idea: missing mandatory code"]),
        ScalarEvaluation(),
        candidate_search_budget=2,
        initial_root_count=1,
        code_max_tokens=100,
        context_token_limit=5000,
    )

    result = method.run()
    root = method._forest.get_state(method._forest.root_state_ids[0])
    invalid = method._forest.attempts()[-1]

    assert result.initialization_complete
    assert result.n_candidates == 2
    assert result.n_evaluations == 1
    assert root.generation_count_n == 1
    assert invalid.attempt_kind is AttemptKind.INVALID
    assert invalid.direct_outcome is DirectOutcome.INVALID
    assert invalid.failure_category == "parse"


def test_evaluator_preparation_failure_stops_run_without_finalizing_invalid() -> None:
    method = TraceAADV95(
        QueueLLM([response(1.0)]),
        ScalarEvaluation(),
        candidate_search_budget=1,
        initial_root_count=1,
        code_max_tokens=100,
        context_token_limit=5000,
    )
    method._evaluator.evaluate_program_record_time_with_details = (  # type: ignore[method-assign]
        lambda _program: (
            EvaluationOutcome(
                result=None,
                failure_kind="prepare_error",
                error_type="RuntimeError",
                error="worker could not start",
            ),
            0.1,
        )
    )

    with pytest.raises(EvaluatorInfrastructureFailure, match="worker could not start"):
        method.run()

    assert method._candidate_count == 1
    assert method._evaluation_count == 1
    assert method._pending_attempt is not None
    assert method._pending_attempt.processing_stage.value == "parsed"
    assert method._forest.attempts() == ()


def test_minimize_mode_uses_directed_quality_for_search_and_final_best() -> None:
    method = TraceAADV95(
        QueueLLM([response(3.0), response(1.0), response(2.0)]),
        ScalarEvaluation(),
        candidate_search_budget=3,
        initial_root_count=1,
        maximize=False,
        code_max_tokens=100,
        context_token_limit=5000,
    )

    result = method.run()

    assert result.best_artifact is not None
    assert result.best_artifact.fitness == 1.0
    assert method._forest.attempts()[-1].direct_outcome is DirectOutcome.REGRESS


def test_pending_anchor_response_resume_does_not_recall_model_or_recount_n(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    first_llm = QueueLLM([response(1.0), response(2.0), response(3.0)])
    method = TraceAADV95(
        first_llm,
        ScalarEvaluation(),
        candidate_search_budget=3,
        initial_root_count=1,
        code_max_tokens=100,
        context_token_limit=5000,
        checkpoint_dir=checkpoint_dir,
    )
    method._initialize()
    selected_state_id, _ = select_anchor(method._forest, method._optimism_scale or 0.0)
    prompt = method._build_anchor_prompt(selected_state_id)

    class Interrupted(Exception):
        pass

    def interrupt_processing():
        raise Interrupted

    method._process_pending_attempt = interrupt_processing  # type: ignore[method-assign]
    with pytest.raises(Interrupted):
        method._request_candidate(
            prompt,
            anchor_state_id=selected_state_id,
            stage="search",
            iteration=0,
        )
    count_after_response = method._forest.get_state(
        selected_state_id
    ).generation_count_n
    assert method._pending_attempt is not None
    assert method._candidate_count == 3

    resumed_llm = QueueLLM([])
    resumed = TraceAADV95(
        resumed_llm,
        ScalarEvaluation(),
        candidate_search_budget=3,
        initial_root_count=1,
        code_max_tokens=100,
        context_token_limit=5000,
        checkpoint_dir=checkpoint_dir,
        resume_from=checkpoint_dir / "latest.json",
    )
    result = resumed.run()

    assert result.n_candidates == 3
    assert result.n_iterations == 1
    assert resumed_llm.calls == 0
    assert resumed._forest.get_state(selected_state_id).generation_count_n == (
        count_after_response
    )
    assert resumed._pending_attempt is None


def test_evaluated_pending_resume_does_not_recall_model_or_evaluator(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    evaluation = ScalarEvaluation()
    method = TraceAADV95(
        QueueLLM([response(1.0), response(2.0), response(3.0)]),
        evaluation,
        candidate_search_budget=3,
        initial_root_count=1,
        code_max_tokens=100,
        context_token_limit=5000,
        checkpoint_dir=checkpoint_dir,
    )
    method._initialize()
    selected_state_id, _ = select_anchor(method._forest, method._optimism_scale or 0.0)

    class Interrupted(Exception):
        pass

    method._finalize_pending_attempt = (  # type: ignore[method-assign]
        lambda **_kwargs: (_ for _ in ()).throw(Interrupted())
    )
    with pytest.raises(Interrupted):
        method._request_candidate(
            method._build_anchor_prompt(selected_state_id),
            anchor_state_id=selected_state_id,
            stage="search",
            iteration=0,
        )
    assert method._pending_attempt is not None
    assert method._pending_attempt.evaluated_fitness == 3.0

    resumed_evaluation = ScalarEvaluation()
    resumed_llm = QueueLLM([])
    resumed = TraceAADV95(
        resumed_llm,
        resumed_evaluation,
        candidate_search_budget=3,
        initial_root_count=1,
        code_max_tokens=100,
        context_token_limit=5000,
        checkpoint_dir=checkpoint_dir,
        resume_from=checkpoint_dir / "latest.json",
    )
    result = resumed.run()

    assert result.n_candidates == 3
    assert resumed_llm.calls == 0
    assert resumed_evaluation.executed_inputs == []
    assert resumed._pending_attempt is None


def test_checkpoint_loader_rejects_another_schema(tmp_path: Path) -> None:
    method = TraceAADV95(
        QueueLLM([]),
        ScalarEvaluation(),
        candidate_search_budget=2,
        initial_root_count=1,
        code_max_tokens=100,
        context_token_limit=5000,
    )

    with pytest.raises(ValueError, match="checkpoint version"):
        load_state(method, {"version": 99})
