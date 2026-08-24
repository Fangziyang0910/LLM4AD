"""Mechanism and recovery tests for TraceAAD V9.17."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad_v9_17 import RunArtifacts, TraceAADV917
from llm4ad.method.traceaad_v9_17.checkpoint import save_checkpoint
from llm4ad.method.traceaad_v9_17.schema import (
    BlockKind,
    GenerationState,
    Hypothesis,
    HypothesisStatus,
    Pending,
    Phase,
)
from llm4ad.method.traceaad_v9_17.selection import (
    rank_hypotheses,
    refine_score,
    select_refine_parent,
)
from llm4ad.method.traceaad_v9_17.tree import Tree, VIRTUAL_ROOT_ID

TEMPLATE = """def choose(value: int) -> int:
    return value
"""


def _program(value: float) -> str:
    return (
        f"Idea: return {value}\n"
        "Code:\n```python\n"
        "def choose(value: int) -> int:\n"
        f"    return {value}\n"
        "```"
    )


class ScriptedLLM(LLM):
    def __init__(self, responses: list[str]) -> None:
        super().__init__()
        self.responses = responses
        self.calls = 0
        self.prompts: list[str] = []

    def draw_sample(self, prompt: str, *args, **kwargs) -> str:
        self.prompts.append(prompt)
        response = self.responses[self.calls]
        self.calls += 1
        return response


class ConstantLLM(LLM):
    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.response = _program(value)
        self.calls = 0

    def draw_sample(self, prompt: str, *args, **kwargs) -> str:
        self.calls += 1
        return self.response


class InterruptedRepairLLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def draw_sample(self, prompt: str, *args, **kwargs) -> str:
        self.calls += 1
        if self.calls == 1:
            return _program(1)
        if self.calls == 2:
            return "unusable response"
        raise RuntimeError("simulated interruption before repair response")


class ValueEvaluation(Evaluation):
    def __init__(self) -> None:
        super().__init__(
            template_program=TEMPLATE,
            task_description="Improve choose.",
            safe_evaluate=False,
            timeout_seconds=5,
        )

    def evaluate_program(self, program_str: str, callable_func, **kwargs) -> float | None:
        return None if callable_func is None else float(callable_func(10))


def test_refine_parent_score_is_local_and_uses_primary_refine_count() -> None:
    tree = Tree(maximize=True)
    root = tree.add_algorithm(
        code="a",
        fitness=10.0,
        parent_id=VIRTUAL_ROOT_ID,
        hypothesis_id=1,
        idea=None,
        diff="",
        added=0,
        removed=0,
        result="initial",
        created_by=None,
    )
    child = tree.add_algorithm(
        code="b",
        fitness=9.5,
        parent_id=root.id,
        hypothesis_id=1,
        idea=None,
        diff="",
        added=0,
        removed=0,
        result="worse",
        created_by="refine",
    )
    root.refine_count = 9
    assert refine_score(tree, root, 2.0) == pytest.approx(10.0 + 2.0 / math.sqrt(10))
    assert select_refine_parent(tree, 1, scale=2.0) is child


def test_hypothesis_competition_uses_quality_then_creation_only() -> None:
    hypotheses = [
        Hypothesis(3, 3, None, HypothesisStatus.MATURING, 3, 7.0, primary_slots=99),
        Hypothesis(1, 1, None, HypothesisStatus.ACTIVE, 1, 7.0, primary_slots=1),
        Hypothesis(2, 2, None, HypothesisStatus.ACTIVE, 2, 8.0, primary_slots=1),
    ]
    assert [item.id for item in rank_hypotheses(hypotheses)] == [2, 1, 3]


def test_valid_discovery_is_matured_then_displaces_by_frontier_quality(
    tmp_path: Path,
) -> None:
    values = [
        10,
        20,
        *([10] * 3),
        *([20] * 3),
        *([20] * 3),
        *([10] * 3),
        30,
        *([30] * 3),
    ]
    method = TraceAADV917(
        llm=ScriptedLLM([_program(value) for value in values]),
        evaluation=ValueEvaluation(),
        artifacts=RunArtifacts(tmp_path, console_output=False),
        budget=len(values),
        n_roots=2,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    method.run()

    assert method._n_eval == 18
    assert method._explore_slots == 1
    assert method._valid_discoveries == 1
    assert method._active_ids == [3, 2]
    assert method._reserve_ids == [1]
    assert method._hypotheses[3].best_quality == 30.0
    assert method._block_counts[BlockKind.MATURATION.value] == 1
    events = (tmp_path / "mechanism_events.jsonl").read_text(encoding="utf-8")
    assert '"event": "competition"' in events
    assert '"candidate_hypothesis_id": 3' in events


def test_positive_maturation_gain_cannot_bypass_competition_quality() -> None:
    values = [10, *([10] * 3), *([10] * 3), 5, *([9] * 3)]
    method = TraceAADV917(
        llm=ScriptedLLM([_program(value) for value in values]),
        evaluation=ValueEvaluation(),
        budget=len(values),
        n_roots=1,
    )
    method.run()

    assert method._hypotheses[2].last_block_gain == 4.0
    assert method._active_ids == [1]
    assert method._reserve_ids == [2]


def test_positive_block_gain_gets_one_more_sweep_without_discovery() -> None:
    values = [1, *([1] * 3), *([2] * 3), *([2] * 3)]
    method = TraceAADV917(
        llm=ScriptedLLM([_program(value) for value in values]),
        evaluation=ValueEvaluation(),
        budget=len(values),
        n_roots=1,
    )
    method.run()

    assert method._block_counts[BlockKind.DEVELOPMENT.value] == 2
    assert method._explore_slots == 0
    assert method._hypotheses[1].best_quality == 2.0
    assert method._hypotheses[1].last_block_gain == 0.0


def test_fixed_cycle_discovers_after_one_full_sweep_even_when_gain_is_positive() -> None:
    values = [1, *([1] * 3), *([2] * 3), 3, *([3] * 3)]
    method = TraceAADV917(
        llm=ScriptedLLM([_program(value) for value in values]),
        evaluation=ValueEvaluation(),
        budget=len(values),
        n_roots=1,
        adaptive_sweeps=False,
    )
    method.run()

    assert method._block_counts[BlockKind.DEVELOPMENT.value] == 1
    assert method._explore_slots == 1
    assert method._valid_discoveries == 1
    assert method._sweep == 1


def test_initial_maturation_freezes_median_absolute_refine_scale() -> None:
    values = [10, 20, 11, 9, 10, 22, 20, 21, 21]
    method = TraceAADV917(
        llm=ScriptedLLM([_program(value) for value in values]),
        evaluation=ValueEvaluation(),
        budget=len(values),
        n_roots=2,
    )
    method.run()

    assert method._s_r_frozen is True
    assert method._bootstrap_deltas == pytest.approx([1, 2, 1, 2, 2, 1])
    assert method._s_r == pytest.approx(1.5)


def test_generation_prompt_contains_actual_idea_change_result_and_fitness() -> None:
    llm = ScriptedLLM([_program(1), _program(2), _program(3)])
    method = TraceAADV917(
        llm=llm,
        evaluation=ValueEvaluation(),
        budget=3,
        n_roots=1,
    )
    method.run()

    prompt = llm.prompts[2]
    assert "[Recent Algorithm Improvement History]" in prompt
    assert "Idea: return 2" in prompt
    assert "Change: +1/-1 lines" in prompt
    assert "Result: improved" in prompt
    assert "Fitness: 1 -> 2" in prompt


def test_repair_calls_do_not_consume_primary_slots_or_refine_attempts(
    tmp_path: Path,
) -> None:
    responses = [
        _program(1),
        "unusable response",
        _program(1),
        _program(1),
        _program(1),
    ]
    method = TraceAADV917(
        llm=ScriptedLLM(responses),
        evaluation=ValueEvaluation(),
        artifacts=RunArtifacts(tmp_path, console_output=False),
        budget=4,
        n_roots=1,
    )
    method.run()

    assert method._n_eval == 4
    assert method._n_calls == 5
    assert method._n_llm_calls == 5
    assert method._repair_llm_calls == 1
    assert method._refine_slots == 3
    assert sum(item.refine_count for item in method._tree.valid_algorithms()) == 3
    with (tmp_path / "evaluations.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["attempt_kind"] for row in rows] == [
        "initial",
        "initial",
        "repair",
        "initial",
        "initial",
    ]
    assert [row["eval_count"] for row in rows] == ["1", "2", "2", "3", "4"]


def test_pending_candidate_resume_does_not_repeat_llm_or_primary_slot(
    tmp_path: Path,
) -> None:
    first = TraceAADV917(
        llm=ConstantLLM(),
        evaluation=ValueEvaluation(),
        budget=4,
        n_roots=1,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    first._generation = GenerationState(
        prompt="root",
        parent_id=VIRTUAL_ROOT_ID,
        intent=None,
        mode="root",
        hypothesis_id=None,
    )
    first._pending = Pending(
        response=_program(1), attempt=1, attempt_kind="initial"
    )
    first._n_llm_calls = 1
    checkpoint = save_checkpoint(first)
    assert checkpoint is not None

    resumed_llm = ConstantLLM()
    resumed = TraceAADV917(
        llm=resumed_llm,
        evaluation=ValueEvaluation(),
        budget=4,
        n_roots=1,
        checkpoint_dir=tmp_path / "checkpoints",
        resume_from=checkpoint,
    )
    resumed.run()

    assert resumed._n_eval == 4
    assert resumed._n_calls == 4
    assert resumed._n_llm_calls == 4
    assert resumed_llm.calls == 3
    assert resumed._phase is Phase.INITIAL_MATURATION
    assert resumed._active_block is None


def test_fixed_cycle_can_only_fork_adaptive_checkpoint_at_initial_boundary(
    tmp_path: Path,
) -> None:
    adaptive = TraceAADV917(
        llm=ConstantLLM(),
        evaluation=ValueEvaluation(),
        budget=11,
        n_roots=1,
        checkpoint_dir=tmp_path / "adaptive",
    )
    while not (
        adaptive._phase is Phase.DEVELOPMENT
        and adaptive._active_block is None
        and adaptive._sweep_order == []
    ):
        adaptive._advance()
    checkpoint = save_checkpoint(adaptive)
    assert checkpoint is not None

    with pytest.raises(ValueError, match="scheduler mismatch"):
        TraceAADV917(
            llm=ConstantLLM(),
            evaluation=ValueEvaluation(),
            budget=11,
            n_roots=1,
            adaptive_sweeps=False,
            resume_from=checkpoint,
        )
    fixed = TraceAADV917(
        llm=ConstantLLM(),
        evaluation=ValueEvaluation(),
        budget=11,
        n_roots=1,
        adaptive_sweeps=False,
        fork_from_initialization=True,
        resume_from=checkpoint,
        checkpoint_dir=tmp_path / "fixed",
    )
    fixed.run()
    assert fixed._n_eval == 11
    assert fixed._explore_slots == 1
    assert fixed._block_counts[BlockKind.DEVELOPMENT.value] == 1


def test_interrupted_repair_resumes_same_block_step_without_new_slot(
    tmp_path: Path,
) -> None:
    first = TraceAADV917(
        llm=InterruptedRepairLLM(),
        evaluation=ValueEvaluation(),
        budget=4,
        n_roots=1,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        first.run()

    assert first._n_eval == 2
    assert first._n_calls == 2
    assert first._generation is not None
    assert first._generation.attempt == 2
    assert first._generation.primary_charged is True
    assert first._active_block is not None
    assert first._active_block.completed_steps == 1

    resumed = TraceAADV917(
        llm=ConstantLLM(),
        evaluation=ValueEvaluation(),
        budget=4,
        n_roots=1,
        checkpoint_dir=tmp_path / "checkpoints",
        resume_from=tmp_path / "checkpoints" / "latest.json",
    )
    resumed.run()

    assert resumed._n_eval == 4
    assert resumed._n_calls == 5
    assert resumed._repair_llm_calls == 1
    assert resumed._refine_slots == 3
    assert sum(item.refine_count for item in resumed._tree.valid_algorithms()) == 3


def test_invalid_discovery_reopens_development_without_a_hypothesis() -> None:
    responses = [
        *([_program(1)] * 7),
        "invalid",
        "invalid",
        "invalid",
        *([_program(1)] * 3),
    ]
    method = TraceAADV917(
        llm=ScriptedLLM(responses),
        evaluation=ValueEvaluation(),
        budget=11,
        n_roots=1,
    )
    method.run()

    assert method._explore_slots == 1
    assert method._discovery_attempts == 1
    assert method._valid_discoveries == 0
    assert len(method._hypotheses) == 1
    assert method._active_ids == [1]
    assert method._block_counts[BlockKind.DEVELOPMENT.value] == 2
    assert method._block_counts[BlockKind.MATURATION.value] == 0


def test_insufficient_discovery_budget_is_spent_on_refine() -> None:
    method = TraceAADV917(
        llm=ConstantLLM(),
        evaluation=ValueEvaluation(),
        budget=10,
        n_roots=1,
    )
    method.run()

    assert method._n_eval == 10
    assert method._explore_slots == 0
    assert method._block_counts[BlockKind.TERMINAL.value] == 1
    assert method._phase is Phase.TERMINAL
