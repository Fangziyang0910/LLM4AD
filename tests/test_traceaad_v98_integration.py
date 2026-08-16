from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.runners.traceaad import launch_v98, run
from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad_v9_8 import (
    CHECKPOINT_VERSION,
    PROTOCOL_ID,
    AllocationPolicy,
    RunArtifacts,
    TraceAADV98,
)
from llm4ad.method.traceaad_v9_8.forest import Forest
from llm4ad.method.traceaad_v9_8.checkpoint import dump_state, load_state
from llm4ad.method.traceaad_v9_8.history import parent_path, render_path
from llm4ad.method.traceaad_v9_8.schema import Attempt, Intent, Outcome, Pending
from llm4ad.method.traceaad_v9_8.selection import hypothesis_scores, select
from llm4ad.method.traceaad_v9_8.traceaad import draw_intent

TEMPLATE = """def choose(value: int) -> int:
    return value
"""


class ScriptedLLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def draw_sample(self, prompt, *args, **kwargs):
        self.calls += 1
        return (
            f"Idea: candidate {self.calls}\n"
            "Code:\n```python\n"
            "def choose(value: int) -> int:\n"
            f"    return value + {self.calls}\n"
            "```"
        )

    def count_tokens(self, text: str) -> int:
        return len(text) // 4 + 1


class IncreasingEvaluation(Evaluation):
    def __init__(self) -> None:
        super().__init__(
            template_program=TEMPLATE,
            task_description="Improve choose.",
            use_numba_accelerate=False,
            safe_evaluate=False,
            timeout_seconds=10,
        )
        self.calls = 0

    def evaluate_program(self, program_str, callable_func, **kwargs):
        self.calls += 1
        return float(self.calls)


def _program(forest: Forest, code: str, q: float, order: int):
    return forest.add_program(code=code, fitness=q, order=order)


def _attempt(
    attempt_id: int,
    *,
    parent_id: int,
    source_hypothesis_id: int,
    child_id: int,
    child_hypothesis_id: int,
    program_id: int,
    intent: Intent,
    order: int,
) -> Attempt:
    return Attempt(
        id=attempt_id,
        response_id=f"r{attempt_id}",
        anchor_id=parent_id,
        source_hypothesis_id=source_hypothesis_id,
        child_id=child_id,
        child_hypothesis_id=child_hypothesis_id,
        program_id=program_id,
        intent=intent.value,
        idea=f"{intent.value} idea",
        diff="--- parent.py\n+++ candidate.py\n-old\n+new",
        added=1,
        removed=1,
        parent_fitness=1.0,
        child_fitness=0.5,
        dq=-0.5,
        frontier_before=1.0,
        frontier_after=1.0 if intent is Intent.EXPLORE else 0.5,
        realized_gain=None if intent is Intent.EXPLORE else 0.0,
        outcome=Outcome.REGRESS,
        kind="new_hypothesis" if intent is Intent.EXPLORE else "new",
        order=order,
        stage="search",
        iteration=order,
    )


def test_explore_creates_segment_and_refine_inherits_it() -> None:
    forest = Forest(maximize=True)
    root_program = _program(forest, "root", 1.0, 1)
    root, root_h = forest.add_root(program_id=root_program.id, order=1)
    explore_program = _program(forest, "explore", 0.5, 2)
    explore_attempt = forest.next_attempt_id()
    entry, child_h = forest.add_explore_child(
        parent_id=root.id,
        program_id=explore_program.id,
        attempt_id=explore_attempt,
        order=2,
    )
    forest.add_attempt(
        _attempt(
            explore_attempt,
            parent_id=root.id,
            source_hypothesis_id=root_h.id,
            child_id=entry.id,
            child_hypothesis_id=child_h.id,
            program_id=explore_program.id,
            intent=Intent.EXPLORE,
            order=2,
        )
    )
    refined_program = _program(forest, "refined", 0.8, 3)
    refine_attempt = forest.next_attempt_id()
    refined = forest.add_refine_child(
        parent_id=entry.id,
        program_id=refined_program.id,
        attempt_id=refine_attempt,
        order=3,
    )

    assert root.hypothesis_id == root_h.id
    assert child_h.parent_hypothesis_id == root_h.id
    assert child_h.q0 == 0.5
    assert child_h.q_base == 1.0
    assert refined.hypothesis_id == child_h.id
    assert refined.root_id == root.root_id


def test_boundary_grace_restores_source_baseline_at_creation() -> None:
    forest = Forest(maximize=True)
    root_program = _program(forest, "root", 10.0, 1)
    root, _ = forest.add_root(program_id=root_program.id, order=1)
    child_program = _program(forest, "child", 4.0, 2)
    child, hypothesis = forest.add_explore_child(
        parent_id=root.id,
        program_id=child_program.id,
        attempt_id=forest.next_attempt_id(),
        order=2,
    )

    scores = {
        item.hypothesis_id: item
        for item in hypothesis_scores(
            forest,
            s0=2.0,
            intent=Intent.REFINE,
            policy=AllocationPolicy.FULL,
        )
    }
    score = scores[hypothesis.id]
    assert score.q == 4.0
    assert score.u == 2.0
    assert score.c == 6.0
    assert score.m == 0.0
    assert score.score == 12.0

    hypothesis.increment(Intent.REFINE)
    child.increment(Intent.REFINE)
    decayed = {
        item.hypothesis_id: item
        for item in hypothesis_scores(
            forest,
            s0=2.0,
            intent=Intent.REFINE,
            policy=AllocationPolicy.FULL,
        )
    }[hypothesis.id]
    assert decayed.c == pytest.approx(6.0 / 2**0.5)
    assert decayed.u == pytest.approx(2.0 / 2**0.5)


def test_q_u_c_m_ablation_changes_only_declared_components() -> None:
    forest = Forest(maximize=True)
    root_program = _program(forest, "root", 5.0, 1)
    root, _ = forest.add_root(program_id=root_program.id, order=1)
    entry_program = _program(forest, "entry", 3.0, 2)
    entry, hypothesis = forest.add_explore_child(
        parent_id=root.id,
        program_id=entry_program.id,
        attempt_id=forest.next_attempt_id(),
        order=2,
    )
    improved_program = _program(forest, "improved", 4.0, 3)
    forest.add_refine_child(
        parent_id=entry.id,
        program_id=improved_program.id,
        attempt_id=forest.next_attempt_id(),
        order=3,
    )
    hypothesis.n_refine = 2

    def score(policy: AllocationPolicy):
        return {
            item.hypothesis_id: item
            for item in hypothesis_scores(
                forest, s0=2.0, intent=Intent.REFINE, policy=policy
            )
        }[hypothesis.id]

    q_u = score(AllocationPolicy.Q_U)
    q_u_c = score(AllocationPolicy.Q_U_C)
    full = score(AllocationPolicy.FULL)
    assert q_u.c == q_u.m == 0.0
    assert q_u_c.m == 0.0 and q_u_c.c > 0
    assert full.c == q_u_c.c
    assert full.m == pytest.approx((4.0 - 3.0) / 2)
    assert full.score == pytest.approx(q_u_c.score + full.m)


def test_choice_is_operator_specific_and_route_provenance_not_used_by_full_policy() -> None:
    forest = Forest(maximize=True)
    first_program = _program(forest, "first", 1.0, 1)
    first, first_h = forest.add_root(program_id=first_program.id, order=1)
    second_program = _program(forest, "second", 1.0, 2)
    second, second_h = forest.add_root(program_id=second_program.id, order=2)
    first_h.n_refine = first.n_refine = 9
    second_h.n_refine = second.n_refine = 0
    first_h.n_explore = first.n_explore = 0
    second_h.n_explore = second.n_explore = 9

    refine = select(forest, s0=1.0, intent=Intent.REFINE)
    explore = select(forest, s0=1.0, intent=Intent.EXPLORE)
    assert refine.hypothesis_id == second_h.id
    assert explore.hypothesis_id == first_h.id
    assert refine.routes == explore.routes == ()


def test_history_marks_hypothesis_boundary() -> None:
    forest = Forest(maximize=True)
    root_program = _program(forest, "root", 1.0, 1)
    root, root_h = forest.add_root(program_id=root_program.id, order=1)
    child_program = _program(forest, "child", 0.5, 2)
    attempt_id = forest.next_attempt_id()
    child, child_h = forest.add_explore_child(
        parent_id=root.id,
        program_id=child_program.id,
        attempt_id=attempt_id,
        order=2,
    )
    forest.add_attempt(
        _attempt(
            attempt_id,
            parent_id=root.id,
            source_hypothesis_id=root_h.id,
            child_id=child.id,
            child_hypothesis_id=child_h.id,
            program_id=child_program.id,
            intent=Intent.EXPLORE,
            order=2,
        )
    )
    text = render_path(forest, parent_path(forest, child.id))
    assert "Operator: Explore" in text
    assert f"create H{child_h.id} from H{root_h.id}" in text


def test_intent_draw_is_deterministic_and_fixed_prior() -> None:
    draws = [draw_intent(5, index) for index in range(2000)]
    assert draws == [draw_intent(5, index) for index in range(2000)]
    assert abs(draws.count(Intent.REFINE) / len(draws) - 0.7) < 0.03


def test_runner_builds_frozen_v98_and_records_config(tmp_path: Path) -> None:
    spec = run.make_run_spec(
        task="tsp_construct",
        version="v9_8",
        backend="server3",
        budget=1000,
        run_name="v98",
        experiments_root=tmp_path,
    )
    method = run.build_method(spec, tmp_path / "run")
    assert isinstance(method, TraceAADV98)
    assert method.search_configuration() == run._v98_method_params(spec)
    assert method.search_configuration()["protocol_id"] == PROTOCOL_ID
    assert method.search_configuration()["checkpoint_schema_version"] == CHECKPOINT_VERSION
    run_dir, run_name, _ = run.resolve_run_dir(spec)
    run.write_run_config(spec, run_dir, run_name)
    payload = json.loads((run_dir / "run_config.json").read_text())
    assert payload["method"] == "traceaad_v9_8"
    assert payload["method_params"] == run._v98_method_params(spec)
    assert payload["generator_environment"]["logical_model_name"] == "Qwen3.6-27B"
    assert len(payload["implementation"]["protocol_source_sha256"]) == 64
    assert payload["implementation"]["git_commit"]
    assert payload["implementation"]["source_files"]
    assert "backend" not in payload and "llm" not in payload
    method._llm.close()


def test_end_to_end_v98_smoke_streams_facts_and_exhausts_eval_budget(
    tmp_path: Path,
) -> None:
    artifacts = RunArtifacts(tmp_path, console_output=False)
    method = TraceAADV98(
        llm=ScriptedLLM(),
        evaluation=IncreasingEvaluation(),
        artifacts=artifacts,
        budget=20,
        n_roots=8,
        context_limit=32768,
        checkpoint_dir=tmp_path / "checkpoints",
        seed=3,
    )
    method.run()

    summary = json.loads((tmp_path / "logs" / "summary.json").read_text())
    assert summary["status"] == "finished"
    assert summary["evaluator_call_count"] == 20
    assert summary["n_roots"] == 8
    assert summary["n_hypotheses"] >= 8
    assert (tmp_path / "checkpoints" / "latest.json").is_file()
    assert (tmp_path / "logs" / "events.jsonl").stat().st_size > 0
    assert (tmp_path / "logs" / "llm_calls.jsonl").stat().st_size > 0
    with (tmp_path / "evaluations.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 20
    assert {row["operator"] for row in rows if row["stage"] == "search"} <= {
        "refine",
        "explore",
    }


def test_checkpoint_with_completed_pending_response_does_not_call_llm_again(
    tmp_path: Path,
) -> None:
    source_llm = ScriptedLLM()
    source = TraceAADV98(
        llm=source_llm,
        evaluation=IncreasingEvaluation(),
        budget=20,
        n_roots=8,
        context_limit=32768,
        checkpoint_dir=tmp_path / "source",
    )
    program = source._forest.add_program(
        code="def choose(value: int) -> int:\n    return value\n",
        fitness=1.0,
        order=1,
    )
    anchor, hypothesis = source._forest.add_root(program_id=program.id, order=1)
    hypothesis.n_refine = 1
    anchor.n_refine = 1
    source._n_candidates = 1
    source._n_eval = 1
    source._pending = Pending(
        id=source._forest.next_attempt_id(),
        response_id="v98-r000001-a000000",
        anchor_id=anchor.id,
        hypothesis_id=hypothesis.id,
        stage="search",
        iteration=0,
        order=1,
        intent="refine",
        prompt="saved prompt",
        generation_seed=1,
        selection={"selected_anchor_id": anchor.id},
        response=(
            "Idea: resumed\nCode:\n```python\n"
            "def choose(value: int) -> int:\n    return value + 10\n```"
        ),
    )
    payload = json.loads(json.dumps(dump_state(source)))

    resumed_llm = ScriptedLLM()
    resumed = TraceAADV98(
        llm=resumed_llm,
        evaluation=IncreasingEvaluation(),
        budget=20,
        n_roots=8,
        context_limit=32768,
        checkpoint_dir=tmp_path / "resumed",
    )
    load_state(resumed, payload)
    resumed._resume_pending()

    assert resumed_llm.calls == 0
    assert resumed._pending is None
    assert resumed._n_candidates == 1
    assert resumed._n_eval == 2
    assert resumed._forest.get_hypothesis(hypothesis.id).n_refine == 1


def test_pending_request_recovers_durably_logged_response_before_redraw(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    first_artifacts = RunArtifacts(run_dir, console_output=False)
    response = (
        "Idea: durable\nCode:\n```python\n"
        "def choose(value: int) -> int:\n    return value + 20\n```"
    )
    first_artifacts.record_llm_call(
        response_id="v98-r000001-a000000",
        status="ok",
        transport_attempt=1,
        raw_response=response,
    )
    first_artifacts.finish()

    artifacts = RunArtifacts(run_dir, console_output=False)
    llm = ScriptedLLM()
    method = TraceAADV98(
        llm=llm,
        evaluation=IncreasingEvaluation(),
        artifacts=artifacts,
        budget=20,
        n_roots=8,
        context_limit=32768,
        checkpoint_dir=run_dir / "checkpoints",
    )
    program = method._forest.add_program(
        code="def choose(value: int) -> int:\n    return value\n",
        fitness=1.0,
        order=1,
    )
    anchor, hypothesis = method._forest.add_root(program_id=program.id, order=1)
    method._pending = Pending(
        id=method._forest.next_attempt_id(),
        response_id="v98-r000001-a000000",
        anchor_id=anchor.id,
        hypothesis_id=hypothesis.id,
        stage="search",
        iteration=0,
        order=1,
        intent="refine",
        prompt="original prompt",
        generation_seed=1,
        selection=None,
        response=None,
    )

    method._resume_pending()

    assert llm.calls == 0
    assert method._n_candidates == 1
    assert method._n_eval == 1
    artifacts.finish()


def test_formal_launcher_resumes_existing_incomplete_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "existing"
    run_dir.mkdir()
    item = launch_v98.Item(
        task="tsp_constructive",
        repeat=1,
        seed=0,
        session="v98_resume_test",
        run_name="existing",
        run_dir=run_dir,
    )
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(launch_v98.subprocess, "run", fake_run)
    launch_v98.launch(
        item,
        backend="server3",
        allocation_policy=AllocationPolicy.FULL.value,
        dry_run=False,
    )

    command = calls[0]
    assert "--resume-from" in command
    assert str(run_dir) in command
    assert "--run-name" not in command
