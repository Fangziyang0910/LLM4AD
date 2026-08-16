from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from experiments.runners.traceaad import launch_v99, run
from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad_v9_9 import (
    CHECKPOINT_VERSION,
    PROTOCOL_ID,
    ROOT_CANDIDATE_COUNT,
    RunArtifacts,
    TraceAADV99,
)
from llm4ad.method.traceaad_v9_9.checkpoint import dump_state, load_state
from llm4ad.method.traceaad_v9_9.forest import Forest
from llm4ad.method.traceaad_v9_9.history import parent_path, render_path
from llm4ad.method.traceaad_v9_9.schema import Attempt, Intent, Outcome, Pending
from llm4ad.method.traceaad_v9_9.selection import (
    distance_grace,
    geometric_rank_weights,
    midrank_percentiles,
    score_anchors,
    select,
)

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
    child_id: int,
    program_id: int,
    intent: Intent,
    order: int,
) -> Attempt:
    return Attempt(
        id=attempt_id,
        response_id=f"r{attempt_id}",
        anchor_id=parent_id,
        child_id=child_id,
        program_id=program_id,
        intent=intent.value,
        idea=f"{intent.value} idea",
        diff="--- parent.py\n+++ candidate.py\n-old\n+new",
        added=1,
        removed=1,
        parent_fitness=1.0,
        child_fitness=0.5,
        dq=-0.5,
        outcome=Outcome.REGRESS,
        kind="new",
        order=order,
        stage="search",
        iteration=order,
    )


def test_midrank_uses_average_ranks_of_ties() -> None:
    ranks = midrank_percentiles({0: 1.0, 1: 2.0, 2: 2.0})
    assert ranks[0] == 0.0
    assert ranks[1] == ranks[2] == 0.75
    assert midrank_percentiles({7: 3.0}) == {7: 0.5}


def test_distance_grace_prefers_a_near_ancestor_over_a_distant_best() -> None:
    forest = Forest(maximize=True)
    root = forest.add_root(program_id=_program(forest, "root", 10.0, 1).id, order=1)
    current = root
    for index in range(18):
        current = forest.add_child(
            parent_id=current.id,
            program_id=_program(forest, f"step-{index}", 5.0, index + 2).id,
            attempt_id=forest.next_attempt_id(),
            order=index + 2,
        )
    parent = forest.get_anchor(current.parent_id)
    ranks = {
        root.program_id: 1.0,
        parent.program_id: 0.6,
        current.program_id: 0.5,
    }
    for anchor in forest.anchors():
        ranks.setdefault(anchor.program_id, 0.5)
    grace = distance_grace(forest, current.id, ranks, half_life=4.0)
    near = (2.0 ** (-1 / 4)) * 0.1
    far = (2.0 ** (-18 / 4)) * 0.5
    assert grace == pytest.approx(near)
    assert near > far


def test_refine_grace_does_not_enter_explore_score() -> None:
    forest = Forest(maximize=True)
    root = forest.add_root(program_id=_program(forest, "root", 10.0, 1).id, order=1)
    child = forest.add_child(
        parent_id=root.id,
        program_id=_program(forest, "child", 4.0, 2).id,
        attempt_id=forest.next_attempt_id(),
        order=2,
    )
    scores = {item.anchor_id: item for item in score_anchors(forest)}
    assert scores[root.id].c_refine == 0.0
    assert scores[child.id].c_refine > 0
    assert scores[child.id].s_refine - scores[child.id].q == pytest.approx(
        0.25 * scores[child.id].u_refine + scores[child.id].c_refine
    )
    assert scores[child.id].s_explore - scores[child.id].q == pytest.approx(
        0.25 * scores[child.id].u_explore
    )


def test_geometric_rank_keeps_a_bounded_tail_and_shares_tied_mass() -> None:
    distinct = geometric_rank_weights(
        tuple((index, float(200 - index)) for index in range(200))
    )
    assert all(weight > 0 for weight in distinct.values())
    assert sum(distinct[index] for index in range(10)) == pytest.approx(0.75, abs=0.01)
    tied = geometric_rank_weights(((0, 1.0), (1, 1.0), (2, 0.0)))
    expected_top = ((1.0 + 2.0 ** (-1 / 5)) / 2) / (
        1.0 + 2.0 ** (-1 / 5) + 2.0 ** (-2 / 5)
    )
    assert tied[0] == tied[1] == pytest.approx(expected_top)
    assert tied[2] == pytest.approx(
        (2.0 ** (-2 / 5)) / (1.0 + 2.0 ** (-1 / 5) + 2.0 ** (-2 / 5))
    )


def test_joint_selection_is_deterministic_and_quality_cancels_in_operator_odds() -> None:
    forest = Forest(maximize=True)
    weak = forest.add_root(program_id=_program(forest, "weak", 1.0, 1).id, order=1)
    strong = forest.add_root(program_id=_program(forest, "strong", 9.0, 2).id, order=2)
    first = select(forest, seed=5, iteration=0)
    second = select(forest, seed=5, iteration=0)
    assert first.anchor_id == second.anchor_id
    assert first.intent is second.intent
    scores = {item.anchor_id: item for item in first.scores}
    assert scores[weak.id].pi_refine == pytest.approx(scores[strong.id].pi_refine)
    assert scores[weak.id].pi_refine == pytest.approx(0.7)
    assert math.isclose(sum(item.mu for item in first.scores), 1.0)


def test_history_omits_hypothesis_markers() -> None:
    forest = Forest(maximize=True)
    root = forest.add_root(program_id=_program(forest, "root", 1.0, 1).id, order=1)
    attempt_id = forest.next_attempt_id()
    child = forest.add_child(
        parent_id=root.id,
        program_id=_program(forest, "child", 0.5, 2).id,
        attempt_id=attempt_id,
        order=2,
    )
    forest.add_attempt(
        _attempt(
            attempt_id,
            parent_id=root.id,
            child_id=child.id,
            program_id=child.program_id,
            intent=Intent.EXPLORE,
            order=2,
        )
    )
    text = render_path(forest, parent_path(forest, child.id))
    assert "Operator: Explore" in text
    assert "Hypothesis" not in text


def test_cached_code_creates_a_child_for_both_operators() -> None:
    forest = Forest(maximize=True)
    root = forest.add_root(program_id=_program(forest, "root", 1.0, 1).id, order=1)
    seen = _program(forest, "seen", 2.0, 2)
    other = forest.add_root(program_id=seen.id, order=2)
    refine_child = forest.add_child(
        parent_id=root.id,
        program_id=seen.id,
        attempt_id=forest.next_attempt_id(),
        order=3,
    )
    explore_parent = forest.add_root(program_id=_program(forest, "third", 0.5, 4).id, order=4)
    explore_child = forest.add_child(
        parent_id=explore_parent.id,
        program_id=seen.id,
        attempt_id=forest.next_attempt_id(),
        order=5,
    )
    assert refine_child.program_id == explore_child.program_id == seen.id
    assert refine_child.parent_id == root.id
    assert explore_child.parent_id == explore_parent.id
    assert other.program_id == seen.id


def test_initialization_keeps_the_best_roots_and_skips_bootstrap(
    tmp_path: Path,
) -> None:
    artifacts = RunArtifacts(tmp_path, console_output=False)
    method = TraceAADV99(
        llm=ScriptedLLM(),
        evaluation=IncreasingEvaluation(),
        artifacts=artifacts,
        budget=6,
        n_roots=2,
        n_root_candidates=4,
        context_limit=32768,
        checkpoint_dir=tmp_path / "checkpoints",
        seed=1,
    )
    method.run()

    summary = json.loads((tmp_path / "logs" / "summary.json").read_text())
    assert summary["status"] == "finished"
    assert summary["evaluator_call_count"] == 6
    assert summary["n_root_candidates"] == 4
    assert summary["n_roots"] == 2
    assert summary["n_discarded_roots"] == 2
    root_programs = [
        method._forest.get_program(method._forest.get_anchor(root_id).program_id)
        for root_id in method._forest.root_ids
    ]
    discarded = [
        method._forest.get_program(program_id)
        for program_id in method._forest.discarded_program_ids
    ]
    assert {program.q for program in root_programs} == {3.0, 4.0}
    assert {program.q for program in discarded} == {1.0, 2.0}
    assert all(anchor.parent_id is None for anchor in method._forest.anchors() if anchor.id in method._forest.root_ids)
    with (tmp_path / "evaluations.csv").open(newline="") as handle:
        stages = {row["stage"] for row in csv.DictReader(handle)}
    assert "bootstrap" not in stages
    assert "root_generation" in stages
    assert "search" in stages


def test_runner_builds_frozen_v99_and_records_config(tmp_path: Path) -> None:
    spec = run.make_run_spec(
        task="tsp_construct",
        version="v9_9",
        backend="server3",
        budget=1000,
        run_name="v99",
        experiments_root=tmp_path,
    )
    method = run.build_method(spec, tmp_path / "run")
    assert isinstance(method, TraceAADV99)
    assert method.search_configuration() == run._v99_method_params(spec)
    assert method.search_configuration()["protocol_id"] == PROTOCOL_ID
    assert method.search_configuration()["checkpoint_schema_version"] == CHECKPOINT_VERSION
    assert spec.n_init == 8
    assert method._n_root_candidates == ROOT_CANDIDATE_COUNT
    run_dir, run_name, _ = run.resolve_run_dir(spec)
    run.write_run_config(spec, run_dir, run_name)
    payload = json.loads((run_dir / "run_config.json").read_text())
    assert payload["method"] == "traceaad_v9_9"
    assert payload["method_params"] == run._v99_method_params(spec)
    assert payload["generator_environment"]["logical_model_name"] == "Qwen3.6-27B"
    assert len(payload["implementation"]["protocol_source_sha256"]) == 64
    assert "backend" not in payload and "llm" not in payload
    method._llm.close()


def test_end_to_end_v99_smoke_streams_facts_and_exhausts_eval_budget(
    tmp_path: Path,
) -> None:
    artifacts = RunArtifacts(tmp_path, console_output=False)
    method = TraceAADV99(
        llm=ScriptedLLM(),
        evaluation=IncreasingEvaluation(),
        artifacts=artifacts,
        budget=20,
        context_limit=32768,
        checkpoint_dir=tmp_path / "checkpoints",
        seed=3,
    )
    method.run()

    summary = json.loads((tmp_path / "logs" / "summary.json").read_text())
    assert summary["status"] == "finished"
    assert summary["evaluator_call_count"] == 20
    assert summary["n_roots"] == 8
    assert summary["n_root_candidates"] == 12
    assert summary["n_discarded_roots"] == 4
    assert (tmp_path / "checkpoints" / "latest.json").is_file()
    assert (tmp_path / "logs" / "events.jsonl").stat().st_size > 0
    with (tmp_path / "evaluations.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 20
    assert {row["operator"] for row in rows if row["stage"] == "search"} <= {
        "refine",
        "explore",
    }
    assert all(row["stage"] != "bootstrap" for row in rows)


def test_checkpoint_with_completed_pending_response_does_not_call_llm_again(
    tmp_path: Path,
) -> None:
    source_llm = ScriptedLLM()
    source = TraceAADV99(
        llm=source_llm,
        evaluation=IncreasingEvaluation(),
        budget=20,
        n_roots=1,
        n_root_candidates=1,
        context_limit=32768,
        checkpoint_dir=tmp_path / "source",
    )
    program = source._forest.add_program(
        code="def choose(value: int) -> int:\n    return value\n",
        fitness=1.0,
        order=1,
    )
    anchor = source._forest.add_root(program_id=program.id, order=1)
    source._root_candidate_ids = [program.id]
    source._initialization_complete = True
    source._n_candidates = 1
    source._n_eval = 1
    anchor.n_refine = 1
    source._pending = Pending(
        id=source._forest.next_attempt_id(),
        response_id="v99-r000001-a000000",
        anchor_id=anchor.id,
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
    resumed = TraceAADV99(
        llm=resumed_llm,
        evaluation=IncreasingEvaluation(),
        budget=20,
        n_roots=1,
        n_root_candidates=1,
        context_limit=32768,
        checkpoint_dir=tmp_path / "resumed",
    )
    load_state(resumed, payload)
    resumed._resume_pending()

    assert resumed_llm.calls == 0
    assert resumed._pending is None
    assert resumed._n_candidates == 1
    assert resumed._n_eval == 2
    assert resumed._forest.get_anchor(anchor.id).n_refine == 1


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
        response_id="v99-r000001-a000000",
        status="ok",
        transport_attempt=1,
        raw_response=response,
    )
    first_artifacts.finish()

    artifacts = RunArtifacts(run_dir, console_output=False)
    llm = ScriptedLLM()
    method = TraceAADV99(
        llm=llm,
        evaluation=IncreasingEvaluation(),
        artifacts=artifacts,
        budget=20,
        n_roots=1,
        n_root_candidates=1,
        context_limit=32768,
        checkpoint_dir=run_dir / "checkpoints",
    )
    program = method._forest.add_program(
        code="def choose(value: int) -> int:\n    return value\n",
        fitness=1.0,
        order=1,
    )
    anchor = method._forest.add_root(program_id=program.id, order=1)
    method._root_candidate_ids = [program.id]
    method._initialization_complete = True
    method._pending = Pending(
        id=method._forest.next_attempt_id(),
        response_id="v99-r000001-a000000",
        anchor_id=anchor.id,
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
    item = launch_v99.Item(
        task="tsp_constructive",
        repeat=1,
        seed=0,
        session="v99_resume_test",
        run_name="existing",
        run_dir=run_dir,
    )
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(launch_v99.subprocess, "run", fake_run)
    launch_v99.launch(item, backend="server3", dry_run=False)

    command = calls[0]
    assert "--resume-from" in command
    assert str(run_dir) in command
    assert "--run-name" not in command
    assert "v9_9" in command
