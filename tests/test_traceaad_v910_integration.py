from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from experiments.runners.traceaad import launch_v910, run
from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad_v9_10 import (
    CHILD_WINDOW,
    RunArtifacts,
    TraceAADV910,
)
from llm4ad.method.traceaad_v9_10.checkpoint import load_checkpoint, save_checkpoint
from llm4ad.method.traceaad_v9_10.forest import Forest
from llm4ad.method.traceaad_v9_10.schema import (
    Action,
    ActionStatus,
    Intent,
    Outcome,
    Pending,
)
from llm4ad.method.traceaad_v9_10.selection import (
    discounted_posterior,
    parent_chain_window,
    prior_counts,
    select,
    settle_pending_actions,
)

TEMPLATE = """def choose(value: int) -> int:
    return value
"""


class ScriptedLLM(LLM):
    def __init__(self, start: int = 0) -> None:
        super().__init__()
        self.calls = start

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


def _action(
    action_id: int,
    *,
    anchor_id: int,
    child_id: int | None = None,
    program_id: int | None = None,
    intent: str = "refine",
    order: int,
    kind: str = "new",
    outcome: Outcome = Outcome.REGRESS,
) -> Action:
    return Action(
        id=action_id,
        response_id=f"a{action_id}",
        anchor_id=anchor_id,
        child_id=child_id,
        program_id=program_id,
        intent=intent,
        idea=f"{intent} idea",
        diff="--- parent.py\n+++ candidate.py\n-old\n+new",
        added=1,
        removed=1,
        parent_fitness=1.0,
        child_fitness=0.5,
        dq=-0.5,
        outcome=outcome,
        kind=kind,
        order=order,
        stage="search",
        iteration=order,
    )


def _settle(
    action: Action,
    *,
    result: int,
    order: int,
) -> Action:
    action.settle(result, order)
    return action


def _chain(
    forest: Forest,
    parent_id: int,
    q: float,
    order: int,
) -> int:
    """Extend the formation tree by one child anchor holding a new program."""
    program = _program(forest, f"code-{order}", q, order)
    return forest.add_child(
        parent_id=parent_id,
        program_id=program.id,
        action_id=forest.next_action_id(),
        order=order,
    ).id


def test_prior_counts_match_intent_prior_strength() -> None:
    assert prior_counts(Intent.REFINE) == pytest.approx((1.4, 0.6))
    assert prior_counts(Intent.EXPLORE) == pytest.approx((0.6, 1.4))


def test_childless_action_settles_as_failure_without_a_window() -> None:
    forest = Forest(maximize=True)
    root = forest.add_root(program_id=_program(forest, "root", 1.0, 1).id, order=1)
    action = _action(
        forest.next_action_id(),
        anchor_id=root.id,
        intent="explore",
        order=4,
        kind="invalid",
        outcome=Outcome.INVALID,
    )
    forest.add_action(action)

    settled = settle_pending_actions(forest, now_order=4)

    assert [item.id for item in settled] == [action.id]
    assert action.status is ActionStatus.SETTLED
    assert action.result == 0
    assert action.settled_order == 4
    assert action.window_best_q is None
    assert action.observed_depth == 0


def test_improving_child_settles_success_immediately() -> None:
    forest = Forest(maximize=True)
    root = forest.add_root(program_id=_program(forest, "root", 1.0, 1).id, order=1)
    child = forest.add_child(
        parent_id=root.id,
        program_id=_program(forest, "child", 2.0, 2).id,
        action_id=forest.next_action_id(),
        order=2,
    )
    action = _action(
        forest.next_action_id(),
        anchor_id=root.id,
        child_id=child.id,
        program_id=child.program_id,
        order=2,
        outcome=Outcome.IMPROVE,
    )
    forest.add_action(action)

    settled = settle_pending_actions(forest, now_order=2)

    assert [item.id for item in settled] == [action.id]
    assert action.result == 1
    assert action.window_best_q == pytest.approx(2.0)
    assert action.observed_depth == 0


def test_regressed_child_stays_pending_until_window_depth_is_reached() -> None:
    forest = Forest(maximize=True)
    root = forest.add_root(program_id=_program(forest, "root", 5.0, 1).id, order=1)
    action_id = forest.next_action_id()
    child = forest.add_child(
        parent_id=root.id,
        program_id=_program(forest, "c1", 4.0, 2).id,
        action_id=action_id,
        order=2,
    )
    action = _action(
        action_id,
        anchor_id=root.id,
        child_id=child.id,
        program_id=child.program_id,
        order=2,
    )
    forest.add_action(action)

    assert settle_pending_actions(forest, now_order=2) == ()
    assert action.status is ActionStatus.PENDING
    assert action.observed_depth == 0

    grandchild = _chain(forest, child.id, 4.5, 3)
    assert settle_pending_actions(forest, now_order=3) == ()
    assert action.observed_depth == 1

    great_grandchild = _chain(forest, grandchild, 4.8, 4)
    assert settle_pending_actions(forest, now_order=4) == ()
    assert action.observed_depth == 2

    _chain(forest, great_grandchild, 4.9, 5)
    settled = settle_pending_actions(forest, now_order=5)

    assert [item.id for item in settled] == [action.id]
    assert action.result == 0
    assert action.observed_depth == CHILD_WINDOW
    assert action.window_best_q == pytest.approx(4.9)
    assert action.settled_order == 5


def test_response_age_settlement_closes_unvisited_child() -> None:
    forest = Forest(maximize=True)
    root = forest.add_root(program_id=_program(forest, "root", 5.0, 1).id, order=1)
    action_id = forest.next_action_id()
    child = forest.add_child(
        parent_id=root.id,
        program_id=_program(forest, "child", 4.0, 2).id,
        action_id=action_id,
        order=2,
    )
    action = _action(
        action_id,
        anchor_id=root.id,
        child_id=child.id,
        program_id=child.program_id,
        order=2,
    )
    forest.add_action(action)

    assert settle_pending_actions(
        forest, now_order=5, child_window=3, settlement_mode="response_age"
    ) == (action,)
    assert action.result == 0
    assert action.status is ActionStatus.SETTLED


def test_uniform_allocation_ignores_sampled_arm_weights() -> None:
    forest = Forest(maximize=True)
    forest.add_root(program_id=_program(forest, "root1", 1.0, 1).id, order=1)
    forest.add_root(program_id=_program(forest, "root2", 2.0, 2).id, order=2)

    choice = select(forest, seed=11, order=3, allocation_mode="uniform")

    assert len({round(item.omega, 12) for item in choice.arms}) == 1
    assert sum(item.omega for item in choice.arms) == pytest.approx(1.0)


def test_descendant_recovery_inside_the_window_settles_success() -> None:
    forest = Forest(maximize=True)
    root = forest.add_root(program_id=_program(forest, "root", 5.0, 1).id, order=1)
    action_id = forest.next_action_id()
    child = forest.add_child(
        parent_id=root.id,
        program_id=_program(forest, "c1", 4.0, 2).id,
        action_id=action_id,
        order=2,
    )
    action = _action(action_id, anchor_id=root.id, child_id=child.id, order=2)
    forest.add_action(action)
    assert settle_pending_actions(forest, now_order=2) == ()

    _chain(forest, child.id, 5.5, 3)
    settled = settle_pending_actions(forest, now_order=3)

    assert [item.id for item in settled] == [action.id]
    assert action.result == 1
    assert action.window_best_q == pytest.approx(5.5)
    assert action.observed_depth == 1


def test_window_ignores_descendants_beyond_child_window() -> None:
    forest = Forest(maximize=True)
    root = forest.add_root(program_id=_program(forest, "root", 5.0, 1).id, order=1)
    action_id = forest.next_action_id()
    child = forest.add_child(
        parent_id=root.id,
        program_id=_program(forest, "c1", 4.0, 2).id,
        action_id=action_id,
        order=2,
    )
    action = _action(action_id, anchor_id=root.id, child_id=child.id, order=2)
    forest.add_action(action)

    current = child
    for order, q in ((3, 4.5), (4, 4.8), (5, 4.9)):
        current = forest.get_anchor(_chain(forest, current.id, q, order))
    _chain(forest, current.id, 100.0, 6)

    settled = settle_pending_actions(forest, now_order=6)

    assert [item.id for item in settled] == [action.id]
    assert action.result == 0
    assert action.window_best_q == pytest.approx(4.9)


def test_posterior_mixes_parent_chain_distance_and_recency_discounts() -> None:
    forest = Forest(maximize=True)
    root = forest.add_root(program_id=_program(forest, "root", 1.0, 1).id, order=1)
    a1 = forest.add_child(
        parent_id=root.id,
        program_id=_program(forest, "a1", 1.2, 2).id,
        action_id=forest.next_action_id(),
        order=2,
    )
    a2 = forest.add_child(
        parent_id=a1.id,
        program_id=_program(forest, "a2", 0.8, 3).id,
        action_id=forest.next_action_id(),
        order=3,
    )
    root_refine = _settle(
        _action(100, anchor_id=root.id, intent="refine", order=2, outcome=Outcome.IMPROVE),
        result=1,
        order=2,
    )
    forest.add_action(root_refine)
    a1_explore = _settle(
        _action(101, anchor_id=a1.id, intent="explore", order=3),
        result=0,
        order=3,
    )
    forest.add_action(a1_explore)

    assert parent_chain_window(forest, a2.id) == (a2.id, a1.id, root.id)

    refine = discounted_posterior(forest, a2.id, Intent.REFINE, now_order=12)
    explore = discounted_posterior(forest, a2.id, Intent.EXPLORE, now_order=12)

    # The root refine success sits at parent-chain distance 2 and is 10
    # responses old; the a1 explore failure sits at distance 1 and is 9 old.
    assert refine.a_post == pytest.approx(1.4 + 0.5 * 2.0 ** (-10 / 20))
    assert refine.b_post == pytest.approx(0.6)
    assert refine.n_settled == 1
    assert explore.a_post == pytest.approx(0.6)
    assert explore.b_post == pytest.approx(1.4 + 2.0 ** (-1 / 2) * 2.0 ** (-9 / 20))
    assert explore.n_settled == 1


def test_parent_chain_window_cuts_beyond_four_ancestors() -> None:
    forest = Forest(maximize=True)
    current = forest.add_root(program_id=_program(forest, "root", 1.0, 1).id, order=1)
    ids = [current.id]
    for index in range(5):
        current = forest.add_child(
            parent_id=current.id,
            program_id=_program(forest, f"a{index}", 1.0, index + 2).id,
            action_id=forest.next_action_id(),
            order=index + 2,
        )
        ids.append(current.id)
    root, a1, a5 = ids[0], ids[1], ids[5]
    distant = _settle(
        _action(100, anchor_id=root, intent="refine", order=2, outcome=Outcome.IMPROVE),
        result=1,
        order=2,
    )
    forest.add_action(distant)
    near = _settle(
        _action(101, anchor_id=a1, intent="refine", order=3, outcome=Outcome.IMPROVE),
        result=1,
        order=3,
    )
    forest.add_action(near)

    assert parent_chain_window(forest, a5) == tuple(reversed(ids[-(4 + 1) :]))

    posterior = discounted_posterior(forest, a5, Intent.REFINE, now_order=10)
    # The root action is five generations away and outside the window; the
    # a1 action sits exactly at the distance-four boundary.
    assert posterior.n_settled == 1
    assert posterior.a_post == pytest.approx(1.4 + 2.0 ** (-4 / 2) * 2.0 ** (-7 / 20))
    assert posterior.b_post == pytest.approx(0.6)


def test_pending_actions_never_enter_the_posterior() -> None:
    forest = Forest(maximize=True)
    root = forest.add_root(program_id=_program(forest, "root", 1.0, 1).id, order=1)
    child = forest.add_child(
        parent_id=root.id,
        program_id=_program(forest, "child", 0.5, 2).id,
        action_id=forest.next_action_id(),
        order=2,
    )
    forest.add_action(_action(100, anchor_id=root.id, child_id=child.id, order=2))
    settle_pending_actions(forest, now_order=2)

    posterior = discounted_posterior(forest, child.id, Intent.REFINE, now_order=9)

    assert posterior.n_settled == 0
    assert (posterior.a_post, posterior.b_post) == prior_counts(Intent.REFINE)


def test_fresh_anchor_arm_keeps_its_own_intent_prior() -> None:
    forest = Forest(maximize=True)
    root = forest.add_root(program_id=_program(forest, "root", 1.0, 1).id, order=1)
    child = forest.add_child(
        parent_id=root.id,
        program_id=_program(forest, "child", 0.5, 2).id,
        action_id=forest.next_action_id(),
        order=2,
    )
    forest.add_action(
        _settle(_action(100, anchor_id=root.id, intent="refine", order=2), result=0, order=2)
    )

    refine = discounted_posterior(forest, child.id, Intent.REFINE, now_order=6)
    explore = discounted_posterior(forest, child.id, Intent.EXPLORE, now_order=6)

    # The parent's refine failure leaks into the child's refine arm at
    # distance-decayed weight, but the untried explore arm stays at its prior.
    assert refine.n_settled == 1
    assert refine.b_post > prior_counts(Intent.REFINE)[1]
    assert (explore.a_post, explore.b_post) == prior_counts(Intent.EXPLORE)


def test_thompson_table_normalizes_and_repeats_identically() -> None:
    forest = Forest(maximize=True)
    forest.add_root(program_id=_program(forest, "weak", 1.0, 1).id, order=1)
    forest.add_root(program_id=_program(forest, "strong", 9.0, 2).id, order=2)

    first = select(forest, seed=13, order=9)
    second = select(forest, seed=13, order=9)

    assert first.anchor_id == second.anchor_id
    assert first.intent is second.intent
    assert len(first.arms) == 4
    assert math.isclose(sum(item.omega for item in first.arms), 1.0)
    for item in first.arms:
        assert (item.a_post, item.b_post) == prior_counts(item.intent)
        assert item.theta > 0.0

    payload = first.to_dict()
    assert payload["selected_arm"]["anchor_id"] == first.anchor_id
    assert payload["selected_arm"]["intent"] == first.intent.value
    diagnostics = payload["diagnostics"]
    assert diagnostics["n_anchors"] == 2
    assert diagnostics["n_arms"] == 4
    assert diagnostics["refine_share"] == pytest.approx(1.0 - diagnostics["explore_share"])
    assert diagnostics["effective_anchors"] > 0.0
    assert diagnostics["selected_n_selected"] == 0


def test_smoke_run_settles_every_improving_action_and_streams_facts(
    tmp_path: Path,
) -> None:
    artifacts = RunArtifacts(tmp_path, console_output=False)
    method = TraceAADV910(
        llm=ScriptedLLM(),
        evaluation=IncreasingEvaluation(),
        artifacts=artifacts,
        budget=20,
        checkpoint_dir=tmp_path / "checkpoints",
        seed=3,
    )
    method.run()

    summary = json.loads((tmp_path / "logs" / "summary.json").read_text())
    assert summary["status"] == "finished"
    assert summary["evaluator_call_count"] == 20
    assert summary["n_roots"] == 8
    assert summary["n_actions"] == 12
    assert summary["n_settled_success"] == 12
    assert summary["n_pending"] == 0
    assert summary["n_settled_failure"] == 0
    assert (tmp_path / "checkpoints" / "latest.json").is_file()

    events = [
        json.loads(line)
        for line in (tmp_path / "logs" / "events.jsonl")
        .read_text()
        .splitlines()
    ]
    requests = [item for item in events if item["event"] == "request_prepared"]
    settlements = [item for item in events if item["event"] == "action_settled"]
    assert len(settlements) == 12
    assert {item["settled_result"] for item in settlements} == {1}
    assert len({item["action_id"] for item in settlements}) == 12
    search_requests = [
        item for item in requests if item.get("selection") is not None
    ]
    assert len(search_requests) == 12
    arm = search_requests[0]["selection"]["selected_arm"]
    assert {"theta", "a_post", "b_post", "omega"} <= set(arm)
    assert search_requests[0]["selection"]["diagnostics"]["n_arms"] == 16

    with (tmp_path / "evaluations.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 20
    search_rows = [row for row in rows if row["stage"] == "search"]
    assert len(search_rows) == 12
    assert {row["action_status"] for row in search_rows} == {"settled"}
    assert {row["action_result"] for row in search_rows} == {"1"}
    assert {row["operator"] for row in search_rows} <= {"refine", "explore"}
    assert all(row["stage"] != "bootstrap" for row in rows)


class InvalidSearchLLM(LLM):
    """Root prompts get valid programs; search prompts never parse."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def draw_sample(self, prompt, *args, **kwargs):
        self.calls += 1
        if "[Current Algorithm]" in prompt:
            return "Idea: broken\nCode:\n```text\nnot a program\n```"
        return (
            f"Idea: root {self.calls}\n"
            "Code:\n```python\n"
            "def choose(value: int) -> int:\n"
            f"    return value + {self.calls}\n"
            "```"
        )

    def count_tokens(self, text: str) -> int:
        return len(text) // 4 + 1


def test_response_safety_limit_marks_resumable_protocol_failure(
    tmp_path: Path,
) -> None:
    artifacts = RunArtifacts(tmp_path, console_output=False)
    method = TraceAADV910(
        llm=InvalidSearchLLM(),
        evaluation=IncreasingEvaluation(),
        artifacts=artifacts,
        budget=20,
        n_roots=2,
        max_responses=6,
        checkpoint_dir=tmp_path / "checkpoints",
        seed=1,
    )
    method.run()

    summary = json.loads((tmp_path / "logs" / "summary.json").read_text())
    assert summary["status"] == "aborted"
    assert summary["stop_reason"] == "response_safety_limit"
    assert summary["evaluator_call_count"] == 6
    assert summary["n_actions"] == 4
    assert summary["n_settled_failure"] == 4
    assert summary["n_pending"] == 0
    assert all(item.result == 0 for item in method._forest.actions())


def test_runner_builds_v910_and_records_config(tmp_path: Path) -> None:
    spec = run.make_run_spec(
        task="tsp_construct",
        version="v9_10",
        backend="server3",
        budget=1000,
        run_name="v910",
        experiments_root=tmp_path,
    )
    method = run.build_method(spec, tmp_path / "run")
    assert isinstance(method, TraceAADV910)
    assert spec.n_init == 8
    assert method._n_roots == 8
    assert not hasattr(method, "_context_limit")
    run_dir, run_name, _ = run.resolve_run_dir(spec)
    run.write_run_config(spec, run_dir, run_name)
    payload = json.loads((run_dir / "run_config.json").read_text())
    assert payload["method"] == "traceaad_v9_10"
    assert payload["method_params"] == run._v910_method_params(spec)
    assert payload["generator_environment"]["logical_model_name"] == "Qwen3.6-27B"
    assert "implementation" not in payload
    assert "backend" not in payload and "llm" not in payload
    method._llm.close()


def test_checkpoint_round_trip_preserves_actions_and_pending_resume(
    tmp_path: Path,
) -> None:
    source_llm = ScriptedLLM()
    source = TraceAADV910(
        llm=source_llm,
        evaluation=IncreasingEvaluation(),
        budget=20,
        n_roots=1,
        checkpoint_dir=tmp_path / "source",
    )
    program = source._forest.add_program(
        code="def choose(value: int) -> int:\n    return value\n",
        fitness=1.0,
        order=1,
    )
    anchor = source._forest.add_root(program_id=program.id, order=1)
    source._initialization_complete = True
    source._n_candidates = 1
    source._n_eval = 1
    anchor.n_refine = 1
    source._pending = Pending(
        id=source._forest.next_action_id(),
        response_id="v910-r000001-a000000",
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
    save_checkpoint(source)

    resumed_llm = ScriptedLLM()
    resumed_evaluation = IncreasingEvaluation()
    resumed_evaluation.calls = 5
    resumed = TraceAADV910(
        llm=resumed_llm,
        evaluation=resumed_evaluation,
        budget=20,
        n_roots=1,
        checkpoint_dir=tmp_path / "resumed",
    )
    load_checkpoint(resumed, tmp_path / "source" / "latest.json")
    resumed._resume_pending()

    assert resumed_llm.calls == 0
    assert resumed._pending is None
    assert resumed._n_candidates == 1
    assert resumed._n_eval == 2
    assert resumed._forest.get_anchor(anchor.id).n_refine == 1
    actions = resumed._forest.actions()
    assert len(actions) == 1
    assert actions[0].status is ActionStatus.SETTLED
    assert actions[0].result == 1


class CodeValueEvaluation(Evaluation):
    """Fitness read from the generated code, so evaluation is replay-safe."""

    def __init__(self) -> None:
        super().__init__(
            template_program=TEMPLATE,
            task_description="Improve choose.",
            use_numba_accelerate=False,
            safe_evaluate=False,
            timeout_seconds=10,
        )

    def evaluate_program(self, program_str, callable_func, **kwargs):
        import re

        match = re.search(r"return value \+ (\d+)", program_str)
        return float(int(match.group(1)))


def test_interrupted_resume_reproduces_uninterrupted_trajectory(
    tmp_path: Path,
) -> None:
    common = {
        "evaluation": CodeValueEvaluation(),
        "budget": 24,
        "seed": 9,
    }
    uninterrupted = TraceAADV910(
        llm=ScriptedLLM(),
        artifacts=RunArtifacts(tmp_path / "a", console_output=False),
        checkpoint_dir=tmp_path / "a" / "ck",
        **common,
    )
    uninterrupted.run()

    interrupted = TraceAADV910(
        llm=ScriptedLLM(),
        artifacts=RunArtifacts(tmp_path / "b", console_output=False),
        checkpoint_dir=tmp_path / "b" / "ck",
        **common,
    )
    interrupted._initialize()
    while interrupted._n_eval < 14:
        choice = select(interrupted._forest, seed=9, order=interrupted._n_candidates + 1)
        interrupted._request(
            interrupted._prompt(choice.anchor_id, choice.intent),
            anchor_id=choice.anchor_id,
            stage="search",
            iteration=interrupted._iteration,
            intent=choice.intent.value,
            selection=choice.to_dict(),
        )
    stopped_at = interrupted._n_candidates

    resumed = TraceAADV910(
        llm=ScriptedLLM(start=stopped_at),
        checkpoint_dir=tmp_path / "b" / "ck",
        resume_from=tmp_path / "b" / "ck" / "latest.json",
        **common,
    )
    resumed.run()

    assert resumed._forest.to_dict() == uninterrupted._forest.to_dict()
    assert resumed._n_eval == uninterrupted._n_eval == 24
    assert resumed.best == uninterrupted.best


def test_formal_launcher_resumes_existing_incomplete_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "existing"
    run_dir.mkdir()
    item = launch_v910.Item(
        task="tsp_constructive",
        repeat=1,
        seed=0,
        session="v910_resume_test",
        run_name="existing",
        run_dir=run_dir,
    )
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(launch_v910.subprocess, "run", fake_run)
    launch_v910.launch(item, backend="server3", dry_run=False)

    command = calls[0]
    assert "--resume-from" in command
    assert str(run_dir) in command
    assert "--run-name" not in command
    assert "v9_10" in command
