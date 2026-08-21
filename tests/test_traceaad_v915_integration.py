"""Core mechanism tests for TraceAAD V9.15."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad_v9_15 import RunArtifacts, TraceAADV915
from llm4ad.method.traceaad_v9_15.prompt import preflight_code
from llm4ad.method.traceaad_v9_15_eh import TraceAADV915EH
from llm4ad.method.traceaad_v9_15.checkpoint import save_checkpoint
from llm4ad.method.traceaad_v9_15.selection import (
    boltzmann_probabilities,
    decide,
    effective_sample_size,
    explore_probability,
    formation_gains,
    protection_bonus,
    selection_score,
    solve_beta,
    target_ess,
    trajectory_bonus,
)
from llm4ad.method.traceaad_v9_15.schema import Intent, Pending
from llm4ad.method.traceaad_v9_15.tree import Tree

TEMPLATE = """def choose(value: int) -> int:
    return value
"""


class ScriptedLLM(LLM):
    """Returns programs whose fitness follows an explicit value script."""

    def __init__(
        self, values: list[float], *, invalid_at: set[int] | None = None
    ) -> None:
        super().__init__()
        self.values = values
        self.calls = 0
        self.invalid_at = invalid_at or set()

    def draw_sample(self, prompt: str, *args, **kwargs) -> str:
        self.calls += 1
        if self.calls in self.invalid_at:
            return "unusable response"
        value = self.values[self.calls - 1]
        return (
            f"Idea: candidate {self.calls}\n"
            "Code:\n```python\n"
            "def choose(value: int) -> int:\n"
            f"    return {value}\n"
            "```"
        )


class IncrementalEvaluation(Evaluation):
    def __init__(self) -> None:
        super().__init__(
            template_program=TEMPLATE,
            task_description="Improve choose.",
            safe_evaluate=False,
            timeout_seconds=5,
        )

    def evaluate_program(self, program_str: str, callable_func, **kwargs) -> float | None:
        return None if callable_func is None else float(callable_func(10))


def _pool() -> tuple[Tree, object, object]:
    """root(10) -> explore child(8), refine child(11); scale becomes 1.0."""
    tree = Tree(maximize=True)
    root = tree.add_algorithm(code="r", fitness=10.0)
    explore_child = tree.add_algorithm(
        code="e", fitness=8.0, parent_id=root.id, created_by=Intent.EXPLORE.value
    )
    tree.add_algorithm(code="s", fitness=11.0, parent_id=root.id, created_by=Intent.REFINE.value)
    return tree, root, explore_child


def test_explore_probability_schedule() -> None:
    assert explore_probability(0) == pytest.approx(0.20)
    assert explore_probability(50) == pytest.approx(0.35)
    assert explore_probability(1_000_000) == pytest.approx(0.50, abs=1e-3)
    values = [explore_probability(n) for n in range(0, 1000, 50)]
    assert values == sorted(values)
    assert all(0.20 <= value <= 0.50 for value in values)


def test_tree_derives_depth_gains_scale_and_round_trips() -> None:
    tree = Tree(maximize=False)
    root = tree.add_algorithm(code="r", fitness=10.0)
    child = tree.add_algorithm(
        code="c", fitness=8.0, parent_id=root.id, created_by=Intent.EXPLORE.value
    )
    child.refine_count = 2
    child.explore_count = 1

    assert tree.depth(root.id) == 1
    assert tree.depth(child.id) == 2
    assert tree.formation_gain(child) == pytest.approx(-8.0 - (-10.0))
    assert tree.formation_gain(root) is None
    assert tree.positive_gain_scale() == pytest.approx(2.0)
    assert tree.best_quality() == pytest.approx(-8.0)

    restored = Tree.from_dict(tree.to_dict())
    assert restored.to_dict() == tree.to_dict()
    assert restored.get_algorithm(child.id).refine_count == 2
    assert restored.get_algorithm(child.id).created_by == "explore"


def test_formation_gains_window_skips_virtual_root_and_caps_at_six() -> None:
    tree = Tree(maximize=True)
    parent = tree.add_algorithm(code="r", fitness=0.0)
    node = parent
    for step in range(1, 9):
        node = tree.add_algorithm(
            code=f"n{step}", fitness=float(step**2), parent_id=node.id
        )

    assert tree.depth(node.id) == 9
    gains = formation_gains(tree, node)
    assert len(gains) == 6
    # fitness squares give per-step gains 2i-1; the window keeps the last six
    assert gains == (5.0, 7.0, 9.0, 11.0, 13.0, 15.0)
    assert formation_gains(tree, parent) == ()


def test_protection_bonus_bounds_decay_and_eligibility() -> None:
    tree, root, explore_child = _pool()
    scale = tree.positive_gain_scale()
    assert scale == pytest.approx(1.0)

    # gap 2, cap 2*s_t = 2, gamma(0) = 1
    assert protection_bonus(
        tree, explore_child, intent=Intent.REFINE, scale=scale
    ) == pytest.approx(2.0)
    explore_child.refine_count = 1
    assert protection_bonus(
        tree, explore_child, intent=Intent.REFINE, scale=scale
    ) == pytest.approx(1.0)
    explore_child.refine_count = 2
    assert protection_bonus(
        tree, explore_child, intent=Intent.REFINE, scale=scale
    ) == pytest.approx(2.0 / 3.0)

    # not under Explore intent, not Explore-born, or no scale evidence
    assert (
        protection_bonus(tree, explore_child, intent=Intent.EXPLORE, scale=scale)
        == 0.0
    )
    assert protection_bonus(tree, root, intent=Intent.REFINE, scale=scale) == 0.0
    assert protection_bonus(tree, explore_child, intent=Intent.REFINE, scale=None) == 0.0

    # gap above the cap is clipped to 2*s_t before decay
    broken = tree.add_algorithm(
        code="b", fitness=5.0, parent_id=root.id, created_by=Intent.EXPLORE.value
    )
    assert protection_bonus(
        tree, broken, intent=Intent.REFINE, scale=scale
    ) == pytest.approx(2.0)

    # Explore-born child at or above its parent gets no grace
    lifted = tree.add_algorithm(
        code="l", fitness=12.0, parent_id=root.id, created_by=Intent.EXPLORE.value
    )
    assert protection_bonus(tree, lifted, intent=Intent.REFINE, scale=scale) == 0.0


def test_trajectory_bonus_matches_the_product_formula() -> None:
    tree = Tree(maximize=True)
    rising = tree.add_algorithm(code="r", fitness=10.0)
    mid = tree.add_algorithm(code="m", fitness=11.0, parent_id=rising.id)
    top = tree.add_algorithm(code="t", fitness=13.0, parent_id=mid.id)

    scale = tree.positive_gain_scale()
    assert scale == pytest.approx(1.5)

    # mid: gains (1.0), f_succ 1, mean+ 1.0, headroom 13-11=2
    assert trajectory_bonus(tree, mid, scale=scale) == pytest.approx(
        1.0 * 1.0 * 2.0 / 3.5
    )
    # top is the best: headroom zero kills the bonus despite the rising path
    assert trajectory_bonus(tree, top, scale=scale) == 0.0
    # roots have no evaluated formation step
    assert trajectory_bonus(tree, rising, scale=scale) == 0.0

    # weak self-grinding far below the frontier: multiplier ~1, product tiny
    base = tree.add_algorithm(code="b", fitness=0.0)
    weak = tree.add_algorithm(code="w", fitness=0.1, parent_id=base.id)
    weak_scale = tree.positive_gain_scale()
    assert weak_scale == pytest.approx(1.0)  # median of {1, 2, 0.1}
    assert trajectory_bonus(tree, weak, scale=weak_scale) == pytest.approx(
        0.1 * 12.9 / (12.9 + 1.0)
    )


def test_selection_score_combines_the_three_terms() -> None:
    tree, _, explore_child = _pool()
    scale = tree.positive_gain_scale()
    score = selection_score(tree, explore_child, intent=Intent.REFINE, scale=scale)
    expected = (
        tree.quality(explore_child)
        + protection_bonus(tree, explore_child, intent=Intent.REFINE, scale=scale)
        + trajectory_bonus(tree, explore_child, scale=scale)
    )
    assert score == pytest.approx(expected)
    assert score == pytest.approx(8.0 + 2.0)


def test_boltzmann_ess_and_beta_solver() -> None:
    scores = [1.0, 2.0, 3.0, 4.0, 10.0, 10.5, 11.0, 30.0, 31.0, 100.0]
    uniform = boltzmann_probabilities(0.0, scores)
    assert effective_sample_size(uniform) == pytest.approx(len(scores))

    target = target_ess(len(scores))
    assert target == pytest.approx(2.0)  # floor of two dominates up to 20 nodes
    assert target_ess(100) == pytest.approx(10.0)
    beta = solve_beta(scores, target)
    ess = effective_sample_size(boltzmann_probabilities(beta, scores))
    assert ess == pytest.approx(target, rel=1e-3)

    # pools below ten algorithms clamp the target to the floor of two
    assert target_ess(8) == pytest.approx(2.0)
    beta_small = solve_beta(scores[:8], target_ess(8))
    ess_small = effective_sample_size(
        boltzmann_probabilities(beta_small, scores[:8])
    )
    assert ess_small == pytest.approx(2.0, rel=1e-3)

    # degenerate all-equal scores stay uniform at any beta without crashing
    equal = solve_beta([5.0] * 4, 2.0)
    assert effective_sample_size(boltzmann_probabilities(equal, [5.0] * 4)) == pytest.approx(4.0)


def test_decide_is_deterministic_and_reports_diagnostics() -> None:
    tree, _, _ = _pool()
    first = decide(tree, n_stag=0, seed=0, n_eval=5)
    second = decide(tree, n_stag=0, seed=0, n_eval=5)
    assert first.parent.id == second.parent.id
    assert first.intent == second.intent
    assert first.beta == second.beta
    assert first.n_valid == 3
    assert first.parent_q == pytest.approx(tree.quality(first.parent))
    assert 0.0 < first.ess <= 3.0


def test_search_loop_updates_counters_stagnation_and_artifacts(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    method = TraceAADV915(
        llm=ScriptedLLM([10.0, 12.0, 11.0, 14.0, 13.0, 20.0]),
        evaluation=IncrementalEvaluation(),
        artifacts=RunArtifacts(run_dir, console_output=False),
        budget=6,
        n_roots=2,
        checkpoint_dir=run_dir / "checkpoints",
    )
    method.run()

    assert method._n_eval == 6
    assert len(method._tree.root_algorithms()) == 2
    assert len(method._tree.valid_algorithms()) == 6
    assert method.best.fitness == 20.0
    assert method._n_stag == 0

    loop_children = [
        algorithm
        for algorithm in method._tree.valid_algorithms()
        if algorithm.parent_id != method._tree.virtual_root_id
    ]
    assert len(loop_children) == 4
    for algorithm in method._tree.valid_algorithms():
        assert algorithm.count == algorithm.refine_count + algorithm.explore_count
    assert sum(a.count for a in method._tree.valid_algorithms()) == 4
    for child in loop_children:
        assert child.created_by in {Intent.REFINE.value, Intent.EXPLORE.value}

    with (run_dir / "evaluations.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    # initialization is reset to zero stagnation; loop row diagnostics present
    assert rows[2]["n_stag"] == "1"  # fitness 11 below best 12
    assert rows[3]["n_stag"] == "0"  # fitness 14 is a new best
    assert rows[4]["n_stag"] == "1"  # fitness 13 below best 14
    assert rows[5]["n_stag"] == "0"  # fitness 20 is a new best
    for row in rows[2:]:
        assert row["beta"] and row["ess"] and row["p_explore"]
        assert int(row["n_valid"]) == int(row["eval_count"]) - 1
        child = method._tree.get_algorithm(int(row["child_id"]))
        assert row["intent"] == child.created_by
    for row in rows[:2]:
        assert row["beta"] == ""  # initialization carries no decision

    summary = json.loads((run_dir / "logs/summary.json").read_text())
    assert summary["status"] == "finished"
    assert summary["best_score"] == 20.0
    assert summary["evaluator_call_count"] == 6
    assert (run_dir / "best_program.py").is_file()


def test_failed_evaluation_consumes_budget_counts_and_stagnation(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "failed"
    method = TraceAADV915(
        llm=ScriptedLLM([10.0, 99.0], invalid_at={2}),
        evaluation=IncrementalEvaluation(),
        artifacts=RunArtifacts(run_dir, console_output=False),
        budget=2,
        n_roots=1,
    )
    method.run()

    root = method._tree.root_algorithms()[0]
    assert method._n_eval == 2
    assert root.count == 1
    assert root.refine_count + root.explore_count == 1
    assert len(method._tree.valid_algorithms()) == 1
    assert method._n_stag == 1
    with (run_dir / "evaluations.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["status"] == "ok"
    assert rows[1]["status"] == "exec_error"
    assert rows[1]["n_stag"] == "1"


def test_preflight_checks_syntax_and_target_function() -> None:
    assert preflight_code("def choose(value):\n    return value\n", "choose") is None
    assert preflight_code("def choose(value):\n    return value\n", "other")
    assert preflight_code("def choose(value)\n    return value\n", "choose")


def test_error_handling_repairs_once_and_logs_each_real_evaluation(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "error_handling"
    method = TraceAADV915EH(
        llm=ScriptedLLM([10.0, 11.0, 12.0], invalid_at={2}),
        evaluation=IncrementalEvaluation(),
        artifacts=RunArtifacts(run_dir, console_output=False),
        budget=3,
        n_roots=1,
    )
    method.run()

    assert method._n_eval == 3
    assert method._llm.calls == 3
    assert method.best.fitness == 12.0
    with (run_dir / "evaluations.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["attempt_kind"] for row in rows] == ["initial", "initial", "repair"]
    assert rows[1]["status"] == "exec_error"
    assert rows[1]["error_type"] == "SyntaxError"
    assert rows[2]["status"] == "ok"
    assert rows[2]["attempt"] == "2"
    assert all(row["candidate_hash"] for row in rows)
    assert all(float(row["elapsed_seconds"]) >= 0.0 for row in rows)


def test_checkpoint_round_trips_stagnation_and_counters(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    method = TraceAADV915(
        llm=ScriptedLLM([10.0]),
        evaluation=IncrementalEvaluation(),
        budget=3,
        n_roots=1,
        checkpoint_dir=checkpoint_dir,
    )
    root = method._tree.add_algorithm(code=TEMPLATE, fitness=10.0)
    method._n_eval = 1
    method._n_stag = 5
    method._pending = Pending(
        parent_id=root.id,
        intent=Intent.REFINE.value,
        response=(
            "Idea: recovered step\nCode:\n```python\n"
            "def choose(value: int) -> int:\n    return 19\n```"
        ),
    )
    path = save_checkpoint(method)
    state = json.loads(path.read_text())
    assert set(state) == {"tree", "pending", "n_eval", "n_stag"}
    assert state["n_stag"] == 5

    resumed = TraceAADV915(
        llm=ScriptedLLM([10.0, 21.0]),
        evaluation=IncrementalEvaluation(),
        budget=3,
        n_roots=1,
        resume_from=path,
    )
    assert resumed._n_stag == 5
    resumed.run()

    assert resumed._n_eval == 3
    assert resumed._pending is None
    assert resumed._tree.get_algorithm(2).idea == "recovered step"
    assert sum(a.count for a in resumed._tree.valid_algorithms()) == 2
    for algorithm in resumed._tree.valid_algorithms():
        assert algorithm.count == algorithm.refine_count + algorithm.explore_count


def test_resume_preserves_stagnation_counter(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    method = TraceAADV915(
        llm=ScriptedLLM([10.0]),
        evaluation=IncrementalEvaluation(),
        budget=1,
        n_roots=1,
        checkpoint_dir=checkpoint_dir,
    )
    method._tree.add_algorithm(code=TEMPLATE, fitness=10.0)
    method._n_eval = 1
    method._n_stag = 5
    path = save_checkpoint(method)

    resumed = TraceAADV915(
        llm=ScriptedLLM([8.0, 9.0]),
        evaluation=IncrementalEvaluation(),
        budget=3,
        n_roots=1,
        resume_from=path,
    )
    resumed.run()

    # two non-best children continue the checkpointed counter: 5 -> 6 -> 7
    assert resumed._n_eval == 3
    assert resumed._n_stag == 7


def test_interrupted_run_resumes_to_identical_trajectory(tmp_path: Path) -> None:
    values = [10.0, 12.0, 11.0, 14.0, 13.0, 20.0]
    full_dir = tmp_path / "full"
    full = TraceAADV915(
        llm=ScriptedLLM(values),
        evaluation=IncrementalEvaluation(),
        artifacts=RunArtifacts(full_dir, console_output=False),
        budget=6,
        n_roots=2,
        checkpoint_dir=full_dir / "checkpoints",
    )
    full.run()

    part_dir = tmp_path / "part"
    part = TraceAADV915(
        llm=ScriptedLLM(values),
        evaluation=IncrementalEvaluation(),
        artifacts=RunArtifacts(part_dir, console_output=False),
        budget=3,
        n_roots=2,
        checkpoint_dir=part_dir / "checkpoints",
    )
    part.run()

    resumed = TraceAADV915(
        llm=ScriptedLLM(values[3:]),
        evaluation=IncrementalEvaluation(),
        artifacts=RunArtifacts(part_dir, console_output=False),
        budget=6,
        n_roots=2,
        resume_from=part_dir / "checkpoints" / "latest.json",
    )
    resumed.run()

    def trajectory(run_dir: Path) -> list[tuple[str, ...]]:
        with (run_dir / "evaluations.csv").open() as handle:
            return [
                (
                    row["parent_id"],
                    row["child_id"],
                    row["intent"],
                    row["fitness"],
                    row["n_stag"],
                )
                for row in csv.DictReader(handle)
            ]

    assert resumed._n_eval == full._n_eval == 6
    assert trajectory(part_dir) == trajectory(full_dir)
    # idea strings are model output; every decision-relevant field matches
    stripped = lambda payload: [
        {key: value for key, value in item.items() if key != "idea"}
        for item in payload["algorithms"]
    ]
    assert stripped(resumed._tree.to_dict()) == stripped(full._tree.to_dict())


def test_model_transport_failure_is_immediately_visible(tmp_path: Path) -> None:
    class FailingLLM(LLM):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def draw_sample(self, prompt: str, *args, **kwargs) -> str:
            self.calls += 1
            raise ConnectionError("offline")

    llm = FailingLLM()
    method = TraceAADV915(
        llm=llm,
        evaluation=IncrementalEvaluation(),
        budget=2,
        n_roots=1,
    )
    with pytest.raises(ConnectionError, match="offline"):
        method.run()
    assert llm.calls == 1
    assert method._n_eval == 0
