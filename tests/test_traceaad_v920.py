from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.runners.traceaad_v9_20 import run as runner
from llm4ad.method.traceaad_v9_20 import RunArtifacts, TraceAADV920
from llm4ad.method.traceaad_v9_20.schema import Action, Algorithm
from llm4ad.method.traceaad_v9_20.schema import Pending
from llm4ad.method.traceaad_v9_20.selection import (
    allocation_stats,
    continuation_value,
    decide_action,
    uncertainty_value,
)
from llm4ad.method.traceaad_v9_20.landscape import Landscape, region_statistics
from llm4ad.method.traceaad_v9_20.prompt import build_generation_prompt, parse_program_response
from llm4ad.method.traceaad_v9_20.tree import Tree
from tests.test_traceaad_v919_mechanism import BlendLLM, FastEvaluation


def _landscape() -> tuple[Tree, Landscape]:
    tree = Tree()
    landscape = Landscape(task="tsp_construct", protocol={"protocol_id": "test"})
    for index, fitness in enumerate((1.0, 2.0, 3.0), start=1):
        node = tree.add_algorithm(code=f"code-{index}", fitness=fitness)
        landscape.add(node.id, [[[0], [index]]])
    return tree, landscape


def test_allocation_mixes_quality_and_coverage() -> None:
    tree, landscape = _landscape()
    quality = {node.id: tree.quality(node) for node in tree.valid_algorithms()}
    stats = region_statistics(
        landscape=landscape,
        quality=quality,
        opportunities={1: 8, 2: 0, 3: 0},
    )
    allocation = allocation_stats(tree, stats)
    assert sum(allocation.probabilities.values()) == 1.0
    assert allocation.probabilities[2] > 0.0
    assert allocation.probabilities[3] > 0.0
    assert allocation.coverage[2] >= allocation.coverage[1]


def test_continuation_and_uncertainty_have_expected_limits() -> None:
    fresh = Algorithm(1, "x", 1.0, 0)
    tried = Algorithm(2, "y", 1.0, 0, opportunities=3, improvements=2, failures=1)
    assert continuation_value(fresh) == 0.5
    assert continuation_value(tried) == 3.0 / 5.0
    assert uncertainty_value(fresh) > uncertainty_value(tried)


def test_action_excludes_crossover_without_reference() -> None:
    decision = decide_action(
        algorithm=Algorithm(1, "x", 1.0, 0),
        reference_value=None,
        rng=__import__("random").Random(0),
    )
    assert Action.CROSSOVER.value not in decision.probabilities
    assert abs(sum(decision.probabilities.values()) - 1.0) < 1e-12


def test_action_prompt_contracts_are_distinct() -> None:
    common = dict(
        task_description="task",
        code="def f(x): return x",
        fitness=1.0,
        history_text="FULL FORMATION HISTORY",
        action=Action.EXPLORE,
        context_mode="explore",
        attempt_summary="Direct attempts: 2",
    )
    prompt = build_generation_prompt(**common)
    assert "Direct attempts: 2" in prompt
    assert "FULL FORMATION HISTORY" not in prompt
    crossover = build_generation_prompt(
        **{**common, "action": Action.CROSSOVER, "context_mode": "crossover", "reference_code": "def g(x): return x", "reference_fitness": 2.0}
    )
    assert "Crossover Reference Algorithm" in crossover
    assert "def g(x)" in crossover


def test_v920_short_integration_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    method = TraceAADV920(
        BlendLLM(),
        FastEvaluation(),
        RunArtifacts(run_dir, console_output=False),
        budget=10,
        n_roots=8,
        checkpoint_dir=run_dir / "checkpoints",
        seed=0,
    )
    method.run()
    assert method._n_eval == 10
    assert method._n_calls == 10
    assert len(method._tree.valid_algorithms()) == 10
    summary = json.loads((run_dir / "logs" / "summary.json").read_text())
    assert summary["status"] == "finished"
    assert summary["allocation_policy"] == "quality_continuation_plus_behavior_coverage"
    resumed = TraceAADV920(
        BlendLLM(),
        FastEvaluation(),
        budget=10,
        n_roots=8,
        checkpoint_dir=tmp_path / "resumed" / "checkpoints",
        resume_from=run_dir / "checkpoints" / "latest.json",
    )
    assert resumed._n_eval == method._n_eval
    assert np.allclose(resumed._landscape.matrix, method._landscape.matrix)


def test_repair_replaces_direct_credit_in_v920(tmp_path: Path) -> None:
    method = TraceAADV920(
        BlendLLM(), FastEvaluation(), budget=2, n_roots=1, checkpoint_dir=tmp_path / "cp"
    )
    root = method._tree.add_algorithm(code="root", fitness=1.0)
    parsed = parse_program_response(
        "Idea: repaired\nCode:\n```python\ndef f(x): return x\n```"
    )
    method._pending = Pending(root.id, "develop", "", "")
    method._finish(
        attempt=1,
        outcome="invalid",
        status="error",
        fitness=None,
        child=None,
        error="bad",
        error_type="InvalidEvaluationResult",
        elapsed=0.0,
        parsed=parsed,
        continuing=True,
    )
    method._pending = Pending(root.id, "develop", "", "")
    method._finish(
        attempt=2,
        outcome="improve",
        status="ok",
        fitness=2.0,
        child=None,
        error=None,
        error_type=None,
        elapsed=0.0,
        parsed=parsed,
        continuing=False,
    )
    assert (root.opportunities, root.improvements, root.failures) == (1, 1, 0)


def test_duplicate_still_consumes_primary_evaluator_slot(tmp_path: Path) -> None:
    llm = BlendLLM()
    original = llm.draw_sample

    def same(prompt, *args, **kwargs):
        if not hasattr(llm, "_same"):
            llm._same = original(prompt, *args, **kwargs)
        return llm._same

    llm.draw_sample = same  # type: ignore[method-assign]
    method = TraceAADV920(
        llm,
        FastEvaluation(),
        RunArtifacts(tmp_path / "run", console_output=False),
        budget=9,
        n_roots=8,
        checkpoint_dir=tmp_path / "run" / "checkpoints",
    )
    method.run()
    assert method._n_eval == 9
    assert method._n_calls == 9
    assert len(method._tree.valid_algorithms()) == 1
    assert method._outcome_counts["duplicate"] == 8


def test_runner_builds_v920_without_running() -> None:
    spec = runner.make_run_spec(task="tsp_construct", backend="local")
    assert spec.method_name == "traceaad_v9_20"
    assert spec.n_init == 8
    assert runner._method_params(spec)["actions"] == ["develop", "explore", "crossover"]
