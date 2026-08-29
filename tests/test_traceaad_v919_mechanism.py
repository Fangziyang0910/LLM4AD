"""Focused mechanism tests for TraceAAD V9.19."""

from __future__ import annotations

import csv
import json
import math
import random
import zlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from llm4ad.base import LLM, TextFunctionProgramConverter
from llm4ad.method.traceaad_v9_19 import (
    RunArtifacts,
    TraceAADV919,
    TrackedResult,
)
from llm4ad.method.traceaad_v9_19 import behave, landscape as landscape_module
from llm4ad.task.optimization.tsp_construct.evaluation import TSPEvaluation
from llm4ad.method.traceaad_v9_19.landscape import (
    Landscape,
    behavior_tag,
    midrank_percentile,
    neighborhood_size,
    region_statistics,
)
from llm4ad.method.traceaad_v9_19.history import render_path
from llm4ad.method.traceaad_v9_19.selection import (
    boltzmann_probabilities,
    decide_action,
    edge_value,
    effective_sample_size,
    explore_probability,
    node_score,
    solve_beta,
    t_response,
    trajectory_response,
    target_ess,
)
from llm4ad.method.traceaad_v9_19.prompt import (
    ACTION_INSTRUCTIONS,
    build_generation_prompt,
    build_root_prompt,
    parse_program_response,
)
from llm4ad.method.traceaad_v9_19.schema import Action
from llm4ad.method.traceaad_v9_19.tree import Tree, VIRTUAL_ROOT_ID

TSP_TEMPLATE = """import numpy as np
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    return int(unvisited_nodes[0])
"""

RESPONSE_TEMPLATE = """Idea: blend {tag}
Code:
```python
import numpy as np

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    d = distance_matrix[current_node][unvisited_nodes]
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    scores = -({a} * d + {b} * d_dest)
    return int(unvisited_nodes[int(np.argmax(scores))])
```
"""


class BlendLLM(LLM):
    """Returns one unique valid TSP heuristic per call."""

    def __init__(self, *, broken_first: int = 0) -> None:
        super().__init__()
        self.calls = 0
        self.broken_first = broken_first
        self.prompts: list[str] = []

    def draw_sample(self, prompt: str, *args, **kwargs) -> str:
        self.prompts.append(prompt)
        self.calls += 1
        if self.calls <= self.broken_first:
            return "Idea: broken\nCode:\n```python\ndef select_next_node():\n    return (\n```"
        a = 1.0 + 0.071 * self.calls
        b = 0.05 * self.calls
        return RESPONSE_TEMPLATE.format(tag=self.calls, a=a, b=b)


class FastEvaluation(TSPEvaluation):
    """A TSP-typed evaluation with a two-node scoring shortcut.

    Returns the benchmark-style fitness plus deterministic synthetic
    trajectories keyed by the program text, standing in for the tracked
    training evaluation.
    """

    def __init__(self) -> None:
        super().__init__(
            n_instance=2,
            problem_size=8,
            seed=0,
            safe_evaluate=False,
            timeout_seconds=10,
        )

    def evaluate_program(self, program_str: str, callable_func, **kwargs):
        if callable_func is None:
            return None
        distances = np.array([[0.0, 1.0, 3.0], [1.0, 0.0, 1.0], [3.0, 1.0, 0.0]])
        unvisited = np.array([1, 2])
        fitness = -float(callable_func(0, 0, unvisited, distances))
        return TrackedResult(fitness, synthetic_trajectories(program_str))


def synthetic_trajectories(program_str: str, *, seed_offset: int = 0) -> list:
    """Two instance trajectories, deterministic per program text."""
    import random as _random

    trajectories = []
    for instance_index in range(2):
        rng = _random.Random(
            f"{zlib.crc32(program_str.encode())}:{seed_offset}:{instance_index}"
        )
        order = list(range(8))
        rng.shuffle(order)
        states = [order[: k + 1] for k in range(8)]
        trajectories.append(states[:12])
    return trajectories


def _make_method(
    tmp_path: Path,
    llm: LLM,
    *,
    budget: int = 12,
    n_roots: int = 8,
) -> TraceAADV919:
    run_dir = tmp_path / "run"
    return TraceAADV919(
        llm=llm,
        evaluation=FastEvaluation(),
        artifacts=RunArtifacts(run_dir, console_output=False),
        budget=budget,
        n_roots=n_roots,
        seed=0,
        checkpoint_dir=run_dir / "checkpoints",
    )


# ---------------------------------------------------------------------------
# landscape layer
# ---------------------------------------------------------------------------


def test_midrank_percentile_orders_and_ties() -> None:
    values = {1: 1.0, 2: 3.0, 3: 2.0, 4: 3.0}
    percentiles = midrank_percentile(values)
    assert percentiles[1] == pytest.approx(0.0)
    assert percentiles[3] == pytest.approx(1.0 / 3.0)
    assert percentiles[2] == pytest.approx(2.5 / 3.0)
    assert percentiles[4] == pytest.approx(2.5 / 3.0)


def test_midrank_percentile_all_tied_is_one_half() -> None:
    assert set(midrank_percentile({1: 5.0, 2: 5.0}).values()) == {0.5}
    assert midrank_percentile({1: 5.0}) == {1: 0.5}


def test_neighborhood_size_formula() -> None:
    assert neighborhood_size(1) == 0
    assert neighborhood_size(8) == 2
    assert neighborhood_size(20) == 2
    assert neighborhood_size(41) == 2
    assert neighborhood_size(61) == 3
    assert neighborhood_size(200) == 10


def test_singleton_region_statistics_is_defined() -> None:
    landscape = Landscape(task="tsp_construct", protocol={"protocol_id": "test"})
    landscape.add(1, [[[0], [0, 1]]])
    stats = region_statistics(
        landscape=landscape, quality={1: 1.0}, opportunities={1: 0}
    )
    assert stats.neighbors[1] == ()
    assert stats.neighborhood_size == 0
    assert stats.promise[1] == pytest.approx(1.0 / 2.0)


def _fake_landscape(distances: dict[tuple[int, int], float]) -> Landscape:
    landscape = Landscape(task="tsp_construct", protocol={"protocol_id": "test"})
    ids = sorted({node for pair in distances for node in pair})
    if not ids:
        return landscape
    index = {node_id: position for position, node_id in enumerate(ids)}
    size = len(ids)
    matrix = np.zeros((size, size), dtype=np.float32)
    for (left, right), value in distances.items():
        matrix[index[left], index[right]] = value
        matrix[index[right], index[left]] = value
    landscape._ids = list(ids)
    landscape._index = index
    landscape._matrix = matrix
    return landscape


def test_landscape_add_computes_incremental_distances() -> None:
    landscape = Landscape(task="tsp_construct", protocol={"protocol_id": "test"})
    trajectory_a = [[[0], [0, 1], [0, 1, 2]]]
    landscape.add(1, trajectory_a)
    landscape.add(2, [[[0], [0, 2], [0, 2, 1]]])
    landscape.add(3, trajectory_a)
    assert landscape.distance(1, 3) == pytest.approx(0.0)
    assert landscape.distance(1, 2) > 0.0
    assert landscape.matrix.shape == (3, 3)
    assert np.allclose(landscape.matrix, landscape.matrix.T)


def test_region_promise_weights_self_two_thirds() -> None:
    distances = {
        (1, 2): 0.1,
        (1, 3): 0.9,
        (2, 3): 0.9,
        (1, 4): 0.2,
        (2, 4): 0.2,
        (3, 4): 0.9,
        (1, 5): 0.8,
        (2, 5): 0.8,
        (3, 5): 0.1,
        (4, 5): 0.8,
    }
    landscape = _fake_landscape(distances)
    quality = {1: 10.0, 2: 8.0, 3: 2.0, 4: 6.0, 5: 4.0}
    stats = region_statistics(
        landscape=landscape,
        quality=quality,
        opportunities={1: 0, 2: 0, 3: 9, 4: 0, 5: 0},
    )
    assert set(stats.neighbors[1]) == {2, 4}
    assert stats.neighborhood_size == 2
    assert stats.q[1] == pytest.approx(1.0)
    assert stats.q[3] == pytest.approx(0.0)
    # neighbors of 1 are {2, 4} with Q 0.75 and 0.5; median = 0.625
    # P = (2*1.0 + 0.625) / 3
    assert stats.promise[1] == pytest.approx((2.0 * 1.0 + 0.625) / 3.0)
    assert stats.raw_coverage[3] == 9.0
    assert min(stats.underdevelopment, key=stats.underdevelopment.get) == 3


def test_archive_novelty_is_percentile_of_nearest_radii() -> None:
    landscape = Landscape(task="tsp_construct", protocol={"protocol_id": "test"})
    shared = [[[0], [0, 1], [0, 1, 2]]]
    landscape.add(1, shared)
    landscape.add(2, [[[0], [0, 2], [0, 2, 1]]])
    landscape.add(3, [[[0], [0, 3], [0, 3, 1]]])
    radii = landscape.nearest_radii()
    assert set(radii) == {1, 2, 3}
    novelty, row = landscape.archive_novelty(shared)
    rho = float(np.min(row))
    assert rho == pytest.approx(0.0)
    ranked = {node_id: radius for node_id, radius in radii.items()}
    ranked[-1] = rho
    assert novelty == pytest.approx(midrank_percentile(ranked)[-1])
    assert novelty < 1.0 / 3.0
    assert behavior_tag(0.0) == "near-known"
    assert behavior_tag(1.0 / 3.0) == "intermediate"
    assert behavior_tag(2.0 / 3.0) == "far-from-archive"
    assert behavior_tag(1.0) == "far-from-archive"


def test_crossover_reference_requires_quality_and_behavior_separation() -> None:
    landscape = _fake_landscape(
        {
            (1, 2): 0.1,
            (1, 3): 0.8,
            (1, 4): 0.9,
            (1, 5): 0.7,
            (2, 3): 0.7,
            (2, 4): 0.8,
            (2, 5): 0.6,
            (3, 4): 0.2,
            (3, 5): 0.3,
            (4, 5): 0.4,
        }
    )
    reference = landscape.select_crossover_reference(
        1,
        {1: 10.0, 2: 1.0, 3: 9.0, 4: 8.0, 5: 2.0},
        random.Random(0),
    )
    assert reference in {3, 4}


def test_landscape_store_roundtrip(tmp_path: Path) -> None:
    landscape = Landscape(task="tsp_construct", protocol={"protocol_id": "test"})
    landscape.add(1, [[[0], [0, 1]]])
    landscape.add(2, [[[0], [0, 1], [0, 1, 2]]])
    arrays = landscape.state_arrays()
    restored = Landscape.from_state_arrays(
        task="tsp_construct", protocol={"protocol_id": "test"}, arrays=arrays
    )
    assert restored.node_ids == landscape.node_ids
    assert np.allclose(restored.matrix, landscape.matrix)
    assert restored.distance(1, 2) == landscape.distance(1, 2)
    restored.add(3, [[[0], [0, 3]]])
    assert restored.distance(1, 3) >= 0.0


# ---------------------------------------------------------------------------
# trajectory layer
# ---------------------------------------------------------------------------


def _chain(tree: Tree, qualities: list[float], novelties: list[float]) -> None:
    parent = VIRTUAL_ROOT_ID
    for index, (quality, novelty) in enumerate(zip(qualities, novelties)):
        parent = tree.add_algorithm(
            code=f"node-{index}",
            fitness=quality,
            parent_id=parent,
            novelty=None if index == 0 else novelty,
            behavior_tag=None if index == 0 else behavior_tag(novelty),
        ).id
        node = tree.get_algorithm(parent)
        node.t_response = t_response(tree, parent)


def test_t_response_root_is_neutral() -> None:
    tree = Tree()
    root = tree.add_algorithm(code="root", fitness=1.0)
    assert t_response(tree, root.id) == 0.5


def test_edge_value_caps_unimproved_credit() -> None:
    assert edge_value(improved=True, novelty=0.0) == 1.0
    assert edge_value(improved=True, novelty=1.0) == 1.0
    assert edge_value(improved=False, novelty=0.0) == 0.0
    assert edge_value(improved=False, novelty=1.0) == 0.5
    assert edge_value(improved=False, novelty=0.4) == pytest.approx(0.2)


def test_t_response_uses_recent_edges_and_novelty() -> None:
    tree = Tree()
    # root 1.0; then improve, improve, regress-near, improve
    _chain(tree, [1.0, 2.0, 3.0, 2.5, 3.5], [0.5, 0.1, 0.2, 0.0, 0.8])
    child = tree.add_algorithm(
        code="child", fitness=4.0, parent_id=5, novelty=0.2, behavior_tag="near-known"
    )
    # last four edges of the child: improve (1), regress-near (0.5*0.0=0),
    # improve (1), improve (1)
    expected = (1.0 + (1.0 + 0.0 + 1.0 + 1.0)) / 6.0
    assert t_response(tree, child.id) == pytest.approx(expected)


def test_unimproved_far_edge_keeps_limited_credit() -> None:
    tree = Tree()
    tree.add_algorithm(code="root", fitness=5.0)
    child = tree.add_algorithm(
        code="child",
        fitness=4.0,
        parent_id=1,
        novelty=1.0,
        behavior_tag="far-from-archive",
    )
    # one unimproved far edge: v = 0.5 -> T = (1 + 0.5) / 3
    assert t_response(tree, child.id) == pytest.approx(1.5 / 3.0)


def test_repeated_near_no_improve_lowers_t() -> None:
    tree = Tree()
    _chain(tree, [5.0, 4.0, 3.0], [0.5, 0.0, 0.0])
    # two unimproved near edges: v=0,0 -> T = 1/4 = 0.25
    assert t_response(tree, 3) == pytest.approx(0.25)


def test_trajectory_response_reacts_to_direct_failures() -> None:
    tree = Tree()
    node = tree.add_algorithm(code="root", fitness=1.0)
    assert trajectory_response(node) == pytest.approx(0.5)
    node.successful_opportunities = 1
    node.failed_opportunities = 3
    assert trajectory_response(node) == pytest.approx(0.5 * 0.5 + 0.5 * (2.0 / 6.0))


# ---------------------------------------------------------------------------
# node score and Boltzmann allocation
# ---------------------------------------------------------------------------


def test_node_score_uses_promise_opportunity_and_t() -> None:
    stats = landscape_module.RegionStats(
        q={1: 1.0, 2: 0.0},
        promise={1: 0.9, 2: 0.1},
        underdevelopment={1: 0.1, 2: 0.9},
        raw_coverage={1: 5.0, 2: 0.0},
        neighbors={1: (2,), 2: (1,)},
        pool_size=2,
        neighborhood_size=1,
    )
    scores = node_score(stats, {1: 0.5, 2: 0.5})
    assert scores[1] == pytest.approx(0.75 * 0.9 + 0.10 * 0.1 + 0.15 * 0.5)
    assert scores[2] == pytest.approx(0.75 * 0.1 + 0.10 * 0.9 + 0.15 * 0.5)


def test_solve_beta_reaches_target_ess() -> None:
    scores = [0.0, 0.2, 0.5, 0.9, 1.0, 0.4, 0.3, 0.6]
    beta = solve_beta(scores, target_ess(len(scores)))
    probabilities = boltzmann_probabilities(beta, scores)
    assert effective_sample_size(probabilities) == pytest.approx(
        target_ess(len(scores)), abs=1e-2
    )
    assert solve_beta([1.0] * 8, 2.0) == 0.0


def test_target_ess_formula() -> None:
    assert target_ess(8) == pytest.approx(2.0)
    assert target_ess(100) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# action layer
# ---------------------------------------------------------------------------


def test_explore_probability_follows_t() -> None:
    assert explore_probability(0.5) == pytest.approx(0.30)
    assert explore_probability(1.0) == pytest.approx(0.10)
    assert explore_probability(0.0) == pytest.approx(0.60)
    assert explore_probability(0.25) == pytest.approx(0.45)


def test_decide_action_explores_when_t_is_low() -> None:
    class RngStub:
        def __init__(self, draw: float) -> None:
            self.draw = draw

        def random(self) -> float:
            return self.draw

    # T=0 gives p_E=0.60; draw 0.5 selects Explore
    decision = decide_action(t_value=0.0, rng=RngStub(0.5))
    assert decision.p_explore == pytest.approx(0.60)
    assert decision.action is Action.EXPLORE

    # T=1 gives p_E=0.10; draw 0.5 selects Develop
    decision = decide_action(t_value=1.0, rng=RngStub(0.5))
    assert decision.p_explore == pytest.approx(0.10)
    assert decision.action is Action.DEVELOP


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------


def test_render_path_includes_behavior_tag() -> None:
    tree = Tree()
    tree.add_algorithm(code="root", fitness=1.0)
    tree.add_algorithm(
        code="child",
        fitness=2.0,
        parent_id=1,
        idea="blend",
        novelty=0.1,
        behavior_tag="near-known",
    )
    text = render_path(tree, 2)
    assert "Idea: blend" in text
    assert "Result: improve" in text
    assert "Fitness: 1 -> 2" in text
    assert "Behavior: near-known" in text


def test_generation_prompt_carries_trajectory_and_action_only() -> None:
    history = (
        "[Recent Algorithm Improvement History]\n"
        "\n[History 1] Formation step\nIdea: blend\nResult: improve\n"
        "Fitness: 1 -> 2\nBehavior: near-known"
    )
    prompt = build_generation_prompt(
        task_description="Improve the TSP heuristic.",
        code="def select_next_node(): ...",
        fitness=2.0,
        history_text=history,
        action=Action.DEVELOP,
    )
    assert ACTION_INSTRUCTIONS[Action.DEVELOP] in prompt
    assert "Do not write a module or function docstring, and do not write comments." in prompt
    assert "Idea: blend" in prompt
    assert "Result: improve" in prompt
    assert "Behavior: near-known" in prompt
    assert "Reference Algorithm" not in prompt
    assert "BehaveSim" not in prompt
    assert "percentile" not in prompt
    assert ACTION_INSTRUCTIONS[Action.EXPLORE] not in prompt


def test_explore_prompt_uses_explore_instruction() -> None:
    prompt = build_generation_prompt(
        task_description="Improve the TSP heuristic.",
        code="def select_next_node(): ...",
        fitness=2.0,
        history_text="[Recent Algorithm Improvement History]",
        action=Action.EXPLORE,
    )
    assert ACTION_INSTRUCTIONS[Action.EXPLORE] in prompt
    assert "Reference Algorithm" not in prompt


def test_crossover_prompt_contains_reference_algorithm() -> None:
    prompt = build_generation_prompt(
        task_description="Improve the TSP heuristic.",
        code="def select_next_node(): ...",
        fitness=2.0,
        history_text="[Recent Algorithm Improvement History]",
        action=Action.CROSSOVER,
        reference_code="def select_next_node(): return 2",
        reference_fitness=1.5,
        reference_behavior="far-from-archive",
        reference_distance=0.72,
    )
    assert ACTION_INSTRUCTIONS[Action.CROSSOVER] in prompt
    assert "[Crossover Reference Algorithm]" in prompt
    assert "def select_next_node(): return 2" in prompt
    assert "Fitness: 1.5" in prompt
    assert "Behavior distance from current algorithm: 0.72" in prompt


def test_parse_program_response() -> None:
    parsed = parse_program_response(
        "Idea: x\nCode:\n```python\ndef select_next_node(): return 1\n```"
    )
    assert parsed.declared_idea == "x"
    assert "def select_next_node" in parsed.code
    assert "return 1" in parsed.code


def test_parse_program_response_drops_docstring_and_comments() -> None:
    parsed = parse_program_response(
        "Idea: x\nCode:\n```python\n"
        "import numpy as np\n\n"
        "def select_next_node():\n"
        '    """a long strategy essay"""\n'
        "    # History 6 worked\n"
        "    return 1\n"
        "```"
    )
    assert parsed.declared_idea == "x"
    assert "long strategy" not in parsed.code
    assert "History 6" not in parsed.code
    assert "#" not in parsed.code
    assert "return 1" in parsed.code


def test_root_prompt_shows_signature_without_template_docstring() -> None:
    evaluation = FastEvaluation()
    template = TextFunctionProgramConverter.text_to_program(evaluation.template_program)
    assert template is not None
    prompt = build_root_prompt(
        task_description=evaluation.task_description,
        template_function=template.functions[0],
    )
    assert "def select_next_node" in prompt
    assert "Design a novel algorithm" not in prompt
    assert "Do not write a module or function docstring, and do not write comments." in prompt


# ---------------------------------------------------------------------------
# search-loop behavior
# ---------------------------------------------------------------------------


def test_run_completes_and_charges_one_slot_per_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = BlendLLM()
    method = _make_method(tmp_path, llm, budget=12)
    method.run()
    assert method._n_eval == 12
    assert method._n_calls == 12
    assert method._repair_llm_calls == 0
    assert len(method._tree.valid_algorithms()) == 12
    assert len(method._landscape.node_ids) == 12
    with (tmp_path / "run" / "evaluations.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    events = [
        json.loads(line)
        for line in (tmp_path / "run" / "mechanism_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    pre = [event for event in events if event["event"] == "pre_decision"]
    actions = [event for event in events if event["event"] == "action_decision"]
    assert len(pre) == 4 and len(actions) == 4
    snapshot = pre[0]["snapshot"][0]
    assert {"id", "q", "P", "U", "B", "c_t", "T", "S"} == set(snapshot)
    assert pre[0]["pool_size"] == 8 and pre[0]["neighborhood_size"] == 2
    assert set(event["action"] for event in actions) <= {"develop", "explore", "crossover"}
    children = [node for node in method._tree.valid_algorithms() if node.parent_id]
    assert children
    assert all(node.novelty is not None for node in children)
    assert all(
        node.behavior_tag
        in {"near-known", "intermediate", "far-from-archive"}
        for node in children
    )
    decisions = [
        json.loads(line)
        for line in (tmp_path / "run" / "decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(decisions) == 4
    record = decisions[0]
    assert record["task"] == "tsp_construct"
    assert record["action"] in {"develop", "explore", "crossover"}
    assert "exact_prompt" in record and "exact_response" in record
    assert record["P"] is not None and record["U"] is not None and record["T"] is not None


def test_duplicate_consumes_slot_without_creating_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = BlendLLM()
    method = _make_method(tmp_path, llm, budget=9)
    original = llm.draw_sample

    def always_same(prompt, *args, **kwargs):
        if not hasattr(llm, "_frozen"):
            llm._frozen = original(prompt, *args, **kwargs)
        return llm._frozen

    llm.draw_sample = always_same  # type: ignore[method-assign]
    method.run()
    assert method._n_eval == 9
    assert method._n_calls == 9
    assert len(method._tree.valid_algorithms()) == 1
    assert method._outcome_counts["duplicate"] == 8
    assert sum(method._outcome_counts.values()) == 9


def test_repair_replaces_direct_failure_credit_with_final_outcome(
    tmp_path: Path,
) -> None:
    method = _make_method(tmp_path, BlendLLM(), budget=2, n_roots=1)
    root = method._tree.add_algorithm(code="root", fitness=1.0)
    parsed = parse_program_response(
        "Idea: repaired\nCode:\n```python\ndef select_next_node(): return 1\n```"
    )
    from llm4ad.method.traceaad_v9_19.schema import Pending

    method._pending = Pending(
        parent_id=root.id,
        action="develop",
        response="",
        exact_prompt="",
    )
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
    method._pending = Pending(
        parent_id=root.id,
        action="develop",
        response="",
        exact_prompt="",
    )
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
    assert root.opportunities == 1
    assert root.successful_opportunities == 1
    assert root.failed_opportunities == 0


def test_bounded_repair_recovers_from_broken_first_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = BlendLLM(broken_first=1)
    method = _make_method(tmp_path, llm, budget=8)
    method.run()
    assert method._n_eval == 8
    assert method._n_calls == 9
    assert method._repair_llm_calls == 1
    assert method._repair_eval_calls == 1
    assert len(method._tree.valid_algorithms()) == 8
    assert len(method._attempts) == 8
    assert len({attempt.slot for attempt in method._attempts}) == 8
    assert sum(method._outcome_counts.values()) == 8


def test_checkpoint_roundtrip_preserves_landscape_and_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    method = _make_method(tmp_path, BlendLLM(), budget=10)
    method.run()
    matrix = method._landscape.matrix.copy()
    attempts = [asdict(attempt) for attempt in method._attempts]

    resumed = TraceAADV919(
        llm=BlendLLM(),
        evaluation=FastEvaluation(),
        budget=10,
        n_roots=8,
        seed=0,
        checkpoint_dir=tmp_path / "resumed",
        resume_from=tmp_path / "run" / "checkpoints" / "latest.json",
    )
    assert resumed._n_eval == 10
    assert np.allclose(resumed._landscape.matrix, matrix)
    assert [asdict(attempt) for attempt in resumed._attempts] == attempts
    assert dict(resumed._outcome_counts) == dict(method._outcome_counts)
    quality = {
        algorithm.id: resumed._tree.quality(algorithm)
        for algorithm in resumed._tree.valid_algorithms()
    }
    stats = region_statistics(
        landscape=resumed._landscape,
        quality=quality,
        opportunities={
            algorithm.id: algorithm.opportunities
            for algorithm in resumed._tree.valid_algorithms()
        },
    )
    assert stats.pool_size == len(quality)
    restored = resumed._tree.valid_algorithms()
    assert all(
        node.t_response == method._tree.get_algorithm(node.id).t_response
        for node in restored
    )


def test_resumed_method_runs_and_checkpoints_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    method = _make_method(tmp_path, BlendLLM(), budget=9)
    method.run()

    resumed = TraceAADV919(
        llm=BlendLLM(),
        evaluation=FastEvaluation(),
        budget=12,
        n_roots=8,
        seed=0,
        checkpoint_dir=tmp_path / "resumed",
        resume_from=tmp_path / "run" / "checkpoints" / "latest.json",
    )
    resumed.run()
    assert resumed._n_eval == 12
    assert len(resumed._tree.valid_algorithms()) == 9
    assert (tmp_path / "resumed" / "latest.json").is_file()

    reloaded = TraceAADV919(
        llm=BlendLLM(),
        evaluation=FastEvaluation(),
        budget=12,
        n_roots=8,
        seed=0,
        checkpoint_dir=tmp_path / "reloaded",
        resume_from=tmp_path / "resumed" / "latest.json",
    )
    assert reloaded._n_eval == 12
    assert np.allclose(reloaded._landscape.matrix, resumed._landscape.matrix)


def test_resume_rejects_missing_profile_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    method = _make_method(tmp_path, BlendLLM(), budget=9)
    method.run()
    checkpoint = tmp_path / "run" / "checkpoints" / "latest.json"
    (tmp_path / "run" / "checkpoints" / "behave.npz").unlink()
    with pytest.raises(FileNotFoundError):
        TraceAADV919(
            llm=BlendLLM(),
            evaluation=FastEvaluation(),
            budget=9,
            n_roots=8,
            seed=0,
            resume_from=checkpoint,
        )


def test_pure_python_distance_matches_numba_kernel() -> None:
    left = [[[0], [0, 1], [0, 1, 2]]]
    right = [[[0], [0, 2], [0, 2, 1]]]
    combined = behave.trajectory_distances(left, [right], prefix_mode=True)

    def normalized_edit(a, b):
        if not a and not b:
            return 0.0
        previous = list(range(len(b) + 1))
        for i, av in enumerate(a, 1):
            current = [i] + [0] * len(b)
            for j, bv in enumerate(b, 1):
                current[j] = min(
                    previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (av != bv)
                )
            previous = current
        return previous[-1] / max(len(a), len(b))

    def dtw(a_states, b_states):
        n, m = len(a_states), len(b_states)
        previous = [math.inf] * (m + 1)
        previous[0] = 0.0
        for i in range(1, n + 1):
            current = [math.inf] * (m + 1)
            for j in range(1, m + 1):
                local = normalized_edit(a_states[i - 1], b_states[j - 1])
                current[j] = local + min(previous[j], current[j - 1], previous[j - 1])
            previous = current
        return previous[m] / min(n, m)

    assert combined[0] == pytest.approx(dtw(left[0], right[0]))
