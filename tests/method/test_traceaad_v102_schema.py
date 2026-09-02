import json
import math
import random
from types import SimpleNamespace

from llm4ad.method.traceaad_v10_2.schema import SearchTree
from llm4ad.method.traceaad_v10_2.traceaad import TraceAADV102


def test_resume_loads_tree_and_budget_from_checkpoint(tmp_path) -> None:
    state_path = tmp_path / "tree_state.json"
    state_path.write_text(
        json.dumps(
            {
                "started_at": "2026-09-02T00:00:00",
                "nodes": [
                    {
                        "id": 0,
                        "code": "def f():\n    return 1",
                        "idea": "seed",
                        "fitness": 1.0,
                        "parent_id": None,
                        "origin_operator": "Init",
                        "donor_id": None,
                    }
                ],
                "rng_state": list(random.Random(0).getstate()),
                "parent_selection_counts": {"0": 4},
                "batch_counter": 3,
                "budget_used": 17,
            }
        ),
        encoding="utf-8",
    )
    method = TraceAADV102.__new__(TraceAADV102)
    method.state_path = state_path
    method.tree = SearchTree()
    method.rng = random.Random()

    method._load_state()

    assert method.budget_used == 17
    assert method.parent_selection_counts == {0: 4}
    assert method.tree.nodes[0].operator == "Init"
    assert method.tree.nodes[0].evaluation_id is None

    child = method.tree.add(
        code="def f():\n    return 2",
        idea="improve",
        fitness=2.0,
        evaluation_id=18,
        parent_id=0,
        operator="Refine",
    )
    assert child is not None
    assert child.evaluation_id == 18
    assert child.operator == "Refine"

    method._save_state()
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["parent_selection_counts"] == {"0": 4}


def test_render_trajectory_step_indexing_and_trends():
    from llm4ad.method.traceaad_v10_2.prompts import render_trajectory
    from llm4ad.method.traceaad_v10_2.schema import Node

    # ancestors: nearest first [parent, grandparent, great_grandparent]
    n_great = Node(id=0, code="c0", idea="i0", fitness=1.0, evaluation_id=1, parent_id=None, operator="Init")
    n_grand = Node(id=1, code="c1", idea="i1", fitness=1.5, evaluation_id=2, parent_id=0, operator="Refine")
    n_parent = Node(id=2, code="c2", idea="i2", fitness=1.2, evaluation_id=3, parent_id=1, operator="Pivot")

    # Display 2 nodes out of 3 ancestors (grandparent, parent), plus great_grandparent for trend
    text = render_trajectory([n_parent, n_grand, n_great], display=2)
    assert "Step 1" in text
    assert "Step 2" in text
    assert "Generation" not in text
    # Step 1 is grandparent (1.5 vs great 1.0 -> Improved)
    assert "Step 1\nIdea: i1\nFitness: 1.5 (Improved)" in text
    # Step 2 is parent (1.2 vs grand 1.5 -> Degraded)
    assert "Step 2\nIdea: i2\nFitness: 1.2 (Degraded)" in text


def test_calibrate_beta_and_parent_selection():
    from llm4ad.method.traceaad_v10_2.traceaad import calibrate_beta

    fitnesses = [1.0, 2.0, 3.0, 4.0, 5.0]
    beta, target, actual = calibrate_beta(fitnesses, ess_fraction=0.5, ess_minimum=2)
    assert beta > 0.0
    assert abs(actual - target) < 0.1
    # Check that higher fitness gets higher probability
    weights = [math.exp(beta * f) for f in fitnesses]
    assert weights[-1] > weights[0]


def test_parent_selection_corrects_quality_probability_by_selection_count() -> None:
    class ChoicesRecorder:
        def __init__(self) -> None:
            self.weights: list[float] = []

        def choices(self, population, weights):
            self.weights = list(weights)
            return [1]

    method = TraceAADV102.__new__(TraceAADV102)
    method.tree = SearchTree()
    method.tree.add(
        code="a", idea="a", fitness=1.0, evaluation_id=1,
        parent_id=None, operator="Init",
    )
    method.tree.add(
        code="b", idea="b", fitness=1.0, evaluation_id=2,
        parent_id=None, operator="Init",
    )
    method.ess_fraction = 0.1
    method.ess_minimum = 2
    method.parent_selection_counts = {0: 0, 1: 3}
    method.rng = ChoicesRecorder()

    parent, probability, _, _, _ = method.select_parent()

    assert parent.id == 1
    assert math.isclose(method.rng.weights[0], 2 / 3)
    assert math.isclose(method.rng.weights[1], 1 / 3)
    assert math.isclose(probability, 1 / 3)
    assert method.parent_selection_counts == {0: 0, 1: 4}


def test_expand_parent_uniformly_selects_one_available_operator() -> None:
    class ChoiceRecorder:
        def __init__(self, picked: str) -> None:
            self.picked = picked
            self.available: list[str] = []

        def choice(self, available):
            self.available = list(available)
            assert self.picked in self.available
            return self.picked

    for donor_available, picked, expected in [
        (True, "Fuse", ["Refine", "Pivot", "Fuse"]),
        (False, "Pivot", ["Refine", "Pivot"]),
    ]:
        method = TraceAADV102.__new__(TraceAADV102)
        method.tree = SearchTree()
        parent = method.tree.add(
            code="parent", idea="parent", fitness=1.0, evaluation_id=1,
            parent_id=None, operator="Init",
        )
        donor = method.tree.add(
            code="donor", idea="donor", fitness=0.9, evaluation_id=2,
            parent_id=None, operator="Init",
        )
        method.budget = 10
        method.budget_used = 2
        method.traj_gens = 8
        method.step_counter = 0
        method.rng = ChoiceRecorder(picked)
        method.select_parent = lambda: (parent, 0.5, 1.0, 2.0, 2.0)
        method.select_donor = lambda _: donor if donor_available else None
        generated = []

        def generate(operator, current, ancestors, selected_donor):
            generated.append((operator, selected_donor))
            return SimpleNamespace(
                operator=operator,
                program=object(),
                idea="child",
                code="child",
                fitness=None,
                evaluation_id=None,
                node_id=None,
            )

        def evaluate(attempt) -> None:
            method.budget_used += 1
            attempt.evaluation_id = method.budget_used
            attempt.fitness = 1.1

        method._generate = generate
        method._evaluate = evaluate
        method._log_attempt = lambda *args, **kwargs: None
        method._save_state = lambda: None

        method._expand_parent()

        assert method.rng.available == expected
        assert len(generated) == 1
        assert generated[0][1] is (donor if picked == "Fuse" else None)
        assert method.budget_used == 3
        assert method.step_counter == 1
        assert len(method.tree.nodes) == 3
