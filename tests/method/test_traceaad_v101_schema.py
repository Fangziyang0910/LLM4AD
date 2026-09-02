import json
import math
import random

from llm4ad.method.traceaad_v10_1.schema import SearchTree
from llm4ad.method.traceaad_v10_1.traceaad import TraceAADV101


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
                "batch_counter": 3,
                "budget_used": 17,
            }
        ),
        encoding="utf-8",
    )
    method = TraceAADV101.__new__(TraceAADV101)
    method.state_path = state_path
    method.tree = SearchTree()
    method.rng = random.Random()

    method._load_state()

    assert method.budget_used == 17
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


def test_render_trajectory_step_indexing_and_trends():
    from llm4ad.method.traceaad_v10_1.prompts import render_trajectory
    from llm4ad.method.traceaad_v10_1.schema import Node

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
    from llm4ad.method.traceaad_v10_1.traceaad import calibrate_beta, _ess

    fitnesses = [1.0, 2.0, 3.0, 4.0, 5.0]
    beta, target, actual = calibrate_beta(fitnesses, ess_fraction=0.5, ess_minimum=2)
    assert beta > 0.0
    assert abs(actual - target) < 0.1
    # Check that higher fitness gets higher probability
    weights = [math.exp(beta * f) for f in fitnesses]
    assert weights[-1] > weights[0]


