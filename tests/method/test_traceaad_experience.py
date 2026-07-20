from __future__ import annotations

from llm4ad.method.traceaad.derivation_graph import DerivationGraph
from llm4ad.method.traceaad.experience_memory import ExperienceMemory


def test_cross_trajectory_evidence_returns_successes_and_failures() -> None:
    graph = DerivationGraph()
    root = graph.add_node(code="root", idea="root", fitness=1.0)
    better = graph.add_node(code="better", idea="better", fitness=2.0)
    worse = graph.add_node(code="worse", idea="worse", fitness=0.0)
    graph.add_edge(
        parent_id=root.id,
        child_id=better.id,
        action="use local density",
        operator="endpoint_refine",
        delta=1.0,
        outcome="improve",
    )
    graph.add_edge(
        parent_id=root.id,
        child_id=worse.id,
        action="use random choice",
        operator="novelty_jump",
        delta=-1.0,
        outcome="regress",
    )

    evidence = ExperienceMemory(graph).examples(operator="endpoint_refine")

    assert [item.action for item in evidence.positives] == ["use local density"]
    assert [item.action for item in evidence.negatives] == ["use random choice"]
