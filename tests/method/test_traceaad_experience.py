"""ExperienceMemory 边级无标签经验检索测试。"""
from __future__ import annotations

from llm4ad.method.traceaad.context import _contrast_block
from llm4ad.method.traceaad.derivation_graph import DerivationGraph
from llm4ad.method.traceaad.experience_memory import ExperienceMemory, normalize_action_text
from llm4ad.method.traceaad.schema import OperatorName


def _graph_with_edges() -> DerivationGraph:
    graph = DerivationGraph()
    root = graph.add_node(code="root", idea="root", fitness=0.0)
    for index, (action, operator, delta, outcome, iteration) in enumerate(
        (
            ("raise the constant", OperatorName.ENDPOINT, 2.0, "improve", 1),
            ("raise the constant", OperatorName.ENDPOINT, 1.5, "improve", 2),
            ("add a bonus term", OperatorName.ENDPOINT, 0.5, "improve", 3),
            ("shrink the constant", OperatorName.ENDPOINT, -3.0, "regress", 4),
            ("borrow donor scaling", OperatorName.CROSSOVER, 1.0, "improve", 5),
            ("borrow donor scaling", OperatorName.CROSSOVER, -2.0, "regress", 6),
            ("plateau tweak", OperatorName.ENDPOINT, 0.0, "plateau", 7),
            ("", OperatorName.ENDPOINT, 4.0, "improve", 8),
        ),
        start=1,
    ):
        child = graph.add_node(code=f"c{index}", idea=f"c{index}", fitness=float(index))
        graph.add_edge(
            parent_id=root.id,
            child_id=child.id,
            action=action,
            operator=operator,
            delta=delta,
            outcome=outcome,
            iteration=iteration,
        )
    return graph


def test_normalize_action_collapses_whitespace() -> None:
    assert normalize_action_text("  raise   the\nconstant  ") == "raise the constant"


def test_examples_deduplicate_identical_actions_and_respect_k() -> None:
    memory = ExperienceMemory(_graph_with_edges())

    batch = memory.examples(operator=OperatorName.ENDPOINT, positive_k=2, negative_k=1)

    assert len(batch.positives) == 2
    assert [example.action for example in batch.positives] == [
        "raise the constant",
        "add a bonus term",
    ]
    assert batch.positives[0].delta == 2.0
    assert len(batch.negatives) == 1
    assert batch.negatives[0].action == "shrink the constant"
    assert batch.negatives[0].delta == -3.0


def test_examples_default_to_two_successes_and_two_failures() -> None:
    batch = ExperienceMemory(_graph_with_edges()).examples(
        operator=OperatorName.ENDPOINT
    )

    assert len(batch.positives) <= 2
    assert len(batch.negatives) <= 2


def test_examples_prefer_current_operator_then_fallback_globally() -> None:
    memory = ExperienceMemory(_graph_with_edges())

    batch = memory.examples(operator=OperatorName.CROSSOVER, positive_k=2, negative_k=2)

    assert [example.operator for example in batch.positives] == [
        OperatorName.ENDPOINT,
        OperatorName.ENDPOINT,
    ]
    assert batch.negatives[0].operator == OperatorName.CROSSOVER
    assert batch.negatives[1].operator == OperatorName.ENDPOINT


def test_examples_skip_plateau_and_empty_action() -> None:
    memory = ExperienceMemory(_graph_with_edges())

    batch = memory.examples(operator=OperatorName.ENDPOINT, positive_k=10, negative_k=10)

    actions = {example.action for example in batch.positives + batch.negatives}
    assert "" not in actions
    assert "plateau tweak" not in actions


def test_examples_use_directed_delta_without_flipping() -> None:
    graph = DerivationGraph()
    root = graph.add_node(code="root", idea="root", fitness=10.0)
    child = graph.add_node(code="child", idea="child", fitness=8.0)
    # Minimization task already stores directed delta as positive improvement.
    graph.add_edge(
        parent_id=root.id,
        child_id=child.id,
        action="reduce cost",
        operator=OperatorName.ENDPOINT,
        delta=2.0,
        outcome="improve",
        iteration=1,
    )
    memory = ExperienceMemory(graph)

    batch = memory.examples(operator=OperatorName.ENDPOINT, positive_k=1, negative_k=1)

    assert len(batch.positives) == 1
    assert batch.positives[0].delta == 2.0
    assert batch.negatives == ()


def test_examples_keep_one_representative_when_action_has_mixed_outcomes() -> None:
    memory = ExperienceMemory(_graph_with_edges())

    batch = memory.examples(operator=OperatorName.CROSSOVER, positive_k=10, negative_k=10)

    actions = [example.action for example in batch.positives + batch.negatives]
    assert actions.count("borrow donor scaling") == 1
    representative = next(
        example
        for example in batch.positives + batch.negatives
        if example.action == "borrow donor scaling"
    )
    assert representative.outcome == "regress"
    assert representative.delta == -2.0


def test_examples_prefer_newer_iteration_when_delta_ties() -> None:
    graph = DerivationGraph()
    root = graph.add_node(code="root", idea="root", fitness=0.0)
    for iteration, action in ((1, "older action"), (5, "newer action")):
        child = graph.add_node(code=action, idea=action, fitness=1.0)
        graph.add_edge(
            parent_id=root.id,
            child_id=child.id,
            action=action,
            operator=OperatorName.ENDPOINT,
            delta=1.0,
            outcome="improve",
            iteration=iteration,
        )

    batch = ExperienceMemory(graph).examples(
        operator=OperatorName.ENDPOINT,
        positive_k=2,
        negative_k=0,
    )

    assert [example.action for example in batch.positives] == [
        "newer action",
        "older action",
    ]


def test_empty_graph_returns_empty_batch() -> None:
    memory = ExperienceMemory(DerivationGraph())

    batch = memory.examples(operator=OperatorName.ENDPOINT)

    assert batch.positives == ()
    assert batch.negatives == ()


def test_contrast_block_contains_only_fitness_and_idea() -> None:
    block = _contrast_block(
        {
            "best": {"fitness": 2.0, "idea": "strong"},
            "worst": {"fitness": 1.0, "idea": "weak"},
        }
    )

    assert "fitness=2" in block
    assert "strong" in block
    assert "fitness=1" in block
    assert "weak" in block
    assert "mechanism" not in block.lower()
    assert "mech=" not in block.lower()
