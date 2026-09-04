from collections import defaultdict

from experiments.traceaad_v10_prompt_probe.probe import (
    FUSE_CONDITIONS,
    INIT_CONDITIONS,
    build_prompt_variant,
    build_schedule,
)
from llm4ad.method.traceaad_v10_1 import prompts as prompts_v101
from llm4ad.method.traceaad_v10_1.schema import Node
from llm4ad.method.traceaad_v10_2 import prompts as prompts_v102


def _nodes() -> tuple[Node, list[Node], Node]:
    root = Node(
        id=0,
        code="def f(x):\n    # root note\n    return x",
        idea="root",
        fitness=1.0,
    )
    parent = Node(
        id=1,
        code="def f(x):\n    # current note\n    return x + 1",
        idea="current",
        fitness=2.0,
        parent_id=0,
        operator="Refine",
    )
    donor = Node(
        id=2,
        code="def f(x):\n    # donor note\n    return x * 2",
        idea="donor",
        fitness=1.5,
    )
    return parent, [root], donor


def test_endpoint_conditions_are_exact_version_prompts() -> None:
    parent, ancestors, donor = _nodes()
    kwargs = {
        "task_contract": "# Task Contract",
        "current": parent,
        "ancestors": ancestors,
        "trajectory_display": 1,
        "operator": "Fuse",
        "donor": donor,
    }

    v101 = build_prompt_variant(condition="v101", **kwargs)
    assert "Generation -1\nIdea: root" in v101
    assert "Step 1" not in v101
    assert prompts_v101.OPERATOR_INSTRUCTIONS["Fuse"] in v101
    assert prompts_v101.OUTPUT_CONTRACT in v101
    assert build_prompt_variant(condition="v102", **kwargs) == prompts_v102._assemble(
        kwargs["task_contract"], parent, ancestors, 1, "Fuse", donor
    )


def test_cumulative_conditions_change_only_the_named_prompt_components() -> None:
    parent, ancestors, donor = _nodes()
    kwargs = {
        "task_contract": "# Task Contract",
        "current": parent,
        "ancestors": ancestors,
        "trajectory_display": 1,
        "operator": "Fuse",
        "donor": donor,
    }
    prompts = {
        condition: build_prompt_variant(condition=condition, **kwargs)
        for condition in FUSE_CONDITIONS
    }

    assert prompts_v102.IMPLEMENTATION_PRINCIPLE not in prompts["v101"]
    assert prompts_v102.IMPLEMENTATION_PRINCIPLE in prompts["implementation_principle"]
    assert "current note" in prompts["implementation_principle"]
    assert "donor note" in prompts["implementation_principle"]
    assert "current note" not in prompts["comment_free"]
    assert "donor note" not in prompts["comment_free"]
    assert prompts_v101.OPERATOR_INSTRUCTIONS["Fuse"] in prompts["comment_free"]
    assert prompts_v102.OPERATOR_INSTRUCTIONS["Fuse"] in prompts["operator_instruction"]
    assert prompts_v101.OUTPUT_CONTRACT in prompts["operator_instruction"]
    assert prompts_v102.OUTPUT_CONTRACT in prompts["v102"]


def test_schedule_keeps_each_randomized_complete_block_together() -> None:
    anchors = [
        {"anchor_id": "a1", "task": "tsp_construct"},
        {"anchor_id": "a2", "task": "cvrp_aco"},
    ]
    schedule = build_schedule(anchors, init_pairs_per_task=1, seed=7)
    blocks = defaultdict(list)
    for row in schedule:
        blocks[row["block_id"]].append(row)

    for anchor in anchors:
        rows = blocks[anchor["anchor_id"]]
        assert {row["condition"] for row in rows} == set(FUSE_CONDITIONS)
        assert len({row["sampling_seed"] for row in rows}) == 1
        assert sorted(row["within_block_order"] for row in rows) == list(range(5))
    init_rows = blocks["tsp_construct:init:1"]
    assert {row["condition"] for row in init_rows} == set(INIT_CONDITIONS)
    assert len({row["sampling_seed"] for row in init_rows}) == 1
