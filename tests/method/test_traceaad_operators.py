from __future__ import annotations

from llm4ad.method.traceaad.derivation_graph import DerivationGraph
from llm4ad.method.traceaad.experience_memory import ExperienceMemory
from llm4ad.method.traceaad.islands import IslandsManager
from llm4ad.method.traceaad.operators import (
    BacktrackBranchOp,
    MechanismCrossoverOp,
    NoveltyJumpOp,
    OperatorContext,
    SimplifyOp,
)
from llm4ad.method.traceaad.schema import ValueVec
from llm4ad.method.traceaad.trajectory_memory import TrajectoryMemory


def _context(*, graph: DerivationGraph, memory: TrajectoryMemory, selected, **fields):
    return OperatorContext(
        graph=graph,
        memory=memory,
        experience_memory=ExperienceMemory(graph),
        islands=fields.pop("islands", IslandsManager(n_islands=1)),
        selected=selected,
        maximize=True,
        **fields,
    )


def _trajectory_with_fitnesses(*fitnesses: float, complexities: tuple[int, ...] | None = None):
    graph = DerivationGraph()
    memory = TrajectoryMemory(max_trajectory_length=8)
    complexities = complexities or tuple(0 for _ in fitnesses)
    root = graph.add_node(
        code="root", idea="root", fitness=fitnesses[0], complexity=complexities[0],
    )
    trajectory = memory.create_initial(node_id=root.id)
    for index, (fitness, complexity) in enumerate(zip(fitnesses[1:], complexities[1:]), start=1):
        child = graph.add_node(
            code=f"child-{index}", idea=f"child-{index}", fitness=fitness, complexity=complexity,
        )
        edge = graph.add_edge(
            parent_id=trajectory.endpoint_id,
            child_id=child.id,
            action=f"step-{index}",
        )
        trajectory = memory.extend(
            trajectory_id=trajectory.id,
            parent_id=trajectory.endpoint_id,
            child_id=child.id,
            edge_id=edge.id,
        )
    return graph, memory, trajectory


def _initial_trajectory(
    *,
    graph: DerivationGraph,
    memory: TrajectoryMemory,
    fitness: float,
    code: str,
    idea: str,
    quality: float,
    scalar: float,
    island_id: int = 0,
):
    node = graph.add_node(code=code, idea=idea, fitness=fitness)
    trajectory = memory.create_initial(node_id=node.id, island_id=island_id)
    memory.set_value(trajectory.id, ValueVec(quality=quality), scalar)
    return memory.get_trajectory(trajectory.id)


def test_backtrack_does_not_rename_an_endpoint_rewrite_as_internal():
    graph, memory, trajectory = _trajectory_with_fitnesses(10.0, 0.0)
    ctx = _context(graph=graph, memory=memory, selected=trajectory)

    base_id, reason = BacktrackBranchOp().select_base(ctx)

    assert base_id == trajectory.endpoint_id
    assert reason == "endpoint"


def test_backtrack_is_ineligible_when_base_selection_stays_at_endpoint():
    graph, memory, trajectory = _trajectory_with_fitnesses(10.0, 0.0)
    ctx = _context(graph=graph, memory=memory, selected=trajectory)

    operator = BacktrackBranchOp()

    assert operator.select_trajectory(ctx) is None
    assert not operator.trigger(ctx)


def test_backtrack_remains_eligible_when_base_selection_is_strictly_internal():
    graph, memory, trajectory = _trajectory_with_fitnesses(10.0, 9.0)
    ctx = _context(graph=graph, memory=memory, selected=trajectory)

    operator = BacktrackBranchOp()
    target = operator.select_trajectory(ctx)

    assert target is not None
    ctx.selected = target
    base_id, _ = operator.select_base(ctx)
    assert base_id != target.endpoint_id


def test_simplify_triggers_on_relative_complexity_alone():
    graph, memory, trajectory = _trajectory_with_fitnesses(
        10.0, 10.0, complexities=(10, 20),
    )
    ctx = _context(graph=graph, memory=memory, selected=trajectory)

    assert SimplifyOp().trigger(ctx)


def test_simplify_does_not_trigger_without_relative_complexity_pressure():
    # selected endpoint is the simpler of the two active endpoints
    graph, memory, trajectory = _trajectory_with_fitnesses(
        10.0, 11.0, complexities=(100, 10),
    )
    ctx = _context(graph=graph, memory=memory, selected=trajectory)

    assert not SimplifyOp().trigger(ctx)


def test_simplify_does_not_trigger_with_single_active_endpoint_pool():
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    node = graph.add_node(code="only", idea="only", fitness=10.0, complexity=100)
    trajectory = memory.create_initial(node_id=node.id)
    ctx = _context(graph=graph, memory=memory, selected=trajectory)

    assert not SimplifyOp().trigger(ctx)


def test_crossover_rejects_a_low_quality_donor_despite_high_scalar_value():
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    selected = _initial_trajectory(
        graph=graph, memory=memory, fitness=10.0,
        code=(
            "def score(current, candidates):\n"
            "    dens = [local_density(current, c) for c in candidates]\n"
            "    return dens.index(max(dens))\n"
        ),
        idea="density scoring",
        quality=0.9, scalar=0.9,
    )
    low_quality = _initial_trajectory(
        graph=graph, memory=memory, fitness=1.0,
        code=(
            "import random\n"
            "def pick_next(state, options):\n"
            "    ranks = sorted(options, key=lambda item: random.random() * priority(item))\n"
            "    return ranks[0]\n"
        ),
        idea="sparse list",
        quality=0.1, scalar=10.0,
    )
    qualified = _initial_trajectory(
        graph=graph, memory=memory, fitness=8.0,
        code=(
            "import random\n"
            "def pick_next(state, options):\n"
            "    ranks = sorted(options, key=lambda item: random.random() * priority(item))\n"
            "    return ranks[0]\n"
        ),
        idea="edge contrast",
        quality=0.8, scalar=0.6,
    )
    ctx = _context(graph=graph, memory=memory, selected=selected)

    MechanismCrossoverOp().select_base(ctx)

    assert ctx.hints["donor_idea"] == "edge contrast"
    assert graph.get_node(qualified.endpoint_id).idea == "edge contrast"
    assert graph.get_node(low_quality.endpoint_id).idea == "sparse list"


def test_crossover_prefers_higher_complementarity_plus_quality():
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    selected = _initial_trajectory(
        graph=graph, memory=memory, fitness=10.0,
        code=(
            "def score(current, candidates):\n"
            "    dens = [local_density(current, c) for c in candidates]\n"
            "    return dens.index(max(dens))\n"
        ),
        idea="density scoring",
        quality=0.9, scalar=0.9,
    )
    similar = _initial_trajectory(
        graph=graph, memory=memory, fitness=8.0,
        code=(
            "def score(current, candidates):\n"
            "    dens = [local_density(current, c) for c in candidates]\n"
            "    return dens.index(max(dens))\n"
        ),
        idea="almost same density",
        quality=0.85, scalar=0.85,
    )
    complementary = _initial_trajectory(
        graph=graph, memory=memory, fitness=8.0,
        code=(
            "import random\n"
            "def pick_next(state, options):\n"
            "    ranks = sorted(options, key=lambda item: random.random() * priority(item))\n"
            "    return ranks[0]\n"
        ),
        idea="rank scaling",
        quality=0.8, scalar=0.8,
    )
    ctx = _context(graph=graph, memory=memory, selected=selected)

    MechanismCrossoverOp().select_base(ctx)

    assert ctx.hints["donor_idea"] == "rank scaling"
    assert ctx.hints["donor_idea"] != graph.get_node(similar.endpoint_id).idea
    assert "donor_mechanism" not in ctx.hints


def test_crossover_constraint_mentions_donor_idea_not_mechanism_family():
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    selected = _initial_trajectory(
        graph=graph, memory=memory, fitness=10.0,
        code=(
            "def score(current, candidates):\n"
            "    dens = [local_density(current, c) for c in candidates]\n"
            "    return dens.index(max(dens))\n"
        ),
        idea="density scoring",
        quality=0.9, scalar=0.9,
    )
    _initial_trajectory(
        graph=graph, memory=memory, fitness=8.0,
        code=(
            "import random\n"
            "def pick_next(state, options):\n"
            "    ranks = sorted(options, key=lambda item: random.random() * priority(item))\n"
            "    return ranks[0]\n"
        ),
        idea="rank scaling",
        quality=0.8, scalar=0.8,
    )
    op = MechanismCrossoverOp()
    ctx = _context(graph=graph, memory=memory, selected=selected)
    base_id, _ = op.select_base(ctx)
    constraint = op.build_constraint(ctx, base_id)

    assert "rank scaling" in constraint
    assert "mechanism family" not in constraint.lower()


def test_novelty_always_triggers():
    graph, memory, trajectory = _trajectory_with_fitnesses(10.0)
    op = NoveltyJumpOp()
    ctx = _context(graph=graph, memory=memory, selected=trajectory, best_stagnation=0, iteration=0)
    assert op.trigger(ctx)
    ctx.best_stagnation = 100
    ctx.iteration = 99
    assert op.trigger(ctx)


def test_novelty_constraint_lists_existing_ideas_without_preset_families():
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    selected = _initial_trajectory(
        graph=graph, memory=memory, fitness=10.0,
        code="def a():\n    return 1", idea="alpha idea",
        quality=0.9, scalar=0.9,
    )
    _initial_trajectory(
        graph=graph, memory=memory, fitness=8.0,
        code="def b():\n    return 2", idea="beta idea",
        quality=0.8, scalar=0.8,
    )
    op = NoveltyJumpOp()
    ctx = _context(
        graph=graph, memory=memory, selected=selected,
        best_stagnation=12, iteration=0,
    )
    constraint = op.build_constraint(ctx, None)

    assert "alpha idea" in constraint
    assert "beta idea" in constraint
    assert "local_density" not in constraint
    assert "nearest neighbor" not in constraint.lower()


def test_novelty_assigns_to_least_loaded_island():
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    selected = _initial_trajectory(
        graph=graph, memory=memory, fitness=10.0,
        code="def a():\n    return 1", idea="a",
        quality=0.9, scalar=0.9, island_id=0,
    )
    _initial_trajectory(
        graph=graph, memory=memory, fitness=8.0,
        code="def b():\n    return 2", idea="b",
        quality=0.8, scalar=0.8, island_id=0,
    )
    _initial_trajectory(
        graph=graph, memory=memory, fitness=7.0,
        code="def c():\n    return 3", idea="c",
        quality=0.7, scalar=0.7, island_id=1,
    )
    child = graph.add_node(code="def d():\n    return 4", idea="d", fitness=6.0)
    islands = IslandsManager(n_islands=3)
    ctx = _context(
        graph=graph, memory=memory, selected=selected, islands=islands,
    )

    traj = NoveltyJumpOp().insert(ctx, child.id, None, None)

    assert traj.island_id == 2
