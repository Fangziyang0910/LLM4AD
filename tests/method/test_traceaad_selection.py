from __future__ import annotations

import pytest

from llm4ad.method.traceaad.derivation_graph import DerivationGraph
from llm4ad.method.traceaad.islands import IslandsManager
from llm4ad.method.traceaad.schema import ValueVec
from llm4ad.method.traceaad.trajectory_memory import TrajectoryMemory
from llm4ad.method.traceaad.value import (
    ValueWeights,
    compute_value_vec,
    pareto_survival_order,
    scalarize,
    select_trajectory,
    ucb_bonus,
)


def _add_initial(
    *,
    graph: DerivationGraph,
    memory: TrajectoryMemory,
    fitness: float,
    island_id: int = 0,
):
    node = graph.add_node(
        code=f"def solve(): return {fitness}",
        idea=f"fitness {fitness}",
        fitness=fitness,
        is_valid=True,
    )
    return node, memory.create_initial(node_id=node.id, island_id=island_id)


def test_selection_uses_clipped_active_pool_fitness_bounds(monkeypatch) -> None:
    graph = DerivationGraph()
    memory = TrajectoryMemory()

    by_fitness = {}
    for fitness in (-1000.0, 0.0, 5.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0):
        _, trajectory = _add_initial(graph=graph, memory=memory, fitness=fitness)
        by_fitness[fitness] = trajectory

    _, archived = _add_initial(graph=graph, memory=memory, fitness=-10000.0)
    memory.archive(archived.id)
    graph.add_node(code="def solve(): return 10000", idea="untracked outlier",
                   fitness=10000.0, is_valid=True)

    monkeypatch.setattr("llm4ad.method.traceaad.value.random.random", lambda: 0.0)
    select_trajectory(
        memory=memory,
        graph=graph,
        maximize=True,
        iteration=0,
        max_iter=100,
        w=ValueWeights(fitness_clip_quantile=0.1),
    )

    # The 10th/90th active-pool percentiles are 0 and 16. Extreme values clip.
    assert memory.get_trajectory(by_fitness[-1000.0].id).value.quality == 0.0
    assert memory.get_trajectory(by_fitness[17.0].id).value.quality == 1.0
    assert memory.get_trajectory(by_fitness[5.0].id).value.quality == pytest.approx(5.0 / 16.0)


def test_endpoint_quality_gates_heroic_recovery_potential() -> None:
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    root, root_trajectory = _add_initial(graph=graph, memory=memory, fitness=-100.0)
    recovered = graph.add_node(
        code="def solve(): return 5", idea="large recovery to a mediocre endpoint",
        fitness=5.0, is_valid=True,
    )
    edge = graph.add_edge(parent_id=root.id, child_id=recovered.id, action="recover")
    recovery_trajectory = memory.extend(
        trajectory_id=root_trajectory.id,
        parent_id=root.id,
        child_id=recovered.id,
        edge_id=edge.id,
    )
    _, elite_trajectory = _add_initial(graph=graph, memory=memory, fitness=10.0)
    weights = ValueWeights(
        potential_quality_floor=0.5,
        w_quality=0.30,
        w_potential=0.25,
        w_diversity=0.0,
        w_novelty=0.0,
    )

    recovery_value = compute_value_vec(
        trajectory=recovery_trajectory,
        graph=graph,
        active_others=(elite_trajectory,),
        fmin=0.0,
        fmax=10.0,
        maximize=True,
        w=weights,
    )
    elite_value = compute_value_vec(
        trajectory=elite_trajectory,
        graph=graph,
        active_others=(recovery_trajectory,),
        fmin=0.0,
        fmax=10.0,
        maximize=True,
        w=weights,
    )

    assert recovery_value.quality == 0.5
    assert recovery_value.potential == 0.0
    assert scalarize(recovery_value, weights) < scalarize(elite_value, weights)


def test_ucb_keeps_a_late_floor_and_grows_during_stagnation() -> None:
    late_bonus = ucb_bonus(
        visit_count=0,
        total_visits=100,
        c0=0.4,
        iteration=100,
        max_iter=100,
        ucb_floor=0.05,
    )
    stagnant_bonus = ucb_bonus(
        visit_count=0,
        total_visits=100,
        c0=0.4,
        iteration=100,
        max_iter=100,
        ucb_floor=0.05,
        stagnation=50,
        stagnation_boost=0.2,
    )

    assert late_bonus > 0.0
    assert stagnant_bonus > late_bonus


def _elite_under_ucb_pressure():
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    _, elite = _add_initial(graph=graph, memory=memory, fitness=10.0)
    _, unvisited = _add_initial(graph=graph, memory=memory, fitness=9.0)
    for _ in range(100):
        memory.record_visit(elite.id)
    weights = ValueWeights(
        w_quality=0.0,
        w_potential=0.0,
        w_diversity=0.0,
        w_novelty=0.0,
        c0=1.0,
        ucb_floor=0.0,
        top_k=1,
    )
    return graph, memory, elite, unvisited, weights


def test_elite_has_a_minimum_direct_sampling_probability(monkeypatch) -> None:
    graph, memory, elite, _, weights = _elite_under_ucb_pressure()
    weights = ValueWeights(**{
        **weights.__dict__,
        "elite_sampling_prob": 1.0,
    })
    monkeypatch.setattr("llm4ad.method.traceaad.value.random.random", lambda: 0.0)

    selected = select_trajectory(
        memory=memory,
        graph=graph,
        maximize=True,
        iteration=0,
        max_iter=100,
        w=weights,
    )

    assert selected.id == elite.id


def test_elite_remains_in_candidate_pool_when_outside_top_k(monkeypatch) -> None:
    graph, memory, elite, _, weights = _elite_under_ucb_pressure()
    weights = ValueWeights(**{
        **weights.__dict__,
        "elite_sampling_prob": 0.0,
    })
    monkeypatch.setattr("llm4ad.method.traceaad.value.random.random", lambda: 1.0)

    selected = select_trajectory(
        memory=memory,
        graph=graph,
        maximize=True,
        iteration=0,
        max_iter=100,
        w=weights,
    )

    assert selected.id == elite.id


def test_candidate_pool_covers_each_active_island(monkeypatch) -> None:
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    _add_initial(graph=graph, memory=memory, fitness=3.0, island_id=0)
    _add_initial(graph=graph, memory=memory, fitness=2.0, island_id=1)
    _, island_two = _add_initial(graph=graph, memory=memory, fitness=1.0, island_id=2)
    weights = ValueWeights(
        w_quality=0.0,
        w_potential=0.0,
        w_diversity=0.0,
        w_novelty=0.0,
        c0=0.0,
        ucb_floor=0.0,
        top_k=1,
        island_top_k=1,
        elite_sampling_prob=0.0,
    )
    monkeypatch.setattr("llm4ad.method.traceaad.value.random.random", lambda: 1.0)

    selected = select_trajectory(
        memory=memory,
        graph=graph,
        maximize=True,
        iteration=0,
        max_iter=100,
        w=weights,
    )

    assert selected.id == island_two.id


def test_selection_samples_each_unique_path_only_once(monkeypatch) -> None:
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    _, original = _add_initial(graph=graph, memory=memory, fitness=2.0, island_id=0)
    memory.create_initial(node_id=original.endpoint_id, island_id=1)
    memory.create_initial(node_id=original.endpoint_id, island_id=2)
    _add_initial(graph=graph, memory=memory, fitness=1.0, island_id=3)
    captured: list[int] = []

    def capture(scored, *, temperature, rng=None):
        captured.extend(trajectory.id for trajectory, _, _ in scored)
        return scored[0][0]

    monkeypatch.setattr("llm4ad.method.traceaad.value._softmax_sample", capture)
    select_trajectory(
        memory=memory,
        graph=graph,
        maximize=True,
        iteration=0,
        max_iter=100,
        w=ValueWeights(
            top_k=10,
            island_top_k=1,
            elite_sampling_prob=0.0,
        ),
    )

    assert len(captured) == 2
    assert len({memory.get_trajectory(item).path_key for item in captured}) == 2


def test_default_value_weights_match_the_audit_driven_search_configuration() -> None:
    weights = ValueWeights()

    assert (
        weights.w_quality,
        weights.w_potential,
        weights.w_diversity,
        weights.w_novelty,
        weights.top_k,
    ) == (0.50, 0.20, 0.15, 0.15, 12)


def test_move_to_island_preserves_visit_count() -> None:
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    _, trajectory = _add_initial(graph=graph, memory=memory, fitness=1.0)
    for _ in range(7):
        memory.record_visit(trajectory.id)

    moved = memory.move_to_island(trajectory.id, island_id=1)

    assert moved.visit_count == 7
    assert moved.island_id == 1


def test_migration_rotates_existing_identities_and_preserves_visits() -> None:
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    _, top_zero = _add_initial(graph=graph, memory=memory, fitness=4.0, island_id=0)
    _, low_zero = _add_initial(graph=graph, memory=memory, fitness=1.0, island_id=0)
    _, top_one = _add_initial(graph=graph, memory=memory, fitness=5.0, island_id=1)
    _, low_one = _add_initial(graph=graph, memory=memory, fitness=2.0, island_id=1)
    for trajectory_id, scalar in (
        (top_zero.id, 2.0),
        (low_zero.id, 1.0),
        (top_one.id, 3.0),
        (low_one.id, 0.5),
    ):
        memory.set_value(trajectory_id, ValueVec(quality=scalar), scalar)
    for _ in range(3):
        memory.record_visit(top_zero.id)
    for _ in range(7):
        memory.record_visit(top_one.id)
    original_ids = {trajectory.id for trajectory in memory.trajectories()}

    moved = IslandsManager(n_islands=2).migrate(memory=memory, top_per_island=1)

    assert moved == 2
    assert {trajectory.id for trajectory in memory.trajectories()} == original_ids
    assert len(memory.trajectories()) == 4
    assert memory.get_trajectory(top_zero.id).island_id == 1
    assert memory.get_trajectory(top_one.id).island_id == 0
    assert memory.get_trajectory(top_zero.id).visit_count == 3
    assert memory.get_trajectory(top_one.id).visit_count == 7
    assert memory.get_trajectory(low_zero.id).island_id == 0
    assert memory.get_trajectory(low_one.id).island_id == 1


def test_island_assignment_has_a_stable_known_mapping() -> None:
    islands = IslandsManager(n_islands=4)

    assert (
        islands.assign("local_density"),
        islands.assign("sparsified_candidate"),
        islands.assign("adaptive_exponent"),
        islands.assign("nn_rank"),
    ) == (3, 1, 3, 0)


def test_pareto_survival_keeps_non_dominated_diversity_before_dominated_scalar() -> None:
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    _, diverse = _add_initial(graph=graph, memory=memory, fitness=1.0)
    _, strong = _add_initial(graph=graph, memory=memory, fitness=2.0)
    _, dominated = _add_initial(graph=graph, memory=memory, fitness=3.0)
    memory.set_value(
        diverse.id,
        ValueVec(quality=0.6, potential=0.6, diversity=1.0, novelty=1.0),
        scalar=0.8,
    )
    memory.set_value(
        strong.id,
        ValueVec(quality=1.0, potential=1.0, diversity=0.5, novelty=0.5),
        scalar=0.7,
    )
    memory.set_value(
        dominated.id,
        ValueVec(quality=0.9, potential=0.9, diversity=0.4, novelty=0.4),
        scalar=0.99,
    )

    ordered = pareto_survival_order(memory.active())

    assert [trajectory.id for trajectory in ordered] == [diverse.id, strong.id, dominated.id]
