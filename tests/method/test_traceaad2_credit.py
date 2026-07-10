from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from llm4ad.method.traceaad2.context import _patterns_block
from llm4ad.method.traceaad2.credit import step_generalization_signal
from llm4ad.method.traceaad2.portfolio import OperatorPortfolio, PortfolioWeights
from llm4ad.method.traceaad2.pattern_memory import PatternMemory
from llm4ad.method.traceaad2.schema import OperatorName
from llm4ad.method.traceaad2.derivation_graph import DerivationGraph
from llm4ad.method.traceaad2.reflection import distill
from llm4ad.method.traceaad2.trajectory_memory import TrajectoryMemory
from llm4ad.method.traceaad2.operators.novelty import NoveltyJumpOp


@dataclass
class _StubOperator:
    name: str
    role: str = "exploit"

    def trigger(self, ctx) -> bool:
        return True


def test_portfolio_records_one_batch_outcome_per_operator_iteration() -> None:
    op = _StubOperator("test_operator")
    portfolio = OperatorPortfolio((op,), PortfolioWeights())

    portfolio.update_batch(
        op=op,
        iteration=7,
        normalized_reward=0.4,
        best_valid=True,
        best_novel=False,
        best_regress=False,
        total_cost=123.0,
        global_best=True,
    )

    stats = portfolio.snapshot()[op.name]
    assert stats["n_calls"] == 1
    assert stats["mean_gain"] == pytest.approx(0.4)
    assert stats["valid_rate"] == pytest.approx(1.0)
    assert stats["novel_rate"] == pytest.approx(0.0)
    assert stats["regress_rate"] == pytest.approx(0.0)
    assert stats["mean_cost"] == pytest.approx(123.0)
    assert stats["global_best_rate"] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="already updated"):
        portfolio.update_batch(
            op=op,
            iteration=7,
            normalized_reward=-1.0,
            best_valid=False,
            best_novel=False,
            best_regress=True,
            total_cost=999.0,
        )


def test_portfolio_credit_follows_recent_outcomes_instead_of_lifetime_mean() -> None:
    op = _StubOperator("test_operator")
    portfolio = OperatorPortfolio((op,), PortfolioWeights())

    for iteration in range(8):
        portfolio.update_batch(
            op=op,
            iteration=iteration,
            normalized_reward=-1.0,
            best_valid=True,
            best_novel=False,
            best_regress=True,
            total_cost=10.0,
        )
    for iteration in range(8, 16):
        portfolio.update_batch(
            op=op,
            iteration=iteration,
            normalized_reward=1.0,
            best_valid=True,
            best_novel=True,
            best_regress=False,
            total_cost=10.0,
        )

    assert portfolio.snapshot()[op.name]["mean_gain"] > 0.5


def test_global_best_bonus_increases_an_operators_selection_probability() -> None:
    record_op = _StubOperator("record_operator")
    ordinary_op = _StubOperator("ordinary_operator")
    weights = PortfolioWeights(
        alpha=0.0,
        beta_v=0.0,
        beta_n=0.0,
        delta_r=0.0,
        delta_c=0.0,
        global_best_bonus=1.0,
        min_probability=0.0,
    )
    portfolio = OperatorPortfolio((record_op, ordinary_op), weights)
    common = dict(
        iteration=1,
        normalized_reward=0.0,
        best_valid=True,
        best_novel=False,
        best_regress=False,
        total_cost=10.0,
    )
    portfolio.update_batch(op=record_op, global_best=True, **common)
    portfolio.update_batch(op=ordinary_op, global_best=False, **common)

    probabilities = portfolio.probabilities(ctx=None, iteration=1, max_iter=10)

    assert probabilities[record_op.name] > probabilities[ordinary_op.name]


def test_near_record_bonus_increases_operator_probability() -> None:
    near = _StubOperator("near")
    ordinary = _StubOperator("ordinary")
    weights = PortfolioWeights(
        alpha=0.0,
        beta_v=0.0,
        beta_n=0.0,
        delta_r=0.0,
        delta_c=0.0,
        global_best_bonus=0.0,
        near_record_bonus=1.0,
        min_probability=0.0,
    )
    portfolio = OperatorPortfolio((near, ordinary), weights)
    common = dict(
        iteration=1,
        normalized_reward=0.0,
        best_valid=True,
        best_novel=False,
        best_regress=False,
        total_cost=0.0,
        global_best=False,
    )
    portfolio.update_batch(op=near, near_record=True, **common)
    portfolio.update_batch(op=ordinary, near_record=False, **common)

    probabilities = portfolio.probabilities(ctx=None, iteration=1, max_iter=10)

    assert probabilities[near.name] > probabilities[ordinary.name]


def test_late_phase_caps_novelty_without_removing_softmax_exploration() -> None:
    novelty = _StubOperator(OperatorName.NOVELTY)
    exploit = _StubOperator(OperatorName.ENDPOINT)
    weights = PortfolioWeights(
        alpha=10.0,
        beta_v=0.0,
        beta_n=0.0,
        delta_r=0.0,
        delta_c=0.0,
        global_best_bonus=0.0,
        min_probability=0.05,
        late_novelty_max_probability=0.2,
    )
    portfolio = OperatorPortfolio((novelty, exploit), weights)
    common = dict(
        iteration=0,
        best_valid=True,
        best_novel=False,
        best_regress=False,
        total_cost=0.0,
    )
    portfolio.update_batch(op=novelty, normalized_reward=1.0, **common)
    portfolio.update_batch(op=exploit, normalized_reward=-1.0, **common)

    early = portfolio.probabilities(ctx=None, iteration=0, max_iter=100)
    late = portfolio.probabilities(ctx=None, iteration=90, max_iter=100)

    assert min(early.values()) >= 0.05
    assert late[OperatorName.NOVELTY] == pytest.approx(0.2)


def test_temperature_floor_prevents_late_softmax_from_freezing() -> None:
    better = _StubOperator("better")
    worse = _StubOperator("worse")
    weights = PortfolioWeights(
        alpha=1.0,
        beta_v=0.0,
        beta_n=0.0,
        delta_r=0.0,
        delta_c=0.0,
        global_best_bonus=0.0,
        tau_init=0.0,
        tau_end=0.0,
        tau_floor=0.5,
        min_probability=0.0,
    )
    portfolio = OperatorPortfolio((better, worse), weights)
    common = dict(
        iteration=0,
        best_valid=True,
        best_novel=False,
        best_regress=False,
        total_cost=0.0,
    )
    portfolio.update_batch(op=better, normalized_reward=1.0, **common)
    portfolio.update_batch(op=worse, normalized_reward=-1.0, **common)

    probabilities = portfolio.probabilities(ctx=None, iteration=100, max_iter=100)

    assert probabilities[worse.name] > 0.01


def test_actual_batch_cost_remains_comparable_in_portfolio_selection() -> None:
    fast = _StubOperator("fast")
    slow = _StubOperator("slow")
    weights = PortfolioWeights(
        alpha=0.0,
        beta_v=0.0,
        beta_n=0.0,
        delta_r=0.0,
        delta_c=1.0,
        global_best_bonus=0.0,
        min_probability=0.0,
        cost_scale=120.0,
    )
    portfolio = OperatorPortfolio((fast, slow), weights)
    common = dict(
        iteration=0,
        normalized_reward=0.0,
        best_valid=True,
        best_novel=False,
        best_regress=False,
    )
    portfolio.update_batch(op=fast, total_cost=30.0, **common)
    portfolio.update_batch(op=slow, total_cost=300.0, **common)

    probabilities = portfolio.probabilities(ctx=None, iteration=0, max_iter=100)

    assert probabilities[fast.name] > probabilities[slow.name]


def test_pattern_memory_tracks_mechanism_credit_per_operator() -> None:
    memory = PatternMemory()
    memory.record_mechanism_outcome(
        operator=OperatorName.CROSSOVER,
        mechanism_tag="adaptive_exponent",
        support_id=10,
        success=True,
        iteration=1,
    )
    memory.record_mechanism_outcome(
        operator=OperatorName.CROSSOVER,
        mechanism_tag="adaptive_exponent",
        support_id=11,
        success=True,
        iteration=2,
    )
    memory.record_mechanism_outcome(
        operator=OperatorName.NOVELTY,
        mechanism_tag="adaptive_exponent",
        support_id=20,
        success=False,
        iteration=3,
    )
    memory.record_mechanism_outcome(
        operator=OperatorName.NOVELTY,
        mechanism_tag="adaptive_exponent",
        support_id=21,
        success=False,
        iteration=4,
    )

    assert memory.mechanism_attempts(
        "adaptive_exponent", operator=OperatorName.CROSSOVER
    ) == 2
    assert memory.mechanism_improve_rate(
        "adaptive_exponent", operator=OperatorName.CROSSOVER
    ) == pytest.approx(1.0)
    assert memory.mechanism_attempts(
        "adaptive_exponent", operator=OperatorName.NOVELTY
    ) == 2
    assert memory.mechanism_improve_rate(
        "adaptive_exponent", operator=OperatorName.NOVELTY
    ) == pytest.approx(0.0)
    assert memory.mechanism_improve_rate("adaptive_exponent") == pytest.approx(0.5)


def test_pattern_prompt_reports_operator_conditioned_improve_rate_and_support() -> None:
    memory = PatternMemory()
    memory.upsert_mechanism(
        mechanism_tag="local_density",
        text="density evidence",
        generalization_score=0.75,
        support_id=10,
        updated_iter=1,
    )
    for support_id, success in ((10, True), (11, False)):
        memory.record_mechanism_outcome(
            operator=OperatorName.ENDPOINT,
            mechanism_tag="local_density",
            support_id=support_id,
            success=success,
            iteration=support_id,
        )

    block = _patterns_block(memory, operator=OperatorName.ENDPOINT)

    assert "aggregate_improve_rate=0.75" in block
    assert "operator_improve_rate=0.50" in block
    assert "operator_support=2" in block
    assert "generalization=" not in block


def test_pattern_memory_replaces_stale_scores_and_keeps_real_support_ids() -> None:
    memory = PatternMemory()
    first = memory.upsert_mechanism(
        mechanism_tag="local_density",
        text="early evidence",
        generalization_score=0.9,
        support_id=101,
        updated_iter=10,
    )
    updated = memory.upsert_mechanism(
        mechanism_tag="local_density",
        text="later evidence",
        generalization_score=0.2,
        support_id=202,
        updated_iter=20,
    )

    assert updated.id == first.id
    assert updated.generalization_score == pytest.approx(0.2)
    assert updated.support_ids == (101, 202)


def test_pattern_memory_keeps_all_unique_mechanism_support_ids() -> None:
    memory = PatternMemory()

    for support_id in range(60):
        memory.upsert_mechanism(
            mechanism_tag="local_density",
            text="density evidence",
            generalization_score=0.5,
            support_id=support_id,
            updated_iter=support_id,
        )

    pattern = memory.mechanism_pattern("local_density")
    assert pattern is not None
    assert pattern.support_ids == tuple(range(60))


def test_pattern_memory_deduplicates_repeated_lessons() -> None:
    memory = PatternMemory()
    first = memory.add(
        kind="lesson",
        text="Prefer density-aware candidate pruning.",
        mechanism_tag="local_density",
        support_ids=(1,),
        generalization_score=0.6,
        confidence=0.7,
        updated_iter=10,
    )
    repeated = memory.add(
        kind="lesson",
        text="  prefer   density-aware candidate pruning. ",
        mechanism_tag="local_density",
        support_ids=(2,),
        generalization_score=0.5,
        confidence=0.8,
        updated_iter=20,
    )

    assert repeated.id == first.id
    assert repeated.support_ids == (1, 2)
    assert len(memory.top_lessons()) == 1


def test_pattern_memory_merges_differently_worded_lessons_for_one_mechanism_scope() -> None:
    memory = PatternMemory()
    first = memory.add(
        kind="lesson",
        text="Density variant A ranked strongly.",
        mechanism_tag="local_density",
        support_ids=(1,),
        updated_iter=10,
    )
    revised = memory.add(
        kind="lesson",
        text="Density variant B is the current best.",
        mechanism_tag="local_density",
        support_ids=(2,),
        updated_iter=20,
    )

    assert revised.id == first.id
    assert revised.support_ids == (1, 2)
    assert revised.text == "Density variant B is the current best."
    assert len(memory.top_lessons()) == 1


def test_step_generalization_signal_uses_per_instance_parent_child_outcomes() -> None:
    signal = step_generalization_signal(
        parent_fitness_vector=(1.0, 1.0, 1.0),
        child_fitness_vector=(2.0, 0.5, 1.0),
        maximize=True,
    )

    assert signal == pytest.approx(0.5)
    assert step_generalization_signal(
        parent_fitness_vector=None,
        child_fitness_vector=(2.0, 2.0),
        maximize=True,
    ) == 0.0


def test_distill_counts_shared_graph_edges_once_and_updates_conditioned_credit() -> None:
    graph = DerivationGraph()
    trajectories = TrajectoryMemory()
    patterns = PatternMemory()
    root = graph.add_node(code="root", idea="root", fitness=0.0, is_valid=True)
    initial = trajectories.create_initial(node_id=root.id)
    improved = graph.add_node(code="a", idea="a", fitness=1.0, is_valid=True)
    good_edge = graph.add_edge(
        parent_id=root.id,
        child_id=improved.id,
        action="add candidates",
        operator=OperatorName.CROSSOVER,
        mechanism_tag="sparsified_candidate",
        delta=1.0,
        outcome="improve",
        iteration=1,
    )
    improved_trajectory = trajectories.extend(
        trajectory_id=initial.id,
        parent_id=root.id,
        child_id=improved.id,
        edge_id=good_edge.id,
    )
    trajectories.fork(improved_trajectory.id, island_id=1)
    trajectories.fork(improved_trajectory.id, island_id=2)

    distill(
        memory=trajectories,
        graph=graph,
        pattern_memory=patterns,
        maximize=True,
        iteration=10,
        min_support=1,
    )

    assert patterns.mechanism_attempts(
        "sparsified_candidate", operator=OperatorName.CROSSOVER
    ) == 1
    first = patterns.mechanism_pattern("sparsified_candidate")
    assert first is not None
    assert first.support_ids == (good_edge.id,)
    assert first.generalization_score == pytest.approx(1.0)

    regressed = graph.add_node(code="b", idea="b", fitness=0.5, is_valid=True)
    bad_edge = graph.add_edge(
        parent_id=improved.id,
        child_id=regressed.id,
        action="bad candidates",
        operator=OperatorName.CROSSOVER,
        mechanism_tag="sparsified_candidate",
        delta=-0.5,
        outcome="regress",
        iteration=11,
    )
    regressed_trajectory = trajectories.extend(
        trajectory_id=improved_trajectory.id,
        parent_id=improved.id,
        child_id=regressed.id,
        edge_id=bad_edge.id,
    )
    trajectories.fork(regressed_trajectory.id, island_id=3)

    distill(
        memory=trajectories,
        graph=graph,
        pattern_memory=patterns,
        maximize=True,
        iteration=20,
        min_support=1,
    )

    assert patterns.mechanism_attempts(
        "sparsified_candidate", operator=OperatorName.CROSSOVER
    ) == 2
    updated = patterns.mechanism_pattern("sparsified_candidate")
    assert updated is not None
    assert updated.support_ids == (good_edge.id, bad_edge.id)
    assert updated.generalization_score == pytest.approx(0.5)


def test_distill_scopes_and_reverses_operator_anti_patterns() -> None:
    graph = DerivationGraph()
    trajectories = TrajectoryMemory()
    patterns = PatternMemory()
    root = graph.add_node(code="root", idea="root", fitness=0.0, is_valid=True)

    for i in range(5):
        child = graph.add_node(code=f"bad-{i}", idea="bad", fitness=-1.0, is_valid=True)
        graph.add_edge(
            parent_id=root.id,
            child_id=child.id,
            action="failed density rewrite",
            operator=OperatorName.ENDPOINT,
            mechanism_tag="local_density",
            delta=-1.0,
            outcome="regress",
            iteration=i,
        )
    for i in range(2):
        child = graph.add_node(code=f"cross-{i}", idea="good", fitness=1.0, is_valid=True)
        graph.add_edge(
            parent_id=root.id,
            child_id=child.id,
            action="successful density crossover",
            operator=OperatorName.CROSSOVER,
            mechanism_tag="local_density",
            delta=1.0,
            outcome="improve",
            iteration=10 + i,
        )

    distill(
        memory=trajectories,
        graph=graph,
        pattern_memory=patterns,
        maximize=True,
        iteration=20,
        min_support=2,
    )

    assert patterns.is_anti_pattern("local_density", operator=OperatorName.ENDPOINT)
    assert not patterns.is_anti_pattern("local_density", operator=OperatorName.CROSSOVER)
    assert not patterns.is_anti_pattern("local_density")

    for i in range(2):
        child = graph.add_node(code=f"recovered-{i}", idea="good", fitness=1.0, is_valid=True)
        graph.add_edge(
            parent_id=root.id,
            child_id=child.id,
            action="recovered density rewrite",
            operator=OperatorName.ENDPOINT,
            mechanism_tag="local_density",
            delta=1.0,
            outcome="improve",
            iteration=21 + i,
        )

    distill(
        memory=trajectories,
        graph=graph,
        pattern_memory=patterns,
        maximize=True,
        iteration=30,
        min_support=2,
    )

    assert not patterns.is_anti_pattern("local_density", operator=OperatorName.ENDPOINT)


def test_distill_recovers_a_global_anti_pattern_after_positive_unique_evidence() -> None:
    graph = DerivationGraph()
    trajectories = TrajectoryMemory()
    patterns = PatternMemory()
    root = graph.add_node(code="root", idea="root", fitness=0.0, is_valid=True)
    patterns.add(
        kind="anti_pattern",
        text="density underperformed in an earlier contrast",
        mechanism_tag="local_density",
        support_ids=(root.id,),
        generalization_score=0.0,
        confidence=0.6,
        updated_iter=1,
    )
    for index in range(5):
        child = graph.add_node(
            code=f"good-{index}", idea="good", fitness=1.0, is_valid=True
        )
        graph.add_edge(
            parent_id=root.id,
            child_id=child.id,
            action="successful density rewrite",
            operator=OperatorName.ENDPOINT,
            mechanism_tag="local_density",
            delta=1.0,
            outcome="improve",
            iteration=10 + index,
        )

    distill(
        memory=trajectories,
        graph=graph,
        pattern_memory=patterns,
        maximize=True,
        iteration=20,
        min_support=2,
    )

    assert not patterns.is_anti_pattern("local_density")


def test_pattern_memory_failure_cooldown_expires_and_success_resets_it() -> None:
    memory = PatternMemory()
    for support_id, iteration in ((1, 10), (2, 11)):
        memory.record_mechanism_outcome(
            operator=OperatorName.NOVELTY,
            mechanism_tag="adaptive_exponent",
            support_id=support_id,
            success=False,
            iteration=iteration,
        )

    assert memory.mechanism_in_failure_cooldown(
        "adaptive_exponent",
        operator=OperatorName.NOVELTY,
        iteration=20,
        failure_limit=2,
        cooldown=24,
    )
    assert not memory.mechanism_in_failure_cooldown(
        "adaptive_exponent",
        operator=OperatorName.NOVELTY,
        iteration=35,
        failure_limit=2,
        cooldown=24,
    )

    memory.record_mechanism_outcome(
        operator=OperatorName.NOVELTY,
        mechanism_tag="adaptive_exponent",
        support_id=3,
        success=True,
        iteration=35,
    )
    assert not memory.mechanism_in_failure_cooldown(
        "adaptive_exponent",
        operator=OperatorName.NOVELTY,
        iteration=36,
        failure_limit=2,
        cooldown=24,
    )


def test_novelty_requires_deeper_stagnation_and_respects_trigger_cooldown() -> None:
    operator = NoveltyJumpOp()
    ctx = SimpleNamespace(
        best_stagnation=5,
        iteration=20,
        pattern_memory=PatternMemory(),
        memory=TrajectoryMemory(),
        graph=DerivationGraph(),
        hints={},
    )

    assert not operator.trigger(ctx)
    ctx.best_stagnation = 12
    assert operator.trigger(ctx)
    operator.build_constraint(ctx, None)

    ctx.iteration = 27
    assert not operator.trigger(ctx)
    ctx.iteration = 28
    assert operator.trigger(ctx)


def test_novelty_family_choice_uses_fresh_start_history_and_rotates_after_failures() -> None:
    patterns = PatternMemory()
    patterns.upsert_mechanism(
        mechanism_tag="adaptive_exponent",
        text="strong in crossover",
        generalization_score=0.95,
        support_id=100,
        updated_iter=10,
    )
    for support_id in range(4):
        patterns.record_mechanism_outcome(
            operator=OperatorName.CROSSOVER,
            mechanism_tag="adaptive_exponent",
            support_id=support_id,
            success=True,
            iteration=support_id,
        )
    for support_id, iteration in ((20, 20), (21, 21)):
        patterns.record_mechanism_outcome(
            operator=OperatorName.NOVELTY,
            mechanism_tag="adaptive_exponent",
            support_id=support_id,
            success=False,
            iteration=iteration,
        )
    patterns.record_mechanism_outcome(
        operator=OperatorName.NOVELTY,
        mechanism_tag="nn_rank",
        support_id=30,
        success=True,
        iteration=20,
    )
    ctx = SimpleNamespace(
        best_stagnation=20,
        iteration=22,
        pattern_memory=patterns,
        memory=TrajectoryMemory(),
        graph=DerivationGraph(),
        hints={},
    )

    NoveltyJumpOp().build_constraint(ctx, None)

    assert ctx.hints["mechanism_tag_hint"] == "nn_rank"


def test_novelty_is_ineligible_when_every_family_is_blocked() -> None:
    patterns = PatternMemory()
    families = (
        "local_density",
        "nn_rank",
        "row_normalize",
        "edge_contrast",
        "sparsified_candidate",
        "adaptive_exponent",
        "hybrid_distance",
        "randomization",
    )
    for family in families:
        patterns.add(
            kind="anti_pattern",
            text=f"blocked {family}",
            mechanism_tag=family,
            updated_iter=1,
        )
    ctx = SimpleNamespace(
        best_stagnation=20,
        iteration=20,
        pattern_memory=patterns,
        memory=TrajectoryMemory(),
        graph=DerivationGraph(),
        hints={},
    )

    assert not NoveltyJumpOp().trigger(ctx)
