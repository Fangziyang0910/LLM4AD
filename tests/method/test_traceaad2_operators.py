from __future__ import annotations

from llm4ad.method.traceaad2.derivation_graph import DerivationGraph
from llm4ad.method.traceaad2.feedback import RankingModel
from llm4ad.method.traceaad2.islands import IslandsManager
from llm4ad.method.traceaad2.operators import (
    BacktrackBranchOp,
    DistillSimplifyOp,
    MechanismCrossoverOp,
    OperatorContext,
    ScaleTransferOp,
    infer_mechanism_tag,
)
from llm4ad.method.traceaad2.pattern_memory import PatternMemory
from llm4ad.method.traceaad2.schema import ValueVec
from llm4ad.method.traceaad2.trajectory_memory import TrajectoryMemory


def _context(*, graph: DerivationGraph, memory: TrajectoryMemory, selected, **fields):
    pattern_memory = fields.pop("pattern_memory", PatternMemory())
    return OperatorContext(
        graph=graph,
        memory=memory,
        pattern_memory=pattern_memory,
        ranking=RankingModel(),
        islands=IslandsManager(n_islands=1),
        selected=selected,
        maximize=True,
        **fields,
    )


def _trajectory_with_fitnesses(*fitnesses: float, complexities: tuple[int, ...] | None = None):
    graph = DerivationGraph()
    memory = TrajectoryMemory(max_trajectory_length=8)
    complexities = complexities or tuple(0 for _ in fitnesses)
    root = graph.add_node(
        code="root", idea="root", fitness=fitnesses[0], is_valid=True,
        complexity=complexities[0], mechanism_tag="local_density",
    )
    trajectory = memory.create_initial(node_id=root.id)
    for index, (fitness, complexity) in enumerate(zip(fitnesses[1:], complexities[1:]), start=1):
        child = graph.add_node(
            code=f"child-{index}", idea=f"child-{index}", fitness=fitness,
            is_valid=True, complexity=complexity, mechanism_tag="local_density",
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
    *, graph: DerivationGraph, memory: TrajectoryMemory, fitness: float,
    mechanism_tag: str, quality: float, scalar: float,
):
    node = graph.add_node(
        code=mechanism_tag,
        idea=mechanism_tag,
        fitness=fitness,
        is_valid=True,
        mechanism_tag=mechanism_tag,
    )
    trajectory = memory.create_initial(node_id=node.id)
    memory.set_value(trajectory.id, ValueVec(quality=quality), scalar)
    return memory.get_trajectory(trajectory.id)


def test_mechanism_inference_does_not_credit_removed_randomization():
    action = "Remove randomization and use a deterministic tie break."

    assert infer_mechanism_tag(action) == "other"


def test_mechanism_inference_treats_replaced_stochasticity_as_negated():
    action = "Replace the stochastic random choice with a deterministic ordering."

    assert infer_mechanism_tag(action) == "other"


def test_mechanism_inference_does_not_let_a_hint_override_explicit_removal():
    action = "Replace randomization with a deterministic ordering."

    assert infer_mechanism_tag(action, hint="randomization") == "other"


def test_mechanism_inference_prefers_observed_family_over_requested_hint():
    action = "Use local density to score the remaining nodes."

    assert infer_mechanism_tag(action, hint="adaptive_exponent") == "local_density"


def test_mechanism_inference_understands_avoid_using_as_negation():
    action = "Avoid using stochastic noise in the selection rule."

    assert infer_mechanism_tag(action) == "other"


def test_mechanism_inference_keeps_positive_randomization_after_with():
    action = "Replace the greedy rule with randomized tie breaking."

    assert infer_mechanism_tag(action) == "randomization"


def test_mechanism_inference_does_not_treat_generic_candidates_as_sparsification():
    action = "Score every candidate with its nearest-neighbor distance."

    assert infer_mechanism_tag(action) == "other"


def test_mechanism_inference_still_recognizes_an_explicit_candidate_list():
    action = "Restrict selection to a sparsified candidate list."

    assert infer_mechanism_tag(action) == "sparsified_candidate"


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


def test_simplify_uses_active_pool_relative_complexity_on_plateau():
    graph, memory, trajectory = _trajectory_with_fitnesses(
        10.0, 10.0, complexities=(10, 20),
    )
    ctx = _context(graph=graph, memory=memory, selected=trajectory)

    assert DistillSimplifyOp().trigger(ctx)


def test_simplify_does_not_use_default_robustness_as_a_trigger():
    graph, memory, trajectory = _trajectory_with_fitnesses(
        10.0, 11.0, complexities=(10, 100),
    )
    ctx = _context(graph=graph, memory=memory, selected=trajectory)

    assert not DistillSimplifyOp().trigger(ctx)


def test_simplify_can_use_global_stagnation_for_a_relatively_complex_endpoint():
    graph, memory, trajectory = _trajectory_with_fitnesses(
        10.0, 11.0, complexities=(10, 100),
    )
    ctx = _context(
        graph=graph, memory=memory, selected=trajectory, best_stagnation=5,
    )

    assert DistillSimplifyOp().trigger(ctx)


def test_scale_transfer_requires_explicit_real_generalization_evidence():
    graph, memory, trajectory = _trajectory_with_fitnesses(10.0)
    ctx = _context(
        graph=graph,
        memory=memory,
        selected=trajectory,
        has_generalization_evidence=False,
    )

    assert not ScaleTransferOp().trigger(ctx)


def test_scale_transfer_is_eligible_when_real_generalization_evidence_exists():
    graph, memory, trajectory = _trajectory_with_fitnesses(10.0)
    ctx = _context(
        graph=graph,
        memory=memory,
        selected=trajectory,
        has_generalization_evidence=True,
    )

    assert ScaleTransferOp().trigger(ctx)


def test_crossover_rejects_a_low_quality_donor_despite_high_scalar_value():
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    selected = _initial_trajectory(
        graph=graph, memory=memory, fitness=10.0,
        mechanism_tag="local_density", quality=0.9, scalar=0.9,
    )
    low_quality = _initial_trajectory(
        graph=graph, memory=memory, fitness=1.0,
        mechanism_tag="sparsified_candidate", quality=0.1, scalar=10.0,
    )
    qualified = _initial_trajectory(
        graph=graph, memory=memory, fitness=8.0,
        mechanism_tag="edge_contrast", quality=0.8, scalar=0.6,
    )
    ctx = _context(graph=graph, memory=memory, selected=selected)

    MechanismCrossoverOp().select_base(ctx)

    assert ctx.hints["donor_id"] == qualified.id
    assert ctx.hints["donor_id"] != low_quality.id


def test_crossover_prefers_operator_conditioned_successful_donor_mechanisms():
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    selected = _initial_trajectory(
        graph=graph, memory=memory, fitness=10.0,
        mechanism_tag="local_density", quality=0.9, scalar=0.9,
    )
    failed_donor = _initial_trajectory(
        graph=graph, memory=memory, fitness=8.0,
        mechanism_tag="edge_contrast", quality=0.8, scalar=0.8,
    )
    successful_donor = _initial_trajectory(
        graph=graph, memory=memory, fitness=8.0,
        mechanism_tag="sparsified_candidate", quality=0.8, scalar=0.8,
    )
    patterns = PatternMemory()
    for support_id in range(3):
        patterns.record_mechanism_outcome(
            operator="mechanism_crossover",
            mechanism_tag="edge_contrast",
            support_id=support_id,
            success=False,
            iteration=support_id,
        )
        patterns.record_mechanism_outcome(
            operator="mechanism_crossover",
            mechanism_tag="sparsified_candidate",
            support_id=10 + support_id,
            success=True,
            iteration=support_id,
        )
    ctx = _context(
        graph=graph,
        memory=memory,
        selected=selected,
        pattern_memory=patterns,
    )

    MechanismCrossoverOp().select_base(ctx)

    assert ctx.hints["donor_id"] == successful_donor.id
    assert ctx.hints["donor_id"] != failed_donor.id


class _ConditionedAntiPatternMemory(PatternMemory):
    def mechanism_improve_rate(
        self, mechanism_tag: str, *, operator: str | None = None,
    ) -> float | None:
        return 1.0 if mechanism_tag == "edge_contrast" else 0.5

    def is_anti_pattern(
        self, mechanism_tag: str, *, operator: str | None = None,
    ) -> bool:
        return operator == "mechanism_crossover" and mechanism_tag == "edge_contrast"


def test_crossover_excludes_an_operator_conditioned_anti_pattern():
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    selected = _initial_trajectory(
        graph=graph, memory=memory, fitness=10.0,
        mechanism_tag="local_density", quality=0.9, scalar=0.9,
    )
    anti_pattern = _initial_trajectory(
        graph=graph, memory=memory, fitness=8.0,
        mechanism_tag="edge_contrast", quality=0.8, scalar=0.8,
    )
    qualified = _initial_trajectory(
        graph=graph, memory=memory, fitness=8.0,
        mechanism_tag="sparsified_candidate", quality=0.8, scalar=0.8,
    )
    ctx = _context(
        graph=graph,
        memory=memory,
        selected=selected,
        pattern_memory=_ConditionedAntiPatternMemory(),
    )

    MechanismCrossoverOp().select_base(ctx)

    assert ctx.hints["donor_id"] == qualified.id
    assert ctx.hints["donor_id"] != anti_pattern.id


class _LegacyPatternMemory:
    def mechanism_improve_rate(self, mechanism_tag: str) -> float | None:
        return None

    def is_anti_pattern(self, mechanism_tag: str) -> bool:
        return False


def test_crossover_remains_compatible_with_legacy_pattern_memory_api():
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    selected = _initial_trajectory(
        graph=graph, memory=memory, fitness=10.0,
        mechanism_tag="local_density", quality=0.9, scalar=0.9,
    )
    donor = _initial_trajectory(
        graph=graph, memory=memory, fitness=8.0,
        mechanism_tag="sparsified_candidate", quality=0.8, scalar=0.8,
    )
    ctx = _context(
        graph=graph,
        memory=memory,
        selected=selected,
        pattern_memory=_LegacyPatternMemory(),
    )

    MechanismCrossoverOp().select_base(ctx)

    assert ctx.hints["donor_id"] == donor.id
