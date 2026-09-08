"""Select complete archived records for a concrete algorithm-design task."""

import ast
from functools import lru_cache
import math

MAX_FIT_ATTEMPTS = 32
#: Multiplicative weak hint: a structure-different candidate gets at most
#: (1 + STRUCTURE_PREFERENCE) times its task weight, never an independent
#: probability mass. A near-zero task weight stays near zero after the bonus.
STRUCTURE_PREFERENCE = 0.25

#: The only context mechanism. Kept as an explicit identity in checkpoints,
#: events and manifests; retired policies are not valid values.
CONTEXT_POLICY = 'task_evidence_v1'


def quality_layers(nodes):
    scores = sorted(node.fitness for node in nodes)
    if not scores:
        return {}, []

    def quantile(fraction):
        position = (len(scores) - 1) * fraction
        lo = int(position)
        hi = min(lo + 1, len(scores) - 1)
        weight = position - lo
        return (1 - weight) * scores[lo] + weight * scores[hi]

    lower, upper = quantile(1 / 3), quantile(2 / 3)
    return {
        node.id: 'low' if node.fitness <= lower else
        'middle' if node.fitness <= upper else 'high'
        for node in nodes
    }, [lower, upper]


@lru_cache(maxsize=None)
def structure_signature(code):
    """A conservative control-structure hint, never a semantic identity claim."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    structural = (
        ast.FunctionDef, ast.AsyncFunctionDef, ast.For, ast.AsyncFor, ast.While,
        ast.If, ast.Try, ast.With, ast.AsyncWith, ast.Match, ast.ListComp,
        ast.SetComp, ast.DictComp, ast.GeneratorExp,
    )
    return tuple(type(item).__name__ for item in ast.walk(tree) if isinstance(item, structural))


def _deduplicate_records(nodes, rng):
    groups = {}
    for node in nodes:
        groups.setdefault(node.code, []).append(node)
    # Keep Idea, Code and fitness from one real record; never splice duplicates.
    return [rng.choice(records) for records in groups.values()]


def _task_base_weights(candidates, layers, operator):
    """Task-only sampling distribution each operator is built on."""
    if operator == 'Pivot':
        layer_sizes = {}
        for layer in layers.values():
            layer_sizes[layer] = layer_sizes.get(layer, 0) + 1
        present = len(layer_sizes)
        return [1 / (present * layer_sizes[layers[node.id]]) for node in candidates]
    if operator == 'Fuse':
        ranked_scores = {score: rank for rank, score in enumerate(
            sorted({node.fitness for node in candidates}), 1
        )}
        raw = [ranked_scores[node.fitness] for node in candidates]
        total = sum(raw)
        return [weight / total for weight in raw]
    raise ValueError(f'unsupported structure-bonus operator: {operator}')


def _bounded_structure_weights(candidates, parent, operator, layers):
    """Multiply the task weight of structure-different candidates by (1 + λ).

    Every candidate keeps full support through its positive task weight, so a
    tiny structure-different set can no longer capture an independent share of
    the total probability mass.
    """
    base = _task_base_weights(candidates, layers, operator)
    parent_signature = structure_signature(parent.code)
    boosted = [
        weight * (1 + STRUCTURE_PREFERENCE)
        if structure_signature(node.code) != parent_signature else weight
        for weight, node in zip(base, candidates)
    ]
    total = sum(boosted)
    return [weight / total for weight in boosted]


def _weighted_order(candidates, weights, rng, limit=MAX_FIT_ATTEMPTS):
    """Sample without replacement while preserving every positive weight."""
    pool = list(candidates)
    pool_weights = list(weights)
    ordered = []
    while pool and len(ordered) < limit:
        threshold = rng.random() * sum(pool_weights)
        cumulative = 0.0
        for index, weight in enumerate(pool_weights):
            cumulative += weight
            if threshold <= cumulative:
                break
        ordered.append(pool.pop(index))
        pool_weights.pop(index)
    return ordered


def _evidence_relation(node, parent, role):
    """Describe a real direct generation edge, or None for archive references.

    Pivot/Fuse references are already recorded through reference_roles and
    context_program_roles; giving them a second non-edge relation object only
    duplicates that fact and drifts from the documented analysis contract.
    """
    if role == 'formation_evidence':
        source, target = node, parent
    elif role == 'development_evidence':
        source, target = parent, node
    else:
        return None

    relation = {
        'kind': role, 'source_id': source.id, 'target_id': target.id,
        'operator': target.operator, 'source_fitness': source.fitness,
        'target_fitness': target.fitness,
        'fitness_delta': target.fitness - source.fitness,
        'direct_generation_relation': True,
    }
    if target.donor_id is not None:
        relation['historical_donor_id'] = target.donor_id
    return relation


def sample_task_evidence(nodes, parent, rng, *, operator, limit, fits):
    """Choose role-specific evidence, with at most one reference.

    ``fits`` receives ``(references, donor, roles, relations)``. Structural
    signatures only scale the task weight by at most (1 + λ), never an
    equivalence claim. Quality layers only define the Pivot/Fuse task base
    distribution; Refine evidence is chosen by generation relationship alone.
    """
    desired = min(limit, 1)
    empty = {
        'quality_boundaries': [], 'reference_layers': {}, 'reference_roles': {},
        'reference_fit_rejections': [], 'reference_attempts': [],
        'reference_shortfall': desired, 'evidence_relations': [],
        'structure_preference': STRUCTURE_PREFERENCE,
    }
    if desired == 0:
        return [], None, empty

    eligible = [node for node in nodes
                if math.isfinite(node.fitness) and node.code != parent.code]
    if operator == 'Refine':
        # Relationship is established before code deduplication so an unrelated
        # duplicate can never erase the true predecessor or child record.
        # Pools are tried in fixed order: the formation edge first, then a
        # previous Refine attempt from the same base, then children made by
        # other operators. Formation therefore always gets its capacity check
        # before any development pool can consume the attempt budget.
        formation = _deduplicate_records(
            [node for node in eligible if node.id == parent.parent_id], rng,
        )
        development = [node for node in eligible if node.parent_id == parent.id]
        same_operator = _deduplicate_records(
            [node for node in development if getattr(node, 'operator', None) == 'Refine'],
            rng,
        )
        other_operator = _deduplicate_records(
            [node for node in development if getattr(node, 'operator', None) != 'Refine'],
            rng,
        )
        pools = [
            ('formation_evidence', formation),
            ('development_evidence', same_operator),
            ('development_evidence', other_operator),
        ]
        candidates = [*formation, *same_operator, *other_operator]
        all_layers, boundaries = {}, []
    else:
        pools = []
        candidates = _deduplicate_records(eligible, rng)
    if not candidates:
        return [], None, empty
    if operator in ('Pivot', 'Fuse'):
        all_layers, boundaries = quality_layers(candidates)
    else:
        all_layers, boundaries = {}, []
    selected, donor, rejected, attempts, roles, relations = [], None, [], [], {}, []

    def try_nodes(ordered, role, donor_candidate=False, attempt_start=None,
                   selection_weights=None):
        nonlocal donor
        attempt_start = len(attempts) if attempt_start is None else attempt_start
        remaining_attempts = MAX_FIT_ATTEMPTS - (len(attempts) - attempt_start)
        for node in ordered[:remaining_attempts]:
            proposed = [*selected, node]
            proposed_donor = node if donor_candidate else donor
            proposed_roles = {**roles, node.id: role}
            relation = _evidence_relation(node, parent, role)
            proposed_relations = (
                [*relations, relation] if relation is not None else list(relations)
            )
            accepted = fits(
                proposed, proposed_donor, proposed_roles, proposed_relations,
            )
            attempts.append({
                'node_id': node.id, 'layer': all_layers.get(node.id),
                'role': role, 'accepted': accepted,
                'structure_different': (
                    structure_signature(node.code) != structure_signature(parent.code)
                ),
                **({'selection_weight': selection_weights[node.id]}
                   if selection_weights else {}),
            })
            if accepted:
                selected.append(node)
                roles[node.id] = role
                if relation is not None:
                    relations.append(relation)
                if donor_candidate:
                    donor = node
                return True
            rejected.append(node.id)
        return False

    if operator == 'Refine':
        attempt_start = len(attempts)
        for role, pool in pools:
            if not pool:
                continue
            rng.shuffle(pool)
            if try_nodes(pool, role, attempt_start=attempt_start):
                break
    elif operator == 'Pivot':
        weights = _bounded_structure_weights(candidates, parent, operator, all_layers)
        try_nodes(
            _weighted_order(candidates, weights, rng), 'alternative_reference',
            selection_weights={node.id: weight for node, weight in zip(candidates, weights)},
        )
    elif operator == 'Fuse':
        weights = _bounded_structure_weights(candidates, parent, operator, all_layers)
        try_nodes(
            _weighted_order(candidates, weights, rng), 'transfer_source',
            donor_candidate=True,
            selection_weights={node.id: weight
                               for node, weight in zip(candidates, weights)},
        )
    else:
        raise ValueError(f'unsupported operator: {operator}')

    return selected, donor, {
        'quality_boundaries': boundaries,
        'reference_layers': {str(node.id): all_layers[node.id] for node in selected}
        if operator in ('Pivot', 'Fuse') else {},
        'reference_roles': {str(node_id): role for node_id, role in roles.items()},
        'reference_fit_rejections': rejected,
        'reference_attempts': attempts,
        'reference_shortfall': desired - len(selected),
        'evidence_relations': relations,
        'structure_preference': STRUCTURE_PREFERENCE,
    }
