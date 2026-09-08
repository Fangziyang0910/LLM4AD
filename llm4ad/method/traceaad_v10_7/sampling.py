"""Select complete archived records for a concrete algorithm-design task."""

import ast
from functools import lru_cache
import math

MAX_FIT_ATTEMPTS = 32
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


def _layered_order(candidates, rng):
    layers, _ = quality_layers(candidates)
    pools = {}
    for node in candidates:
        pools.setdefault(layers[node.id], []).append(node)
    for pool in pools.values():
        rng.shuffle(pool)
    names = list(pools)
    rng.shuffle(names)
    return [node for name in names for node in pools[name]]


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


def _soft_structure_weights(candidates, parent, operator):
    """Mix a task base distribution with a bounded structure preference."""
    layers, _ = quality_layers(candidates)
    if operator == 'Pivot':
        layer_sizes = {}
        for layer in layers.values():
            layer_sizes[layer] = layer_sizes.get(layer, 0) + 1
        present = len(layer_sizes)
        base = [1 / (present * layer_sizes[layers[node.id]]) for node in candidates]
    elif operator == 'Fuse':
        ranked_scores = {score: rank for rank, score in enumerate(
            sorted({node.fitness for node in candidates}), 1
        )}
        raw = [ranked_scores[node.fitness] for node in candidates]
        total = sum(raw)
        base = [weight / total for weight in raw]
    else:
        raise ValueError(f'unsupported soft-structure operator: {operator}')

    parent_signature = structure_signature(parent.code)
    preferred = [index for index, node in enumerate(candidates)
                 if structure_signature(node.code) != parent_signature]
    if not preferred:
        return base
    structure_mass = STRUCTURE_PREFERENCE / len(preferred)
    return [
        (1 - STRUCTURE_PREFERENCE) * probability
        + (structure_mass if index in preferred else 0)
        for index, probability in enumerate(base)
    ]


def _evidence_relation(node, parent, role, node_by_id):
    if role == 'formation_evidence':
        source, target = node, parent
    elif role == 'development_evidence':
        source, target = parent, node
    else:
        return {
            'kind': 'archive_reference', 'base_id': parent.id,
            'reference_id': node.id, 'direct_generation_relation': False,
        }

    relation = {
        'kind': role, 'source_id': source.id, 'target_id': target.id,
        'operator': target.operator, 'source_fitness': source.fitness,
        'target_fitness': target.fitness,
        'fitness_delta': target.fitness - source.fitness,
        'direct_generation_relation': True,
    }
    if target.donor_id is not None:
        historical_donor = node_by_id.get(target.donor_id)
        relation['historical_donor_id'] = target.donor_id
        relation['historical_donor_fitness'] = (
            historical_donor.fitness if historical_donor is not None else None
        )
    return relation


def sample_task_evidence(nodes, parent, rng, *, operator, limit, fits):
    """Choose role-specific evidence, with at most two references.

    ``fits`` receives ``(references, donor, roles, relations)``. Structural
    signatures provide a bounded preference, never an equivalence claim.
    """
    desired = min(limit, 2 if operator == 'Fuse' else 1)
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
    node_by_id = {node.id: node for node in nodes}
    if operator == 'Refine':
        # Relationship is established before code deduplication so an unrelated
        # duplicate can never erase the true predecessor or child record.
        formation = _deduplicate_records(
            [node for node in eligible if node.id == parent.parent_id], rng,
        )
        development = _deduplicate_records(
            [node for node in eligible if node.parent_id == parent.id], rng,
        )
        candidates = [*formation, *development]
    else:
        candidates = _deduplicate_records(eligible, rng)
    if not candidates:
        return [], None, empty
    all_layers, boundaries = quality_layers(candidates)
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
            relation = _evidence_relation(node, parent, role, node_by_id)
            proposed_relations = [*relations, relation]
            accepted = fits(
                proposed, proposed_donor, proposed_roles, proposed_relations,
            )
            attempts.append({
                'node_id': node.id, 'layer': all_layers[node.id],
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
                relations.append(relation)
                if donor_candidate:
                    donor = node
                return True
            rejected.append(node.id)
        return False

    if operator == 'Refine':
        attempt_start = len(attempts)
        pools = []
        if formation:
            rng.shuffle(formation)
            pools.append(('formation_evidence', formation))
        if development:
            rng.shuffle(development)
            pools.append(('development_evidence', development))
        rng.shuffle(pools)
        for role, pool in pools:
            if try_nodes(pool, role, attempt_start=attempt_start):
                break
    elif operator == 'Pivot':
        weights = _soft_structure_weights(candidates, parent, operator)
        try_nodes(
            _weighted_order(candidates, weights, rng), 'alternative_reference',
            selection_weights={node.id: weight for node, weight in zip(candidates, weights)},
        )
    elif operator == 'Fuse':
        weights = _soft_structure_weights(candidates, parent, operator)
        donor_order = _weighted_order(candidates, weights, rng)
        if try_nodes(
                donor_order, 'transfer_source', donor_candidate=True,
                selection_weights={node.id: weight
                                   for node, weight in zip(candidates, weights)}) \
                and desired > 1:
            remaining = [node for node in candidates if node.id != donor.id]
            ordered = _layered_order(remaining, rng)
            try_nodes(ordered, 'auxiliary_contrast')
    else:
        raise ValueError(f'unsupported operator: {operator}')

    return selected, donor, {
        'quality_boundaries': boundaries,
        'reference_layers': {str(node.id): all_layers[node.id] for node in selected},
        'reference_roles': {str(node_id): role for node_id, role in roles.items()},
        'reference_fit_rejections': rejected,
        'reference_attempts': attempts,
        'reference_shortfall': desired - len(selected),
        'evidence_relations': relations,
        'structure_preference': STRUCTURE_PREFERENCE,
    }
