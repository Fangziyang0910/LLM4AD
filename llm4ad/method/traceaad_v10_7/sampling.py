"""Select complete archived records for a concrete algorithm-design task."""

import ast
from functools import lru_cache
import math

MAX_FIT_ATTEMPTS = 32

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


def _unique_candidates(nodes, parent, rng):
    groups = {}
    for node in nodes:
        if math.isfinite(node.fitness) and node.code != parent.code:
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


def sample_task_evidence(nodes, parent, rng, *, operator, limit, fits):
    """Choose role-specific evidence, with at most two references.

    ``fits`` receives ``(references, donor, roles)``. Structural signatures only
    prioritize alternatives; they never declare two algorithms equivalent.
    """
    desired = min(limit, 2 if operator == 'Fuse' else 1)
    empty = {
        'quality_boundaries': [], 'reference_layers': {}, 'reference_roles': {},
        'reference_fit_rejections': [], 'reference_attempts': [],
        'reference_shortfall': desired,
    }
    if desired == 0:
        return [], None, empty

    candidates = _unique_candidates(nodes, parent, rng)
    if not candidates:
        return [], None, empty
    all_layers, boundaries = quality_layers(candidates)
    parent_signature = structure_signature(parent.code)
    different = [node for node in candidates
                 if structure_signature(node.code) != parent_signature]
    similar = [node for node in candidates if node not in different]
    selected, donor, rejected, attempts, roles = [], None, [], [], {}

    def try_nodes(ordered, role, donor_candidate=False, attempt_start=None):
        nonlocal donor
        attempt_start = len(attempts) if attempt_start is None else attempt_start
        remaining_attempts = MAX_FIT_ATTEMPTS - (len(attempts) - attempt_start)
        for node in ordered[:remaining_attempts]:
            proposed = [*selected, node]
            proposed_donor = node if donor_candidate else donor
            proposed_roles = {**roles, node.id: role}
            accepted = fits(proposed, proposed_donor, proposed_roles)
            attempts.append({
                'node_id': node.id, 'layer': all_layers[node.id],
                'role': role, 'accepted': accepted,
            })
            if accepted:
                selected.append(node)
                roles[node.id] = role
                if donor_candidate:
                    donor = node
                return True
            rejected.append(node.id)
        return False

    if operator == 'Refine':
        attempt_start = len(attempts)
        adjacent = [node for node in candidates
                    if node.id == parent.parent_id or node.parent_id == parent.id]
        rng.shuffle(adjacent)
        accepted = try_nodes(adjacent, 'formation_contrast', attempt_start=attempt_start)
        if not accepted:
            remaining = [node for node in candidates if node not in adjacent]
            ordered = _layered_order(remaining, rng)
            try_nodes(ordered, 'archive_contrast', attempt_start=attempt_start)
    elif operator == 'Pivot':
        preferred = _layered_order(different, rng)
        fallback = _layered_order(similar, rng)
        try_nodes([*preferred, *fallback], 'alternative_reference')
    elif operator == 'Fuse':
        # Quality is the primary donor condition; structural difference only
        # breaks the order within the same global quality layer.
        donor_order = []
        for layer in ('high', 'middle', 'low'):
            for group in (different, similar):
                pool = [node for node in group if all_layers[node.id] == layer]
                rng.shuffle(pool)
                donor_order.extend(pool)
        if try_nodes(donor_order, 'transfer_source', donor_candidate=True) \
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
    }
