"""Sample complete archived records; scores define coverage, not algorithm classes."""

import math

CONTEXT_POLICIES = ('ancestor_history', 'uniform_trajectory_v1', 'sampled_trajectory_v1')
MAX_FIT_ATTEMPTS = 32


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


def sample_references(nodes, parent, rng, *, limit, policy, fits):
    if limit == 0:
        return [], {'quality_boundaries': [], 'reference_layers': {},
                    'reference_fit_rejections': [], 'reference_attempts': [],
                    'reference_shortfall': 0}
    groups = {}
    for node in nodes:
        if math.isfinite(node.fitness) and node.code != parent.code:
            groups.setdefault(node.code, []).append(node)
    # Choose one intact observation per exact code, without best-score bias.
    candidates = [rng.choice(records) for records in groups.values()]
    rejected = []
    layers, boundaries = quality_layers(candidates)
    remaining = {}
    for node in candidates:
        remaining.setdefault(layers[node.id], []).append(node)
    selected, used, attempts = [], set(), []
    for _ in range(limit):
        layer = None
        for _ in range(MAX_FIT_ATTEMPTS):
            available = [layer for layer, pool in remaining.items() if pool]
            if not available:
                break
            if policy == 'sampled_trajectory_v1':
                if layer is None or not remaining[layer]:
                    layer = rng.choice([item for item in available if item not in used] or available)
                pool = remaining[layer]
                node = pool.pop(rng.randrange(len(pool)))
            else:
                node = rng.choice([node for pool in remaining.values() for node in pool])
                layer = layers[node.id]
                remaining[layer].remove(node)
            accepted = fits([*selected, node])
            if not accepted:
                rejected.append(node.id)
            attempts.append({'node_id': node.id, 'layer': layer, 'accepted': accepted})
            if accepted:
                selected.append(node)
                used.add(layer)
                break
        else:
            break
        if not any(remaining.values()):
            break
    return selected, {
        'quality_boundaries': boundaries,
        'reference_layers': {str(node.id): layers[node.id] for node in selected},
        'reference_fit_rejections': rejected,
        'reference_attempts': attempts,
        'reference_shortfall': limit - len(selected),
    }
