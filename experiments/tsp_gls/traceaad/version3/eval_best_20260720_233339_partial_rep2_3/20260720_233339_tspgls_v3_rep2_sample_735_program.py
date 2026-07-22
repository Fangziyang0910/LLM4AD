import numpy as np

def update_edge_distance(edge_distance: np.ndarray, local_opt_tour: np.ndarray, edge_n_used: np.ndarray) -> np.ndarray:
    """
    Design a novel algorithm to update the distance matrix.

    Args:
    edge_distance: A matrix of the distance.
    local_opt_tour: An array of the local optimal tour of IDs.
    edge_n_used: A matrix of the number of each edge used during permutation.

    Return:
    updated_edge_distance: A matrix of the updated distance.
    """
    updated_edge_distance = edge_distance.copy()
    n = len(local_opt_tour)

    if n == 0:
        return updated_edge_distance

    # Extract edges from the local optimal tour and calculate current tour cost
    tour_edges = []
    current_tour_cost = 0.0

    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        # Ensure consistent ordering (u < v) for undirected graph representation
        if u < v:
            tour_edges.append((u, v))
        else:
            tour_edges.append((v, u))

        current_tour_cost += edge_distance[u, v]

    # Calculate baseline estimate
    # Baseline: Average distance of edges in the current tour
    if n > 0:
        baseline_estimate = current_tour_cost / n
    else:
        baseline_estimate = 1.0 # Fallback

    # Avoid division by zero or very small baselines
    if baseline_estimate < 1e-9:
        baseline_estimate = 1.0

    # Calculate stagnation ratio as the ratio of current tour cost to baseline estimate
    # If the tour is much worse than the average edge, ratio is high.
    # If the tour is much better, ratio is low.
    stagnation_ratio = current_tour_cost / baseline_estimate

    # Clamp stagnation ratio between 0.5 and 2.0 to prevent erratic penalties
    clamped_stagnation_ratio = np.clip(stagnation_ratio, 0.5, 2.0)

    # Refined decay factor calculation using logarithmic scaling with a tunable constant alpha
    # alpha controls the sensitivity of the decay to the stagnation ratio
    alpha = 1.0
    base = 1.0

    # decay_factor = base / (1.0 + alpha * log(1.0 + clamped_stagnation_ratio))
    # This ensures smoother changes in decay as the stagnation ratio varies.
    decay_factor = base / (1.0 + alpha * np.log(1.0 + clamped_stagnation_ratio))

    # Ensure decay factor is positive and <= 1
    # The logarithmic function grows slowly, so decay_factor will vary smoothly.
    # We clamp it to ensure stability within reasonable bounds [0.1, 1.0]
    decay_factor = np.clip(decay_factor, 0.1, 1.0)

    # Apply uniform decay to all edges
    updated_edge_distance *= decay_factor

    # Apply penalties to edges in the current tour
    # Penalty formula: lambda * (edge_cost / baseline) / (usage_count + 1)
    # This makes penalties proportional to the edge's contribution to the tour cost relative to the average
    penalty_scale = 0.5

    for (u, v) in tour_edges:
        edge_cost = edge_distance[u, v]
        usage_count = edge_n_used[u, v]

        # Calculate individual edge ratio relative to baseline
        # If an edge is much more expensive than the average, it gets a higher penalty factor
        edge_ratio = edge_cost / baseline_estimate

        # Calculate penalty
        # Higher edge cost relative to baseline -> higher penalty
        # Higher usage -> lower penalty
        penalty = penalty_scale * edge_ratio / (usage_count + 1.0)

        # Add penalty to the distance matrix
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance
