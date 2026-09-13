import numpy as np

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray,
                     distance_matrix: np.ndarray) -> int:
    n = len(unvisited_nodes)
    if n == 0:
        return destination_node
    if n == 1:
        return int(unvisited_nodes[0])

    idxs = unvisited_nodes
    d_curr = distance_matrix[current_node, idxs]
    d_dest = distance_matrix[idxs, destination_node]

    # Cluster structure among remaining nodes
    sub = distance_matrix[np.ix_(idxs, idxs)].copy()
    np.fill_diagonal(sub, np.inf)
    n_rem = n

    # Cohesion: median distance to other remaining nodes (robust)
    cohesion = np.median(sub, axis=1)

    # Exit cost: cheapest next edge from this candidate (min over sub) plus destination
    exit_cost = np.minimum(np.min(sub, axis=1), d_dest)

    # Second nearest for orphan detection (avoid isolating last nodes)
    sorted_sub = np.sort(sub, axis=1)
    second_nearest = sorted_sub[:, 1] if n_rem > 2 else np.full(n_rem, np.inf)

    # Normalize by global scale for stability
    scale = np.mean(d_curr)
    if scale < 1e-9:
        scale = 1.0

    d_curr_n = d_curr / scale
    d_dest_n = d_dest / scale
    cohesion_n = cohesion / scale
    exit_n = exit_cost / scale
    second_n = second_nearest / scale

    # Progress in [0,1]
    n_total = distance_matrix.shape[0]
    progress = (n_total - n) / max(n_total - 1, 1)

    # Geometric (multiplicative) cost: product of factors
    # We want to MINIMIZE cost, so larger factors = worse.
    # Use log to keep numerical stability and combine additively in log-space.
    log_step = np.log(d_curr_n + 1e-6)
    log_cohesion = 0.3 * np.log(cohesion_n + 1e-6)
    log_exit = 0.4 * np.log(exit_n + 1e-6)

    # Orphan penalty only strong near the end
    orphan_weight = 0.5 * max(0.0, (progress - 0.7) / 0.3)
    log_orphan = orphan_weight * np.log(second_n + 1e-6)

    # Destination pull only in final 30% to avoid early trapping
    dest_weight = 0.0
    if progress > 0.7:
        dest_weight = 0.8 * ((progress - 0.7) / 0.3) ** 2
    log_dest = dest_weight * np.log(d_dest_n + 1e-6)

    log_cost = log_step + log_cohesion + log_exit + log_orphan + log_dest

    # Connectivity tie-breaker: prefer nodes with more nearby options (higher degree at small radius)
    connectivity = np.sum(sub < (cohesion_n * scale * 1.2), axis=1)
    tie_break = -1e-5 * connectivity  # higher connectivity => lower cost

    final = log_cost + tie_break

    if np.any(np.isnan(final)) or np.any(np.isinf(final)):
        # Fallback to pure nearest neighbor
        final = d_curr_n

    pos = int(np.argmin(final))
    return int(idxs[pos])