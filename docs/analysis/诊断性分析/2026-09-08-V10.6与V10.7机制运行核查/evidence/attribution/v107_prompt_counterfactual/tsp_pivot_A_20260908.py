import numpy as np

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Selects the next node using a spectral cohesion and lookahead heuristic.
    
    The score is a weighted sum of:
    1. Immediate distance from current node.
    2. Distance to destination (return cost), weighted heavily at late stages.
    3. Cohesion: Average distance to all other unvisited nodes (prevents stragglers).
    4. Lookahead: Estimated cost of the next hop (min(d(i,j) + d(j,dest))).
    5. Regret: A convex penalty on the gap from the best score to avoid ties.
    """
    if len(unvisited_nodes) == 0:
        return destination_node
        
    n_remaining = len(unvisited_nodes)
    
    if n_remaining == 1:
        return int(unvisited_nodes[0])

    n_total = distance_matrix.shape[0]
    if n_total <= 1:
        return destination_node

    # 1. Basic Distances
    d_curr = distance_matrix[current_node, unvisited_nodes]
    d_dest = distance_matrix[unvisited_nodes, destination_node]

    # 2. Sub-matrix for unvisited nodes
    # We need distances between all pairs of unvisited nodes
    sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)].copy()
    
    # Set diagonal to 0 for average calculation, but we need inf for min-lookahead later
    # Let's compute metrics separately to be safe
    
    # Cohesion: Average distance to other unvisited nodes
    sub_avg = sub.copy()
    np.fill_diagonal(sub_avg, 0)
    avg_cohesion = np.sum(sub_avg, axis=1) / (n_remaining - 1)
    
    # Lookahead: min_{j != i} ( d(i, j) + d(j, dest) )
    # We need to handle the case where i == j (should be ignored)
    sub_inf = sub.copy()
    np.fill_diagonal(sub_inf, np.inf)
    
    # Cost to go from i to j then to dest
    # C[i, j] = d(i, j) + d(j, dest)
    # d(i, j) is sub_inf[i, j]
    # d(j, dest) is d_dest[j]
    next_hop_cost = sub_inf + d_dest[np.newaxis, :]
    min_lookahead = np.min(next_hop_cost, axis=1)
    
    # 3. Progress and Weights
    # progress: 0 at start (n_remaining = n_total), 1 at end (n_remaining = 1)
    if n_total > 1:
        progress = (n_total - n_remaining) / (n_total - 1)
    else:
        progress = 1.0

    # Dynamic Weighting:
    # Early: Focus on immediate distance and cohesion (building compact clusters)
    # Late: Focus on return to destination to ensure feasibility
    
    # Return weight increases quadratically to become dominant at the end
    w_return = 0.1 + 0.5 * (progress ** 2)
    
    # Cohesion weight decreases as the set of unvisited nodes shrinks
    # It is less useful when few nodes remain
    w_cohesion = 0.2 * (1.0 - progress)
    
    # Lookahead weight is constant, providing a consistent 2-step view
    w_lookahead = 0.3
    
    # Base immediate distance weight
    w_curr = 1.0

    # 4. Composite Score
    scores = (w_curr * d_curr + 
              w_return * d_dest + 
              w_cohesion * avg_cohesion + 
              w_lookahead * min_lookahead)
    
    # 5. Regret-based Tie Breaking
    # If scores are close, prefer the one that is significantly better than others.
    # This helps in scenarios where multiple nodes are equidistant.
    min_score = np.min(scores)
    # Use a squared penalty to strongly favor the minimum
    # Note: (scores - min_score) is 0 for the min, positive for others.
    # We add this penalty, effectively reducing the relative chance of picking a "second best" node
    # if the gap is small, but it's additive so it doesn't change the min index directly,
    # it just helps if we were doing a soft-max selection. Since we use argmin,
    # this term is constant for the argmin selection unless we want to modify the landscape.
    # Actually, for a hard argmin, adding a constant or a term that is 0 for the min 
    # doesn't change the winner. 
    # However, floating point errors can cause ties. 
    # A better tie-breaker for argmin is to add a tiny epsilon based on index or 
    # prefer lower distance to destination on tie.
    
    # Let's use a subtle tie-breaker: if scores are within 1e-6, prefer node with lower d_dest
    # or lower avg_cohesion.
    
    # Since we return a single argmin, the regret penalty above doesn't help.
    # Let's refine: Add a very small penalty to break ties based on d_dest
    # This ensures that if two nodes are equally good in total score, the one closer 
    # to the destination is picked, which is generally better for future steps.
    
    tie_breaker = 1e-9 * d_dest
    final_scores = scores + tie_breaker
    
    # Safety check for NaN/Inf
    if np.any(np.isnan(final_scores)) or np.any(np.isinf(final_scores)):
        final_scores = d_curr + d_dest
        
    best_idx = np.argmin(final_scores)
    return int(unvisited_nodes[best_idx])