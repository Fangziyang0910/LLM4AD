import numpy as np
def heuristics(prize: np.ndarray, distance: np.ndarray, maxlen: float) -> np.ndarray:
    """Return edge desirability values for OP ant colony optimization.

    Args:
        prize: Node prizes with shape (n,). Node 0 is the depot.
        distance: Pairwise Euclidean distances with shape (n, n).
            Diagonal entries are large sentinels so self-loops are unused.
        maxlen: Maximum allowed tour length (return-to-depot constrained).

    Returns:
        An (n, n) edge-prior matrix. Larger values make an edge more likely
        to be sampled. Values at or below zero are treated as 1e-9.
    """
    n = prize.shape[0]
    
    # Small epsilon to avoid division by zero
    eps = 1e-7
    
    # Distance matrix with epsilon added for stability in division
    d = distance + eps
    
    # Create a matrix of prizes for each destination node
    # prize_dest[i, j] = prize[j]
    prize_dest = prize[np.newaxis, :]  # Shape (1, n) -> broadcast to (n, n)
    
    # Compute mean prize of non-depot nodes
    if n > 1:
        mean_prize = np.mean(prize[1:])
    else:
        mean_prize = 1.0
        
    # Compute median distance of edges that are individually feasible (shorter than maxlen)
    # Filter out diagonal sentinels by ensuring d < maxlen (sentinels are usually very large)
    feasible_mask_dist = d < maxlen
    if np.any(feasible_mask_dist):
        median_dist = np.median(d[feasible_mask_dist])
        # Compute mean distance for feasible edges for instance-aware alpha
        mean_dist_feasible = np.mean(d[feasible_mask_dist])
    else:
        median_dist = 1.0
        mean_dist_feasible = 1.0
    
    # Avoid division by zero in scaling
    mean_prize = max(mean_prize, eps)
    median_dist = max(median_dist, eps)
    mean_dist_feasible = max(mean_dist_feasible, eps)
    
    # Compute instance-aware alpha:
    # alpha = 1.0 + 0.5 * (maxlen / mean_dist_feasible)
    # This adapts the sensitivity to prize-to-distance ratio based on budget tightness.
    alpha = 1.0 + 0.5 * (maxlen / mean_dist_feasible)
    
    # Scaling factor with distance-dependent exponent:
    # Base: (prize[j] / mean_prize) * (median_dist / distance[i,j])
    # Exponent: 1 / (1 + log1p(distance[i,j] / median_dist))
    # This reduces the influence of global scaling for long-distance edges.
    base_term = (prize_dest / mean_prize) * (median_dist / d)
    exponent = 1.0 / (1.0 + np.log1p(d / median_dist))
    
    scaling_factor = base_term ** exponent
    
    # Compute heuristic: (prize[j] / distance[i,j]) ** alpha * scaling_factor
    # The base term rewards high prize-to-distance ratios with adaptive sensitivity.
    heuristic_matrix = (prize_dest / d) ** alpha * scaling_factor
    
    # --- Dynamic Prize-Density Gradient Lookahead Heuristic ---
    # Instead of total reachable prize, we estimate the average prize density 
    # of the k-nearest nodes reachable within the remaining slack after visiting j.
    # Density = Total Prize of Reachable Nodes / Number of Reachable Nodes (or distance extent)
    # We use the ratio of reachable prize to the average distance to these nodes to penalize sparse clusters.
    
    # Compute return cost explicitly from destination node j to depot (node 0)
    return_dist = distance[0, :]  # Shape (n,)
    
    # Slack is the budget remaining after traveling i->j and returning j->0
    # slack[i, j] = maxlen - distance[i, j] - distance[j, 0]
    total_cost = distance + return_dist[np.newaxis, :] # (n, n)
    slack = maxlen - total_cost
    
    # Precompute sorted indices for all destination nodes
    # sorted_indices[j, k] is the index of the k-th closest node to j
    sorted_indices = np.argsort(distance, axis=1) # Shape (n, n)
    
    # Gather sorted distances for each destination node j
    # sorted_dists[j, k] = distance[j, sorted_indices[j, k]]
    sorted_dists = np.take_along_axis(distance, sorted_indices, axis=1)
    
    # Gather sorted prizes for each destination node j
    # sorted_prizes[j, k] = prize[sorted_indices[j, k]]
    sorted_prizes = np.take_along_axis(prize[np.newaxis, :], sorted_indices, axis=1)
    
    # Cumulative sum of prizes along axis 1 (for each j, cumsum over k)
    cum_prizes = np.cumsum(sorted_prizes, axis=1) # Shape (n, n)
    
    # Initialize lookahead matrix
    # We will store a "density score" here. 
    # Density = Reachable_Prize / (Count_of_Reachable + epsilon)
    # This prioritizes clusters where many high-prize nodes are close by.
    lookahead = np.zeros((n, n))
    
    # Vectorized lookup for each destination node j
    for j in range(n):
        # Slack values for all i leading to j
        slacks_j = slack[:, j] # Shape (n,)
        
        # Sorted distances from j to all other nodes
        dists_j = sorted_dists[j, :] # Shape (n,)
        
        # Find insertion indices for slacks into sorted distances
        # side='right' ensures we count all nodes with dist <= slack
        idx = np.searchsorted(dists_j, slacks_j, side='right')
        
        # Clamp indices to valid range [0, n-1] for indexing into cum_prizes
        idx_clamped = np.clip(idx, 0, n - 1)
        
        # Get cumulative prize sums for node j
        cum_p_j = cum_prizes[j, :]
        
        # Map indices to prize sums
        prize_sums = cum_p_j[idx_clamped]
        
        # Count of reachable nodes
        count_reachable = idx_clamped
        
        # If idx == 0, no nodes are reachable within slack, so prize sum is 0
        # Also if slack < 0, searchsorted returns 0
        valid = idx > 0
        prize_sums[~valid] = 0
        count_reachable[~valid] = 0
        
        # Calculate Density: Prize / (Count + 1) to avoid division by zero
        # Adding 1 in denominator ensures that if count is 0, density is 0.
        # If count > 0, this gives average prize per reachable node.
        density_scores = prize_sums / (count_reachable + 1.0)
        
        lookahead[:, j] = density_scores

    # Normalize lookahead scores to be comparable with heuristic_matrix
    # Scale by alpha * mean_prize to couple lookahead bonus with budget sensitivity.
    normalization_factor = alpha * mean_prize
    normalized_lookahead = lookahead / max(normalization_factor, eps)
    
    # Combine: heuristic_matrix * (1 + normalized_lookahead)
    # The lookahead acts as a bonus for future potential value density.
    heuristic_matrix = heuristic_matrix * (1.0 + normalized_lookahead)
    
    # --- Smooth Feasibility Constraints ---
    # Replace hard binary mask with smooth continuous feasibility constraints
    # derived from the reference program's validated structure.
    
    # Compute return cost vector for broadcasting
    return_dist_vec = distance[0, :].reshape(1, -1)  # Shape (1, n)
    
    # Reachability weight: sqrt of normalized remaining budget
    # Normalized by mean_dist_feasible to align budget slack scaling with average 
    # feasible edge length, providing a robust geometric estimate of future visitable nodes.
    remaining_budget = maxlen - d - return_dist_vec
    reachability_weight = np.sqrt(np.maximum(remaining_budget, 0.0) / mean_dist_feasible)
    
    # Dynamic budget-sensitivity term:
    # Penalize edges that consume significant budget relative to maxlen and problem density.
    # Uses alpha to scale the exponential decay penalty.
    dynamic_penalty = np.exp(-total_cost / (maxlen * 0.5 * alpha))
    
    # Multiply by feasibility constraints
    heuristic_matrix = heuristic_matrix * reachability_weight * dynamic_penalty
    
    # Set diagonal to 0 (no self-loops)
    np.fill_diagonal(heuristic_matrix, 0)
    
    # Ensure all values are positive (or at least non-negative)
    # Values at or below zero are treated as 1e-9 by the ACO logic, 
    # but we enforce a floor here for safety.
    heuristic_matrix = np.maximum(heuristic_matrix, 1e-9)
    
    return heuristic_matrix
