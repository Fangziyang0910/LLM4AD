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
    n = len(prize)
    if n <= 1:
        return np.zeros((n, n))

    eps = 1e-9
    
    # --- 1. Distance Decay ---
    dist_safe = np.maximum(distance, eps)
    inv_sq_dist = 1.0 / (dist_safe ** 2)
    
    # --- 2. Dynamic Slack Fraction ---
    dist_from_depot = distance[0, :]  # shape (n,)
    dist_to_depot = distance[:, 0]    # shape (n,)
    
    # Broadcasting setup
    dist_from_depot_2d = dist_from_depot[:, np.newaxis]
    dist_to_depot_2d = dist_to_depot[np.newaxis, :]
    
    # Effective cost matrix: cost to go 0 -> i -> j -> 0
    effective_cost = dist_from_depot_2d + distance + dist_to_depot_2d
    
    # Slack: remaining budget after taking edge i->j and returning
    slack = maxlen - effective_cost
    
    # Normalized slack fraction
    maxlen_safe = maxlen if maxlen > eps else eps
    slack_ratio = slack / maxlen_safe
    
    # Graceful suppression of infeasible edges using np.maximum
    slack_fraction = np.maximum(slack_ratio, 0.0)
    
    # --- 3. Approximate Reachable Potential (O(N^2)) ---
    # Precompute cost(j, k) + dist(k, 0) for all j, k
    dist_to_depot_row = dist_to_depot[np.newaxis, :] # (1, n)
    cost_jk_0 = distance + dist_to_depot_row # (n, n)
    
    # Precompute term(j, k) = prize[k] / cost(j, k, 0)
    # We use cost(j, k, 0) in denominator to normalize by feasibility
    prize_col = prize[:, np.newaxis] # (n, 1)
    cost_jk_0_safe = np.maximum(cost_jk_0, eps)
    term_jk_static = prize_col / cost_jk_0_safe # (n, n)
    
    # Sum over k to get a "static reachability potential" for each node j
    # This represents the value of node j as a gateway to other prizes
    static_potential = np.sum(term_jk_static, axis=1) # shape (n,)
    
    # Now, we need to scale this by the remaining budget slack for edge (i, j).
    slack_safe = np.maximum(slack_fraction, eps)
    dynamic_potential = static_potential[np.newaxis, :] * (slack_safe ** 1.5)

    # --- 4. Dynamic Prize-to-Cost Efficiency ---
    # Replace static raw prize with prize/j-cost ratio to reward efficient moves
    # prize[j] / distance[i, j]
    dest_prize = prize[np.newaxis, :] # shape (1, n)
    inv_dist = 1.0 / dist_safe # shape (n, n)
    
    # Efficiency term: Prize / Distance for the immediate edge
    efficiency_term = dest_prize * inv_dist
    
    # Combine efficiency with dynamic potential
    dcp_weighted = dynamic_potential * efficiency_term
    
    # --- 5. Return Feasibility Penalty ---
    # Exponentially penalize edges (i, j) where distance[j, 0] exceeds maxlen/2.
    # Penalty = exp(-max(0, dist(j,0) - maxlen/2) / maxlen)
    # This steers the ant away from nodes from which returning is risky.
    threshold = maxlen_safe * 0.5
    excess_dist = np.maximum(0.0, dist_to_depot_2d - threshold)
    return_penalty = np.exp(-excess_dist / maxlen_safe)
    
    # --- 6. Critical Slack Penalty (Refined Centralization Bias) ---
    # Piecewise linear reward: strongly penalize moves from nodes where the 
    # remaining slack is less than 10% of maxlen.
    # np.where(slack_fraction < 0.1, 0.1, 1.0)
    critical_slack_threshold = 0.1
    centralization_bias = np.where(slack_fraction < critical_slack_threshold, 0.1, 1.0)
    
    # --- 7. Prize Density ---
    # Encourage moving towards clusters of high-value nodes.
    # Identify high-value nodes (prize > median)
    high_value_threshold = np.median(prize)
    is_high_value = prize > high_value_threshold
    
    # Calculate mean distance from each node to other high-value nodes
    # Mask for high value nodes
    hv_mask = is_high_value[np.newaxis, :] # (1, n)
    
    # Initialize sums and counts
    sum_dist = np.zeros(n)
    count_dist = np.zeros(n)
    
    # If we have high value nodes, compute their contribution
    if np.any(is_high_value):
        hv_indices = np.where(is_high_value)[0]
        # Distance from HV nodes to all nodes
        dist_from_hv = distance[hv_indices, :] # (num_hv, n)
        
        # Create a zeroed copy to handle self-distances
        dist_hv_safe = dist_from_hv.copy()
        
        # Set self-distances to 0 using vectorized advanced indexing
        if len(hv_indices) > 0:
            dist_hv_safe[np.arange(len(hv_indices)), hv_indices] = 0.0
            
        # Sum over HV sources
        sum_dist = np.sum(dist_hv_safe, axis=0) # shape (n,)
        
        # Count: for each node j, how many HV nodes contribute?
        # If j is HV, it contributes (num_hv - 1) because self is 0.
        # If j is not HV, it contributes num_hv.
        num_hv = len(hv_indices)
        count_dist = np.where(is_high_value, num_hv - 1, num_hv)
    else:
        num_hv = 0

    # Handle case where num_hv is 0 or 1 to avoid division by zero or meaningless stats
    if num_hv < 2:
        mean_dist_to_hv = np.ones(n) * eps # No cluster effect, neutral
    else:
        mean_dist_to_hv = np.divide(sum_dist, count_dist, out=np.zeros_like(sum_dist), where=(count_dist>0))
        # Replace zeros (if count was 0) with eps
        mean_dist_to_hv = np.where(mean_dist_to_hv > 0, mean_dist_to_hv, eps)

    # Prize Density: Prize[j] / MeanDistance[j, nearest/avg HV]
    # If a node is in a dense cluster, mean_dist is small -> Density is high.
    prize_density = np.divide(prize, mean_dist_to_hv, out=np.zeros_like(prize), where=(mean_dist_to_hv>0))
    prize_density = np.where(prize_density > 0, prize_density, eps)
    prize_density_2d = prize_density[np.newaxis, :] # shape (1, n)

    # --- 8. Composite Heuristic ---
    # Heuristic = Efficiency[i,j] * InvSqDist[i,j] * ReturnPenalty[j] * WeightedReachablePotential[i,j] * CentralizationBias[i,j] * PrizeDensity[j]
    heuristic_matrix = efficiency_term * inv_sq_dist * return_penalty * dcp_weighted * centralization_bias * prize_density_2d
    
    return heuristic_matrix
