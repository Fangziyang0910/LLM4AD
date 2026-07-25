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
    eps = 1e-8
    
    # 1. Compute Prize-to-Distance Ratio (Base Attraction)
    # prize[np.newaxis, :] broadcasts prize[j] to all rows i
    safe_distance = np.where(distance == 0, eps, distance)
    prize_matrix = prize[np.newaxis, :]
    prize_ratio = prize_matrix / safe_distance  # shape (n, n)
    
    # 2. Compute Budget Constraints and Costs
    # Feasibility is based on remaining budget if we take edge (i,j) and return directly to depot.
    # Estimated cost: dist(0, i) + dist(i, j) + dist(j, 0)
    
    # dist(0, i): distance from depot to node i. Column 0 of distance matrix.
    dist_0_i = distance[:, 0]  # shape (n,)
    
    # dist(j, 0): distance from node j to depot. Row 0 of distance matrix.
    dist_j_0 = distance[0, :]  # shape (n,)
    
    # Broadcast for matrix operations:
    # dist_0_i[:, None] -> shape (n, 1)
    # distance          -> shape (n, n)
    # dist_j_0[None, :] -> shape (1, n)
    
    estimated_round_trip_cost = dist_0_i[:, None] + distance + dist_j_0[None, :]
    
    remaining_budget = maxlen - estimated_round_trip_cost
    
    # 3. Compute Adaptive Alpha (Reference Strategy - Unbounded)
    # Revert to simpler unbounded formulation: alpha = 1 + max(0, remaining/maxlen)
    # This provides a smoother gradient for exponential decay.
    if maxlen <= 0:
        alpha = 1.0 + np.zeros_like(remaining_budget)
    else:
        alpha = 1.0 + np.maximum(0.0, remaining_budget / maxlen)
    
    # 4. Compute Exponential Feasibility Factor
    # Exponent: remaining_budget / (edge_cost * alpha)
    denominator = safe_distance * alpha
    # Avoid division by zero in denominator if dist is 0 and alpha is 0
    denominator = np.where(denominator == 0, eps, denominator)
    
    exponent = remaining_budget / denominator
    
    # Clip exponent to avoid overflow/underflow in exp
    exponent = np.clip(exponent, -50, 50)
    
    feasibility_factor = np.exp(exponent)
    
    # 5. Compute Local Cluster Density Ratio
    # We look at nodes k such that dist(j, k) <= 2 * dist(i, j).
    # density(i, j) = (sum of prizes of such k) / (2 * dist(i, j))
    
    # dist(j, k) is distance[j, k]. 
    # To get dist(j, k) for all i, j, k:
    # distance[None, :, :] broadcasts to (n, n, n) where [i, j, k] = distance[j, k].
    dist_j_k = distance[None, :, :]  # shape (n, n, n), [i, j, k] = dist(j, k)
    
    # dist(i, j) depends on i and j.
    # distance[:, :, None] gives [i, j, k] = dist(i, j).
    dist_i_j = distance[:, :, None]  # shape (n, n, n)
    
    # Threshold: 2 * dist(i, j)
    threshold = 2.0 * dist_i_j
    
    # Mask: dist(j, k) <= 2 * dist(i, j)
    mask = dist_j_k <= threshold  # shape (n, n, n)
    
    # Prizes: prize[k] needs to be broadcast to (n, n, n).
    prize_k = prize[None, None, :]  # shape (1, 1, n) -> broadcasts to (n, n, n)
    
    # Local Sum of Prizes within radius for each i, j
    # Sum over k axis (axis 2)
    local_prize_sum = np.sum(mask * prize_k, axis=2)  # shape (n, n)
    
    # Normalize by distance to get density per unit cost.
    # Avoid division by zero for dist(i,j)
    # dist_i_j[:, :, 0] extracts the distance matrix from the 3D array
    dist_ij_matrix = dist_i_j[:, :, 0]
    local_density = np.divide(local_prize_sum, (2.0 * dist_ij_matrix), out=np.zeros_like(local_prize_sum), where=dist_ij_matrix > eps)
    
    # Clip density to avoid extreme values and ensure stability
    local_density = np.clip(local_density, 0, 1e6)
    
    # Convert density to a multiplicative boost factor
    # Normalize by global mean prize to keep scale consistent
    mean_prize = np.mean(prize)
    if mean_prize == 0:
        mean_prize = 1.0
        
    density_boost = 1.0 + local_density / mean_prize
    
    # 6. Compute Simplified Prize Gradient Alignment
    # Direct prize difference: delta_prize = prize[j] - prize[i]
    # alignment_score = delta_prize / safe_distance
    
    # Broadcast prizes
    prize_i = prize[:, None]  # shape (n, 1)
    prize_j = prize[None, :]  # shape (1, n)
    
    delta_prize = prize_j - prize_i
    
    # Score normalized by distance
    alignment_score = np.divide(delta_prize, safe_distance, out=np.zeros_like(delta_prize), where=safe_distance > eps)
    
    # Scale by 1.0/mean_prize
    alignment_boost = 1.0 + alignment_score / mean_prize
    
    # Clamp alignment boost to reasonable range to prevent dominance
    alignment_boost = np.clip(alignment_boost, 0.5, 2.0)
    
    # 7. Combine Heuristics
    # Multiply prize ratio by feasibility factor by density bonus by alignment bonus
    heuristic = prize_ratio * feasibility_factor * density_boost * alignment_boost
    
    # 8. Post-processing
    # Ensure finite values
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    # Ensure all values are positive (values <= 0 treated as 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Set diagonal to small value (self-loops not allowed/used)
    np.fill_diagonal(heuristic, 1e-9)
    
    return heuristic
