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
    
    # Create a matrix of prizes for each destination node
    # prize_dest[i, j] = prize[j]
    prize_dest = np.tile(prize, (n, 1))
    
    # Avoid division by zero in distance
    dist_safe = np.maximum(distance, 1e-10)
    
    # Dynamic "prize density" exponent based on local prize-to-distance ratios
    # alpha = log1p(prize) / log1p(dist)
    # This adapts the steepness of the distance penalty: 
    # High prize relative to distance -> lower alpha (less penalty for distance)
    # Low prize relative to distance -> higher alpha (steeper penalty for distance)
    log_prize = np.log1p(prize_dest)
    log_dist = np.log1p(dist_safe)
    
    # Compute dynamic alpha
    alpha_dynamic = log_prize / log_dist
    
    # Determine data-dependent clipping bounds based on global instance density
    # Calculate the mean and std of valid off-diagonal alphas to set bounds.
    
    # Create a mask for valid off-diagonal entries to compute statistics
    valid_mask = np.ones_like(alpha_dynamic, dtype=bool)
    np.fill_diagonal(valid_mask, False)
    
    # Get the distribution of alphas from the valid edges
    valid_alphas = alpha_dynamic[valid_mask]
    
    if len(valid_alphas) > 0:
        mean_alpha = np.mean(valid_alphas)
        std_alpha = np.std(valid_alphas)
        
        # Define bounds dynamically: 
        # Lower bound: max(0.1, mean - 1.5*std)
        # Upper bound: min(4.0, mean + 1.5*std)
        # This allows the heuristic to be more sensitive in varied instances
        # and tighter in consistent ones.
        
        alpha_lower = max(0.1, mean_alpha - 1.5 * std_alpha)
        alpha_upper = min(4.0, mean_alpha + 1.5 * std_alpha)
        
        # Ensure a minimum range to avoid collapse
        if alpha_upper - alpha_lower < 0.5:
            mid = (alpha_lower + alpha_upper) / 2.0
            alpha_lower = mid - 0.25
            alpha_upper = mid + 0.25
            
    else:
        # Fallback to default bounds if no valid edges (shouldn't happen for n>1)
        alpha_lower = 0.5
        alpha_upper = 2.0
        
    # Clamp alpha to [alpha_lower, alpha_upper]
    alpha_dynamic = np.clip(alpha_dynamic, alpha_lower, alpha_upper)
    
    # Compute a base attractiveness: prize / distance^alpha_dynamic
    base_attract = prize_dest / (dist_safe ** alpha_dynamic)
    
    # Feasibility-aware lookahead term
    # Estimate remaining budget after visiting j from i and returning to depot (0)
    # remaining = maxlen - distance[i, j] - distance[j, 0]
    # distance[j, 0] is the column 0 of the distance matrix (return to depot from j)
    # We need to broadcast distance[j, 0] across rows.
    # dist_return[j] = distance[j, 0]
    dist_return = distance[:, 0]
    
    # Calculate remaining budget for each edge (i, j)
    # shape (n, n)
    # distance[i, j] is the cost to move from i to j
    # dist_return[j] is the cost to move from j to depot
    # remaining_budget[i, j] = maxlen - distance[i, j] - dist_return[j]
    remaining_budget = maxlen - distance - dist_return[np.newaxis, :]
    
    # "Residual budget efficiency" metric
    # Estimates the number of future nodes potentially visitable with the remaining budget.
    # We use the average distance across the instance as a proxy for the cost of visiting a future node.
    
    # Calculate average distance for normalization (excluding diagonal and large sentinels)
    # We reuse the valid_mask from alpha calculation for consistency
    valid_distances = dist_safe[valid_mask]
    avg_distance = np.mean(valid_distances) if len(valid_distances) > 0 else 1.0
    avg_safe = max(avg_distance, 1e-10)
    
    # Compute lookahead score: remaining_budget / avg_distance
    # This represents the "capacity" for future moves.
    # If remaining_budget is negative, the score should be 0.
    
    eps = 1e-10
    feasible_mask = remaining_budget > 0
    
    # Calculate efficiency score
    # We scale by 1/avg_safe to get a dimensionless count of potential steps
    raw_lookahead = np.zeros_like(base_attract)
    # Only compute where feasible
    raw_lookahead[feasible_mask] = remaining_budget[feasible_mask] / avg_safe
    
    # Combine base attractiveness with lookahead score
    # The lookahead score acts as a multiplier that dampens edges that consume too much budget
    # relative to the average node cost, potentially cutting off future opportunities.
    heur_matrix = base_attract * raw_lookahead
    
    # Set diagonal to 0 (no self-loops)
    np.fill_diagonal(heur_matrix, 0.0)
    
    # Ensure finite values
    heur_matrix = np.isfinite(heur_matrix) * heur_matrix
    
    # Replace negative or zero values with a small positive number as per contract
    heur_matrix = np.where(heur_matrix <= 0, 1e-9, heur_matrix)
    
    return heur_matrix
