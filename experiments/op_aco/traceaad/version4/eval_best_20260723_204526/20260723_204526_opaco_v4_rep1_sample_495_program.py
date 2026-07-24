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
    eps = 1e-9
    
    # --- Step 1: Normalized Prize Density ---
    # Normalize prizes by the mean to get relative importance
    mean_prize = np.mean(prize)
    if mean_prize < eps:
        mean_prize = eps
    normalized_prize = prize / mean_prize
    
    # --- Step 2: Prize-to-Distance Ratio (Base Heuristic) ---
    # Reward nodes with high prizes that are close by
    # Use distance squared for sharper decay
    dist_sq = np.maximum(distance ** 2, eps)
    
    # prize_col: shape (1, n) where prize_col[0, j] = normalized_prize[j]
    prize_col = normalized_prize[np.newaxis, :]
    
    # base_heuristic[i, j] = prize[j] / dist(i, j)^2
    base_heuristic = prize_col / dist_sq
    
    # --- Step 2.5: Dynamic Prize Decay Term ---
    # Calculate the global maximum potential prize density in the instance.
    max_base_heuristic = np.max(base_heuristic)
    if max_base_heuristic < eps:
        max_base_heuristic = eps
        
    # Dynamic decay: scale base_heuristic by its ratio to the global max.
    decay_factor = base_heuristic / max_base_heuristic
    
    # --- Step 3: Prize Density Gradient Factor ---
    # Calculate the global mean distance from any node to all other nodes.
    sum_dist = np.sum(distance, axis=1) # Shape (n,)
    mean_dist_from_i = sum_dist / (n - 1) # Shape (n,)
    
    # For an edge i->j, we want to reward if j is close relative to the average spread.
    mean_dist_row = mean_dist_from_i[:, np.newaxis] # Shape (n, 1)
    dist_from_i = distance # Shape (n, n)
    
    # Avoid division by zero
    dist_from_i_safe = np.maximum(dist_from_i, eps)
    
    gradient_factor = mean_dist_row / dist_from_i_safe
    
    # --- Step 4: Budget-Aware Feasibility & Efficiency Penalty ---
    # Estimate the cost of the partial tour 0 -> i -> j -> 0
    # dist_to_i[i]: distance from depot to node i
    dist_to_i = distance[0, :]  # shape (n,)
    dist_to_i_row = dist_to_i[:, np.newaxis]  # shape (n, 1)
    
    # dist_from_j[j]: distance from node j to depot
    dist_from_j = distance[:, 0]  # shape (n,)
    dist_from_j_col = dist_from_j[np.newaxis, :]  # shape (1, n)
    
    # estimated_tour_len[i, j] = dist[0, i] + dist[i, j] + dist[j, 0]
    estimated_tour_len = dist_to_i_row + distance + dist_from_j_col
    
    # 1. Efficiency Penalty
    est_len_clipped = np.maximum(estimated_tour_len, eps)
    efficiency = np.minimum(maxlen / est_len_clipped, 1.0)
    
    # 2. Feasibility Penalty (Adaptive Sigmoid)
    slack = maxlen - estimated_tour_len
    
    # Modified Adaptive Scaling:
    # Scale the sigmoid based on the ratio of residual budget to return distance.
    
    # Avoid division by zero in ratio calculation
    dist_from_j_col_safe = np.maximum(dist_from_j_col, eps)
    
    base_sharpness = 5.0
    
    # Scale inversely proportional to return distance
    scale = base_sharpness * (maxlen / dist_from_j_col_safe)
    
    # Cap scale to prevent overflow
    max_scale = 100.0
    scale = np.minimum(scale, max_scale)
    
    exp_arg = -scale * slack
    exp_arg = np.clip(exp_arg, -50, 50)
    
    feasibility_smooth = 1.0 / (1.0 + np.exp(exp_arg))
    
    combined_penalty = feasibility_smooth * efficiency
    
    # --- Step 4.5: Return Feasibility Check (Hard Constraint) ---
    # Explicitly penalize edges where distance from j to depot > residual budget.
    # Cost to reach j from 0 via i: dist[0, i] + dist[i, j]
    cost_to_reach_j = dist_to_i_row + distance # Shape (n, n)
    
    # Residual budget available to return from j to 0
    residual_budget_for_return = maxlen - cost_to_reach_j
    
    # Condition: dist_from_j_col <= residual_budget_for_return
    # We use a strict boolean mask for hard constraint.
    return_feasible = (dist_from_j_col <= residual_budget_for_return + eps)
    
    # Convert to float for multiplication
    return_feasibility_mask = return_feasible.astype(float)
    
    # --- Step 5: Residual Capacity Multiplier ---
    # New Idea: Scale heuristic by ratio of remaining budget after visiting j 
    # to the distance required to return from j.
    # Residual budget after visiting j and returning to depot:
    # slack = maxlen - (dist[0,i] + dist[i,j] + dist[j,0])
    # However, the "remaining budget after visiting j" usually implies the budget left 
    # to continue the tour, i.e., maxlen - (dist[0,i] + dist[i,j]).
    # But the prompt says "remaining budget after visiting j to the distance required to return from j".
    # Let's interpret "remaining budget after visiting j" as the slack left after the full 
    # hypothetical return trip 0->i->j->0? No, that would be zero if we returned.
    # It likely means the budget left after arriving at j: Residual = maxlen - (dist[0,i] + dist[i,j]).
    # And "distance required to return from j" is dist[j, 0].
    # So ratio = Residual / dist[j, 0].
    # If Residual < dist[j, 0], the ratio is negative/zero, implying infeasibility.
    # This acts as a continuous measure of feasibility margin.
    
    # Cost to reach j: dist[0, i] + dist[i, j]
    # residual_after_j = maxlen - (dist[0, i] + dist[i, j])
    residual_after_j = maxlen - cost_to_reach_j
    
    # Distance to return from j
    dist_return_j = dist_from_j_col # Shape (1, n), broadcasted to (n, n)
    dist_return_j_safe = np.maximum(dist_return_j, eps)
    
    # Capacity multiplier: residual_after_j / dist_return_j
    # Clip at 0 to ensure non-negative scaling for feasible edges.
    # If residual_after_j is negative, the edge is infeasible for return, 
    # and return_feasibility_mask will zero it out.
    capacity_ratio = np.maximum(residual_after_j, 0.0) / dist_return_j_safe
    
    # Normalize capacity ratio to [0, 1] or keep as is? 
    # Let's keep it as a multiplier but cap it to avoid extreme values dominating.
    # A ratio of 1 means we have exactly enough budget to return.
    # A ratio > 1 means we have extra budget.
    # We can cap at e.g., 2.0 to prevent huge multipliers for very short return trips with huge slack.
    max_capacity_ratio = 2.0
    capacity_multiplier = np.minimum(capacity_ratio, max_capacity_ratio)
    
    # --- Step 5.5: Remaining Budget Utilization Heuristic ---
    # Residual budget after visiting i and j and returning from j
    residual_budget = maxlen - estimated_tour_len
    # Only consider edges that leave some budget
    residual_budget_safe = np.maximum(residual_budget, eps)
    
    # Estimate potential collectible prize within residual budget.
    # Approximate by: Residual_Budget * (Average Prize Density of other nodes from j).
    
    # prize_row: shape (1, n) with prize[k]
    prize_row = prize[np.newaxis, :]
    
    # dist_from_j_matrix: shape (n, n) where element [j, k] is dist[j, k]
    dist_from_j_matrix_safe = np.maximum(distance, eps)
    
    # Density matrix: density[j, k] = prize[k] / dist[j, k]
    density_matrix = prize_row / dist_from_j_matrix_safe
    
    # Compute mean density for each node j
    sum_density_from_j = np.sum(density_matrix, axis=1) # Shape (n,)
    mean_density_from_j = sum_density_from_j / (n - 1) # Shape (n,)
    
    # Expand to shape (1, n)
    mean_density_from_j_col = mean_density_from_j[np.newaxis, :] # Shape (1, n)
    
    # Mask out edges where estimated_tour_len > maxlen
    feasible_mask = (estimated_tour_len <= maxlen).astype(float)
    
    # Utilization term
    utilization_term = (residual_budget_safe * mean_density_from_j_col) * feasible_mask
    
    # Normalize utilization term
    max_potential_util = np.max(utilization_term)
    if max_potential_util < eps:
        max_potential_util = eps
    utilization_normalized = utilization_term / max_potential_util
    
    # --- Step 6: Combine All Components ---
    # Integrate the new capacity multiplier.
    heuristic = (base_heuristic * gradient_factor * combined_penalty * decay_factor * 
                 (1.0 + utilization_normalized) * return_feasibility_mask * 
                 capacity_multiplier)
    
    # --- Step 7: Clean up ---
    # Ensure diagonal is negligible
    np.fill_diagonal(heuristic, 0.0)
    
    # Clamp negative or zero values
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
