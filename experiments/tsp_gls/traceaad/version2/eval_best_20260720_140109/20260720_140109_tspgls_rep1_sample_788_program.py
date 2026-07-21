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
    # Create a copy to avoid modifying the original
    updated = edge_distance.copy()
    
    n = edge_distance.shape[0]
    
    # Identify edges in the local optimal tour using vectorized operations
    # The tour is a sequence of city indices
    u_indices = local_opt_tour
    v_indices = np.roll(local_opt_tour, -1)
    
    # Create a mask for edges in the tour
    in_tour_mask = np.zeros_like(edge_distance, dtype=bool)
    in_tour_mask[u_indices, v_indices] = True
    in_tour_mask[v_indices, u_indices] = True
    
    # Find the maximum usage count to normalize
    max_usage = np.max(edge_n_used) if np.max(edge_n_used) > 0 else 1
    
    # Normalize usage counts to [0, 1]
    normalized_usage = edge_n_used / max_usage
    
    # --- Dynamic Volatility-Based Scaling ---
    # Calculate Coefficient of Variation (CV) of edge_n_used
    # CV = std / mean. If mean is 0, CV is undefined, but usage should be > 0 if tour exists.
    mean_usage = np.mean(edge_n_used)
    std_usage = np.std(edge_n_used)
    
    if mean_usage > 0:
        cv = std_usage / mean_usage
    else:
        cv = 0
        
    # Define a volatility scale. 
    # Low CV -> High Rigidity -> High Penalty Scaling
    # High CV -> Low Rigidity -> Low Penalty Scaling
    # We map CV to a scaling factor [0, 1].
    # Assume max possible CV for a binary-like distribution is around 1.0 or slightly higher depending on graph density.
    # Let's clamp CV to a reasonable range, e.g., [0, 2], and invert it.
    # If CV is 0, penalty_scale_factor is 1.0 (Max penalty)
    # If CV is high, penalty_scale_factor is 0.0 (Min penalty)
    
    max_cv_estimate = 2.0 # Heuristic upper bound for CV in TSP context
    cv_clamped = np.clip(cv, 0, max_cv_estimate)
    
    # Scaling factor: inversely proportional to CV
    # penalty_scale_factor = 1 - (cv_clamped / max_cv_estimate)
    # This gives 1.0 when cv=0 and 0.0 when cv=max_cv_estimate.
    penalty_scale_factor = 1.0 - (cv_clamped / max_cv_estimate)
    
    # Ensure a minimum baseline penalty to always discourage used edges slightly
    min_penalty_factor = 0.2
    penalty_scale_factor = np.maximum(penalty_scale_factor, min_penalty_factor)
    
    # --- Rigid Segment Detection (Enhanced with Dynamic Scaling) ---
    # Get normalized usage for tour edges
    step_usages = normalized_usage[u_indices, v_indices] # Shape (n,)
    
    # The usage of the next edge in the sequence
    next_step_usages = np.roll(step_usages, -1)
    
    # Rigid weight: product of current edge usage and next edge usage
    # This identifies "stuck" segments where two consecutive edges are heavily used.
    rigid_weight = step_usages * next_step_usages
    
    # Threshold for rigid detection remains static or can be adjusted, 
    # but the INTENSITY of the penalty is now driven by penalty_scale_factor.
    rigid_threshold = 0.1 
    
    is_rigid = rigid_weight > rigid_threshold
    
    # --- Apply Penalties and Rewards ---
    
    # Base scales
    penalty_scale_base = 0.1
    penalty_scale_rigid = 0.2 # Amplified base for rigid, further scaled by volatility
    
    # Initialize penalty matrix
    penalty_matrix = np.zeros_like(edge_distance)
    
    # Apply penalties to tour edges
    for i in range(n):
        u, v = u_indices[i], v_indices[i]
        usage = step_usages[i]
        dist = edge_distance[u, v]
        
        if is_rigid[i]:
            # Amplified penalty for rigid segments
            # Scaled by global volatility factor
            # High penalty_scale_factor (low variance) -> Stronger push to break rigidity
            penalty = penalty_scale_rigid * rigid_weight[i] * dist * (1.0 + penalty_scale_factor)
        else:
            # Standard penalty
            # Scaled by global volatility factor
            penalty = penalty_scale_base * usage * dist * (1.0 + penalty_scale_factor)
            
        penalty_matrix[u, v] += penalty
        penalty_matrix[v, u] += penalty
        
    updated += penalty_matrix
    
    # --- Apply Rewards for Non-Tour Edges ---
    
    # Reward is based on inverse usage: low usage -> high reward
    # Cap the reduction to prevent negative or artificially small distances.
    cap_fraction = 0.5
    
    raw_reward_weight = 1.0 / (1.0 + normalized_usage)
    scaled_reward = (raw_reward_weight - 0.5) * 2.0 # Maps [0.5, 1.0] -> [0, 1]
    
    max_reduction = cap_fraction * edge_distance
    reduction = scaled_reward * max_reduction
    reduction = np.minimum(reduction, max_reduction)
    
    non_tour_mask = ~in_tour_mask
    updated -= reduction * non_tour_mask
    
    # --- Add Noise and Enforce Constraints ---
    
    # Add small random perturbation to all edges to encourage exploration
    # Noise scale can also be influenced by volatility: high rigidity -> more noise?
    # Let's keep noise static for stability, or slightly increase with penalty_scale_factor
    perturbation_scale = 0.01 * np.max(edge_distance) * (1.0 + 0.5 * penalty_scale_factor)
    random_noise = np.random.uniform(-perturbation_scale, perturbation_scale, size=updated.shape)
    
    # Add noise and ensure symmetry
    updated += random_noise
    updated = (updated + updated.T) / 2
    
    # Ensure no negative distances
    updated = np.maximum(updated, 0)
    
    return updated
