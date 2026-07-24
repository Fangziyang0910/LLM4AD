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
    n = distance.shape[0]
    
    # 1. Compute Dynamic Alpha and Immediate Prize Density
    dist_to_depot = distance[:, 0]  # Shape (n,)
    dist_j_to_0 = dist_to_depot[np.newaxis, :]  # Shape (1, n)
    
    total_trip_cost = distance + dist_j_to_0  # Shape (n, n)
    residual_budget = maxlen - total_trip_cost
    budget_ratio = residual_budget / maxlen
    budget_ratio = np.clip(budget_ratio, 1e-9, 1.0) 
    
    alpha_min = 1.0
    alpha_max = 3.0
    k_tanh = 2.0  
    val = np.tanh(k_tanh * (1.0 - budget_ratio))
    max_val = np.tanh(k_tanh)
    normalized_intensity = val / max_val
    alpha_map = alpha_min + (alpha_max - alpha_min) * normalized_intensity
    
    dist_safe = np.maximum(distance, 1e-9)
    dist_alpha = np.power(dist_safe, alpha_map)
    
    prize_col = prize[np.newaxis, :]
    density = prize_col / dist_alpha  
    
    # 2. Compute Dynamic Residual Budget and Feasibility (Sigmoid Penalty + Return Trip Urgency)
    cost_fraction = total_trip_cost / maxlen
    
    sigmoid_k = 5.0  
    sigmoid_threshold = 0.7 
    
    sigmoid_arg = sigmoid_k * (cost_fraction - sigmoid_threshold)
    feasibility_factor = 1.0 / (1.0 + np.exp(sigmoid_arg))
    
    dist_j_to_0_safe = np.maximum(dist_j_to_0, 1e-9)
    slack_ratio = residual_budget / dist_j_to_0_safe 
    
    k_urgency = 5.0
    severity = np.maximum(0.0, 1.5 - slack_ratio)
    return_urgency_factor = np.exp(-k_urgency * severity)
    
    feasibility_combined = feasibility_factor * return_urgency_factor

    # 3. Return Penalty Term
    safe_residual = np.maximum(residual_budget, 1e-9)
    return_ratio = dist_j_to_0 / safe_residual
    return_penalty = np.exp(-return_ratio)
    
    # 4. Compute Clustering Density Term with Dynamic Radius
    prize_dist_matrix = prize[np.newaxis, :] / dist_safe 
    sort_indices = np.argsort(distance, axis=1)
    row_indices = np.arange(n)[:, np.newaxis]
    sorted_dist = distance[row_indices, sort_indices]
    sorted_prize_dist = prize_dist_matrix[row_indices, sort_indices]
    prefix_sum = np.cumsum(sorted_prize_dist, axis=1)
    prefix_sum = np.concatenate((np.zeros((n, 1)), prefix_sum), axis=1)
    
    threshold = residual_budget / 2.0
    sorted_dist_expanded = sorted_dist[:, np.newaxis, :] 
    threshold_expanded = threshold[np.newaxis, :, :] 
    valid_neighbors = sorted_dist_expanded <= threshold_expanded 
    counts = np.sum(valid_neighbors, axis=2) 
    counts_T = counts.T 
    row_indices_matrix = np.tile(np.arange(n), (n, 1)) 
    col_indices_matrix = np.clip(counts_T, 0, n)
    clustering_scores = prefix_sum[row_indices_matrix, col_indices_matrix]
    
    max_clustering = np.max(clustering_scores)
    if max_clustering > 0:
        clustering_normalized = clustering_scores / max_clustering
    else:
        clustering_normalized = np.zeros_like(clustering_scores)
        
    # 5. Compute Dynamic Clustering Weight based on Local Node Density
    mask_diag = ~np.eye(n, dtype=bool)
    avg_dist = np.mean(distance[mask_diag]) if np.sum(mask_diag) > 0 else 1.0
    density_radius = avg_dist * 1.5 
    density_counts = np.sum(distance <= density_radius, axis=0) 
    max_density_count = np.max(density_counts) if np.max(density_counts) > 0 else 1.0
    local_density_norm = density_counts / max_density_count
    dynamic_clustering_weight = local_density_norm[np.newaxis, :]
    dynamic_clustering_weight = np.clip(dynamic_clustering_weight, 0.1, 1.0)
    
    decay_factor = np.exp(-dist_safe / avg_dist)
    
    # 6. Global Prize Scarcity Modulator
    total_prize_potential = np.sum(prize) 
    budget_for_clustering = np.maximum(residual_budget, 1e-9)
    prize_budget_ratio = total_prize_potential / budget_for_clustering
    
    avg_prize = np.mean(prize[1:]) if n > 1 else 1.0 
    avg_dist_between_nodes = avg_dist 
    avg_efficiency = avg_prize / (avg_dist_between_nodes + 1e-9)
    
    k_scarcity = 2.0
    scarcity_threshold = avg_efficiency 
    scarcity_arg = k_scarcity * (prize_budget_ratio - scarcity_threshold)
    global_scarcity_modulator = 1.0 / (1.0 + np.exp(-scarcity_arg))
    global_scarcity_modulator = 0.1 + 0.9 * global_scarcity_modulator
    
    clustering_term = dynamic_clustering_weight * decay_factor * clustering_normalized * global_scarcity_modulator
    
    # 7. Opportunity Cost Metric
    feasible_mask = residual_budget > 1e-9
    density_feasible = np.where(feasible_mask, density, -np.inf)
    max_density_row = np.max(density_feasible, axis=1) 
    max_density_safe = np.maximum(max_density_row, 0.0)
    max_density_expanded = max_density_safe[:, np.newaxis] 
    relative_efficiency = density / (max_density_expanded + 1e-9)
    relative_efficiency = np.clip(relative_efficiency, 0.0, 2.0)
    k_eff = 5.0
    efficiency_arg = k_eff * (relative_efficiency - 0.5)
    opportunity_cost_factor = 1.0 / (1.0 + np.exp(-efficiency_arg))

    # 8. Prize-to-Remaining-Budget Ratio Heuristic
    budget_efficiency = prize_col / (total_trip_cost + 1e-9)
    budget_efficiency_normalized = budget_efficiency * maxlen
    k_budget = 2.0
    budget_efficiency_factor = 1.0 / (1.0 + np.exp(-k_budget * (budget_efficiency_normalized - 0.5)))

    # 9. Path Length Conservation Penalty (Flexibility)
    dist_from_j = distance[:, np.newaxis, :] 
    residual_expanded = residual_budget[:, :, np.newaxis]
    
    feasible_from_j = dist_from_j < residual_expanded
    
    dist_successors = np.where(feasible_from_j, dist_from_j, 0.0)
    count_successors = np.sum(feasible_from_j, axis=2)
    count_successors_safe = np.maximum(count_successors, 1.0)
    
    mean_dist_successors = np.sum(dist_successors, axis=2) / count_successors_safe
    
    dist_sq_successors = np.where(feasible_from_j, dist_from_j**2, 0.0)
    mean_sq_dist = np.sum(dist_sq_successors, axis=2) / count_successors_safe
    var_dist_successors = np.maximum(mean_sq_dist - mean_dist_successors**2, 0.0)
    
    mean_norm = mean_dist_successors / (avg_dist + 1e-9)
    var_norm = var_dist_successors / (avg_dist**2 + 1e-9)
    
    k_flex_mean = 2.0
    k_flex_var = 1.5
    
    mean_arg = k_flex_mean * (mean_norm - 0.5)
    mean_factor = 1.0 / (1.0 + np.exp(mean_arg))
    
    var_arg = k_flex_var * (var_norm - 1.0)
    var_factor = 1.0 / (1.0 + np.exp(var_arg))
    
    flexibility_factor = mean_factor * var_factor

    # 10. Diversity Preservation Mechanism
    # Penalize edges leading to nodes that are "over-attractive" based on static metrics
    # to simulate reducing visit frequency of popular nodes.
    
    # Node Attraction Score: Prize / Distance_to_Depot (a proxy for how often a node is visited in standard greedy approaches)
    node_attraction = prize / (dist_to_depot + 1e-9)
    
    # Normalize attraction scores to [0, 1] based on the distribution
    min_att = np.min(node_attraction)
    max_att = np.max(node_attraction)
    if max_att > min_att:
        norm_att = (node_attraction - min_att) / (max_att - min_att)
    else:
        norm_att = np.zeros_like(node_attraction)
        
    # Identify "Popular" nodes (Top 20% by attraction)
    popularity_threshold = 0.8
    is_popular = norm_att > popularity_threshold
    
    # Create a penalty matrix: if j is popular, penalize the edge i->j
    # We don't penalize if this is the single best move (greedy choice) to avoid completely breaking exploitation.
    # But since this is a heuristic matrix, we just scale down the weight.
    
    # Scale penalty strength based on how "crowded" the graph is perceived to be.
    # If many nodes are popular, we penalize more strongly.
    popularity_density = np.mean(is_popular)
    penalty_strength = 0.5 * popularity_density # Stronger penalty if many nodes are attractive
    
    # Penalty factor: 1.0 for non-popular, (1 - penalty_strength) for popular
    popular_mask = is_popular[np.newaxis, :] # Shape (1, n)
    popularity_penalty = 1.0 - penalty_strength * popular_mask
    
    # 11. Combine Heuristics
    heuristic_matrix = density * feasibility_combined * return_penalty * (1.0 + clustering_term) * opportunity_cost_factor * budget_efficiency_factor * flexibility_factor * popularity_penalty
    
    # Ensure no NaNs or Infs
    heuristic_matrix = np.nan_to_num(heuristic_matrix, nan=0.0, posinf=1e-9, neginf=0.0)
    
    # Replace values <= 0 with 1e-9
    mask = heuristic_matrix <= 0
    heuristic_matrix[mask] = 1e-9
    
    return heuristic_matrix
