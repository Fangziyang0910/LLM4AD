
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
    updated_distance = edge_distance.copy()
    
    n = len(local_opt_tour)
    if n < 2:
        return updated_distance

    # Identify the edges in the local optimal tour
    edges_in_tour = []
    tour_costs = []
    tour_usage_counts = []
    
    for i in range(n):
        u = int(local_opt_tour[i])
        v = int(local_opt_tour[(i + 1) % n])
        edges_in_tour.append((u, v))
        tour_costs.append(edge_distance[u, v])
        tour_usage_counts.append(edge_n_used[u, v])
    
    if not tour_costs:
        return updated_distance

    tour_costs_arr = np.array(tour_costs)
    usage_counts_arr = np.array(tour_usage_counts)
    
    # Calculate statistics for normalization
    median_cost = np.median(tour_costs_arr)
    
    # Hyperparameters
    base_penalty_scale = 0.20          # Balanced base scale between No.1 (0.35) and No.2 (0.15)
    quadratic_scale = 2.0              # Quadratic growth for deviation (balanced)
    stagnation_base = 1.5              # Threshold for stagnation pressure (lower than No.2's 2.0)
    quadratic_steepness = 0.5          # Steepness of stagnation pressure
    exploration_weight = 0.5           # Weight for exploration factor
    exponential_decay_factor = 0.10    # Exponential decay for high-frequency edges (inspired by No.1)
    eps = 1e-10
    
    # Calculate robust deviation metric using Median Absolute Deviation (MAD)
    mad_cost = np.median(np.abs(tour_costs_arr - median_cost))
    if mad_cost < eps:
        mad_cost = eps
    
    # Calculate global usage statistics for stagnation pressure (Inspired by No.1)
    avg_usage = np.mean(usage_counts_arr)
    excess_usage = max(0, avg_usage - stagnation_base)
    stagnation_pressure = 1 + quadratic_steepness * (excess_usage ** 2)
    
    # Calculate exploration factor (Inspired by No.2)
    # Harmonic mean emphasizes low usage counts (underused edges)
    safe_usage = usage_counts_arr + eps
    harmonic_mean_usage = n / np.sum(1.0 / safe_usage)
    
    # Exploration factor: Higher value when edges are underused (low harmonic mean)
    exploration_factor = 1.0 + (exploration_weight * (1.0 / (1.0 + harmonic_mean_usage)))

    for i, (u, v) in enumerate(edges_in_tour):
        current_edge_cost = tour_costs_arr[i]
        usage_count = usage_counts_arr[i]
        
        # Only penalize edges that are above median cost to focus on the worst half
        if current_edge_cost > median_cost:
            
            # Normalize deviation by MAD
            normalized_deviation = (current_edge_cost - median_cost) / mad_cost
            
            # Quadratic penalty term: Penalizes large deviations more severely
            # Ensures the penalty grows rapidly with cost deviation
            cost_penalty_raw = np.power(max(0, normalized_deviation), quadratic_scale)
            
            # Usage modulation: 
            # 1. Exploration favorability: Penalize more if the edge is rarely used globally (Inspired by No.2)
            # 2. Exponential Decay: Protects frequently used links from excessive distortion (Inspired by No.1)
            usage_modulator = 1.0 / (1.0 + usage_count * 0.1)
            decay_term = np.exp(-usage_count * exponential_decay_factor)
            
            # Combine components
            # Base scale * Quadratic Deviation * Stagnation Pressure * Exploration Factor * Usage Modulator * Exponential Decay
            penalty = (base_penalty_scale * 
                       cost_penalty_raw * 
                       stagnation_pressure * 
                       exploration_factor * 
                       usage_modulator *
                       decay_term)
            
            # Ensure non-negative
            if penalty < 0:
                penalty = 0
                
            # Apply penalty to the symmetric matrix
            updated_distance[u, v] += penalty
            updated_distance[v, u] += penalty
            
    return updated_distance
