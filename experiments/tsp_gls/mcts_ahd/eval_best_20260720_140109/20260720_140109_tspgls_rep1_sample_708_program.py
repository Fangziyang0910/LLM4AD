
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
    result = edge_distance.copy()
    n = len(local_opt_tour)
    
    if n == 0 or n < 2:
        return result

    # 1. Global Robust Statistics
    flat_usage = edge_n_used.flatten()
    if len(flat_usage) > 0:
        median_usage = np.median(flat_usage)
        q1, q3 = np.percentile(flat_usage, [25, 75])
        iqr = q3 - q1
        if iqr < 1e-9:
            iqr = 1.0
            
        mean_usage = np.mean(flat_usage)
        std_usage = np.std(flat_usage)
        if std_usage < 1e-9:
            std_usage = 1.0
    else:
        median_usage = 0.0
        iqr = 1.0
        mean_usage = 0.0
        std_usage = 1.0

    # 2. Concentration Factor based on Global CV
    # Higher CV indicates more concentrated usage -> Stronger penalty needed to escape local optima
    if mean_usage > 1e-9:
        cv = std_usage / mean_usage
        # Tuned weight: 0.6 provides a balanced response between Algo 2 (0.55) and Algo 3 (0.5)
        concentration_factor = 1.0 + 0.6 * cv
    else:
        concentration_factor = 1.0

    # 3. Extract current tour edges and their usages
    tour_edges = []
    current_tour_usages = []
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_edges.append((u, v))
        current_tour_usages.append(edge_n_used[u, v])
    
    # Sort usages for rank calculation
    sorted_tour_usages = np.sort(current_tour_usages)

    # 4. Algorithm Parameters
    sigmoid_steepness = 3.0
    base_penalty_mult = 0.8   # Base multiplier
    noise_intensity_base = 0.3
    noise_rank_scale = 0.5
    noise_dev_scale = 0.2
    max_penalty_clip = 4.0
    min_penalty_clip = 1.0

    # 5. Apply penalties
    for idx, (u, v) in enumerate(tour_edges):
        usage = edge_n_used[u, v]
        
        # Robust Z-score
        robust_z = (usage - median_usage) / iqr
        
        # Sigmoid mapping to [0, 1]
        normalized_intensity = 1.0 / (1.0 + np.exp(-sigmoid_steepness * robust_z))
        
        # Scaled Penalty Component
        sigmoid_penalty = base_penalty_mult * normalized_intensity * concentration_factor
        
        # Rank Calculation for Adaptive Noise
        # Using searchsorted for efficiency and consistency
        rank = np.searchsorted(sorted_tour_usages, usage, side='right') - 1
        if rank < 0: rank = 0
        if rank >= n: rank = n - 1
        
        norm_rank = rank / (n - 1) if n > 1 else 0.5
        
        # Adaptive Noise: Scaled by rank and deviation magnitude
        deviation_magnitude = abs(robust_z)
        noise_scale = noise_intensity_base + noise_dev_scale * deviation_magnitude + noise_rank_scale * norm_rank
        noise = np.random.normal(0, noise_scale)
        
        # Final Penalty Factor
        penalty_factor = 1.0 + sigmoid_penalty + noise
        
        # Clip for stability
        penalty_factor = np.clip(penalty_factor, min_penalty_clip, max_penalty_clip)
            
        # Apply to matrix
        result[u, v] *= penalty_factor
        result[v, u] *= penalty_factor
        
    return result
