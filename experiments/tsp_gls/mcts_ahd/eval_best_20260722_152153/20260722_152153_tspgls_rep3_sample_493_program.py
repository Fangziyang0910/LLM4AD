
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
    import numpy as np

    result = edge_distance.copy()
    n_cities = len(local_opt_tour)
    
    if n_cities == 0:
        return result

    # 1. Extract edges in the local optimal tour
    u_nodes = np.array([int(local_opt_tour[i]) for i in range(n_cities)])
    v_nodes = np.array([int(local_opt_tour[(i + 1) % n_cities]) for i in range(n_cities)])
    
    # 2. Get current usage counts and distances for tour edges
    tour_usages = edge_n_used[u_nodes, v_nodes]
    tour_distances = edge_distance[u_nodes, v_nodes]
    
    # 3. Calculate global statistics for normalization
    global_mean_dist = np.mean(edge_distance)
    if global_mean_dist == 0:
        global_mean_dist = 1.0
        
    total_edges = n_cities * (n_cities - 1) // 2
    global_usage_sum = np.sum(edge_n_used)
    if global_usage_sum == 0:
        global_usage_sum = 1.0
        
    epsilon = 1e-9
    
    # 4. Usage Term: Logarithmic Inverse Frequency (Inspired by Algo 1)
    # Calculate mean usage per edge
    mean_usage = global_usage_sum / total_edges
    
    # Normalize usage relative to mean usage
    # Avoid division by zero by adding epsilon
    normalized_freq = (tour_usages + epsilon) / (mean_usage + epsilon)
    
    # Logarithmic inverse: penalizes low usage more strongly
    # log(1 + 1/freq) is stable for low and high frequencies
    usage_penalty_raw = np.log(1.0 + 1.0 / (normalized_freq + epsilon))
    
    # Normalize by max usage penalty in the tour to [0, 1]
    max_usage_penalty = np.max(usage_penalty_raw)
    if max_usage_penalty == 0:
        max_usage_penalty = 1.0
    normalized_usage_penalty = usage_penalty_raw / max_usage_penalty
    
    # 5. Cost Term: Linear Deviation from Mean (Inspired by Algo 1)
    # Normalize distance by global mean
    normalized_dist_ratio = tour_distances / global_mean_dist
    
    # Penalty increases linearly with distance above mean
    # Shift to ensure non-negative values for normalization
    cost_penalty_raw = normalized_dist_ratio - 1.0
    
    # Handle negative deviations (edges cheaper than mean)
    min_cost_penalty = np.min(cost_penalty_raw)
    if min_cost_penalty < 0:
        cost_penalty_raw = cost_penalty_raw - min_cost_penalty
        
    max_cost_penalty = np.max(cost_penalty_raw)
    if max_cost_penalty == 0:
        max_cost_penalty = 1.0
    normalized_cost_penalty = cost_penalty_raw / max_cost_penalty
    
    # 6. Combine Penalties Additively (Inspired by Algo 1)
    # Equal weighting for balance between diversity (usage) and quality (cost)
    combined_factor = normalized_usage_penalty + normalized_cost_penalty
    
    # Scale by a factor proportional to the global mean distance
    # Using 1.35 as a tuned scale between Algo 1 (1.25) and Algo 2 (1.5)
    base_penalty_scale = 1.35 * global_mean_dist
    
    penalty = base_penalty_scale * combined_factor
    
    # 7. Apply penalties symmetrically
    # Clip penalty to avoid extreme distortion
    # Clip at 2.5 * global_mean_dist to allow moderate exploration
    penalty = np.clip(penalty, 0, 2.5 * global_mean_dist)
    
    result[u_nodes, v_nodes] += penalty
    result[v_nodes, u_nodes] += penalty
    
    return result
