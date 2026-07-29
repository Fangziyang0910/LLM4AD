import numpy as np
def heuristics(distance_matrix: np.ndarray, coordinates: np.ndarray, demands: np.ndarray, capacity: int) -> np.ndarray:
    """Return edge desirability values for CVRP ant colony optimization.

    Args:
        distance_matrix: Pairwise Euclidean distances with shape (n, n).
        coordinates: Node coordinates with shape (n, 2). Node 0 is the depot.
        demands: Node demands with shape (n,). The depot demand is zero.
        capacity: Capacity shared by all vehicles.

    Returns:
        An (n, n) edge-prior matrix. Larger values make an edge more likely
        to be sampled. Values at or below zero are treated as 1e-9.
    """
    n = distance_matrix.shape[0]
    depot = coordinates[0]
    epsilon = 1e-10
    
    # 1. Angular Coherence Term (Global Exponential)
    # Vector from depot to each node
    vec_depot_i = coordinates - depot  # shape (n, 2)
    
    # Compute angles from depot for each node
    angles = np.arctan2(vec_depot_i[:, 1], vec_depot_i[:, 0])  # (n,)
    
    # Create a grid of angles for all pairs
    angles_i = angles[np.newaxis, :]  # (1, n)
    angles_j = angles[:, np.newaxis]  # (n, 1)
    
    # Compute angular difference
    angle_diff = np.abs(angles_i - angles_j)  # (n, n)
    # Normalize to [0, pi] range for cosine
    angle_diff_normalized = np.minimum(angle_diff, 2 * np.pi - angle_diff)
    
    # Global Exponential Angular Coherence:
    alpha_angle = 5.0
    angular_coherence = np.exp(-alpha_angle * angle_diff_normalized / np.pi)
    
    # 2. Inverse Distance Heuristic
    inv_dist = 1.0 / (distance_matrix + epsilon)  # shape (n, n)
    
    # 3. Capacity-aware penalty for distance
    demands_j = demands[np.newaxis, :]  # shape (1, n)
    capacity_penalty_factor = (demands_j / (capacity + epsilon)) * distance_matrix
    dynamic_inv_dist = inv_dist / (1.0 + capacity_penalty_factor)
    
    # 4. Distance-Decay Neighbor Score
    flat_dist = distance_matrix.ravel()
    non_zero_dist = flat_dist[flat_dist > 0]
    if len(non_zero_dist) > 0:
        lambda_dist = np.median(non_zero_dist)
    else:
        lambda_dist = 1.0
        
    if lambda_dist < 1e-6:
        lambda_dist = 1e-6
        
    knn_weight = np.exp(-distance_matrix / lambda_dist)
    
    # 5. Demand Urgency
    demand_urgency = demands / (capacity + epsilon)  # shape (n,)
    demand_urgency_mat = demand_urgency[np.newaxis, :]  # shape (1, n)
    
    # 6. Target-Centric Radial Efficiency Term (Refined)
    coords_i = coordinates[np.newaxis, :, :]  # (1, n, 2)
    coords_j = coordinates[:, np.newaxis, :]  # (n, 1, 2)
    move_vec = coords_j - coords_i  # (n, n, 2)
    
    # Vector from node j to depot
    vec_j_depot = depot - coords_j  # (n, 1, 2)
    
    # Norms for normalization
    norm_move = np.linalg.norm(move_vec, axis=2)  # (n, n)
    norm_j_depot = np.linalg.norm(vec_j_depot, axis=2)  # (n, 1)
    
    norm_move_safe = norm_move + epsilon
    norm_j_depot_safe = norm_j_depot + epsilon
    
    # Dot product of move_vec and vec_j_depot
    dot_product = np.sum(move_vec * vec_j_depot, axis=2)  # (n, n)
    
    # Cosine similarity
    cosine_sim = dot_product / (norm_move_safe * norm_j_depot_safe)
    cosine_sim = np.minimum(cosine_sim, 1.0)
    
    alpha_radial = 1.0
    radial_efficiency = np.exp(alpha_radial * cosine_sim)
    radial_term = radial_efficiency * inv_dist

    # 7. Depot-Return Bias
    vec_to_depot = depot - coordinates  # shape (n, 2)
    vec_to_depot_i = vec_to_depot[np.newaxis, :, :] # (1, n, 2)
    dot_depot = np.sum(move_vec * vec_to_depot_i, axis=2) # (n, n)
    dist_i_depot = np.linalg.norm(vec_depot_i[np.newaxis, :, :], axis=2) # (1, n)
    dist_i_depot_safe = dist_i_depot + epsilon
    
    alpha_depot = 2.0
    depot_return_score = np.exp(alpha_depot * dot_depot / dist_i_depot_safe)
    
    # 8. Remaining Capacity Heuristic
    remaining_capacity_ratio = np.maximum(0, (capacity - demands[:, np.newaxis]) / (demands[np.newaxis, :] + epsilon))
    
    # 9. Clarke-Wright Savings Heuristic with Dynamic Angular Coupling
    dist_depot_i = distance_matrix[:, 0] # (n,)
    dist_depot_j = distance_matrix[0, :] # (n,)
    savings = dist_depot_i[:, np.newaxis] + dist_depot_j[np.newaxis, :] - distance_matrix # (n, n)
    
    # Geometric Savings Refinement:
    cosine_angle_diff = np.cos(angle_diff_normalized)
    geometric_savings = savings * np.maximum(cosine_angle_diff, 0.0) # Only positive contributions
    
    # Dynamic Angular Coherence Coupling:
    # Compute local angular density of target node j (mean angular distance to all nodes)
    # Lower mean angle dist -> Higher density
    mean_angle_dist_j = np.mean(angle_diff_normalized, axis=0) # (n,) - mean angular dist of each j
    
    # Normalize density score
    gamma_density = 2.0
    # Higher density (lower mean_angle_dist) gets higher score
    density_score_j = gamma_density / (mean_angle_dist_j / np.pi + epsilon)
    
    # Couple savings with target density
    # Boost savings only when merging into high-density sectors
    savings_density_coupled = geometric_savings * density_score_j[np.newaxis, :]
    
    # Normalize savings
    mean_depot_dist = np.mean(dist_depot_i[1:])
    if mean_depot_dist > 0:
        normalized_savings = savings_density_coupled / mean_depot_dist
    else:
        normalized_savings = savings_density_coupled
        
    # Capacity availability ratio
    capacity_avail_ratio = np.maximum(0, (capacity - demands[:, np.newaxis]) / (capacity + epsilon))
    
    # Prioritize savings when capacity is abundant
    savings_weighted = normalized_savings * capacity_avail_ratio

    # 10. Reintegrate Sector Density Bias
    # Compute mean angular distance of each node j from all other nodes to estimate local angular density.
    mean_angle_dist = np.mean(angle_diff_normalized, axis=1)  # (n,)
    
    gamma_density_bias = 2.0
    # Density score for bias: higher values for denser clusters
    density_score_bias = gamma_density_bias / (mean_angle_dist / np.pi + epsilon)
    sector_density_bias = np.exp(density_score_bias)
    
    # Reshape to (1, n) for broadcasting against (n, n) matrix
    density_bias_mat = sector_density_bias[np.newaxis, :]

    # 11. Combine Heuristics
    # Multiply all terms. The depot return bias adds a multiplicative factor rewarding inward moves.
    # Use angular_coupled_savings in the additive term.
    # Reintegrate sector_density_bias as a multiplicative factor.
    heuristic_matrix = dynamic_inv_dist * angular_coherence * demand_urgency_mat * knn_weight * radial_term * depot_return_score * remaining_capacity_ratio * (1.0 + savings_weighted) * density_bias_mat
    
    # Ensure diagonal is zero
    np.fill_diagonal(heuristic_matrix, 0.0)
    
    # Ensure non-negative
    heuristic_matrix = np.maximum(heuristic_matrix, 0.0)
    
    return heuristic_matrix
