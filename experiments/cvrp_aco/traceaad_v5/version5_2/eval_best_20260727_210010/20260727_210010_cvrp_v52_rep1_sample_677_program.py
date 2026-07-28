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
    
    # --- Base Heuristic: Inverse Distance ---
    # Avoid division by zero for self-loops and coincident points
    inv_dist = np.where(distance_matrix > 0, 1.0 / distance_matrix, 0.0)
    
    # --- Local Density Penalty (Next-Best-Neighbor) ---
    # Discourage long jumps in sparse regions by scaling edge length against local node density
    K = 5
    k_clip = min(K, n - 1)
    
    # Sort distances to find nearest neighbors for each node
    sorted_distances = np.sort(distance_matrix, axis=1)
    
    # Average distance to K nearest neighbors (excluding self)
    sum_k_nn = np.sum(sorted_distances[:, 1:k_clip+1], axis=1)
    avg_k_nn = sum_k_nn / k_clip
    avg_k_nn = np.maximum(avg_k_nn, 1e-9)
    
    # Combined local scale for edge (i, j) as geometric mean of individual scales
    local_scale_i = avg_k_nn[:, np.newaxis]
    local_scale_j = avg_k_nn[np.newaxis, :]
    combined_scale = np.sqrt(local_scale_i * local_scale_j)
    
    # Penalty factor: exp(-0.5 * (dist / scale)^2)
    ratio = distance_matrix / combined_scale
    density_penalty = np.exp(-0.5 * ratio ** 2)
    
    # --- Angular Consistency ---
    # Reward edges between nodes that are close in angle relative to the depot
    depot_coords = coordinates[0:1, :]
    dx = coordinates[:, 0] - depot_coords[0, 0]
    dy = coordinates[:, 1] - depot_coords[0, 1]
    angles = np.arctan2(dy, dx)
    
    # Angular difference matrix normalized to [-pi, pi]
    angle_diff = angles[:, np.newaxis] - angles[np.newaxis, :]
    angle_diff = np.mod(angle_diff + np.pi, 2 * np.pi) - np.pi
    
    # Calculate median distance from depot to normalize scale
    dists_from_depot_all = distance_matrix[0, :]
    median_dist_from_depot = np.median(dists_from_depot_all[dists_from_depot_all > 0])
    median_dist_from_depot = np.maximum(median_dist_from_depot, 1e-9)
    
    # Compute adaptive sigma for each pair using geometric mean of local K-NN distances
    # This aligns angular guidance with local cluster density rather than global depot proximity
    base_sigma = np.pi / 4.0
    sigma_matrix = base_sigma * np.sqrt(avg_k_nn[:, np.newaxis] * avg_k_nn[np.newaxis, :]) / median_dist_from_depot
    
    # Clamp sigma to prevent numerical instability
    sigma_matrix = np.clip(sigma_matrix, 1e-9, np.pi)
    
    # Angular compatibility
    angular_compatibility = np.exp(-0.5 * (angle_diff / sigma_matrix) ** 2)
    
    # --- Capacity Feasibility Mask ---
    # Zero out edges where the combined demand exceeds vehicle capacity
    combined_demands = demands[:, np.newaxis] + demands[np.newaxis, :]
    capacity_penalty = (combined_demands <= capacity).astype(float)
    
    # --- Combine Heuristics ---
    # Multiplicative combination of factors
    heuristic = inv_dist * angular_compatibility * density_penalty * capacity_penalty
    
    # --- Depot-Specific Adjustments ---
    # Global-mean-normalized depot urgency formulation with adaptive alpha
    if n > 1:
        dists_from_depot = distance_matrix[0, 1:]
        
        # Calculate global mean K-NN for customers (excluding depot)
        global_mean_knn = np.mean(avg_k_nn[1:])
        global_mean_knn = np.maximum(global_mean_knn, 1e-9)
        
        # Adaptive alpha derived from the ratio of global mean K-NN to median depot distance
        # This allows the urgency decay to dynamically adjust to the spatial spread of customers
        # relative to the depot cluster density.
        # A higher ratio implies customers are spread out relative to their local density,
        # so we might want a different decay rate.
        # Base alpha is 1.8. We scale it by the ratio.
        # If ratio is 1 (uniform spread), alpha is 1.8.
        adaptive_alpha = 1.8 * (global_mean_knn / median_dist_from_depot)
        # Clamp adaptive alpha to reasonable bounds to prevent extreme behavior
        adaptive_alpha = np.clip(adaptive_alpha, 0.5, 5.0)
        
        # Apply adaptive urgency factor
        urgency_factor = np.exp(-dists_from_depot / (global_mean_knn * adaptive_alpha)) * (demands[1:] / capacity)
        
        # Apply urgency factor to edges connecting depot and customers
        heuristic[0, 1:] *= urgency_factor
        heuristic[1:, 0] *= urgency_factor
    
    # Ensure diagonal is zero (no self-loops)
    np.fill_diagonal(heuristic, 0.0)
    
    # Ensure non-negative values and replace zeros with small positive value
    heuristic = np.maximum(heuristic, 0.0)
    heuristic = np.where(heuristic <= 0, 1e-9, heuristic)
    
    return heuristic
