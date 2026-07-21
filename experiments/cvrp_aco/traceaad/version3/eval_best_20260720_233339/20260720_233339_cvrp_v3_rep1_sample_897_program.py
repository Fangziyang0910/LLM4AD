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
    if n <= 1:
        return np.zeros((n, n))
        
    epsilon = 1e-9
    
    # 1. Demand-Weighted Proximity
    # Calculate inverse distance for proximity
    inv_dist_raw = 1.0 / (distance_matrix + epsilon)
    
    # Scale by destination node's demand ratio to vehicle capacity.
    # This penalizes edges leading to very high demand nodes relative to capacity,
    # encouraging tighter packing (i.e., preferring nodes that fit well with remaining capacity 
    # implicitly via the heuristic bias, though exact feasibility is handled elsewhere).
    # We use the ratio of the destination demand (j) to capacity.
    # Note: In ACO, we don't know the exact remaining capacity during construction easily without state,
    # but a static bias based on demand/capacity ratio can still guide towards efficient packing 
    # by not overly penalizing large nodes if they are close, or conversely, by adjusting weights.
    # Here, we simply modulate the proximity by (1 + demand_j/capacity) to give higher weight 
    # to nodes that are "worth" the capacity usage if they are close, or conversely, 
    # use 1/(1 + demand_j/capacity) to penalize. 
    # The prompt suggests: "penalizing edges leading to high-demand nodes when remaining capacity is low".
    # Since we don't have dynamic remaining capacity in this static heuristic function, 
    # we interpret this as a static preference for efficient packing. 
    # A common approach is to favor edges that leave "room". 
    # Let's use: Proximity * (1 - demand_j / capacity). This reduces desirability for very large nodes.
    # However, to avoid negative values if demand > capacity (infeasible anyway), we clip.
    
    demand_ratio_dest = demands[np.newaxis, :] / (capacity + epsilon)
    # Clip to [0, 0.99] to ensure positive weight for feasible nodes
    demand_factor = 1.0 - np.clip(demand_ratio_dest, 0.0, 0.99)
    
    demand_weighted_proximity = inv_dist_raw * demand_factor
    
    # 2. Angular Deflection Heuristic (Relative to Depot)
    depot_coord = coordinates[0]
    # Vector from node i to depot
    u = depot_coord - coordinates  # Shape (n, 2)
    
    # Vector from node i to node j
    # Shape (n, n, 2)
    v = coordinates[np.newaxis, :, :] - coordinates[:, np.newaxis, :] 
    
    dot_product_deflect = np.sum(u[:, np.newaxis, :] * v, axis=2)
    
    mag_u = np.linalg.norm(u, axis=1)  # Shape (n,)
    mag_v = distance_matrix  # Shape (n, n)
    
    denom_deflect = mag_u[:, np.newaxis] * mag_v
    denom_deflect = np.maximum(denom_deflect, epsilon)
    
    cos_theta_deflect = dot_product_deflect / denom_deflect
    cos_theta_deflect = np.clip(cos_theta_deflect, -1.0, 1.0)
    
    # Angular factor: 1 - cos_theta. 
    # If going towards depot (angle ~0), cos ~1, factor ~0.
    # If going away/perpendicular, cos ~0, factor ~1.
    angular_factor = 1.0 - cos_theta_deflect
    
    # 3. Sweep-Angle Heuristic
    # Calculate polar angles of all nodes relative to the depot
    deltas = coordinates - depot_coord
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    
    # Compute angular differences for all pairs (i, j)
    # diff[i, j] = angle[j] - angle[i]
    angle_diff = angles[np.newaxis, :] - angles[:, np.newaxis]
    
    # Normalize differences to [-pi, pi] to handle the wrap-around at 2pi
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    
    abs_angle_diff = np.abs(angle_diff)
    
    # 4. Dynamic Lambda based on Local Demand Coefficient of Variation (CV)
    
    # Determine k for neighbor selection
    k = min(10, n - 1)
    
    # Get indices of k nearest neighbors for each node
    sorted_indices = np.argsort(distance_matrix, axis=1)
    
    # Skip the node itself (index 0 in sorted_indices for each row)
    knn_indices = sorted_indices[:, 1:k+1]
    knn_distances = distance_matrix[np.arange(n)[:, np.newaxis], knn_indices]
    
    # A. Estimate Local Demand Statistics via Gaussian Kernel
    
    # Bandwidth h is based on the distance to the k-th nearest neighbor
    kth_dist = knn_distances[:, -1]
    bandwidth = np.maximum(kth_dist, epsilon)
    
    # Calculate Gaussian weights for all nodes j relative to node i
    h_matrix = bandwidth[:, np.newaxis]
    std_dist_sq = (distance_matrix / h_matrix) ** 2
    gaussian_weights = np.exp(-0.5 * std_dist_sq)
    
    # Local weighted mean demand for node i
    weighted_demand_sum = np.sum(gaussian_weights * demands[np.newaxis, :], axis=1)
    weight_sum = np.sum(gaussian_weights, axis=1)
    weight_sum = np.maximum(weight_sum, epsilon)
    
    local_mean_demand = weighted_demand_sum / weight_sum
    
    # Local weighted variance of demand for node i
    # Var(X) = E[X^2] - (E[X])^2
    weighted_demand_sq_sum = np.sum(gaussian_weights * (demands[np.newaxis, :] ** 2), axis=1)
    local_var_demand = (weighted_demand_sq_sum / weight_sum) - (local_mean_demand ** 2)
    local_var_demand = np.maximum(local_var_demand, epsilon)
    
    local_std_demand = np.sqrt(local_var_demand)
    
    # B. Calculate Local Coefficient of Variation (CV)
    # CV = Std / Mean. If Mean is very small, CV is large.
    local_cv = local_std_demand / (local_mean_demand + epsilon)
    
    # C. Dynamic Lambda Calculation
    min_cv = np.min(local_cv)
    max_cv = np.max(local_cv)
    range_cv = max_cv - min_cv
    
    if range_cv < epsilon:
        norm_cv = np.ones(n) * 0.5
    else:
        norm_cv = (local_cv - min_cv) / range_cv
        
    # Map norm_cv [0, 1] to Lambda range [max_lambda, min_lambda]
    min_lambda = 1.0
    max_lambda = 10.0
    
    dynamic_lambda_base = max_lambda * (1.0 - norm_cv) + min_lambda * norm_cv
    
    # Create matrix for broadcasting
    adaptive_lambda_matrix = dynamic_lambda_base[:, np.newaxis]
    
    # Calculate Gaussian weights for dynamic angular decay
    dynamic_angular_decay = np.exp(-adaptive_lambda_matrix * abs_angle_diff)
    
    # 5. Continuous Local Cluster Coherence
    global_mean_dist = np.mean(np.triu(distance_matrix, k=1))
    dist_matrix_for_ratio = np.maximum(distance_matrix, epsilon)
    coherence_boost = (global_mean_dist) / dist_matrix_for_ratio
    
    # 6. Depot Return Proximity Bias
    is_depot_return = np.zeros((n, n))
    is_depot_return[:, 0] = 1.0
    
    demand_ratio = demands / capacity
    depot_return_bias = np.exp(2.0 * demand_ratio)
    depot_return_factor = 1.0 + is_depot_return * (depot_return_bias[:, np.newaxis] - 1.0)
    
    # 7. Capacity Penalty
    demand_sum_ij = demands[:, np.newaxis] + demands[np.newaxis, :]
    heavy_penalty = 1.0 / (1.0 + demand_sum_ij / capacity)
    
    # 8. Sweep Factor (Standard)
    k_sweep = 2.0
    sweep_factor = 1.0 / (1.0 + k_sweep * abs_angle_diff)
    
    # 9. Combine All Heuristics
    # Use the new Demand-Weighted Proximity instead of inv_dist
    heuristic_matrix = (demand_weighted_proximity * angular_factor * dynamic_angular_decay * 
                        heavy_penalty * depot_return_factor * coherence_boost *
                        sweep_factor)
    
    # 10. Mask Infeasible Edges
    np.fill_diagonal(heuristic_matrix, 0.0)
    
    infeasible = demand_sum_ij > capacity
    heuristic_matrix[infeasible] = 0.0
    
    # Ensure non-negative
    heuristic_matrix = np.maximum(heuristic_matrix, 0.0)
    
    return heuristic_matrix
