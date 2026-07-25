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
    
    # Depot is node 0
    depot_coords = coordinates[0:1, :]  # Shape (1, 2)
    
    # Vectors from depot to all nodes
    node_vectors = coordinates - depot_coords  # Shape (n, 2)
    
    # Radial distances from depot to each node
    radial_distances = np.sqrt(np.sum(node_vectors**2, axis=1, keepdims=True))  # Shape (n, 1)
    
    # Avoid division by zero for depot itself
    radial_distances_safe = np.maximum(radial_distances, 1e-10)
    
    # Normalized direction vectors from depot to each nodes
    direction_vectors = node_vectors / radial_distances_safe  # Shape (n, 2)
    
    # Spatial compactness: Cosine similarity between direction vectors
    # High values indicate nodes are in similar angular sectors from the depot
    cos_angles = np.dot(direction_vectors, direction_vectors.T)  # Shape (n, n)
    
    # Gravity Component: Symmetric demand attraction based on both nodes
    # Formula: 1 + (demand_i + demand_j) / capacity
    combined_demand = demands[:, np.newaxis] + demands[np.newaxis, :]
    demand_attraction = 1.0 + combined_demand / capacity
    
    # Capacity-Weighted Distance Factor (Simplified Stable Version):
    # effective_dist = dist * (1 + demand_j / capacity)
    # This penalizes long edges to high-demand nodes more severely.
    demand_scale = 1.0 + demands[np.newaxis, :] / capacity
    effective_dist = distance_matrix * demand_scale
    effective_dist_safe = np.maximum(effective_dist, 1e-10)
    distance_factor = 1.0 / (effective_dist_safe ** 2)
    
    # Proximity to Depot Bias (Exponential with Dynamic Alpha):
    dist_i = radial_distances.T  # Shape (1, n)
    dist_j = radial_distances    # Shape (n, 1)
    
    # Calculate max radial distance for scaling the exponent
    max_dist = np.max(radial_distances_safe) if np.max(radial_distances_safe) > 0 else 1.0
    
    # Normalize difference
    normalized_diff = (dist_i - dist_j) / max_dist
    
    # Calculate dynamic alpha based on vehicle load ratio
    total_demand = np.sum(demands)
    num_vehicles = np.ceil(total_demand / capacity) if capacity > 0 else 1.0
    
    # The load factor represents how "tight" the capacity constraint is.
    avg_load_ratio = total_demand / (num_vehicles * capacity) if (num_vehicles * capacity) > 0 else 0.0
    
    # Clamp ratio to reasonable bounds to prevent extreme values
    avg_load_ratio = np.clip(avg_load_ratio, 0.0, 1.0)
    
    # Scale dynamic alpha based on load ratio.
    dynamic_alpha = 2.0 * (1.0 + avg_load_ratio)

    # Apply exponential bias.
    depot_bias = np.exp(dynamic_alpha * normalized_diff)
    
    # Angular Momentum Bonus (Refined with Robust Linear Blend):
    vec_ij = coordinates[np.newaxis, :, :] - coordinates[:, np.newaxis, :]
    
    # Normalize vec_ij
    dist_ij = np.linalg.norm(vec_ij, axis=2, keepdims=True)
    dist_ij_safe = np.maximum(dist_ij, 1e-10)
    vec_ij_norm = vec_ij / dist_ij_safe
    
    dir_i_expanded = direction_vectors[:, np.newaxis, :]
    
    # Dot product
    dot_prod = np.sum(dir_i_expanded * vec_ij_norm, axis=2) # Shape (n, n)
    
    # Clip dot product to [-1, 1] for numerical stability
    dot_prod = np.clip(dot_prod, -1.0, 1.0)
    
    # Calculate Local Density for each node using k-Nearest Neighbors (k=5)
    k = 5
    sorted_indices = np.argsort(distance_matrix, axis=1)
    nearest_k_indices = sorted_indices[:, 1:k+1] # Shape (n, k)
    
    rows = np.arange(n)[:, np.newaxis]
    local_dists = distance_matrix[rows, nearest_k_indices] # Shape (n, k)
    
    mean_knn_dist = np.mean(local_dists, axis=1) # Shape (n,)
    
    max_knn_dist = np.max(mean_knn_dist) if np.max(mean_knn_dist) > 0 else 1.0
    normalized_knn_dist = mean_knn_dist / max_knn_dist
    
    # Robust Linear Blend for k_ang:
    base_k = 2.0
    min_adaptive_k = 0.5
    max_adaptive_k = 4.0
    
    adaptive_k = max_adaptive_k - (max_adaptive_k - min_adaptive_k) * normalized_knn_dist
    
    # Linear blend: 50% static base, 50% density-adaptive
    k_ang = 0.5 * base_k + 0.5 * adaptive_k
    
    k_ang_expanded = k_ang[:, np.newaxis]
    
    # Bonus = exp(k_ang * dot_prod)
    angular_momentum_bonus = np.exp(k_ang_expanded * dot_prod)

    # Cluster Affinity Term (with Density Adaptation and Exponential Angular Affinity):
    
    # Estimate area covered by nodes (bounding box area)
    min_x = np.min(coordinates[:, 0])
    max_x = np.max(coordinates[:, 0])
    min_y = np.min(coordinates[:, 1])
    max_y = np.max(coordinates[:, 1])
    area = (max_x - min_x) * (max_y - min_y)
    area_safe = np.maximum(area, 1e-10)
    
    # Node density: nodes per unit area
    density = n / area_safe
    
    # Scale coefficients by density
    density_factor = np.log1p(density)
    
    # Adjust c_1 and c_2 based on density factor
    c_1 = 1.0 * (1.0 + 0.5 * density_factor)
    c_2 = 1.0 * (1.0 + 0.5 * density_factor)
    
    # 1. Angular Proximity (Exponential Decay)
    # Replaced linear cosine with exponential to strengthen angular alignment preference
    angular_affinity = np.exp(c_1 * cos_angles)
    
    # 2. Radial Proximity
    radial_diff = np.abs(dist_j - dist_i) # Shape (n, n)
    
    # Sigma: characteristic scale for radial decay
    sigma = max_dist * 0.5 if max_dist > 0 else 1.0
    
    radial_affinity = np.exp(-c_2 * radial_diff / sigma)
    
    # Combine Angular and Radial Affinity
    cluster_affinity = angular_affinity * radial_affinity

    # Dynamic Remaining Capacity Probability (Refined from Requested Change):
    # Uses geometric distance-to-depot as a proxy for remaining route length to dynamically adjust capacity penalties.
    # Estimate remaining capacity probability based on distance from node i to depot.
    # If dist_to_depot is large, we are "far" from depot, implying potentially less room for more stops if we return.
    # We bias towards nodes j where demand_j is low if est_dist_to_depot is large.
    
    # Estimate remaining capacity capacity_rem proportional to capacity minus estimated travel cost?
    # Simpler: Use radial distance as proxy for "urgency" to return.
    # dist_i: distance from depot to node i. 
    # Let's define a "urgency" factor based on how far node i is from depot.
    # urgency_i = dist_i / max_dist. Higher urgency means we are likely near end of route or deep in space.
    
    dist_i_flat = radial_distances.flatten() # Shape (n,)
    urgency_i = dist_i_flat / (max_dist if max_dist > 0 else 1.0) # Shape (n,)
    
    # Slack available at node i: capacity - demand_i
    slack_i = capacity - demands # Shape (n,)
    slack_i_safe = np.maximum(slack_i, 1e-5)
    
    # Demand of target node j
    demand_j_mat = demands[np.newaxis, :] # Shape (1, n)
    
    # Calculate a dynamic penalty coefficient beta based on urgency.
    # If urgency is high (far from depot or deep in route), we penalize high-demand nodes more.
    # beta_i scales with urgency.
    # Let beta be a matrix broadcasted from urgency.
    urgency_mat = urgency_i[:, np.newaxis] # Shape (n, 1)
    
    # Base penalty ratio: demand_j / slack_i
    demand_slack_ratio = demand_j_mat / slack_i_safe[:, np.newaxis] # Shape (n, n)
    
    # Dynamic exponent: increases with urgency.
    # When urgency is 0 (at depot/start), exponent is base (e.g., 1.0).
    # When urgency is 1 (far), exponent is higher (e.g., 3.0).
    dynamic_exponent = 1.0 + 2.0 * urgency_mat # Shape (n, 1)
    
    # Apply exponential penalty with dynamic exponent
    # cap_proximity_bias = exp(-dynamic_exponent * demand_slack_ratio)
    cap_proximity_bias = np.exp(-dynamic_exponent * demand_slack_ratio)

    # Combine all components
    heur_matrix = (cos_angles * demand_attraction * distance_factor * depot_bias * 
                   angular_momentum_bonus * cluster_affinity * cap_proximity_bias)
    
    # Ensure non-negative values
    heur_matrix = np.maximum(heur_matrix, 1e-9)
    
    return heur_matrix
