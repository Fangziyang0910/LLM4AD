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
    
    # --- Precomputations ---
    
    # Ensure capacity is treated as float for calculations
    cap_float = float(capacity)
    epsilon = 1e-9
    
    # Compute demand sum matrix
    demand_sum = demands[:, np.newaxis] + demands[np.newaxis, :]
    
    # Distance matrix safety
    dist_safe = distance_matrix + epsilon
    
    # Coordinates relative to depot (node 0)
    depot_coord = coordinates[0]
    depot_to_node = coordinates - depot_coord  # Shape (n, 2)
    
    # --- Base Heuristic: Residual Capacity Gradient ---
    # Metric: (Residual^2 / Distance^1.5)
    # Amplifies reward for edges that leave significant residual capacity relative to distance cost.
    
    residual = cap_float - demand_sum
    
    # Non-linear Residual Capacity Gradient
    # grad_cap = residual^2 / dist_safe
    # We want residual^2 / dist^1.5 => grad_cap / sqrt(dist_safe)
    grad_cap = (residual ** 2) / dist_safe
    base = grad_cap / np.sqrt(dist_safe)
    
    # --- Directional Cosine Similarity Bonus ---
    # Reward edges that continue in a similar direction to the vector from depot to current node.
    
    norms_dep = np.linalg.norm(depot_to_node, axis=1, keepdims=True)
    norms_dep_safe = np.where(norms_dep > 0, norms_dep, 1.0)
    unit_dep = depot_to_node / norms_dep_safe
    
    # Compute edge vectors: for each i, j, vector i->j
    coord_diff = coordinates[np.newaxis, :, :] - coordinates[:, np.newaxis, :]  # Shape (n, n, 2)
    
    norms_edge = np.linalg.norm(coord_diff, axis=2, keepdims=True)
    norms_edge_safe = np.where(norms_edge > 0, norms_edge, 1.0)
    unit_edge = coord_diff / norms_edge_safe
    
    # Compute dot product of unit_dep[i] and unit_edge[i, j]
    dot_products = np.sum(unit_dep[:, np.newaxis, :] * unit_edge, axis=2)  # Shape (n, n)
    
    # Boost edges with positive cosine (continuing in similar direction)
    # Cosine + 1 gives range [0, 2], where 0 is opposite, 2 is same
    direction_factor = 1.0 + dot_products
    direction_factor = np.maximum(direction_factor, 0.0)
    
    # --- Capacity-Conditioned Directional Bonus ---
    # Modify the directional cosine bonus to be conditional on capacity feasibility.
    # Reward maintaining directionality only when the subsequent node j can fit into 
    # the vehicle's residual capacity after visiting i.
    # Condition: residual > demands[j] implies that j fits in the remaining capacity.
    # residual = capacity - (demands[i] + demands[j])
    # demands[j] shape: (n,), broadcast to (n, n) as demands[np.newaxis, :]
    
    residual_gt_demand_j = (residual > demands[np.newaxis, :]).astype(np.float64)
    
    # Weight: 1.0 + 0.5 * residual_gt_demand_j
    # If feasible (residual > demands[j]), weight is 1.5. If not, weight is 1.0.
    capacity_condition_weight = 1.0 + 0.5 * residual_gt_demand_j
    
    # Apply this weight to the direction factor to make it conditional
    direction_factor = direction_factor * capacity_condition_weight
    
    heuristic = base * direction_factor
    
    # --- Nearest-Neighbor Distance Gradient Penalty ---
    
    # Find min distance from each node to any other node
    dist_copy = distance_matrix.copy()
    np.fill_diagonal(dist_copy, np.inf)
    min_dists = np.min(dist_copy, axis=1)  # Shape (n,)
    
    # Avoid division by zero if min_dists is 0
    min_dists_safe = np.maximum(min_dists, epsilon)
    
    # Calculate relative distance: dist[i,j] / min_dist[i]
    relative_dists = distance_matrix / min_dists_safe[:, np.newaxis]
    
    # Calculate deviation from nearest neighbor (only positive deviations penalized)
    dist_deviation = np.maximum(0, relative_dists - 1.0)
    
    # Scaling factor based on target node demand relative to capacity
    if capacity <= 0:
        cap_scale = np.ones((n, n))
    else:
        cap_scale = cap_float / (cap_float + demands[np.newaxis, :])
        
    # Apply scaling to deviation
    scaled_deviation = dist_deviation * cap_scale
    
    # Compute penalty factor
    alpha_penalty = 1.0
    penalty_factor = 1.0 / (1.0 + alpha_penalty * scaled_deviation)
    
    heuristic = heuristic * penalty_factor
    
    # --- Savings Heuristic Component ---
    # Reward edges where visiting j after i yields distance saving compared to separate depot trips
    # saving_ij = dist(0,i) + dist(0,j) - dist(i,j)
    
    # Distance from depot (node 0) to all nodes
    depot_dists = distance_matrix[0, :]  # Shape (n,)
    
    # Calculate savings matrix
    savings = depot_dists[:, np.newaxis] + depot_dists[np.newaxis, :] - distance_matrix
    
    # Capacity-aware scaling factor for sigma
    if capacity <= 0:
        sigma_factor = np.ones((n, n))
    else:
        sigma_factor = cap_float / (demand_sum + cap_float)
    
    # Calculate mean of non-zero distances to avoid trivial scaling
    nonzero_mask = distance_matrix > 0
    if np.any(nonzero_mask):
        mean_dist = np.sum(distance_matrix[nonzero_mask]) / np.sum(nonzero_mask)
    else:
        mean_dist = 1.0
        
    # Apply capacity-aware scaling to mean distance to get sigma matrix
    sigma_matrix = mean_dist * 0.5 * sigma_factor
    
    # Exponential boost based on savings
    sigma_safe = np.maximum(sigma_matrix, epsilon)
    savings_boost = np.exp(savings / sigma_safe)
    
    # Capacity-utilization penalty term: exp(-demands[j] / capacity)
    if capacity > 0:
        capacity_penalty = np.exp(-demands[np.newaxis, :] / cap_float)
    else:
        capacity_penalty = np.ones((n, n))
        
    savings_factor = savings_boost * capacity_penalty
    
    heuristic = heuristic * savings_factor
    
    # --- Dynamic Angular Proximity Bonus ---
    # Encourage sweep-like route construction by rewarding edges with similar polar angles
    # Applied after savings boost, before final masking
    
    # Calculate polar angles for each node relative to depot
    angles = np.arctan2(depot_to_node[:, 1], depot_to_node[:, 0])  # Shape (n,)
    
    # Calculate angular differences for all pairs (i, j)
    # diff[i, j] = angle[j] - angle[i]
    angle_diff = angles[np.newaxis, :] - angles[:, np.newaxis]
    
    # Normalize angles to [-pi, pi] to handle wrap-around correctly
    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
    
    # Absolute angular difference
    abs_angle_diff = np.abs(angle_diff)
    
    # Convert angle difference to a similarity score (heuristic boost)
    # Use exponential decay: exp(-abs_angle_diff)
    # This strongly rewards adjacent nodes in angular space (sweep order)
    angular_proximity_bonus = np.exp(-abs_angle_diff)
    
    heuristic = heuristic * angular_proximity_bonus
    
    # --- Strict Feasibility Mask for Customer-to-Customer Edges ---
    # Set customer-to-customer edge heuristics to zero if demands[i] + demands[j] > capacity
    
    # Create a mask for customer-to-customer edges (i > 0 and j > 0)
    is_customer = np.arange(n) > 0
    mask_cc = is_customer[:, np.newaxis] & is_customer[np.newaxis, :]
    
    # Create a mask where the demand constraint is violated for customer-to-customer edges
    infeasible_cc = mask_cc & (demand_sum > cap_float)
    
    # Zero out infeasible edges
    heuristic = np.where(infeasible_cc, 0.0, heuristic)
    
    # Ensure finite and handle any potential NaNs or Infs
    heuristic = np.where(np.isfinite(heuristic), heuristic, 0.0)
    
    # Set diagonal to small value to avoid self-transitions
    np.fill_diagonal(heuristic, 0.0)
    
    # Ensure no negative values
    heuristic = np.maximum(heuristic, 0.0)
    
    return heuristic
