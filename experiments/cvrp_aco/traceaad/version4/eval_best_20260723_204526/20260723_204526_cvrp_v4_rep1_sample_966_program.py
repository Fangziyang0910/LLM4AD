import numpy as np

def heuristics(distance_matrix: np.ndarray, coordinates: np.ndarray, demands: np.ndarray, capacity: int) -> np.ndarray:
    """Return edge desirability values for CVRP ant colony optimization.

    Args:
        distance_matrix: Pairwise Euclidean distances with shape (n, n).
        coordinates: Node coordinates with shape (n, 2). Node 0 is the depot.
        demands: Node demands with shape (n). The depot demand is zero.
        capacity: Capacity shared by all vehicles.

    Returns:
        An (n, n) edge-prior matrix. Larger values make an edge more likely
        to be sampled. Values at or below zero are treated as 1e-9.
    """
    n = distance_matrix.shape[0]
    
    # Initialize heuristics matrix with zeros
    heur = np.zeros((n, n), dtype=np.float64)
    
    # Depot coordinates
    depot_coord = coordinates[0]
    
    # Compute angles and radial distances of all customers relative to the depot
    dx = coordinates[1:, 0] - depot_coord[0]
    dy = coordinates[1:, 1] - depot_coord[1]
    angles_cust = np.arctan2(dy, dx)
    dists_cust = np.sqrt(dx**2 + dy**2)
    
    # Create full angle and distance arrays including depot. 
    # Depot angle is 0, distance is 0.
    full_angles = np.zeros(n)
    full_angles[1:] = angles_cust
    
    full_dists = np.zeros(n)
    full_dists[1:] = dists_cust
    
    # Precompute 1/distance to avoid division by zero issues later
    inv_dist = 1.0 / (distance_matrix + 1e-9)
    
    # --- Direction-Aware Angular Penalty ---
    angle_i = full_angles[:, np.newaxis]
    angle_j = full_angles[np.newaxis, :]
    
    delta = angle_j - angle_i
    delta = delta % (2 * np.pi)
    delta[delta > np.pi] -= 2 * np.pi
    
    delta_norm = delta / np.pi
    abs_delta_norm = np.abs(delta_norm)
    
    angular_bonus = np.exp(-2.0 * abs_delta_norm)
    direction_bias = 1.0 + 0.3 * delta_norm
    angular_bonus_final = angular_bonus * direction_bias
    
    # --- Cluster-Contextual Feasibility Heuristic (Refined) ---
    std_ang = np.std(angles_cust)
    std_rad = np.std(dists_cust)
    
    alpha_ang = max(std_ang, 1e-9) * 1.5
    alpha_rad = max(std_rad, 1e-9) * 1.5
    
    diff_ang = angles_cust[:, np.newaxis] - angles_cust[np.newaxis, :]
    diff_ang = diff_ang % (2 * np.pi)
    diff_ang[diff_ang > np.pi] -= 2 * np.pi
    diff_ang_abs = np.abs(diff_ang)
    
    diff_rad = dists_cust[:, np.newaxis] - dists_cust[np.newaxis, :]
    diff_rad_abs = np.abs(diff_rad)
    
    exp_ang = np.exp(-diff_ang_abs**2 / (2 * alpha_ang**2))
    exp_rad = np.exp(-diff_rad_abs**2 / (2 * alpha_rad**2))
    cluster_weights = exp_ang * exp_rad
    
    local_demand_sum = cluster_weights @ demands[1:]
    local_influence_sum = cluster_weights.sum(axis=1)
    
    local_demand_density = local_demand_sum / (local_influence_sum + 1e-9)
    
    max_sat = np.max(local_demand_density) if np.max(local_demand_density) > 1e-9 else 1.0
    norm_sat = local_demand_density / (max_sat + 1e-9)
    
    cluster_feasibility_cust = np.exp(-0.5 * norm_sat)
    
    cluster_feasibility = np.zeros(n)
    cluster_feasibility[1:] = cluster_feasibility_cust
    
    same_cluster_bonus = np.exp(-diff_ang_abs**2 / (2 * alpha_ang**2)) * np.exp(-diff_rad_abs**2 / (2 * alpha_rad**2))
    
    same_cluster_full = np.ones((n, n))
    same_cluster_full[1:, 1:] = same_cluster_bonus
    
    cluster_context_bonus = same_cluster_full * cluster_feasibility[np.newaxis, :]
    
    # --- Dynamic Capacity Risk Heuristic ---
    sum_demands = demands[:, np.newaxis] + demands[np.newaxis, :]
    valid_pair = (sum_demands <= capacity).astype(np.float64)
    
    remaining_cap_after_ij = capacity - sum_demands
    median_demand_all = np.median(demands[1:])
    remaining_cap_safe = np.maximum(remaining_cap_after_ij, 0.0)
    cap_ratio = remaining_cap_safe / (median_demand_all + 1e-9)
    
    dynamic_cap_bonus = np.exp(-0.5 * np.maximum(0, 1.0 - cap_ratio))
    
    demand_bonus = np.where(
        valid_pair == 1, 
        dynamic_cap_bonus, 
        0.01
    )
    
    # --- Radial Gradient Alignment Heuristic ---
    diff_radial = np.abs(full_dists[:, np.newaxis] - full_dists[np.newaxis, :])
    
    max_depot_dist = np.max(full_dists)
    if max_depot_dist < 1e-9:
        max_depot_dist = 1.0
        
    norm_diff_radial = diff_radial / max_depot_dist
    
    sigma_strict = 0.5
    radial_alignment_bonus = np.exp(-0.5 * (norm_diff_radial / sigma_strict)**2)
    
    # --- Cluster Locality Heuristic ---
    sorted_indices = np.argsort(full_angles)
    rank = np.empty(n, dtype=np.int32)
    rank[sorted_indices] = np.arange(n)
    
    rank_i = rank[:, np.newaxis]
    rank_j = rank[np.newaxis, :]
    
    delta_rank = rank_j - rank_i
    delta_rank = (delta_rank + n // 2) % n - n // 2
    abs_delta_rank = np.abs(delta_rank)
    
    sigma = n / 10.0 
    if sigma < 1.0:
        sigma = 1.0
        
    cluster_locality_bonus = np.exp(-0.5 * (abs_delta_rank / sigma)**2)
    
    # --- Dynamic Geometric Detour Penalty ---
    dist_0_i = full_dists[:, np.newaxis]
    dist_0_j = full_dists[np.newaxis, :]
    dist_i_j = distance_matrix
    
    denom = dist_0_i + dist_0_j
    denom_safe = np.maximum(denom, 1e-9)
    
    numerator = dist_0_i + dist_i_j + dist_0_j
    
    detour_ratio = numerator / denom_safe
    
    k = 3
    sorted_dists_idx = np.argsort(distance_matrix, axis=1)
    
    kth_neighbor_idx = sorted_dists_idx[:, k]
    kth_neighbor_dist = distance_matrix[np.arange(n), kth_neighbor_idx]
    
    kth_neighbor_dist = np.maximum(kth_neighbor_dist, 1e-9)
    
    median_dist = np.median(kth_neighbor_dist)
    
    density_score = median_dist / kth_neighbor_dist
    
    k_detour_base = 1.5
    
    modulation_i = np.exp(density_score[:, np.newaxis])
    modulation_j = np.exp(density_score[np.newaxis, :])
    
    k_detour_dynamic = k_detour_base * np.sqrt(modulation_i * modulation_j)
    
    detour_excess = (detour_ratio - 1.0)
    
    detour_bonus = np.exp(-k_detour_dynamic * detour_excess)
    
    # --- Continuous Angular Density Heuristic ---
    angle_diff = np.abs(angle_j - angle_i)
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
    
    angular_scale = 0.2 * np.pi 
    
    continuous_angular_bonus = np.exp(-0.5 * (angle_diff / angular_scale)**2)
    
    avg_dist = np.mean(distance_matrix[~np.eye(n, dtype=bool)])
    spatial_scale = avg_dist * 0.5
    
    spatial_bonus = np.exp(-0.5 * (distance_matrix / spatial_scale)**2)
    
    sector_bonus = continuous_angular_bonus * spatial_bonus
    
    # --- Probabilistic Geometric Turning Cost Penalty ---
    coords_i = coordinates[np.newaxis, :, :]
    coords_k = coordinates[:, np.newaxis, :]
    
    vec_ki = coords_i - coords_k
    
    norm_ki = np.linalg.norm(vec_ki, axis=2)
    norm_ki_safe = np.maximum(norm_ki, 1e-9)
    
    inv_dist_safe = 1.0 / (distance_matrix + 1e-9)
    decay_scale = avg_dist if avg_dist > 0 else 1.0
    dist_decay = np.exp(-distance_matrix / decay_scale)
    
    demand_ratio = demands / (capacity + 1e-9)
    cap_weight_k = np.exp(-0.5 * demand_ratio[:, np.newaxis])
    
    weight_ki = inv_dist_safe * dist_decay * cap_weight_k
    
    weight_ki_exp = weight_ki[:, :, np.newaxis]
    
    weighted_vec_ki = weight_ki_exp * vec_ki
    
    expected_vec_i = np.sum(weighted_vec_ki, axis=0)
    
    norm_expected = np.linalg.norm(expected_vec_i, axis=1, keepdims=True)
    norm_expected_safe = np.maximum(norm_expected, 1e-9)
    
    incoming_unit_i = expected_vec_i / norm_expected_safe
    
    coords_j = coordinates[np.newaxis, :, :]
    coords_i_expand = coordinates[:, np.newaxis, :]
    
    vec_ij = coords_j - coords_i_expand
    
    norm_ij = np.linalg.norm(vec_ij, axis=2)
    norm_ij_safe = np.maximum(norm_ij, 1e-9)
    
    outgoing_unit_ij = vec_ij / norm_ij_safe[:, :, np.newaxis]
    
    incoming_unit_i_expand = incoming_unit_i[:, np.newaxis, :]
    
    cos_angle = np.sum(incoming_unit_i_expand * outgoing_unit_ij, axis=2)
    
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    alpha_turn = 1.0
    turning_bonus = np.exp(-alpha_turn * (1.0 - cos_angle))
    
    turning_bonus[0, :] = 1.0
    
    # --- Depot Proximity Penalty ---
    sector_width = np.pi / 3.0
    
    # diff_ang_abs is already computed as (n-1, n-1) for customers
    is_in_sector = (diff_ang_abs < sector_width).astype(np.float64)
    
    max_sector_dist = (is_in_sector * dists_cust[np.newaxis, :]).max(axis=1)
    
    max_sector_dist_full = np.zeros(n)
    max_sector_dist_full[1:] = max_sector_dist
    
    residual_cap = capacity - sum_demands
    
    norm_sector_dist = max_sector_dist_full / (max_depot_dist + 1e-9)
    
    res_cap_ratio = residual_cap / (capacity + 1e-9)
    
    beta = 2.0
    depot_proximity_bonus = np.exp(-beta * np.maximum(0, 1.0 - res_cap_ratio) * norm_sector_dist[np.newaxis, :])
    
    depot_proximity_bonus[0, :] = 1.0
    depot_proximity_bonus[:, 0] = 1.0
    
    # --- Refined Route Length Budget Heuristic ---
    customer_mask = np.ones((n, n), dtype=bool)
    customer_mask[0, :] = False
    customer_mask[:, 0] = False
    np.fill_diagonal(customer_mask, False)
    
    if np.sum(customer_mask) > 0:
        avg_step_dist = np.mean(distance_matrix[customer_mask])
    else:
        avg_step_dist = avg_dist
        
    frac_cap = residual_cap / (capacity + 1e-9)
    
    dist_j_to_depot = distance_matrix[:, 0]
    
    max_route_dist = 2.0 * max_depot_dist + avg_step_dist
    
    allowed_total_dist = max_route_dist * frac_cap
    
    cost_ij_plus_return = distance_matrix + dist_j_to_depot[np.newaxis, :]
    
    excess = cost_ij_plus_return - allowed_total_dist
    
    alpha_budget = 3.0
    route_length_bonus = np.exp(-alpha_budget * np.maximum(0.0, excess))
    
    route_length_bonus[0, :] = 1.0
    route_length_bonus[:, 0] = 1.0
    
    # --- Cluster Boundary Crossing Penalty ---
    # Penalize edges connecting customers in different sectors if distance > local median
    inf = 1e15
    
    # Calculate median distance within sector for each customer j
    dists_cust_nn = distance_matrix[1:, 1:] # (n-1, n-1)
    
    # is_in_sector is (n-1, n-1)
    dists_sector = np.where(is_in_sector, dists_cust_nn, inf)
    
    # Use nanmedian to handle inf
    median_sector_dist = np.nanmedian(dists_sector, axis=1)
    
    # Handle inf or nan from empty sectors
    median_sector_dist = np.nan_to_num(median_sector_dist, nan=1e15, posinf=1e15)
    
    # Avoid division by zero
    median_sector_dist_safe = np.maximum(median_sector_dist, 1e-9)
    
    # Create mask for cross-sector edges
    cross_sector_mask = (is_in_sector == 0).astype(np.float64) # (n-1, n-1)
    
    # Ratio of edge distance to local median
    dist_ratio_cust = dists_cust_nn / median_sector_dist_safe[:, np.newaxis]
    
    # Penalty factor: if cross-sector and dist > median, penalize
    # exp(-alpha * max(0, ratio - 1))
    alpha_boundary = 2.0
    diff_boundary = np.maximum(0.0, dist_ratio_cust - 1.0)
    
    boundary_penalty_factor = np.exp(-alpha_boundary * diff_boundary)
    
    # Apply penalty only if cross-sector
    boundary_bonus_cust = np.where(cross_sector_mask, boundary_penalty_factor, 1.0)
    
    # Expand to full matrix
    boundary_bonus = np.ones((n, n))
    boundary_bonus[1:, 1:] = boundary_bonus_cust
    
    # --- Refined Nearest Neighbor Safety Margin Heuristic ---
    # 1. Identify feasible successors for each node j
    feasible_jk = (demands[np.newaxis, :] + demands[:, np.newaxis] <= capacity)
    
    # 2. Calculate min distance to feasible neighbor for each j
    dist_for_nn = np.where(feasible_jk, distance_matrix, inf)
    min_dist_to_feasible_neighbor = np.min(dist_for_nn, axis=1)
    
    # 3. Calculate median inter-customer distance within the same angular sector for each j
    # Use the same sector definition as Depot Proximity Penalty for consistency
    
    # Median sector dist already computed above as median_sector_dist (n-1)
    
    # 4. Calculate Dynamic Risk Score
    # Ratio of nearest feasible neighbor distance to local median distance
    risk_ratio = min_dist_to_feasible_neighbor[1:] / median_sector_dist_safe
    
    # 5. Calculate Safety Bonus
    # Penalize if risk_ratio is significantly greater than 1
    alpha_nn = 3.0  # Refined from 2.5 to 3.0 for stricter penalty
    diff_nn = np.maximum(0.0, risk_ratio - 1.0)
    
    nn_safety_bonus_cust = np.exp(-alpha_nn * diff_nn)
    
    nn_safety_bonus = np.ones(n)
    nn_safety_bonus[1:] = nn_safety_bonus_cust
    
    # Expand to matrix
    nn_safety_bonus_matrix = nn_safety_bonus[np.newaxis, :]
    
    # --- Combine Heuristics ---
    temp_heur = (inv_dist * 
                 angular_bonus_final * 
                 demand_bonus * 
                 radial_alignment_bonus * 
                 cluster_locality_bonus * 
                 detour_bonus * 
                 sector_bonus *
                 turning_bonus *
                 cluster_context_bonus *
                 depot_proximity_bonus *
                 route_length_bonus *
                 boundary_bonus *
                 nn_safety_bonus_matrix)
    
    # Mask out self-loops
    np.fill_diagonal(temp_heur, 0.0)
    
    heur = temp_heur
    
    # Ensure values are positive
    heur = np.maximum(heur, 1e-9)
    
    return heur
