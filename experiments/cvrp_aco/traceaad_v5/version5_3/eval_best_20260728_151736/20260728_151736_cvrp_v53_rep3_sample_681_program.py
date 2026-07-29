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
    
    # Compute inverse distance, handling self-loops and zero distances
    # Use a small epsilon to avoid division by zero
    epsilon = 1e-8
    inv_dist = np.ones_like(distance_matrix) / (distance_matrix + epsilon)
    
    # Zero out self-loops
    np.fill_diagonal(inv_dist, 0.0)
    
    # 1. Angular Similarity Heuristic (Sweep angle)
    depot_coord = coordinates[0]
    vectors = coordinates - depot_coord
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    
    angles_col = angles.reshape(1, -1)
    angles_row = angles.reshape(-1, 1)
    
    diff_sweep = angles_col - angles_row
    diff_sweep = np.arctan2(np.sin(diff_sweep), np.cos(diff_sweep))
    
    alpha_angle = 3.0
    angular_factor = np.exp(-alpha_angle * np.abs(diff_sweep))

    # 2. Local Distance Decay Heuristic
    mean_dist = np.mean(distance_matrix[distance_matrix > 0]) if np.any(distance_matrix > 0) else 1.0
    if mean_dist == 0:
        mean_dist = 1.0
        
    beta_dist = 2.0
    alpha_local = 1.0
    local_gap_factor = 1.0 + alpha_local * np.exp(-beta_dist * distance_matrix / mean_dist)

    # 3. Nearest-Neighbor Spatial Bias
    # Explicitly favor short edges regardless of angular alignment
    alpha_nn = 2.0
    nn_bias = alpha_nn * np.exp(-distance_matrix / mean_dist)
    
    # Combine Spatial Heuristics
    spatial_heuristic = inv_dist * angular_factor * local_gap_factor * nn_bias
    
    # 4. Route Continuation Bias (Angular Continuity)
    # Compute the angle of the vector from depot to i (incoming direction proxy)
    # And the angle of the vector from i to j (outgoing direction)
    # We want to encourage small turns.
    
    # Vector from depot (0) to i
    vec_depot_i = coordinates - coordinates[0] # (n, 2)
    angle_depot_i = np.arctan2(vec_depot_i[:, 1], vec_depot_i[:, 0]) # (n,)
    
    # Vector from i to j
    # Using broadcasting: coordinates[None, :, :] - coordinates[:, None, :] -> (n, n, 2)
    vec_ij = coordinates[np.newaxis, :, :] - coordinates[:, np.newaxis, :] # (n, n, 2)
    
    # Angle of vector from i to j
    # Handle zero vectors (self-loops) by setting angle to 0 or ignoring later
    angle_ij = np.arctan2(vec_ij[:, :, 1], vec_ij[:, :, 0]) # (n, n)
    
    # Angle of incoming vector to i (from depot)
    # This is constant for a given i, regardless of j.
    angle_in_i = angle_depot_i[:, np.newaxis] # (n, 1) -> broadcast to (n, n)
    
    # Difference in angle
    angle_diff = angle_ij - angle_in_i # (n, n)
    
    # Normalize angle difference to [-pi, pi]
    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
    
    # Apply continuity factor
    lambda_angle = 2.0
    continuity_factor = np.exp(-lambda_angle * np.abs(angle_diff))
    
    # Zero out self-loops in continuity factor (though spatial heuristic already does)
    np.fill_diagonal(continuity_factor, 1.0) # Identity factor for diagonal, will be zeroed out later
    
    # Multiply spatial heuristic by continuity factor
    spatial_heuristic *= continuity_factor
    
    # 5. Dynamic "Remaining Capacity Proxy" Heuristic
    # Compute desirability for edge (i, j) as exp(-k * (demands[i] + demands[j] - capacity))
    # for feasible pairs (where sum <= capacity) and a heavily penalized value otherwise.
    
    demands_col = demands[np.newaxis, :] # (1, n)
    demands_row = demands[:, np.newaxis] # (n, 1)
    pair_demand = demands_row + demands_col # (n, n)
    
    k_cap = 10.0
    
    # Calculate the deviation from capacity
    # If pair_demand <= capacity, deviation is negative or zero, so exp is >= 1
    # If pair_demand > capacity, deviation is positive, so exp is < 1 (penalty)
    deviation = (pair_demand - capacity) / capacity if capacity > 0 else 0.0
    
    # To ensure feasible pairs are strongly favored and infeasible are penalized,
    # we can use a step-like function or a sharp sigmoid. 
    # The prompt suggests: exp(-k * (demands[i] + demands[j] - capacity))
    # Let's normalize the argument to make the penalty sharp.
    # If capacity is 0, handle gracefully (though CVRP capacity > 0 usually)
    if capacity > 0:
        # Scale deviation to be dimensionless
        scaled_deviation = deviation 
        # Apply exponential penalty
        # Note: If deviation is negative (feasible), -k * dev is positive -> exp > 1
        # If deviation is positive (infeasible), -k * dev is negative -> exp < 1
        cap_feasibility_factor = np.exp(-k_cap * scaled_deviation)
    else:
        cap_feasibility_factor = np.ones_like(distance_matrix)

    # Apply capacity feasibility factor to spatial heuristic
    spatial_heuristic *= cap_feasibility_factor
    
    # 6. Cluster Cohesion Term (Chebyshev Distance Affinity)
    # Replace Voronoi sector with Chebyshev distance based affinity
    # Compute pairwise Chebyshev distance
    # coordinates shape: (n, 2)
    # diff_coords shape: (n, n, 2)
    diff_coords = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    # Chebyshev distance is max absolute difference along axes
    chebyshev_dist = np.max(np.abs(diff_coords), axis=2)
    
    # Normalize by maximum possible Chebyshev distance in the dataset
    max_coord_range = np.max(chebyshev_dist)
    if max_coord_range == 0:
        max_coord_range = 1.0
        
    # Normalized Chebyshev distance
    normalized_chebyshev = chebyshev_dist / max_coord_range
    
    # Cluster affinity: favor nodes that are close in Chebyshev distance (dense clusters)
    alpha_cluster = 5.0
    cluster_affinity = 1.0 + alpha_cluster / (normalized_chebyshev + epsilon)
    
    # Multiply spatial heuristic by cluster affinity
    spatial_heuristic *= cluster_affinity

    # 7. Demand-Based Scaling
    # Scale by demands[j] / capacity to prioritize high-demand customers
    if capacity > 0:
        demand_score = demands / capacity
    else:
        demand_score = demands
    
    demand_matrix = demand_score[np.newaxis, :]
    
    # Final Heuristic Matrix
    heuristic_matrix = spatial_heuristic * demand_matrix
    
    # 8. Route Balance Heuristic
    # Replace exponential pairing with rational decay: 1.0 / (1.0 + |d_i - d_j| / capacity)
    # This discourages connecting nodes with vastly different demands
    if capacity > 0:
        demand_diff = np.abs(demands_row - demands_col) # (n, n)
        route_balance_factor = 1.0 / (1.0 + demand_diff / capacity)
        heuristic_matrix *= route_balance_factor

    # 9. Depot-Proximity Coupling Heuristic
    # Favors edges between nodes at similar distances from the depot
    # Promotes radial sweep routes that minimize backtracking and zig-zagging
    dist_to_depot = distance_matrix[:, 0]
    
    # Compute absolute difference in distance to depot for all pairs
    # dist_to_depot[:, np.newaxis] has shape (n, 1)
    # dist_to_depot[np.newaxis, :] has shape (1, n)
    dist_diff = np.abs(dist_to_depot[:, np.newaxis] - dist_to_depot[np.newaxis, :])
    
    # Refinement: Use max(dist_to_depot) for normalization and increase decay constant to 3.0
    max_dist_to_depot = np.max(dist_to_depot)
    if max_dist_to_depot == 0:
        max_dist_to_depot = 1.0
        
    coupling_factor = np.exp(-3.0 * dist_diff / max_dist_to_depot)
    
    heuristic_matrix *= coupling_factor

    # Zero out self-loops and depot-to-depot
    np.fill_diagonal(heuristic_matrix, 0.0)
    heuristic_matrix[0, 0] = 0.0
    
    # Ensure non-negative
    heuristic_matrix = np.maximum(heuristic_matrix, 0.0)
    
    return heuristic_matrix
