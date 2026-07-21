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
    
    # 1. Distance Component: Inverse distance raised to power 3 for stronger nearest-neighbor bias
    dist_inv = 1.0 / (distance_matrix + 1e-9)
    dist_reinforced = np.power(dist_inv, 3)
    
    # 2. Capacity/Demand Component: 
    # Favor edges leading to nodes with higher demand relative to capacity.
    demand_importance = demands / capacity
    demand_weighted = dist_reinforced * demand_importance[np.newaxis, :]
    
    # 3. Geometric Components: Curvature, Sector Cohesion, Radial Progression
    
    depot_coord = coordinates[0]
    deltas = coordinates - depot_coord
    
    # --- Sector Cohesion Bonus ---
    num_sectors = 10
    sector_angle = 2 * np.pi / num_sectors
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    angles = np.mod(angles, 2 * np.pi)
    node_sectors = np.floor(angles / sector_angle).astype(int)
    node_sectors = np.mod(node_sectors, num_sectors)
    
    sector_diff = np.abs(node_sectors[:, np.newaxis] - node_sectors[np.newaxis, :])
    circular_diff = np.minimum(sector_diff, num_sectors - sector_diff)
    sector_cohesion = np.exp(-0.5 * circular_diff)
    
    # --- Radial Progression Bias ---
    dist_from_depot = np.linalg.norm(deltas, axis=1)
    dist_j = dist_from_depot[np.newaxis, :]
    dist_i = dist_from_depot[:, np.newaxis]
    is_closer = dist_j < dist_i
    is_sector_close = circular_diff <= 1
    delta_dist = dist_i - dist_j
    radial_bonus = np.where(is_closer & is_sector_close, np.exp(0.5 * delta_dist / (np.max(dist_from_depot) + 1e-9)), 1.0)
    
    # --- Curvature Consistency ---
    d_i_matrix = deltas[np.newaxis, :, :].transpose(1, 0, 2) 
    coords_j = coordinates[np.newaxis, :, :]
    coords_i = coordinates[:, np.newaxis, :]
    e_ij_matrix = coords_j - coords_i
    
    d_i_x = d_i_matrix[:, :, 0]
    d_i_y = d_i_matrix[:, :, 1]
    e_ij_x = e_ij_matrix[:, :, 0]
    e_ij_y = e_ij_matrix[:, :, 1]
    
    cross_z = d_i_x * e_ij_y - d_i_y * e_ij_x
    
    mag_d_i = np.linalg.norm(d_i_matrix, axis=2)
    mag_d_i = np.maximum(mag_d_i, 1e-9)
    
    mag_e_ij = np.linalg.norm(e_ij_matrix, axis=2)
    mag_e_ij = np.maximum(mag_e_ij, 1e-9)
    
    dot_product = np.sum(d_i_matrix * e_ij_matrix, axis=2)
    
    sin_theta = cross_z / (mag_d_i * mag_e_ij)
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    
    cos_theta = dot_product / (mag_d_i * mag_e_ij)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    alpha = 2.0
    beta = 1.0
    curvature_score = np.exp(alpha * cos_theta - beta * (sin_theta ** 2))
    
    geometric_bonus = curvature_score * sector_cohesion * radial_bonus

    # 4. Static Angular Capacity Penalty (Replacement for Demand Compatibility Score)
    
    # Determine k for k-nearest neighbors
    # Use sqrt(n) or a fixed small number, capped by n-1
    k = max(1, min(int(np.sqrt(n)), n - 1))
    
    # Compute average demand of k-nearest neighbors for each node
    # distance_matrix[i, j] gives distance from i to j
    # We want neighbors OF j, so we look at row j in distance_matrix
    # Sort distances along axis 1 to find nearest neighbors
    # We exclude self-loops by setting diagonal to infinity temporarily for sorting
    
    dist_mat_copy = distance_matrix.copy()
    np.fill_diagonal(dist_mat_copy, np.inf)
    
    # Get indices of k nearest neighbors for each node
    # argsort returns indices sorted by distance
    neighbor_indices = np.argsort(dist_mat_copy, axis=1)[:, :k]
    
    # Get demands of these neighbors
    # neighbor_indices shape (n, k)
    neighbor_demands = demands[neighbor_indices] # shape (n, k)
    
    # Average demand of neighbors
    avg_neighbor_demand = np.mean(neighbor_demands, axis=1) # shape (n,)
    
    # For each destination node j, we have avg_neighbor_demand[j]
    # The heuristic estimates the capacity usage if we visit j and a typical neighbor.
    # Estimated Remaining Capacity = Capacity - demand[j] - avg_neighbor_demand[j]
    
    demand_j = demands # shape (n,)
    avg_neigh_d_j = avg_neighbor_demand # shape (n,)
    
    # Broadcast to matrix for edge (i, j): depends only on j
    # Residual depends only on j
    residual_matrix = capacity - demand_j[np.newaxis, :] - avg_neigh_d_j[np.newaxis, :] # shape (n, n)
    
    # Scaling factor: ratio of estimated remaining capacity to demand of j
    # If residual is negative, it means even j + avg_neighbor exceeds capacity -> penalize heavily
    # If residual is small, capacity is tight -> dampen geometric bonus
    # If residual is large, capacity is loose -> allow geometric bonus
    
    # Avoid division by zero
    demand_j_matrix = np.maximum(demand_j[np.newaxis, :], 1e-9) # shape (1, n) -> (n, n)
    
    # Calculate ratio
    capacity_ratio = residual_matrix / demand_j_matrix # shape (n, n)
    
    # Clamp ratio to be at least 0 (negative means infeasible cluster)
    # We want to dampen geometric bonus when ratio is low.
    # Let's use the ratio directly as a multiplier for the geometric bonus.
    # However, if ratio is very large, we don't want to over-amplify. 
    # Let's clip the max ratio to some reasonable value or just use it as is.
    # Actually, the instruction says "dampen... when capacity is constrained".
    # So high ratio -> no dampening (factor ~1 or high). Low ratio -> dampening (factor ~0).
    
    # Let's cap the benefit of high residual to avoid exploding values, 
    # but primarily focus on dampening low residuals.
    # A simple clamp: factor = min(ratio, 2.0) ? 
    # Or just use ratio. If ratio > 1, it boosts. If ratio < 1, it shrinks.
    # Let's use the ratio as the scaling factor for the geometric bonus.
    
    capacity_penalty = capacity_ratio
    
    # Combine geometric bonus with capacity penalty
    # geometric_bonus * capacity_penalty
    # If capacity_penalty is negative (infeasible), we want to kill the edge.
    # So we clamp capacity_penalty to be >= 0.
    effective_geometric_bonus = geometric_bonus * np.maximum(capacity_penalty, 0.0)

    # Combine all components
    heur_matrix = demand_weighted * effective_geometric_bonus
    
    # Boost depot edges significantly to ensure route initiation and completion
    depot_boost = 10.0
    
    # Boost leaving depot (0 -> j)
    heur_matrix[0, :] *= depot_boost
    
    # Boost returning to depot (j -> 0)
    heur_matrix[:, 0] *= depot_boost
    
    # Handle diagonal (self-loops): set to 0 so they are never chosen
    np.fill_diagonal(heur_matrix, 0.0)
    
    # Ensure all values are positive (replace zeros/negatives with a small epsilon)
    heur_matrix = np.maximum(heur_matrix, 1e-9)
    
    return heur_matrix
