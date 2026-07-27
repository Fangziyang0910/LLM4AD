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
        return np.ones((n, n)) * 1e-9

    # --- Precompute Geometric Properties ---
    
    # Depot coordinates
    depot_coords = coordinates[0]
    
    # Vector from depot to each node
    vectors_from_depot = coordinates - depot_coords  # shape (n, 2)
    
    # Distances from depot to each node
    dist_from_depot = np.sqrt(np.sum(vectors_from_depot**2, axis=1))  # shape (n,)
    
    # Angles from depot to each node
    angles = np.arctan2(vectors_from_depot[:, 1], vectors_from_depot[:, 0])  # shape (n,)
    angles[0] = 0.0  # Depot angle
    
    # --- Compute Individual Penalty Terms ---
    
    # 1. Angular Locality Penalty
    angle_diff = angles[np.newaxis, :] - angles[:, np.newaxis]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    
    max_depot_dist = np.max(dist_from_depot[1:]) if n > 1 else 1.0
    sigma_angle = np.pi * (max_depot_dist / (n - 1)) ** 0.5 if n > 1 else np.pi
    sigma_angle = max(sigma_angle, 1e-9)
    
    angular_factor = np.exp(-0.5 * (angle_diff / sigma_angle) ** 2)
    
    # 2. Radial Locality Penalty
    if n > 1:
        sigma_radial = np.std(dist_from_depot[1:])
    else:
        sigma_radial = 1.0
    sigma_radial = max(sigma_radial, 1e-9)
    
    radial_diff = np.abs(dist_from_depot[np.newaxis, :] - dist_from_depot[:, np.newaxis])
    radial_factor = np.exp(-0.5 * (radial_diff / sigma_radial) ** 2)
    
    # 3. Angular Momentum Penalty
    edge_vectors = coordinates[np.newaxis, :, :] - coordinates[:, np.newaxis, :]
    edge_angles = np.arctan2(edge_vectors[:, :, 1], edge_vectors[:, :, 0])
    source_angles = angles[:, np.newaxis]
    
    continuity_angle_diff = edge_angles - source_angles
    continuity_angle_diff = (continuity_angle_diff + np.pi) % (2 * np.pi) - np.pi
    
    mask = ~np.eye(n, dtype=bool)
    all_dists = distance_matrix[mask]
    mean_dist = np.mean(all_dists) if len(all_dists) > 0 else 1.0
    mean_dist = max(mean_dist, 1e-9)
    
    if max_depot_dist > 0:
        sigma_momentum_ang = 0.5 * np.pi * (mean_dist / max_depot_dist)
    else:
        sigma_momentum_ang = 0.5 * np.pi
    sigma_momentum_ang = max(sigma_momentum_ang, 1e-9)
    
    momentum_factor = np.exp(-0.5 * (continuity_angle_diff / sigma_momentum_ang) ** 2)
    
    # 4. Capacity Margin Term
    # Modified: Bidirectional capacity margin encoding
    demand_ratio = np.zeros_like(demands)
    demand_ratio[1:] = demands[1:] / capacity
    
    dist_ratio = distance_matrix / mean_dist
    
    gamma = 0.5
    # Numerator: (1.0 - demand_ratio_j) * (1.0 + demand_ratio_i)
    # This rewards transitions from nodes with high demand (low residual capacity) 
    # to nodes with low demand.
    numerator = (1.0 - demand_ratio[np.newaxis, :]) * (1.0 + demand_ratio[:, np.newaxis])
    denominator = dist_ratio + 1e-9
    
    capacity_exponent = gamma * (numerator / denominator)
    capacity_factor = np.exp(capacity_exponent)
    
    # --- Final Desirability Calculation ---
    
    inv_dist = 1.0 / (distance_matrix + 1e-9)
    
    desirability = inv_dist * angular_factor * radial_factor * momentum_factor * capacity_factor
    
    # --- Cleanup ---
    
    # Zero out depot-related edges and diagonal in the final matrix
    desirability[0, :] = 0.0
    desirability[:, 0] = 0.0
    np.fill_diagonal(desirability, 0.0)
    
    # Ensure non-negative and finite
    desirability = np.maximum(desirability, 1e-9)
    desirability = np.where(np.isfinite(desirability), desirability, 1e-9)
    
    # Final safety zero out
    desirability[0, :] = 0.0
    desirability[:, 0] = 0.0
    np.fill_diagonal(desirability, 0.0)
    
    return desirability
