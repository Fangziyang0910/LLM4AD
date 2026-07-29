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
    
    depot_coords = coordinates[0]
    
    # Vectors from depot to all nodes
    dx = coordinates[:, 0] - depot_coords[0]
    dy = coordinates[:, 1] - depot_coords[1]
    
    # Calculate angles using atan2
    # Shape: (n,)
    angles = np.arctan2(dy, dx)
    
    # Inverse distance heuristic (proximity)
    # Avoid division by zero for self-loops
    inv_dist = 1.0 / (distance_matrix + 1e-9)
    
    # Initialize heuristic matrix
    H = np.zeros((n, n))
    
    # Precompute demand ratios for customers
    cust_demands = demands[1:]
    demand_ratio = cust_demands / capacity
    
    # 1. Depot to Customer Edges (0 -> j)
    # Modified: Use combined geometric and demand term plus residual capacity penalty.
    # Bias toward customers aligned with primary x-axis (cos(angle)^2 ~ 1) and high demand.
    # Heuristic = inv_dist * (geometric_demand_factor + residual_capacity_factor)
    # residual_capacity_factor = (1 - demand_ratio)
    
    # Calculate cosine squared of angles for customers
    cust_cos_sq = np.cos(angles[1:])**2
    geometric_demand_factor = 1.0 + cust_cos_sq * demand_ratio
    
    # Add residual capacity term to penalize starting with high-demand customers
    # that leave little room for subsequent additions.
    residual_capacity_factor = 1.0 - demand_ratio
    
    combined_factor = geometric_demand_factor + residual_capacity_factor
    H[0, 1:] = inv_dist[0, 1:] * combined_factor
    
    # 2. Customer to Depot Edges (i -> 0)
    # Non-linear Capacity Utilization Pressure:
    # Reward returning to depot from nodes that have high demand (filling the vehicle).
    # Using a power law to sharpen the incentive for high-demand nodes.
    # H[i, 0] = (Demand[i] / Capacity)^2 * (1 / Distance)
    pressure = demand_ratio**2
    H[1:, 0] = pressure * inv_dist[1:, 0]
    
    # 3. Customer to Customer Edges (i -> j)
    # Combine angular similarity, inverse distance, dynamic capacity awareness, and greedy insertion cost
    
    # Extract customer angles
    cust_angles = angles[1:]
    
    # Compute pairwise angle differences for customers
    # Shape: (n-1, n-1)
    cust_angle_diff = cust_angles[:, np.newaxis] - cust_angles[np.newaxis, :]
    
    # Handle periodicity of angles: map to [-pi, pi]
    # This gives the smallest angular difference in [0, pi]
    cust_min_angle_diff = np.abs(np.mod(cust_angle_diff + np.pi, 2 * np.pi) - np.pi)
    
    # Normalize by pi (max possible minimal angle difference is pi)
    cust_rel_angle_diff = cust_min_angle_diff / np.pi
    
    # Exponential decay: similar angles get high weight
    # Adjusted sigma_angle from 0.2 to 0.15 to increase selectivity pressure
    sigma_angle = 0.15
    cust_angular_sim = np.exp(-cust_rel_angle_diff**2 / (2 * sigma_angle**2))
    
    # Inverse distance for customers
    cust_inv_dist = inv_dist[1:, 1:]
    
    # Greedy insertion cost approximation
    # Prioritize edges that are short relative to their individual return costs to depot.
    # Insertion ratio = dist(i, j) / (dist(i, depot) + dist(j, depot))
    # We want SMALL insertion cost, so we use inverse of this ratio.
    
    # dist_2depot[i] for i in 1..n-1
    dist_2depot_cust = distance_matrix[1:, 0]
    
    # Create grid of distances
    dist_i_depot = dist_2depot_cust[:, np.newaxis] # Shape (N-1, 1)
    dist_j_depot = dist_2depot_cust[np.newaxis, :] # Shape (1, N-1)
    
    # Sum of distances to depot for pair (i, j)
    sum_depot_dists = dist_i_depot + dist_j_depot
    
    # Distance between i and j
    dist_ij = distance_matrix[1:, 1:]
    
    # Insertion cost ratio
    insertion_ratio = dist_ij / (sum_depot_dists + 1e-9)
    
    # Heuristic contribution: prefer small ratios. Use inverse of squared ratio for sharper penalty.
    insertion_heuristic = 1.0 / (insertion_ratio**2 + 1e-9)
    
    # Local cluster density heuristic:
    # For each customer, find the distance to their nearest neighbor (excluding self).
    # A smaller distance indicates higher local density.
    # We use the inverse squared distance to this nearest neighbor as a density score.
    # This score is applied to all edges connected to that customer to prioritize 
    # connections within tight clusters.
    
    # Find min distance to any other customer for each customer
    # dist_ij has shape (N-1, N-1). Diagonal is ~0 (or 1e-9 due to inv_dist calc, but here it's dist).
    # We need to ignore self-distances. Set diagonal to infinity before argmin/min.
    dist_ij_copy = dist_ij.copy()
    np.fill_diagonal(dist_ij_copy, np.inf)
    
    nearest_neighbor_dist = np.min(dist_ij_copy, axis=1) # Shape (N-1,)
    
    # Density score: inverse of nearest neighbor distance squared (sharpened)
    # Add epsilon to avoid division by zero if nodes coincide
    density_score = 1.0 / (nearest_neighbor_dist**2 + 1e-9)
    
    # Broadcast density score to the pairwise matrix
    # density_score shape (N-1,) -> (N-1, 1)
    density_score_2d = density_score[:, np.newaxis]
    
    # Dynamic Residual Capacity Awareness (from Reference)
    # Penalize outgoing edges from high-demand nodes to simulate residual capacity constraints.
    # This replaces the static symmetric demand_compat term.
    # Create broadcastable demand ratio for source nodes (i)
    # Shape: (n-1, 1)
    source_demand_ratio = demand_ratio[:, np.newaxis]
    
    # Apply penalty: inversely proportional to source demand
    # Adding a small constant to prevent division by zero and moderate the effect
    capacity_awareness = 1.0 / (source_demand_ratio + 1e-9)
    
    # Depot-Avoidance Penalty (Modified)
    # Replace linear penalty with exponential decay to smoothly penalize internal edges
    # when the source node is far from the depot, preventing abrupt route fragmentation.
    # Factor = exp(-dist(i, depot) / capacity)
    depot_avoidance_factor = np.exp(-dist_i_depot / capacity)
    
    # Combine
    # H_cc = angular_sim * inv_dist * insertion_heuristic * density_score * capacity_awareness * depot_avoidance
    H_cc = cust_angular_sim * cust_inv_dist * insertion_heuristic * density_score_2d * capacity_awareness * depot_avoidance_factor
    
    # Set customer-to-customer edges
    H[1:, 1:] = H_cc
    
    # Zero out diagonal (self-loops)
    np.fill_diagonal(H, 0)
    
    # Ensure positive values
    H = np.maximum(H, 1e-9)
    
    return H
