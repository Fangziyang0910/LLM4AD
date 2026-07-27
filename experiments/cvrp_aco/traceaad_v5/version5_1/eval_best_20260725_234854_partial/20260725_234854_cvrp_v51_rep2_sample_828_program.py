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
    
    # Implement demand-aware distance exponent:
    # Higher demand nodes get higher exponents, penalizing long distances more heavily for them.
    # Base exponent 4.0, scaled by demand/capacity ratio.
    # demands[np.newaxis, :] broadcasts against the distance_matrix (n, n) such that 
    # the exponent depends on the destination node's demand.
    epsilon = 1e-9
    demand_weighted_dist = distance_matrix ** (4.0 + 2.0 * demands[np.newaxis, :] / capacity)
    
    # Clip to avoid division by zero/inf, then invert to get desirability (higher for shorter dists)
    inv_dist_weighted = 1.0 / np.clip(demand_weighted_dist, epsilon, None)
    
    # Calculate capacity-aware demand scaling using global urgency metric
    # Use max(demands) to stabilize against residual capacity noise
    max_demand = np.max(demands)
    
    # Global urgency: demands[j] / (capacity - max_demand + epsilon)
    # This amplifies priority for high-demand nodes relative to the worst-case constraint
    capacity_scaled_demands = demands / (capacity - max_demand + epsilon)
    
    # capacity_scaled_demands has shape (n,), so we expand dims to (1, n) for broadcasting
    # heuristic_matrix[i, j] = urgency(j) * inv_dist_weighted(i, j)
    heuristic_matrix = capacity_scaled_demands[np.newaxis, :] * inv_dist_weighted
    
    # Add depot-proximity bias for the final leg
    # Calculate distance from each node to the depot (node 0)
    depot_coords = coordinates[0]
    # Euclidean distance from each node to depot
    coordinates_to_depot = np.sqrt(np.sum((coordinates - depot_coords) ** 2, axis=1))
    
    # Apply octic depot bias structure (dist^-8.2) scaled by 0.92
    # Numerator: coordinates_to_depot raised to power 8.2, broadcasted to (1, n)
    # Denominator: distance_matrix raised to power 8.2
    dist_power = distance_matrix ** 8.2
    depot_bias = 0.92 * (coordinates_to_depot ** 8.2)[np.newaxis, :] / np.clip(dist_power, epsilon, None)
    
    # Apply multiplicative scaling factor
    heuristic_matrix *= (1.0 + depot_bias)
    
    # Zero out row 0 to prevent ants from starting new routes from the depot mid-construction
    # (Depot is only visited at start and end of a route, handled by ACO framework logic)
    heuristic_matrix[0, :] = 0.0
    
    return heuristic_matrix
