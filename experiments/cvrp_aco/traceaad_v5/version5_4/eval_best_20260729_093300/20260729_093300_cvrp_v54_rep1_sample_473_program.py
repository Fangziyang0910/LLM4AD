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
    if n < 2:
        return np.ones((n, n)) * 1e-9

    # Use smooth epsilon addition to handle zero distances robustly for distance ratios
    eps = 1e-9
    dist_safe = distance_matrix + eps
    
    # Compute demand-to-distance ratio for each directed edge
    # For edge i -> j, reward is proportional to demand[j] / distance(i, j)
    # We further penalize distance by squaring it (effectively demand[j] / dist^3)
    
    # demands is shape (n,), we need it to broadcast as (1, n) for j index
    ratio = demands[np.newaxis, :] / dist_safe
    inv_dist_sq = 1.0 / (dist_safe ** 2)
    
    heuristic_dist_demand = ratio * inv_dist_sq
    
    # Angular heuristic component:
    # Favor edges i -> j that continue in the direction of the current route from the depot.
    # Compute the angle between vector (0 -> i) and vector (i -> j).
    # Optimized: Use explicit norm calculations with independent zero-masking as per reference.
    
    depot_coord = coordinates[0]  # (2,)
    
    # Vector from depot to i: shape (n, 2)
    v = coordinates - depot_coord[np.newaxis, :]
    
    # Vector from i to j: shape (n, n, 2)
    w = coordinates[np.newaxis, :, :] - coordinates[:, np.newaxis, :]
    
    # Dot product v . w: shape (n, n)
    dot_product = np.sum(v[:, np.newaxis, :] * w, axis=2)  # (n, n)
    
    # Norms calculated explicitly via np.linalg.norm as per reference
    norm_v = np.linalg.norm(v, axis=1)  # (n,)
    norm_w = np.linalg.norm(w, axis=2)  # (n, n)
    
    # Independent zero-masking for norms to prevent artificial bias in short-edge cosine similarities
    norm_v_safe = np.where(norm_v > 0, norm_v, eps)
    norm_w_safe = np.where(norm_w > 0, norm_w, eps)
    
    # Cosine similarity: dot / (norm_v * norm_w)
    cosine_sim = dot_product / (norm_v_safe[:, np.newaxis] * norm_w_safe)
    
    # Clamp cosine similarity to [0, 1]
    angular_factor = np.clip(cosine_sim, 0, 1)
    
    # For edges starting from depot (i=0), norm_v is 0, so cosine_sim is 0/0 -> NaN -> clipped to 0.
    # We want neutral behavior for depot edges, so set angular_factor to 1 for i=0
    angular_factor[0, :] = 1.0
    
    # Combine distance/demand heuristic with angular heuristic
    # Increase weight to 2.0 to more strongly bias towards fan-like routes
    heuristic = heuristic_dist_demand * (1.0 + 2.0 * angular_factor)
    
    # Capacity-feasibility heuristic:
    # Use dynamic quadratic factor from reference: (1 - demands[j]/capacity)^2
    # This naturally approaches 0 as demand approaches capacity, and 1 for small demands.
    
    demand_ratio = demands[np.newaxis, :] / capacity
    demand_ratio = np.clip(demand_ratio, 0, 1)
    capacity_factor = (1.0 - demand_ratio) ** 2
    
    heuristic = heuristic * capacity_factor
    
    # Local route-start detection heuristic:
    # Set heuristic[0, j] to a significantly higher value proportional to demands[j] / dist_safe[0, j]
    # This boosts the probability of starting new routes with high-demand, near-depot customers.
    # Use a dynamic multiplier proportional to capacity to scale appropriately with instance size.
    depot_boost_multiplier = capacity * 10.0
    depot_boost = depot_boost_multiplier * (demands / dist_safe[0, :]) * capacity_factor[0, :]
    heuristic[0, :] = np.maximum(heuristic[0, :], depot_boost)
    
    # Ensure no infinite or NaN values
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    # Ensure all values are positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Zero out self-edges
    np.fill_diagonal(heuristic, 1e-9)
    
    return heuristic
