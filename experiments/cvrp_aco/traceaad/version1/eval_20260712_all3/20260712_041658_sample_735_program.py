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
    epsilon = 1e-9

    # Calculate dynamic neighborhood size
    # Ensures k is at least 3, at most n-1, and scales linearly with problem size (n/3)
    k_dynamic = max(3, min(int(n / 3), n - 1))

    # 1. Calculate inverse distance for all edges
    # Avoid division by zero by adding a small epsilon
    inv_dist = 1.0 / (distance_matrix + epsilon)

    # 2. Local Density Adjustment
    # Create a copy of distance matrix with infinity on diagonal for sorting/partitioning
    dist_temp = distance_matrix.copy()
    np.fill_diagonal(dist_temp, np.inf)

    if k_dynamic > 0:
        # Get indices of k nearest neighbors using argpartition for efficiency
        k_nn_indices = np.argpartition(dist_temp, k_dynamic, axis=1)[:, :k_dynamic]

        # Gather the distances to these k neighbors
        # Shape: (n, k_dynamic)
        k_nn_distances = dist_temp[np.arange(n)[:, np.newaxis], k_nn_indices]

        # Calculate inverse distances
        k_nn_inv_dist = 1.0 / (k_nn_distances + epsilon)

        # Sum of inverse distances for each node
        # Shape: (n,)
        sum_inv_dist_knn = np.sum(k_nn_inv_dist, axis=1)

        # Dynamic gamma scaling constant proportional to capacity tightness
        total_demand = np.sum(demands[1:])
        tightness = total_demand / capacity
        gamma = 0.1 * tightness

        # Local density factor: 1.0 + gamma * sum(1/dist)
        # Shape: (n,) -> expand to (n, 1) for broadcasting
        local_density_factor = 1.0 + gamma * sum_inv_dist_knn

        # Expand to (n, n) to multiply with edge heuristics
        local_density_factor_matrix = local_density_factor[:, np.newaxis] * np.ones((n, 1), dtype=np.float64)
    else:
        local_density_factor_matrix = np.ones((n, 1), dtype=np.float64)
        sum_inv_dist_knn = np.zeros(n, dtype=np.float64)
        total_demand = np.sum(demands[1:])
        tightness = total_demand / capacity

    # 3. Capacity Feasibility Factor
    # Create a capacity factor matrix initialized to 1.0
    capacity_factor = np.ones((n, n), dtype=np.float64)

    depot_row_mask = np.zeros(n, dtype=bool)
    depot_row_mask[0] = True

    depot_col_mask = np.zeros(n, dtype=bool)
    depot_col_mask[0] = True

    # Mask for customer-to-customer edges
    customer_mask = ~depot_row_mask[:, np.newaxis] & ~depot_col_mask[np.newaxis, :]

    # Calculate combined demands for all pairs
    # Shape: (n, n)
    demands_sum_matrix = demands[:, np.newaxis] + demands[np.newaxis, :]

    # Dynamic penalty factor for customer-to-customer edges
    # Formula: 1.0 / (1.0 + beta * (d_i + d_j) / capacity)
    beta = 0.5

    # Calculate penalty term
    penalty_term = beta * (demands_sum_matrix / capacity)

    # Calculate capacity factor for customer edges
    cap_factor_val = 1.0 / (1.0 + penalty_term)

    # Apply to customer mask
    capacity_factor[customer_mask] = cap_factor_val[customer_mask]

    # 4. Compute Base Heuristic Scores
    # Heuristic(i, j) = inv_dist(i, j) * capacity_factor(i, j) * local_density_factor(i)
    base_heuristics = inv_dist * capacity_factor * local_density_factor_matrix

    # 5. Adaptive Exponent Mechanism
    # total_demand already calculated above
    alpha = 1.0 + (total_demand / capacity)

    # Apply exponent to customer-to-customer edges
    powered_heuristics = base_heuristics.copy()
    powered_heuristics[customer_mask] = np.power(base_heuristics[customer_mask], alpha)

    # 6. Handle diagonal and self-loops
    np.fill_diagonal(powered_heuristics, 0.0)

    # Ensure no negative or zero values
    heuristics_matrix = np.maximum(powered_heuristics, epsilon)

    return heuristics_matrix
