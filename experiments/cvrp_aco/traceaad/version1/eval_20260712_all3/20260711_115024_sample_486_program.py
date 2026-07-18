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

    # 1. Calculate Average Inter-Customer Distance for Normalization
    # We focus on customer-to-customer distances to gauge local density,
    # excluding the depot row/col to avoid bias from long depot routes.
    # Mask for customer indices (1 to n-1)
    customer_mask = np.ones(n, dtype=bool)
    customer_mask[0] = False

    if np.sum(customer_mask) > 1:
        # Extract submatrix of customer distances
        cust_dist_sub = distance_matrix[np.ix_(customer_mask, customer_mask)]
        # Compute mean of upper triangle to avoid double counting and self-loops
        avg_dist = np.sum(cust_dist_sub) / (cust_dist_sub.size) * 2.0
        # Avoid division by zero if all customers are at same spot (unlikely but safe)
        if avg_dist < 1e-9:
            avg_dist = 1.0
    else:
        # Fallback if less than 2 customers: use global mean distance excluding diagonal
        dist_flat = distance_matrix.flatten()
        # Exclude diagonal elements
        diag_indices = np.arange(n)
        dist_no_diag = dist_flat[np.isin(np.arange(dist_flat.size), diag_indices * (n + 1), invert=True)]
        avg_dist = np.mean(dist_no_diag)
        if avg_dist < 1e-9:
            avg_dist = 1.0

    # 2. Local Edge-Length Adaptive Exponent using Reciprocal Power Law
    # Normalize distances by the average inter-customer distance
    # We set diagonal to 1.0 temporarily to avoid division by zero, will reset later
    dist = distance_matrix.copy()
    np.fill_diagonal(dist, 1.0)

    normalized_dist = dist / avg_dist

    # Define alpha range: [alpha_min, alpha_max]
    # High alpha (e.g., 3.0) for very short edges (normalized_dist -> 0)
    # Low alpha (e.g., 1.5) for very long edges (normalized_dist -> large)
    # Formula: alpha = alpha_min + (alpha_max - alpha_min) / (1.0 + normalized_dist ** 2.0)
    # normalized_dist=0 -> alpha = alpha_max
    # normalized_dist->inf -> alpha = alpha_min

    alpha_min = 1.5
    alpha_max = 3.0

    # Compute local alpha for each edge
    local_alpha = alpha_min + (alpha_max - alpha_min) / (1.0 + normalized_dist ** 2.0)

    # 3. Base Heuristic: Inverse Distance ** local_alpha
    # heuristic[i, j] = 1 / dist[i, j] ^ local_alpha[i, j]
    # Use np.exp and np.log for stable computation of x^y
    # 1 / d^a = exp(-a * ln(d))
    # Note: dist has diagonal 1.0, ln(1)=0, so diagonal becomes exp(0)=1. We will zero it out later.

    # Handle potential zeros in distance matrix (though diagonal is set to 1.0, others shouldn't be 0 unless same coords)
    # Safe log: log(dist) where dist >= 1e-9
    safe_dist = np.maximum(dist, 1e-9)
    log_dist = np.log(safe_dist)

    heuristic = np.exp(-local_alpha * log_dist)

    # Set diagonal to 0 as ants should not visit same node twice
    np.fill_diagonal(heuristic, 0.0)

    # 4. Distance-Aware Demand Feasibility Penalty
    # Calculate distances from depot
    depot_coord = coordinates[0]
    vec_from_depot = coordinates - depot_coord
    dist_from_depot = np.linalg.norm(vec_from_depot, axis=1)

    if np.max(dist_from_depot) > 1e-9:
        normalized_dist_from_depot = dist_from_depot / np.max(dist_from_depot)
    else:
        normalized_dist_from_depot = np.zeros_like(dist_from_depot)

    # Demand factor: 1 / (1 + demand/capacity)
    base_demand_factor = 1.0 / (1.0 + demands / capacity)

    # Distance-aware modification:
    # Close nodes (dist ~ 0): W ~ 1.0 (less penalty)
    # Far nodes (dist ~ 1): W ~ base_demand_factor (full penalty)
    distance_weighted_demand_factor = base_demand_factor + (1.0 - base_demand_factor) * (1.0 - normalized_dist_from_depot)

    # Apply demand factor to columns (destination node j)
    heuristic *= distance_weighted_demand_factor[np.newaxis, :]

    # 5. Magnitude-Weighted Angular Consistency Heuristic
    # Calculate angle of each node relative to the depot (node 0)
    norms = dist_from_depot

    # Handle nodes at the depot (norm=0) to avoid division by zero
    safe_norms = norms.copy()
    safe_norms[safe_norms == 0] = 1.0

    unit_vecs = vec_from_depot / safe_norms[:, np.newaxis]

    # Compute dot product matrix: G[i,j] = u_i . u_j = cos(theta_i - theta_j)
    G = np.dot(unit_vecs, unit_vecs.T)

    # Map cosine to [0, 1] range
    angular_weight_base = 0.5 + 0.5 * G

    # Magnitude weighting: harmonic mean of distances from depot
    # Harmonic mean: 2 / (1/r1 + 1/r2) = 2 * r1 * r2 / (r1 + r2)
    # This penalizes edges connecting nodes at significantly different radii more sharply than geometric mean.
    # Avoid division by zero by ensuring norms are positive
    r1 = norms[:, np.newaxis]
    r2 = norms[np.newaxis, :]

    # Calculate harmonic mean: 2 * (r1 * r2) / (r1 + r2 + eps)
    eps = 1e-9
    harmonic_mean_dist = 2.0 * (r1 * r2) / (r1 + r2 + eps)

    # Normalize the harmonic mean to [0, 1] for consistent scaling
    if np.max(dist_from_depot) > 1e-9:
        magnitude_weight = harmonic_mean_dist / np.max(dist_from_depot)
    else:
        magnitude_weight = np.zeros_like(harmonic_mean_dist)

    # Combine angular weight with magnitude weight
    angular_weight = angular_weight_base * magnitude_weight

    # Integrate local alpha with angular consistency:
    # Raise angular_weight to the power of local_alpha.
    # For short edges (high local_alpha), angular consistency is amplified.
    # For long edges (low local_alpha), angular consistency is diminished.
    # Use exp(local_alpha * log(angular_weight)) for stability.
    # angular_weight is in [0, 1], so log is <= 0.

    # Add small epsilon to angular_weight to avoid log(0)
    safe_angular_weight = np.maximum(angular_weight, 1e-12)
    log_angular_weight = np.log(safe_angular_weight)

    modulated_angular_weight = np.exp(local_alpha * log_angular_weight)

    # Apply modulated angular weight to heuristic
    heuristic *= modulated_angular_weight

    # 6. Capacity Feasibility Boost
    # Identify pairs of customers (i, j) such that demand[i] + demand[j] <= capacity.
    # These edges are more likely to be part of a feasible route.
    # Note: demands[0] is 0 for depot.

    # Create compatibility matrix for all nodes
    # demands shape (n,).
    # compatibility[i, j] is True if demands[i] + demands[j] <= capacity
    demands_col = demands[:, np.newaxis]
    demands_row = demands[np.newaxis, :]
    compatible = (demands_col + demands_row) <= capacity

    # Define a boost factor for compatible edges.
    # A moderate boost (e.g., 1.1 or 1.2) encourages using these edges without dominating the heuristic.
    # Edges involving the depot (demand 0) are always compatible with any single customer demand <= capacity,
    # which is always true for valid inputs. This naturally boosts depot connections, which is good.
    boost_value = 0.2

    # Create boost matrix: 1.0 for incompatible, (1.0 + boost_value) for compatible
    capacity_boost = np.where(compatible, 1.0 + boost_value, 1.0)

    # Apply capacity boost
    heuristic *= capacity_boost

    # Ensure non-negative and handle potential zeros/nans
    # Values at or below 1e-9 are treated as 1e-9 by the caller/framework,
    # but we ensure finite positive values here.
    heuristic = np.where(heuristic < 1e-9, 1e-9, heuristic)

    return heuristic
