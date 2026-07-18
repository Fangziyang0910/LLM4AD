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

    # Baseline value for infeasible or non-existent edges
    BASELINE = 1e-9

    # 1. Compute Local Density
    # Local density is inverse of average distance to k-nearest neighbors
    # We choose k as a small fixed number, e.g., 5, or n-1 if n < 5
    k = min(5, n - 1)

    # To find k-nearest neighbors efficiently:
    # We can use np.argpartition for each row to find the indices of the smallest distances
    # Note: distance_matrix[i, i] is 0, so it will be the first element.
    # We want the smallest distances excluding self?
    # Usually KNN includes self as distance 0.
    # If we include self, the average will be dominated by 0.
    # So we must exclude self.

    # Strategy:
    # For each row, find the indices of the k+1 smallest distances (including self).
    # Then take the top k (excluding the first which is self).

    # Create a copy of distance matrix to avoid modifying original if needed,
    # but argpartition doesn't modify original, it returns indices.

    # Get indices of k+1 smallest distances for each row
    # We need to be careful with the "k" value.
    # If n=2, k=1. We need 2 smallest.

    knn_dists = np.zeros((n, k))

    # Use np.argpartition to get indices of the smallest k+1 elements
    # k_indices[i, j] is the index of the j-th smallest distance in row i
    k_indices = np.argpartition(distance_matrix, k + 1, axis=1)[:, :k+1]

    # Gather the distances for these indices
    # np.take_along_axis is efficient
    k_dists = np.take_along_axis(distance_matrix, k_indices, axis=1)

    # The first one is self (distance 0), remove it
    k_dists_no_self = k_dists[:, 1:]

    # Average distance to k-nearest neighbors
    avg_knn_dist = k_dists_no_self.mean(axis=1)

    # Local density: inverse of average distance
    # Add small epsilon to avoid division by zero
    eps = 1e-6
    local_density = 1.0 / (avg_knn_dist + eps)

    # 2. Compute Inverse Squared Distance Matrix
    # Avoid division by zero for identical points (distance 0)
    dist_safe = np.where(distance_matrix == 0, eps, distance_matrix)
    inv_sq_dist = 1.0 / (dist_safe ** 2)

    # 3. Compute Angular Coherence Matrix
    # Depot coordinates
    depot_coords = coordinates[0]

    # Radial vectors: from Depot to each node i
    # shape (n, 2)
    radial_vectors = coordinates - depot_coords

    # Normalize radial vectors
    radial_norms = np.linalg.norm(radial_vectors, axis=1, keepdims=True)
    radial_norms = np.where(radial_norms == 0, eps, radial_norms)
    radial_directions = radial_vectors / radial_norms  # shape (n, 2)

    # Edge vectors: from node i to node j
    # edge_vecs[i, j] = coords[j] - coords[i]
    # shape (n, n, 2)

    # Efficient computation using broadcasting
    # src coords: (n, 1, 2)
    src_coords = coordinates[:, np.newaxis, :]
    # dst coords: (1, n, 2)
    dst_coords = coordinates[np.newaxis, :, :]

    edge_vectors = dst_coords - src_coords  # shape (n, n, 2)

    # Normalize edge vectors
    edge_norms = np.linalg.norm(edge_vectors, axis=2, keepdims=True)  # shape (n, n, 1)
    edge_norms = np.where(edge_norms == 0, eps, edge_norms)
    edge_directions = edge_vectors / edge_norms  # shape (n, n, 2)

    # Compute dot product for cosine similarity
    # radial_directions shape (n, 2) -> broadcast to (n, 1, 2) for dot product with (n, n, 2)
    radial_dirs_broadcast = radial_directions[:, np.newaxis, :]

    # Dot product sum over last axis (cosine of angle between radial and edge vector)
    cos_angles = np.sum(radial_dirs_broadcast * edge_directions, axis=2)  # shape (n, n)

    # Transform cosine to coherence score [0, 2]
    # 1 + cos(theta): 2 if aligned, 0 if opposed
    angular_coherence = 1.0 + cos_angles

    # 4. Capacity Feasibility Filter
    # Create a mask where destination demand <= capacity
    dest_feasible = demands[np.newaxis, :] <= capacity  # shape (1, n) -> broadcast to (n, n)

    # 5. Combine Scores with Local Density
    # Weighting: Geometric mean of source and destination densities
    # shape (n, 1) * shape (1, n) -> shape (n, n)
    density_factor = np.sqrt(local_density[:, np.newaxis] * local_density[np.newaxis, :])

    # Raw score = inv_sq_dist * angular_coherence * density_factor
    raw_scores = inv_sq_dist * angular_coherence * density_factor

    # Define valid mask: not diagonal and feasible demand
    diag_mask = np.eye(n, dtype=bool)
    valid_mask = (~diag_mask) & dest_feasible

    # Zero out invalid edges
    raw_scores = np.where(valid_mask, raw_scores, 0.0)

    # 6. Dense Row-Normalization (L1)
    # Compute row sums
    row_sums = raw_scores.sum(axis=1, keepdims=True)

    # Avoid division by zero for rows with no feasible edges (sum=0)
    # If sum is 0, we assign BASELINE to all entries in that row later
    row_sums_safe = np.where(row_sums == 0, 1.0, row_sums)

    normalized_scores = raw_scores / row_sums_safe

    # Replace 0s resulting from invalid edges with BASELINE.
    # Valid edges that have 0 score (e.g. due to 0 angular coherence) remain 0.
    heuristic_matrix = np.where(valid_mask, normalized_scores, BASELINE)

    return heuristic_matrix
