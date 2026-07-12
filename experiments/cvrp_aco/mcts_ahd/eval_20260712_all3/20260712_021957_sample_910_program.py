
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
    import numpy as np

    n = distance_matrix.shape[0]
    depot_idx = 0
    eps = 1e-10

    # 1. Sharp Inverse Cubed Distance Metric
    # Inspired by No.1 to strongly prefer very close neighbors for tight clustering.
    dist_safe = np.maximum(distance_matrix, eps)
    dist_factor = 1.0 / (dist_safe ** 3.0)

    # 2. Radial Directionality Bias
    # From No.2: Encourages outward sweeps via exponential dot products.
    depot_coord = coordinates[depot_idx]
    vectors_from_depot = coordinates - depot_coord  # Shape: (n, 2)

    dist_from_depot = np.linalg.norm(vectors_from_depot, axis=1, keepdims=True)
    unit_radial_i = vectors_from_depot / np.maximum(dist_from_depot, eps)

    # Edge vectors i->j: coordinates[j] - coordinates[i]
    edge_vectors_ij = coordinates[np.newaxis, :, :] - coordinates[:, np.newaxis, :]
    edge_dist_ij = np.linalg.norm(edge_vectors_ij, axis=2, keepdims=True)
    unit_edge_ij = edge_vectors_ij / np.maximum(edge_dist_ij, eps)

    # Dot product: radial consistency
    radial_consistency = np.sum(unit_radial_i[:, np.newaxis, :] * unit_edge_ij, axis=2)

    # Exponential bias. alpha_dir from No.2.
    alpha_dir = 2.8
    dir_factor = np.exp(radial_consistency * alpha_dir)

    # 3. Exponential Capacity Penalty
    # From No.2: Sharp penalty for high-demand pairs.
    demands_2d = demands[:, np.newaxis] + demands[np.newaxis, :]
    load_ratio = demands_2d / (capacity + eps)

    # Parameters from No.2 for sharp decay
    beta_cap = 4.0
    gamma_cap = 3.0
    cap_factor = np.exp(-beta_cap * (load_ratio ** gamma_cap))

    # 4. Combine all factors with tuned weights
    # Adjusted weights to balance the new sharp distance metric with the others.
    # Since dist_factor is now sharper (inv-d^3 vs balanced inv-d^1/2), we might reduce its exponent slightly
    # or keep it high to emphasize clustering. Let's try weights similar to No.2 but adapted for the sharper base.
    w_dir = 1.6
    w_dist = 2.5  # Slightly lower than No.2's 3.0 because the base metric is already very sharp
    w_cap = 2.0

    heuristics_matrix = (dir_factor ** w_dir) * \
                        (dist_factor ** w_dist) * \
                        (cap_factor ** w_cap)

    # 5. Suppress self-loops and all depot edges
    # Inspired by No.1: Strictly suppress depot edges to allow pheromones to guide closure.
    np.fill_diagonal(heuristics_matrix, 0.0)
    heuristics_matrix[depot_idx, :] = 0.0
    heuristics_matrix[:, depot_idx] = 0.0

    # 6. Ensure finite positive values
    heuristics_matrix = np.maximum(heuristics_matrix, 1e-9)

    return heuristics_matrix
