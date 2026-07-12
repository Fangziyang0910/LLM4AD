
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
    eps = 1e-9

    # 1. Distance Term: Strong inverse distance bias
    # Increased power to 4.0 (from Algorithm 1) for stronger local clustering
    dist_safe = np.where(distance_matrix == 0, eps, distance_matrix)
    dist_score = (1.0 / dist_safe) ** 4.0

    # 2. Geometric Coherence: Sector Alignment + Angular Sweep

    # Vector Depot -> i
    v_depot = coordinates - depot_coords[np.newaxis, :]  # Shape (n, 2)

    # --- Sector Cosine ---
    # Cosine similarity between vector from depot to i and vector from depot to j
    dot_prod = np.dot(v_depot, v_depot.T)  # Shape (n, n)
    norms = np.linalg.norm(v_depot, axis=1)  # Shape (n,)
    norms_safe = np.where(norms == 0, eps, norms)
    outer_norms = norms_safe[:, np.newaxis] * norms_safe[np.newaxis, :]
    sector_cosine = dot_prod / outer_norms
    sector_cosine = np.clip(sector_cosine, 0.0, 1.0)
    # Power 2.0 (from Algorithm 2) for moderate clustering effect
    sector_score = sector_cosine ** 2.0

    # --- Angular Sweep Coherence ---
    # Prioritize edges where nodes are close in polar angle relative to depot
    angles = np.arctan2(v_depot[:, 1], v_depot[:, 0])  # Shape (n,)
    angle_diff = angles[np.newaxis, :] - angles[:, np.newaxis]  # Shape (n, n)
    # Wrap angle differences to [-pi, pi]
    angle_diff = np.mod(angle_diff + np.pi, 2 * np.pi) - np.pi

    # Use a Gaussian kernel with narrower bandwidth pi/9 (from Algorithm 1) for stricter local angular ordering
    angle_std = np.pi / 9.0
    angle_score = np.exp(-0.5 * (angle_diff / angle_std) ** 2)

    # Combine Sector and Angle scores multiplicatively
    geo_score = sector_score * angle_score

    # 3. Radial Outward Movement Preference
    # Encourages edges that move radially outward from the depot

    # Vector i -> j
    vec_ij = coordinates[np.newaxis, :, :] - coordinates[:, np.newaxis, :]  # shape (n, n, 2)
    norms_ij = np.linalg.norm(vec_ij, axis=2, keepdims=True)  # shape (n, n, 1)
    norms_ij_safe = np.where(norms_ij == 0, eps, norms_ij)
    unit_ij = vec_ij / norms_ij_safe  # shape (n, n, 2)

    # Vector i -> Depot
    vec_i_depot = depot_coords[np.newaxis, np.newaxis, :] - coordinates[:, np.newaxis, :] # shape (n, n, 2)
    norms_i_depot = np.linalg.norm(vec_i_depot, axis=2, keepdims=True)
    norms_i_depot_safe = np.where(norms_i_depot == 0, eps, norms_i_depot)
    unit_i_depot = vec_i_depot / norms_i_depot_safe # shape (n, n, 2)

    # Cosine similarity between edge i->j and vector i->depot
    # If edge goes towards depot, cos ~ 1. If away, cos ~ -1.
    cos_towards_depot = np.sum(unit_ij * unit_i_depot, axis=2)
    cos_towards_depot = np.clip(cos_towards_depot, -1.0, 1.0)

    # Radial Score: 1.0 if moving away (cos ~ -1), 0.0 if moving towards (cos ~ 1)
    # Increased power to 1.5 (from Algorithm 1) for stronger outward preference
    radial_score_raw = (1.0 - cos_towards_depot) / 2.0
    radial_score = np.power(radial_score_raw, 1.5)

    # 4. Capacity Feasibility Score
    # Use a sigmoid penalty to strictly enforce capacity constraints
    pair_demands = demands[:, np.newaxis] + demands[np.newaxis, :]
    util_ratio = pair_demands / capacity

    # Strict infeasibility mask: demand exceeds capacity
    infeasible_mask = util_ratio > 1.0

    # Sigmoid penalty with sharp drop-off near threshold 0.5 (from Algorithm 1) and slope 15 (from Algorithm 1)
    k = 15.0
    threshold = 0.5
    cap_score = 1.0 / (1.0 + np.exp(k * (util_ratio - threshold)))

    # Zero out infeasible edges explicitly
    cap_score = np.where(infeasible_mask, 0.0, cap_score)

    # 5. Combine Heuristics
    # Multiplicative combination of all components
    heuristics_matrix = dist_score * geo_score * radial_score * cap_score

    # 6. Post-processing
    # Zero out self-loops
    np.fill_diagonal(heuristics_matrix, 0.0)

    # Zero out infeasible edges explicitly again for safety
    heuristics_matrix = np.where(infeasible_mask, 0.0, heuristics_matrix)

    # Ensure non-negative
    heuristics_matrix = np.maximum(heuristics_matrix, 0.0)

    # Handle numerical stability: replace zeros or very small numbers with a small epsilon
    heuristics_matrix = np.where(heuristics_matrix < eps, eps, heuristics_matrix)

    # Ensure finiteness
    heuristics_matrix = np.where(np.isfinite(heuristics_matrix), heuristics_matrix, eps)

    return heuristics_matrix
