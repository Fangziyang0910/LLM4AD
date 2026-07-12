
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
    eps = 1e-9

    # 1. Base Attractiveness: High-power inverse distance
    # Using power 6.0 for very strong local preference, as seen in top performers
    safe_dist = np.where(distance_matrix == 0, eps, distance_matrix)
    inv_dist = 1.0 / safe_dist
    base_attraction = inv_dist ** 6.0

    # 2. Geometric Heuristics

    depot_coord = coordinates[0]
    vectors_from_depot = coordinates - depot_coord[np.newaxis, :]  # (n, 2)

    # Radial distances from depot
    radial_dist = np.sqrt(np.sum(vectors_from_depot ** 2, axis=1))
    radial_dist[0] = 0.0

    # --- Angular Sector Score ---
    angles = np.arctan2(vectors_from_depot[:, 1], vectors_from_depot[:, 0])  # (n,)
    angle_diff = angles[np.newaxis, :] - angles[:, np.newaxis]  # (n, n)

    # Normalize angle difference to [-pi, pi]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

    # Sector score: Sharp Gaussian for tight clustering
    # Using sigma=pi/10 for even tighter clustering than pi/8 (Algo 4,5)
    sigma_angle = np.pi / 10.0
    sector_score = np.exp(-0.5 * (angle_diff / sigma_angle) ** 2)

    # --- Radial Inward Bias ---
    delta_radial = radial_dist[np.newaxis, :] - radial_dist[:, np.newaxis]  # (n, n)
    max_radial = np.max(radial_dist) + eps
    normalized_delta_radial = delta_radial / max_radial

    # Inward score: Sigmoid high when moving closer to depot
    # Using k=6.0 for slightly stronger inward bias than 5.0
    k_radial = 6.0
    inward_score = 1.0 / (1.0 + np.exp(-k_radial * normalized_delta_radial))

    # Local score: Gaussian peak at similar radii
    sigma_radial = 0.15
    local_score = np.exp(-0.5 * (normalized_delta_radial / sigma_radial) ** 2)

    # Blend inward and local scores
    # Higher weight on inward to encourage returning to depot
    blended_radial = 0.8 * inward_score + 0.2 * local_score

    # --- Directional Alignment ---
    norms_from_depot = radial_dist[:, np.newaxis]  # (n, 1)
    norms_from_depot = np.where(norms_from_depot < 1e-9, 1e-9, norms_from_depot)
    dir_from_depot = vectors_from_depot / norms_from_depot  # (n, 2)

    # Vector from i to j
    coord_diff = coordinates[np.newaxis, :, :] - coordinates[:, np.newaxis, :]  # (n, n, 2)

    # Dot product: alignment between (i->j) and (depot->i)
    dir_depot_expanded = dir_from_depot[:, np.newaxis, :]  # (n, 1, 2)
    dot_products = np.sum(dir_depot_expanded * coord_diff, axis=2)  # (n, n)

    # Normalize by distance to get cosine-like score [-1, 1]
    angular_score = np.zeros((n, n))
    mask_valid_dist = distance_matrix > 1e-9
    angular_score[mask_valid_dist] = dot_products[mask_valid_dist] / distance_matrix[mask_valid_dist]

    # Map cosine similarity from [-1, 1] to [0, 2] for weighting
    direction_weight = 1.0 + angular_score

    # --- Detour Efficiency ---
    # Penalize edges that are long relative to the sum of their radial distances to the depot.
    radial_sum = radial_dist[:, np.newaxis] + radial_dist[np.newaxis, :]
    radial_sum_safe = np.where(radial_sum < eps, eps, radial_sum)

    # Ratio: distance(i,j) / (dist(i,depot) + dist(j,depot))
    detour_ratio = distance_matrix / radial_sum_safe

    # Score: 1 when ratio is 0 (perfectly aligned with depot), 0 when ratio >= 1.
    # Using a sharper decay for detour penalty
    detour_score = np.clip(1.0 - detour_ratio**1.5, 0.0, 1.0)

    # Combine geometric scores:
    geometric_score = blended_radial * sector_score * direction_weight * detour_score

    # 3. Demand-Weighted Locality & Capacity Feasibility
    max_demand = np.max(demands)
    if max_demand == 0:
        max_demand = 1.0
    normalized_demands = demands / max_demand

    # Geometric mean of normalized demands for the pair
    demand_weights = np.sqrt(normalized_demands[:, np.newaxis] * normalized_demands[np.newaxis, :])

    # Gaussian decay based on distance for locality bonus
    # Tighter locality bonus to reinforce clustering
    sigma_loc = 15.0
    locality_bonus = np.exp(-0.5 * (distance_matrix / sigma_loc) ** 2)

    demand_locality_factor = demand_weights * locality_bonus

    # Capacity Feasibility Penalty (Steep Sigmoid)
    demand_sum = demands[:, np.newaxis] + demands[np.newaxis, :]

    is_depot = np.zeros(n, dtype=bool)
    is_depot[0] = True
    is_customer = ~is_depot

    # Mask for customer-to-customer edges
    cc_mask = is_customer[:, np.newaxis] & is_customer[np.newaxis, :]

    # Steep sigmoid for capacity violation
    # Using k=70.0 for sharper penalty than previous bests
    k = 70.0
    exponent = k * (demand_sum - capacity)
    exponent = np.clip(exponent, -500, 500) # Prevent overflow
    sigmoid_val = 1.0 / (1.0 + np.exp(exponent))

    # Apply capacity factor only to customer-customer edges
    capacity_factor = np.where(cc_mask, sigmoid_val, 1.0)

    # 4. Construct Final Heuristic Matrix
    heuristics_matrix = base_attraction * geometric_score * demand_locality_factor * capacity_factor

    # Ensure no zeros or negatives for ACO probability calculation
    heuristics_matrix = np.where(heuristics_matrix <= 0, eps, heuristics_matrix)

    # Handle diagonal (no self-loops)
    heuristics_matrix[np.arange(n), np.arange(n)] = eps

    return heuristics_matrix
