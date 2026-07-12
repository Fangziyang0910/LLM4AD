import random
import math
import scipy
try:
    import torch
except Exception:
    torch = None
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
    eps = 1e-9

    # 1. Base heuristic: Inverse distance^2.05 (Slightly sharper than 2.0 per reflection)
    dist_safe = np.maximum(distance_matrix, eps)
    inv_dist = 1.0 / (dist_safe ** 2.05)

    # 2. Pairwise Demand Calculations using broadcasting
    demands_col = demands[:, np.newaxis]  # (n, 1)
    demands_row = demands[np.newaxis, :]  # (1, n)
    demand_sum = demands_col + demands_row

    # 3. Strict Feasibility Mask
    # Edges that exceed capacity are set to 0.
    feasible_mask = (demand_sum <= capacity).astype(float)

    # 4. Depot-Aware Masking
    # Create a mask for non-depot edges (exclude row 0 and col 0)
    depot_mask = np.ones((n, n))
    depot_mask[0, :] = 0
    depot_mask[:, 0] = 0
    non_depot_mask = depot_mask

    # 5. Utilization Boost: Exponent 2.2
    # ONLY for non-depot edges
    if capacity > 0:
        utilization_ratio = np.clip(demand_sum / capacity, 0.0, 1.0)
        utilization_boost = utilization_ratio ** 2.2
        utilization_boost = utilization_boost * non_depot_mask
    else:
        utilization_boost = np.zeros_like(demand_sum)

    # 6. Feasibility Slack Bonus: Penalize high remaining capacity
    # Coefficient 0.1
    if capacity > 0:
        remaining_capacity = capacity - demand_sum
        remaining_capacity = np.maximum(remaining_capacity, 0.0)

        # Normalize slack by capacity
        slack_norm = remaining_capacity / capacity

        # Inverse squared slack factor: High score for low slack (tight fits)
        slack_factor_internal = 1.0 / (1.0 + 0.1 * (slack_norm ** 2))

        # Apply slack factor only to non-depot edges
        # For depot edges, slack_factor is effectively 1.0 (neutral)
        slack_factor = slack_factor_internal * non_depot_mask + (1.0 - non_depot_mask)
    else:
        slack_factor = np.ones_like(demand_sum)

    # 7. Exploration Nudge: Angular Sector Consistency & Convex Hull Proximity
    # Calculate angles of all nodes relative to depot
    offsets = coordinates - coordinates[0] # (n, 2)
    angles = np.arctan2(offsets[:, 1], offsets[:, 0]) # (n,)

    # Angle difference for edge i->j
    angles_col = angles[:, np.newaxis]
    angles_row = angles[np.newaxis, :]

    # Handle wrap around
    diff = angles_col - angles_row
    diff = np.mod(diff + np.pi, 2*np.pi) - np.pi # Normalize to [-pi, pi]

    # Normalize diff to [0, 1] by dividing by pi
    norm_diff = np.abs(diff) / np.pi

    # Apply a Gaussian-like bonus for small angular differences
    angle_bonus = np.exp(-5.0 * (norm_diff ** 2))

    # New Mechanism: Convex Hull Proximity Bonus
    # Edges between customers on the convex hull are often critical for TSP/CVRP structure.
    # We approximate hull importance by distance to the depot (hull nodes are typically far).
    # However, simple distance is already handled by inv_dist.
    # Instead, let's look at "local" hull density.
    # A simpler heuristic for "hull-like" edges is connecting nodes that have few neighbors
    # in a specific angular sector, but that's expensive.
    # Let's use a simple geometric proxy: The cross product magnitude relative to distance
    # indicates how "straight" a path is relative to the origin? No.

    # Let's stick to the directive's angular bonus but ensure depot neutrality.
    angle_bonus = angle_bonus * non_depot_mask + (1.0 - non_depot_mask)

    # 8. Combine components multiplicatively
    heuristic = inv_dist * feasible_mask * utilization_boost * slack_factor * angle_bonus

    # 9. Zero out self-loops
    np.fill_diagonal(heuristic, 0)

    return heuristic
