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
    if n <= 1:
        return np.zeros_like(distance_matrix)

    eps = 1e-10

    # 1. Base Heuristic: Inverse Distance * Demand Weighting
    safe_distance = np.where(distance_matrix == 0, eps, distance_matrix)
    inv_dist = 1.0 / safe_distance

    demand_j = demands[np.newaxis, :]  # Shape (1, n)
    demand_ratio = np.clip(demand_j / capacity, 0.0, 1.0)
    # Use exponent 0.5 for stability
    demand_weight = np.power(demand_ratio, 0.5)

    # 2. Hybrid Capacity Penalty
    demands_i = demands[:, np.newaxis]  # Shape (n, 1)
    combined_demand = demands_i + demand_j

    # Determine max and min single customer demand for penalty components
    if n > 1:
        customer_demands = demands[1:]
        max_demand = np.max(customer_demands)
        min_demand = np.min(customer_demands)
        if max_demand <= eps:
            max_demand = eps
        if min_demand <= eps:
            min_demand = eps
    else:
        max_demand = eps
        min_demand = eps

    residual = capacity - combined_demand

    # Linear component: 0.0 when residual < 0, linear growth up to max_demand
    linear_term = np.clip(residual / max_demand, 0.0, 1.0)

    # Convex component: Smooth penalty based on deviation from a safe buffer (min_demand)
    deficit = np.maximum(0.0, min_demand - residual)
    normalized_deficit = deficit / capacity
    convex_term = 1.0 / (1.0 + (normalized_deficit ** 2.0))

    # Hybrid: Product of linear (0 to 1) and convex (0 to 1) factors
    penalty_factor = linear_term * convex_term

    # Combine base heuristic and capacity penalty
    heuristic = inv_dist * demand_weight * penalty_factor

    # 3. Geometric Constraints for Internal Edges (i>0, j>0)
    depot_coords = coordinates[0]
    vec_from_depot = coordinates - depot_coords  # Shape (n, 2)

    # Edge vectors i -> j
    coords_j = coordinates[np.newaxis, :, :]  # Shape (1, n, 2)
    coords_i = coordinates[:, np.newaxis, :]  # Shape (n, 1, 2)
    edge_vec = coords_j - coords_i  # Shape (n, n, 2)

    # Normalize vectors for dot product calculation
    norms_from_depot = np.linalg.norm(vec_from_depot, axis=1, keepdims=True)
    norms_from_depot = np.clip(norms_from_depot, eps, None)
    unit_from_depot = vec_from_depot / norms_from_depot  # Shape (n, 2)

    norms_edge = np.linalg.norm(edge_vec, axis=2, keepdims=True)
    norms_edge = np.clip(norms_edge, eps, None)
    unit_edge = edge_vec / norms_edge  # Shape (n, n, 2)

    # Dot product: cos(theta) between radial vector from depot (to i) and edge vector (i->j)
    unit_from_depot_bc = unit_from_depot[:, np.newaxis, :]  # Shape (n, 1, 2)
    dot_prod = np.sum(unit_from_depot_bc * unit_edge, axis=2)  # Shape (n, n)

    # --- Internal Edges (i>0, j>0) ---
    internal_mask = np.ones((n, n), dtype=bool)
    internal_mask[0, :] = False # From depot
    internal_mask[:, 0] = False # To depot

    if np.any(internal_mask):
        # A. Radial Clustering
        # Strict spatial coherence: gamma=3.5
        depot_dists = distance_matrix[0]
        diff_radial = np.abs(depot_dists[:, np.newaxis] - depot_dists[np.newaxis, :])
        local_scale = np.maximum(depot_dists[:, np.newaxis], depot_dists[np.newaxis, :])
        normalized_diff = diff_radial / (local_scale + eps)
        gamma = 3.5
        radial_penalty = 1.0 / (1.0 + (normalized_diff ** gamma))

        # B. Local Angular Penalty
        # Relaxed angular penalty to allow smoother turns
        dev = 1.0 - dot_prod
        dev = np.clip(dev, 0.0, None)
        lambda_ang = 0.6
        angular_factor = 1.0 / (1.0 + lambda_ang * (dev ** 2.0))

        # C. Directional Flow Incentive
        directional_factor = 0.5 + 0.5 * np.clip(dot_prod, 0.0, 1.0)

        # D. Nearest Neighbor Directionality Nudge
        # Increased to 1.40 to further strengthen local cluster coherence
        nn_mask = np.zeros((n, n), dtype=bool)
        customer_dists = distance_matrix[1:, 1:]
        cust_indices = np.arange(1, n)
        nearest_j_indices = np.argmin(customer_dists, axis=1)

        for idx, i in enumerate(cust_indices):
            j = cust_indices[nearest_j_indices[idx]]
            nn_mask[i, j] = True

        nn_factor = np.ones((n, n))
        nn_factor[nn_mask] = 1.40

        # Tangential bonus removed as per directive

        # Apply factors multiplicatively to internal edges
        internal_indices = np.where(internal_mask)
        heuristic[internal_indices] *= radial_penalty[internal_indices]
        heuristic[internal_indices] *= angular_factor[internal_indices]
        heuristic[internal_indices] *= directional_factor[internal_indices]
        heuristic[internal_indices] *= nn_factor[internal_indices]

    # 4. Depot Edge Geometry
    depot_edge_mask = np.zeros((n, n), dtype=bool)
    depot_edge_mask[0, 1:] = True # From depot to customers
    depot_edge_mask[1:, 0] = True # From customers to depot

    if np.any(depot_edge_mask):
        depot_indices = np.where(depot_edge_mask)

        # Vector towards depot for all nodes
        vec_to_depot = depot_coords[np.newaxis, :] - coordinates[:, np.newaxis, :]

        # Normalize vec_to_depot
        norms_to_depot = np.linalg.norm(vec_to_depot, axis=2, keepdims=True)
        norms_to_depot = np.clip(norms_to_depot, eps, None)
        unit_to_depot = vec_to_depot / norms_to_depot

        # Dot product between Edge Vector and Unit Vector Towards Depot
        depot_align_dot = np.sum(unit_edge * unit_to_depot, axis=2)

        # Strict returns to depot
        lambda_depot = 16.0
        depot_dev = 1.0 - depot_align_dot
        depot_dev_sq = depot_dev ** 2
        depot_factor = 1.0 / (1.0 + lambda_depot * depot_dev_sq)

        # Asymmetric logic: 1.0 for exit, calculated factor for return
        depot_exit_mask = np.zeros((n, n), dtype=bool)
        depot_exit_mask[0, 1:] = True
        adjusted_depot_factor = np.where(depot_exit_mask, 1.0, depot_factor)

        heuristic[depot_indices] *= adjusted_depot_factor[depot_indices]

    # 5. Finalization
    np.fill_diagonal(heuristic, 0.0)

    heuristic = np.clip(heuristic, 0.0, None)
    heuristic = np.where(heuristic <= 0.0, 1e-9, heuristic)
    heuristic = np.nan_to_num(heuristic, nan=1e-9, posinf=1e9, neginf=1e-9)

    return heuristic
