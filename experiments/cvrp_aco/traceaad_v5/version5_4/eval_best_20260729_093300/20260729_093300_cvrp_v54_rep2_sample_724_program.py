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
    epsilon = 1e-8
    
    # 1. Squared Inverse Distance Heuristic
    # Strongly prefer short hops
    dists = distance_matrix + epsilon
    inv_dist_sq = np.power(1.0 / dists, 2.0)
    
    # --- Geometric Factors ---
    # Computes vectors for geometric heuristics
    depot_coords = coordinates[0:1, :]
    v_in = coordinates - depot_coords  # Shape: (n, 2), vector from depot to node i
    
    # v2: Vector from node i to node j
    # Shape: (n, n, 2)
    v2 = coordinates[np.newaxis, :, :] - coordinates[:, np.newaxis, :]
    
    # Expand v_in for broadcasting: (n, 1, 2)
    v_in_expanded = v_in[:, np.newaxis, :]
    
    # --- Path Continuity Factor (Cosine Similarity) ---
    # Dot product
    dot_product = np.sum(v_in_expanded * v2, axis=2)  # Shape: (n, n)
    
    # Magnitudes
    mag_v_in = np.linalg.norm(v_in, axis=1)  # Shape: (n,)
    mag_v2 = dists  # Shape: (n, n)
    
    # Avoid division by zero
    mag_v_in_safe = np.maximum(mag_v_in, epsilon)
    
    # Cosine similarity
    denom = mag_v_in_safe[:, np.newaxis] * mag_v2
    cos_continuity = np.clip(dot_product / (denom + epsilon), -1.0, 1.0)
    
    # Continuity factor: normalize to [0, 1] range
    path_continuity_factor = 0.5 * (1.0 + cos_continuity)
    
    # --- Convex-Hull Boundary Bias ---
    # Calculate cross-product magnitude between depot-to-current (v_in) and current-to-next (v2).
    # Cross product in 2D is x1*y2 - x2*y1.
    # We want to boost edges that turn "inward" relative to the depot-cluster geometry.
    cross_product_magnitude = np.abs(
        v_in_expanded[:, :, 0] * v2[:, :, 1] - v_in_expanded[:, :, 1] * v2[:, :, 0]
    )
    
    # Normalize cross product by the product of magnitudes to get sin(angle)
    sin_term = cross_product_magnitude / (denom + epsilon)
    sin_term = np.clip(sin_term, 0.0, 1.0)
    
    # Bias: Penalize large sin_term (wide swings). 
    boundary_bias = np.exp(-1.0 * sin_term)
    
    # Combine path continuity (straightness) with boundary bias (inward/cluster focus)
    geometric_factor = path_continuity_factor * boundary_bias
    
    # --- Unified Density-Aware Radial Bias ---
    # Compute average nearest-neighbor distance for each source node i.
    k = 5
    
    # Sort distances for each row, skip the first element (self-loop at 0)
    sorted_dists = np.sort(distance_matrix, axis=1)[:, 1:]
    # Take the first k neighbors
    k_nearest_dists = sorted_dists[:, :k]
    # Compute mean
    avg_nnd = np.mean(k_nearest_dists, axis=1, keepdims=True) # Shape (n, 1)
    
    # Density component: exp(-dists / avg_nnd)
    density_component = np.exp(-dists / (avg_nnd + epsilon))
    
    # Radial component: exp(-sin_term)
    radial_component = np.exp(-sin_term)
    
    # Combined unified factor
    local_density_factor = density_component * radial_component

    # --- Hard Capacity Feasibility Mask ---
    demands_col = demands[:, np.newaxis]  # Shape (n, 1)
    demands_row = demands[np.newaxis, :]  # Shape (1, n)
    demand_outer = demands_col * demands_row  # Shape: (n, n)
    demand_sum = demands_col + demands_row
    
    # Hard feasibility mask: strictly prevent infeasible edges
    feasible_mask = (demand_sum <= capacity).astype(np.float64)
    
    # Demand Gravity: Prioritize edges connecting high-demand nodes, scaled by distance
    demand_gravity = demand_outer / dists
    
    # --- Route Completion Bias ---
    # Approximates remaining load based on the current node's demand context.
    # remaining_load approx = capacity - demands[i]
    # Bias encourages closing routes (returning to depot or ending) when load is high.
    # Since this is a static heuristic matrix, we bias edges FROM node i.
    # If node i has high demand, remaining capacity is low, so we penalize outgoing edges
    # to non-depot nodes less? No, we want to encourage returning to depot.
    # The heuristic is for ALL edges. Returning to depot is edge (i, 0).
    # A static matrix cannot know dynamic load. However, the prompt asks for:
    # np.exp(-remaining_load / capacity) where remaining_load is approximated by capacity - demands[i].
    # This implies: bias = exp(-(capacity - demands[i]) / capacity) = exp(-1 + demands[i]/capacity).
    # If demands[i] is high, bias is higher (closer to 1). If demands[i] is low, bias is low (closer to e^-1).
    # This seems to reward high-demand nodes for outgoing edges, which might help "clearing" them.
    
    remaining_load_approx = capacity - demands_col  # Shape (n, 1)
    route_completion_bias = np.exp(-remaining_load_approx / capacity)
    
    # Combine heuristics
    heuristic = inv_dist_sq * geometric_factor * local_density_factor * demand_gravity * feasible_mask * route_completion_bias
    
    # Ensure diagonal is zero
    np.fill_diagonal(heuristic, 0.0)
    
    # Ensure non-negative and apply minimum floor
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
