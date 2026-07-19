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
    depot = 0
    
    # Distance from depot to each node
    dep_i = distance_matrix[depot, :]  # shape (n,)
    
    # Create matrices for broadcasting
    dep_i_matrix = dep_i[:, np.newaxis]
    dep_j_matrix = dep_i[np.newaxis, :]
    
    # Calculate savings for each pair (i, j)
    # Savings(i, j) = dist(depot, i) + dist(j, depot) - dist(i, j)
    savings_matrix = dep_i_matrix + dep_j_matrix - distance_matrix
    
    # Avoid division by zero for distance matrix
    dist_mat = distance_matrix.copy()
    dist_mat[np.diag_indices_from(dist_mat)] = 1.0
    
    # Savings-based heuristic: favors high savings and short distances
    # H_savings[i, j] = savings[i, j] / dist_mat[i, j]
    heuristic_savings = savings_matrix / dist_mat
    
    # Angular Component
    # Vectors from depot to each node
    depot_coords = coordinates[depot:depot+1, :]  # shape (1, 2)
    vectors_from_depot = coordinates - depot_coords  # shape (n, 2)
    
    # Normalize vectors to get unit vectors
    norms_depot = np.linalg.norm(vectors_from_depot, axis=1, keepdims=True)
    # Avoid division by zero for nodes at depot (if any)
    norms_depot_safe = np.where(norms_depot == 0, 1e-9, norms_depot)
    unit_vectors_from_depot = vectors_from_depot / norms_depot_safe  # shape (n, 2)
    
    # Calculate cosine similarity between vector i and vector j from depot
    cosine_similarity_depot = unit_vectors_from_depot @ unit_vectors_from_depot.T  # shape (n, n)
    
    # Clamp cosine similarity to non-negative range [0, 1] for angular preference
    non_neg_cosine_depot = np.clip(cosine_similarity_depot, 0, 1.0)
    
    # Apply non-linear amplification (power of 3) to sharply increase preference for high alignment
    angular_factor = np.power(non_neg_cosine_depot, 3)
    
    # Savings-Boosted Angular Alignment Component
    alpha = 0.5
    
    # Normalize savings by the maximum savings in the matrix
    max_savings = np.max(savings_matrix)
    if max_savings > 0:
        normalized_savings = savings_matrix / max_savings
    else:
        normalized_savings = np.zeros_like(savings_matrix)
    
    # Base boosted angular factor
    boosted_angular_factor = angular_factor * (1 + alpha * normalized_savings)
    
    # Radial Consistency Component integrated into Angular Bonus
    # Factor: min(dep_i[i], dep_i[j]) / max(dep_i[i], dep_i[j])
    # This penalizes edges connecting nodes at significantly different distances from the depot.
    
    max_dep = np.maximum(dep_i_matrix, dep_j_matrix)
    min_dep = np.minimum(dep_i_matrix, dep_j_matrix)
    
    # Create a safe denominator to avoid division by zero
    denom = np.maximum(max_dep, 1e-9)
    radial_ratio = min_dep / denom
    
    non_depot_mask = (dep_i_matrix > 1e-9) & (dep_j_matrix > 1e-9)
    
    # Where non-depot, use the radial ratio. Where depot is involved, use 1.0 (no penalty/bonus from radial consistency)
    radial_factor = np.where(non_depot_mask, radial_ratio, 1.0)
    
    # Incorporate radial consistency into the angular bonus
    angular_bonus = boosted_angular_factor * radial_factor
    
    # Combine savings-based and enhanced angular-based heuristics
    heuristic_matrix = heuristic_savings * angular_bonus
    
    # --- Directional Continuity Factor ---
    # Calculate the dot product of vector from Depot to i (u) and vector from i to j (v).
    # u = coordinates[i] - coordinates[depot]
    # v = coordinates[j] - coordinates[i]
    # Cosine = dot(u, v) / (|u| * |v|)
    # Factor = 1 + 0.5 * clipped_cosine
    
    # u vectors (n, 2)
    u = vectors_from_depot 
    
    # v vectors (n, n, 2) where v[i, j] = coords[j] - coords[i]
    coords_i = coordinates[:, np.newaxis, :] # (n, 1, 2)
    coords_j = coordinates[np.newaxis, :, :] # (1, n, 2)
    v = coords_j - coords_i # (n, n, 2)
    
    # Compute dot product u . v for each pair (i, j)
    u_bc = u[:, np.newaxis, :] # (n, 1, 2)
    dot_product = np.sum(u_bc * v, axis=2) # (n, n)
    
    # Compute norms |u| and |v|
    norm_u = np.linalg.norm(u, axis=1) # (n,)
    norm_u_mat = norm_u[:, np.newaxis] # (n, 1)
    
    norm_v = np.linalg.norm(v, axis=2) # (n, n)
    
    # Avoid division by zero
    denom_angle = norm_u_mat * norm_v
    
    # Safe denominator
    denom_angle_safe = np.where(denom_angle == 0, 1e-9, denom_angle)
    
    cos_angle_raw = dot_product / denom_angle_safe
    
    # Clip cos_angle to [-1, 1]
    cos_angle_clipped = np.clip(cos_angle_raw, -1, 1)
    
    # Directional Continuity Factor: 1 + 0.5 * cos_angle
    # If cos is 1 (straight), factor is 1.5. If cos is -1 (backwards), factor is 0.5.
    continuity_factor = 1.0 + 0.5 * cos_angle_clipped
    
    # Mask edges where i or j is the depot to avoid geometric ambiguity
    # Also mask self-loops
    depot_mask_i = (dep_i_matrix == 0)
    depot_mask_j = (dep_j_matrix == 0)
    depot_edge_mask = depot_mask_i | depot_mask_j
    
    self_loop_mask = np.eye(n, dtype=bool)
    
    invalid_mask = depot_edge_mask | self_loop_mask
    
    # For invalid edges, use neutral factor 1.0
    continuity_factor_matrix = np.where(invalid_mask, 1.0, continuity_factor)
    
    # Apply directional continuity factor to the heuristic matrix
    heuristic_matrix = heuristic_matrix * continuity_factor_matrix
    
    # Set diagonal to 0 (self-loops are not useful)
    np.fill_diagonal(heuristic_matrix, 0.0)
    
    # Ensure no negative values. Values <= 0 are treated as 1e-9 by the solver.
    heuristic_matrix[heuristic_matrix <= 0] = 1e-9
    
    return heuristic_matrix
