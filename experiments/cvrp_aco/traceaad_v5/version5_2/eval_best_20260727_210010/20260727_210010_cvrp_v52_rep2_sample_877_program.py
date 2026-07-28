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
    depot_coord = coordinates[0]
    
    # Vectorized coordinates for all nodes
    node_coords = coordinates  # shape (n, 2)
    
    # Initialize heuristic matrix with zeros
    heur_matrix = np.zeros_like(distance_matrix, dtype=float)
    
    # --- 1. Depot-to-Customer Edges (Row 0) ---
    # Use the improved heuristic from Current Program History Step 1
    # Heuristic = Inverse Distance * (Capacity / (Demand + 1.0))
    if n > 1:
        customers = np.arange(1, n)
        
        # Get distances and demands for customers
        dists_depot_to_cust = distance_matrix[0, customers]
        demands_cust = demands[customers]
        
        # Calculate inverse distance safely
        epsilon = 1e-8
        inv_dist_depot = np.zeros_like(dists_depot_to_cust)
        nonzero_mask = dists_depot_to_cust > 0
        inv_dist_depot[nonzero_mask] = 1.0 / dists_depot_to_cust[nonzero_mask]
        
        # Calculate demand scaling factor
        demand_scale = capacity / (demands_cust + 1.0)
        
        # Compute specific depot heuristic
        depot_heur = inv_dist_depot * demand_scale
        
        # Assign to row 0, customer columns
        heur_matrix[0, customers] = depot_heur

    # --- 2. Customer-to-Customer Edges (Submatrix 1:, 1:) ---
    # Use the Reference Program's validated structure:
    # inv_dist * exp(-7.0 * |angle_diff|) * (1 + cos_theta)
    if n > 2:
        c_coords = coordinates[1:]
        d_coord = coordinates[0]
        
        # Vectors from depot to each customer
        v_di = c_coords - d_coord[np.newaxis, :]
        
        # Polar angles relative to depot
        angles = np.arctan2(v_di[:, 1], v_di[:, 0])
        
        # Pairwise angle differences normalized to [-pi, pi]
        angle_diff = angles[np.newaxis, :] - angles[:, np.newaxis]
        angle_diff_normalized = np.mod(angle_diff + np.pi, 2 * np.pi) - np.pi
        
        # Angular proximity factor: exp(-alpha * |angle_diff|)
        # alpha = 7.0
        alpha = 7.0
        angular_factor = np.exp(-alpha * np.abs(angle_diff_normalized))
        
        # Inverse distance
        dist_ij = distance_matrix[1:, 1:]
        inv_dist_ij = np.zeros_like(dist_ij)
        nonzero_dist_mask = dist_ij > 0
        inv_dist_ij[nonzero_dist_mask] = 1.0 / dist_ij[nonzero_dist_mask]
        
        # Local Geometric Consistency: Cosine similarity between depot->i and i->j vectors
        # Vector u: Depot -> i (v_di[:, i])
        # Vector w: i -> j (c_coords[j] - c_coords[i])
        
        u_x = v_di[:, 0]
        u_y = v_di[:, 1]
        
        w_x = c_coords[:, 0][np.newaxis, :] - c_coords[:, 0][:, np.newaxis]
        w_y = c_coords[:, 1][np.newaxis, :] - c_coords[:, 1][:, np.newaxis]
        
        dot_prod = u_x[:, np.newaxis] * w_x + u_y[:, np.newaxis] * w_y
        
        norm_u = np.sqrt(u_x**2 + u_y**2)
        norm_w = np.sqrt(w_x**2 + w_y**2)
        
        eps = 1e-10
        cos_theta = dot_prod / (norm_u[:, np.newaxis] * norm_w + eps)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        # Local consistency bonus: 1 + cos(theta)
        local_consistency_factor = 1.0 + cos_theta
        
        # Combine all factors
        customer_heuristics = inv_dist_ij * angular_factor * local_consistency_factor
        
        heur_matrix[1:, 1:] = customer_heuristics

    # --- 3. Customer-to-Depot Edges (Column 0) ---
    # Heuristics for returning to depot are set to 0 to defer to pheromones/constraints
    heur_matrix[:, 0] = 0.0

    # --- Soft Capacity Penalty ---
    # Apply the sigmoid-based soft capacity penalty from Current Program History Step 2
    # This penalizes edges where the sum of connected node demands approaches vehicle capacity.
    
    # Compute sum of demands for each edge (i, j)
    demands_col = demands[:, np.newaxis]  # shape (n, 1)
    demands_row = demands[np.newaxis, :]  # shape (1, n)
    pair_demands = demands_col + demands_row  # shape (n, n)
    
    threshold = 0.8
    k = 10.0  # Steepness of the sigmoid
    
    ratio = pair_demands / capacity
    # Sigmoid: 1 / (1 + exp(k * (ratio - threshold)))
    capacity_penalty = 1.0 / (1.0 + np.exp(k * (ratio - threshold)))
    
    # Apply the penalty to the heuristic matrix
    heur_matrix = heur_matrix * capacity_penalty
    
    # Ensure diagonal is zero (no self-loops)
    np.fill_diagonal(heur_matrix, 0.0)
    
    # Ensure no negative values; small positive value for invalid/zero entries
    epsilon = 1e-9
    heur_matrix = np.maximum(heur_matrix, epsilon)
    
    return heur_matrix
