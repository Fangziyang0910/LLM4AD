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
    
    # Create a copy of the distance matrix for the heuristic
    heuristic_matrix = np.copy(distance_matrix)
    
    # Avoid division by zero where distance is zero (same node)
    # Set diagonal to infinity temporarily so it becomes 0 in inverse
    np.fill_diagonal(heuristic_matrix, 0.0)
    
    # Calculate the inverse distance (visibility)
    # We use a small epsilon to avoid division by zero
    epsilon = 1e-8
    visibility = 1.0 / (heuristic_matrix + epsilon)
    
    # Penalize self-loops more strongly
    np.fill_diagonal(visibility, 0.0)
    
    # Calculate angles of all nodes relative to the depot (node 0)
    # Depot is at coordinates[0]
    depot_coords = coordinates[0]
    
    # Calculate vectors from depot to all nodes
    vectors = coordinates - depot_coords
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    
    # For each pair (i, j), calculate the angular difference
    # We want to favor edges that connect nodes with similar angles (cluster affinity)
    # but also consider the direction of travel
    
    # Create angle matrix for all pairs
    angles_matrix = angles[:, np.newaxis]  # (n, 1)
    angles_matrix_T = angles[np.newaxis, :]  # (1, n)
    
    # Calculate angular difference for all pairs
    angular_diff = np.abs(angles_matrix - angles_matrix_T)
    
    # Handle wrap-around for angular difference
    angular_diff = np.minimum(angular_diff, 2 * np.pi - angular_diff)
    
    # Create a factor that rewards small angular differences
    # Tuned to 3.0 to strengthen bias towards geographically clustered routes
    k = 3.0  # Tunable parameter for angular sensitivity
    angular_factor = np.exp(-k * angular_diff)
    
    # Combine visibility with angular factor
    # This encourages short edges between nodes in similar directions from the depot
    heuristic_combined = visibility * angular_factor
    
    # Additional consideration: penalize edges that cross long distances significantly
    # relative to the average distance
    avg_dist = np.mean(distance_matrix[distance_matrix > 0])
    if avg_dist > 0:
        dist_ratio = distance_matrix / avg_dist
        # Exponential decay for long edges
        # Reverted coefficient from 1.2 to 1.0 to align with Reference Program Step 6
        dist_penalty = np.exp(-dist_ratio * 1.0)
        heuristic_combined = heuristic_combined * dist_penalty
    
    # Ensure non-negative values
    heuristic_combined = np.maximum(heuristic_combined, 0.0)
    
    # Apply depot-specific bias
    
    # Edges leaving depot: row 0, columns j (j > 0)
    # Use a linear factor to ensure medium-demand nodes are not suppressed
    if n > 1:
        demand_ratio = demands[1:] / capacity
        # Linear factor: 1.0 base + 0.5 scaling based on demand ratio
        depot_bias_out = 1.0 + 0.5 * demand_ratio
        heuristic_combined[0, 1:] *= depot_bias_out
        
        # Edges entering depot: rows j (j > 0), column 0
        # Only favor returning to depot if the customer's demand is large enough to justify a route end
        # Threshold set to 0.5 to align with reference program and historical Step 4
        threshold = 0.5
        depot_bias_in = np.where(demand_ratio > threshold, demand_ratio, 0.0)
        heuristic_combined[1:, 0] *= depot_bias_in

    # Zero out diagonal explicitly to ensure no self-loops
    heuristic_combined[np.arange(n), np.arange(n)] = 0.0
    
    # Ensure non-negative
    heuristic_combined = np.maximum(heuristic_combined, 0.0)
    
    # Capacity-aware feasibility mask removed as per instructions.
    # Feasibility is handled by the ACO construction phase.

    # Augment with local deflection penalty for customer-to-customer edges
    # Compute angle between vector (i - depot) and (j - i)
    # Vector v1 = i - depot, Vector v2 = j - i
    # cos(theta) = (v1 . v2) / (|v1| * |v2|)
    if n > 1:
        # Get coordinates for customers (1 to n-1)
        cust_coords = coordinates[1:]
        depot_coords = coordinates[0]
        
        # Vectors from depot to customers: (n-1, 2)
        v_depot_to_cust = cust_coords - depot_coords
        
        # Vectors from customer i to customer j: (n-1, n-1, 2)
        # v_ij = coord_j - coord_i
        v_ij = cust_coords[np.newaxis, :, :] - cust_coords[:, np.newaxis, :]
        
        # Dot product v_depot_to_i . v_ij
        # Shape: (n-1, n-1)
        dot_products = np.sum(v_depot_to_cust[:, np.newaxis, :] * v_ij, axis=2)
        
        # Norms
        # Norm of v_depot_to_i: (n-1,)
        norm_depot_to_cust = np.linalg.norm(v_depot_to_cust, axis=1)
        # Norm of v_ij: (n-1, n-1)
        norm_ij = np.linalg.norm(v_ij, axis=2)
        
        # Avoid division by zero
        norm_depot_to_cust_expanded = norm_depot_to_cust[:, np.newaxis]
        denominator = norm_depot_to_cust_expanded * norm_ij
        denominator = np.where(denominator < epsilon, epsilon, denominator)
        
        cos_angles = dot_products / denominator
        # Clip to [-1, 1] for numerical stability
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        
        # Calculate local distance ratio for customer-to-customer edges
        # dist_ratio_local is the edge length normalized by the average distance
        if avg_dist > 0:
            dist_matrix_cust = distance_matrix[1:, 1:]
            dist_ratio_local = dist_matrix_cust / avg_dist
        else:
            dist_ratio_local = np.zeros((n-1, n-1))
            
        # Length-scaled deflection factor
        # Base deflection: 1.0 + 0.95 * cos_angles (Increased from 0.8 to 0.95)
        # Length scaling: 1.0 + 0.2 * (1.0 - dist_ratio_local)
        # This reinforces straightness specifically on short, high-quality edges
        deflection_base = 1.0 + 0.95 * cos_angles
        length_scale = 1.0 + 0.2 * (1.0 - dist_ratio_local)
        deflection_factor = deflection_base * length_scale
        
        # Introduce capacity slack scaling factor
        # This gently penalizes edges that connect nodes with high combined demand
        # relative to vehicle capacity, providing soft capacity awareness.
        # Shape of demands[1:] is (n-1,)
        # demands[1:, np.newaxis] is (n-1, 1)
        # demands[np.newaxis, 1:] is (1, n-1) -> broadcasts to (n-1, n-1)
        # Note: demands[1:] refers to customer demands. 
        # We need to align indices carefully. 
        # heuristic_combined[1:, 1:] corresponds to customers 1..n-1.
        # demands[1:] are demands for customers 1..n-1.
        
        # Create demand matrix for customers
        cust_demands = demands[1:] # Shape (n-1,)
        # Sum of demands for edge (i, j)
        # i corresponds to row index in cust_demands
        # j corresponds to col index in cust_demands
        demand_sum = cust_demands[:, np.newaxis] + cust_demands[np.newaxis, :]
        
        # Capacity slack factor: capacity / (demand_i + demand_j)
        # Clipped to [0.0, 2.0] to prevent extreme values
        capacity_slack = np.clip(capacity / (demand_sum + epsilon), 0.0, 2.0)
        
        # Multiply deflection factor by capacity slack
        deflection_factor *= capacity_slack
        
        # Apply only to customer-to-customer edges
        heuristic_combined[1:, 1:] *= deflection_factor

    # Normalize the heuristic matrix to have a reasonable range
    max_val = np.max(heuristic_combined)
    if max_val > 0:
        heuristic_combined = heuristic_combined / max_val
    
    return heuristic_combined
