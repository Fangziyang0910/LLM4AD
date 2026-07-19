import numpy as np

def heuristics(distance_matrix: np.ndarray, coordinates: np.ndarray, demands: np.ndarray, capacity: int) -> np.ndarray:
    """Return edge desirability values for CVRP ant colony optimization.

    Args:
        distance_matrix: Pairwise Euclidean distances with shape (n, n).
        coordinates: Node coordinates with shape (n, 2). Node 0 is the depot.
        demands: Node demands with shape (n, n). The depot demand is zero.
        capacity: Capacity shared by all vehicles.

    Returns:
        An (n, n) edge-prior matrix. Larger values make an edge more likely
        to be sampled. Values at or below zero are treated as 1e-9.
    """
    n = distance_matrix.shape[0]
    heur = np.zeros_like(distance_matrix)
    
    # 1. Inverse Distance
    # Avoid division by zero on diagonal
    safe_dist = np.where(distance_matrix == 0, 1e-9, distance_matrix)
    inv_dist = 1.0 / safe_dist
    
    # 2. Radial Momentum / Angular Consistency
    # Coordinates of depot
    depot_coord = coordinates[0:1, :]  # Shape (1, 2)
    
    # Vector from depot to each node i
    vec_depot_to_i = coordinates - depot_coord  # Shape (n, 2)
    
    # Normalize these vectors to get unit directions from depot
    magnitudes_depot_i = np.linalg.norm(vec_depot_to_i, axis=1, keepdims=True)
    # Avoid division by zero for depot itself
    magnitudes_depot_i = np.where(magnitudes_depot_i == 0, 1.0, magnitudes_depot_i)
    unit_depot_to_i = vec_depot_to_i / magnitudes_depot_i  # Shape (n, 2)
    
    # We want to compute the angle between (Depot->i) and (i->j)
    # Vector from i to j
    vec_i_to_j = coordinates[np.newaxis, :, :] - coordinates[:, np.newaxis, :]  # Shape (n, n, 2)
    
    # Unit vector from i to j
    magnitudes_i_j = np.linalg.norm(vec_i_to_j, axis=2, keepdims=True)
    magnitudes_i_j = np.where(magnitudes_i_j == 0, 1.0, magnitudes_i_j)
    unit_i_to_j = vec_i_to_j / magnitudes_i_j  # Shape (n, n, 2)
    
    # Cosine of the angle between unit_depot_to_i and unit_i_to_j
    dot_products = np.sum(unit_depot_to_i[:, np.newaxis, :] * unit_i_to_j, axis=2)  # Shape (n, n)
    
    # Clip to [-1, 1] for numerical stability
    dot_products = np.clip(dot_products, -1.0, 1.0)
    
    # Angular factor: 
    # High when continuing outward/sideways, low when turning back sharply.
    angular_factor = 1.0 + dot_products
    
    # 3. Clarke-Wright Savings Potential Bias (Exponential) with Residual Efficiency
    # Savings(i, j) = dist(0, i) + dist(0, j) - dist(i, j)
    # Ratio: Savings(i, j) / dist(i, j)
    
    # Distance from depot (node 0) to all nodes
    dist_from_depot = distance_matrix[0, :]  # Shape (n,)
    
    # Broadcasting:
    # dist_depot_i shape (n, 1)
    # dist_depot_j shape (1, n)
    dist_depot_i = dist_from_depot[:, np.newaxis]
    dist_depot_j = dist_from_depot[np.newaxis, :]
    
    # Calculate raw savings numerator: dist(0, i) + dist(0, j) - dist(i, j)
    savings_numerator = dist_depot_i + dist_depot_j - distance_matrix
    
    # Calculate ratio: savings_numerator / dist(i, j)
    # Use safe_dist to avoid division by zero
    savings_ratio = savings_numerator / safe_dist
    
    # Feasibility Mask: Edge (i, j) is only feasible if demands[i] + demands[j] <= capacity
    # This applies strictly to customer-to-customer edges.
    demands_i = demands[:, np.newaxis]  # (n, 1)
    demands_j = demands[np.newaxis, :]  # (1, n)
    demands_sum = demands_i + demands_j
    
    feasible_customers = demands_sum <= capacity  # (n, n)
    
    # Define a mask for "customer-to-customer" edges.
    # i != 0 AND j != 0
    mask_customer_edge = (np.arange(n)[:, np.newaxis] != 0) & (np.arange(n)[np.newaxis, :] != 0)
    
    # Define a mask for "customer-to-depot" edges.
    # i != 0 AND j == 0
    mask_return_depot = (np.arange(n)[:, np.newaxis] != 0) & (np.arange(n)[np.newaxis, :] == 0)
    
    # Define a mask for "depot-to-customer" edges.
    # i == 0 AND j != 0
    mask_depot_edge = (np.arange(n)[:, np.newaxis] == 0) & (np.arange(n)[np.newaxis, :] != 0)
    
    # Apply exponential savings bias
    # Bias = exp(alpha * savings_ratio)
    # Alpha scales the strength of the savings preference
    alpha_savings = 0.5
    exp_savings_bias = np.exp(alpha_savings * savings_ratio)
    
    # 4. Residual Savings Efficiency
    # Scale the savings bias by the ratio of remaining capacity at node i to total capacity.
    # Residual Capacity at i: capacity - demands[i]
    # Note: For the depot (i=0), demand is 0, so residual is full capacity.
    # This term encourages using savings-based edges when there is room in the vehicle.
    
    # Remaining capacity ratio for node i: (capacity - demands[i]) / capacity
    # Shape (n, 1)
    residual_cap_ratio = (capacity - demands_i) / capacity
    
    # Apply this scaling to the exponential savings bias
    # This creates a dynamic bias that decreases as the vehicle fills up
    dynamic_savings_bias = exp_savings_bias * residual_cap_ratio
    
    # Initialize bias factor matrix to 1.0
    bias_factor = np.ones_like(distance_matrix)
    
    # Apply dynamic savings bias for feasible customer-to-customer edges
    # We use the dynamic bias directly as the multiplier for these edges
    apply_dynamic_savings_mask = feasible_customers & mask_customer_edge
    
    bias_factor = np.where(apply_dynamic_savings_mask, dynamic_savings_bias, 1.0)
    
    # 5. Combine Factors
    # Heuristic = inv_dist * angular_factor * bias_factor
    heur = inv_dist * angular_factor * bias_factor
    
    # 6. Special handling for Depot and Return edges
    
    # a) Return to depot (j=0)
    # Boost return to depot for high demand nodes to encourage closing routes with heavy loads.
    if n > 1:
        # Normalize demands to [0, 1] range relative to capacity
        norm_demands = demands / capacity
        
        # Return boost: higher demand -> higher boost to return to depot
        # This applies to all i -> 0 edges where i is a customer
        return_boost = 1.0 + norm_demands[:, np.newaxis]
        heur[:, 0] *= return_boost[:, 0]
        
    # b) Leaving depot (i=0)
    # Boost starting routes to high-density customers (high demand, close to depot)
    # This helps form dense clusters early.
    if n > 1:
        # Efficiency for depot edge: demands[j] / dist(depot, j)
        # Avoid division by zero for j=0 (depot to depot), though mask excludes it.
        safe_depot_dist_j = np.where(dist_from_depot == 0, 1.0, dist_from_depot)
        depot_efficiency = demands / safe_depot_dist_j  # Shape (n,)
        
        # We want to apply this boost to row 0 (i=0) for columns j != 0
        # Boost = 1 + gamma * efficiency
        gamma_depot = 1.0
        depot_boost_values = 1.0 + gamma_depot * depot_efficiency
        
        # Apply to the specific cells in heur[0, :]
        # Ensure we don't touch diagonal or infeasible edges (though depot->j is always feasible capacity-wise if demand[j] <= capacity)
        # The mask mask_depot_edge handles i=0, j!=0
        if n > 1:
            # Create a full boost matrix for broadcasting or direct assignment
            full_depot_boost = np.ones((n, n))
            full_depot_boost[0, :] = depot_boost_values
            
            # Apply boost only to valid depot edges
            heur[mask_depot_edge] *= full_depot_boost[mask_depot_edge]

    # 7. Enforce Capacity Constraints on the Heuristic
    # Edges that violate capacity must be strongly discouraged (value <= 0 or very small)
    
    # Infeasible customer-to-customer edges
    infeasible_customers = ~feasible_customers & mask_customer_edge
    
    # Set heuristic to 0 for infeasible customer-to-customer edges
    heur[infeasible_customers] = 0.0
    
    # Ensure diagonal is 0
    np.fill_diagonal(heur, 0.0)
    
    # Ensure non-negative values
    # The contract says values at or below zero are treated as 1e-9 by the ACO sampler.
    # We set infeasible to 0.0 which satisfies this.
    
    return heur
