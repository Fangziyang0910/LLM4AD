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
    epsilon = 1e-9
    
    # 1. Base Heuristic: Inverse distance
    # Use soft inverse to avoid division by zero and extreme values
    inv_dist = 1.0 / (distance_matrix + epsilon)
    
    # 2. Capacity-Aware Urgency
    # Urgency based on destination demand relative to capacity.
    # Higher demand customers are more "urgent" to place correctly.
    demand_j = demands.reshape(1, -1) # (1, n)
    urgency = demand_j / capacity     # (1, n)
    
    # 3. Enhanced Sector Continuity Factor
    # Compute angles of all nodes relative to the depot (node 0)
    depot_coord = coordinates[0]
    
    # Vector from depot to all nodes
    diffs = coordinates - depot_coord # (n, 2)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0]) # (n,)
    
    # Compute angular difference matrix
    # angle_i is (n, 1), angle_j is (1, n)
    angle_i = angles.reshape(n, 1)
    angle_j = angles.reshape(1, n)
    
    # Absolute difference
    diff = np.abs(angle_i - angle_j)
    
    # Wrap around 2*pi: if difference > pi, use 2*pi - diff
    # This gives the shortest angular distance on the circle
    angular_diff = np.where(diff > np.pi, 2 * np.pi - diff, diff)
    
    # Distances from depot for all nodes
    # dist_i: (n, 1) for source nodes
    # dist_j: (1, n) for destination nodes
    dist_from_depot = np.linalg.norm(diffs, axis=1) # (n,)
    dist_i = dist_from_depot.reshape(n, 1) # (n, 1)
    dist_j = dist_from_depot.reshape(1, n) # (1, n)
    
    # Compute harmonic mean of distances
    # H(i, j) = 2 * d_i * d_j / (d_i + d_j)
    # Add epsilon to denominator to avoid division by zero (e.g., depot to depot)
    sum_dist = dist_i + dist_j + epsilon
    product_dist = dist_i * dist_j
    harmonic_mean_dist = 2.0 * product_dist / sum_dist
    
    # Scale angular penalty by the harmonic mean of distances
    # Base angular penalty coefficient
    alpha_angle_base = 2.0
    
    # Scaling factor constant to normalize the effect of distance
    scale_constant = 50.0 
    
    # Effective angular difference penalty scaled by distance
    # Higher harmonic mean means at least one node is far, increasing penalty for angular deviation
    # If both are close (small harmonic mean), penalty is reduced, allowing wider turns.
    scaled_angular_diff = angular_diff * (1.0 + harmonic_mean_dist / scale_constant)
    
    # Sector continuity factor: higher value for smaller scaled angular difference
    sector_continuity = np.exp(-alpha_angle_base * scaled_angular_diff)
    
    # 4. Radial Progression Bonus for Depot-to-Customer Edges (Row 0)
    # Encourages starting routes to customers in sparse angular sectors.
    # Calculate angular density for each customer (node 1 to n-1)
    # Exclude depot (node 0) from density calculation as it's the center
    
    customer_angles = angles[1:] # (n-1,)
    n_customers = n - 1
    
    # Define angular window for density calculation
    angular_window = np.pi / 4.0 # 45 degrees
    
    # Compute pairwise angular differences between customers
    # Shape: (n-1, n-1)
    diff_cust = np.abs(customer_angles[:, np.newaxis] - customer_angles[np.newaxis, :])
    diff_cust = np.where(diff_cust > np.pi, 2 * np.pi - diff_cust, diff_cust)
    
    # Count neighbors within angular window for each customer
    # A neighbor is any other customer within the angular window
    # We want low density to be good, so we count how many are CLOSE
    density_mask = diff_cust < angular_window
    # Subtract self (diagonal is 0, so it's included in mask, set to 0)
    np.fill_diagonal(density_mask, False)
    density_counts = np.sum(density_mask, axis=1) # (n-1,)
    
    # Max possible density for normalization
    max_density = n_customers - 1 if n_customers > 1 else 1
    
    # Sparse score: 1.0 is most sparse, 0.0 is most dense
    # Normalize counts to [0, 1]
    norm_density = density_counts / max_density
    sparse_score = 1.0 - norm_density
    
    # Radial bonus: increase score if sparse AND far from depot
    # dist_from_depot[1:] corresponds to customers
    customer_dists = dist_from_depot[1:]
    
    # Bonus term: scale by distance, modulated by sparsity
    # If sparse_score is high, the bonus is larger.
    # We add this bonus multiplicatively to the heuristic for row 0.
    # Let's define a radial_boost factor for each customer j.
    radial_boost = 1.0 + 2.0 * sparse_score * (customer_dists / scale_constant)
    
    # Initialize full matrix with 1s (no bonus for non-depot rows)
    radial_factor = np.ones((n, n))
    
    # Apply to row 0, columns 1 to n-1
    radial_factor[0, 1:] = radial_boost
    
    # 5. Residual Capacity Fit for Customer-to-Customer Edges
    # Estimate remaining capacity as (capacity - demand_i)
    # Favor edges where demand_j fits comfortably in residual capacity.
    
    # demand_i: (n, 1)
    demand_i = demands.reshape(n, 1)
    
    # Residual capacity after visiting node i
    # If demand_i is 0 (depot), residual is capacity.
    residual_cap = capacity - demand_i # (n, 1)
    
    # We want to avoid routes where demand_j > residual_cap.
    # We want to favor cases where demand_j is small relative to residual_cap.
    # Ratio: demand_j / residual_cap
    # If ratio is low, it's a good fit (lots of space left).
    # If ratio is high (close to 1), it's a tight fit.
    # If ratio > 1, it's invalid (should be strongly penalized).
    
    # Calculate ratio. Add epsilon to residual_cap to avoid div by zero if capacity=demand_i
    # Note: If demand_i == capacity, residual is 0. This node cannot go anywhere.
    ratio = demand_j / (residual_cap + epsilon) # (n, n)
    
    # Fit factor: Higher is better for smaller ratios.
    # Using an exponential decay or inverse function.
    # Let's use: 1 / (1 + ratio). 
    # If ratio=0, factor=1. If ratio=1, factor=0.5. If ratio=2, factor=0.33.
    # This gently penalizes tight fits and invalid fits without zeroing them out completely
    # (allowing pheromones to potentially override if necessary, though invalid edges 
    # should ideally be masked in ACO construction, here we just downweight).
    fit_factor = 1.0 / (1.0 + ratio)
    
    # Scale the boost slightly to make it more significant
    # 1.0 + alpha * (fit_factor - 0.5) ? 
    # Or just multiply by a factor that emphasizes the fit.
    # Let's stick to a multiplicative factor.
    # To make it stronger, we can exponentiate the fit_factor.
    demand_factor = np.ones((n, n))
    
    # Apply only to customer-to-customer edges
    # Create a mask for i != 0 and j != 0
    mask_cust = np.zeros((n, n), dtype=bool)
    mask_cust[1:, 1:] = True
    
    # Clip fit_factor to be at least epsilon to avoid zeroing out completely if needed,
    # but mathematically it won't be zero.
    demand_factor[mask_cust] = fit_factor[mask_cust]
    
    # 6. Enhanced Turn-Angle Smoothness Bonus with Directional Momentum
    # Favor edges where the angle formed by Depot-i-j is close to 180 degrees.
    # New: Incorporate "Directional Momentum" scaling the smoothness factor by the 
    # cosine similarity between the radial vector (Depot -> i) and the current edge vector (i -> j).
    
    # Initialize smoothness matrix to 1.0 (neutral)
    smoothness = np.ones_like(distance_matrix)
    
    if n > 1:
        # Coordinates for all nodes
        # We need vectors for each pair (i, j)
        # i is source (rows), j is destination (cols)
        
        # Get coordinates for source nodes i (n, 1, 2)
        coord_i = coordinates[:, np.newaxis, :] # (n, 1, 2)
        # Get coordinates for destination nodes j (1, n, 2)
        coord_j = coordinates[np.newaxis, :, :] # (1, n, 2)
        
        # Vector u = Depot - i (Radial vector reversed, points from i to Depot)
        # Or better, use vector R_i = i - Depot (points from Depot to i).
        # Let's define Momentum Direction as the radial vector from Depot to i.
        # R_i = coord_i - depot_coord
        R_i = coord_i - depot_coord[np.newaxis, np.newaxis, :] # (n, 1, 2)
        
        # Vector v = j - i (Edge vector)
        v = coord_j - coord_i # (n, n, 2)
        
        # Compute norms
        norm_R = np.linalg.norm(R_i, axis=2, keepdims=True) # (n, 1, 1)
        norm_v = np.linalg.norm(v, axis=2, keepdims=True) # (n, n, 1)
        
        # Normalize vectors
        # Add epsilon to norms to avoid division by zero
        R_i_normed = R_i / (norm_R + epsilon) # (n, 1, 2)
        v_normed = v / (norm_v + epsilon) # (n, n, 2)
        
        # Compute cosine similarity between Radial Vector (Depot->i) and Edge Vector (i->j)
        # dot_product = sum(R_i_normed * v_normed, axis=2)
        # Shape: (n, n)
        momentum_cosine = np.sum(R_i_normed * v_normed, axis=2) # (n, n)
        
        # Momentum Factor: 
        # We want to encourage movement that continues in the general radial direction.
        # If cos ~ 1, we are moving away from depot in same sector.
        # If cos ~ -1, we are moving back towards depot.
        # If cos ~ 0, we are moving perpendicular.
        # Boost edges with positive cosine similarity (outward spiral/continuation).
        # Clip to [0, 1] range for weighting? Or use directly?
        # Let's use: 1.0 + alpha * max(0, cosine). This gives neutral (1.0) for negative/zero, 
        # and boosts positive cosine.
        momentum_factor = 1.0 + 2.0 * np.maximum(0.0, momentum_cosine)
        
        # Geometric Straightness Component
        # Compute angle at i formed by Depot-i-j.
        # Vector w = Depot - i = -R_i
        w_normed = -R_i_normed
        
        # Cosine of angle between w (Depot->i reversed) and v (i->j)
        # cos_theta = dot(-R_i, v) / (|R_i| * |v|)
        cos_angle = np.sum(w_normed * v_normed, axis=2) # (n, n)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angle_at_i = np.arccos(cos_angle) # (n, n)
        
        # We want angle close to pi (180 degrees) for straight line continuation from depot perspective?
        # Actually, if we want a "straight" route relative to the incoming path, we need the previous edge.
        # Since we don't have previous edge, we assume "ideal" path is tangential or radial?
        # The previous code used deviation from pi relative to the vector from i to Depot.
        # Let's stick to: Deviation from 180 degrees between vector (Depot->i) and (i->j) is not quite right.
        # Vector Depot->i is R_i. Vector i->j is v.
        # If we continue "straight" past i, we want v to be aligned with R_i (cos=1).
        # The previous code calculated angle between (Depot-i) and (j-i).
        # Depot-i is -R_i. j-i is v.
        # Angle between -R_i and v. If they are aligned (180 deg total), cos is -1.
        # So cos_angle == -1 means straight line continuation of the radial vector? No.
        # If Depot, i, j are collinear and i is between Depot and j:
        # R_i points Depot->i. v points i->j. They are same direction. Cosine = 1.
        # The angle between -R_i (i->Depot) and v (i->j) is 180 degrees.
        # So cos(angle) = -1.
        # Let's calculate deviation from 180 degrees (pi) for the angle between vector (i->Depot) and (i->j).
        
        # angle_between_u_v where u=Depot-i, v=j-i
        # cos_val = dot(u, v) / (|u||v|)
        # We calculated cos_angle above as dot(-R_i_normed, v_normed).
        # This is the cosine of the angle between vector from i to Depot and vector from i to j.
        # We want this angle to be close to PI (180) for a "straight" line extending away from depot?
        # No, if angle is PI, then i->Depot and i->j are opposite. So j is in direction opposite to Depot.
        # This means j is further out radially. This is good.
        
        deviation = np.abs(angle_at_i - np.pi)
        
        # Base smoothness factor inversely proportional to deviation
        smooth_scale = 2.0 
        base_smoothness_values = 1.0 / (1.0 + smooth_scale * deviation)
        
        # Scale this bonus by the inverse distance between i and j
        # short_dist_inv = 1 / (dist(i,j) + epsilon)
        # This makes the smoothness constraint stronger for nearby nodes
        short_dist_inv = 1.0 / (distance_matrix + epsilon)
        
        # Capacity-Aware Scaling Term
        # Estimate remaining capacity as (capacity - destination_demand).
        # Multiply smoothness bonus by ratio: remaining_capacity / destination_demand.
        
        remaining_cap_est = capacity - demand_j # (1, n)
        
        # Avoid division by zero if demand is 0
        cap_ratio = remaining_cap_est / (demand_j + epsilon) # (1, n)
        
        dist_scaling_factor = 1.0 + 5.0 * short_dist_inv
        
        # Combine base smoothness, distance scaling, capacity scaling, and NEW Momentum Factor
        # Momentum factor reinforces the directionality
        smoothness_values = base_smoothness_values * dist_scaling_factor * cap_ratio * momentum_factor
        
        # Apply smoothness only for customer-to-customer edges
        # Create a mask for i != 0 and j != 0
        mask = np.zeros((n, n), dtype=bool)
        mask[1:, 1:] = True
        
        smoothness = np.where(mask, smoothness_values, 1.0)
        
    # 7. Combine components
    # Base heuristic: inv_dist * urgency * sector_continuity
    # Then multiply by radial_factor (for depot edges), demand_factor (for customer edges), and smoothness (for customer edges)
    
    heuristic_matrix = inv_dist * urgency * sector_continuity * radial_factor * demand_factor * smoothness
    
    # Ensure positivity
    heuristic_matrix = np.maximum(heuristic_matrix, epsilon)
    
    return heuristic_matrix
