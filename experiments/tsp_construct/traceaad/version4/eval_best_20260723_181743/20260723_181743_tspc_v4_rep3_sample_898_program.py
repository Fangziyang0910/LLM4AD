import numpy as np

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    if len(unvisited_nodes) == 0:
        return destination_node
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    n_candidates = len(unvisited_nodes)
    
    # Get distances from current node to all unvisited nodes
    dist_from_current = np.array([distance_matrix[current_node, node] for node in unvisited_nodes])
    
    # Get distances from destination node to all unvisited nodes
    dist_from_dest = np.array([distance_matrix[destination_node, node] for node in unvisited_nodes])
    
    # Handle case where distances are zero or very small
    epsilon = 1e-10
    
    # --- Vectorized Soft-min Kernel Repulsion Score ---
    # Extract the submatrix of pairwise distances among unvisited nodes
    pairwise_dist_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Estimate sigma using median of upper triangle of pairwise distances
    mask = np.triu(np.ones_like(pairwise_dist_matrix, dtype=bool), k=1)
    upper_triangular_dists = pairwise_dist_matrix[mask]
    
    if len(upper_triangular_dists) > 0:
        median_dist = np.median(upper_triangular_dists)
        if median_dist < epsilon:
            median_dist = epsilon
        sigma = median_dist 
    else:
        sigma = 1.0

    # Calculate Gaussian Kernel Values: K(i, j) = exp(-d_ij^2 / (2*sigma^2))
    dist_sq = pairwise_dist_matrix ** 2
    exp_term = -dist_sq / (2.0 * sigma ** 2)
    kernel_matrix = np.exp(exp_term)
    
    # Sum along axis 1 to get closeness score for each node
    # Subtract 1.0 from diagonal sum because K(i,i) = 1.0, but we don't want self-interaction
    closeness_scores = np.sum(kernel_matrix, axis=1) - 1.0
    
    # We want to PREFER isolated nodes (low closeness).
    max_closeness = np.max(closeness_scores)
    min_closeness = np.min(closeness_scores)
    
    if max_closeness > min_closeness:
        # Normalize to [0, 1] then invert
        norm_closeness = (closeness_scores - min_closeness) / (max_closeness - min_closeness)
        norm_repulsion = 1.0 - norm_closeness
    else:
        norm_repulsion = np.ones(n_candidates)
        
    # Calculate Future Cost Estimation using Greedy Nearest Neighbor Approximation
    future_costs = np.zeros(n_candidates)
    
    for i in range(n_candidates):
        candidate_i = unvisited_nodes[i]
        
        # 1. Distance from current to candidate
        cost_to_candidate = dist_from_current[i]
        
        # 2. Remaining nodes if we pick candidate_i
        remaining_indices = [idx for idx in range(n_candidates) if idx != i]
        remaining_nodes = unvisited_nodes[remaining_indices]
        
        if len(remaining_nodes) == 0:
            # If no nodes left, just need to return to destination
            future_costs[i] = cost_to_candidate + distance_matrix[candidate_i, destination_node]
        else:
            # Estimate cost of completing the tour for remaining nodes using Greedy NN
            nn_cost = 0.0
            current_nn = candidate_i
            remaining_set = set(remaining_nodes)
            
            # Build the NN path through remaining nodes
            while remaining_set:
                dists_to_remaining = [distance_matrix[current_nn, node] for node in remaining_set]
                min_idx = np.argmin(dists_to_remaining)
                next_node = list(remaining_set)[min_idx]
                min_dist = dists_to_remaining[min_idx]
                
                nn_cost += min_dist
                remaining_set.remove(next_node)
                current_nn = next_node
            
            # Add distance from last visited node to destination
            nn_cost += distance_matrix[current_nn, destination_node]
            
            future_costs[i] = cost_to_candidate + nn_cost

    # Normalize scores to [0, 1] range for combination
    
    # 1. Attraction scores (inverse distance from current)
    attraction_scores = np.array([1.0 / (d + epsilon) for d in dist_from_current])
    max_attraction = np.max(attraction_scores)
    min_attraction = np.min(attraction_scores)
    if max_attraction > min_attraction:
        norm_attraction = (attraction_scores - min_attraction) / (max_attraction - min_attraction)
    else:
        norm_attraction = np.ones(n_candidates)
    
    # 2. Future Cost scores (inverse so low cost = high score)
    min_future_val = np.min(future_costs)
    max_future_val = np.max(future_costs)
    range_future = max_future_val - min_future_val
    
    if range_future > epsilon:
        norm_future = 1.0 - (future_costs - min_future_val) / range_future
    else:
        norm_future = np.ones(n_candidates)
        
    # 3. Regret scores
    if range_future > epsilon:
        norm_regret = (min_future_val - future_costs) / range_future
    else:
        norm_regret = np.ones(n_candidates)

    # --- Refined Local Cycle Penalty (Including Destination) ---
    # Penalize candidates that form tight triangles with Current and Destination.
    # This avoids picking a node that is very close to both Current and Destination,
    # which might isolate other nodes or create a suboptimal geometric closure.
    
    local_cycle_penalty = np.zeros(n_candidates)
    
    dist_current_dest = distance_matrix[current_node, destination_node]
    
    for i in range(n_candidates):
        d_c = dist_from_current[i]
        d_d = dist_from_dest[i]
        
        # Perimeter of triangle Current -> Candidate -> Destination
        perimeter = d_c + d_d + dist_current_dest
        
        # We penalize if the triangle is "flat" or "tight" in a way that suggests
        # the candidate is on the direct path or very close to the edge between Current and Dest.
        # A simple metric: if d_c + d_d is close to dist_current_dest, the node is collinear/close to edge.
        # However, we specifically want to penalize tight clusters where d_c and d_d are both small relative to the global scale.
        
        # Let's define a penalty based on how small d_c and d_d are relative to sigma.
        norm_d_c = d_c / (sigma + epsilon)
        norm_d_d = d_d / (sigma + epsilon)
        
        # High penalty if both distances are small (node is in the middle of Current and Dest close up)
        # This prevents "short-cutting" the tour structure if the destination is far away in terms of topology but close in geometry.
        
        closure_tendency = np.exp(-norm_d_c) * np.exp(-norm_d_d)
        
        # If the candidate is very close to the direct line between Current and Dest, it might be a bad choice if it doesn't help connect to other nodes.
        # Check collinearity factor using Heron's formula for area or cosine rule.
        # Cosine of angle at Current: cos(theta) = (d_c^2 + dist_current_dest^2 - d_d^2) / (2 * d_c * dist_current_dest)
        # If theta is close to 0, node is in direction of dest.
        
        if d_c > epsilon and dist_current_dest > epsilon:
            try:
                cos_theta = (d_c**2 + dist_current_dest**2 - d_d**2) / (2.0 * d_c * dist_current_dest)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angle_factor = np.abs(np.arccos(cos_theta)) # 0 if collinear same direction, pi if opposite
                
                # Penalize less if angle is large (node goes away from dest), penalize more if angle is small (node goes towards dest closely)
                # We want to avoid going directly to dest prematurely if there are other nodes.
                # But we also want to avoid tight triangles.
                
                local_cycle_penalty[i] = closure_tendency * (1.0 - (angle_factor / np.pi))
            except Exception:
                local_cycle_penalty[i] = closure_tendency
        else:
            local_cycle_penalty[i] = closure_tendency

    # Normalize penalty to [0, 1]
    if np.max(local_cycle_penalty) > epsilon:
        norm_penalty = local_cycle_penalty / np.max(local_cycle_penalty)
    else:
        norm_penalty = np.zeros(n_candidates)

    # Dynamic alpha: Exponential transition based on fraction of nodes remaining
    total_nodes = distance_matrix.shape[0]
    fraction_remaining = n_candidates / total_nodes
    fraction_remaining = np.clip(fraction_remaining, 0.0, 1.0)
    
    alpha_min = 0.5
    alpha_max = 1.0
    k = 5.0 
    
    alpha = alpha_min + (alpha_max - alpha_min) * np.exp(-k * (1.0 - fraction_remaining))
    alpha = np.clip(alpha, alpha_min, alpha_max)
    
    # --- Convex Hull Area Metric for Geometric Spread ---
    
    spread_score = 0.0
    
    if n_candidates >= 3:
        try:
            # Calculate angles relative to the first unvisited node as reference
            ref_node = unvisited_nodes[0]
            dist_to_ref = dist_from_current[0]
            
            angles = np.zeros(n_candidates)
            if dist_to_ref > epsilon:
                for i in range(n_candidates):
                    dist_to_cand_i = dist_from_current[i]
                    dist_between_ref_and_i = distance_matrix[ref_node, unvisited_nodes[i]]
                    
                    denom = 2.0 * dist_to_ref * dist_to_cand_i
                    if denom > epsilon:
                        cos_theta = (dist_to_ref**2 + dist_to_cand_i**2 - dist_between_ref_and_i**2) / denom
                        cos_theta = np.clip(cos_theta, -1.0, 1.0)
                        angles[i] = np.arccos(cos_theta)
                    else:
                        angles[i] = 0.0
            
            # The "spread" is the range of these angles.
            angle_range = np.max(angles) - np.min(angles)
            normalized_spread = angle_range / np.pi
            
            # Use sigmoid to determine boost
            steepness = 3.0
            bias = 0.5
            sigmoid_input = steepness * (normalized_spread - bias)
            spread_score = 1.0 / (1.0 + np.exp(-sigmoid_input))
            
        except Exception:
            spread_score = 0.0
    else:
        spread_score = 0.0
        
    # Base weights
    base_weight_future = 0.25
    base_weight_regret = 0.25
    base_weight_attr_rep = 0.5
    
    # Boost future cost weight based on spread via sigmoid
    future_cost_multiplier = 1.0 + spread_score
    weight_future_raw = base_weight_future * future_cost_multiplier
    
    # Renormalize total weights
    total_weight_sum = weight_future_raw + base_weight_regret + base_weight_attr_rep
    
    weight_regret = base_weight_regret / total_weight_sum
    weight_attr_rep = base_weight_attr_rep / total_weight_sum
    weight_future = weight_future_raw / total_weight_sum
    
    # Distribute weight_attr_rep between attraction and repulsion based on alpha
    w_attr = weight_attr_rep * alpha
    w_rep = weight_attr_rep * (1 - alpha)
    
    # Integrate Local Cycle Penalty
    # Subtract a portion of the attraction score based on the penalty
    # Penalty weight: stronger when few nodes remain (exploitation phase)
    penalty_weight = 0.2 * (1.0 - fraction_remaining)
    
    adjusted_attraction = norm_attraction - penalty_weight * norm_penalty
    # Ensure attraction doesn't go negative
    adjusted_attraction = np.clip(adjusted_attraction, 0.0, 1.0)
    
    final_scores = w_attr * adjusted_attraction + w_rep * norm_repulsion + weight_future * norm_future + weight_regret * norm_regret
    
    best_idx = np.argmax(final_scores)
    
    return int(unvisited_nodes[best_idx])
