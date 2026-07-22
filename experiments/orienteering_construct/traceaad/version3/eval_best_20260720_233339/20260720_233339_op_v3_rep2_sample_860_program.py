import numpy as np

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray, prizes: np.ndarray, remaining_budget: float) -> int:
    """
    Design a novel constructive heuristic for the Orienteering Problem.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the route destination node.
    unvisited_nodes: Array of feasible unvisited node IDs. Visiting one of these nodes still leaves enough budget to return to the destination.
    distance_matrix: Pairwise Euclidean distance matrix of all nodes.
    prizes: Prize values of all nodes. The depot prize is 0.
    remaining_budget: Remaining travel budget before selecting the next node.

    Return:
    ID of the next node to visit.
    """
    if len(unvisited_nodes) == 0:
        return destination_node

    # Precompute constants
    epsilon = 1e-9
    
    # Costs from current node to all unvisited nodes
    costs_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Costs from all unvisited nodes to destination
    costs_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Direct cost from current to destination
    direct_cost_to_dest = distance_matrix[current_node, destination_node]
    
    # Prizes of unvisited nodes
    unvisited_prizes = prizes[unvisited_nodes]
    
    # Calculate Reachability Index: cost to node / remaining budget
    reachability_index = costs_from_current / (remaining_budget + epsilon)
    
    # --- Step 1: Cost Matrix Calculation for Lookahead ---
    
    # Get indices of unvisited nodes in the global array
    unvisited_ids = unvisited_nodes
    
    # Construct the sub-matrix of distances between unvisited nodes
    dist_unvisited = distance_matrix[np.ix_(unvisited_ids, unvisited_ids)]
    
    # Broadcast costs_from_current to column vector (K, 1)
    c_from = costs_from_current[:, np.newaxis]
    # Broadcast costs_to_dest to row vector (1, K)
    c_to = costs_to_dest[np.newaxis, :]
    
    # Total cost matrix (K, K) for path: current -> i -> j -> dest
    total_cost_matrix = c_from + dist_unvisited + c_to
    
    # --- Step 2: Smooth Budget Slack Probability ---
    
    # Calculate remaining budget for each path (current -> i -> j -> dest)
    slack = remaining_budget - total_cost_matrix
    
    # Calculate average prize density of the remaining feasible nodes to scale the sigmoid
    avg_dist_to_dest = np.mean(costs_to_dest)
    avg_prize_density = np.mean(unvisited_prizes) / (avg_dist_to_dest + epsilon)
    
    # Define a steepness parameter. 
    feasible_count = np.sum(total_cost_matrix <= remaining_budget)
    total_pairs = total_cost_matrix.size
    feasibility_ratio = feasible_count / total_pairs if total_pairs > 0 else 0.0
    
    # If very few feasible pairs, soften the penalty to allow near-feasible ones
    if feasibility_ratio < 0.1:
        alpha = 5.0
    elif feasibility_ratio < 0.5:
        alpha = 10.0
    else:
        alpha = 20.0
        
    # Smooth probability function: 1 / (1 + exp(-alpha * slack))
    slack_clipped = np.clip(slack, -500, 500)
    budget_slack_prob = 1.0 / (1.0 + np.exp(-alpha * slack_clipped))
    
    # --- Step 3: Base Score Calculation (Log-Scaled Detour Penalty) ---
    
    # Sum of prizes weighted by budget slack probability for each row i.
    horizon_prize_sum = np.sum(budget_slack_prob * unvisited_prizes[np.newaxis, :], axis=1)
    
    # Dynamic Detour Penalty for the first step (current -> i)
    round_trip_costs = costs_from_current + costs_to_dest
    detour_costs_first_step = round_trip_costs - direct_cost_to_dest
    
    # Residual Budget after visiting candidate i
    residual_budget = remaining_budget - costs_from_current
    res_budget_denom = residual_budget + epsilon
    
    # Residual Prize Density for the first step
    residual_prize_density_first = horizon_prize_sum / res_budget_denom
    res_prize_denom_first = residual_prize_density_first + epsilon
    
    # Effective Detour Cost for first step
    effective_detour_first = detour_costs_first_step / res_prize_denom_first
    
    # Log-Scaled Penalty for first step
    log_scaled_penalty_first = 1.0 / (1.0 + np.log1p(effective_detour_first))
    
    # Base Score Components
    cost_denom = costs_from_current + epsilon
    numerator = unvisited_prizes * horizon_prize_sum
    denominator = cost_denom * (1 + reachability_index**2)
    base_scores = numerator / denominator
    
    # Penalized Base Score
    penalized_base_scores = base_scores * log_scaled_penalty_first
    
    # --- Step 4: Top-K Weighted Lookahead Synergy ---
    
    # Create a mask for j != i to prevent self-loop in lookahead
    identity_matrix = np.eye(budget_slack_prob.shape[0], dtype=bool)
    synergy_mask = ~identity_matrix 

    # Calculate the cost of the segment i -> j -> dest
    segment_cost_ij_dest = dist_unvisited + costs_to_dest[np.newaxis, :]
    
    # Calculate the direct cost from i to dest for comparison
    direct_cost_i_dest = costs_to_dest[:, np.newaxis]
    
    # Second-Order Detour Cost: Extra distance incurred by going i->j->dest vs i->dest
    second_order_detour = np.maximum(segment_cost_ij_dest - direct_cost_i_dest, 0.0)
    
    # Scale the detour by the residual budget after visiting i
    scaled_detour_penalty = second_order_detour / (residual_budget[:, np.newaxis] + epsilon)
    
    # Prize Density of node j relative to the segment cost i->j->dest
    synergy_denom_safe = segment_cost_ij_dest + epsilon
    prize_density_j = unvisited_prizes[np.newaxis, :] / synergy_denom_safe
    
    # Augmented Synergy Score for pair (i, j)
    augmented_synergy_matrix = prize_density_j * budget_slack_prob * (1.0 / (1.0 + scaled_detour_penalty))
    
    # Apply mask: only consider j != i
    masked_augmented_synergy = np.where(synergy_mask, augmented_synergy_matrix, 0.0)
    
    # --- Step 5: Directional Consistency Bonus (Local Coordinate Embedding + Angular Entropy) ---
    
    # Identify reference nodes for local embedding: Current, Destination, and Top-3 high-value successors
    ref_nodes = [current_node, destination_node]
    
    # Get top 3 high prize unvisited nodes to define the local frame
    sorted_unvisited_by_prize = unvisited_ids[np.argsort(-unvisited_prizes)]
    top_k_prize_count = min(3, len(unvisited_ids))
    top_prize_nodes = sorted_unvisited_by_prize[:top_k_prize_count]
    
    # Ensure unique reference nodes
    unique_refs = list(dict.fromkeys(ref_nodes + list(top_prize_nodes)))
    num_refs = len(unique_refs)
    
    directional_consistency_bonus = np.zeros(len(unvisited_nodes))
    
    if num_refs >= 3:
        # Build local distance matrix for references
        global_ref_ids = unique_refs
        ref_dist_sq = distance_matrix[np.ix_(global_ref_ids, global_ref_ids)] ** 2
        
        # Find index of current_node and destination_node in unique_refs
        idx_current_in_ref = global_ref_ids.index(current_node)
        idx_dest_in_ref = global_ref_ids.index(destination_node)
        
        # Trilateration / Simple Projection to 2D
        # We set current_node as origin (0,0) and destination_node on X-axis (L, 0)
        r1_id = global_ref_ids[idx_current_in_ref]
        r2_id = global_ref_ids[idx_dest_in_ref]
        
        L = distance_matrix[r1_id, r2_id] # Distance between current and dest
        
        if L < epsilon:
            # Fallback if current and dest are same (shouldn't happen usually)
            pass
        else:
            # Find a third reference not collinear to define Y axis
            r3_id = None
            for i, rid in enumerate(global_ref_ids):
                if i != idx_current_in_ref and i != idx_dest_in_ref:
                    r3_id = rid
                    break
            
            if r3_id is not None:
                # Coordinates of refs in local frame:
                # r1: (0, 0)
                # r2: (L, 0)
                # r3: (x3, y3)
                
                d_sq_r3_r1 = distance_matrix[r3_id, r1_id]**2
                d_sq_r3_r2 = distance_matrix[r3_id, r2_id]**2
                
                x3 = (d_sq_r3_r1 + L**2 - d_sq_r3_r2) / (2 * L)
                y3_sq = d_sq_r3_r1 - x3**2
                y3_sq = max(y3_sq, 0)
                y3 = np.sqrt(y3_sq)
                
                # Now project all unvisited nodes
                d_sq_to_r1 = distance_matrix[unvisited_ids, r1_id]**2
                d_sq_to_r2 = distance_matrix[unvisited_ids, r2_id]**2
                d_sq_to_r3 = distance_matrix[unvisited_ids, r3_id]**2
                
                # Calculate x coords for all unvisited nodes
                x_coords = (d_sq_to_r1 + L**2 - d_sq_to_r2) / (2 * L)
                
                # Calculate y^2
                y_sq = d_sq_to_r1 - x_coords**2
                y_sq = np.maximum(y_sq, 0)
                
                # Determine sign of y using r3 to resolve ambiguity
                # We use the distance to r3 to check which side of the line r1-r2 the point is on
                # Error if y is positive vs negative
                
                y_pos = np.sqrt(y_sq)
                y_neg = -y_pos
                
                # Distance squared from (x, y) to (x3, y3)
                err_pos = (x_coords - x3)**2 + (y_pos - y3)**2 - d_sq_to_r3
                err_neg = (x_coords - x3)**2 + (y_neg - y3)**2 - d_sq_to_r3
                
                # Choose sign with smaller error
                y_coords = np.where(np.abs(err_pos) < np.abs(err_neg), y_pos, y_neg)
                
                local_coords_unvisited = np.column_stack((x_coords, y_coords))
                
                # Calculate directional consistency for each candidate i
                # We check if the top-K successors of i are in the same general direction as i->dest
                
                K_look = min(10, len(unvisited_nodes) - 1)
                actual_K = min(K_look, len(unvisited_nodes) - 1)
                
                if actual_K > 0:
                    negated_synergy = -masked_augmented_synergy
                    # Get top K indices for each row
                    top_k_indices = np.argpartition(negated_synergy, actual_K, axis=1)[:, :actual_K]
                    
                    # Precompute vector from current to dest in local frame
                    # In our frame, current is (0,0) and dest is (L, 0).
                    # So the target direction vector is [1, 0]
                    vec_target_norm = np.array([1.0, 0.0])
                    
                    for i in range(len(unvisited_nodes)):
                        indices_i = top_k_indices[i]
                        top_k_synergies = masked_augmented_synergy[i, indices_i]
                        
                        if np.sum(top_k_synergies) < epsilon:
                            directional_consistency_bonus[i] = 0.0
                            continue
                        
                        # Get local coords of candidate i and its top K successors
                        coord_i = local_coords_unvisited[i]
                        successor_coords = local_coords_unvisited[indices_i]
                        
                        # Vectors from i to successors
                        vec_i_to_successors = successor_coords - coord_i
                        
                        # Normalize vectors
                        norms = np.linalg.norm(vec_i_to_successors, axis=1, keepdims=True)
                        norms = np.maximum(norms, epsilon)
                        vec_i_to_successors_norm = vec_i_to_successors / norms
                        
                        # Weights based on synergy
                        weights = top_k_synergies / (np.sum(top_k_synergies) + epsilon)
                        
                        # Calculate angles of each successor relative to the target vector (1,0)
                        # angle = acos(dot(vec, target))
                        # Since target is [1,0], dot product is just the x-component of the normalized vector
                        dot_products = vec_i_to_successors_norm[:, 0]
                        # Clip to avoid numerical issues with acos
                        dot_products = np.clip(dot_products, -1.0, 1.0)
                        angles = np.arccos(dot_products) # Angles in [0, pi]
                        
                        # Calculate Angular Entropy (Variance of angles)
                        # Weighted variance
                        mean_angle = np.sum(weights * angles)
                        variance_angles = np.sum(weights * (angles - mean_angle)**2)
                        
                        # Normalize variance to [0, 1] roughly. 
                        # Max possible variance for angles in [0, pi] is limited.
                        # A simple normalization: 1 / (1 + lambda * variance)
                        # Or map directly. Let's use a decay factor.
                        # High variance -> high penalty -> low bonus
                        angular_consistency_factor = 1.0 / (1.0 + 10.0 * variance_angles)
                        
                        # Calculate average direction for alignment score (as before)
                        avg_dir = np.sum(vec_i_to_successors_norm * weights[:, np.newaxis], axis=0)
                        
                        # Normalize average direction
                        avg_dir_norm_val = np.linalg.norm(avg_dir)
                        if avg_dir_norm_val > epsilon:
                            avg_dir_norm = avg_dir / avg_dir_norm_val
                        else:
                            avg_dir_norm = vec_target_norm # Fallback
                        
                        # Cosine similarity between avg_dir and target (current->dest)
                        cos_angle = np.dot(avg_dir_norm, vec_target_norm)
                        
                        # Map cos_angle [-1, 1] to [0, 1]
                        alignment_score = (cos_angle + 1) / 2.0
                        
                        # Combine Alignment and Angular Consistency
                        # directional_consistency_bonus[i] = alignment_score * angular_consistency_factor
                        
                        # Optional: Cone tightness penalty?
                        # If successors are spread out, alignment is less meaningful.
                        # Dot products of individual successors with avg_dir
                        dot_products_avg = np.sum(vec_i_to_successors_norm * avg_dir_norm, axis=1)
                        # Variance of dot products. If low, they are tightly clustered.
                        # We can use 1 - std as a tightness score.
                        if len(dot_products_avg) > 1:
                            cone_tightness = 1.0 - np.std(dot_products_avg)
                            cone_tightness = np.clip(cone_tightness, 0, 1)
                        else:
                            cone_tightness = 1.0
                        
                        # Final bonus combines alignment, cone tightness, and angular entropy
                        directional_consistency_bonus[i] = alignment_score * cone_tightness * angular_consistency_factor

    # --- Step 6: Lookahead Synergy Score Calculation ---
    
    # Determine K based on the number of unvisited nodes.
    K = min(10, len(unvisited_nodes) - 1)
    
    lookahead_synergy_score = np.zeros_like(penalized_base_scores)
    
    if K > 0 and len(unvisited_nodes) > 1:
        actual_K = min(K, len(unvisited_nodes) - 1)
        
        if actual_K > 0:
            # Get indices of top actual_K synergy scores for each row
            negated_synergy = -masked_augmented_synergy
            top_k_indices = np.argpartition(negated_synergy, actual_K, axis=1)[:, :actual_K]
            
            for i in range(len(unvisited_nodes)):
                indices_i = top_k_indices[i]
                top_k_synergies = masked_augmented_synergy[i, indices_i]
                
                # Sum of top K synergies
                sum_top_k_synergies = np.sum(top_k_synergies)
                
                # Simple feasibility heuristic for the cluster:
                # We approximate the cost of visiting the top K successors of i in order.
                # This is O(K) per node, total O(N*K).
                
                # Sort indices by synergy descending to form a potential path
                sorted_indices_local = np.argsort(-top_k_synergies)
                sorted_global_indices = indices_i[sorted_indices_local]
                
                # Calculate path cost: current -> i -> p1 -> p2 -> ... -> dest
                # Note: current -> i is already paid in base score.
                # Here we look at i -> p1 -> ... -> dest
                
                # Start cost from i to first successor
                path_cost = dist_unvisited[i, sorted_global_indices[0]]
                
                # Chain successors
                for k in range(len(sorted_global_indices) - 1):
                    u = sorted_global_indices[k]
                    v = sorted_global_indices[k+1]
                    path_cost += dist_unvisited[u, v]
                
                # Add cost from last successor to dest
                path_cost += costs_to_dest[sorted_global_indices[-1]]
                
                # Baseline cost: i -> dest directly
                baseline_cost = costs_to_dest[i]
                
                detour = max(path_cost - baseline_cost, 0.0)
                
                # Normalize detour by residual budget after visiting i
                normalized_detour = detour / (residual_budget[i] + epsilon)
                
                # Feasibility factor
                feasibility_factor = 1.0 / (1.0 + normalized_detour)
                
                lookahead_synergy_score[i] = sum_top_k_synergies * feasibility_factor

    # --- Step 7: Final Score Calculation ---
    
    final_scores = penalized_base_scores + lookahead_synergy_score + directional_consistency_bonus
    
    # Select the node with the maximum score
    best_idx = np.argmax(final_scores)
    
    return unvisited_nodes[best_idx]
