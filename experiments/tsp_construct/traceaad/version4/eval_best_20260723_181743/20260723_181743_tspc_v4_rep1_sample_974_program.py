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
        return unvisited_nodes[0]
    
    candidates = unvisited_nodes
    n_candidates = len(candidates)
    
    # Pre-fetch distances from current node to all candidates for speed
    dists_from_current = np.array([distance_matrix[current_node, c] for c in candidates])
    
    # Pre-fetch distance from current node to destination for gradient calculation
    dist_current_to_dest = distance_matrix[current_node, destination_node]
    
    estimated_costs = np.zeros(n_candidates)
    
    # Define the base alpha factor for the bottleneck penalty
    # Scale inversely with the number of remaining nodes to increase sensitivity near the end
    base_alpha = 0.1
    dynamic_alpha = base_alpha * (1.0 / len(unvisited_nodes))
    
    # Total number of nodes in the problem (estimated from distance matrix)
    total_nodes = distance_matrix.shape[0]
    
    # Destination Alignment Bias factor using a piecewise linear ramp
    # Activates only when fewer than 15% of nodes remain to ensure zero interference
    # in early/mid phases.
    ratio_remaining = len(unvisited_nodes) / total_nodes
    
    max_bias = 0.5
    threshold_bias = 0.15
    
    if ratio_remaining < threshold_bias:
        # Linear ramp from 0 at threshold to max_bias at 0 remaining
        bias_factor = max_bias * (1.0 - (ratio_remaining / threshold_bias))
    else:
        bias_factor = 0.0
        
    # Local Density Penalty factor
    # Cubic decay to diminish influence even more rapidly in the middle phase
    density_weight = 0.2 * (ratio_remaining ** 3)
    
    # Tour Continuity Weight
    # Replaced linear decay with Gaussian decay centered at 50% remaining nodes.
    # This prioritizes smoothness in the mid-tour phase.
    sigma = 0.2
    continuity_weight = 0.1 * np.exp(-((ratio_remaining - 0.5) ** 2) / (2 * sigma ** 2))
    
    # Destination Proximity Gradient Weight
    # Refine from linear ramp to exponential decay.
    # exp(-5 * ratio_remaining) creates a sharper emphasis in the final 10-20%
    # while being very low in the 30-70% zone, preventing premature commitment.
    max_gradient_weight = 0.5
    gradient_weight = max_gradient_weight * np.exp(-5.0 * ratio_remaining)
    
    # Cluster Cohesion Reward Weight
    # Peaks when 50-70% of nodes remain to encourage exhausting local clusters
    # Adjusted sigma to 0.2 and peak to 0.6 to handle transition better
    sigma_cohesion = 0.2
    max_cohesion_weight = 0.3
    cohesion_weight = max_cohesion_weight * np.exp(-((ratio_remaining - 0.6) ** 2) / (2 * sigma_cohesion ** 2))

    # Angular Deviation Penalty Weight
    # Encourages smoother paths by penalizing sharp turns relative to the approximated previous trajectory
    # Gaussian peak at 70% remaining (early-mid tour) to enforce global structure early on
    angular_weight = 0.1 * np.exp(-((ratio_remaining - 0.7) ** 2) / (2 * 0.15 ** 2))

    # Note: The heuristic does not have access to the previous node ID explicitly in the signature.
    # Therefore, we approximate the previous node.
    continuity_bonus = 0.0

    # Determine Visited Nodes
    all_indices = np.arange(total_nodes)
    visited_mask = np.isin(all_indices, unvisited_nodes, invert=True)
    visited_nodes = all_indices[visited_mask]
    
    # Approximate Previous Node for Angular Penalty using Trajectory Inertia
    # We look for the two closest visited nodes to the current node.
    # The closest is likely the true previous node.
    # The second closest helps establish the "incoming" direction if we treat the 
    # vector from the 2nd closest to the current as the inertia vector.
    ref_node_for_angle = None
    
    if len(visited_nodes) >= 2:
        dists_to_visited = distance_matrix[current_node, visited_nodes]
        # Get indices of the two smallest distances among visited nodes
        sorted_idx = np.argsort(dists_to_visited)
        
        closest_visited = visited_nodes[sorted_idx[0]]
        second_closest_visited = visited_nodes[sorted_idx[1]]
        
        # Use the second closest as the reference point for the incoming vector
        # Vector: SecondClosest -> Current
        # This implies the path was coming from SecondClosest through Current (or nearby)
        # This provides a more stable directional reference.
        ref_node_for_angle = second_closest_visited
    elif len(visited_nodes) == 1:
        # Only start node visited. Cannot define a direction.
        ref_node_for_angle = None
    else:
        ref_node_for_angle = None

    for i, cand in enumerate(candidates):
        # 1. Cost to move from current to candidate
        step_cost = dists_from_current[i]
        
        # 2. Estimate completion cost using Dynamic Regret Nearest Neighbor on remaining nodes
        remaining_nodes = np.delete(candidates, i)
        
        # Calculate bottleneck penalty
        other_nodes = np.concatenate([remaining_nodes, [destination_node]])
        if len(other_nodes) > 0:
            dists_to_others = distance_matrix[cand, other_nodes]
            max_dist = np.max(dists_to_others)
        else:
            max_dist = 0
        
        bottleneck_penalty = dynamic_alpha * max_dist
        
        if len(remaining_nodes) == 0:
            completion_cost = distance_matrix[cand, destination_node]
        else:
            # Simulate Dynamic Regret Nearest Neighbor starting from candidate
            nn_current = cand
            nn_remaining = list(remaining_nodes)
            nn_cost = 0.0
            
            remaining_ids = np.array(nn_remaining)
            
            while len(remaining_ids) > 0:
                avg_dists = np.mean(distance_matrix[remaining_ids, :][:, remaining_ids], axis=1)
                global_avg = np.mean(distance_matrix[remaining_ids, :][:, remaining_ids])
                regret_scores = avg_dists - global_avg
                
                dists_to_remaining = distance_matrix[nn_current, remaining_ids]
                
                current_ratio_remaining = len(remaining_ids) / total_nodes
                base_beta = 0.5
                adaptive_beta = base_beta * (1.0 + (1.0 - current_ratio_remaining) * 2.0)
                
                heuristic_costs = dists_to_remaining - adaptive_beta * regret_scores
                
                next_idx_local = np.argmin(heuristic_costs)
                next_node = remaining_ids[next_idx_local]
                
                nn_cost += dists_to_remaining[next_idx_local]
                nn_current = next_node
                
                remaining_ids = np.delete(remaining_ids, next_idx_local)
                
            completion_cost = nn_cost + distance_matrix[nn_current, destination_node]
            
        # 3. Destination Alignment Bias
        dist_to_dest = distance_matrix[cand, destination_node]
        dest_bias_penalty = bias_factor * dist_to_dest
        
        # 4. Local Density Penalty
        k = min(5, len(remaining_nodes)) if len(remaining_nodes) > 0 else 0
        
        density_penalty = 0.0
        if k > 0:
            dists_to_remaining_unvisited = distance_matrix[cand, remaining_nodes]
            sorted_dists = np.sort(dists_to_remaining_unvisited)
            avg_knn_dist = np.mean(sorted_dists[:k])
            density_penalty = density_weight * avg_knn_dist
        
        # 5. Destination Proximity Gradient Penalty
        # Penalize if the candidate is significantly further from the destination than the current node
        # during the final phase of the tour.
        dist_cand_to_dest = dist_to_dest
        gradient_change = dist_cand_to_dest - dist_current_to_dest
        
        # Only apply penalty if moving away from destination (gradient_change > 0)
        gradient_penalty = 0.0
        if gradient_change > 0:
            gradient_penalty = gradient_weight * gradient_change

        # 6. Cluster Cohesion Reward
        # Calculate average distance to k-nearest neighbors among remaining unvisited nodes
        # Subtract from cost to reward cohesive cluster visiting
        k_cohesion = min(10, len(remaining_nodes)) if len(remaining_nodes) > 0 else 0
        cohesion_reward = 0.0
        if k_cohesion > 0:
            dists_to_remaining = distance_matrix[cand, remaining_nodes]
            sorted_dists_cohesion = np.sort(dists_to_remaining)
            avg_knn_dist_cohesion = np.mean(sorted_dists_cohesion[:k_cohesion])
            cohesion_reward = cohesion_weight * avg_knn_dist_cohesion

        # 7. Angular Deviation Penalty (Trajectory-Inertia Anchored)
        angular_penalty = 0.0
        
        if ref_node_for_angle is not None:
            # Vector 1: From Reference Node (2nd closest visited) to Current Node
            # Vector 2: From Current Node to Candidate Node
            
            dist_ref_to_current = distance_matrix[ref_node_for_angle, current_node]
            dist_ref_to_cand = distance_matrix[ref_node_for_angle, cand]
            
            a = dist_ref_to_current
            b = step_cost
            c = dist_ref_to_cand
            
            # We want to minimize the turning angle.
            # The angle at 'current' node is what matters for smoothness.
            
            if a > 1e-9 and b > 1e-9:
                denominator = 2 * a * b
                # Law of Cosines: c^2 = a^2 + b^2 - 2ab cos(angle)
                cos_angle = (a**2 + b**2 - c**2) / denominator
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                
                angle = np.arccos(cos_angle)
                
                # Ideal angle for straight line is PI (180 degrees).
                deviation_from_pi = np.pi - angle
                norm_deviation = np.abs(deviation_from_pi) / np.pi
                
                # Penalize deviation from straight line
                angular_penalty = angular_weight * norm_deviation * step_cost
            else:
                angular_penalty = 0.0
        else:
            # If no reference node (start of tour), no angular penalty
            angular_penalty = 0.0

        # 8. Dynamic Connectivity Cost (Replaces Regret Potential)
        # Estimates the "damage" to the unvisited graph's connectivity if this node is removed.
        # We calculate the sum of distances to the 1st and 2nd nearest neighbors in the REMAINING unvisited set.
        # Lower cost means the node is well-connected to the remaining cluster, so removing it doesn't fragment the graph much.
        connectivity_bonus = 0.0
        
        if len(remaining_nodes) >= 2:
            dists_from_cand_to_rem = distance_matrix[cand, remaining_nodes]
            # Sort distances to remaining nodes
            sorted_dists_rem = np.sort(dists_from_cand_to_rem)
            
            # Sum of the two smallest distances
            min_conn_cost = sorted_dists_rem[0] + sorted_dists_rem[1]
            
            # Normalize by step_cost to make it comparable across different scales
            # We subtract this cost because we want to prefer nodes that are "cheap" to remove from the graph structure
            # i.e., nodes that have close neighbors left behind, keeping the rest of the graph tight.
            # Actually, if a node is far from everyone, removing it is "cheap" for connectivity? No.
            # If a node is a leaf (far from others), removing it breaks nothing.
            # If a node is central (close to many), removing it might increase MST weight of remainder?
            # The metric here is: How connected is this node to the FUTURE?
            # If it's very close to remaining nodes, visiting it now might be good to "clear" a dense cluster.
            # Let's stick to the idea: Prioritize nodes that maintain low-cost global connectivity.
            # A simple heuristic: Reward nodes where the nearest remaining neighbor is very close.
            # This encourages eating away at dense clusters.
            
            connectivity_bonus = 0.05 * min_conn_cost / (step_cost + 1e-6)
            
        elif len(remaining_nodes) == 1:
            # Only one remaining, connectivity is trivial
            pass

        # Total estimated cost
        total_estimated_cost = step_cost + completion_cost - bottleneck_penalty + dest_bias_penalty - density_penalty + gradient_penalty - continuity_bonus - cohesion_reward + angular_penalty - connectivity_bonus
        
        estimated_costs[i] = total_estimated_cost
    
    # Lookahead Stability Penalty
    if n_candidates <= 3:
        top_k_indices = np.arange(n_candidates)
    else:
        top_k_indices = np.argpartition(estimated_costs, 3)[:3]
    
    stability_penalty = np.zeros(n_candidates)
    
    lookahead_depth = 3
    
    for idx in top_k_indices:
        cand = candidates[idx]
        
        edges = []
        edges.append(distance_matrix[current_node, cand])
        
        sim_remaining = np.delete(candidates, idx)
        sim_current = cand
        
        for _ in range(lookahead_depth):
            if len(sim_remaining) == 0:
                next_node = destination_node
            else:
                dists_to_rem = distance_matrix[sim_current, sim_remaining]
                next_idx = np.argmin(dists_to_rem)
                next_node = sim_remaining[next_idx]
                sim_remaining = np.delete(sim_remaining, next_idx)
            
            edges.append(distance_matrix[sim_current, next_node])
            sim_current = next_node
        
        edges_array = np.array(edges)
        std_edge = np.std(edges_array)
        mean_edge = np.mean(edges_array)
        
        fixed_weight = 0.02
        
        if mean_edge > 1e-6:
            cv = std_edge / mean_edge
            stability_penalty[idx] = fixed_weight * cv
        else:
            stability_penalty[idx] = 0.0

    # Add stability penalty to the total estimated costs
    final_costs = estimated_costs + stability_penalty
    
    best_idx = np.argmin(final_costs)
    return candidates[best_idx]
