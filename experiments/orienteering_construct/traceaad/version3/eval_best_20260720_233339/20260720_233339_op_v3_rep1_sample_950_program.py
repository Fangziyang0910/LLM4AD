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
        # No unvisited nodes, just go to destination
        return destination_node
    
    if len(unvisited_nodes) == 1:
        # Only one unvisited node, visit it if possible
        return unvisited_nodes[0]
    
    n_candidates = len(unvisited_nodes)
    scores = np.zeros(n_candidates)
    
    # Pre-calculate costs from current node to each candidate
    costs_from_current = np.array([distance_matrix[current_node, node] for node in unvisited_nodes])
    
    # Pre-calculate prizes for each candidate
    candidate_prizes = np.array([prizes[node] for node in unvisited_nodes])
    
    # Calculate Global Scale Metric: Average pairwise distance among unvisited nodes
    # This helps normalize scores across instances with different spatial densities
    n_unvisited = len(unvisited_nodes)
    total_dist = 0.0
    count = 0
    for i in range(n_unvisited):
        for j in range(i + 1, n_unvisited):
            total_dist += distance_matrix[unvisited_nodes[i], unvisited_nodes[j]]
            count += 1
    
    if count > 0:
        avg_pairwise_dist = total_dist / count
    else:
        avg_pairwise_dist = 1.0 # Fallback if only 1 or 0 nodes (handled by earlier checks)

    # Avoid division by zero
    scale_factor = max(avg_pairwise_dist, 1e-9)
    
    # For each candidate, calculate a lookahead score based on two branching strategies
    for i, candidate in enumerate(unvisited_nodes):
        # Calculate remaining budget after visiting this candidate
        cost_to_candidate = costs_from_current[i]
        budget_after_candidate = remaining_budget - cost_to_candidate
        
        # Base prize from the candidate itself
        base_prize = candidate_prizes[i]
        
        # Identify other unvisited nodes for simulation (excluding the current candidate)
        other_unvisited_ids = np.delete(unvisited_nodes, i)
        
        # Strategy 1: Prize-to-Cost Greedy Simulation
        sim_budget_1 = budget_after_candidate
        current_node_1 = candidate
        sim_prize_1 = base_prize
        
        # Create a mutable list for tracking available nodes in sim 1
        available_1 = list(other_unvisited_ids)
        
        while sim_budget_1 > 0 and len(available_1) > 0:
            best_next_node_1 = -1
            best_ratio_1 = -1
            best_dist_1 = float('inf')
            
            for next_node in available_1:
                dist_to_next = distance_matrix[current_node_1, next_node]
                dist_to_dest = distance_matrix[next_node, destination_node]
                total_move_cost = dist_to_next + dist_to_dest
                
                if total_move_cost <= sim_budget_1:
                    marginal_cost = dist_to_next
                    if marginal_cost == 0:
                        marginal_cost = 1e-9
                    
                    prize = prizes[next_node]
                    ratio = prize / marginal_cost
                    
                    if ratio > best_ratio_1:
                        best_ratio_1 = ratio
                        best_next_node_1 = next_node
                        best_dist_1 = dist_to_next
            
            if best_next_node_1 == -1:
                break
                
            sim_prize_1 += prizes[best_next_node_1]
            sim_budget_1 -= best_dist_1
            current_node_1 = best_next_node_1
            
            # Remove from available list
            available_1.remove(best_next_node_1)

        # Strategy 2: Prize-Discounted Potential Field Simulation
        sim_budget_2 = budget_after_candidate
        current_node_2 = candidate
        sim_prize_2 = base_prize
        
        # Create a mutable list for tracking available nodes in sim 2
        available_2 = list(other_unvisited_ids)
        
        while sim_budget_2 > 0 and len(available_2) > 0:
            best_next_node_2 = -1
            best_score_2 = float('inf')
            best_dist_2 = float('inf')
            
            if len(available_2) == 0:
                break

            # --- Prize-Discounted Potential Field Implementation ---
            
            # We calculate a potential field score for each available next node.
            # Potential(X) = Sum( Prize(U) / (Dist(X, U) + epsilon) ) for all U in Available.
            # This metric captures the "gravitational pull" of high-value nodes.
            # A higher potential means the node X is closer to many high-value nodes.
            
            # 1. Calculate Potential Field for each candidate next node
            potentials = np.zeros(len(available_2))
            epsilon = 1e-9
            
            # Pre-fetch prizes for available nodes to speed up loop
            avail_node_ids = available_2
            avail_prizes_arr = np.array([prizes[node] for node in avail_node_ids])
            
            for j, node_x in enumerate(avail_node_ids):
                # Calculate sum of prize/dist for all other available nodes U relative to X
                # We include X itself in the sum? Usually self-distance is 0, causing inf.
                # Let's exclude X from its own potential calculation or handle 0 dist.
                # If X is in available_2, dist(X,X)=0. prize(X)/0 -> inf.
                # To avoid singularity, we can either:
                # 1. Exclude self from potential sum.
                # 2. Use a small epsilon offset for all distances.
                # Let's use epsilon offset for stability and include self.
                
                # Vectorized distance calculation from node_x to all available nodes
                dists_x_to_all = np.array([distance_matrix[node_x, u] for u in avail_node_ids])
                
                # Potential = sum(prize / (dist + eps))
                potentials[j] = np.sum(avail_prizes_arr / (dists_x_to_all + epsilon))

            # --- Dynamic Influence Radius Implementation ---
            # Calculate the distance from the current simulation node to the destination
            dist_to_dest_current = distance_matrix[current_node_2, destination_node]
            
            # Calculate the Return Viability Ratio
            if sim_budget_2 > 1e-9:
                return_viability_ratio = dist_to_dest_current / sim_budget_2
            else:
                return_viability_ratio = 1.0
            
            # Clamp ratio to [0, 1]
            return_viability_ratio = np.clip(return_viability_ratio, 0.0, 1.0)
            
            # Determine weights for local distance vs global attraction (potential field)
            # We want to maximize potential (so we subtract it from cost) or minimize negative potential.
            # Score = w_local * dist_to_next - w_global * normalized_potential
            
            w_local = 1.0
            
            # Scale global attraction weight based on viability
            # If viability is 0.2 (loose), factor is high.
            # If viability is 0.9 (tight), factor is low.
            max_influence = 0.5 
            
            # Non-linear scaling: exponential decay of influence as viability decreases (ratio increases)
            influence_factor = ((1.0 - return_viability_ratio) ** 2) * max_influence
            
            w_global = influence_factor
            
            # Normalize potentials to be comparable with distances
            # Max potential can vary wildly. Let's normalize by max potential in current step.
            if len(available_2) > 0:
                max_potential = np.max(potentials)
                if max_potential > 1e-9:
                    normalized_potentials = potentials / max_potential
                else:
                    normalized_potentials = potentials
            else:
                normalized_potentials = potentials

            # Calculate composite score for each candidate next node
            for j, next_node in enumerate(available_2):
                dist_to_next = distance_matrix[current_node_2, next_node]
                dist_to_dest = distance_matrix[next_node, destination_node]
                total_move_cost = dist_to_next + dist_to_dest
                
                if total_move_cost <= sim_budget_2:
                    # Score: Minimize distance, Maximize Potential
                    # Score = dist - (weight * normalized_potential)
                    pot_val = normalized_potentials[j]
                    score = w_local * dist_to_next - w_global * pot_val
                    
                    if score < best_score_2:
                        best_score_2 = score
                        best_next_node_2 = next_node
                        best_dist_2 = dist_to_next
            
            if best_next_node_2 == -1:
                break
                
            sim_prize_2 += prizes[best_next_node_2]
            sim_budget_2 -= best_dist_2
            current_node_2 = best_next_node_2
            
            # Remove from available list
            available_2.remove(best_next_node_2)

        # Score: Max prize from either simulation branch
        max_sim_prize = max(sim_prize_1, sim_prize_2)
        
        # Apply Global Scale-Normalized Chain Gain
        # Normalize the simulated prize by the global average distance to make it scale-invariant
        normalized_gain = max_sim_prize / scale_factor
        
        # Calculate immediate efficiency penalty
        if candidate_prizes[i] == 0:
            efficiency_penalty = cost_to_candidate
        else:
            # Higher ratio means more distance per prize, so higher penalty
            efficiency_penalty = (cost_to_candidate / candidate_prizes[i])
            
        # --- Return Path Viability Coefficient (Outer Loop Penalty) ---
        # This penalizes the selection of the first candidate if it puts us in a tight spot for the rest of the route.
        dist_to_dest_from_candidate = distance_matrix[candidate, destination_node]
        
        # Calculate the ratio of return distance to remaining budget after candidate visit
        if budget_after_candidate > 1e-9:
            return_ratio = dist_to_dest_from_candidate / budget_after_candidate
        else:
            return_ratio = 1.0 # Should not happen for feasible nodes
            
        # Clamp return_ratio to avoid extreme values
        return_ratio = np.clip(return_ratio, 0.0, 1.0)
        
        # Base weight for penalty
        base_weight = 0.1
        
        # RVP modulates the base weight. 
        if return_ratio < 0.5:
            rvp_coeff = 1.0 
        else:
            excess_ratio = (return_ratio - 0.5) * 2 
            rvp_coeff = 1.0 + (excess_ratio ** 2) * 5.0 
            
        adaptive_weight = base_weight * rvp_coeff
        
        # Scale the penalty by the adaptive weight
        scaled_penalty = efficiency_penalty * adaptive_weight
        
        # Final Score: Normalized Gain - Scaled Penalty
        scores[i] = normalized_gain - scaled_penalty

    # Select the node with the highest score
    best_idx = np.argmax(scores)
    return unvisited_nodes[best_idx]
