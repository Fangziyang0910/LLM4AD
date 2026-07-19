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
        return -1

    best_score = -np.inf
    best_node = unvisited_nodes[0]
    
    # Pre-fetch distances from current node to all unvisited nodes
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Pre-fetch prizes for unvisited nodes
    prizes_unvisited = prizes[unvisited_nodes]
    
    # Pre-fetch distance from each unvisited node to destination
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    num_candidates = len(unvisited_nodes)
    
    # Parameters for Local Density Bonus
    # Dynamic radius: alpha * mean(dists_from_current)
    alpha = 1.5
    mean_dist = np.mean(dists_from_current)
    proximity_threshold = alpha * mean_dist
    
    # Dynamic beta:
    # Simplified linear scaling.
    # Estimate initial budget for this phase as remaining_budget + distance to destination.
    # This serves as a proxy for the total budget available for the rest of the tour.
    dist_current_to_dest = distance_matrix[current_node, destination_node]
    initial_budget_estimate = remaining_budget + dist_current_to_dest
    
    max_beta = 1.0
    
    if initial_budget_estimate > 0:
        # Linear scaling: beta is 0 when budget is full (ratio=1) and max_beta when budget is tight (ratio=0).
        # ratio = remaining_budget / initial_budget_estimate
        ratio = remaining_budget / initial_budget_estimate
        beta_dynamic = max_beta * (1.0 - ratio)
        # Clamp to [0, max_beta]
        beta_dynamic = max(0.0, min(max_beta, beta_dynamic))
    else:
        # Fallback if estimate is 0
        beta_dynamic = max_beta
        
    total_remaining_prizes = np.sum(prizes_unvisited)
    
    for idx in range(num_candidates):
        node_id = unvisited_nodes[idx]
        cost_to_node = dists_from_current[idx]
        
        # Check if this node is feasible (should be by contract, but good to ensure)
        if cost_to_node > remaining_budget:
            continue
            
        remaining_after_visit = remaining_budget - cost_to_node
        
        # Lookahead estimation
        # Start at node_id. Current position is node_id.
        # Budget is remaining_after_visit.
        # We want to estimate how much more prize we can get.
        
        estimated_prize = prizes[node_id]
        
        # Greedy lookahead
        current_pos = node_id
        current_budget = remaining_after_visit
        visited_in_lookahead = {node_id}
        
        # We will iterate until no more nodes can be added
        while True:
            best_next_node = -1
            best_ratio = -np.inf
            
            # Find best next node from current_pos among unvisited_nodes excluding visited_in_lookahead
            for next_idx in range(num_candidates):
                next_node_id = unvisited_nodes[next_idx]
                
                if next_node_id in visited_in_lookahead:
                    continue
                
                dist_to_next = distance_matrix[current_pos, next_node_id]
                dist_next_to_dest = distance_matrix[next_node_id, destination_node]
                
                # Check feasibility: dist_to_next + dist_next_to_dest <= current_budget
                if dist_to_next + dist_next_to_dest <= current_budget:
                    # Calculate ratio: prize / distance
                    # Add small epsilon to avoid division by zero if dist is 0 (unlikely for distinct nodes)
                    ratio = prizes[next_node_id] / (dist_to_next + 1e-9)
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_next_node = next_node_id
            
            if best_next_node == -1:
                break
            
            # Move to best_next_node
            estimated_prize += prizes[best_next_node]
            dist_traveled = distance_matrix[current_pos, best_next_node]
            current_budget -= dist_traveled
            current_pos = best_next_node
            visited_in_lookahead.add(best_next_node)
            
            # Safety break to prevent infinite loops
            if len(visited_in_lookahead) > num_candidates:
                break

        # Calculate Local Density Bonus
        # Sum prizes of unvisited nodes within proximity_threshold of the candidate node
        sum_prizes_nearby = 0.0
        for next_idx in range(num_candidates):
            next_node_id = unvisited_nodes[next_idx]
            if next_node_id == node_id:
                continue
            
            dist_to_nearby = distance_matrix[node_id, next_node_id]
            if dist_to_nearby <= proximity_threshold:
                sum_prizes_nearby += prizes[next_node_id]
        
        # Calculate density factor
        # Avoid division by zero if total_remaining_prizes is 0
        if total_remaining_prizes > 0:
            density_factor = 1.0 + beta_dynamic * (sum_prizes_nearby / total_remaining_prizes)
        else:
            density_factor = 1.0
            
        # Calculate score for this candidate
        # Score = (Estimated Total Prize * Density Factor) / Cost to reach the entry node (plus regularization)
        score = (estimated_prize * density_factor) / (cost_to_node + 1e-9)
        
        if score > best_score:
            best_score = score
            best_node = node_id

    return best_node
