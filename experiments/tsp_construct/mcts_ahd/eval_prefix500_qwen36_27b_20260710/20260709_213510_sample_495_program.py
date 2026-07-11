
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
    import numpy as np

    if len(unvisited_nodes) == 0:
        return destination_node
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])

    best_node = None
    best_score = float('inf')

    n_unvisited = len(unvisited_nodes)
    unvisited_ids = unvisited_nodes
    
    # Precompute distances from current node to all unvisited nodes
    dist_current_to_unvisited = distance_matrix[current_node, unvisited_nodes]
    
    # Precompute distances from all unvisited nodes to destination
    dist_unvisited_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Precompute distances from current node to destination
    dist_current_to_dest = distance_matrix[current_node, destination_node]

    # Pre-fetch destination row for speed
    dist_to_dest_row = distance_matrix[destination_node]

    # Precompute Connectivity Metric (Mean distance to OTHER unvisited nodes)
    # This is inspired by Algo 1 but calculated efficiently here.
    # dist_unvisited_matrix[i, j] is distance between u_indices[i] and u_indices[j]
    dist_unvisited_matrix = distance_matrix[np.ix_(unvisited_ids, unvisited_ids)]
    
    # Sum of distances from each unvisited node to ALL other unvisited nodes (including self=0)
    sum_dists_others = np.sum(dist_unvisited_matrix, axis=1)
    
    # Mean distance to OTHERS (exclude self)
    if n_unvisited > 1:
        mean_dist_to_others = sum_dists_others / (n_unvisited - 1)
    else:
        mean_dist_to_others = np.zeros(n_unvisited)

    # Weights inspired by Algo 1 for the local terms
    alpha = 1.25      # Weight for immediate distance
    beta = 0.65       # Weight for connectivity metric (Mean dist to others)
    gamma_local = 0.55 # Weight for destination proximity penalty (local term)

    for i, node in enumerate(unvisited_ids):
        d_curr_node = dist_current_to_unvisited[i]
        
        # 1. Calculate Dynamic Alignment Heuristic (from Algo 2)
        detour_cost = d_curr_node + dist_unvisited_to_dest[i] - dist_current_to_dest
        detour_cost = max(0.0, detour_cost)
        
        # 2. Calculate Look-Ahead Residual Cost (Greedy NN on remaining)
        remaining_indices = [j for j in range(n_unvisited) if j != i]
        remaining_nodes = unvisited_ids[remaining_indices]
        
        temp_current = node
        temp_remaining = list(remaining_nodes)
        temp_res_cost = 0.0
        
        # Greedy NN: always pick the closest unvisited residual node
        while len(temp_remaining) > 0:
            dists = distance_matrix[temp_current, temp_remaining]
            min_idx = np.argmin(dists)
            next_node_residual = temp_remaining[min_idx]
            dist = dists[min_idx]
            
            temp_res_cost += dist
            temp_current = next_node_residual
            temp_remaining.pop(min_idx)
            
        # Finally go from last visited residual node to destination
        temp_res_cost += distance_matrix[temp_current, destination_node]
        
        # 3. Combine Scores
        
        # Local Score Component (Inspired by Algo 1):
        # Score_local = alpha * dist_now - beta * mean_dist_to_others - gamma_local * dist_to_dest
        # This encourages picking nodes that are close, but "peripheral" (high mean dist) and far from dest.
        # Note: Algo 1's code logic rewarded high mean_dist (peripheral). 
        # We apply this same logic to the immediate step evaluation to guide the initial step of the residual.
        
        dist_to_dest = dist_unvisited_to_dest[i]
        conn_metric = mean_dist_to_others[i]
        
        local_guidance_score = (alpha * d_curr_node) \
                               - (beta * conn_metric) \
                               - (gamma_local * dist_to_dest)
        
        # Global Score Component (Inspired by Algo 2):
        # Base path estimate + Dynamic Alignment Penalty
        
        # Normalize residual cost? Algo 2 used total sum. Let's stick to sum.
        base_path_estimate = d_curr_node + temp_res_cost
        
        if base_path_estimate > 1e-9:
            alignment_ratio = detour_cost / base_path_estimate
        else:
            alignment_ratio = 1.0
            
        # Dynamic weight for alignment: higher when many nodes left
        weight_gamma_global = n_unvisited / (n_unvisited + 2.0)
        
        global_guidance_score = d_curr_node + temp_res_cost + weight_gamma_global * detour_cost
        
        # Hybrid Score:
        # We combine the local structural insight (Algo 1) with the global look-ahead (Algo 2).
        # To prevent scale mismatch, we can weight them. 
        # The local guidance is a "preference" score, the global is a "cost" score.
        # Let's blend them. 
        # However, simply adding them might be noisy. 
        # Let's try: Total = global_guidance_score + lambda * local_guidance_score
        # Since local_guidance_score has negative terms, it acts as a bonus/penalty adjustment.
        
        # Let's use a small blending factor to inject the local structural preference into the global cost.
        blending_factor = 0.3 
        
        total_score = global_guidance_score + blending_factor * local_guidance_score
        
        if total_score < best_score:
            best_score = total_score
            best_node = node
            
    return int(best_node)
