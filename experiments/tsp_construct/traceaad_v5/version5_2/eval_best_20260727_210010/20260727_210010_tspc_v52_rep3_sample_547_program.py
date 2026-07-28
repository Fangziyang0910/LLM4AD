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
    
    # Calculate distances from current node to each unvisited node
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    n_unvisited = len(unvisited_nodes)
    
    # Calculate distance from each unvisited candidate to the destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Extract the submatrix of distances between all unvisited nodes
    # shape: (n_unvisited, n_unvisited)
    dist_unvisited = distance_matrix[unvisited_nodes][:, unvisited_nodes]
    
    epsilon = 1e-10
    
    # Calculate Geometric Mean of distances to other unvisited nodes using log-sum-exp for stability
    # GM = exp( (1/(n-1)) * sum(ln(dist_i_j)) ) for j != i
    
    # Create a mask where diagonal is 0 (False) and off-diagonal is 1 (True)
    mask = np.ones((n_unvisited, n_unvisited), dtype=bool)
    np.fill_diagonal(mask, False)
    
    # Get valid distances for log calculation
    # Use epsilon to avoid log(0) if distances are extremely small or zero
    valid_dists = np.where(mask, dist_unvisited, 1.0) 
    
    # Calculate log distances
    log_dists = np.log(valid_dists + epsilon)
    
    # Set masked (self) log distances to 0 so they don't affect the sum
    log_dists[~mask] = 0.0
    
    # Sum of log distances to others for each candidate
    sum_log_dists = np.sum(log_dists, axis=1)
    
    # Number of other unvisited nodes
    num_others = n_unvisited - 1
    
    # Geometric mean is exp( (1/(n-1)) * sum(ln(dist)) )
    geometric_mean_dist_to_others = np.exp(sum_log_dists / (num_others + epsilon))
    
    # Primary Score: Asymmetric Geometric Mean Denominator with Non-Linear Transform
    # Applied sqrt to numerator and halved the exponents (0.53 -> 0.265, 0.47 -> 0.235)
    a = 0.265
    b = 0.235
    
    denom = (dist_to_dest + epsilon)**a * (geometric_mean_dist_to_others + epsilon)**b
    
    # Lower score means: closer to current AND good balance of closeness to dest and centrality among unvisited.
    # Numerator transformed with square root as per directive
    geometric_score = np.sqrt(dist_from_current + epsilon) / denom
    
    # Lookahead Feasibility Score (from Reference Program Step 7)
    # For each candidate, estimate the cost of completing the tour using Greedy Nearest Neighbor
    lookahead_costs = np.zeros(n_unvisited)
    
    for i, next_node in enumerate(unvisited_nodes):
        # Remaining nodes to visit after choosing next_node
        remaining_indices = np.delete(np.arange(n_unvisited), i)
        remaining_nodes = unvisited_nodes[remaining_indices]
        
        estimated_cost = 0.0
        
        if len(remaining_nodes) > 0:
            current_pos = next_node
            current_unvisited_set = set(remaining_nodes)
            
            # Greedy nearest neighbor on remaining nodes
            while current_unvisited_set:
                # Find distances from current_pos to all current_unvisited
                dists = distance_matrix[current_pos, list(current_unvisited_set)]
                # Find index of min distance
                min_idx = np.argmin(dists)
                nearest_node = list(current_unvisited_set)[min_idx]
                min_dist = dists[min_idx]
                
                estimated_cost += min_dist
                current_pos = nearest_node
                current_unvisited_set.remove(nearest_node)
            
            # Add cost from last visited node back to destination
            estimated_cost += distance_matrix[current_pos, destination_node]
        else:
            # No remaining nodes, just go to destination
            estimated_cost = dist_to_dest[i]
            
        lookahead_costs[i] = estimated_cost

    # Combine scores
    # Primary geometric score + weighted lookahead cost
    # Alpha balances immediate geometric efficiency vs long-term tour cost feasibility
    alpha = 0.1
    final_scores = geometric_score + alpha * lookahead_costs
    
    # Select the node with the minimum score
    best_idx = np.argmin(final_scores)
    return int(unvisited_nodes[best_idx])
