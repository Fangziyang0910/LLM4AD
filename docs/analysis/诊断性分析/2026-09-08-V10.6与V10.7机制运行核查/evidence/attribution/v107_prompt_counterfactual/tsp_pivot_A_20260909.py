import numpy as np

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Selects the next node using an adaptive nearest-neighbor heuristic with 
    dynamic return bias and 2-step lookahead.
    """
    if len(unvisited_nodes) == 0:
        return destination_node
        
    n_remaining = len(unvisited_nodes)
    
    if n_remaining == 1:
        return int(unvisited_nodes[0])

    n_total = distance_matrix.shape[0]
    if n_total <= 1:
        return destination_node

    # Extract distances
    d_curr = distance_matrix[current_node, unvisited_nodes]
    d_dest = distance_matrix[unvisited_nodes, destination_node]

    # Calculate dynamic return weight
    # As we have fewer nodes left, we should prioritize getting back to the destination
    # Use a smooth transition. Early on, weight is low. Late on, weight is high.
    progress = (n_total - n_remaining) / (n_total - 1)
    
    # Smoothstep function for a nice S-curve weighting
    # t * t * (3 - 2t)
    smooth_progress = progress * progress * (3 - 2 * progress)
    
    # Return weight starts at 0.0 and goes up to 1.0
    # We add a small base value so destination is always considered slightly
    return_weight = 0.0 + 0.8 * smooth_progress

    # Calculate 2-step lookahead cost
    # For each candidate i, estimate the cost of visiting i and then moving to 
    # the nearest unvisited node j (j != i) or the destination.
    # Cost(i) = d(i, j) + d(j, dest) if j is unvisited
    #          = d(i, dest) if going to dest
    
    # To do this efficiently without O(N^2) per step (though N is small in typical TSP tests),
    # we can approximate by finding the minimum distance from i to any other unvisited node
    # plus that node's distance to destination, OR i's direct distance to destination.
    
    if n_remaining > 1:
        sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)].copy()
        np.fill_diagonal(sub, np.inf)
        
        # min_neighbor_dist[i] is the distance from node i to its nearest neighbor in unvisited_nodes
        min_neighbor_dist = np.min(sub, axis=1)
        
        # However, we also need to know WHICH neighbor is closest to add its d(j, dest).
        # A simpler approximation for lookahead: 
        # Lookahead(i) = min( d(i, dest), min_{j != i} ( d(i, j) + d(j, dest) ) )
        
        # Let's compute the min_{j != i} (d(i, j) + d(j, dest))
        # This is element-wise min of (sub + d_dest[None, :])
        # Note: sub[i, i] is inf, so it won't be chosen.
        lookahead_costs = np.min(sub + d_dest[np.newaxis, :], axis=1)
        
        # The actual best next step after i is either going to dest directly or via the best neighbor.
        # But the standard NN heuristic just picks the next node. The lookahead is a heuristic 
        # to adjust the score of the CURRENT choice.
        # We want to penalize candidates i that leave us in a bad position.
        # So we look at the *future* cost associated with choosing i.
        # Future cost if we pick i: min( d(i, dest), min_j(d(i,j) + d(j,dest)) )
        
        lookahead_term = np.minimum(d_dest, lookahead_costs)
    else:
        lookahead_term = d_dest

    # Combine scores
    # Primary: Distance from current
    # Secondary: Distance to destination (weighted)
    # Tertiary: Lookahead cost (to avoid dead ends)
    
    # Normalize components to similar scales to prevent one term from dominating if distances vary
    # Simple normalization: divide by max or mean
    scale = np.max(np.maximum(np.max(d_curr), np.max(d_dest)))
    if scale == 0:
        scale = 1.0

    # Weights
    w_curr = 1.0
    w_look = 0.3  # Modest weight for lookahead

    scores = w_curr * d_curr + return_weight * d_dest + w_look * lookahead_term
    
    # Select the node with the minimum score
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])